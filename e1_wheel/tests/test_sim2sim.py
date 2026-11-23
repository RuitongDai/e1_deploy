"""Check the training/deployment boundary and control transitions without a GUI."""

import contextlib
import io
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from e1_wheel import wheel_config
from e1_wheel.gamepad import GamepadDisconnected
from e1_wheel.scripts.sim2sim_e1_wheel import (
    ControlState,
    OnnxPolicy,
    WheelSimulation,
    parse_args,
)
from e1_wheel.wheel_config import (
    DEFAULT_Q,
    JOINT_NAMES,
    KD,
    KP,
    LEG_JOINT_NAMES,
    POLICY_DT,
    POLICY_PATH,
    action_targets,
    build_observation,
)


class ObservationAndActionTests(unittest.TestCase):
    def test_observation_layout_and_clip_before_scale(self):
        q = DEFAULT_Q + np.arange(20) * 0.01
        q[16:] = 10000.0
        obs = build_observation(
            [400, -400, 2],
            [0, 0, -1],
            [0.5, -0.2, 0.3],
            q,
            np.full(20, 200.0),
            np.full(20, 120.0),
        )
        self.assertEqual(obs.shape, (69,))
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_allclose(obs[:9], [25, -25, 0.5, 0, 0, -1, 0.5, -0.2, 0.3])
        np.testing.assert_allclose(obs[9:25], np.arange(16) * 0.01, atol=1e-7)
        np.testing.assert_array_equal(obs[25:29], 0.0)
        np.testing.assert_array_equal(obs[29:49], 5.0)
        np.testing.assert_array_equal(obs[49:69], 100.0)

    def test_mixed_action_scales_offsets_and_post_scale_clipping(self):
        target = action_targets(np.full(20, 120.0))
        np.testing.assert_allclose(target[:16], [30.5, 15, 15, 29] * 4)
        np.testing.assert_array_equal(target[16:], 100.0)
        np.testing.assert_array_equal(action_targets(np.zeros(20)), DEFAULT_Q)
        self.assertEqual(JOINT_NAMES[0], "FR_thigh_joint")
        self.assertEqual(JOINT_NAMES[4], "FL_thigh_joint")
        self.assertEqual(JOINT_NAMES[8], "RR_thigh_joint")
        self.assertEqual(JOINT_NAMES[12], "RL_thigh_joint")
        self.assertEqual(
            JOINT_NAMES[16:],
            ("FR_wheel_joint", "FL_wheel_joint", "RR_wheel_joint", "RL_wheel_joint"),
        )

    def test_nonfinite_network_state_is_rejected(self):
        with self.assertRaises(ValueError):
            action_targets(np.full(20, np.nan))


class FakeGamepad:
    def __init__(self):
        self.command = np.zeros(3)
        self.events = set()
        self.disconnected = False

    def read(self):
        if self.disconnected:
            raise GamepadDisconnected("Test device unplugged")
        events, self.events = self.events, set()
        return self.command.copy(), events

    def press(self, event):
        self.events.add(event)


class SimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = io.StringIO()
        with contextlib.redirect_stdout(cls.output):
            cls.policy = OnnxPolicy(POLICY_PATH)

    def setUp(self):
        self.silence = contextlib.redirect_stdout(self.output)
        self.silence.__enter__()
        self.addCleanup(self.silence.__exit__, None, None, None)
        self.gamepad = FakeGamepad()
        self.sim = WheelSimulation(
            parse_args(["--init-duration", "0.04"]), self.policy, self.gamepad
        )

    def test_joint_mapping_and_servo_force_limits(self):
        sim = self.sim
        native_names = [sim.model.joint(int(j)).name for j in sim.joint_ids]
        self.assertEqual(tuple(native_names), JOINT_NAMES)
        leg_count = len(LEG_JOINT_NAMES)
        np.testing.assert_array_equal(
            sim.model.actuator_gainprm[sim.actuator_ids, 0],
            np.concatenate((KP[:leg_count], KD[leg_count:])),
        )
        np.testing.assert_array_equal(
            sim.model.actuator_biasprm[sim.actuator_ids, 2], -KD
        )
        np.testing.assert_array_equal(sim.data.qpos[sim.q_indices], DEFAULT_Q)

    def test_angular_velocity_uses_link_axes_instead_of_principal_inertia_axes(self):
        sim = self.sim
        # MuJoCo free-joint angular qvel is in the body/link frame. Exercise
        # each axis at both identity and a rotated root orientation. This E1
        # model has a non-identity principal-inertia frame, exposing BODY misuse.
        for quaternion in ([1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]):
            sim.data.qpos[3:7] = quaternion
            for axis in range(3):
                with self.subTest(quaternion=quaternion, axis=axis):
                    sim.data.qvel.fill(0.0)
                    sim.data.qvel[3 + axis] = 1.0
                    mujoco.mj_forward(sim.model, sim.data)
                    expected = np.zeros(3)
                    expected[axis] = 0.25
                    np.testing.assert_allclose(
                        sim.observation()[:3], expected, atol=1e-7
                    )

    def test_idle_stand_inference_and_damping(self):
        sim = self.sim
        sim.update()
        self.assertEqual(sim.data.time, 0.0)
        self.gamepad.press("stand")
        sim.update()
        self.assertEqual(sim.state, ControlState.GO_TO_DEFAULT)
        sim.update()
        self.assertEqual(sim.state, ControlState.STANDBY)
        self.gamepad.press("start")
        sim.update()
        self.assertEqual(sim.state, ControlState.INFERENCE)
        self.assertEqual(sim.total_policy_steps, 1)
        self.assertTrue(np.isfinite(sim.previous_action).all())
        self.gamepad.press("damp")
        sim.update()
        self.assertEqual(sim.state, ControlState.DAMPING)
        np.testing.assert_array_equal(
            sim.model.actuator_gainprm[sim.actuator_ids, 0], 0.0
        )
        np.testing.assert_array_equal(sim.data.ctrl, 0.0)

    def test_gravity_frame_does_not_change_control_mode(self):
        sim = self.sim
        sim.data.qpos[3:7] = [2**-0.5, 2**-0.5, 0, 0]
        mujoco.mj_forward(sim.model, sim.data)
        np.testing.assert_allclose(sim.gravity(), [0, -1, 0], atol=1e-10)
        self.assertAlmostEqual(sim.tilt(), 90.0)
        sim.state = ControlState.INFERENCE
        sim.update()
        self.assertEqual(sim.state, ControlState.INFERENCE)
        self.assertEqual(sim.total_policy_steps, 1)

    @patch.multiple(
        wheel_config,
        COMMAND_RATE_LIMIT_ENABLED=True,
        COMMAND_ACCEL_LIMIT=np.array([0.5, 0.6, 0.4]),
        COMMAND_DECEL_LIMIT=np.array([1.0, 1.2, 0.8]),
    )
    def test_gamepad_command_ramps_up_reverses_and_returns_to_zero(self):
        sim = self.sim
        target = np.array([0.5, -0.2, 0.3])
        self.gamepad.command[:] = target / wheel_config.COMMAND_LIMIT
        sim.state = ControlState.INFERENCE
        sim.update()
        accel_step = wheel_config.COMMAND_ACCEL_LIMIT * POLICY_DT
        decel_step = wheel_config.COMMAND_DECEL_LIMIT * POLICY_DT
        np.testing.assert_allclose(
            sim.observation()[6:9], np.clip(target, -accel_step, accel_step)
        )
        accel_ticks = int(np.ceil(np.max(np.abs(target) / accel_step)))
        decel_ticks = int(np.ceil(np.max(np.abs(target) / decel_step)))
        for _ in range(accel_ticks):
            sim.update()
        np.testing.assert_allclose(sim.observation()[6:9], target)

        self.gamepad.command[:] = -target / wheel_config.COMMAND_LIMIT
        sim.update()
        expected = target + np.clip(-target, -decel_step, decel_step)
        np.testing.assert_allclose(sim.observation()[6:9], expected)
        reached_zero = sim.command == 0.0
        for _ in range(accel_ticks + decel_ticks):
            sim.update()
            reached_zero |= sim.command == 0.0
        self.assertTrue(reached_zero.all())
        np.testing.assert_allclose(sim.observation()[6:9], -target)

        self.gamepad.command.fill(0.0)
        sim.update()
        expected = -target + np.clip(target, -decel_step, decel_step)
        np.testing.assert_allclose(sim.observation()[6:9], expected)
        for _ in range(decel_ticks):
            sim.update()
        np.testing.assert_array_equal(sim.observation()[6:9], 0.0)

    @patch.object(wheel_config, "COMMAND_RATE_LIMIT_ENABLED", False)
    def test_disabled_command_rate_limit_passes_target_directly(self):
        sim = self.sim
        sim.state = ControlState.INFERENCE
        self.gamepad.command[2] = 0.5
        sim.update()
        np.testing.assert_array_equal(sim.observation()[6:9], [0.0, 0.0, 0.75])

        self.gamepad.command[2] = -0.5
        sim.update()
        np.testing.assert_array_equal(sim.observation()[6:9], [0.0, 0.0, -0.75])

        self.gamepad.command.fill(0.0)
        sim.update()
        np.testing.assert_array_equal(sim.observation()[6:9], 0.0)

    @patch.object(wheel_config, "COMMAND_RATE_LIMIT_ENABLED", False)
    def test_full_stick_reaches_configured_velocity_limits(self):
        self.sim.state = ControlState.INFERENCE
        for stick in ([1, 1, 1], [-1, -1, -1], [.5, -.5, .5], [0, 0, 0]):
            self.gamepad.command[:] = stick
            self.sim.update()
            np.testing.assert_allclose(
                self.sim.observation()[6:9], np.array(stick) * wheel_config.COMMAND_LIMIT
            )

    def test_unplug_stops_inference_without_reusing_command(self):
        sim = self.sim
        sim.state = ControlState.INFERENCE
        self.gamepad.disconnected = True
        with self.assertRaises(GamepadDisconnected):
            sim.update()
        self.assertEqual(sim.state, ControlState.EXIT)
        self.assertEqual(sim.total_policy_steps, 0)
        np.testing.assert_array_equal(sim.command, 0.0)
        np.testing.assert_array_equal(sim.data.ctrl, 0.0)

    def test_a_cannot_start_before_standby(self):
        self.gamepad.press("start")
        self.sim.update()
        self.assertEqual(self.sim.state, ControlState.IDLE)

    def test_reset_restores_pose_and_keeps_elapsed_time(self):
        sim = self.sim
        sim.data.time = 3.0
        sim.data.qpos[0] = 5.0
        self.gamepad.command[:] = [0.5, 0.0, 0.0]
        self.gamepad.press("reset")
        sim.update()
        self.assertEqual(sim.state, ControlState.IDLE)
        self.assertEqual(sim.data.time, 3.0)
        self.assertEqual(sim.data.qpos[0], 0.0)
        np.testing.assert_array_equal(sim.command, 0.0)


if __name__ == "__main__":
    unittest.main()
