"""Check IMU frame conversion and the read-only diagnostic boundary offline."""

import contextlib
import io
import itertools
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from e1_wheel.scripts import deploy_real_e1_wheel as deploy
from e1_wheel import wheel_config


class ImuDiagnosticsTests(unittest.TestCase):
    def test_level_stationary_observation(self):
        imu = SimpleNamespace(imu_euler=[0, 0, 1.2], imu_gyro=[0, 0, 0])
        gyro, gravity = deploy.imu_observation(imu, np.eye(3))
        np.testing.assert_allclose(gyro, [0, 0, 0])
        np.testing.assert_allclose(gravity, [0, 0, -1])

    def test_pitch_and_mount_rotation_apply_to_both_inputs(self):
        imu = SimpleNamespace(imu_euler=[0, -math.pi / 2, 0], imu_gyro=[1, 2, 3])
        mount = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        gyro, gravity = deploy.imu_observation(imu, mount)
        np.testing.assert_allclose(gyro, [-2, 1, 3])
        np.testing.assert_allclose(gravity, [0, -1, 0], atol=1e-15)

    def test_monitor_mode_never_loads_policy_or_opens_motor_services(self):
        imu = SimpleNamespace(
            imu_euler=[0, 0, 0], imu_gyro=[0, 0, 0],
            imu_acc=[0, 0, 9.81], imu_stamp=123,
        )
        # The fake stub exposes only a state read, so command writes cannot pass.
        read = Mock(return_value=imu)
        output = io.StringIO()
        with (
            patch.object(deploy.sys, "argv", ["deploy", "--monitor-imu", "--duration", "0.1"]),
            patch.object(deploy.grpc, "insecure_channel"),
            patch.object(deploy, "LegServiceStub", return_value=SimpleNamespace(GetLegState=read)),
            patch.object(deploy, "MotorService", side_effect=AssertionError("motor service opened")),
            patch.object(deploy.ort, "InferenceSession", side_effect=AssertionError("policy loaded")),
            patch.object(deploy, "Gamepad", side_effect=AssertionError("gamepad opened")),
            patch.object(deploy.time, "monotonic", side_effect=[0, 0, .01, .02, .03, .2]),
            patch.object(deploy.time, "sleep"),
            contextlib.redirect_stdout(output),
        ):
            deploy.main()
        read.assert_called_once()
        self.assertIn("stamp=123", output.getvalue())
        self.assertIn("frames=1", output.getvalue())
        self.assertIn("gravity_body=[0.0, 0.0, -1.0]", output.getvalue())


class HardwareCommandTests(unittest.TestCase):
    @patch.object(wheel_config, "COMMAND_RATE_LIMIT_ENABLED", False)
    def test_full_stick_reaches_policy_without_json_scale(self):
        for stick in ([1, 1, 1], [-1, -1, -1], [.5, -.5, .5], [0, 0, 0]):
            with self.subTest(stick=stick):
                pad = SimpleNamespace(read=Mock(side_effect=[
                    (np.zeros(3), set()),
                    (np.zeros(3), {"stand"}),
                    (np.array(stick), {"start"}),
                    (np.zeros(3), {"exit"}),
                ]))
                upper, lower = Mock(), Mock()
                state = (wheel_config.DEFAULT_Q.copy(), np.zeros(20), np.zeros(3), np.array([0, 0, -1]))
                with (
                    patch.object(deploy, "read_state", return_value=state),
                    patch.object(deploy, "infer", return_value=np.zeros(20)) as infer,
                    patch.object(deploy.time, "monotonic", side_effect=itertools.count()),
                    patch.object(deploy.time, "sleep"),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    deploy.run(upper, lower, pad, None, "obs", {"stand_duration_s": 1.0}, np.eye(3))
                infer.assert_called_once()
                np.testing.assert_allclose(
                    infer.call_args.args[2][6:9], np.array(stick) * wheel_config.COMMAND_LIMIT
                )


if __name__ == "__main__":
    unittest.main()
