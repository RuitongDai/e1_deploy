"""MuJoCo deployment for RobotLab-Isaac-Velocity-Flat-E1-Wheel-v0.

The 69-D observation and 20-D mixed position/velocity actions follow RobotLab.
Only the gamepad commands velocity and the stand-up/standby/inference/damping
state machine. This module never imports the robot SDK or opens a robot connection.
"""

from __future__ import annotations

import argparse
import contextlib
import enum
import math
import signal
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from e1_wheel.build_mujoco_scene import build_scene
from e1_wheel.gamepad import Gamepad, GamepadDisconnected, monitor
from e1_wheel.wheel_config import (
    ACTION_SIZE,
    BASE_HEIGHT,
    DECIMATION,
    DEFAULT_Q,
    EFFORT_LIMIT,
    JOINT_NAMES,
    KD,
    KP,
    OBSERVATION_SIZE,
    PHYSICS_DT,
    POLICY_DT,
    POLICY_PATH,
    SCENE_PATH,
    SUBSTEPS,
    action_targets,
    build_observation,
    finite_vector,
    gamepad_command,
    limit_command_rate,
)


class ControlState(enum.Enum):
    IDLE = "idle"
    GO_TO_DEFAULT = "go_to_default"
    STANDBY = "standby"
    INFERENCE = "inference"
    DAMPING = "damping"
    EXIT = "exit"


class OnnxPolicy:
    def __init__(self, path: Path):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Policy does not exist: {path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        try:
            self.session = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as error:
            raise RuntimeError(
                f"Cannot load {path}: {error}\n"
                "For an external-data ONNX export, keep policy.onnx.data beside policy.onnx."
            ) from error
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                "Expected a feed-forward policy with one input and one output"
            )
        self.input, self.output = inputs[0], outputs[0]
        for tensor, size in (
            (self.input, OBSERVATION_SIZE),
            (self.output, ACTION_SIZE),
        ):
            if (
                len(tensor.shape) != 2
                or tensor.shape[-1] != size
                or tensor.type != "tensor(float)"
            ):
                raise ValueError(
                    f"Expected float32 (*,{size}); got {tensor.name} {tensor.shape} {tensor.type}"
                )
            if isinstance(tensor.shape[0], int) and tensor.shape[0] != 1:
                raise ValueError(f"Expected batch size 1, got {tensor.shape}")
        self(np.zeros(OBSERVATION_SIZE, dtype=np.float32))
        print(
            f"[INFO] ONNX: {self.input.name}=(1,69), {self.output.name}=(1,20), onnxruntime {ort.__version__}"
        )

    def __call__(self, observation) -> np.ndarray:
        observation = finite_vector(
            observation, OBSERVATION_SIZE, "observation"
        ).astype(np.float32)
        action = self.session.run(
            [self.output.name], {self.input.name: observation[None, :]}
        )[0]
        if action.shape != (1, ACTION_SIZE):
            raise ValueError(f"Unexpected action shape: {action.shape}")
        # Preserve raw network output for last_action; target clipping happens later.
        return finite_vector(action[0], ACTION_SIZE, "network output").copy()


class WheelSimulation:
    def __init__(self, args, policy: OnnxPolicy, gamepad: Gamepad):
        self.args, self.policy = args, policy
        if args.rebuild_model or not args.model.is_file():
            if args.model != SCENE_PATH:
                raise FileNotFoundError(
                    "Custom --model must already exist; rebuild the default scene separately"
                )
            build_scene()
        self.model = mujoco.MjModel.from_xml_path(str(args.model))
        self.model.opt.timestep = PHYSICS_DT
        self.data = mujoco.MjData(self.model)
        self.base_id = self.model.body("base_link").id
        self.joint_ids = np.array([self.model.joint(name).id for name in JOINT_NAMES])
        self.q_indices = self.model.jnt_qposadr[self.joint_ids]
        self.dq_indices = self.model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = np.array(
            [self.model.actuator(f"servo_{name}").id for name in JOINT_NAMES]
        )
        if self.model.nq != 27 or self.model.nv != 26 or self.model.nu != 20:
            raise ValueError(
                "Scene must contain a floating base, 16 leg joints and 4 wheel joints"
            )
        if not np.array_equal(
            self.model.actuator_trnid[self.actuator_ids, 0], self.joint_ids
        ):
            raise ValueError("Actuator/joint mapping does not match the policy order")
        if not np.all(
            self.model.actuator_forcelimited[self.actuator_ids]
        ) or not np.allclose(
            self.model.actuator_forcerange[self.actuator_ids],
            np.column_stack((-EFFORT_LIMIT, EFFORT_LIMIT)),
        ):
            raise ValueError(
                "Actuator effort limits must match the training configuration"
            )
        if self.model.joint("floating_base").qposadr[0] != 0:
            raise ValueError("floating_base must be the first free joint")
        self.gamepad = gamepad
        self.stop_requested = False
        self.last_input_log = 0.0
        self.records = []
        self.total_policy_steps = 0
        self.min_height = BASE_HEIGHT
        self.reset()
        print(
            f"[INFO] MuJoCo {mujoco.__version__}: physics={1 / PHYSICS_DT:g} Hz, policy=50 Hz, 16 position + 4 velocity servos"
        )

    def set_servos(self, damping: bool = False):
        # MuJoCo affine servos integrate PD damping implicitly (implicitfast).
        kp = np.zeros(ACTION_SIZE) if damping else KP
        kd = KD
        gain = np.zeros(ACTION_SIZE) if damping else np.concatenate((kp[:16], kd[16:]))
        self.model.actuator_gainprm[self.actuator_ids, 0] = gain
        self.model.actuator_biasprm[self.actuator_ids, 1] = -kp
        self.model.actuator_biasprm[self.actuator_ids, 2] = -kd

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = (0.0, 0.0, BASE_HEIGHT)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[self.q_indices] = DEFAULT_Q
        self.previous_action = np.zeros(ACTION_SIZE)
        self.interpolation_time = 0.0
        self.interpolation_start = DEFAULT_Q[:16].copy()
        self.command = np.zeros(3)
        self.set_servos()
        self.data.ctrl[self.actuator_ids] = np.concatenate(
            (DEFAULT_Q[:16], np.zeros(4))
        )
        self.state = ControlState.IDLE
        mujoco.mj_forward(self.model, self.data)
        print(
            "[STATE] IDLE (simulation paused). Gamepad only: X=stand, A=inference, Y=damping, B=exit, Back=reset"
        )

    def gravity(self):
        return self.data.xmat[self.base_id].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])

    def tilt(self):
        return math.degrees(math.acos(float(np.clip(-self.gravity()[2], -1.0, 1.0))))

    def base_velocity(self):
        """Angular then linear velocity, both expressed in the base link frame."""
        velocity = np.zeros(6)
        # BODY + local=1 returns principal-inertia-frame components (ximat).
        # RobotLab observes angular velocity in the root link frame (xmat).
        # The E1 base inertial quaternion is not identity, so these differ.
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, self.base_id, velocity, 1
        )
        return velocity

    def observation(self):
        velocity = self.base_velocity()
        return build_observation(
            velocity[:3],
            self.gravity(),
            self.command,
            self.data.qpos[self.q_indices],
            self.data.qvel[self.dq_indices],
            self.previous_action,
        )

    def start_stand(self):
        self.interpolation_start = self.data.qpos[self.q_indices[:16]].copy()
        self.interpolation_time = 0.0
        self.command.fill(0.0)
        self.previous_action.fill(0.0)
        self.set_servos()
        self.state = ControlState.GO_TO_DEFAULT
        print("[STATE] GO_TO_DEFAULT")

    def enter_damping(self):
        self.state = ControlState.DAMPING
        self.set_servos(damping=True)
        self.command.fill(0.0)
        self.previous_action.fill(0.0)
        self.data.ctrl.fill(0.0)
        print("[STATE] DAMPING")

    def update(self):
        if self.stop_requested:
            self.enter_damping()
            self.state = ControlState.EXIT
            return
        try:
            command, events = self.gamepad.read()
        except GamepadDisconnected:
            self.enter_damping()
            self.state = ControlState.EXIT
            raise
        if self.state is ControlState.INFERENCE:
            target_command = gamepad_command(command)
            self.command = limit_command_rate(self.command, target_command)
        now = time.monotonic()
        if events or (self.args.gamepad_debug and now - self.last_input_log >= 0.2):
            velocity = self.base_velocity()
            print(
                f"[GAMEPAD] events={sorted(events)}, cmd[vx,vy,wz]={np.round(self.command, 3).tolist()}, state={self.state.value}",
                flush=True,
            )
            if self.args.gamepad_debug:
                print(
                    f"[STATE] measured_v_body={np.round(velocity[3:], 3).tolist()}, "
                    f"measured_w_body={np.round(velocity[:3], 3).tolist()}, "
                    f"wheel_targets_rad_s={np.round(self.data.ctrl[self.actuator_ids[16:]], 3).tolist()}",
                    flush=True,
                )
            self.last_input_log = now
        if "exit" in events:
            self.enter_damping()
            self.state = ControlState.EXIT
            return
        if "reset" in events:
            # Back resets the pose; preserve elapsed time for recordings.
            elapsed = self.data.time
            self.reset()
            self.data.time = elapsed
            return
        if "damp" in events:
            self.enter_damping()
        elif "stand" in events:
            self.start_stand()
        elif "start" in events:
            if self.state is ControlState.STANDBY:
                self.command.fill(0.0)
                self.previous_action.fill(0.0)
                self.state = ControlState.INFERENCE
                print("[STATE] INFERENCE")
            else:
                print("[GAMEPAD] A ignored: press X and wait for STANDBY first")

        if self.state is ControlState.IDLE:
            return
        if self.state is ControlState.GO_TO_DEFAULT:
            self.interpolation_time += POLICY_DT
            fraction = min(self.interpolation_time / self.args.init_duration, 1.0)
            alpha = 0.5 * (1.0 - math.cos(math.pi * fraction))
            target = (1.0 - alpha) * self.interpolation_start + alpha * DEFAULT_Q[:16]
            self.data.ctrl[self.actuator_ids] = np.concatenate((target, np.zeros(4)))
            if fraction >= 1.0:
                self.state = ControlState.STANDBY
                print("[STATE] STANDBY")
        elif self.state is ControlState.STANDBY:
            self.data.ctrl[self.actuator_ids] = np.concatenate(
                (DEFAULT_Q[:16], np.zeros(4))
            )
        elif self.state is ControlState.INFERENCE:
            observation = self.observation()
            action = self.policy(observation)
            self.data.ctrl[self.actuator_ids] = action_targets(action)
            self.previous_action[:] = action
            self.total_policy_steps += 1
            if self.args.record:
                self.records.append(
                    (
                        self.data.time,
                        observation.copy(),
                        action.copy(),
                        self.data.qpos[:7].copy(),
                        self.data.qpos[self.q_indices].copy(),
                        self.data.qvel[self.dq_indices].copy(),
                    )
                )
        mujoco.mj_step(self.model, self.data, nstep=DECIMATION * SUBSTEPS)
        mujoco.mj_forward(self.model, self.data)
        for warning in (
            mujoco.mjtWarning.mjWARN_BADQPOS,
            mujoco.mjtWarning.mjWARN_BADQVEL,
            mujoco.mjtWarning.mjWARN_BADQACC,
        ):
            if self.data.warning[warning].number:
                raise RuntimeError(f"MuJoCo reported unstable dynamics: {warning.name}")
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(
            np.isfinite(self.data.qvel)
        ):
            raise RuntimeError("Non-finite simulation state")
        self.min_height = min(self.min_height, float(self.data.qpos[2]))

    def save_recording(self):
        if not self.args.record:
            return
        path = self.args.record
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.records:
            times, observations, actions, base_poses, positions, velocities = map(
                np.asarray, zip(*self.records)
            )
        else:
            times = np.empty(0)
            observations, actions = np.empty((0, 69)), np.empty((0, 20))
            base_poses, positions, velocities = (
                np.empty((0, 7)),
                np.empty((0, 20)),
                np.empty((0, 20)),
            )
        # File handle prevents NumPy silently adding a different extension to --record.
        with path.open("wb") as stream:
            np.savez_compressed(
                stream,
                time=times,
                observations=observations,
                actions=actions,
                base_pose=base_poses,
                joint_position=positions,
                joint_velocity=velocities,
                joint_names=np.array(JOINT_NAMES),
            )
        print(f"[INFO] Recording: {path} ({len(times)} policy samples)")

    def run(self):
        viewer_context = contextlib.nullcontext(None)
        if not self.args.headless:
            import mujoco.viewer

            viewer_context = mujoco.viewer.launch_passive(self.model, self.data)

        def request_stop(*_):
            self.stop_requested = True

        previous_signals = {
            sig: signal.signal(sig, request_stop)
            for sig in (signal.SIGINT, signal.SIGTERM)
        }
        wall_start = time.perf_counter()
        try:
            with viewer_context as viewer:
                if viewer:
                    viewer.cam.distance = 1.7
                    viewer.cam.elevation = -20.0
                    viewer.cam.azimuth = 135.0
                while self.state is not ControlState.EXIT:
                    if viewer and not viewer.is_running():
                        break
                    if (
                        self.args.duration > 0.0
                        and self.data.time >= self.args.duration - 1e-9
                    ):
                        break
                    tick_start = time.perf_counter()
                    with viewer.lock() if viewer else contextlib.nullcontext():
                        self.update()
                        if viewer:
                            viewer.cam.lookat[:] = self.data.xpos[self.base_id]
                    if viewer:
                        viewer.sync()
                    # Hardware input must be sampled in real time, including headless mode.
                    time.sleep(max(0.0, POLICY_DT - (time.perf_counter() - tick_start)))
        finally:
            self.set_servos(damping=True)
            self.data.ctrl.fill(0.0)
            self.save_recording()
            for sig, handler in previous_signals.items():
                signal.signal(sig, handler)
        print(
            f"[RESULT] sim_time={self.data.time:.2f}s, wall_time={time.perf_counter() - wall_start:.2f}s, "
            f"policy_steps={self.total_policy_steps}, state={self.state.value}, "
            f"base_xyz={np.round(self.data.qpos[:3], 3).tolist()}, min_height={self.min_height:.3f}m"
        )
        return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    parser.add_argument("--model", type=Path, default=SCENE_PATH)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="No viewer; still controlled only by the gamepad in real time",
    )
    # Accepted for existing launch commands; gamepad is now always required.
    parser.add_argument("--gamepad", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--gamepad-index",
        type=int,
        default=0,
        help="SDL joystick device index (default 0)",
    )
    parser.add_argument(
        "--test-gamepad",
        action="store_true",
        help="Only print gamepad axes/buttons; no ONNX or simulation",
    )
    parser.add_argument(
        "--gamepad-debug",
        action="store_true",
        help="Print gamepad command and control state at 5 Hz",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Simulation seconds; 0 means unlimited (input monitor: wall seconds)",
    )
    parser.add_argument(
        "--init-duration",
        type=float,
        default=2.0,
        help="Stand-up interpolation duration",
    )
    parser.add_argument(
        "--record",
        type=Path,
        help="Save policy inputs, raw actions and robot states as NPZ",
    )
    parser.add_argument(
        "--rebuild-model",
        action="store_true",
        help="Regenerate scene.xml from the training URDF",
    )
    parser.add_argument(
        "--validate-policy-only",
        action="store_true",
        help="Only check ONNX; does not require a gamepad",
    )
    args = parser.parse_args(argv)
    if args.test_gamepad and args.validate_policy_only:
        parser.error("Choose either --test-gamepad or --validate-policy-only")
    if args.gamepad_index < 0:
        parser.error("--gamepad-index must be non-negative")
    if not np.isfinite(args.init_duration) or args.init_duration <= 0.0:
        parser.error("--init-duration must be finite and positive")
    if not np.isfinite(args.duration) or args.duration < 0.0:
        parser.error("--duration must be finite and non-negative")
    args.model = args.model.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.validate_policy_only:
            OnnxPolicy(args.policy)
            print(
                "[INFO] Policy validation passed (no simulator or robot connection opened)"
            )
            return 0
        with Gamepad(args.gamepad_index) as gamepad:
            if args.test_gamepad:
                monitor(gamepad, args.duration)
                return 0
            policy = OnnxPolicy(args.policy)
            return WheelSimulation(args, policy, gamepad).run()
    except (RuntimeError, ValueError, OSError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
