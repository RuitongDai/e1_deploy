"""DroidUp SDK deployment for the phase-free E1 Dog 16-DOF AMP policy.

The policy interface is identical to
``AMP_for_E1_Vision/sim2sim/sim2sim_e1_dog_16dof_amp.py``:

* ONNX input ``obs``: six chronological 57-D frames (342 values);
* one frame: command(3), angular velocity(3), projected gravity(3),
  joint position error(16), joint velocity(16), previous action(16);
* ONNX output: 16 joint-position actions;
* 50 Hz policy loop and the robot's low-level PD controller;
* no gait phase, clock, depth, or 12-to-16-DOF joint slicing.

The upper SDK service owns FL/FR (8 motors), while the lower service owns
RL/RR (8 motors), matching the successfully deployed E1 Dog 16-DOF path.
Always validate the same ONNX in MuJoCo before enabling the real robot.
"""

from __future__ import annotations

import argparse
import contextlib
import enum
import math
import os
import select
import signal
import sys
import termios
import threading
import time
import tty
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ACTION_SIZE = 16
FRAME_SIZE = 57
HISTORY = 6
OBSERVATION_SIZE = FRAME_SIZE * HISTORY

DT = 0.005
DECIMATION = 4
POLICY_DT = DT * DECIMATION

COMMAND_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
ANGULAR_VELOCITY_SCALE = 0.25
JOINT_VELOCITY_SCALE = 0.05
ACTION_SCALE = 0.25

MAX_FORWARD_VELOCITY = 0.45
MAX_BACKWARD_VELOCITY = 0.45
MAX_LATERAL_VELOCITY = 0.35
MAX_YAW_VELOCITY = 1.2
KEYBOARD_COMMAND_STEP = 0.1
COMMAND_SLEW_RATE = np.array([0.01, 0.01, 0.02], dtype=np.float32)

LEG_JOINTS = (
    "FL_thigh_joint", "FL_hip_roll_joint", "FL_hip_yaw_joint", "FL_calf_joint",
    "FR_thigh_joint", "FR_hip_roll_joint", "FR_hip_yaw_joint", "FR_calf_joint",
    "RL_thigh_joint", "RL_hip_roll_joint", "RL_hip_yaw_joint", "RL_calf_joint",
    "RR_thigh_joint", "RR_hip_roll_joint", "RR_hip_yaw_joint", "RR_calf_joint",
)
UPPER_JOINTS = LEG_JOINTS[:8]
LOWER_JOINTS = LEG_JOINTS[8:]
UPPER_SLICE = slice(0, 8)
LOWER_SLICE = slice(8, 16)

DEFAULT_Q = np.array([0.70, 0.0, 0.0, -1.40] * 4, dtype=np.float32)
KP = np.array([60.0, 40.0, 30.0, 60.0] * 4, dtype=np.float32)
KD = np.array([3.0, 2.0, 1.5, 3.0] * 4, dtype=np.float32)
EFFORT_LIMIT = np.array([60.0, 36.0, 15.0, 60.0] * 4, dtype=np.float32)


class ControlState(enum.Enum):
    IDLE = "idle"                       # Read-only; no motor commands
    GO_TO_DEFAULT = "go_to_default"     # Cosine interpolation to DEFAULT_Q
    STANDBY = "standby"                 # Hold DEFAULT_Q; wait for A
    INFERENCE = "inference"             # Run the policy at 50 Hz
    DAMPING = "damping"                 # Zero Kp, keep Kd (damping)
    EXIT = "exit"


class SafetyCfg:
    # Same convention as the 21-DOF deployment: 60 deg for 3 frames OR 80 deg
    # in a single frame triggers automatic damping.
    fall_tilt_threshold_deg = 60.0
    fall_hard_tilt_threshold_deg = 80.0
    fall_detection_frames = 3
    # Do not apply high-gain stand-up while the robot is still lying down.
    recovery_max_tilt_deg = 30.0
    # Fraction of a second to blend policy actions in.
    fade_in_steps = 100


def _tensor_shape(value_info) -> tuple[int | None, ...]:
    return tuple(
        dimension.dim_value if dimension.HasField("dim_value") else None
        for dimension in value_info.type.tensor_type.shape.dim
    )


def _clip_command(command: Sequence[float]) -> np.ndarray:
    command = np.asarray(command, dtype=np.float32).copy()
    if command.shape != (3,):
        raise ValueError(f"Command must contain three values, got {command.shape}")
    command[0] = np.clip(command[0], -MAX_BACKWARD_VELOCITY, MAX_FORWARD_VELOCITY)
    command[1] = np.clip(command[1], -MAX_LATERAL_VELOCITY, MAX_LATERAL_VELOCITY)
    command[2] = np.clip(command[2], -MAX_YAW_VELOCITY, MAX_YAW_VELOCITY)
    return command


def _slew_command(last: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = np.clip(target - last, -COMMAND_SLEW_RATE, COMMAND_SLEW_RATE)
    return _clip_command(last + delta)


def _gamepad_command(gamepad: Any) -> np.ndarray:
    """Map the shared E1 gamepad physical ranges to this training task."""
    pad_x, pad_y, pad_yaw = gamepad.get_commands()
    return _clip_command(
        (
            float(pad_x) * MAX_FORWARD_VELOCITY,
            float(pad_y) / 0.5 * MAX_LATERAL_VELOCITY,
            float(pad_yaw) * MAX_YAW_VELOCITY,
        )
    )


class KeyboardCommand:
    """Keyboard substitute for the gamepad.

    Velocity keys:
        W/S forward/back, A/D left/right, J/L yaw, R zero.
    Button keys (edge triggered):
        Z -> X (goto default pose)
        X -> A (start inference)
        C -> Y (enter damping)
        Q -> B (exit)
    """

    def __init__(self) -> None:
        self._command = np.zeros(3, dtype=np.float32)
        self._running = True
        self._lock = threading.Lock()
        self._button_pulses = {"a": False, "b": False, "x": False, "y": False}

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def on_key(self, key: int) -> None:
        try:
            char = chr(key).lower()
        except (TypeError, ValueError):
            return
        with self._lock:
            if char == "w":
                self._command[0] += KEYBOARD_COMMAND_STEP
            elif char == "s":
                self._command[0] -= KEYBOARD_COMMAND_STEP
            elif char == "a":
                self._command[1] += KEYBOARD_COMMAND_STEP
            elif char == "d":
                self._command[1] -= KEYBOARD_COMMAND_STEP
            elif char == "j":
                self._command[2] += KEYBOARD_COMMAND_STEP
            elif char == "l":
                self._command[2] -= KEYBOARD_COMMAND_STEP
            elif char == "r":
                self._command.fill(0.0)
            elif char == "z":
                self._button_pulses["x"] = True
            elif char == "x":
                self._button_pulses["a"] = True
            elif char == "c":
                self._button_pulses["y"] = True
            elif char == "q":
                self._button_pulses["b"] = True
                self._running = False
                return
            self._command[:] = _clip_command(self._command)

    def get_command(self) -> np.ndarray:
        with self._lock:
            return self._command.copy()

    def consume_button(self, name: str) -> bool:
        """Return True once, then clear the pulse."""
        with self._lock:
            if self._button_pulses.get(name, False):
                self._button_pulses[name] = False
                return True
            return False


class TerminalKeyboard:
    def __init__(self, command: KeyboardCommand) -> None:
        self.command = command
        self.fd = sys.stdin.fileno()
        self.saved_attributes = None

    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("--keyboard requires a TTY terminal")
        self.saved_attributes = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def poll(self) -> None:
        while select.select([self.fd], [], [], 0.0)[0]:
            value = os.read(self.fd, 1)
            if not value:
                return
            self.command.on_key(value[0])

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.saved_attributes is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved_attributes)


class PhaseFreeAmpOnnxPolicy:
    """Strict one-input/one-output ONNX actor used by sim2sim and sim2real."""

    def __init__(self, policy_path: Path, *, allow_reference: bool = False) -> None:
        if policy_path.suffix.lower() != ".onnx":
            raise ValueError(f"Expected an ONNX policy, got: {policy_path}")
        if not policy_path.is_file():
            raise FileNotFoundError(f"Policy does not exist: {policy_path}")

        import onnx

        model = onnx.load(str(policy_path))
        onnx.checker.check_model(model)
        inputs = list(model.graph.input)
        outputs = list(model.graph.output)
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                f"Expected one ONNX input/output, got {len(inputs)}/{len(outputs)}"
            )
        input_shape = _tensor_shape(inputs[0])
        output_shape = _tensor_shape(outputs[0])
        if input_shape[-1:] != (OBSERVATION_SIZE,):
            raise ValueError(
                f"Expected ONNX observation (*,{OBSERVATION_SIZE}), got {input_shape}"
            )
        if output_shape[-1:] != (ACTION_SIZE,):
            raise ValueError(
                f"Expected ONNX action (*,{ACTION_SIZE}), got {output_shape}"
            )

        try:
            import onnxruntime as ort
        except ImportError as error:
            if not allow_reference:
                raise RuntimeError(
                    "Real-robot 50 Hz deployment requires onnxruntime; the slow "
                    "ONNX ReferenceEvaluator is allowed only with "
                    "--validate-policy-only."
                ) from error
            from onnx.reference import ReferenceEvaluator

            self.session = ReferenceEvaluator(model)
            self.backend = "onnx.reference (validation only)"
            self.input_name = inputs[0].name
            self.output_name = outputs[0].name
        else:
            self.session = ort.InferenceSession(
                str(policy_path), providers=["CPUExecutionProvider"]
            )
            self.backend = f"onnxruntime {ort.__version__}"
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

        # Reject incompatible models before any SDK connection is opened.
        action = self(np.zeros(OBSERVATION_SIZE, dtype=np.float32))
        print(
            f"[INFO] ONNX: {self.input_name}=(*,{OBSERVATION_SIZE}), "
            f"{self.output_name}=(*,{ACTION_SIZE}), backend={self.backend}"
        )
        if action.shape != (ACTION_SIZE,):
            raise RuntimeError(f"Unexpected ONNX action shape: {action.shape}")

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        if observation.shape != (1, OBSERVATION_SIZE):
            raise ValueError(
                f"Expected observation (1,{OBSERVATION_SIZE}), got {observation.shape}"
            )
        action = self.session.run(
            [self.output_name], {self.input_name: observation}
        )[0]
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (ACTION_SIZE,) or not np.all(np.isfinite(action)):
            raise RuntimeError(f"Invalid policy action: shape={action.shape}")
        return np.clip(action, -100.0, 100.0)


class E1Dog16DOFAmpController:
    def __init__(self, args: argparse.Namespace) -> None:
        # Load and validate policy before connecting to the robot.
        self.policy = PhaseFreeAmpOnnxPolicy(args.policy)

        from droidup.client.armbase import ArmBase
        from droidup.client.grpcbase import NanoSleep
        from droidup.client.legbase import LegBase

        self.nano_sleep_cls = NanoSleep
        self.upper = ArmBase(_host=args.host, _port=args.upper_port)
        self.lower = LegBase(_host=args.host, _port=args.lower_port)
        self.init_duration = float(args.init_duration)
        self.use_command_slew = bool(args.command_slew)
        self.fixed_command = _clip_command(args.command)
        self.command = self.fixed_command.copy()
        self.previous_action = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.history = np.zeros((HISTORY, FRAME_SIZE), dtype=np.float32)
        self.last_target_q = DEFAULT_Q.copy()
        self.state = ControlState.IDLE
        self.running = True
        self.inference_step = 0

        self.keyboard = KeyboardCommand() if args.keyboard else None
        if args.gamepad and not args.keyboard:
            from gamepad_controller import GamepadController

            self.gamepad = GamepadController(deadzone=args.gamepad_deadzone)
        else:
            self.gamepad = None

        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)

        # The SDK already exposes all 16 motors: upper FL/FR, lower RL/RR.
        self.upper.set_joint_parameters(
            KP[UPPER_SLICE].tolist(),
            KD[UPPER_SLICE].tolist(),
            EFFORT_LIMIT[UPPER_SLICE].tolist(),
        )
        self.lower.set_joint_parameters(
            KP[LOWER_SLICE].tolist(),
            KD[LOWER_SLICE].tolist(),
            EFFORT_LIMIT[LOWER_SLICE].tolist(),
        )

    # ------------------------------------------------------------------
    # Signal / state helpers
    # ------------------------------------------------------------------
    def _stop(self, _signum=None, _frame=None) -> None:
        print("\n[INFO] Exit requested; switching to damping mode")
        self.running = False

    @staticmethod
    def _fixed_length(
        values, size: int, fallback: np.ndarray | None = None
    ) -> np.ndarray:
        values = np.asarray(list(values)[:size], dtype=np.float32)
        if values.shape == (size,):
            return values
        output = np.zeros(size, dtype=np.float32)
        if fallback is not None and fallback.shape == (size,):
            output[:] = fallback
        output[: values.size] = values
        return output

    def _get_state(self):
        upper = self.upper.get_joint_state()
        lower = self.lower.get_joint_state()
        upper_q = self._fixed_length(
            upper.position, 8, self.last_target_q[UPPER_SLICE]
        )
        lower_q = self._fixed_length(
            lower.position, 8, self.last_target_q[LOWER_SLICE]
        )
        upper_dq = self._fixed_length(upper.velocity, 8)
        lower_dq = self._fixed_length(lower.velocity, 8)
        return (
            np.concatenate((upper_q, lower_q)),
            np.concatenate((upper_dq, lower_dq)),
            self._fixed_length(lower.imu_gyro, 3),
            self._fixed_length(lower.imu_euler, 3),
        )

    @staticmethod
    def _projected_gravity(euler_xyz: Sequence[float]) -> np.ndarray:
        roll, pitch, yaw = [float(value) for value in euler_xyz[:3]]
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        rotation = np.array(
            (
                (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
                (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
                (-sp, cp * sr, cp * cr),
            ),
            dtype=np.float32,
        )
        return rotation.T @ np.array((0.0, 0.0, -1.0), dtype=np.float32)

    def _torso_tilt_deg(self) -> float:
        """Yaw-invariant torso tilt derived from IMU Euler angles."""
        try:
            _, _, _, euler = self._get_state()
        except Exception:
            return float("inf")
        gravity = self._projected_gravity(euler)
        norm = float(np.linalg.norm(gravity))
        if norm < 1.0e-6:
            return float("inf")
        upright_cos = float(np.clip(-gravity[2] / norm, -1.0, 1.0))
        return math.degrees(math.acos(upright_cos))

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    def _update_command(self) -> bool:
        target = self.fixed_command
        if self.keyboard is not None:
            target = self.keyboard.get_command()
            if not self.keyboard.running:
                self.running = False
                return False
        elif self.gamepad is not None:
            target = _gamepad_command(self.gamepad)
        target = _clip_command(target)
        if self.use_command_slew:
            self.command[:] = _slew_command(self.command, target)
        else:
            self.command[:] = target
        return True

    def _read_buttons(self) -> tuple[bool, bool, bool, bool]:
        """Return (A, B, X, Y) held state.

        Gamepad buttons are level-triggered. Keyboard buttons return True
        exactly once per key press (pulse consumed on read).
        """
        if self.gamepad is not None:
            update = getattr(self.gamepad, "update", None)
            if callable(update):
                try:
                    update()
                except Exception:
                    pass
            return (
                bool(self.gamepad.get_button_a()),
                bool(self.gamepad.get_button_b()),
                bool(self.gamepad.get_button_x()),
                bool(self.gamepad.get_button_y()),
            )
        if self.keyboard is not None:
            return (
                self.keyboard.consume_button("a"),
                self.keyboard.consume_button("b"),
                self.keyboard.consume_button("x"),
                self.keyboard.consume_button("y"),
            )
        return False, False, False, False

    # ------------------------------------------------------------------
    # Observation / action
    # ------------------------------------------------------------------
    def _observation(self, q, dq, gyro, euler) -> np.ndarray:
        frame = np.concatenate(
            (
                self.command * COMMAND_SCALE,
                gyro * ANGULAR_VELOCITY_SCALE,
                self._projected_gravity(euler),
                q - DEFAULT_Q,
                dq * JOINT_VELOCITY_SCALE,
                self.previous_action,
            )
        ).astype(np.float32)
        if frame.shape != (FRAME_SIZE,) or not np.all(np.isfinite(frame)):
            raise RuntimeError(f"Invalid observation frame: {frame.shape}")

        # Training deque and sim2sim flatten from oldest to newest.
        self.history[:-1] = self.history[1:]
        self.history[-1] = frame
        observation = self.history.reshape(OBSERVATION_SIZE)
        return np.clip(observation, -100.0, 100.0)

    def _policy_target(self, state) -> np.ndarray:
        q, dq, gyro, euler = state
        action = self.policy(self._observation(q, dq, gyro, euler))
        self.previous_action[:] = action
        self.inference_step += 1
        # Blend from the default pose into the raw policy target to avoid a
        # step input at the very first inference frame.
        alpha = min(self.inference_step / float(SafetyCfg.fade_in_steps), 1.0)
        return DEFAULT_Q + alpha * (ACTION_SCALE * action)

    def _reset_policy_state(self) -> None:
        self.previous_action.fill(0.0)
        self.history.fill(0.0)
        self.command.fill(0.0)
        self.inference_step = 0

    def _current_positions(self) -> np.ndarray:
        try:
            return self._get_state()[0]
        except Exception:
            return self.last_target_q.copy()

    # ------------------------------------------------------------------
    # Motor commands
    # ------------------------------------------------------------------
    def _set_positions(self, target_q: np.ndarray) -> None:
        target_q = np.asarray(target_q, dtype=np.float32)
        if target_q.shape != (ACTION_SIZE,) or not np.all(np.isfinite(target_q)):
            raise RuntimeError(f"Invalid joint target: {target_q.shape}")
        self.upper.set_joint_positions(target_q[UPPER_SLICE].tolist())
        self.lower.set_joint_positions(target_q[LOWER_SLICE].tolist())
        self.last_target_q[:] = target_q

    def _set_damping_mode(self) -> None:
        self.upper.set_joint_parameters(
            np.zeros(8, dtype=np.float32).tolist(),
            KD[UPPER_SLICE].tolist(),
            EFFORT_LIMIT[UPPER_SLICE].tolist(),
        )
        self.lower.set_joint_parameters(
            np.zeros(8, dtype=np.float32).tolist(),
            KD[LOWER_SLICE].tolist(),
            EFFORT_LIMIT[LOWER_SLICE].tolist(),
        )
        self.upper.set_joint_positions(self.last_target_q[UPPER_SLICE].tolist())
        self.lower.set_joint_positions(self.last_target_q[LOWER_SLICE].tolist())

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        print("\n" + "=" * 64)
        print("E1 Dog 16DOF Phase-Free AMP Sim2Real")
        print("  [X] : Goto default pose (standby)")
        print("  [A] : Start Inference (only from standby)")
        print("  [Y] : Enter damping (from standby / inference)")
        print("  [B] : Damping stop and exit")
        print("  Auto: Fall detection enters damping during inference")
        print("  Startup IDLE is read-only; motor commands remain untouched")
        print("=" * 64 + "\n")

        self.state = ControlState.IDLE
        timer = self.nano_sleep_cls(POLICY_DT * 1000.0)

        keyboard_context = (
            TerminalKeyboard(self.keyboard)
            if self.keyboard is not None
            else contextlib.nullcontext()
        )

        if self.keyboard is not None:
            print(
                "[INFO] Keyboard: W/S=fwd/back A/D=left/right J/L=yaw R=zero\n"
                "       Z=X(goto default)  X=A(start)  C=Y(damping)  Q=B(exit)"
            )
        elif self.gamepad is not None:
            print("[INFO] Gamepad: X=standby  A=start  Y=damping  B=exit")
        else:
            print(f"[INFO] Fixed command: {self.fixed_command.tolist()}")

        # Cosine interpolation state.
        q_interp_start = DEFAULT_Q.copy()
        t_interp = 0.0
        T_interp = 2.0

        target_q_real = DEFAULT_Q.copy()
        prev_btn_a = prev_btn_b = prev_btn_x = prev_btn_y = False
        fall_tilt_counter = 0

        with keyboard_context:
            while self.state is not ControlState.EXIT:
                step_start = time.perf_counter()

                # 1) Keyboard poll (consumes bytes and updates button pulses).
                if self.keyboard is not None:
                    keyboard_context.poll()

                # 2) Refresh robot state.
                try:
                    current_state = self._get_state()
                except Exception as exc:
                    print(f"[WARN] get_state failed: {exc}")
                    current_state = None

                # 3) Velocity command handling.
                self._update_command()
                if not self.running:
                    print("\n[INFO] Exit requested; switching to damping mode")
                    self.state = ControlState.DAMPING

                # 4) Buttons with edge detection.
                btn_a, btn_b, btn_x, btn_y = self._read_buttons()
                btn_a_pressed = btn_a and not prev_btn_a
                btn_b_pressed = btn_b and not prev_btn_b
                btn_x_pressed = btn_x and not prev_btn_x
                btn_y_pressed = btn_y and not prev_btn_y
                prev_btn_a, prev_btn_b, prev_btn_x, prev_btn_y = (
                    btn_a, btn_b, btn_x, btn_y,
                )

                # 5) Emergency exit on B (from any state).
                if btn_b_pressed:
                    print("\n" + "=" * 55)
                    print("[EMERGENCY STOP] B pressed: damping and exit")
                    print("=" * 55 + "\n")
                    self.state = ControlState.DAMPING
                    try:
                        self._set_damping_mode()
                    except Exception:
                        pass
                    time.sleep(0.1)
                    break

                # 6) Manual damping via Y (standby/inference).
                if btn_y_pressed and self.state in (
                    ControlState.STANDBY,
                    ControlState.INFERENCE,
                ):
                    prev_state = self.state
                    self.state = ControlState.DAMPING
                    self.command.fill(0.0)
                    fall_tilt_counter = 0
                    self._set_damping_mode()
                    print(f"\n[SAFETY] Y pressed in {prev_state.name}: DAMPING")

                # 7) Automatic fall detection during inference.
                if self.state is ControlState.INFERENCE:
                    tilt = self._torso_tilt_deg()
                    if not math.isfinite(tilt):
                        fall_tilt_counter = SafetyCfg.fall_detection_frames
                    elif tilt >= SafetyCfg.fall_hard_tilt_threshold_deg:
                        fall_tilt_counter = SafetyCfg.fall_detection_frames
                    elif tilt >= SafetyCfg.fall_tilt_threshold_deg:
                        fall_tilt_counter += 1
                    else:
                        fall_tilt_counter = 0

                    if fall_tilt_counter >= SafetyCfg.fall_detection_frames:
                        tilt_text = (
                            f"{tilt:.1f} deg" if math.isfinite(tilt) else "invalid IMU"
                        )
                        print(f"\n[SAFETY] Fall detected ({tilt_text}): DAMPING")
                        self.state = ControlState.DAMPING
                        self.command.fill(0.0)
                        self._set_damping_mode()
                else:
                    fall_tilt_counter = 0

                # 8) FSM.
                if self.state is ControlState.IDLE:
                    # Read-only: do not touch motor commands until X is pressed.
                    if btn_x_pressed:
                        print("[INFO] X pressed: interpolating to default pose")
                        self._reset_policy_state()
                        q_interp_start = self._current_positions()
                        t_interp = 0.0
                        self.state = ControlState.GO_TO_DEFAULT

                elif self.state is ControlState.GO_TO_DEFAULT:
                    st = min(t_interp / T_interp, 1.0)
                    s0 = 0.5 * (1.0 + math.cos(math.pi * st))
                    s1 = 1.0 - s0
                    target_q_real = s0 * q_interp_start + s1 * DEFAULT_Q
                    try:
                        self._set_positions(target_q_real)
                    except Exception as exc:
                        print(f"[WARN] set_positions failed: {exc}")
                    t_interp += POLICY_DT
                    if t_interp >= T_interp:
                        target_q_real = DEFAULT_Q.copy()
                        self.state = ControlState.STANDBY
                        print(
                            "[SUCCESS] Default pose reached. Press A to start inference."
                        )

                elif self.state is ControlState.STANDBY:
                    if btn_a_pressed:
                        print("\n[SUCCESS] A pressed: starting inference")
                        self._reset_policy_state()
                        target_q_real = DEFAULT_Q.copy()
                        fall_tilt_counter = 0
                        self.state = ControlState.INFERENCE
                    elif btn_x_pressed:
                        print("[INFO] X pressed: re-interpolating to default pose")
                        q_interp_start = self._current_positions()
                        t_interp = 0.0
                        self.state = ControlState.GO_TO_DEFAULT
                    else:
                        try:
                            self._set_positions(DEFAULT_Q)
                        except Exception as exc:
                            print(f"[WARN] hold default failed: {exc}")

                elif self.state is ControlState.INFERENCE:
                    if btn_x_pressed:
                        print("[INFO] X pressed: returning to default pose")
                        q_interp_start = self._current_positions()
                        t_interp = 0.0
                        self.state = ControlState.GO_TO_DEFAULT
                        continue

                    if current_state is None:
                        # Could not read sensors this tick; skip the policy
                        # step and keep the previous target.
                        self._set_damping_mode() if False else None
                    else:
                        try:
                            target_q_real = self._policy_target(current_state)
                            self._set_positions(target_q_real)
                        except (FloatingPointError, RuntimeError, ValueError) as exc:
                            print(f"\n[CRITICAL] Inference failed: {exc}")
                            self.state = ControlState.DAMPING
                            self.command.fill(0.0)
                            self._set_damping_mode()
                            continue

                elif self.state is ControlState.DAMPING:
                    if btn_x_pressed:
                        tilt = self._torso_tilt_deg()
                        if (
                            math.isfinite(tilt)
                            and tilt <= SafetyCfg.recovery_max_tilt_deg
                        ):
                            print(
                                f"[SAFETY] X accepted at {tilt:.1f} deg: "
                                "interpolating to default pose"
                            )
                            q_interp_start = self._current_positions()
                            t_interp = 0.0
                            self.state = ControlState.GO_TO_DEFAULT
                        else:
                            print(
                                f"[SAFETY] X ignored: tilt="
                                f"{tilt:.1f} deg; must be <= "
                                f"{SafetyCfg.recovery_max_tilt_deg:.1f} deg"
                            )

                    if self.state is ControlState.DAMPING:
                        try:
                            self._set_damping_mode()
                        except Exception as exc:
                            print(f"[WARN] damping command failed: {exc}")

                timer.waiting(step_start)

    def shutdown(self) -> None:
        if self.state is ControlState.IDLE:
            print(
                "[INFO] Exited from read-only IDLE; motor commands were untouched."
            )
        else:
            try:
                self._set_damping_mode()
                print("[SAFETY] Final damping command sent")
            except Exception as exc:
                print(f"[WARNING] Failed to send final damping command: {exc}")
        if self.gamepad is not None and hasattr(self.gamepad, "close"):
            self.gamepad.close()
        try:
            self.upper.disconnect()
        except Exception:
            pass
        try:
            self.lower.disconnect()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E1 Dog 16DOF phase-free AMP ONNX sim2real"
    )
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--host", default="192.168.254.100")
    parser.add_argument("--upper-port", type=int, default=50054)
    parser.add_argument("--lower-port", type=int, default=50053)
    parser.add_argument("--init-duration", type=float, default=1.5)
    parser.add_argument(
        "--command", type=float, nargs=3, metavar=("VX", "VY", "WZ"),
        default=(0.0, 0.0, 0.0),
    )
    parser.add_argument(
        "--gamepad", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--keyboard", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--command-slew", action=argparse.BooleanOptionalAction, default=True,
        help="Ramp commands for real-robot safety; disable to match sim2sim exactly.",
    )
    parser.add_argument("--gamepad-deadzone", type=float, default=0.15)
    parser.add_argument(
        "--validate-policy-only", action="store_true",
        help="Validate ONNX dimensions/inference without importing DroidUp or connecting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_policy_only:
        PhaseFreeAmpOnnxPolicy(args.policy, allow_reference=True)
        print("[INFO] ONNX policy validation passed; no robot connection was opened")
        return

    controller = E1Dog16DOFAmpController(args)
    try:
        controller.run()
    finally:
        controller.shutdown()


if __name__ == "__main__":
    main()