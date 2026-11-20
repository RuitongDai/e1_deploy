#!/usr/bin/env python3
"""E1 Wheel hardware deployment with the same policy contract as sim2sim."""

import argparse
import json
import math
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import grpc
import numpy as np
import onnxruntime as ort

# Use the repository's SDK protocol and policy helpers for direct script execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from droidup.api.arm_service_pb2_grpc import ArmServiceStub
from droidup.api.droidup_msg_pb2 import DroidCommandRequest, Empty
from droidup.api.leg_service_pb2_grpc import LegServiceStub
from e1_wheel.gamepad import Gamepad
from e1_wheel.wheel_config import (
    ACTION_SIZE,
    COMMAND_LIMIT,
    DEFAULT_Q,
    EFFORT_LIMIT,
    JOINT_NAMES,
    KD,
    KP,
    LEG_JOINT_NAMES,
    OBSERVATION_SIZE,
    POLICY_DT,
    VELOCITY_LIMIT,
    action_targets,
    build_observation,
    finite_vector,
)

LEG_COUNT = len(LEG_JOINT_NAMES)
CONFIG_PATH = Path(__file__).with_name("config.json")


class MotorService:
    def __init__(self, name, stub_class, prefix, config, stack):
        self.address = config[f"{name}_address"]
        self.timeout = config["rpc_timeout_s"]
        self.finger = config.get("upper_finger") if name == "upper" else []
        channel = stack.enter_context(grpc.insecure_channel(self.address))
        self.stub = stub_class(channel)
        self.state_rpc = f"Get{prefix}State"
        self.command_rpc = f"Set{prefix}Command"
        device = self.call(f"Get{prefix}Config", Empty(), timeout=2.0)

        # Each row in config lists [service, SDK joint name, direction, zero offset (rad)].
        mapping = {}
        for index, joint in enumerate(JOINT_NAMES):
            service, sdk_name, direction, offset = config["joints"][joint]
            if service == name:
                mapping[sdk_name] = (index, direction, offset)
        self.names = list(device.joint_name)
        if len(self.names) != 10 or len(mapping) != 10 or set(self.names) != mapping.keys():
            raise ValueError(f"{name}: joint mapping does not match SDK names {self.names}")
        indices, directions, offsets = zip(*(mapping[joint] for joint in self.names))
        self.indices = np.asarray(indices)
        self.directions = np.asarray(directions, dtype=float)
        self.offsets = finite_vector(offsets, 10, f"{name} offsets")
        if not np.isin(self.directions, (-1, 1)).all():
            raise ValueError(f"{name}: joint directions must be +1 or -1")
        self.wheels = self.indices >= LEG_COUNT
        self.modes = [
            config["wheel_mode"] if wheel else config["leg_mode"] for wheel in self.wheels
        ]
        # SDK reports position limits in degrees and joint state/targets in radians.
        self.pmin = np.deg2rad(finite_vector(device.pmin, 10, f"{name} pmin"))[~self.wheels]
        self.pmax = np.deg2rad(finite_vector(device.pmax, 10, f"{name} pmax"))[~self.wheels]
        if np.any(self.pmin >= self.pmax):
            raise ValueError(f"{name}: invalid SDK leg target limits")
        print(f"[SDK] {name} {self.address}: {', '.join(self.names)}")

    def call(self, method, request, timeout=None):
        rpc = getattr(self.stub, method)
        try:
            return rpc(request, timeout=self.timeout if timeout is None else timeout)
        except grpc.RpcError as error:
            raise ConnectionError(
                f"{self.address} {method}: {error.code().name}: {error.details()}"
            ) from error

    def read(self, q, dq, timeout=None):
        state = self.call(self.state_rpc, Empty(), timeout)
        position = finite_vector(state.position, 10, f"{self.address} position")
        velocity = finite_vector(state.velocity, 10, f"{self.address} velocity")
        q[self.indices] = self.directions * position + self.offsets
        dq[self.indices] = self.directions * velocity
        return state

    def request(self, targets, q, damping):
        targets = targets[self.indices]
        i = self.indices
        position = self.directions * (np.where(self.wheels, q[i], targets) - self.offsets)
        position[~self.wheels] = np.clip(position[~self.wheels], self.pmin, self.pmax)
        velocity = self.directions * np.where(self.wheels, targets, 0.0)
        return DroidCommandRequest(
            mode=self.modes,
            position=position.tolist(),
            velocity=velocity.tolist(),
            torque=[0.0] * 10,
            ens=[1] * 10,
            kp=[0.0] * 10 if damping else KP[i].tolist(),
            kd=KD[i].tolist(),
            max_torque=EFFORT_LIMIT[i].tolist(),
            finger=self.finger,
        )

    def send(self, command):
        self.call(self.command_rpc, command)


def read_state(upper, lower, imu_to_body, timeout=None):
    q, dq = np.empty(ACTION_SIZE), np.empty(ACTION_SIZE)
    upper.read(q, dq, timeout)
    imu = lower.read(q, dq, timeout)
    roll, pitch, _ = finite_vector(imu.imu_euler, 3, "lower IMU Euler")
    gyro = finite_vector(imu.imu_gyro, 3, "lower IMU gyro")
    cos_pitch = math.cos(pitch)
    gravity = np.array(
        [math.sin(pitch), -cos_pitch * math.sin(roll), -cos_pitch * math.cos(roll)]
    )
    return q, dq, imu_to_body @ gyro, imu_to_body @ gravity


def infer(session, input_name, observation):
    (actions,) = session.run(None, {input_name: observation[None, :]})
    return finite_vector(actions[0], ACTION_SIZE, "policy output")


def run(upper, lower, gamepad, session, input_name, config, imu_to_body):
    command_scale = finite_vector(config["command_scale"], 3, "command scale")
    stand_duration = config["stand_duration_s"]
    neutral = DEFAULT_Q.copy()
    previous_action = np.zeros(ACTION_SIZE)
    state = "idle"
    wrote = False
    gamepad.read()  # Require a fresh button press after launch.
    print("[STATE] IDLE. X=stand, A=inference after standby, Y=damping, B=exit")
    try:
        while True:
            tick = time.monotonic()
            stick, events = gamepad.read()
            if "exit" in events:
                break
            q, dq, gyro, gravity = read_state(upper, lower, imu_to_body)

            if "damp" in events:
                if wrote:
                    state = "damping"
                    print("[STATE] DAMPING")
            elif "stand" in events:
                start_q, stand_started = q[:LEG_COUNT].copy(), time.monotonic()
                state = "standing"
                print("[STATE] STANDING")
            elif "start" in events and state == "standby":
                previous_action.fill(0.0)
                state = "inference"
                print("[STATE] INFERENCE")

            targets = neutral.copy()
            if state == "standing":
                progress = min((time.monotonic() - stand_started) / stand_duration, 1.0)
                blend = 0.5 * (1.0 - math.cos(math.pi * progress))
                targets[:LEG_COUNT] = (1.0 - blend) * start_q + blend * neutral[:LEG_COUNT]
                if progress == 1.0:
                    state = "standby"
                    print("[STATE] STANDBY")
            elif state == "inference":
                command = np.clip(stick * command_scale, -COMMAND_LIMIT, COMMAND_LIMIT)
                obs = build_observation(gyro, gravity, command, q, dq, previous_action)
                previous_action = infer(session, input_name, obs)
                targets = action_targets(previous_action)
                wheel_limit = VELOCITY_LIMIT[LEG_COUNT:]
                targets[LEG_COUNT:] = np.clip(targets[LEG_COUNT:], -wheel_limit, wheel_limit)
            elif state == "damping":
                targets[:LEG_COUNT] = q[:LEG_COUNT]

            if state != "idle":
                damping = state == "damping"
                upper_cmd = upper.request(targets, q, damping)
                lower_cmd = lower.request(targets, q, damping)
                last_q = q
                wrote = True
                upper.send(upper_cmd)
                lower.send(lower_cmd)
            time.sleep(max(0.0, POLICY_DT - (time.monotonic() - tick)))
    finally:
        if wrote:
            targets = neutral.copy()
            targets[:LEG_COUNT] = last_q[:LEG_COUNT]
            for service in (upper, lower):
                try:
                    service.send(service.request(targets, last_q, damping=True))
                except ConnectionError as error:
                    print(f"[ERROR] Final damping: {error}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Read one joint/IMU frame and run inference without a gamepad or motor writes",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not args.read_only:
        if config["wheel_mode"] is None:
            raise ValueError("Set wheel_mode from the E1 Wheel firmware protocol before enabling motor writes")
        # DogUpper reads all 12 entries even on robots without hands.
        finger = config.get("upper_finger")
        if (
            not isinstance(finger, list)
            or len(finger) != 12
            or any(type(value) is not int or not -(2**31) <= value < 2**31 for value in finger)
        ):
            raise ValueError("upper_finger must contain exactly 12 signed 32-bit integers")
        if not 0.0 < config["stand_duration_s"] < math.inf:
            raise ValueError("stand_duration_s must be a positive finite number")

    policy_path = args.config.parent / config["policy"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(policy_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    (model_input,) = session.get_inputs()
    infer(session, model_input.name, np.zeros(OBSERVATION_SIZE, dtype=np.float32))
    print(f"[ONNX] {policy_path.resolve()} ({OBSERVATION_SIZE} -> {ACTION_SIZE})")

    imu_to_body = np.asarray(config["imu_to_body"], dtype=float)
    with ExitStack() as stack:
        upper = MotorService("upper", ArmServiceStub, "Arm", config, stack)
        lower = MotorService("lower", LegServiceStub, "Leg", config, stack)
        if args.read_only:
            q, dq, gyro, gravity = read_state(upper, lower, imu_to_body, timeout=1.0)
            obs = build_observation(gyro, gravity, np.zeros(3), q, dq, np.zeros(ACTION_SIZE))
            actions = infer(session, model_input.name, obs)
            print(f"[READ ONLY] gyro={gyro.round(3)}, gravity={gravity.round(3)}")
            print(f"[READ ONLY] q={q.round(3)}, dq={dq.round(3)}")
            print(f"[READ ONLY] action={actions.round(3)}")
        else:
            gamepad = stack.enter_context(Gamepad(config["gamepad_index"]))
            run(upper, lower, gamepad, session, model_input.name, config, imu_to_body)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        sys.exit(f"[ERROR] {error}")
