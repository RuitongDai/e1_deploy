"""Deployment contract from RobotLab-Isaac-Velocity-Flat-E1-Wheel-v0.

Source: robot_lab/logs/rsl_rl/e1_wheel_flat/2026-11-17_13-44-56/params/env.yaml.
Observation clipping precedes scaling; action clipping follows scale + offset.
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
POLICY_PATH = ROOT / "model_rough/policy.onnx"
URDF_PATH = ROOT / "robot/E1_DOG_v2/E1_DOG_wheel_no_head.urdf"
SCENE_PATH = ROOT / "robot/E1_DOG_v2/scene.xml"

LEG_ORDER = ("FR", "FL", "RR", "RL")
LEG_JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in LEG_ORDER
    for joint in ("thigh", "hip_roll", "hip_yaw", "calf")
)
WHEEL_JOINT_NAMES = tuple(f"{leg}_wheel_joint" for leg in LEG_ORDER)
JOINT_NAMES = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES
OBSERVATION_SIZE = 69
ACTION_SIZE = 20

SIM_DT = 0.005
DECIMATION = 4
SUBSTEPS = 2
PHYSICS_DT = SIM_DT / SUBSTEPS
POLICY_DT = SIM_DT * DECIMATION
BASE_HEIGHT = 0.46
DEFAULT_Q = np.array([0.5, 0.0, 0.0, -1.0] * 4 + [0.0] * 4, dtype=np.float64)
ACTION_SCALE = np.array([0.25, 0.125, 0.125, 0.25] * 4 + [5.0] * 4)
KP = np.array([
    [60.0, 60.0, 30.0, 60.0],  # 前右腿 FR
    [60.0, 60.0, 30.0, 60.0],  # 前左腿 FL
    [60.0, 60.0, 30.0, 60.0],  # 后右腿 RR
    [60.0, 60.0, 30.0, 60.0],  # 后左腿 RL
    [0.0, 0.0, 0.0, 0.0],  # 四个轮子
]).reshape(-1)

KD = np.array([
    [3.0, 3.0, 1.5, 3.0],  # 前右腿 FR
    [3.0, 3.0, 1.5, 3.0],  # 前左腿 FL
    [3.0, 3.0, 1.5, 3.0],  # 后右腿 RR
    [3.0, 3.0, 1.5, 3.0],  # 后左腿 RL
    [0.5, 0.5, 0.5, 0.5],  # 四个轮子
]).reshape(-1)
EFFORT_LIMIT = np.array([60.0, 36.0, 15.0, 60.0] * 4 + [60.0] * 4)
VELOCITY_LIMIT = np.array([13.09, 13.61, 15.71, 13.09] * 4 + [13.09] * 4)
COMMAND_LIMIT = np.ones(3)


def finite_vector(value, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(
            f"Invalid {label}: expected {size} finite values, got {result.shape}"
        )
    return result


def build_observation(
    angular_velocity, gravity, command, q, dq, previous_action
) -> np.ndarray:
    """[ang_vel(3), gravity(3), command(3), q_rel(20), dq(20), action(20)]."""
    angular_velocity = finite_vector(angular_velocity, 3, "body angular velocity")
    gravity = finite_vector(gravity, 3, "projected gravity")
    command = finite_vector(command, 3, "command")
    q = finite_vector(q, ACTION_SIZE, "joint positions")
    dq = finite_vector(dq, ACTION_SIZE, "joint velocities")
    previous_action = finite_vector(previous_action, ACTION_SIZE, "previous raw action")
    relative_q = q - DEFAULT_Q
    relative_q[16:] = (
        0.0  # Keep all four dimensions, but remove unbounded wheel angles.
    )
    terms = (
        np.clip(angular_velocity, -100.0, 100.0) * 0.25,
        np.clip(gravity, -100.0, 100.0),
        np.clip(command, -100.0, 100.0),
        np.clip(relative_q, -100.0, 100.0),
        np.clip(dq, -100.0, 100.0) * 0.05,
        np.clip(previous_action, -100.0, 100.0),
    )
    return np.concatenate(terms).astype(np.float32)


def action_targets(action) -> np.ndarray:
    """16 position targets (rad), followed by 4 velocity targets (rad/s)."""
    action = finite_vector(action, ACTION_SIZE, "policy action")
    offset = np.concatenate((DEFAULT_Q[:16], np.zeros(4)))
    return np.clip(action * ACTION_SCALE + offset, -100.0, 100.0)
