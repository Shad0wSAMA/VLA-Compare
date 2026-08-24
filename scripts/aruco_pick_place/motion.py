"""IK validation and time-interpolated SO-101 motion control."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable

import numpy as np

from .hardware import MOTOR_NAMES


LOGGER = logging.getLogger(__name__)
RecordStep = Callable[[dict[str, Any], dict[str, float]], None]


def current_joint_vector(observation: dict[str, Any]) -> np.ndarray:
    return np.asarray([observation[f"{name}.pos"] for name in MOTOR_NAMES], dtype=float)


def full_action(joints_deg: np.ndarray, gripper: float) -> dict[str, float]:
    action = {f"{name}.pos": float(value) for name, value in zip(MOTOR_NAMES, joints_deg, strict=True)}
    action["gripper.pos"] = float(gripper)
    return action


def interpolate_joint_actions(
    start_joints: np.ndarray,
    target_joints: np.ndarray,
    start_gripper: float,
    target_gripper: float,
    steps: int,
) -> list[dict[str, float]]:
    """Pure function used by both the controller and its offline test."""
    actions: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        alpha = step / steps
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        joints = start_joints + smooth * (target_joints - start_joints)
        gripper = start_gripper + smooth * (target_gripper - start_gripper)
        actions.append(full_action(joints, gripper))
    return actions


class MotionController:
    def __init__(
        self,
        robot: Any,
        solver: Any,
        config: dict[str, Any],
        record_step: RecordStep | None = None,
    ):
        self.robot = robot
        self.solver = solver
        self.config = config
        self.record_step = record_step or (lambda _observation, _action: None)
        self.fps = int(config["recording"].get("fps", 30))
        self.last_observation: dict[str, Any] | None = None

    def send_interpolated_action(
        self, target_joints_deg: np.ndarray, target_gripper: float, duration_s: float, label: str
    ) -> None:
        if self.last_observation is None:
            self.last_observation = self.robot.get_observation()
        actions = interpolate_joint_actions(
            current_joint_vector(self.last_observation),
            target_joints_deg,
            float(self.last_observation["gripper.pos"]),
            target_gripper,
            max(2, int(math.ceil(duration_s * self.fps))),
        )
        interval = 1.0 / self.fps
        LOGGER.info("Motion: %s (%d steps)", label, len(actions))
        for action in actions:
            loop_start = time.perf_counter()
            observation = self.robot.get_observation()
            sent_action = self.robot.send_action(action)
            self.record_step(observation, sent_action)
            self.last_observation = observation
            remaining = interval - (time.perf_counter() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
        self.last_observation = self.robot.get_observation()

    def solve_pose(self, pose: np.ndarray, label: str) -> np.ndarray:
        if self.last_observation is None:
            self.last_observation = self.robot.get_observation()
        current = current_joint_vector(self.last_observation)
        cfg = self.config["kinematics"]
        target = np.asarray(
            self.solver.inverse_kinematics(
                current,
                pose,
                position_weight=float(cfg.get("position_weight", 1.0)),
                orientation_weight=float(cfg.get("orientation_weight", 0.05)),
            ),
            dtype=float,
        )
        if target.shape != (5,) or not np.isfinite(target).all():
            raise RuntimeError(f"IK produced an invalid solution for {label}: {target}")
        limits = np.asarray(cfg.get("joint_limits_deg"), dtype=float)
        if limits.shape != (5, 2):
            raise ValueError("kinematics.joint_limits_deg must have shape 5x2")
        if np.any(target < limits[:, 0]) or np.any(target > limits[:, 1]):
            raise RuntimeError(f"IK solution for {label} violates configured joint limits: {target.tolist()}")
        reached = self.solver.forward_kinematics(target)
        error_m = float(np.linalg.norm(reached[:3, 3] - pose[:3, 3]))
        tolerance = float(cfg.get("ik_position_tolerance_m", 0.015))
        if error_m > tolerance:
            raise RuntimeError(f"IK position error for {label} is {error_m:.4f}m > {tolerance:.4f}m")
        return target

    def move_to_pose(self, pose: np.ndarray, gripper: float, label: str) -> None:
        self.send_interpolated_action(
            self.solve_pose(pose, label), gripper, float(self.config["motion"]["move_duration_s"]), label
        )
