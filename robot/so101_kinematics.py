"""Windows-friendly FK/IK for the SO-101 arm.

The public interface uses degrees for joints/orientation and metres for XYZ.
Only NumPy and Python's standard library are required.  Geometry, joint axes,
fixed offsets, and limits are read from the URDF instead of being duplicated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import numpy as np


ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
ACTION_NAMES = ("x", "y", "z", "wrist_yaw", "wrist_roll", "gripper")


class IKError(ValueError):
    """Raised when a Cartesian target cannot be reached within joint limits."""


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    joint_type: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


@dataclass(frozen=True)
class IKResult:
    joints: np.ndarray
    position_error_m: float
    yaw_error_deg: float
    iterations: int


def _numbers(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    return np.asarray(default if text is None else [float(x) for x in text.split()], dtype=float)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _origin_matrix(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = _rpy_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s, one_c = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    rotation = np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ]
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    return transform


def _wrap_radians(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class SO101Kinematics:
    """FK and bounded numerical IK for the SO-101's 6D action interface.

    Joint vectors are ``[pan, shoulder, elbow, wrist_flex, wrist_roll,
    gripper]`` in degrees. Actions are ``[x, y, z, wrist_yaw,
    wrist_roll, gripper]`` where XYZ is in metres and angles are degrees.
    As requested, ``wrist_yaw = shoulder + elbow + wrist_flex``. The roll
    and gripper values pass through unchanged.
    """

    def __init__(self, urdf_path: str | Path | None = None) -> None:
        self.urdf_path = Path(urdf_path or Path(__file__).with_name("so101.urdf")).resolve()
        self._joints = self._read_urdf(self.urdf_path)
        self._chain = self._make_chain("base_link", "gripper_frame_link")
        movable = [joint.name for joint in self._chain if joint.joint_type != "fixed"]
        if movable != list(ARM_JOINTS):
            raise ValueError(f"Unexpected SO-101 chain in {self.urdf_path}: {movable}")
        self.lower_rad = np.asarray([self._joints[name].lower for name in ARM_JOINTS])
        self.upper_rad = np.asarray([self._joints[name].upper for name in ARM_JOINTS])

    @staticmethod
    def _read_urdf(path: Path) -> dict[str, Joint]:
        root = ET.parse(path).getroot()
        result: dict[str, Joint] = {}
        for element in root.findall("joint"):
            name = element.attrib["name"]
            joint_type = element.attrib["type"]
            origin_tag = element.find("origin")
            xyz = _numbers(origin_tag.get("xyz") if origin_tag is not None else None, (0.0, 0.0, 0.0))
            rpy = _numbers(origin_tag.get("rpy") if origin_tag is not None else None, (0.0, 0.0, 0.0))
            axis_tag = element.find("axis")
            axis = _numbers(axis_tag.get("xyz") if axis_tag is not None else None, (1.0, 0.0, 0.0))
            limit = element.find("limit")
            lower = float(limit.get("lower", "-inf")) if limit is not None else 0.0
            upper = float(limit.get("upper", "inf")) if limit is not None else 0.0
            result[name] = Joint(
                name=name,
                parent=element.find("parent").attrib["link"],
                child=element.find("child").attrib["link"],
                joint_type=joint_type,
                origin=_origin_matrix(xyz, rpy),
                axis=axis,
                lower=lower,
                upper=upper,
            )
        return result

    def _make_chain(self, base: str, tip: str) -> list[Joint]:
        by_child = {joint.child: joint for joint in self._joints.values()}
        reverse_chain: list[Joint] = []
        link = tip
        while link != base:
            if link not in by_child:
                raise ValueError(f"No URDF path from {base!r} to {tip!r}; stopped at {link!r}")
            joint = by_child[link]
            reverse_chain.append(joint)
            link = joint.parent
        return list(reversed(reverse_chain))

    @property
    def joint_limits_deg(self) -> np.ndarray:
        """Return a copy of the five arm-joint limits as shape ``(5, 2)``."""
        return np.rad2deg(np.column_stack((self.lower_rad, self.upper_rad)))

    def forward_matrix(self, arm_joints_deg: np.ndarray | list[float]) -> np.ndarray:
        """Return the URDF base-to-gripper-frame homogeneous transform."""
        values = np.asarray(arm_joints_deg, dtype=float).reshape(-1)
        if values.size < 5:
            raise ValueError("FK requires at least five arm joint values")
        radians = dict(zip(ARM_JOINTS, np.deg2rad(values[:5]), strict=True))
        transform = np.eye(4)
        for joint in self._chain:
            transform = transform @ joint.origin
            if joint.joint_type != "fixed":
                transform = transform @ _axis_rotation(joint.axis, radians[joint.name])
        return transform

    def forward(self, joints_deg: np.ndarray | list[float]) -> np.ndarray:
        """Convert five joints plus gripper to the requested 6D action."""
        joints = np.asarray(joints_deg, dtype=float).reshape(-1)
        if joints.size != 6 or not np.all(np.isfinite(joints)):
            raise ValueError("Expected six finite values: five joints plus gripper")
        position = self.forward_matrix(joints[:5])[:3, 3]
        wrist_yaw = joints[1] + joints[2] + joints[3]
        return np.array([*position, wrist_yaw, joints[4], joints[5]], dtype=float)

    # Convenient aliases for integrations that prefer explicit names.
    joints_to_action = forward

    def link_positions(self, joints_deg: np.ndarray | list[float]) -> dict[str, np.ndarray]:
        """Return link positions for lightweight GUI/debug visualization."""
        values = np.asarray(joints_deg, dtype=float).reshape(-1)
        if values.size < 5:
            raise ValueError("link_positions requires five arm joint values")
        radians = dict(zip(ARM_JOINTS, np.deg2rad(values[:5]), strict=True))
        transform = np.eye(4)
        positions = {"base_link": transform[:3, 3].copy()}
        for joint in self._chain:
            transform = transform @ joint.origin
            if joint.joint_type != "fixed":
                transform = transform @ _axis_rotation(joint.axis, radians[joint.name])
            positions[joint.child] = transform[:3, 3].copy()
        return positions

    def inverse(
        self,
        action: np.ndarray | list[float],
        current_joints_deg: np.ndarray | list[float] | None = None,
        *,
        position_tolerance_m: float = 5e-5,
        yaw_tolerance_deg: float = 0.02,
        max_iterations: int = 180,
        stop_at_first_valid: bool = False,
    ) -> IKResult:
        """Convert a 6D action to joint values using bounded damped least squares.

        Multiple deterministic seeds cover both elbow branches. ``current_joints_deg``
        is used to choose the closest valid solution when supplied. Real-time callers
        may set ``stop_at_first_valid=True`` to return as soon as the current-state
        seed (or the next valid fallback seed) converges.
        """
        target = np.asarray(action, dtype=float).reshape(-1)
        if target.size != 6 or not np.all(np.isfinite(target)):
            raise ValueError("Expected [x, y, z, wrist_yaw, wrist_roll, gripper]")
        if not self.lower_rad[4] - 1e-12 <= math.radians(target[4]) <= self.upper_rad[4] + 1e-12:
            raise IKError(f"wrist_roll {target[4]:.3f} deg is outside the URDF limit")

        target_yaw = math.radians(target[3])
        target_roll = math.radians(target[4])
        seed_reference = None
        if current_joints_deg is not None:
            current = np.asarray(current_joints_deg, dtype=float).reshape(-1)
            if current.size < 4 or not np.all(np.isfinite(current[:4])):
                raise ValueError("current_joints_deg must contain at least four finite values")
            seed_reference = np.clip(np.deg2rad(current[:4]), self.lower_rad[:4], self.upper_rad[:4])

        pan_hint = math.atan2(target[1], target[0])
        midpoint = (self.lower_rad[:4] + self.upper_rad[:4]) / 2.0
        seeds: list[np.ndarray] = []
        if seed_reference is not None:
            seeds.append(seed_reference)
        for pan in (pan_hint, pan_hint + math.pi, 0.0):
            for shoulder, elbow in ((0.0, 0.0), (0.7, -1.0), (-0.7, 1.0), (1.0, 1.0), (-1.0, -1.0)):
                candidate = midpoint.copy()
                candidate[0] = pan
                candidate[1] = shoulder
                candidate[2] = elbow
                candidate[3] = target_yaw - shoulder - elbow
                seeds.append(np.clip(candidate, self.lower_rad[:4], self.upper_rad[:4]))

        solutions: list[tuple[float, float, int, np.ndarray]] = []
        best_failed: tuple[float, float, int, np.ndarray] | None = None
        for seed in seeds:
            q, pos_error, yaw_error, iterations = self._solve_seed(
                seed, target[:3], target_yaw, target_roll, max_iterations
            )
            candidate = (pos_error, abs(yaw_error), iterations, q)
            if best_failed is None or candidate[:2] < best_failed[:2]:
                best_failed = candidate
            if pos_error <= position_tolerance_m and abs(math.degrees(yaw_error)) <= yaw_tolerance_deg:
                distance = float(np.linalg.norm(q - seed_reference)) if seed_reference is not None else 0.0
                solutions.append((distance, pos_error, iterations, q))
                if stop_at_first_valid:
                    break

        if not solutions:
            assert best_failed is not None
            pos_error, yaw_error, _, _ = best_failed
            raise IKError(
                "Target is unreachable within limits/tolerance: "
                f"best position error={pos_error * 1000:.2f} mm, "
                f"yaw error={math.degrees(yaw_error):.3f} deg"
            )

        _, pos_error, iterations, q = min(solutions, key=lambda item: (item[0], item[1]))
        joints = np.array([*np.rad2deg(q), target[4], target[5]], dtype=float)
        reached = self.forward(joints)
        return IKResult(
            joints=joints,
            position_error_m=float(np.linalg.norm(reached[:3] - target[:3])),
            yaw_error_deg=math.degrees(_wrap_radians(math.radians(reached[3] - target[3]))),
            iterations=iterations,
        )

    action_to_joints = inverse

    def _solve_seed(
        self,
        seed: np.ndarray,
        target_position: np.ndarray,
        target_yaw: float,
        target_roll: float,
        max_iterations: int,
    ) -> tuple[np.ndarray, float, float, int]:
        q = seed.copy()
        yaw_scale = 0.12  # metres/radian: balances angle and XYZ residuals.
        damping = 2e-3
        best_q = q.copy()
        best_score = math.inf
        last_pos_error = math.inf
        last_yaw_error = math.inf

        def residual(value: np.ndarray) -> tuple[np.ndarray, float, float]:
            arm_deg = np.rad2deg(np.r_[value, target_roll])
            position = self.forward_matrix(arm_deg)[:3, 3]
            pos_delta = target_position - position
            yaw_delta = _wrap_radians(target_yaw - float(value[1] + value[2] + value[3]))
            return np.r_[pos_delta, yaw_scale * yaw_delta], float(np.linalg.norm(pos_delta)), yaw_delta

        for iteration in range(1, max_iterations + 1):
            error, pos_error, yaw_error = residual(q)
            score = float(error @ error)
            if score < best_score:
                best_score, best_q = score, q.copy()
                last_pos_error, last_yaw_error = pos_error, yaw_error
            if pos_error <= 2e-5 and abs(yaw_error) <= math.radians(0.005):
                return q, pos_error, yaw_error, iteration

            jacobian = np.zeros((4, 4))
            step = 1e-5
            base_position = self.forward_matrix(np.rad2deg(np.r_[q, target_roll]))[:3, 3]
            for column in range(4):
                perturbed = q.copy()
                perturbed[column] += step
                moved = self.forward_matrix(np.rad2deg(np.r_[perturbed, target_roll]))[:3, 3]
                jacobian[:3, column] = (moved - base_position) / step
            jacobian[3] = yaw_scale * np.array([0.0, 1.0, 1.0, 1.0])

            lhs = jacobian.T @ jacobian + damping * damping * np.eye(4)
            delta = np.linalg.solve(lhs, jacobian.T @ error)
            delta = np.clip(delta, -0.22, 0.22)
            trial = np.clip(q + delta, self.lower_rad[:4], self.upper_rad[:4])
            trial_score = float(residual(trial)[0] @ residual(trial)[0])
            if trial_score < score:
                q = trial
                damping = max(2e-5, damping * 0.7)
            else:
                damping = min(0.5, damping * 4.0)

        return best_q, last_pos_error, last_yaw_error, max_iterations


if __name__ == "__main__":
    model = SO101Kinematics()
    sample = np.array([0.0, 20.0, -40.0, 20.0, 0.0, 30.0])
    cartesian = model.forward(sample)
    solved = model.inverse(cartesian, sample)
    print("joints:", np.round(sample, 4))
    print("action:", np.round(cartesian, 6))
    print("IK:    ", np.round(solved.joints, 4))
