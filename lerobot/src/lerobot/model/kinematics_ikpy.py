# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""URDF forward and inverse kinematics implemented with IKPy.

This module intentionally mirrors :mod:`lerobot.model.kinematics`: public joint
vectors are expressed in degrees and end-effector poses are 4x4 homogeneous
matrices in meters.  IKPy internally uses radians and includes fixed joints in
its vectors, so this adapter performs explicit name-based mapping instead of
depending on chain indices.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.utils.import_utils import require_package

if TYPE_CHECKING:
    from ikpy.chain import Chain
else:
    Chain = Any


class RobotKinematics:
    """Robot kinematics with the same degree-based API as the Placo backend.

    Args:
        urdf_path: Path to the robot URDF.
        target_frame_name: URDF link used as the end-effector/TCP frame.
        joint_names: Ordered joint names corresponding to external joint vectors.
            When omitted, every revolute/continuous joint on the path to the TCP
            is used in URDF order.
        orientation_mode: IKPy orientation constraint: ``None``, ``"X"``,
            ``"Y"``, ``"Z"``, or ``"all"``. For a five-DOF SO101 top-down
            gripper, ``"Z"`` is normally the appropriate constraint.
        max_iterations: Maximum optimizer iterations per IK call.
        regularization_parameter: IKPy regularization toward the supplied current
            joint position. ``None`` disables regularization. The residual uses
            ``sqrt(parameter) * (q - q_current)``.
        symbolic: Whether IKPy should build symbolic transformation matrices.
            The NumPy path is the lighter default for this real-time controller.
    """

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] | None = None,
        *,
        orientation_mode: str | None = "Z",
        max_iterations: int = 200,
        regularization_parameter: float | None = 1e-6,
        symbolic: bool = False,
    ):
        require_package("ikpy", extra="ikpy-kinematics")
        from ikpy.chain import Chain as IKPyChain
        from ikpy.link import OriginLink, URDFLink

        urdf = Path(urdf_path).expanduser().resolve()
        if not urdf.is_file():
            raise FileNotFoundError(f"URDF file does not exist: {urdf}")
        if orientation_mode not in (None, "X", "Y", "Z", "all"):
            raise ValueError("orientation_mode must be one of: None, X, Y, Z, all")
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")
        if regularization_parameter is not None and regularization_parameter < 0:
            raise ValueError("regularization_parameter must be non-negative or None")

        root = ET.parse(urdf).getroot()
        path_joints = self._find_joint_path(root, target_frame_name)
        actuated_path_names = [
            joint.attrib["name"]
            for joint in path_joints
            if joint.attrib.get("type") in ("revolute", "continuous")
        ]
        if joint_names is None:
            joint_names = actuated_path_names
        if not joint_names:
            raise ValueError(f"No active joints were found on the path to '{target_frame_name}'")
        if len(set(joint_names)) != len(joint_names):
            raise ValueError(f"joint_names contains duplicates: {joint_names}")

        path_names = [joint.attrib["name"] for joint in path_joints]
        missing = [name for name in joint_names if name not in path_names]
        if missing:
            raise ValueError(
                f"Joints {missing} are not on the URDF path to '{target_frame_name}'. "
                f"Path joints: {path_names}"
            )

        links = [OriginLink()]
        active_mask = [False]
        joint_indices: dict[str, int] = {}
        for joint in path_joints:
            name = joint.attrib["name"]
            joint_type = joint.attrib.get("type", "fixed")
            if name in joint_names and joint_type not in ("revolute", "continuous"):
                raise ValueError(
                    f"Joint '{name}' has type '{joint_type}', but this degree-based adapter "
                    "only exposes revolute/continuous joints"
                )

            origin = joint.find("origin")
            origin_translation = self._parse_vector(origin, "xyz", [0.0, 0.0, 0.0])
            origin_orientation = self._parse_vector(origin, "rpy", [0.0, 0.0, 0.0])
            rotation = translation = None
            ikpy_joint_type = joint_type
            if joint_type in ("revolute", "continuous"):
                ikpy_joint_type = "revolute"
                axis = joint.find("axis")
                rotation = self._parse_vector(axis, "xyz", [1.0, 0.0, 0.0])
            elif joint_type == "prismatic":
                axis = joint.find("axis")
                translation = self._parse_vector(axis, "xyz", [1.0, 0.0, 0.0])
            elif joint_type != "fixed":
                raise ValueError(f"Unsupported URDF joint type '{joint_type}' for joint '{name}'")

            bounds: tuple[float, float] | None = None
            if joint_type != "continuous":
                limit = joint.find("limit")
                if limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
                    bounds = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))

            links.append(
                URDFLink(
                    name=name,
                    origin_translation=origin_translation,
                    origin_orientation=origin_orientation,
                    rotation=rotation,
                    translation=translation,
                    bounds=bounds,
                    use_symbolic_matrix=symbolic,
                    joint_type=ikpy_joint_type,
                )
            )
            chain_index = len(links) - 1
            is_active = name in joint_names
            active_mask.append(is_active)
            if is_active:
                joint_indices[name] = chain_index

        # IKPy expects the final chain link to be inactive. A zero-transform tip
        # preserves the requested target frame even when the URDF path ends in an
        # actuated joint rather than a fixed TCP joint.
        links.append(
            URDFLink(
                name=f"{target_frame_name}__ikpy_tip",
                origin_translation=np.zeros(3),
                origin_orientation=np.zeros(3),
                rotation=None,
                translation=None,
                use_symbolic_matrix=symbolic,
                joint_type="fixed",
            )
        )
        active_mask.append(False)

        self.chain: Chain = IKPyChain(
            links=links,
            active_links_mask=active_mask,
            name=f"{root.attrib.get('name', 'robot')}:{target_frame_name}",
        )
        self.urdf_path = str(urdf)
        self.target_frame_name = target_frame_name
        self.joint_names = list(joint_names)
        self.orientation_mode = orientation_mode
        self.max_iterations = max_iterations
        self.regularization_parameter = regularization_parameter
        self._joint_indices = joint_indices
        self._empty_chain_position = np.zeros(len(self.chain.links), dtype=np.float64)
        self.last_ik_result: Any | None = None

    @staticmethod
    def _parse_vector(element: ET.Element | None, attribute: str, default: list[float]) -> np.ndarray:
        if element is None or attribute not in element.attrib:
            return np.asarray(default, dtype=np.float64)
        values = np.asarray([float(value) for value in element.attrib[attribute].split()], dtype=np.float64)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError(f"URDF attribute {attribute!r} must contain three finite values")
        return values

    @staticmethod
    def _find_joint_path(root: ET.Element, target_frame_name: str) -> list[ET.Element]:
        link_names = {link.attrib["name"] for link in root.findall("link")}
        if target_frame_name not in link_names:
            raise ValueError(f"Target frame/link '{target_frame_name}' does not exist in the URDF")

        children: dict[str, list[tuple[ET.Element, str]]] = {}
        child_links: set[str] = set()
        for joint in root.findall("joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                raise ValueError(f"URDF joint '{joint.attrib.get('name')}' is missing parent or child")
            parent_name = parent.attrib["link"]
            child_name = child.attrib["link"]
            children.setdefault(parent_name, []).append((joint, child_name))
            child_links.add(child_name)

        roots = sorted(link_names - child_links)
        if len(roots) != 1:
            raise ValueError(f"Expected one URDF root link, found {roots}")
        base_link = roots[0]
        queue: deque[tuple[str, list[ET.Element]]] = deque([(base_link, [])])
        visited: set[str] = set()
        while queue:
            link_name, path = queue.popleft()
            if link_name == target_frame_name:
                return path
            if link_name in visited:
                continue
            visited.add(link_name)
            for joint, child_name in children.get(link_name, []):
                queue.append((child_name, [*path, joint]))
        raise ValueError(f"Target frame/link '{target_frame_name}' is not connected to root '{base_link}'")

    def _to_chain_position(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        values = np.asarray(joint_pos_deg, dtype=np.float64).reshape(-1)
        if len(values) < len(self.joint_names):
            raise ValueError(
                f"Expected at least {len(self.joint_names)} joint values, received {len(values)}"
            )
        if not np.isfinite(values[: len(self.joint_names)]).all():
            raise ValueError("Joint positions must be finite")
        chain_position = self._empty_chain_position.copy()
        for name, value_deg in zip(self.joint_names, values, strict=False):
            chain_position[self._joint_indices[name]] = np.deg2rad(value_deg)
        return chain_position

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """Return the 4x4 TCP pose for degree-valued joints."""
        chain_position = self._to_chain_position(joint_pos_deg)
        pose = np.asarray(self.chain.forward_kinematics(chain_position), dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise RuntimeError(f"IKPy returned an invalid forward-kinematics pose: {pose}")
        return pose

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ) -> np.ndarray:
        """Solve IK and return degree-valued joints, preserving trailing values.

        Position and orientation residuals are evaluated through the IKPy chain
        and optimized with SciPy least-squares. A non-positive
        ``orientation_weight`` disables orientation; a positive value enables
        the configured ``orientation_mode`` and scales that residual relative
        to the position residual.
        """
        current = np.asarray(current_joint_pos, dtype=np.float64).reshape(-1)
        if len(current) < len(self.joint_names):
            raise ValueError(
                f"Expected at least {len(self.joint_names)} current joints, received {len(current)}"
            )
        target = np.asarray(desired_ee_pose, dtype=np.float64)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ValueError("desired_ee_pose must be a finite 4x4 transformation matrix")
        if position_weight <= 0:
            raise ValueError("IKPyRobotKinematics requires position_weight > 0")

        initial_position = self._to_chain_position(current)
        orientation_mode = self.orientation_mode if orientation_weight > 0 else None
        active_initial = self.chain.active_from_full(initial_position)
        active_bounds = np.asarray(
            self.chain.active_from_full([link.bounds for link in self.chain.links]), dtype=np.float64
        )
        lower_bounds, upper_bounds = active_bounds[:, 0], active_bounds[:, 1]

        def residual(active_joints: np.ndarray) -> np.ndarray:
            full_joints = self.chain.active_to_full(active_joints, initial_position)
            reached = np.asarray(self.chain.forward_kinematics(full_joints), dtype=np.float64)
            parts = [position_weight * (reached[:3, 3] - target[:3, 3])]
            if orientation_mode == "X":
                parts.append(orientation_weight * (reached[:3, 0] - target[:3, 0]))
            elif orientation_mode == "Y":
                parts.append(orientation_weight * (reached[:3, 1] - target[:3, 1]))
            elif orientation_mode == "Z":
                parts.append(orientation_weight * (reached[:3, 2] - target[:3, 2]))
            elif orientation_mode == "all":
                parts.append(orientation_weight * (reached[:3, :3] - target[:3, :3]).reshape(-1))
            if self.regularization_parameter is not None and self.regularization_parameter > 0:
                parts.append(np.sqrt(self.regularization_parameter) * (active_joints - active_initial))
            return np.concatenate(parts)

        from scipy.optimize import least_squares

        optimization = least_squares(
            residual,
            active_initial,
            bounds=(lower_bounds, upper_bounds),
            max_nfev=self.max_iterations,
            x_scale="jac",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        self.last_ik_result = optimization
        if not optimization.success:
            raise RuntimeError(
                f"IKPy/SciPy inverse kinematics failed (status={optimization.status}): "
                f"{optimization.message}"
            )
        solution = np.asarray(
            self.chain.active_to_full(optimization.x, initial_position), dtype=np.float64
        )
        if solution.shape != initial_position.shape or not np.isfinite(solution).all():
            raise RuntimeError(f"IKPy returned an invalid inverse-kinematics solution: {solution}")

        result = current.copy()
        for output_index, name in enumerate(self.joint_names):
            result[output_index] = np.rad2deg(solution[self._joint_indices[name]])
        return result


IKPyRobotKinematics = RobotKinematics

__all__ = ["IKPyRobotKinematics", "RobotKinematics"]
