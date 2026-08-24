from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ikpy")

from lerobot.model.kinematics_ikpy import RobotKinematics


PLANAR_URDF = """<?xml version="1.0"?>
<robot name="planar_test">
  <link name="base_link"/>
  <link name="link_1"/>
  <link name="link_2"/>
  <link name="tcp"/>
  <joint name="joint_a" type="revolute">
    <parent link="base_link"/><child link="link_1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="1" velocity="1"/>
  </joint>
  <joint name="joint_b" type="revolute">
    <parent link="link_1"/><child link="link_2"/>
    <origin xyz="1 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="1" velocity="1"/>
  </joint>
  <joint name="tcp_fixed" type="fixed">
    <parent link="link_2"/><child link="tcp"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
  </joint>
</robot>
"""


@pytest.fixture
def urdf_path(tmp_path):
    path = tmp_path / "planar.urdf"
    path.write_text(PLANAR_URDF, encoding="utf-8")
    return path


def test_forward_kinematics_maps_joint_names_in_caller_order(urdf_path):
    solver = RobotKinematics(
        str(urdf_path), target_frame_name="tcp", joint_names=["joint_b", "joint_a"], orientation_mode=None
    )
    pose = solver.forward_kinematics(np.array([-45.0, 30.0]))

    expected_xy = np.array([np.cos(np.deg2rad(30.0)), np.sin(np.deg2rad(30.0))])
    expected_xy += np.array([np.cos(np.deg2rad(-15.0)), np.sin(np.deg2rad(-15.0))])
    np.testing.assert_allclose(pose[:2, 3], expected_xy, atol=1e-7)


def test_inverse_kinematics_reaches_fk_target_and_preserves_gripper(urdf_path):
    solver = RobotKinematics(
        str(urdf_path),
        target_frame_name="tcp",
        joint_names=["joint_a", "joint_b"],
        orientation_mode=None,
        regularization_parameter=1e-8,
    )
    desired_pose = solver.forward_kinematics(np.array([35.0, -50.0]))
    solution = solver.inverse_kinematics(
        np.array([10.0, -10.0, 62.0]), desired_pose, orientation_weight=0.0
    )

    assert solution[2] == pytest.approx(62.0)
    reached_pose = solver.forward_kinematics(solution[:2])
    np.testing.assert_allclose(reached_pose[:3, 3], desired_pose[:3, 3], atol=1e-5)


def test_rejects_joint_outside_target_chain(urdf_path):
    with pytest.raises(ValueError, match="not on the URDF path"):
        RobotKinematics(str(urdf_path), target_frame_name="tcp", joint_names=["missing_joint"])
