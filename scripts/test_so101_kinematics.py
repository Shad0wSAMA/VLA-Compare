"""Offline regression tests for robot/so101_kinematics.py."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from robot.so101_kinematics import IKError, SO101Kinematics  # noqa: E402
from scripts.so101_fk_ik_gui import (  # noqa: E402
    GRIPPER_CLOSE_DEG,
    PICK_GRIPPER_OPEN_DEG,
    PICK_WRIST_ROLL_DEG,
    PICK_WRIST_YAW_DEG,
)


class SO101KinematicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = SO101Kinematics(WORKSPACE / "robot" / "so101.urdf")

    def test_known_fk_pose(self) -> None:
        action = self.model.forward([0.0, 20.0, -40.0, 20.0, 0.0, 30.0])
        np.testing.assert_allclose(action, [0.418260, -0.000009, 0.255929, 0.0, 0.0, 30.0], atol=2e-6)

    def test_wrist_yaw_roll_and_gripper_contract(self) -> None:
        joints = [10.0, 25.0, -30.0, 15.0, 42.0, 73.0]
        action = self.model.forward(joints)
        self.assertAlmostEqual(action[3], 10.0)
        self.assertAlmostEqual(action[4], 42.0)
        self.assertAlmostEqual(action[5], 73.0)

    def test_random_fk_ik_fk_round_trip(self) -> None:
        rng = np.random.default_rng(20260816)
        limits = self.model.joint_limits_deg
        for _ in range(25):
            arm = rng.uniform(limits[:, 0] * 0.8, limits[:, 1] * 0.8)
            joints = np.r_[arm, rng.uniform(0.0, 100.0)]
            target = self.model.forward(joints)
            solved = self.model.inverse(target, joints)
            reached = self.model.forward(solved.joints)
            np.testing.assert_allclose(reached, target, atol=5e-7)

    def test_rejects_out_of_limit_roll(self) -> None:
        target = self.model.forward([0.0, 0.0, 0.0, 0.0, 0.0, 50.0])
        target[4] = 180.0
        with self.assertRaises(IKError):
            self.model.inverse(target)

    def test_realtime_first_valid_ik(self) -> None:
        joints = np.array([5.0, 15.0, -25.0, 35.0, -90.0, 20.0])
        target = self.model.forward(joints)
        solved = self.model.inverse(target, joints, stop_at_first_valid=True)
        np.testing.assert_allclose(self.model.forward(solved.joints), target, atol=5e-7)

    def test_default_cube_pick_path_has_ik(self) -> None:
        current = np.array([0.0, 20.0, -40.0, 20.0, 0.0, 0.0])
        for z, gripper in (
            (0.05, PICK_GRIPPER_OPEN_DEG),
            (-0.01, PICK_GRIPPER_OPEN_DEG),
            (-0.01, GRIPPER_CLOSE_DEG),
            (0.05, GRIPPER_CLOSE_DEG),
        ):
            action = np.array([0.24, 0.01, z, PICK_WRIST_YAW_DEG, PICK_WRIST_ROLL_DEG, gripper])
            result = self.model.inverse(action, current)
            current = result.joints
            self.assertLess(result.position_error_m, 5e-5)

    def test_default_cube_place_path_has_ik(self) -> None:
        current = np.array([0.0, 20.0, -40.0, 20.0, -90.0, GRIPPER_CLOSE_DEG])
        for z, gripper in (
            (0.05, GRIPPER_CLOSE_DEG),
            (0.00, GRIPPER_CLOSE_DEG),
            (0.00, PICK_GRIPPER_OPEN_DEG),
            (0.05, PICK_GRIPPER_OPEN_DEG),
        ):
            action = np.array([0.27, 0.01, z, PICK_WRIST_YAW_DEG, PICK_WRIST_ROLL_DEG, gripper])
            result = self.model.inverse(action, current)
            current = result.joints
            self.assertLess(result.position_error_m, 5e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
