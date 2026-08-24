"""Offline orchestration-order test with fake robot, controller, and recorder."""

from __future__ import annotations

import contextlib

import numpy as np

from common import DEFAULT_CONFIG
from aruco_pick_place import build_pick_place_poses, load_config
from aruco_pick_place.pipeline import execute_pick_place_sequence


class FakeRobot:
    def get_observation(self) -> dict[str, float]:
        return {
            "shoulder_pan.pos": 0.0, "shoulder_lift.pos": 0.0, "elbow_flex.pos": 0.0,
            "wrist_flex.pos": 0.0, "wrist_roll.pos": 0.0, "gripper.pos": 90.0,
        }


class FakeController:
    def __init__(self, robot: FakeRobot):
        self.robot = robot
        self.last_observation = None
        self.events: list[str] = []

    def send_interpolated_action(self, joints, gripper, duration, label) -> None:
        assert np.asarray(joints).shape == (5,)
        self.events.append(label)
        self.last_observation = self.robot.get_observation()

    def move_to_pose(self, pose, gripper, label) -> None:
        assert np.asarray(pose).shape == (4, 4)
        self.events.append(label)
        self.last_observation = self.robot.get_observation()


class FakeRecorder:
    def __init__(self):
        self.saved = False

    def video_manager(self):
        return contextlib.nullcontext()

    def save_episode(self) -> None:
        self.saved = True


def main() -> int:
    config, _ = load_config(DEFAULT_CONFIG)
    robot = FakeRobot()
    controller = FakeController(robot)
    recorder = FakeRecorder()
    poses = build_pick_place_poses(np.asarray([0.0096, 0.01355]), config)
    execute_pick_place_sequence(robot, controller, recorder, poses, config["motion"])
    expected = [
        "open gripper", "grasp approach", "descend to cube", "close gripper",
        "lift cube", "place approach", "descend to place", "release cube", "retreat",
    ]
    assert controller.events == expected
    assert recorder.saved
    print("PASS: orchestration order is correct and the episode is saved after retreat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
