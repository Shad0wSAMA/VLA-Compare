"""Test the recording adapter; add --write-synthetic to create a tiny local episode."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from common import DEFAULT_CONFIG
from aruco_pick_place import load_config
from aruco_pick_place.recording import EpisodeRecorder


class FakeRobot:
    name = "so101_recording_component_test"
    action_features = {
        "shoulder_pan.pos": float, "shoulder_lift.pos": float, "elbow_flex.pos": float,
        "wrist_flex.pos": float, "wrist_roll.pos": float, "gripper.pos": float,
    }
    observation_features = {**action_features, "wrist": (16, 16, 3), "fixed": (16, 16, 3)}


def values(frame: int) -> tuple[dict, dict]:
    joints = {name: float(frame) for name in FakeRobot.action_features}
    image = np.full((16, 16, 3), frame * 20, dtype=np.uint8)
    return {**joints, "wrist": image, "fixed": image}, joints


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-synthetic", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/recording_component_test"))
    args = parser.parse_args()
    config, config_dir = load_config(DEFAULT_CONFIG)
    config = copy.deepcopy(config)
    config["recording"]["enabled"] = args.write_synthetic
    config["recording"]["video"] = False
    config["recording"]["repo_id"] = "local/so101_recording_component_test"
    config["recording"]["root"] = str(args.output.resolve())
    recorder = EpisodeRecorder(FakeRobot(), config, config_dir)
    recorder.start()
    if not args.write_synthetic:
        assert recorder.dataset is None
        observation, action = values(0)
        recorder.add_step(observation, action)
        recorder.save_episode()
        print("PASS: disabled recorder is a safe no-op (use --write-synthetic for dataset I/O).")
        return 0
    for frame in range(3):
        observation, action = values(frame)
        recorder.add_step(observation, action)
    recorder.save_episode()
    assert recorder.dataset is not None
    print(f"PASS: synthetic 3-frame episode saved under {recorder.dataset.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
