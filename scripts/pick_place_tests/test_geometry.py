"""Offline test for table-to-robot transforms and waypoint generation."""

from __future__ import annotations

import numpy as np

from common import DEFAULT_CONFIG
from aruco_pick_place import build_pick_place_poses, load_config


def main() -> int:
    config, _ = load_config(DEFAULT_CONFIG)
    cube_xy = np.asarray([0.0096, 0.01355])
    poses = build_pick_place_poses(cube_xy, config)
    assert set(poses) == {"grasp approach", "grasp", "grasp lift", "place approach", "place"}
    for label, pose in poses.items():
        assert pose.shape == (4, 4)
        assert np.isfinite(pose).all()
        assert np.allclose(pose[3], [0, 0, 0, 1])
        print(f"{label:16s}: xyz_robot={pose[:3, 3].round(4).tolist()}")
    assert poses["grasp approach"][2, 3] > poses["grasp"][2, 3]
    print("PASS: five finite, workspace-checked poses generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
