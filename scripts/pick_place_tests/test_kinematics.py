"""Offline FK/IK round-trip test using robot/so101.urdf and the configured backend."""

from __future__ import annotations

import numpy as np

from common import DEFAULT_CONFIG
from aruco_pick_place import load_config
from aruco_pick_place.hardware import create_kinematics


def main() -> int:
    config, config_dir = load_config(DEFAULT_CONFIG)
    solver = create_kinematics(config, config_dir)
    expected = np.asarray([10.0, -20.0, 30.0, -10.0, 5.0])
    target_pose = solver.forward_kinematics(expected)
    initial = expected + np.asarray([3.0, -3.0, 2.0, -2.0, 1.0])
    cfg = config["kinematics"]
    solved = np.asarray(
        solver.inverse_kinematics(
            initial,
            target_pose,
            position_weight=float(cfg["position_weight"]),
            orientation_weight=float(cfg["orientation_weight"]),
        )
    )
    reached = solver.forward_kinematics(solved)
    position_error = float(np.linalg.norm(reached[:3, 3] - target_pose[:3, 3]))
    assert solved.shape == (5,) and np.isfinite(solved).all()
    assert position_error <= float(cfg["ik_position_tolerance_m"])
    print(f"PASS: {cfg['backend']} FK->IK position error={position_error * 1000:.3f} mm")
    print(f"solution_deg={solved.round(3).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
