"""Offline test of smooth joint/gripper interpolation; never opens a serial port."""

from __future__ import annotations

import numpy as np

from common import SCRIPTS_DIR  # noqa: F401
from aruco_pick_place.motion import interpolate_joint_actions


def main() -> int:
    start = np.asarray([0, -20, 35, -15, 0], dtype=float)
    target = np.asarray([15, -10, 20, -5, 30], dtype=float)
    actions = interpolate_joint_actions(start, target, 90.0, 25.0, steps=30)
    assert len(actions) == 30
    keys = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos"]
    final = np.asarray([actions[-1][key] for key in keys])
    assert np.allclose(final, target)
    assert abs(actions[-1]["gripper.pos"] - 25.0) < 1e-9
    values = np.asarray([[action[key] for key in keys] for action in actions])
    assert np.max(np.abs(np.diff(values, axis=0))) < np.max(np.abs(target - start))
    print("PASS: generated 30 smooth actions and reached the exact target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
