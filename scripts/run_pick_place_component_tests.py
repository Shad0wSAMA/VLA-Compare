"""Run all safe/offline ArUco-YOLO pick-place component tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent / "pick_place_tests"
TESTS = (
    "test_config.py",
    "test_aruco_transform.py",
    "test_geometry.py",
    "test_motion.py",
    "test_kinematics.py",
    "test_cameras.py",
    "test_yolo.py",
    "test_recording.py",
    "test_pipeline.py",
)


def main() -> int:
    failures: list[str] = []
    for filename in TESTS:
        print(f"\n===== {filename} =====", flush=True)
        result = subprocess.run([sys.executable, str(TEST_DIR / filename)], check=False)
        if result.returncode:
            failures.append(f"{filename} (exit {result.returncode})")
    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nPASS: all safe/offline component tests completed. No motors were commanded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
