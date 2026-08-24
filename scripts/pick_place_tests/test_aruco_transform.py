"""Fully offline synthetic test of ArUco detection and border-based homography."""

from __future__ import annotations

import cv2
import numpy as np

from common import DEFAULT_CONFIG
from aruco_pick_place import (
    detect_marker_boundary_points,
    detect_marker_centers,
    estimate_table_homography,
    load_config,
    pixel_to_table_xy,
)


def synthetic_board(perception: dict) -> np.ndarray:
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    dictionary_id = getattr(cv2.aruco, perception["aruco_dictionary"])
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    placements = {0: (50, 50), 1: (510, 50), 2: (510, 350), 3: (50, 350)}
    marker_size = 80
    for marker_id, (x, y) in placements.items():
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
        image[y : y + marker_size, x : x + marker_size] = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB)
    return image


def main() -> int:
    config, _ = load_config(DEFAULT_CONFIG)
    perception = dict(config["perception"])
    perception["minimum_complete_aruco_samples"] = 3
    image = synthetic_board(perception)
    samples = []
    for _ in range(3):
        _centers, corners, ids = detect_marker_centers(image, perception)
        points = detect_marker_boundary_points(corners, ids, perception)
        assert set(points) == {0, 1, 2, 3}, f"Detected IDs: {sorted(points)}"
        samples.append(points)
    homography, _ = estimate_table_homography(samples, perception)
    # The configured outward border corners form this pixel rectangle.
    center_px = np.asarray([(50 + 589) / 2, (50 + 429) / 2], dtype=float)
    center_xy = pixel_to_table_xy(center_px, homography)
    expected = np.asarray([0.0192 / 2, 0.0271 / 2])
    error = float(np.linalg.norm(center_xy - expected))
    assert error < 1e-5, (center_xy, expected, error)
    print(f"PASS: center {center_px.round(1)} px -> {center_xy.round(6)} m; error={error:.2e} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
