"""ArUco table registration and YOLO cube selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    pixel_xy: np.ndarray
    table_xy_m: np.ndarray


def undistort_if_configured(image_rgb: np.ndarray, perception: dict[str, Any]) -> np.ndarray:
    matrix = perception.get("camera_matrix")
    distortion = perception.get("dist_coeffs")
    if matrix is None and distortion is None:
        return image_rgb
    if matrix is None or distortion is None:
        raise ValueError("camera_matrix and dist_coeffs must either both be set or both be null")
    camera_matrix = np.asarray(matrix, dtype=np.float64)
    dist_coeffs = np.asarray(distortion, dtype=np.float64)
    if camera_matrix.shape != (3, 3):
        raise ValueError("perception.camera_matrix must be 3x3")
    return cv2.undistort(image_rgb, camera_matrix, dist_coeffs)


def detect_marker_centers(
    image_rgb: np.ndarray, perception: dict[str, Any]
) -> tuple[dict[int, np.ndarray], list[np.ndarray], np.ndarray | None]:
    dictionary_id = getattr(cv2.aruco, perception["aruco_dictionary"])
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(dictionary_id), cv2.aruco.DetectorParameters()
    )
    corners, ids, _rejected = detector.detectMarkers(image_rgb)
    centers: dict[int, np.ndarray] = {}
    if ids is not None:
        for marker_corners, marker_id in zip(corners, ids.reshape(-1), strict=True):
            centers[int(marker_id)] = marker_corners.reshape(4, 2).mean(axis=0)
    return centers, corners, ids


def detect_marker_boundary_points(
    corners: list[np.ndarray], ids: np.ndarray | None, perception: dict[str, Any]
) -> dict[int, np.ndarray]:
    """Return each marker's configured outward border corner."""
    boundary = perception["marker_boundary"]
    points: dict[int, np.ndarray] = {}
    if ids is None:
        return points
    for marker_corners, marker_id_raw in zip(corners, ids.reshape(-1), strict=True):
        marker_id = int(marker_id_raw)
        entry = boundary.get(str(marker_id))
        if entry is not None:
            points[marker_id] = marker_corners.reshape(4, 2)[int(entry["corner_index"])].astype(np.float32)
    return points


def estimate_table_homography(
    boundary_samples: list[dict[int, np.ndarray]], perception: dict[str, Any]
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    marker_map = {
        int(key): np.asarray(value["xy_m"], dtype=np.float32)
        for key, value in perception["marker_boundary"].items()
    }
    expected_ids = sorted(marker_map)
    complete = [sample for sample in boundary_samples if all(marker_id in sample for marker_id in expected_ids)]
    minimum = int(perception.get("minimum_complete_aruco_samples", 5))
    if len(complete) < minimum:
        raise RuntimeError(
            f"Only {len(complete)} complete ArUco frames were found; need at least {minimum}. "
            f"Expected marker IDs: {expected_ids}"
        )

    averaged: dict[int, np.ndarray] = {}
    max_std = float(perception.get("max_marker_center_std_px", 2.0))
    for marker_id in expected_ids:
        points = np.asarray([sample[marker_id] for sample in complete], dtype=np.float32)
        std_px = float(np.linalg.norm(points.std(axis=0)))
        if std_px > max_std:
            raise RuntimeError(
                f"Marker {marker_id} boundary corner is unstable ({std_px:.2f}px > {max_std:.2f}px)."
            )
        averaged[marker_id] = np.median(points, axis=0).astype(np.float32)

    pixel_points = np.asarray([averaged[marker_id] for marker_id in expected_ids], dtype=np.float32)
    table_points = np.asarray([marker_map[marker_id] for marker_id in expected_ids], dtype=np.float32)
    if abs(float(cv2.contourArea(cv2.convexHull(pixel_points)))) < 100.0:
        raise RuntimeError("Detected marker boundary corners are degenerate or occupy too little image area.")
    homography = cv2.getPerspectiveTransform(pixel_points, table_points)
    if not np.isfinite(homography).all() or abs(float(np.linalg.det(homography))) < 1e-12:
        raise RuntimeError("Could not compute a stable image-to-table homography.")
    return homography, averaged


def pixel_to_table_xy(pixel_xy: np.ndarray, homography: np.ndarray) -> np.ndarray:
    point = np.asarray(pixel_xy, dtype=np.float32).reshape(1, 1, 2)
    return cv2.perspectiveTransform(point, homography).reshape(2).astype(np.float64)


def select_cube(model: Any, image_rgb: np.ndarray, homography: np.ndarray, perception: dict[str, Any]) -> Detection:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    predict_args: dict[str, Any] = {
        "source": image_bgr,
        "conf": float(perception.get("confidence", 0.5)),
        "imgsz": int(perception.get("image_size", 640)),
        "verbose": False,
    }
    if perception.get("device") not in (None, ""):
        predict_args["device"] = perception["device"]
    result = model.predict(**predict_args)[0]
    if result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError("YOLO did not detect a cube above the configured confidence threshold.")

    wanted_class = int(perception.get("class_id", 0))
    candidates: list[Detection] = []
    for xyxy, confidence, class_id in zip(
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.conf.cpu().numpy(),
        result.boxes.cls.cpu().numpy().astype(int),
        strict=True,
    ):
        if int(class_id) != wanted_class:
            continue
        pixel_xy = np.asarray([(xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0])
        table_xy = pixel_to_table_xy(pixel_xy, homography)
        candidates.append(
            Detection(tuple(float(value) for value in xyxy), float(confidence), int(class_id), pixel_xy, table_xy)
        )
    if not candidates:
        raise RuntimeError(f"YOLO returned boxes, but none had class_id={wanted_class}.")

    detection = max(candidates, key=lambda item: item.confidence)
    bounds = np.asarray(perception["table_xy_bounds_m"], dtype=float)
    if bounds.shape != (2, 2):
        raise ValueError("perception.table_xy_bounds_m must be [[xmin,xmax],[ymin,ymax]]")
    if np.any(detection.table_xy_m < bounds[:, 0]) or np.any(detection.table_xy_m > bounds[:, 1]):
        raise RuntimeError(f"Detected cube table position {detection.table_xy_m.tolist()} is outside table bounds.")
    return detection
