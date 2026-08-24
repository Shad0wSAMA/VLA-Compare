"""Reusable components for the ArUco + YOLO SO-101 pick-and-place pipeline."""

from .config import load_config, resolve_local_path, validate_config
from .geometry import build_pick_place_poses, check_workspace, table_target_to_robot_pose
from .perception import (
    Detection,
    detect_marker_boundary_points,
    detect_marker_centers,
    estimate_table_homography,
    pixel_to_table_xy,
    select_cube,
    undistort_if_configured,
)

__all__ = [
    "Detection",
    "build_pick_place_poses",
    "check_workspace",
    "detect_marker_boundary_points",
    "detect_marker_centers",
    "estimate_table_homography",
    "load_config",
    "pixel_to_table_xy",
    "resolve_local_path",
    "select_cube",
    "table_target_to_robot_pose",
    "undistort_if_configured",
    "validate_config",
]
