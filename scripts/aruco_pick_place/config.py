"""Configuration loading and validation for the pick-and-place pipeline."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    path = path.expanduser().resolve()
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Config file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("The top-level JSON value must be an object.")
    return config, path.parent


def resolve_local_path(value: str | None, config_dir: Path) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def validate_config(config: dict[str, Any], config_dir: Path) -> list[str]:
    """Return all known blockers instead of failing at the first placeholder."""
    errors: list[str] = []

    for section in ("robot", "perception", "kinematics", "motion", "placement", "recording"):
        if not isinstance(config.get(section), dict):
            errors.append(f"missing object: {section}")
    if errors:
        return errors

    robot = config["robot"]
    perception = config["perception"]
    kinematics = config["kinematics"]
    motion = config["motion"]
    placement = config["placement"]
    recording = config["recording"]

    if importlib.util.find_spec("ultralytics") is None:
        errors.append("Python package 'ultralytics' is not installed")
    backend = str(kinematics.get("backend", "placo")).lower()
    if backend not in ("ikpy", "placo"):
        errors.append("kinematics.backend must be either 'ikpy' or 'placo'")
    elif backend == "ikpy" and importlib.util.find_spec("ikpy") is None:
        errors.append("Python package 'ikpy' is not installed")
    elif backend == "placo" and importlib.util.find_spec("placo") is None:
        errors.append("Python package 'placo' is not installed")

    if not str(robot.get("port", "")).strip():
        errors.append("robot.port is empty")
    cameras = robot.get("cameras")
    if not isinstance(cameras, dict) or not cameras:
        errors.append("robot.cameras must be a non-empty object")
    else:
        camera_sources: list[str] = []
        for camera_name, camera in cameras.items():
            if not str(camera_name).strip() or not isinstance(camera, dict):
                errors.append(f"robot.cameras.{camera_name} must be a camera object")
                continue
            if str(camera.get("type", "opencv")).lower() != "opencv":
                errors.append(f"robot.cameras.{camera_name}.type must be 'opencv'")
            for key in ("index_or_path", "width", "height", "fps"):
                if key not in camera or camera[key] in (None, ""):
                    errors.append(f"robot.cameras.{camera_name}.{key} is missing")
            try:
                if int(camera.get("width", 0)) <= 0 or int(camera.get("height", 0)) <= 0:
                    raise ValueError
                if int(camera.get("fps", 0)) <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(f"robot.cameras.{camera_name} width, height, and fps must be positive integers")
            fourcc = camera.get("fourcc")
            if fourcc is not None and (not isinstance(fourcc, str) or len(fourcc) != 4):
                errors.append(f"robot.cameras.{camera_name}.fourcc must be null or four characters")
            if camera.get("index_or_path") not in (None, ""):
                camera_sources.append(str(camera["index_or_path"]))
        if len(camera_sources) != len(set(camera_sources)):
            errors.append("robot.cameras must use unique index_or_path values")
        perception_camera = str(robot.get("perception_camera", ""))
        if not perception_camera:
            errors.append("robot.perception_camera is missing")
        elif perception_camera not in cameras:
            errors.append(f"robot.perception_camera '{perception_camera}' is not present in robot.cameras")

    weights = resolve_local_path(perception.get("weights"), config_dir)
    if weights is None:
        errors.append("perception.weights is empty (train YOLO first, then point it to best.pt)")
    elif not weights.is_file():
        errors.append(f"perception.weights does not exist: {weights}")

    if not perception.get("marker_layout_confirmed", False):
        errors.append("perception.marker_layout_confirmed is false")
    marker_boundary = perception.get("marker_boundary", {})
    if not isinstance(marker_boundary, dict) or len(marker_boundary) != 4:
        errors.append("perception.marker_boundary must contain exactly four marker IDs")
    else:
        try:
            marker_ids = [int(marker_id) for marker_id in marker_boundary]
            marker_xy = np.asarray(
                [marker_boundary[str(marker_id)]["xy_m"] for marker_id in marker_ids], dtype=float
            )
            corner_indices = [int(marker_boundary[str(marker_id)]["corner_index"]) for marker_id in marker_ids]
            if marker_xy.shape != (4, 2) or not np.isfinite(marker_xy).all():
                raise ValueError
            if any(index not in range(4) for index in corner_indices):
                raise ValueError
            if abs(float(cv2.contourArea(cv2.convexHull(marker_xy.astype(np.float32))))) < 1e-6:
                errors.append("marker_boundary xy_m values are degenerate; use four non-collinear points")
        except (TypeError, ValueError, KeyError):
            errors.append("marker_boundary entries need corner_index 0..3 and finite xy_m [x,y] in meters")

    dictionary_name = perception.get("aruco_dictionary", "")
    if not hasattr(cv2.aruco, str(dictionary_name)):
        errors.append(f"unknown ArUco dictionary: {dictionary_name}")
    try:
        table_bounds = np.asarray(perception["table_xy_bounds_m"], dtype=float)
        if (table_bounds.shape != (2, 2) or not np.isfinite(table_bounds).all()
                or np.any(table_bounds[:, 0] >= table_bounds[:, 1])):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        errors.append("perception.table_xy_bounds_m must be [[xmin,xmax],[ymin,ymax]]")
    camera_matrix = perception.get("camera_matrix")
    dist_coeffs = perception.get("dist_coeffs")
    if (camera_matrix is None) != (dist_coeffs is None):
        errors.append("perception.camera_matrix and dist_coeffs must either both be set or both be null")
    elif camera_matrix is not None:
        try:
            matrix = np.asarray(camera_matrix, dtype=float)
            distortion = np.asarray(dist_coeffs, dtype=float)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
                raise ValueError
        except (TypeError, ValueError):
            errors.append("camera_matrix must be finite 3x3 and dist_coeffs must contain finite values")

    urdf_path = resolve_local_path(kinematics.get("urdf_path"), config_dir)
    if urdf_path is None:
        errors.append("kinematics.urdf_path is empty")
    elif not urdf_path.is_file():
        errors.append(f"kinematics.urdf_path does not exist: {urdf_path}")
    if not kinematics.get("table_to_robot_confirmed", False):
        errors.append("kinematics.table_to_robot_confirmed is false")
    try:
        table_to_robot = np.asarray(kinematics.get("table_to_robot"), dtype=float)
        if table_to_robot.shape != (4, 4) or not np.isfinite(table_to_robot).all():
            raise ValueError
        if not np.allclose(table_to_robot[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            errors.append("kinematics.table_to_robot last row must be [0, 0, 0, 1]")
        rotation = table_to_robot[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
            errors.append("kinematics.table_to_robot rotation is not orthonormal")
    except (TypeError, ValueError):
        errors.append("kinematics.table_to_robot must be a finite 4x4 matrix")

    joint_names = kinematics.get("joint_names")
    if not isinstance(joint_names, list) or len(joint_names) != 5 or len(set(joint_names)) != 5:
        errors.append("kinematics.joint_names must contain the five unique URDF arm joint names")
    try:
        joint_limits = np.asarray(kinematics["joint_limits_deg"], dtype=float)
        if (joint_limits.shape != (5, 2) or not np.isfinite(joint_limits).all()
                or np.any(joint_limits[:, 0] >= joint_limits[:, 1])):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        errors.append("kinematics.joint_limits_deg must contain five measured [min,max] degree pairs")
    try:
        tool_rpy = np.asarray(kinematics["tool_rotation_rpy_deg_in_table"], dtype=float)
        if tool_rpy.shape != (3,) or not np.isfinite(tool_rpy).all():
            raise ValueError
    except (KeyError, TypeError, ValueError):
        errors.append("kinematics.tool_rotation_rpy_deg_in_table must be three finite angles")

    if not motion.get("poses_confirmed", False):
        errors.append("motion.poses_confirmed is false")
    numeric_motion_keys = (
        "grasp_z_m", "approach_height_m", "lift_height_m", "move_duration_s",
        "gripper_duration_s", "gripper_open", "gripper_closed",
    )
    for key in numeric_motion_keys:
        try:
            if not math.isfinite(float(motion[key])):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            errors.append(f"motion.{key} must be a finite number")
    for key in ("grasp_z_m", "approach_height_m", "lift_height_m", "move_duration_s", "gripper_duration_s"):
        try:
            if key in motion and math.isfinite(float(motion[key])) and float(motion[key]) <= 0:
                errors.append(f"motion.{key} must be positive")
        except (TypeError, ValueError):
            pass
    for key in ("gripper_open", "gripper_closed"):
        try:
            if key in motion and math.isfinite(float(motion[key])) and not 0.0 <= float(motion[key]) <= 100.0:
                errors.append(f"motion.{key} must be in the SO101 gripper range 0..100")
        except (TypeError, ValueError):
            pass

    try:
        place_xyz = np.asarray(placement["table_xyz_m"], dtype=float)
        if place_xyz.shape != (3,) or not np.isfinite(place_xyz).all():
            raise ValueError
    except (KeyError, TypeError, ValueError):
        errors.append("placement.table_xyz_m must be [x, y, z] in meters")
    try:
        bounds = np.asarray(kinematics["workspace_bounds_robot_m"], dtype=float)
        if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        errors.append("kinematics.workspace_bounds_robot_m must be [[xmin,xmax],[ymin,ymax],[zmin,zmax]]")

    if recording.get("enabled", True):
        repo_id = str(recording.get("repo_id", ""))
        if "/" not in repo_id or repo_id.startswith("your_name/"):
            errors.append("recording.repo_id must be changed to a valid '<name>/<dataset>' value")
        try:
            if int(recording.get("fps", 0)) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append("recording.fps must be a positive integer")

    return errors
