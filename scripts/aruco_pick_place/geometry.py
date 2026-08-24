"""Coordinate transforms and pick/place waypoint generation."""

from __future__ import annotations

from typing import Any

import numpy as np


def rpy_matrix_deg(rpy_deg: list[float]) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=float))
    rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return rz @ ry @ rx


def table_target_to_robot_pose(table_xyz_m: np.ndarray, kinematics: dict[str, Any]) -> np.ndarray:
    table_to_robot = np.asarray(kinematics["table_to_robot"], dtype=float)
    pose_table = np.eye(4)
    pose_table[:3, :3] = rpy_matrix_deg(kinematics["tool_rotation_rpy_deg_in_table"])
    pose_table[:3, 3] = np.asarray(table_xyz_m, dtype=float)
    return table_to_robot @ pose_table


def check_workspace(position_robot_m: np.ndarray, bounds: np.ndarray, label: str) -> None:
    if np.any(position_robot_m < bounds[:, 0]) or np.any(position_robot_m > bounds[:, 1]):
        raise RuntimeError(f"{label} target {position_robot_m.tolist()} is outside robot workspace bounds {bounds.tolist()}.")


def build_pick_place_poses(cube_table_xy_m: np.ndarray, config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Build and workspace-check all Cartesian waypoints without solving IK."""
    motion = config["motion"]
    place_xyz = np.asarray(config["placement"]["table_xyz_m"], dtype=float)
    grasp_xyz = np.asarray([cube_table_xy_m[0], cube_table_xy_m[1], float(motion["grasp_z_m"])])
    waypoints = {
        "grasp approach": grasp_xyz + [0.0, 0.0, float(motion["approach_height_m"])],
        "grasp": grasp_xyz,
        "grasp lift": grasp_xyz + [0.0, 0.0, float(motion["lift_height_m"])],
        "place approach": place_xyz + [0.0, 0.0, float(motion["approach_height_m"])],
        "place": place_xyz,
    }
    poses = {label: table_target_to_robot_pose(xyz, config["kinematics"]) for label, xyz in waypoints.items()}
    bounds = np.asarray(config["kinematics"]["workspace_bounds_robot_m"], dtype=float)
    for label, pose in poses.items():
        check_workspace(pose[:3, 3], bounds, label)
    return poses
