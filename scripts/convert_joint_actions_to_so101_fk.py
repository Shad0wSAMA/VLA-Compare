"""Convert a LeRobot dataset's joint-space actions to the SO-101 FK action space.

The source dataset is never modified. ``observation.state`` remains in joint
space; only ``action`` becomes ``[x, y, z, wrist_yaw, wrist_roll, gripper]``.
Unchanged videos are hard-linked when possible to avoid duplicating large files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from robot.so101_kinematics import ACTION_NAMES, SO101Kinematics  # noqa: E402


JOINT_ACTION_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
FK_ACTION_NAMES = list(ACTION_NAMES)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=4, ensure_ascii=False)
        file.write("\n")


def _copy_unchanged_files(source: Path, destination: Path, copy_videos: bool) -> None:
    """Copy metadata/assets, skipping data parquet files that will be regenerated."""
    for source_path in source.rglob("*"):
        relative = source_path.relative_to(source)
        destination_path = destination / relative
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
            continue
        if relative.parts[0] == "data" and source_path.suffix == ".parquet":
            continue
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if relative.parts[0] == "videos" and not copy_videos:
            try:
                os.link(source_path, destination_path)
                continue
            except OSError:
                pass
        shutil.copy2(source_path, destination_path)


def _convert_actions(joints: np.ndarray, model: SO101Kinematics) -> np.ndarray:
    if joints.ndim != 2 or joints.shape[1] != 6:
        raise ValueError(f"Expected action array with shape (N, 6), got {joints.shape}")
    converted = np.empty_like(joints, dtype=np.float32)
    for index, joint_action in enumerate(joints):
        converted[index] = model.forward(joint_action).astype(np.float32)
    return converted


def _action_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    values64 = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(values64, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    return {
        "min": values64.min(axis=0).tolist(),
        "max": values64.max(axis=0).tolist(),
        "mean": values64.mean(axis=0).tolist(),
        "std": values64.std(axis=0).tolist(),
        "count": [int(values64.shape[0])],
        "q01": quantiles[0].tolist(),
        "q10": quantiles[1].tolist(),
        "q50": quantiles[2].tolist(),
        "q90": quantiles[3].tolist(),
        "q99": quantiles[4].tolist(),
    }


def _without_stale_fingerprint(table: pa.Table) -> pa.Table:
    metadata = table.schema.metadata
    if not metadata or b"huggingface" not in metadata:
        return table
    try:
        huggingface = json.loads(metadata[b"huggingface"].decode("utf-8"))
        huggingface.pop("fingerprint", None)
        updated = dict(metadata)
        updated[b"huggingface"] = json.dumps(huggingface, separators=(",", ":")).encode("utf-8")
        return table.replace_schema_metadata(updated)
    except (KeyError, TypeError, ValueError, UnicodeDecodeError):
        return table


def convert_dataset(
    source: Path,
    destination: Path,
    urdf: Path,
    *,
    copy_videos: bool = False,
) -> None:
    source = source.resolve()
    destination = destination.resolve()
    urdf = urdf.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    if destination.exists():
        raise FileExistsError(
            f"Destination already exists: {destination}. Remove it explicitly or choose another --output."
        )
    if source == destination or source in destination.parents:
        raise ValueError("Output must not be the source directory or a child of it")

    info_path = source / "meta" / "info.json"
    stats_path = source / "meta" / "stats.json"
    info = _load_json(info_path)
    source_names = info.get("features", {}).get("action", {}).get("names")
    if source_names != JOINT_ACTION_NAMES:
        raise ValueError(
            "Source action names are not the expected SO-101 joint order:\n"
            f"expected={JOINT_ACTION_NAMES}\nactual={source_names}"
        )

    parquet_paths = sorted((source / "data").rglob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No data parquet files found under {source / 'data'}")

    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex[:8]}")
    model = SO101Kinematics(urdf)
    all_actions: list[np.ndarray] = []
    try:
        temporary.mkdir(parents=True)
        _copy_unchanged_files(source, temporary, copy_videos)
        for file_index, source_parquet in enumerate(parquet_paths, start=1):
            relative = source_parquet.relative_to(source)
            output_parquet = temporary / relative
            output_parquet.parent.mkdir(parents=True, exist_ok=True)
            table = pq.read_table(source_parquet)
            action_index = table.schema.get_field_index("action")
            if action_index < 0:
                raise ValueError(f"Missing action column: {source_parquet}")
            joints = np.asarray(table.column(action_index).to_pylist(), dtype=np.float64)
            fk_actions = _convert_actions(joints, model)
            all_actions.append(fk_actions)
            flat = pa.array(fk_actions.reshape(-1), type=pa.float32())
            action_column = pa.FixedSizeListArray.from_arrays(flat, 6)
            converted_table = table.set_column(action_index, "action", action_column)
            converted_table = _without_stale_fingerprint(converted_table)
            pq.write_table(converted_table, output_parquet, compression="snappy")
            print(f"[{file_index}/{len(parquet_paths)}] {relative}: {len(fk_actions)} frames")

        converted_actions = np.concatenate(all_actions, axis=0)
        output_info = _load_json(temporary / "meta" / "info.json")
        output_info["features"]["action"]["names"] = FK_ACTION_NAMES
        _write_json(temporary / "meta" / "info.json", output_info)

        output_stats = _load_json(temporary / "meta" / "stats.json")
        output_stats["action"] = _action_stats(converted_actions)
        _write_json(temporary / "meta" / "stats.json", output_stats)

        expected_frames = int(output_info["total_frames"])
        if len(converted_actions) != expected_frames:
            raise ValueError(
                f"Frame-count mismatch: converted {len(converted_actions)}, info.json says {expected_frames}"
            )
        first_output = pq.read_table(temporary / parquet_paths[0].relative_to(source), columns=["action"])
        saved_first = np.asarray(first_output["action"][0].as_py(), dtype=np.float64)
        if not np.allclose(saved_first, converted_actions[0], atol=1e-7):
            raise RuntimeError("Verification failed: first saved FK action differs from computed value")

        temporary.rename(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    video_mode = "copied" if copy_videos else "hard-linked when supported"
    print(f"Done: {destination}")
    print(f"Converted action names: {FK_ACTION_NAMES}")
    print(f"Frames: {len(converted_actions)}; videos: {video_mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=WORKSPACE / "training" / "manual_data2_joints",
        help="Source LeRobot dataset containing six joint-space actions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "training" / "manual_data2_fk",
        help="New output dataset. It must not already exist.",
    )
    parser.add_argument("--urdf", type=Path, default=WORKSPACE / "robot" / "so101.urdf")
    parser.add_argument(
        "--copy-videos",
        action="store_true",
        help="Copy video bytes instead of using same-volume hard links.",
    )
    args = parser.parse_args()
    convert_dataset(args.input, args.output, args.urdf, copy_videos=args.copy_videos)


if __name__ == "__main__":
    main()
