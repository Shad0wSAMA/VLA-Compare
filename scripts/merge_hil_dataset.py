r"""Safely merge a corrections-only HIL dataset into a base LeRobot dataset.

The source datasets are never modified. The script first creates a cleaned HIL
copy without the ``intervention`` feature, then merges that copy with the base
dataset by invoking the version-matched ``lerobot-edit-dataset`` executable.

Run from the workspace root::

    .\venv\Scripts\python.exe scripts\merge_hil_dataset.py

The default inputs merge ``training/manual_data2`` with
``training/rollout_stack_cube_data2_hil_run1``. Override them with
``--original-dataset-path``, ``--hil-dataset-path``, and
``--output-dataset-path`` when needed.

Use ``--dry-run`` to validate metadata and print commands without writing data.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from lerobot.datasets.feature_utils import features_equal_for_merge


WORKSPACE = Path(__file__).resolve().parents[1]
EDIT_DATASET = WORKSPACE / "venv" / "Scripts" / "lerobot-edit-dataset.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-dataset-path",
        "--base-root",
        dest="base_root",
        type=Path,
        default=WORKSPACE / "training" / "manual_data2",
        help="Path to the original LeRobot dataset (default: training/manual_data2).",
    )
    parser.add_argument(
        "--hil-dataset-path",
        "--hil-root",
        dest="hil_root",
        type=Path,
        default=WORKSPACE / "training" / "rollout_stack_cube_data2_hil_run1",
        help=(
            "Path to the corrections-only HIL dataset "
            "(default: training/rollout_stack_cube_data2_hil_run1)."
        ),
    )
    parser.add_argument(
        "--normalized-hil-root",
        type=Path,
        default=WORKSPACE / "training" / "rollout_stack_cube_data2_hil_normalized",
        help="Temporary HIL copy re-encoded to the base dataset's video codec when needed.",
    )
    parser.add_argument(
        "--clean-hil-root",
        type=Path,
        default=WORKSPACE / "training" / "rollout_stack_cube_data2_hil_clean",
    )
    parser.add_argument(
        "--output-dataset-path",
        "--output-root",
        dest="output_root",
        type=Path,
        default=WORKSPACE / "training" / "manual_data2_hil_v1",
        help="Path for the merged dataset (default: training/manual_data2_hil_v1).",
    )
    parser.add_argument("--base-repo-id", default="manual_data2")
    parser.add_argument("--hil-repo-id", default="rollout_stack_cube_data2_hil_run1")
    parser.add_argument(
        "--normalized-hil-repo-id", default="rollout_stack_cube_data2_hil_normalized"
    )
    parser.add_argument("--clean-hil-repo-id", default="rollout_stack_cube_data2_hil_clean")
    parser.add_argument("--output-repo-id", default="manual_data2_hil_v1")
    parser.add_argument(
        "--hil-repeat",
        type=int,
        default=1,
        help="Repeat the cleaned HIL dataset in the merge to increase its sampling share.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_info(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Invalid LeRobot dataset; missing {info_path}")
    with info_path.open(encoding="utf-8") as file:
        return json.load(file)


def count_intervention_values(root: Path) -> tuple[int, int]:
    true_count = 0
    false_count = 0
    parquet_files = sorted((root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet data files found under {root / 'data'}")

    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file, columns=["intervention"])
        for value in table.column("intervention").to_pylist():
            # Shape-(1,) numeric features may be materialized as either a scalar
            # or a one-element list, depending on the datasets/Arrow version.
            if isinstance(value, list):
                value = value[0]
            if bool(value):
                true_count += 1
            else:
                false_count += 1
    return true_count, false_count


def video_codecs(info: dict[str, Any]) -> set[str]:
    return {
        str(feature.get("info", {}).get("video.codec"))
        for feature in info["features"].values()
        if feature.get("dtype") == "video"
    }


def _without_video_codec(features: dict[str, Any]) -> dict[str, Any]:
    # Re-encoding below makes the codec equal. Ignore only this field during
    # source validation; the LeRobot merge validation handles the remaining
    # encoder metadata using its own compatibility rules.
    copied = json.loads(json.dumps(features))
    for feature in copied.values():
        info = feature.get("info")
        if isinstance(info, dict):
            info.pop("video.codec", None)
    return copied


def validate_sources(base_root: Path, hil_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    base_info = load_info(base_root)
    hil_info = load_info(hil_root)

    for field in ("fps", "robot_type"):
        if base_info.get(field) != hil_info.get(field):
            raise ValueError(
                f"Dataset {field} mismatch: base={base_info.get(field)!r}, HIL={hil_info.get(field)!r}"
            )

    hil_features = dict(hil_info["features"])
    intervention_feature = hil_features.pop("intervention", None)
    if intervention_feature is None:
        raise ValueError("The HIL dataset has no 'intervention' feature; refusing an ambiguous merge.")
    if not features_equal_for_merge(
        _without_video_codec(base_info["features"]), _without_video_codec(hil_features)
    ):
        raise ValueError(
            "Base and HIL feature schemas are incompatible after removing 'intervention'. "
            "Check camera names/resolution and state/action names."
        )

    true_count, false_count = count_intervention_values(hil_root)
    if false_count:
        raise ValueError(
            f"HIL dataset contains {false_count} autonomous frames and {true_count} intervention frames. "
            "This script only merges corrections-only datasets into behavior-cloning data."
        )
    if true_count == 0:
        raise ValueError("The HIL dataset contains no intervention frames.")

    return base_info, hil_info


def run(command: list[str], *, dry_run: bool) -> None:
    print("\n> " + subprocess.list2cmdline(command))
    if not dry_run:
        subprocess.run(command, cwd=WORKSPACE, check=True)


def main() -> int:
    args = parse_args()
    if args.hil_repeat < 1:
        raise ValueError("--hil-repeat must be at least 1")
    if not EDIT_DATASET.is_file():
        raise FileNotFoundError(f"Missing LeRobot dataset editor: {EDIT_DATASET}")

    base_root = args.base_root.resolve()
    hil_root = args.hil_root.resolve()
    normalized_hil_root = args.normalized_hil_root.resolve()
    clean_hil_root = args.clean_hil_root.resolve()
    output_root = args.output_root.resolve()

    if base_root == hil_root:
        raise ValueError("Base and HIL roots must be different")
    if normalized_hil_root in (base_root, hil_root) or clean_hil_root in (
        base_root,
        hil_root,
        normalized_hil_root,
    ) or output_root in (base_root, hil_root, normalized_hil_root, clean_hil_root):
        raise ValueError("Output roots must be distinct from all source and intermediate roots")
    for target in (normalized_hil_root, clean_hil_root, output_root):
        if target.exists():
            raise FileExistsError(
                f"Output already exists: {target}\n"
                "Choose a new --clean-hil-root/--output-root; the script will not overwrite datasets."
            )

    base_info, hil_info = validate_sources(base_root, hil_root)
    base_frames = int(base_info.get("total_frames", 0))
    hil_frames = int(hil_info.get("total_frames", 0))
    merged_frames = base_frames + hil_frames * args.hil_repeat
    hil_share = (hil_frames * args.hil_repeat / merged_frames * 100.0) if merged_frames else 0.0

    print("Dataset validation passed")
    print(f"  base: {base_info.get('total_episodes', 0)} episodes, {base_frames} frames")
    print(f"  HIL:  {hil_info.get('total_episodes', 0)} episodes, {hil_frames} frames")
    print(f"  HIL repeat: {args.hil_repeat} (estimated merged frame share: {hil_share:.1f}%)")

    base_codecs = video_codecs(base_info)
    hil_codecs = video_codecs(hil_info)
    if len(base_codecs) != 1 or len(hil_codecs) != 1:
        raise ValueError(f"Expected one RGB video codec per dataset; base={base_codecs}, HIL={hil_codecs}")
    base_codec = next(iter(base_codecs))
    hil_codec = next(iter(hil_codecs))

    remove_input_root = hil_root
    remove_input_repo_id = args.hil_repo_id
    if base_codec != hil_codec:
        if base_codec not in {"h264", "hevc", "av1"}:
            raise ValueError(f"Cannot automatically normalize unsupported base video codec: {base_codec}")
        print(f"  video codec normalization required: HIL {hil_codec} -> base {base_codec}")
        reencode_command = [
            str(EDIT_DATASET),
            f"--repo_id={args.hil_repo_id}",
            f"--root={hil_root}",
            f"--new_repo_id={args.normalized_hil_repo_id}",
            f"--new_root={normalized_hil_root}",
            "--operation.type=reencode_videos",
            f"--operation.rgb_encoder.vcodec={base_codec}",
            "--operation.num_workers=2",
            "--push_to_hub=false",
        ]
        run(reencode_command, dry_run=args.dry_run)
        remove_input_root = normalized_hil_root
        remove_input_repo_id = args.normalized_hil_repo_id
    else:
        print(f"  video codecs already match: {base_codec}")

    remove_command = [
        str(EDIT_DATASET),
        f"--repo_id={remove_input_repo_id}",
        f"--root={remove_input_root}",
        f"--new_repo_id={args.clean_hil_repo_id}",
        f"--new_root={clean_hil_root}",
        "--operation.type=remove_feature",
        '--operation.feature_names=["intervention"]',
        "--push_to_hub=false",
    ]
    run(remove_command, dry_run=args.dry_run)

    repo_ids = [args.base_repo_id] + [args.clean_hil_repo_id] * args.hil_repeat
    roots = [str(base_root)] + [str(clean_hil_root)] * args.hil_repeat
    merge_command = [
        str(EDIT_DATASET),
        f"--new_repo_id={args.output_repo_id}",
        f"--new_root={output_root}",
        "--operation.type=merge",
        f"--operation.repo_ids={json.dumps(repo_ids)}",
        f"--operation.roots={json.dumps(roots)}",
        "--push_to_hub=false",
    ]
    run(merge_command, dry_run=args.dry_run)

    if args.dry_run:
        print("\nDry run complete; no files were written.")
    else:
        merged_info = load_info(output_root)
        expected_episodes = int(base_info.get("total_episodes", 0)) + int(
            hil_info.get("total_episodes", 0)
        ) * args.hil_repeat
        actual_episodes = int(merged_info.get("total_episodes", -1))
        if actual_episodes != expected_episodes:
            raise RuntimeError(
                f"Merged episode count mismatch: expected {expected_episodes}, got {actual_episodes}"
            )
        print(f"\nMerge complete: {output_root}")
        print(f"Episodes: {actual_episodes}; frames: {merged_info.get('total_frames', 'unknown')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
