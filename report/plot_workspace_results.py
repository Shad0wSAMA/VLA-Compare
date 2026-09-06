"""Render SO101 VLA / DAgger test outcomes on a clean workspace diagram.

Example::

    cd report
    python3 plot_workspace_results.py

The CSV must contain ``x_mm``, ``y_mm`` and ``result`` columns.  ``result``
accepts success/failure, pass/fail, true/false, or 1/0.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


WORKSPACE_WIDTH_MM = 360
WORKSPACE_HEIGHT_MM = 240
GRID_STEP_MM = 30
SUCCESS_COLOR = "#1aa647"
FAILURE_COLOR = "#e0292e"
ID_REGION_COLOR = "#edf7e8"
OOD_REGION_COLOR = "#e3f0fa"
HIL_REGION_COLOR = "#fff0d8"
ID_REGION_BOUNDS_MM = (60, 60, 240, 120)
HIL_REGION_BOUNDS_MM = (300, 60, 60, 120)
REPORT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_PATH = REPORT_DIR / "hil_test_results.json"
DEFAULT_OUTPUT_PATH = REPORT_DIR / "hil_results_region.png"


@dataclass(frozen=True)
class TestResult:
    x_mm: float
    y_mm: float
    success: bool


def _parse_result(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"success", "pass", "passed", "true", "1", "yes"}:
        return True
    if normalized in {"failure", "fail", "failed", "false", "0", "no"}:
        return False
    raise ValueError(f"Unknown result {value!r}; use success/failure or true/false.")


def load_results(csv_path: Path) -> list[TestResult]:
    """Load spatial test outcomes from a UTF-8 CSV file."""
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"x_mm", "y_mm", "result"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError("CSV must contain x_mm, y_mm, result columns.")
        results = [
            TestResult(float(row["x_mm"]), float(row["y_mm"]), _parse_result(row["result"]))
            for row in reader
        ]
    if not results:
        raise ValueError("CSV contains no test results.")
    return results


def load_results_json(json_path: Path) -> list[TestResult]:
    """Load grid-intersection test outcomes from a JSON test-set file."""
    with json_path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    rows = document["test_results"] if isinstance(document, dict) else document
    if not isinstance(rows, list) or not rows:
        raise ValueError("JSON must contain a non-empty test_results list.")
    try:
        return [TestResult(float(row["x_mm"]), float(row["y_mm"]), _parse_result(str(row["result"]))) for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Each JSON entry needs x_mm, y_mm, and result.") from error


def _validate_grid_intersections(results: list[TestResult]) -> None:
    for item in results:
        if not (0 <= item.x_mm <= WORKSPACE_WIDTH_MM and 0 <= item.y_mm <= WORKSPACE_HEIGHT_MM):
            raise ValueError(f"Point ({item.x_mm}, {item.y_mm}) mm is outside the workspace.")
        if item.x_mm % GRID_STEP_MM != 0 or item.y_mm % GRID_STEP_MM != 0:
            raise ValueError(f"Point ({item.x_mm}, {item.y_mm}) mm is not a {GRID_STEP_MM} mm grid intersection.")


def _draw_aruco_style_marker(axis: plt.Axes, x_mm: float, y_mm: float, size_mm: float, pattern: tuple[tuple[int, ...], ...]) -> None:
    axis.add_patch(Rectangle((x_mm, y_mm), size_mm, size_mm, facecolor="black", edgecolor="black", zorder=2))
    cell_size = size_mm / (len(pattern) + 2)
    for row_index, row in enumerate(pattern):
        for column_index, bit in enumerate(row):
            if bit:
                axis.add_patch(
                    Rectangle(
                        (x_mm + (column_index + 1) * cell_size, y_mm + (row_index + 1) * cell_size),
                        cell_size,
                        cell_size,
                        facecolor="#f6f2e8",
                        edgecolor="none",
                        zorder=3,
                    )
                )


def _draw_workspace(axis: plt.Axes, show_id_ood: bool = False, show_hil: bool = False) -> None:
    axis.set_facecolor(OOD_REGION_COLOR if show_id_ood else "#f5f1e8")
    if show_id_ood:
        x_mm, y_mm, width_mm, height_mm = ID_REGION_BOUNDS_MM
        axis.add_patch(
            Rectangle(
                (x_mm, y_mm), width_mm, height_mm,
                facecolor=ID_REGION_COLOR, edgecolor="#86aa82", linewidth=1.2,
                label="ID region", zorder=-1,
            )
        )
    if show_hil:
        x_mm, y_mm, width_mm, height_mm = HIL_REGION_BOUNDS_MM
        axis.add_patch(
            Rectangle(
                (x_mm, y_mm), width_mm, height_mm,
                facecolor=HIL_REGION_COLOR, edgecolor="#cf9c51", linewidth=1.2,
                label="HIL / DAgger region", zorder=-0.5,
            )
        )
    for position in range(0, WORKSPACE_WIDTH_MM + 1, GRID_STEP_MM):
        axis.axvline(position, color="#c9c3b8", linewidth=0.75, zorder=0)
    for position in range(0, WORKSPACE_HEIGHT_MM + 1, GRID_STEP_MM):
        axis.axhline(position, color="#c9c3b8", linewidth=0.75, zorder=0)

    axis.add_patch(
        Rectangle((120, 170), 120, 70, facecolor="#bb402d", edgecolor="#8d2e21", linewidth=1.5, zorder=1)
    )
    axis.text(180, 205, "Placement area", color="white", ha="center", va="center", fontsize=10, weight="bold", zorder=2)

    patterns = (
        ((1, 1, 0), (0, 1, 1), (0, 0, 1)),
        ((1, 0, 1), (1, 1, 0), (0, 1, 0)),
        ((0, 1, 1), (1, 0, 1), (1, 0, 0)),
        ((1, 0, 0), (0, 1, 1), (1, 1, 0)),
    )
    marker_size = 28
    marker_positions = ((4, 4), (328, 4), (4, 208), (328, 208))
    for position, pattern in zip(marker_positions, patterns, strict=True):
        _draw_aruco_style_marker(axis, *position, marker_size, pattern)

    axis.set_xlim(0, WORKSPACE_WIDTH_MM)
    axis.set_ylim(0, WORKSPACE_HEIGHT_MM)
    axis.set_aspect("equal")
    axis.set_xticks(range(0, WORKSPACE_WIDTH_MM + 1, GRID_STEP_MM))
    axis.set_yticks(range(0, WORKSPACE_HEIGHT_MM + 1, GRID_STEP_MM))
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.tick_params(labelsize=8)


def _add_legends(axis: plt.Axes, show_id_ood: bool, show_hil: bool):
    """Place outcome markers and workspace-region keys on separate legend rows."""
    handles, labels = axis.get_legend_handles_labels()
    outcome_handles = [
        handle for handle, label in zip(handles, labels, strict=True)
        if label.startswith(("Success", "Failure"))
    ]
    outcome_legend = axis.legend(
        handles=outcome_handles, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False,
    )
    if not show_id_ood:
        return outcome_legend, None

    axis.add_artist(outcome_legend)
    region_handles = [
        Patch(facecolor=ID_REGION_COLOR, edgecolor="#86aa82", label="ID region"),
        Patch(facecolor=OOD_REGION_COLOR, edgecolor="none", label="OOD region"),
    ]
    if show_hil:
        region_handles.append(Patch(facecolor=HIL_REGION_COLOR, edgecolor="#cf9c51", label="HIL / DAgger region"))
    region_legend = axis.legend(
        handles=region_handles, loc="lower center", bbox_to_anchor=(0.5, -0.37), ncol=3, frameon=False,
    )
    return outcome_legend, region_legend


def plot_workspace_results(
    results: list[TestResult],
    output_path: Path,
    title: str = "SO101 workspace test results",
    show_id_ood: bool = False,
    show_hil: bool = False,
) -> Path:
    """Draw the workspace and save green success / red failure dots to PNG."""
    _validate_grid_intersections(results)
    figure, axis = plt.subplots(figsize=(11, 8))
    show_id_ood = show_id_ood or show_hil
    _draw_workspace(axis, show_id_ood=show_id_ood, show_hil=show_hil)
    success = [item for item in results if item.success]
    failure = [item for item in results if not item.success]
    if success:
        axis.scatter([item.x_mm for item in success], [item.y_mm for item in success], s=180, c=SUCCESS_COLOR, edgecolors="white", linewidths=1.4, label=f"Success ({len(success)})", zorder=4)
    if failure:
        axis.scatter([item.x_mm for item in failure], [item.y_mm for item in failure], s=180, c=FAILURE_COLOR, edgecolors="white", linewidths=1.4, label=f"Failure ({len(failure)})", zorder=4)
    axis.set_title(title, pad=14, weight="bold")
    _add_legends(axis, show_id_ood=show_id_ood, show_hil=show_hil)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw SO101 workspace success/failure test results.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", type=Path, help="CSV with x_mm,y_mm,result")
    source.add_argument("--json", type=Path, default=DEFAULT_RESULTS_PATH, help="JSON test-set file (default: report/workspace_test_results.json)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--title", default="SO101 VLA / DAgger workspace test results")
    parser.add_argument("--show-id-ood", action="store_true", help="Show the ID rectangle and OOD background.")
    parser.add_argument("--show-hil", action="store_true", help="Show the DAgger/HIL training region (also shows ID/OOD).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = load_results(args.csv) if args.csv else load_results_json(args.json)
    output_path = plot_workspace_results(
        results, args.output.resolve(), args.title,
        show_id_ood=args.show_id_ood, show_hil=args.show_hil,
    )
    print(f"Saved {len(results)} test results to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
