"""Regression test for the clean SO101 workspace result map."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))

from report.plot_workspace_results import (  # noqa: E402
    DEFAULT_RESULTS_PATH,
    GRID_STEP_MM,
    _add_legends,
    _draw_workspace,
    load_results,
    load_results_json,
    plot_workspace_results,
)


class WorkspaceResultMapTest(unittest.TestCase):
    def test_id_ood_hil_legend_uses_two_rows(self) -> None:
        """Combining outcome and region keys in one row makes the report hard to scan."""
        figure, axis = plt.subplots()
        try:
            axis.scatter([60], [60], label="Success (1)")
            axis.scatter([90], [60], label="Failure (1)")
            _draw_workspace(axis, show_id_ood=True, show_hil=True)
            outcome_legend, region_legend = _add_legends(axis, show_id_ood=True, show_hil=True)
            self.assertEqual([text.get_text() for text in outcome_legend.get_texts()], ["Success (1)", "Failure (1)"])
            self.assertEqual(
                [text.get_text() for text in region_legend.get_texts()],
                ["ID region", "OOD region", "HIL / DAgger region"],
            )
        finally:
            plt.close(figure)

    def test_default_json_uses_only_grid_intersections(self) -> None:
        """Off-grid examples would misrepresent the discrete test-set locations."""
        results = load_results_json(DEFAULT_RESULTS_PATH)
        self.assertGreater(len(results), 0)
        for item in results:
            self.assertEqual(item.x_mm % GRID_STEP_MM, 0)
            self.assertEqual(item.y_mm % GRID_STEP_MM, 0)

    def test_workspace_uses_a_twelve_by_eight_cell_grid(self) -> None:
        """A wrong workspace size would make target locations incomparable across runs."""
        figure, axis = plt.subplots()
        try:
            _draw_workspace(axis)
            self.assertEqual(axis.get_xlim(), (0.0, 360.0))
            self.assertEqual(axis.get_ylim(), (0.0, 240.0))
            self.assertEqual(len(axis.get_xticks()) - 1, 12)
            self.assertEqual(len(axis.get_yticks()) - 1, 8)
        finally:
            plt.close(figure)

    def test_id_ood_overlay_marks_the_requested_rectangle(self) -> None:
        """A shifted ID boundary would report in-distribution performance incorrectly."""
        figure, axis = plt.subplots()
        try:
            _draw_workspace(axis, show_id_ood=True)
            id_regions = [patch for patch in axis.patches if patch.get_label() == "ID region"]
            self.assertEqual(len(id_regions), 1)
            id_region = id_regions[0]
            self.assertEqual((id_region.get_x(), id_region.get_y()), (60, 60))
            self.assertEqual((id_region.get_width(), id_region.get_height()), (240, 120))
            np.testing.assert_allclose(axis.get_facecolor()[:3], (0.89, 0.94, 0.98), atol=0.01)
        finally:
            plt.close(figure)

    def test_hil_overlay_marks_the_dagger_training_region(self) -> None:
        """A misplaced HIL region would misstate where DAgger data was collected."""
        figure, axis = plt.subplots()
        try:
            _draw_workspace(axis, show_id_ood=True, show_hil=True)
            hil_regions = [patch for patch in axis.patches if patch.get_label() == "HIL / DAgger region"]
            self.assertEqual(len(hil_regions), 1)
            hil_region = hil_regions[0]
            self.assertEqual((hil_region.get_x(), hil_region.get_y()), (300, 60))
            self.assertEqual((hil_region.get_width(), hil_region.get_height()), (60, 120))
        finally:
            plt.close(figure)

    def test_csv_results_render_as_colored_markers(self) -> None:
        """Changing a result from success to failure must change its visible marker colour."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            csv_path = temporary_path / "results.csv"
            output_path = temporary_path / "workspace_results.png"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["x_mm", "y_mm", "result"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"x_mm": 60, "y_mm": 60, "result": "success"},
                        {"x_mm": 240, "y_mm": 60, "result": "failure"},
                    ]
                )

            results = load_results(csv_path)
            plot_workspace_results(results, output_path, title="Test result map")

            self.assertEqual([(item.x_mm, item.y_mm, item.success) for item in results], [(60.0, 60.0, True), (240.0, 60.0, False)])
            self.assertTrue(output_path.is_file())
            image = mpimg.imread(output_path)[..., :3]
            self.assertGreater(np.count_nonzero(np.all(np.isclose(image, [0.10, 0.65, 0.28], atol=0.02), axis=2)), 100)
            self.assertGreater(np.count_nonzero(np.all(np.isclose(image, [0.88, 0.16, 0.18], atol=0.02), axis=2)), 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
