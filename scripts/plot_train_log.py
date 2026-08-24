"""Plot metrics from a LeRobot training text log.

Example:
    python scripts/plot_train_log.py outputs/log/train_output1.txt

The script ignores tqdm progress-bar text and parses only metric records after
``ot_train.py:609``. It writes a PNG dashboard and a CSV containing the parsed
records.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


METRIC_RE = re.compile(r"(?<!\w)(?P<key>[A-Za-z][A-Za-z0-9_/]*):(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
STEP_RE = re.compile(r"\bstep:(?P<value>[-+]?\d*\.?\d+)(?P<unit>[KMG]?)\b")
SAMPLE_RE = re.compile(r"\bsmpl:(?P<value>[-+]?\d*\.?\d+)(?P<unit>[KMG]?)\b")


def scaled_number(value: str, unit: str) -> float:
    return float(value) * {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9}[unit]


def parse_log(path: Path) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "ot_train.py:609" not in line or " step:" not in line:
            continue

        step_match = STEP_RE.search(line)
        sample_match = SAMPLE_RE.search(line)
        if not step_match or not sample_match:
            continue

        record = {
            "step": scaled_number(step_match["value"], step_match["unit"]),
            "samples": scaled_number(sample_match["value"], sample_match["unit"]),
        }
        # Start after the training logger marker so timestamp fragments are not
        # mistaken for metrics.
        metric_text = line.split("ot_train.py:609", 1)[1]
        for match in METRIC_RE.finditer(metric_text):
            key = match["key"]
            if key not in {"step", "smpl"}:
                record[key] = float(match["value"])
        records.append(record)
    return records


def write_csv(records: list[dict[str, float]], path: Path) -> None:
    keys = ["step", "samples"] + sorted({key for row in records for key in row if key not in {"step", "samples"}})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def plot_dashboard(records: list[dict[str, float]], output: Path) -> None:
    x = [row["step"] for row in records]

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True, constrained_layout=True)
    fig.suptitle("Training metrics", fontsize=16)

    def draw(ax, names: list[str], title: str, ylabel: str, log_y: bool = False) -> None:
        for name in names:
            values = [row.get(name, float("nan")) for row in records]
            ax.plot(x, values, linewidth=1.3, label=name)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if log_y:
            ax.set_yscale("log")
        if len(names) > 1:
            ax.legend(frameon=False)

    draw(axes[0, 0], ["loss", "l1_loss", "kld_loss"], "Loss", "value")
    draw(axes[0, 1], ["grdn"], "Gradient norm", "value")
    draw(axes[1, 0], ["lr"], "Learning rate", "value", log_y=True)
    draw(axes[1, 1], ["smp/s"], "Throughput", "samples / s")
    draw(axes[2, 0], ["updt_s", "data_s"], "Step time", "seconds")
    draw(axes[2, 1], ["mem_gb"], "GPU memory", "GB")

    axes[2, 0].set_xlabel("Training step")
    axes[2, 1].set_xlabel("Training step")
    for ax in axes.flat:
        ax.ticklabel_format(axis="x", style="plain")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="training log text file")
    parser.add_argument("-o", "--output-dir", type=Path, default=None, help="directory for generated files")
    args = parser.parse_args()

    records = parse_log(args.log)
    if not records:
        raise SystemExit(f"No training metric records found in {args.log}")

    output_dir = args.output_dir or args.log.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    dashboard = output_dir / f"{args.log.stem}_dashboard.png"
    csv_path = output_dir / f"{args.log.stem}_metrics.csv"
    plot_dashboard(records, dashboard)
    write_csv(records, csv_path)
    print(f"Parsed {len(records)} records")
    print(f"Dashboard: {dashboard}")
    print(f"CSV:       {csv_path}")


if __name__ == "__main__":
    main()
