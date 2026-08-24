"""从 LeRobot 训练日志中提取并绘制 learning curve。

用法：
    python scripts/plot_learning_curve.py outputs/train/act_data3/train.log

会在日志同目录生成：
    learning_curve.csv
    learning_curve.png
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


METRIC_RE = re.compile(
    r"step:(?P<step>\d+)(?P<step_unit>K|M)?\b.*?loss:(?P<loss>[-+]?\d*\.?\d+)"
    r".*?l1_loss:(?P<l1>[-+]?\d*\.?\d+)"
    r".*?kld_loss:(?P<kld>[-+]?\d*\.?\d+)",
    re.DOTALL,
)


def parse_step(match: re.Match[str]) -> int:
    value = int(match.group("step"))
    if match.group("step_unit") == "K":
        return value * 1_000
    if match.group("step_unit") == "M":
        return value * 1_000_000
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="绘制 LeRobot ACT learning curve")
    parser.add_argument("log_file", type=Path, help="训练输出日志文件")
    args = parser.parse_args()
    log_file = args.log_file.resolve()
    if not log_file.is_file():
        raise FileNotFoundError(f"找不到日志文件：{log_file}")

    rows: list[dict[str, float | int]] = []
    raw_text = log_file.read_bytes()
    encoding = "utf-16" if raw_text.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
    text = raw_text.decode(encoding, errors="replace")
    for match in METRIC_RE.finditer(text):
        rows.append(
            {
                "step": parse_step(match),
                "loss": float(match.group("loss")),
                "l1_loss": float(match.group("l1")),
                "kld_loss": float(match.group("kld")),
            }
        )

    if not rows:
        raise RuntimeError("日志中没有找到 step/loss/l1_loss/kld_loss 记录。")

    # 同一个 step 在追加日志时可能出现多次，只保留最后一次。
    rows = list({int(row["step"]): row for row in rows}.values())
    rows.sort(key=lambda row: int(row["step"]))
    csv_path = log_file.with_name("learning_curve.csv")
    png_path = log_file.with_name("learning_curve.png")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "l1_loss", "kld_loss"])
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib.pyplot as plt

    steps = [row["step"] for row in rows]
    plt.figure(figsize=(10, 5.5))
    plt.plot(steps, [row["loss"] for row in rows], label="total loss", linewidth=2)
    plt.plot(steps, [row["l1_loss"] for row in rows], label="L1 loss")
    plt.plot(steps, [row["kld_loss"] for row in rows], label="KL loss")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("ACT learning curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()

    print(f"提取了 {len(rows)} 个训练点")
    print(f"CSV：{csv_path}")
    print(f"图片：{png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
