"""Aggregate eval_soccer.py CSVs into a comparison bar chart.

Each CSV is expected to have columns: step, env, tar_vx, tar_vy, ball_vx,
ball_vy, dist (as written by eval_soccer.py).

Usage:
    python scripts/plot_phase_comparison.py \\
        --csv output/phase_compare/eval_phase1.csv --label "Phase 1 only" \\
        --csv output/phase_compare/eval_phase2_scratch.csv --label "Phase 2 only" \\
        --csv output/phase_compare/eval_phase2.csv --label "Phase 1 + Phase 2" \\
        --out output/phase_compare/comparison.png
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", action="append", required=True,
                   help="Path to an eval CSV. Repeat for each variant.")
    p.add_argument("--label", action="append", required=True,
                   help="Display label for the corresponding --csv. Order matches.")
    p.add_argument("--out", default="output/phase_compare/comparison.png")
    return p.parse_args()


def load_csv(path):
    tar_vx, tar_vy, ball_vx, ball_vy, dist = [], [], [], [], []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tar_vx.append(float(row["tar_vx"]))
            tar_vy.append(float(row["tar_vy"]))
            ball_vx.append(float(row["ball_vx"]))
            ball_vy.append(float(row["ball_vy"]))
            dist.append(float(row["dist"]))
    return (np.array(tar_vx), np.array(tar_vy),
            np.array(ball_vx), np.array(ball_vy),
            np.array(dist))


def compute_metrics(tvx, tvy, bvx, bvy, dist):
    tar_speed = np.sqrt(tvx ** 2 + tvy ** 2)
    ball_speed = np.sqrt(bvx ** 2 + bvy ** 2)
    speed_err = np.abs(tar_speed - ball_speed)

    eps = 1e-6
    moving = ball_speed > 0.05
    ball_dir = np.stack([bvx, bvy], axis=-1) / np.clip(ball_speed[:, None], eps, None)
    tar_dir = np.stack([tvx, tvy], axis=-1) / np.clip(tar_speed[:, None], eps, None)
    cos_align = np.sum(ball_dir * tar_dir, axis=-1)
    cos_moving = cos_align[moving]
    angle_err_deg = np.degrees(np.arccos(np.clip(cos_moving, -1.0, 1.0)))

    return {
        "speed_err_mean": float(speed_err.mean()),
        "speed_err_std": float(speed_err.std()),
        "angle_err_mean": float(angle_err_deg.mean()) if angle_err_deg.size else float("nan"),
        "angle_err_std": float(angle_err_deg.std()) if angle_err_deg.size else 0.0,
        "dist_mean": float(dist.mean()),
        "dist_std": float(dist.std()),
        "cos_align_mean": float(cos_moving.mean()) if cos_moving.size else float("nan"),
    }


def main():
    args = parse_args()
    if (len(args.csv) != len(args.label)):
        raise SystemExit("number of --csv and --label flags must match")

    variants = []
    for path, label in zip(args.csv, args.label):
        data = load_csv(path)
        m = compute_metrics(*data)
        variants.append((label, m))
        print(f"{label}: {m}")

    metric_keys = [
        ("speed_err_mean", "speed_err_std", "Speed error (m/s)"),
        ("angle_err_mean", "angle_err_std", "Angle error (deg)"),
        ("dist_mean",      "dist_std",      "Char-ball distance (m)"),
    ]

    labels = [v[0] for v in variants]
    n_variants = len(variants)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"][:n_variants]

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(4.2 * len(metric_keys), 4.5))
    if (len(metric_keys) == 1):
        axes = [axes]
    x = np.arange(n_variants)
    for ax, (mean_k, std_k, title) in zip(axes, metric_keys):
        means = [v[1][mean_k] for v in variants]
        stds = [v[1][std_k] for v in variants]
        bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Soccer policy comparison")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
