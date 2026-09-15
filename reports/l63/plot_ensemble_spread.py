#!/usr/bin/env python3
"""Plot an EnKF/ETKF analysis trajectory with its ensemble-spread shadow.

Reads experiments/l63/baselines/baselines_trajectories.npz (produced by
eval_baselines.py / evaluation/run.py) and, for one test window, plots the
truth, the ensemble-mean analysis, and a shaded band of
mean +/- sqrt(ensemble_variance) for each state dimension (X, Y, Z).

Usage:
    python reports/l63/plot_ensemble_spread.py --case s0 --method EnKF --window 0
    python reports/l63/plot_ensemble_spread.py --case s1 --method ETKF --window 5 --output my_plot.png
"""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAJ_PATH = os.path.join(BASE, "experiments", "l63", "baselines", "baselines_trajectories.npz")
OUT_DIR = os.path.join(BASE, "reports", "l63", "outputs", "figs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["s0", "s1"], default="s0")
    parser.add_argument("--method", choices=["EnKF", "ETKF"], default="EnKF",
                         help="Only ensemble methods have a spread to plot.")
    parser.add_argument("--window", type=int, default=0, help="Test window index.")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if not os.path.exists(TRAJ_PATH):
        raise FileNotFoundError(
            f"{TRAJ_PATH} not found -- run eval_baselines.py to (re)generate it."
        )
    data = np.load(TRAJ_PATH)

    prefix = f"{args.case}_{args.method}"
    traj_key = f"{prefix}_trajectories"
    truth_key = f"{prefix}_truths"
    var_key = f"{prefix}_ensemble_variance"
    for key in (traj_key, truth_key, var_key):
        if key not in data.files:
            raise KeyError(
                f"'{key}' not in {TRAJ_PATH}. Available keys: {sorted(data.files)}"
            )

    mean = data[traj_key][args.window]      # (num_steps, 3)
    truth = data[truth_key][args.window]    # (num_steps, 3)
    spread = np.sqrt(data[var_key][args.window])  # (num_steps, 3)
    t = np.arange(mean.shape[0]) * args.dt

    dims = ["X", "Y", "Z"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for i, (ax, d) in enumerate(zip(axes, dims)):
        ax.fill_between(t, mean[:, i] - spread[:, i], mean[:, i] + spread[:, i],
                         color="tab:blue", alpha=0.25, label="ensemble spread (+/- 1 sd)")
        ax.plot(t, mean[:, i], color="tab:blue", lw=1.5, label="ensemble mean")
        ax.plot(t, truth[:, i], color="black", lw=1.0, ls="--", label="truth")
        ax.set_ylabel(d)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"{args.method} — {args.case.upper()} — test window {args.window}")
    axes[-1].set_xlabel("time")
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    output = args.output or os.path.join(
        OUT_DIR, f"{args.case}_{args.method}_window{args.window}_spread.png")
    fig.savefig(output, dpi=150)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
