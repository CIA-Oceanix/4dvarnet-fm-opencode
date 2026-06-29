#!/usr/bin/env python3
"""
Generate a synthesis PDF from baseline evaluation results.

Usage:
    python reports/generate_baseline_report.py \\
        --json experiments/baselines_dws50_inf1.2.json \\
        --trajs experiments/baselines_trajectories_dws50_inf1.2.npz \\
        --output reports/outputs/synthesis_dws50_inf12.pdf
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

np.random.seed(0)

METHODS = ["Weak-4DVar", "Strong-4DVar", "EnKF"]
METHOD_COLORS = {
    "Weak-4DVar": "#ff7f0e",
    "Strong-4DVar": "#2ca02c",
    "EnKF": "#d62728",
}
CASES = ["cs1", "cs2"]
CASE_LABELS = {"cs1": "CS1", "cs2": "CS2"}
COMPONENTS = ["X", "Y", "Z"]
T_MAX = 3.0
DT = 0.01
NUM_STEPS = int(T_MAX / DT)
OBS_INTERVAL = 20

METHOD_PARAMS = {
    "Weak-4DVar": {"opt_steps": 150, "lr": 0.02, "da_window_steps": 50},
    "Strong-4DVar": {"max_iter": 40, "lr": 0.1, "da_window_steps": 50},
    "EnKF": {"N_ensemble": 30, "inflation": 1.2},
}

CS_PARAMS = {
    "cs1": {
        "Forcing": "Noise-free (true W_L)",
        "Parameter bias": "0%",
        "Forcing-state bias": "0.0",
        "Coupling": "Linear",
    },
    "cs2": {
        "Forcing": "Corrupted (OU process)",
        "Parameter bias": "15%",
        "Forcing-state bias": "0.15",
        "Coupling": "Quartic",
    },
}

def load_results(json_path, trajs_path):
    with open(json_path) as f:
        metrics = json.load(f)

    trajs_data = {}
    if os.path.exists(trajs_path):
        trajs = np.load(trajs_path)
        for key in trajs.files:
            trajs_data[key] = trajs[key]
        trajs.close()
    else:
        # fallback: try individual files
        base = os.path.dirname(trajs_path)
        dws_inf = os.path.basename(trajs_path).replace("baselines_trajectories", "baselines_trajs").replace(".npz", "")
        for case in CASES:
            for meth in METHODS:
                key = f"{case}_{meth.replace('-', '_').replace(' ', '_')}"
                fname = f"{dws_inf}_{key}.npz"
                fpath = os.path.join(base, fname)
                if os.path.exists(fpath):
                    data = np.load(fpath)
                    for k in data.files:
                        trajs_data[f"{key}_{k}"] = data[k]
                    data.close()

    return metrics, trajs_data


def find_best_worst(trajs_data, case, method):
    key_traj = f"{case}_{method.replace('-', '_').replace(' ', '_')}_trajectories"
    key_truth = f"{case}_{method.replace('-', '_').replace(' ', '_')}_truths"
    if key_traj not in trajs_data or key_truth not in trajs_data:
        return None, None, None, None
    trajs = trajs_data[key_traj]
    truths = trajs_data[key_truth]
    rmse_per = np.sqrt(np.mean((trajs - truths) ** 2, axis=(1, 2)))
    best_idx = int(np.argmin(rmse_per))
    worst_idx = int(np.argmax(rmse_per))
    return best_idx, worst_idx, trajs, truths


def draw_trajectory(ax, truth, recon, obs_mask, title, rmse_val, color, var_idx, var_name):
    time = np.linspace(0, T_MAX, len(truth))
    ax.plot(time, truth[:, var_idx], "k-", lw=1.5, alpha=0.8, label="Truth")
    ax.plot(time, recon[:, var_idx], "--", color=color, lw=1.5, alpha=0.8, label="Recon")
    if obs_mask is not None:
        obs_t = time[obs_mask]
        ax.scatter(obs_t, truth[obs_mask, var_idx], c="gray", s=8, alpha=0.4, zorder=3)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel(var_name, fontsize=9)
    ax.set_title(f"{title}  |  RMSE={rmse_val:.3f}", fontsize=9)
    ax.grid(True, alpha=0.3, ls="--")
    ax.legend(fontsize=7, loc="upper right")


def make_trajectory_page(pdf, trajs_data, case, method):
    color = METHOD_COLORS[method]
    obs_mask = np.zeros(NUM_STEPS, dtype=bool)
    obs_mask[np.arange(OBS_INTERVAL, NUM_STEPS, OBS_INTERVAL)] = True

    bw = find_best_worst(trajs_data, case, method)
    if bw[0] is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, f"No trajectory data for {CASE_LABELS[case]} / {method}",
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        pdf.savefig(fig)
        plt.close()
        return

    best_idx, worst_idx, trajs, truths = bw
    best_traj = trajs[best_idx]
    best_truth = truths[best_idx]
    worst_traj = trajs[worst_idx]
    worst_truth = truths[worst_idx]
    best_rmse = np.sqrt(np.mean((best_traj - best_truth) ** 2))
    worst_rmse = np.sqrt(np.mean((worst_traj - worst_truth) ** 2))

    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5))
    fig.suptitle(f"{CASE_LABELS[case]} — {method}  (best & worst reconstruction)",
                 fontsize=13, fontweight="bold", y=1.01)

    for i, comp in enumerate(COMPONENTS):
        draw_trajectory(
            axes[0, i], best_truth, best_traj, obs_mask,
            f"Best (traj #{best_idx})", best_rmse, color, i, comp
        )
        draw_trajectory(
            axes[1, i], worst_truth, worst_traj, obs_mask,
            f"Worst (traj #{worst_idx})", worst_rmse, color, i, comp
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close()
    print(f"  Page: {CASE_LABELS[case]} / {method}")


def main():
    parser = argparse.ArgumentParser(description="Generate baseline synthesis PDF")
    parser.add_argument("--json", required=True, help="Path to metrics JSON")
    parser.add_argument("--trajs", required=True, help="Path to trajectory NPZ")
    parser.add_argument("--output", default="reports/outputs/synthesis_dws50_inf12.pdf")
    args = parser.parse_args()

    metrics, trajs_data = load_results(args.json, args.trajs)
    config = metrics.get("config", {})

    print(f"Generating PDF: {args.output}")

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

    with PdfPages(args.output) as pdf:
        # ── Page 1: Title + Parameterization ──
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.axis("off")

        inf_val = config.get("enkf_inflation", 1.2)
        dws_val = config.get("da_window_steps", 50)

        lines = [
            "4DVarNet-FM: Baseline Verification Report",
            "=" * 100,
            "",
            "Common parameters:",
            f"  T_max = {T_MAX}s    dt = {DT}    Steps = {NUM_STEPS}",
            f"  obs_interval = {OBS_INTERVAL}    R_var = 0.5    B_var = 2.0",
            "",
            "DA Baseline parameters:",
            f"  DA window steps (DWS) = {dws_val}",
            f"  Weak-4DVar : opt_steps=150, lr=0.02",
            f"  Strong-4DVar: max_iter=40, lr=0.1",
            f"  EnKF       : N_ensemble=30, inflation={inf_val}",
            "",
            "Case studies:",
            "-" * 100,
            f"{'Parameter':<28} {'CS1':<36} {'CS2':<36}",
            "-" * 100,
        ]
        for param, cs1_val in CS_PARAMS["cs1"].items():
            cs2_val = CS_PARAMS["cs2"].get(param, "")
            lines.append(f"  {param:<26} {cs1_val:<36} {cs2_val:<36}")
        lines += [
            "-" * 100,
            "",
            "Test data:",
            "  200 trajectories per case study, generated with seed=123 (CS1) / 124 (CS2)",
            "  Forcing corrupted by Ornstein-Uhlenbeck noise (tau_eta=5.0, sigma_eta=0.707)",
        ]

        ax.text(0.05, 0.98, "\n".join(lines), transform=ax.transAxes,
                fontsize=8.5, fontfamily="monospace", verticalalignment="top")
        fig.suptitle("Baseline Verification — Synthesis Report",
                     fontsize=15, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close()
        print("  Page 1: Title + parameterization")

        # ── Page 2: Summary metrics table ──
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.axis("off")

        lines = [
            "Summary of Results — Mean RMSE",
            "=" * 110,
            f"{'Method':<20} {'CS1 X':<10} {'CS1 Y':<10} {'CS1 Z':<10} {'CS1 μ':<10}"
            f" {'CS2 X':<10} {'CS2 Y':<10} {'CS2 Z':<10} {'CS2 μ':<10} {'Deg':<8}",
            "-" * 110,
        ]

        for method in METHODS:
            row = f"{method:<20}"
            means_cs1 = []
            means_cs2 = []
            for case in CASES:
                cm = metrics.get(case, {}).get(method, {})
                for comp in COMPONENTS:
                    val = cm.get(comp, {}).get("mean", float("nan"))
                    row += f" {val:<10.4f}"
                    if case == "cs1":
                        means_cs1.append(val)
                    else:
                        means_cs2.append(val)
                mu = cm.get("mean", float("nan"))
                row += f" {mu:<10.4f}"
            deg = np.mean(means_cs2) / (np.mean(means_cs1) + 1e-10)
            row += f" {deg:<8.2f}x"
            lines.append(row)

        lines += ["-" * 110]
        # find best per column
        for col_idx, col_name in enumerate(["CS1 μ", "CS2 μ"]):
            vals = []
            for method in METHODS:
                cm = metrics.get("cs1" if "CS1" in col_name else "cs2", {}).get(method, {})
                vals.append(cm.get("mean", float("nan")))
            best_m = METHODS[int(np.nanargmin(vals))]
            lines.append(f"  Best in {col_name}: {best_m} ({min(vals):.4f})")

        lines += [
            "",
            f"Degradation = CS2 μ / CS1 μ.   Lower = more robust.",
            "",
            "Config file: " + args.json,
        ]

        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
                fontsize=9, fontfamily="monospace", verticalalignment="top")
        fig.suptitle("Baseline Verification — Metrics",
                     fontsize=15, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close()
        print("  Page 2: Metrics table")

        # ── Pages 3-8: Trajectories ──
        for case in CASES:
            for method in METHODS:
                make_trajectory_page(pdf, trajs_data, case, method)

    print(f"\nDone: {args.output}")


if __name__ == "__main__":
    main()
