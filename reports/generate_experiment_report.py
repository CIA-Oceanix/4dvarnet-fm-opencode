#!/usr/bin/env python3
"""
Generate synthesis PDF from DL experiment results.
Usage:
    python reports/generate_experiment_report.py
    python reports/generate_experiment_report.py --output my_report.pdf
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evaluation.metrics import rmse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE, "experiments")
OUTPUT_DIR = os.path.join(BASE, "reports", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

T_MAX = 3.0
DT = 0.01
NUM_STEPS = int(T_MAX / DT)
OBS_INTERVAL = 20
COMPONENTS = ["X", "Y", "Z"]

MODEL_COLORS = {
    "direct_unet": ["#1f77b4", "#4ecdc4"],
    "vanilla_cfm": ["#ff7f0e", "#ffbb78"],
    "tweedie": ["#2ca02c", "#98df8a"],
}
MODEL_LABELS = {
    "direct_unet": "DirectUNet",
    "vanilla_cfm": "VanillaCFM",
    "tweedie": "Tweedie",
}
BEST_BASELINE_COLOR = "#888888"


def get_model_type(exp):
    mt = exp.get("model_type")
    if mt is None:
        mt = exp.get("config", {}).get("model_type", "tweedie")
    return mt


def get_model_color(mt, shade=0):
    return MODEL_COLORS.get(mt, ["#888888", "#bbbbbb"])[shade % 2]


def load_experiments():
    experiments = []
    for d in sorted(os.listdir(EXP_DIR)):
        rpath = os.path.join(EXP_DIR, d, "results.json")
        if os.path.exists(rpath):
            with open(rpath) as f:
                result = json.load(f)
            if "fm_cs1" in result and "fm_cs2" in result:
                result["_dir"] = d
                experiments.append(result)
    return experiments


def sort_key(e):
    order = {"direct_unet": 0, "vanilla_cfm": 1, "tweedie": 2}
    return (order.get(get_model_type(e), 99), e.get("experiment_id", ""))


def load_baseline_reference():
    path = os.path.join(EXP_DIR, "baselines_dws50_inf1.2.json")
    if os.path.exists(path):
        with open(path) as f:
            bl = json.load(f)
        cs1_vals = []
        cs2_vals = []
        for meth in ["Weak-4DVar", "Strong-4DVar", "EnKF", "ETKF"]:
            c1 = bl.get("cs1", {}).get(meth, {}).get("mean")
            c2 = bl.get("cs2", {}).get(meth, {}).get("mean")
            if c1 is not None:
                cs1_vals.append(c1)
            if c2 is not None:
                cs2_vals.append(c2)
        best_bl_cs1 = min(cs1_vals) if cs1_vals else None
        best_bl_cs2 = min(cs2_vals) if cs2_vals else None
    else:
        best_bl_cs1 = None
        best_bl_cs2 = None

    best_fm = None
    b1_path = os.path.join(EXP_DIR, "B1_small_unet", "results.json")
    if os.path.exists(b1_path):
        with open(b1_path) as f:
            best_fm = json.load(f)
    return best_bl_cs1, best_bl_cs2, best_fm


def load_trajectories(exp_dir):
    trajs = {}
    for cs in ["cs1", "cs2"]:
        tpath = os.path.join(exp_dir, f"trajectories_{cs}.npz")
        if os.path.exists(tpath):
            data = np.load(tpath)
            trajs[cs] = {"trajectories": data["trajectories"], "truths": data["truths"]}
            data.close()
    return trajs


def find_best_worst(trajs, truths):
    N = trajs.shape[0]
    rmses = np.array([float(np.sqrt(np.mean((trajs[i] - truths[i]) ** 2))) for i in range(N)])
    best_idx = int(np.argmin(rmses))
    worst_idx = int(np.argmax(rmses))
    return best_idx, worst_idx, rmses[best_idx], rmses[worst_idx]


def draw_trajectory(ax, truth, recon, title, rmse_val, color, var_idx, var_name):
    time = np.linspace(0, T_MAX, len(truth))
    obs_mask = np.zeros(NUM_STEPS, dtype=bool)
    obs_mask[np.arange(OBS_INTERVAL, NUM_STEPS, OBS_INTERVAL)] = True
    ax.plot(time, truth[:, var_idx], "k-", lw=1.5, alpha=0.8, label="Truth")
    ax.plot(time, recon[:, var_idx], "--", color=color, lw=1.5, alpha=0.8, label="Recon")
    obs_t = time[obs_mask]
    ax.scatter(obs_t, truth[obs_mask, var_idx], c="gray", s=8, alpha=0.4, zorder=3)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel(var_name, fontsize=9)
    ax.set_title(f"{title}  |  RMSE={rmse_val:.3f}", fontsize=9)
    ax.grid(True, alpha=0.3, ls="--")
    ax.legend(fontsize=7, loc="upper right")


def make_trajectory_page(pdf, exp, exp_dir, model_type):
    trajs_data = load_trajectories(exp_dir)
    if not trajs_data:
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.text(0.5, 0.5, f"No trajectory data for {exp.get('experiment_id', '?')}",
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        fig.suptitle(f"{exp.get('experiment_id', '?')} — Trajectories (no data)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close()
        return

    exp_id = exp.get("experiment_id", "?")
    color = get_model_color(model_type, 0)

    for case in ["cs1", "cs2"]:
        if case not in trajs_data:
            continue
        trajs = trajs_data[case]["trajectories"]
        truths = trajs_data[case]["truths"]
        best_idx, worst_idx, best_rmse, worst_rmse = find_best_worst(trajs, truths)
        best_traj = trajs[best_idx]
        best_truth = truths[best_idx]
        worst_traj = trajs[worst_idx]
        worst_truth = truths[worst_idx]

        fig, axes = plt.subplots(2, 3, figsize=(14, 6.5))
        case_label = "CS1" if case == "cs1" else "CS2"
        fig.suptitle(f"{exp_id} ({MODEL_LABELS.get(model_type, model_type)}) — "
                     f"{case_label} best & worst reconstruction",
                     fontsize=13, fontweight="bold", y=1.01)

        for i, comp in enumerate(COMPONENTS):
            draw_trajectory(
                axes[0, i], best_truth, best_traj,
                f"Best (traj #{best_idx})", best_rmse, color, i, comp
            )
            draw_trajectory(
                axes[1, i], worst_truth, worst_traj,
                f"Worst (traj #{worst_idx})", worst_rmse, color, i, comp
            )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close()
        print(f"  Page: {exp_id} / {case_label}")


def main():
    parser = argparse.ArgumentParser(description="Generate DL experiment synthesis PDF")
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, "synthesis_experiments.pdf"))
    args = parser.parse_args()

    experiments = load_experiments()
    experiments.sort(key=sort_key)

    best_bl_cs1, best_bl_cs2, best_fm_ref = load_baseline_reference()

    if not experiments:
        print("No experiments with valid results.json found.")
        return

    print(f"Experiments found: {len(experiments)}")
    for e in experiments:
        mt = get_model_type(e)
        print(f"  {e.get('experiment_id', '?'):<20} model_type={mt}")

    print(f"\nGenerating PDF: {args.output}")

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

    with PdfPages(args.output) as pdf:
        # ── Page 1: Title + Experiment Config Table ──
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.axis("off")

        lines = [
            "4DVarNet-FM: DL Experiment Report",
            "=" * 110,
            "",
            "Experiment configurations:",
            "-" * 110,
        ]
        header = f"{'ID':<20} {'Model':<14} {'Channels':<18} {'Epochs':<10} {'Train Mix':<12} {'Params':<14}"
        lines.append(header)
        lines.append("-" * 110)
        for e in experiments:
            cfg = e.get("config", {})
            mt = get_model_type(e)
            ch = str(cfg.get("hidden_channels", "?"))
            ep = cfg.get("epochs_stage1", "?")
            if not cfg.get("skip_stage2", False):
                ep2 = cfg.get("epochs_stage2", 0)
                ep = f"{ep}+{ep2}" if ep2 else str(ep)
            tm = cfg.get("train_mix", "?")
            rand = "rand" if cfg.get("randomize_params", False) else "fixed"
            lines.append(
                f"{e.get('experiment_id', '?'):<20} {MODEL_LABELS.get(mt, mt):<14} "
                f"{ch:<18} {str(ep):<10} {tm:<12} {rand:<14}"
            )
        lines += [
            "-" * 110,
            "",
            "Test data:",
            "  200 trajectories per case study (seed=123 CS1, seed=124 CS2)",
            "  T_max = 3.0s  dt = 0.01  15 obs per window (obs_interval=20)",
        ]
        if best_fm_ref:
            lines += [
                "",
                f"Reference (B1_small_unet): CS1 μ={best_fm_ref['fm_cs1']['mean']:.4f}  "
                f"CS2 μ={best_fm_ref['fm_cs2']['mean']:.4f}  "
                f"Deg={best_fm_ref['fm_degradation']:.2f}x",
            ]

        ax.text(0.05, 0.98, "\n".join(lines), transform=ax.transAxes,
                fontsize=8.5, fontfamily="monospace", verticalalignment="top")
        fig.suptitle("DL Experiment Synthesis Report",
                     fontsize=15, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close()
        print("  Page 1: Title + config table")

        # ── Page 2: Summary Metrics Table ──
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.axis("off")

        lines = [
            "Summary of Results — Mean RMSE per Component",
            "=" * 130,
            f"{'ID':<20} {'Model':<12} "
            f"{'CS1 X':<9} {'CS1 Y':<9} {'CS1 Z':<9} {'CS1 μ':<9} "
            f"{'CS2 X':<9} {'CS2 Y':<9} {'CS2 Z':<9} {'CS2 μ':<9} {'Deg':<8}",
            "-" * 130,
        ]
        for e in experiments:
            c1 = e["fm_cs1"]
            c2 = e["fm_cs2"]
            mt = get_model_type(e)
            row = f"{e.get('experiment_id', '?'):<20} {MODEL_LABELS.get(mt, mt):<12}"
            for case_data in [c1, c2]:
                for comp in COMPONENTS:
                    val = case_data.get(comp, {}).get("mean", float("nan"))
                    row += f" {val:<9.4f}"
                row += f" {case_data.get('mean', float('nan')):<9.4f}"
            row += f" {e.get('fm_degradation', float('nan')):<8.2f}x"
            lines.append(row)

        lines += ["-" * 130, ""]
        lines.append(f"{'Best baseline reference':<20} "
                     f"{'':<12} "
                     f"{'':<9} {'':<9} {'':<9} "
                     f"{f'{best_bl_cs1:.4f}' if best_bl_cs1 else 'N/A':<9} "
                     f"{'':<9} {'':<9} {'':<9} "
                     f"{f'{best_bl_cs2:.4f}' if best_bl_cs2 else 'N/A':<9} "
                     f"{'':<8}")
        if best_bl_cs1 and best_bl_cs2:
            lines.append(f"  (best baseline CS1 μ={best_bl_cs1:.4f}, CS2 μ={best_bl_cs2:.4f})")
        lines += [
            "",
            "Degradation = CS2 μ / CS1 μ.   Lower = more robust.",
        ]

        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
                fontsize=8, fontfamily="monospace", verticalalignment="top")
        fig.suptitle("DL Experiments — Metrics",
                     fontsize=15, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close()
        print("  Page 2: Metrics table")

        # ── Page 3: Bar Charts ──
        fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
        fig.suptitle("DL Experiment Comparison — Mean RMSE",
                     fontsize=14, fontweight="bold")

        labels = [e.get("experiment_id", "?") for e in experiments]
        cs1_vals = [e["fm_cs1"]["mean"] for e in experiments]
        cs2_vals = [e["fm_cs2"]["mean"] for e in experiments]
        deg_vals = [e.get("fm_degradation", float("nan")) for e in experiments]

        model_types = [get_model_type(e) for e in experiments]
        bar_colors = [get_model_color(mt, 0) for mt in model_types]

        titles = ["CS1 Mean RMSE", "CS2 Mean RMSE", "Degradation (CS2 μ / CS1 μ)"]
        datasets = [cs1_vals, cs2_vals, deg_vals]
        ref_vals = [best_bl_cs1, best_bl_cs2, None]
        ylabels = ["Mean RMSE", "Mean RMSE", "Degradation Ratio"]

        for idx, (ax_, title, vals, ref, ylbl) in enumerate(
                zip(axes, titles, datasets, ref_vals, ylabels)):
            colors = bar_colors
            x = np.arange(len(vals))
            bars = ax_.bar(x, vals, color=colors, width=0.55, edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if np.isfinite(val):
                    va = "bottom"
                    if idx == 2:
                        ax_.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                                 f"{val:.2f}x", ha="center", va=va, fontsize=7, fontweight="bold")
                    else:
                        ax_.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                                 f"{val:.4f}", ha="center", va=va, fontsize=7)

            if ref is not None and np.isfinite(ref):
                ax_.axhline(ref, color=BEST_BASELINE_COLOR, ls="--", lw=1.5, alpha=0.7,
                            label=f"Best baseline ({ref:.4f})")
            if idx == 2:
                ax_.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.5, label="Ideal (1.0x)")

            if best_fm_ref:
                ref_val = best_fm_ref["fm_cs1"]["mean"] if idx == 0 else \
                          best_fm_ref["fm_cs2"]["mean"] if idx == 1 else \
                          best_fm_ref["fm_degradation"]
                if np.isfinite(ref_val):
                    ax_.axhline(ref_val, color="#1f77b4", ls="-.", lw=1.5, alpha=0.5,
                                label=f"B1_small_unet ({ref_val:.4f})")

            ax_.set_xticks(x)
            ax_.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
            ax_.set_ylabel(ylbl, fontsize=10)
            ax_.set_title(title, fontsize=11)
            ax_.legend(fontsize=7, loc="upper right")
            ax_.grid(True, axis="y", alpha=0.3, ls="--")
            clean = [v for v in vals if np.isfinite(v)]
            if clean:
                ax_.set_ylim(0, max(clean) * 1.35)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close()
        print("  Page 3: Bar charts")

        # ── Pages 4+: Trajectory Pages ──
        for e in experiments:
            exp_dir = os.path.join(EXP_DIR, e.get("_dir", e.get("experiment_id", "")))
            model_type = get_model_type(e)
            make_trajectory_page(pdf, e, exp_dir, model_type)

    print(f"\nDone: {args.output}")
    print(f"  Total experiments: {len(experiments)}")
    print(f"  Best baseline CS1 μ: {best_bl_cs1:.4f}" if best_bl_cs1 is not None else "")
    print(f"  Best baseline CS2 μ: {best_bl_cs2:.4f}" if best_bl_cs2 is not None else "")


if __name__ == "__main__":
    main()
