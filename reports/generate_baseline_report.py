#!/usr/bin/env python3
"""
Generate a synthesis PDF comparing DA baselines (default vs. joint
parameter-estimation variants) on the S0/S1 benchmark.

S0: perfect model (DA coupling exponent matches truth, no bias).
S1: model mismatch (DA coupling exponent 1.0 vs. truth 1.6, param_bias=0.15,
    forcing_state_bias=0.1).

Default variants (Weak-4DVar, Strong-4DVar, EnKF, ETKF) solve with fixed,
nominal sigma/rho/beta/c1. Joint variants (Joint-Weak-4DVar, ...) additionally
estimate those parameters online, so this report puts every method side by
side against its own joint counterpart.

Usage:
    python reports/generate_baseline_report.py \\
        --json experiments/baselines_joint_dws50_s0s1_inf2.0_etkf_inf2.0.json \\
        --output reports/outputs/baseline_default_vs_joint_s0_s1.pdf
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(BASE, "experiments", "baselines_joint_dws50_s0s1_inf2.0_etkf_inf2.0.json")
DEFAULT_SAMPLES = os.path.join(BASE, "experiments", "baselines_joint_dws50_s0s1_inf2.0_etkf_inf2.0_samples.npz")
DEFAULT_OUTPUT = os.path.join(BASE, "reports", "outputs", "baseline_default_vs_joint_s0_s1.pdf")

METHODS = ["Weak-4DVar", "Strong-4DVar", "EnKF", "ETKF"]
VARIANTS = ["Default", "Joint"]
COMPONENTS = ["X", "Y", "Z"]
PARAM_COMPONENTS = ["sigma", "rho", "beta", "c1"]
CASES = ["s0", "s1"]
CASE_LABELS = {"s0": "S0", "s1": "S1"}

# Fixed hue per method, never cycled by rank/value (same categorical palette
# used throughout reports/, see generate_ablation_bias_report.py).
METHOD_COLOR = {
    "Weak-4DVar": "#2a78d6",    # blue
    "Strong-4DVar": "#eda100",  # yellow
    "EnKF": "#1baf7a",          # aqua
    "ETKF": "#e34948",          # red
}
VARIANT_HATCH = {"Default": "", "Joint": "//"}

T_MAX = 3.0
DT = 0.01
NUM_STEPS = int(T_MAX / DT)
OBS_INTERVAL = 20
DA_WINDOW_STEPS = 50
ENKF_INFLATION = 2.0
ETKF_INFLATION = 2.0

METHOD_PARAMS = {
    "Weak-4DVar": "opt_steps=150, lr=0.02, B_var=2.0, Q_var=0.05",
    "Strong-4DVar": "max_iter=40, lr=0.1, B_var=2.0",
    "EnKF": f"N_ensemble=30, inflation={ENKF_INFLATION}",
    "ETKF": f"N_ensemble=30, inflation={ETKF_INFLATION} (deterministic transform)",
}

CASE_META = {
    "s0": {
        "Truth coupling exponent": "1.6",
        "DA coupling exponent": "1.6 (perfect model)",
        "Param bias": "0.0",
        "Forcing-state bias": "0.0",
    },
    "s1": {
        "Truth coupling exponent": "1.6",
        "DA coupling exponent": "1.0 (mismatch)",
        "Param bias": "0.15",
        "Forcing-state bias": "0.1",
    },
}


def method_key(method, variant):
    return method if variant == "Default" else f"Joint-{method}"


def load_metrics(json_path):
    with open(json_path) as f:
        return json.load(f)


def state_mean(metrics, case, method, variant):
    entry = metrics.get(case, {}).get(method_key(method, variant), {}).get("state_rmse")
    return entry["mean"] if entry else float("nan")


def state_mean_std(metrics, case, method, variant):
    entry = metrics.get(case, {}).get(method_key(method, variant), {}).get("state_rmse")
    if not entry:
        return float("nan"), float("nan")
    return entry["mean"], entry.get("std", float("nan"))


def state_component(metrics, case, method, variant, comp):
    entry = metrics.get(case, {}).get(method_key(method, variant), {}).get("state_rmse")
    return entry[comp]["mean"] if entry else float("nan")


def state_component_std(metrics, case, method, variant, comp):
    entry = metrics.get(case, {}).get(method_key(method, variant), {}).get("state_rmse")
    if not entry:
        return float("nan"), float("nan")
    c = entry.get(comp, {})
    return c.get("mean", float("nan")), c.get("std", float("nan"))


def param_component(metrics, case, method, comp):
    entry = metrics.get(case, {}).get(method_key(method, "Joint"), {}).get("param_rmse")
    return entry[comp]["mean"] if entry else float("nan")


def param_component_std(metrics, case, method, comp):
    entry = metrics.get(case, {}).get(method_key(method, "Joint"), {}).get("param_rmse")
    if not entry:
        return float("nan"), float("nan")
    c = entry.get(comp, {})
    return c.get("mean", float("nan")), c.get("std", float("nan"))


def fmt_ms(mean, std=None):
    """'mean +/- std' for a table cell; falls back to plain mean if std is
    missing/non-finite (e.g. no std stored for this metric)."""
    if not (isinstance(mean, (int, float)) and np.isfinite(mean)):
        return "n/a"
    if isinstance(std, (int, float)) and np.isfinite(std):
        return f"{mean:.4f}±{std:.4f}"
    return f"{mean:.4f}"


def make_title_page(pdf, metrics, json_path):
    lines = [
        "4DVarNet-FM: DA Baselines — Default vs. Joint Parameter Estimation",
        "=" * 100,
        "",
        "Common parameters:",
        f"  T_max = {T_MAX}s    dt = {DT}    Steps = {NUM_STEPS}",
        f"  obs_interval = {OBS_INTERVAL} (15 obs / window, includes step 0)    R_var = 0.5    B_var = 2.0",
        f"  DA window steps (DWS) = {DA_WINDOW_STEPS}",
        "",
        "Method parameters (shared by Default and Joint variants):",
    ]
    for m in METHODS:
        lines.append(f"  {m:<14} {METHOD_PARAMS[m]}")
    lines += [
        "",
        "Default vs. Joint:",
        "  Default : solves with fixed, nominal sigma/rho/beta/c1.",
        "  Joint   : additionally estimates sigma/rho/beta/c1 online (log-space,",
        "            Gaussian prior P_var=1.0 around the DA-side nominal values,",
        "            re-estimated every DA window and carried forward).",
        "",
        "Case studies:",
        "-" * 100,
        f"{'Parameter':<28} {'S0':<34} {'S1':<34}",
        "-" * 100,
    ]
    for param, s0_val in CASE_META["s0"].items():
        s1_val = CASE_META["s1"].get(param, "")
        lines.append(f"  {param:<26} {s0_val:<34} {s1_val:<34}")
    lines += [
        "-" * 100,
        "",
        "Test data:",
        "  200 test windows per case, RandomParamLorenz63Dataset (per-window random",
        "  sigma/rho/beta +/-20%), make_s0_s1_trainval data setup.",
        "",
        f"Source: {os.path.relpath(json_path, BASE)}",
        f"Total run time: {metrics.get('total_time_seconds', float('nan')):.0f}s",
    ]

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(0.04, 0.98, "\n".join(lines), transform=ax.transAxes,
            fontsize=8.5, fontfamily="monospace", verticalalignment="top")
    fig.suptitle("Baseline Verification — Default vs. Joint", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close()
    print("  Page 1: Title + parameterization")


def make_summary_table_page(pdf, metrics):
    lines = [
        "Summary of Results — Mean State RMSE ± std, Default vs. Joint",
        "=" * 140,
        f"{'Method':<16} {'S0 Default':<18} {'S0 Joint':<18} {'S0 dJ%':<9}"
        f" {'S1 Default':<18} {'S1 Joint':<18} {'S1 dJ%':<9} {'Deg (Def)':<11} {'Deg (Joint)':<11}",
        "-" * 140,
    ]
    for m in METHODS:
        d_s0, d_s0_std = state_mean_std(metrics, "s0", m, "Default")
        j_s0, j_s0_std = state_mean_std(metrics, "s0", m, "Joint")
        d_s1, d_s1_std = state_mean_std(metrics, "s1", m, "Default")
        j_s1, j_s1_std = state_mean_std(metrics, "s1", m, "Joint")
        d_pct_s0 = (j_s0 - d_s0) / d_s0 * 100 if d_s0 else float("nan")
        d_pct_s1 = (j_s1 - d_s1) / d_s1 * 100 if d_s1 else float("nan")
        deg_def = d_s1 / d_s0 if d_s0 else float("nan")
        deg_joint = j_s1 / j_s0 if j_s0 else float("nan")
        lines.append(
            f"{m:<16} {fmt_ms(d_s0, d_s0_std):<18} {fmt_ms(j_s0, j_s0_std):<18} {d_pct_s0:>+7.1f}%"
            f" {fmt_ms(d_s1, d_s1_std):<18} {fmt_ms(j_s1, j_s1_std):<18} {d_pct_s1:>+7.1f}%"
            f" {f'{deg_def:.2f}x':<11} {f'{deg_joint:.2f}x':<11}"
        )
    lines += [
        "-" * 140,
        "",
        "dJ% = (Joint mean - Default mean) / Default mean.   Positive = joint estimation costs state accuracy.",
        "Deg = S1 mean / S0 mean (degradation under model mismatch).   Lower = more robust.",
        "",
    ]

    best_def_s0 = min(METHODS, key=lambda m: state_mean(metrics, "s0", m, "Default"))
    best_joint_s0 = min(METHODS, key=lambda m: state_mean(metrics, "s0", m, "Joint"))
    best_def_s1 = min(METHODS, key=lambda m: state_mean(metrics, "s1", m, "Default"))
    best_joint_s1 = min(METHODS, key=lambda m: state_mean(metrics, "s1", m, "Joint"))
    lines += [
        f"Best Default on S0: {best_def_s0} ({state_mean(metrics, 's0', best_def_s0, 'Default'):.4f})",
        f"Best Joint on S0:   {best_joint_s0} ({state_mean(metrics, 's0', best_joint_s0, 'Joint'):.4f})",
        f"Best Default on S1: {best_def_s1} ({state_mean(metrics, 's1', best_def_s1, 'Default'):.4f})",
        f"Best Joint on S1:   {best_joint_s1} ({state_mean(metrics, 's1', best_joint_s1, 'Joint'):.4f})",
    ]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.axis("off")
    ax.text(0.03, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=8.3, fontfamily="monospace", verticalalignment="top")
    fig.suptitle("Baseline Verification — Metrics (Default vs. Joint)", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close()
    print("  Page 2: Summary metrics table")


def _grouped_bars(ax, values_by_variant, title, ylabel, fmt="{:.4f}"):
    n = len(METHODS)
    width = 0.35
    x = np.arange(n)
    for i, variant in enumerate(VARIANTS):
        offset = (i - 0.5) * width
        vals = values_by_variant[variant]
        colors = [METHOD_COLOR[m] for m in METHODS]
        bars = ax.bar(x + offset, vals, width=width, color=colors, edgecolor="black",
                      linewidth=0.6, hatch=VARIANT_HATCH[variant], label=variant, alpha=0.9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt.format(val), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, fontsize=8, rotation=25, ha="right")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="y", alpha=0.3, ls="--")
    all_vals = values_by_variant["Default"] + values_by_variant["Joint"]
    finite = [v for v in all_vals if np.isfinite(v)]
    if finite:
        ax.set_ylim(0, max(finite) * 1.3)


def make_bar_chart_page(pdf, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    fig.suptitle("Baseline Comparison — Default vs. Joint", fontsize=14, fontweight="bold")

    s0_vals = {v: [state_mean(metrics, "s0", m, v) for m in METHODS] for v in VARIANTS}
    s1_vals = {v: [state_mean(metrics, "s1", m, v) for m in METHODS] for v in VARIANTS}
    deg_vals = {
        v: [state_mean(metrics, "s1", m, v) / state_mean(metrics, "s0", m, v)
            if state_mean(metrics, "s0", m, v) else float("nan") for m in METHODS]
        for v in VARIANTS
    }

    _grouped_bars(axes[0], s0_vals, "S0 Mean RMSE", "Mean RMSE")
    _grouped_bars(axes[1], s1_vals, "S1 Mean RMSE", "Mean RMSE")
    _grouped_bars(axes[2], deg_vals, "Robustness (S1/S0)", "Degradation", fmt="{:.2f}x")
    axes[2].axhline(1.0, color="gray", ls=":", lw=1, alpha=0.5)

    # Single legend distinguishing hatch (variant), color already keys method.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="white", edgecolor="black", hatch=VARIANT_HATCH[v], label=v) for v in VARIANTS]
    fig.legend(handles=handles, loc="upper right", fontsize=9, bbox_to_anchor=(0.99, 0.97))

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    pdf.savefig(fig)
    plt.close()
    print("  Page 3: Comparison bar charts")


def make_component_table_page(pdf, metrics):
    lines = ["Per-Component RMSE ± std — Default vs. Joint", "=" * 140]
    for case in CASES:
        lines += [
            f"--- {CASE_LABELS[case]} ---",
            f"{'Method':<16} {'Variant':<8} {'X':<18} {'Y':<18} {'Z':<18} {'Mean':<18}",
            "-" * 140,
        ]
        for m in METHODS:
            for variant in VARIANTS:
                vals = [state_component_std(metrics, case, m, variant, c) for c in COMPONENTS]
                mean_v, mean_std = state_mean_std(metrics, case, m, variant)
                lines.append(
                    f"{m:<16} {variant:<8} {fmt_ms(*vals[0]):<18} {fmt_ms(*vals[1]):<18} "
                    f"{fmt_ms(*vals[2]):<18} {fmt_ms(mean_v, mean_std):<18}"
                )
        lines.append("")

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.axis("off")
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes,
            fontsize=8, fontfamily="monospace", verticalalignment="top")
    fig.suptitle("Per-Component RMSE Breakdown", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close()
    print("  Page 4: Per-component RMSE table")


def make_param_rmse_page(pdf, metrics):
    lines = [
        "Parameter RMSE ± std — Joint Variants Only",
        "=" * 124,
    ]
    for case in CASES:
        lines += [
            f"--- {CASE_LABELS[case]} ---",
            f"{'Method':<16} {'sigma':<18} {'rho':<18} {'beta':<18} {'c1':<18}",
            "-" * 124,
        ]
        for m in METHODS:
            vals = [param_component_std(metrics, case, m, c) for c in PARAM_COMPONENTS]
            lines.append(
                f"{m:<16} {fmt_ms(*vals[0]):<18} {fmt_ms(*vals[1]):<18} "
                f"{fmt_ms(*vals[2]):<18} {fmt_ms(*vals[3]):<18}"
            )
        lines.append("")

    fig, axes = plt.subplots(2, 4, figsize=(15, 8), squeeze=False)
    fig.suptitle("Parameter RMSE Breakdown — Joint Variants (S0 vs. S1)", fontsize=14, fontweight="bold")

    for row, case in enumerate(CASES):
        for ci, comp in enumerate(PARAM_COMPONENTS):
            ax = axes[row, ci]
            vals = [param_component(metrics, case, m, comp) for m in METHODS]
            colors = [METHOD_COLOR[m] for m in METHODS]
            bars = ax.bar(range(len(METHODS)), vals, color=colors, width=0.55, edgecolor="black", linewidth=0.6, alpha=0.9)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}",
                        ha="center", va="bottom", fontsize=7)
            ax.set_xticks(range(len(METHODS)))
            ax.set_xticklabels(METHODS, fontsize=7, rotation=30, ha="right")
            ax.set_ylabel("RMSE", fontsize=9)
            ax.set_title(f"{CASE_LABELS[case]} — {comp}", fontsize=10)
            ax.grid(True, axis="y", alpha=0.3, ls="--")
            finite = [v for v in vals if np.isfinite(v)]
            if finite:
                ax.set_ylim(0, max(finite) * 1.35)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close()
    print("  Page 5: Parameter RMSE breakdown")

    # Also emit the table as its own page since a 2x4 bar grid leaves no
    # room for the raw numbers.
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.axis("off")
    ax2.text(0.05, 0.95, "\n".join(lines), transform=ax2.transAxes,
             fontsize=9, fontfamily="monospace", verticalalignment="top")
    fig2.suptitle("Parameter RMSE Table — Joint Variants", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig2)
    plt.close()
    print("  Page 5b: Parameter RMSE table")


SAMPLE_TAGS = ["best", "median", "worst"]
SAMPLE_LABEL = {"best": "Good", "median": "Median", "worst": "Worst"}
VARIANT_COLOR = {"Default": "#2a78d6", "Joint": "#e34948"}


def load_samples(samples_path):
    if not os.path.exists(samples_path):
        return None
    return dict(np.load(samples_path))


def has_sample(samples, case, method, variant, tag):
    key = method_key(method, variant).replace("-", "_")
    return f"{case}__{key}__{tag}_traj" in samples


def get_sample(samples, case, method, variant, tag):
    key = method_key(method, variant).replace("-", "_")
    prefix = f"{case}__{key}__{tag}"
    return samples[f"{prefix}_traj"], samples[f"{prefix}_truth"], samples[f"{prefix}_obs_mask"], float(samples[f"{prefix}_rmse"])


def make_trajectory_sample_page(pdf, samples, case, method):
    time_grid = np.linspace(0, T_MAX, NUM_STEPS)

    fig, axes = plt.subplots(3, 3, figsize=(14, 9.5))
    fig.suptitle(f"{CASE_LABELS[case]} — {method}: Default vs. Joint (good / median / worst reconstructions)",
                 fontsize=13, fontweight="bold", y=0.99)

    for row, tag in enumerate(SAMPLE_TAGS):
        variants_present = [v for v in VARIANTS if has_sample(samples, case, method, v, tag)]
        if not variants_present:
            for ci in range(3):
                axes[row, ci].axis("off")
            continue

        truth = None
        obs_mask = None
        rmse_by_variant = {}
        traj_by_variant = {}
        for v in variants_present:
            traj, tr, om, rmse_v = get_sample(samples, case, method, v, tag)
            traj_by_variant[v] = traj
            rmse_by_variant[v] = rmse_v
            truth, obs_mask = tr, om

        for ci, comp in enumerate(COMPONENTS):
            ax = axes[row, ci]
            ax.plot(time_grid, truth[:, ci], "-", color="black", lw=1.5, alpha=0.85, label="Truth")
            for v in variants_present:
                ax.plot(time_grid, traj_by_variant[v][:, ci], "-", color=VARIANT_COLOR[v],
                        lw=1.6, alpha=0.70, label=v)
            obs_t = time_grid[obs_mask.astype(bool)]
            ax.scatter(obs_t, truth[obs_mask.astype(bool), ci], c="gray", s=8, alpha=0.4, zorder=3)
            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_ylabel(comp, fontsize=9)
            rmse_str = " / ".join(f"{v}={rmse_by_variant[v]:.3f}" for v in variants_present)
            title = f"{SAMPLE_LABEL[tag]}" if ci != 1 else f"{SAMPLE_LABEL[tag]} — RMSE: {rmse_str}"
            ax.set_title(title, fontsize=9)
            ax.grid(True, alpha=0.3, ls="--")
            if row == 0 and ci == 2:
                ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close()
    print(f"  Page: Trajectory samples — {CASE_LABELS[case]} / {method}")


def main():
    parser = argparse.ArgumentParser(description="Generate baseline Default-vs-Joint synthesis PDF (S0/S1)")
    parser.add_argument("--json", default=DEFAULT_INPUT, help="Path to metrics JSON")
    parser.add_argument("--samples", default=DEFAULT_SAMPLES, help="Path to trajectory-samples NPZ")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"No baseline results found at {args.json}")
        print("Run batch/run_baselines_s0s1_full_local.py first.")
        return

    metrics = load_metrics(args.json)
    samples = load_samples(args.samples)
    if samples is None:
        print(f"No trajectory samples found at {args.samples} (run batch/run_baselines_s0s1_full_local.py to generate them)")
    print(f"Generating PDF: {args.output}")

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

    with PdfPages(args.output) as pdf:
        make_title_page(pdf, metrics, args.json)
        make_summary_table_page(pdf, metrics)
        make_bar_chart_page(pdf, metrics)
        make_component_table_page(pdf, metrics)
        make_param_rmse_page(pdf, metrics)
        if samples is not None:
            for case in CASES:
                for m in METHODS:
                    make_trajectory_sample_page(pdf, samples, case, m)

    print(f"\nDone: {args.output}")


if __name__ == "__main__":
    main()
