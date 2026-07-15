#!/usr/bin/env python3
"""Generate a report for the S0/S1 DA-baseline bias ablation produced by
batch/run_ablation_bias_s0_s1.py: explains what each swept experiment is,
then plots mean RMSE vs the swept bias magnitude for every (scenario, axis),
for both the vanilla baselines and their Joint (state+parameter estimating)
counterparts, plus the Joint methods' own parameter-RMSE recovery.
"""
import os
import json
import argparse
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE, "experiments")
OUTPUT_DIR = os.path.join(BASE, "reports", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_INPUT = os.path.join(EXP_DIR, "ablation_bias_s0_s1.json")
DEFAULT_OUTPUT = os.path.join(OUTPUT_DIR, "ablation_bias_report.pdf")

BASE_METHODS = ["Weak-4DVar", "Strong-4DVar", "EnKF", "ETKF"]
JOINT_METHODS = [f"Joint-{m}" for m in BASE_METHODS]
# Order matches batch/run_ablation_bias_s0_s1.py::METHODS (vanilla immediately
# followed by its Joint counterpart).
METHODS = [m for pair in zip(BASE_METHODS, JOINT_METHODS) for m in pair]

# Fixed hue order, never cycled by rank/value (same categorical palette used
# throughout reports/, see reports/generate_training_report.py). A Joint
# method shares its base method's color/marker identity -- it is the same
# estimator, just also solving for parameters -- since vanilla and Joint now
# live on separate pages there's no need to distinguish them within a plot.
METHOD_COLOR = {
    "Weak-4DVar": "#2a78d6",    # blue
    "Strong-4DVar": "#eda100",  # yellow
    "EnKF": "#1baf7a",          # aqua
    "ETKF": "#e34948",          # red
}
METHOD_MARKER = {"Weak-4DVar": "o", "Strong-4DVar": "s", "EnKF": "^", "ETKF": "D"}
METHOD_CFG = {
    "Weak-4DVar": "opt_steps=150, lr=0.02",
    "Strong-4DVar": "max_iter=40, lr=0.1",
    "EnKF": "N_ensemble=30, inflation=2.0",
    "ETKF": "N_ensemble=30, inflation=2.0",
}


def is_joint(method):
    return method.startswith("Joint-")


def base_name(method):
    return method[len("Joint-"):] if is_joint(method) else method


def method_style(method):
    """(color, marker, linestyle, markerfacecolor) for a method, keeping
    Joint-X visually paired with X via shared color/marker."""
    base = base_name(method)
    color = METHOD_COLOR[base]
    marker = METHOD_MARKER[base]
    return color, marker, "-", color


SCENARIOS = [
    ("s0", "S0", "perfect operating point (no bias)"),
    ("s1", "S1", "biased operating point"),
]

# Fixed, unswept model-form mismatch present at every grid point (both S0/S1,
# all 3 axes): truth always couples the large-scale forcing W_L into dX
# nonlinearly (data/lorenz63.py::generate_long_trajectory ->
# _coupling(W_L, c1, exponent) = c1 * sign(W_L) * |W_L|**exponent), while every
# DA baseline here (vanilla and Joint alike) is built with a LINEAR forward
# model (batch/run_ablation_bias_s0_s1.py::build_methods ->
# evaluation/baselines.py::_apply_coupling). Neither S0 nor S1 config
# overrides coupling_exponent_truth, so this gap is identical across the whole
# report -- it is not one of the swept axes, it's a constant offset baked into
# the RMSE floor even at bias=0.
COUPLING_EXPONENT_TRUTH = 1.6  # Lorenz63Config.coupling_exponent_truth default
COUPLING_EXPONENT_DA = 1.0     # Lorenz63Config.coupling_exponent_da default; = batch script's COUPLING_EXPONENT_DA

AXES = [
    ("param_bias", "Parameter bias",
     "biases the sigma/rho/beta values DA solves with; forcing_state_bias held at the scenario's own baseline"),
    ("forcing_state_bias", "Forcing/state bias",
     "biases the corrupted forcing fed to DA; param_bias held at the scenario's own baseline"),
    ("joint_bias", "Joint bias",
     "param_bias and forcing_state_bias vary together from 0 at matching values"),
]


def load_data(path):
    with open(path) as f:
        return json.load(f)


def series_for(data, scenario, axis):
    """(sorted bias values, {method: [mean RMSE per value]}) for one
    (scenario, axis) sweep, gathered from run_key = f"{scenario}__{axis}__{value}" entries."""
    entries = [e for e in data.values() if e.get("scenario") == scenario and e.get("axis") == axis]
    entries.sort(key=lambda e: e["value"])
    values = [e["value"] for e in entries]
    per_method = {m: [e["results"].get(m, {}).get("mean", float("nan")) for e in entries] for m in METHODS}
    return values, per_method


def param_series_for(data, scenario, axis):
    """(sorted bias values, {joint_method: [mean param RMSE per value]}) for
    one (scenario, axis) sweep, averaging each Joint method's sigma/rho/beta/c1
    param_rmse into a single number per grid point."""
    entries = [e for e in data.values() if e.get("scenario") == scenario and e.get("axis") == axis]
    entries.sort(key=lambda e: e["value"])
    values = [e["value"] for e in entries]
    per_method = {}
    for m in JOINT_METHODS:
        series = []
        for e in entries:
            pr = e["results"].get(m, {}).get("param_rmse")
            series.append(float(np.mean(list(pr.values()))) if pr else float("nan"))
        per_method[m] = series
    return values, per_method


def scenario_meta(data, scenario):
    """(base_param_bias, base_forcing_state_bias, num_windows) reported by this
    scenario's entries, or ('?', '?', '?')."""
    for e in data.values():
        if e.get("scenario") == scenario:
            return e.get("base_param_bias", "?"), e.get("base_forcing_state_bias", "?"), e.get("num_windows", "?")
    return "?", "?", "?"


def _fmt_grid(values):
    """'0, 0.05, 0.10, 0.15, 0.20, 0.30' style grid string (0 bare, others 2dp)."""
    return ", ".join("0" if v == 0 else f"{v:.2f}" for v in values)


def _fmt_base(v):
    """'0.0' / '0.15' style anchor-value string."""
    return "?" if v == "?" else str(float(v))


def make_sweep_table_page(pdf, data):
    """One table per scenario spelling out exactly what param_bias/forcing_state_bias
    each of the 3 sweep axes uses -- mirrors the anchoring logic in
    batch/run_ablation_bias_s0_s1.py::main (sweeps list)."""
    fig, subplot_axes = plt.subplots(len(SCENARIOS), 1, figsize=(14, 9.5))
    fig.suptitle("Sweep Definitions — What Each Axis Actually Varies", fontsize=15, fontweight="bold")

    for ax, (scenario, s_label, s_desc) in zip(subplot_axes, SCENARIOS):
        base_p, base_f, _ = scenario_meta(data, scenario)
        values, _ = series_for(data, scenario, "param_bias")
        grid_str = _fmt_grid(values) if values else "n/a"
        base_p_str, base_f_str = _fmt_base(base_p), _fmt_base(base_f)

        joint_forcing_cell = f"same value ({grid_str})"
        if base_p not in (0, 0.0, "?") or base_f not in (0, 0.0, "?"):
            joint_forcing_cell = textwrap.fill(
                joint_forcing_cell + f" — note: joint sweep starts both at 0, "
                f"not anchored to {s_label}'s {base_p_str}/{base_f_str} base",
                width=48,
            )

        rows = [
            ["param_bias sweep", grid_str, f"fixed at {base_f_str}"],
            ["forcing_state_bias sweep", f"fixed at {base_p_str}", grid_str],
            ["joint_bias sweep", grid_str, joint_forcing_cell],
        ]

        ax.axis("off")
        ax.set_title(
            f"{s_label} sweeps (anchored at param_bias={base_p_str}, forcing_state_bias={base_f_str})",
            fontsize=11, fontweight="bold", loc="left", pad=10,
        )
        table = ax.table(
            cellText=rows,
            colLabels=["axis", "param_bias", "forcing_state_bias"],
            cellLoc="left", colLoc="left", loc="center",
            colWidths=[0.22, 0.28, 0.50],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#bbbbbb")
            if r == 0:
                cell.set_text_props(fontweight="bold")
                cell.set_facecolor("#eeeeee")
            cell.PAD = 0.02
        table.scale(1, 3.0 if scenario == "s1" else 2.4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close()
    print("  Page 2: Sweep definition tables (S0 / S1)")


def make_title_page(pdf, data, input_path):
    lines = [
        "DA Baseline Bias Ablation — S0/S1",
        "=" * 78,
        "",
        "What this ablation does",
        "-" * 78,
        "Sweeps how badly DA baselines (Weak-4DVar, Strong-4DVar, EnKF, ETKF) degrade",
        "when the model they solve with is biased relative to the true Lorenz63",
        "dynamics. Each baseline's Joint-* counterpart (Joint-Weak-4DVar, etc.) is",
        "swept alongside it -- same hyperparameters, but it also solves for",
        "sigma/rho/beta/c1 rather than using the (biased) DA-supplied values, so its",
        "state RMSE and its own parameter-recovery RMSE are both reported. Two",
        "independent bias mechanisms are swept, plus a joint sweep where both move",
        "together:",
        "",
    ]
    for axis, label, note in AXES:
        values, _ = series_for(data, SCENARIOS[0][0], axis)
        grid_str = ", ".join(f"{v:g}" for v in values) if values else "n/a"
        lines += [f"  * {label} ({axis})", f"      {note}", f"      grid: [{grid_str}]", ""]

    lines += [
        "Unlike the default S0/S1 baselines (evaluation/run.py::run_and_cache_baselines,",
        "which always feeds DA the TRUE forcing), this ablation feeds DA the CORRUPTED",
        "forcing, so forcing_state_bias actually perturbs what DA sees. Truth trajectories",
        "are unaffected by either bias — only the DA-facing sigma/rho/beta and corrupted",
        "forcing are biased — so varying bias does not change the physical windows.",
        "",
        "Coupling exponent: a 4th, UNSWEPT bias fixed at every grid point",
        "-" * 78,
        f"Truth couples W_L into dX nonlinearly, c1*sign(W_L)*|W_L|^{COUPLING_EXPONENT_TRUTH:g}",
        f"(coupling_exponent_truth); DA -- vanilla AND Joint alike -- always solves with",
        f"the LINEAR form c1*W_L (coupling_exponent_da={COUPLING_EXPONENT_DA:g}, never estimated, so Joint's",
        "sigma/rho/beta/c1 recovery can't fix it). Same gap for S0 and S1, all axes --",
        "so the RMSE floor even at bias=0 already reflects this fixed model-form error.",
        "(Unrelated to forcing_coupling 'linear'/'quartic' in conf/schema.py, used only",
        "to train the NN surrogates elsewhere in this repo, not this DA ablation.)",
        "",
        "Scenarios",
        "-" * 78,
    ]
    for scenario, s_label, s_desc in SCENARIOS:
        base_p, base_f, num_windows = scenario_meta(data, scenario)
        lines.append(
            f"  {s_label}: {s_desc}  "
            f"(param_bias={base_p}, forcing_state_bias={base_f}, num_windows={num_windows})"
        )

    lines += [
        "",
        "Methods",
        "-" * 78,
    ]
    for m in BASE_METHODS:
        lines.append(f"  {m:<18} {METHOD_CFG[m]}")
        lines.append(f"  Joint-{m:<12} {METHOD_CFG[m]}  (+ jointly estimates sigma/rho/beta/c1)")

    lines += [
        "",
        f"Source: {os.path.relpath(input_path, BASE)}",
        f"Runs found: {len(data)}",
    ]

    fig, ax = plt.subplots(figsize=(11, 10.5))
    ax.axis("off")
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
             fontsize=9.5, fontfamily="monospace", verticalalignment="top")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    print("  Page 1: Title + experiment explanation")


def make_state_rmse_grid_page(pdf, data, methods, title, page_num, page_desc):
    fig, axes = plt.subplots(len(SCENARIOS), len(AXES), figsize=(16, 9), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for row, (scenario, s_label, _) in enumerate(SCENARIOS):
        base_p, base_f, _ = scenario_meta(data, scenario)
        for col, (axis, axis_label, _) in enumerate(AXES):
            ax = axes[row][col]
            values, per_method = series_for(data, scenario, axis)
            if not values:
                ax.axis("off")
                ax.set_title(f"{s_label} — {axis_label} (no data)", fontsize=9)
                continue
            for method in methods:
                color, marker, ls, mfc = method_style(method)
                ax.plot(values, per_method[method], ls, marker=marker, color=color,
                         markerfacecolor=mfc, markeredgecolor=color,
                         lw=1.8, ms=5, alpha=0.9, label=method)
            ax.set_title(f"{s_label} (base p={base_p}, f={base_f}) — {axis_label}", fontsize=10)
            ax.set_xlabel(f"{axis} value", fontsize=9)
            ax.set_ylabel("Mean RMSE", fontsize=9)
            ax.grid(True, alpha=0.3, ls="--")
            ax.legend(fontsize=7.5, loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close()
    print(f"  Page {page_num}: {page_desc}")


def make_param_rmse_grid_page(pdf, data):
    fig, axes = plt.subplots(len(SCENARIOS), len(AXES), figsize=(16, 9), squeeze=False)
    fig.suptitle("Joint Methods — Mean Parameter RMSE vs Bias Magnitude  "
                 "(avg of sigma/rho/beta/c1)", fontsize=14, fontweight="bold")

    for row, (scenario, s_label, _) in enumerate(SCENARIOS):
        base_p, base_f, _ = scenario_meta(data, scenario)
        for col, (axis, axis_label, _) in enumerate(AXES):
            ax = axes[row][col]
            values, per_method = param_series_for(data, scenario, axis)
            if not values or all(np.all(np.isnan(v)) for v in per_method.values()):
                ax.axis("off")
                ax.set_title(f"{s_label} — {axis_label} (no data)", fontsize=9)
                continue
            for method in JOINT_METHODS:
                color, marker, ls, mfc = method_style(method)
                ax.plot(values, per_method[method], ls, marker=marker, color=color,
                         markerfacecolor=mfc, markeredgecolor=color,
                         lw=1.8, ms=5, alpha=0.9, label=method)
            ax.set_title(f"{s_label} (base p={base_p}, f={base_f}) — {axis_label}", fontsize=10)
            ax.set_xlabel(f"{axis} value", fontsize=9)
            ax.set_ylabel("Mean Param RMSE", fontsize=9)
            ax.grid(True, alpha=0.3, ls="--")
            ax.legend(fontsize=7.5, loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close()
    print("  Page 5: Joint param RMSE vs bias grid (2 scenarios x 3 axes, 4 Joint methods)")


def main():
    parser = argparse.ArgumentParser(description="Generate the S0/S1 DA bias-ablation report")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"No ablation results found at {args.input}")
        print("Run batch/run_ablation_bias_s0_s1.py first.")
        return

    data = load_data(args.input)
    print(f"Loaded {len(data)} runs from {args.input}")

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

    with PdfPages(args.output) as pdf:
        make_title_page(pdf, data, args.input)
        make_sweep_table_page(pdf, data)
        make_state_rmse_grid_page(
            pdf, data, BASE_METHODS,
            "Default Methods — Mean State RMSE vs Bias Magnitude",
            3, "Default state RMSE vs bias grid (2 scenarios x 3 axes, 4 methods)")
        make_state_rmse_grid_page(
            pdf, data, JOINT_METHODS,
            "Joint Methods — Mean State RMSE vs Bias Magnitude",
            4, "Joint state RMSE vs bias grid (2 scenarios x 3 axes, 4 methods)")
        make_param_rmse_grid_page(pdf, data)

    print(f"\nReport saved: {args.output}")


if __name__ == "__main__":
    main()
