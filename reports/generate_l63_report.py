#!/usr/bin/env python3
"""Plotting/loading helpers for the L63 experiments notebook
(reports/l63_experiments_report.ipynb).

Each config/models/<eid>.yaml has a matching experiments/l63/<eid>/ directory
holding independently trained s0 and s1 models (experiments/l63/<eid>/s0/,
.../s1/), each evaluated only on its own matching held-out case -- unlike the
T*-experiments layout (4dvarnet-fm-opencode/reports/generate_training_report.py)
there is no cross-eval / trained_on split here, so every table and plot below
is keyed by a single "case" (s0 or s1), not a (trained_on, eval_case) pair.
"""
import os
import json
import csv
import numpy as np
import yaml
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE, "experiments", "l63")
CFG_DIR = os.path.join(BASE, "config", "models")

with open(os.path.join(BASE, "config", "lorenz63.yaml")) as _f:
    _DATA_CFG = yaml.safe_load(_f)["data"]
OBS_INTERVAL = _DATA_CFG["obs_interval"]
T_MAX = _DATA_CFG["T_max"]
STATE_NAMES = _DATA_CFG["state_names"]

CASES = ["s0", "s1"]
CASE_LABEL = {"s0": "S0 (in-distribution)", "s1": "S1 (biased / OOD)"}

def discover_exp_ids():
    """Every config/models/*.yaml with a matching experiments/l63/<eid>/ dir,
    sorted alphabetically for a deterministic (and therefore stable-colored)
    plot order."""
    ids = []
    for fname in sorted(os.listdir(CFG_DIR)):
        if not fname.endswith(".yaml"):
            continue
        eid = fname[: -len(".yaml")]
        if os.path.isdir(os.path.join(EXP_DIR, eid)):
            ids.append(eid)
    return ids

EXP_IDS = discover_exp_ids()

MODEL_LABEL = {
    "direct_unet": "DirectUNet",
    "vanilla_cfm": "VanillaCFM",
    "tweedie_cfm": "Tweedie",
    "joint_cfm": "JointCFM",
    "joint_direct_unet": "JointDirectUNet",
    "joint_tweedie_cfm": "JointTweedie",
}

# Same CVD-safe categorical palette convention used by the T*-experiments
# report (4dvarnet-fm-opencode/reports/generate_training_report.py) -- fixed
# hue order, never reassigned by rank/value.
CATEGORICAL_PALETTE = [
    "#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7",
    "#e34948", "#e87ba4", "#eb6834", "#7a5230", "#5f9ea0", "#9b30ff",
]
EXP_COLOR = {eid: CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)] for i, eid in enumerate(EXP_IDS)}

def display_label(eid):
    return eid

def load_config(eid):
    with open(os.path.join(CFG_DIR, f"{eid}.yaml")) as f:
        return yaml.safe_load(f)

def model_type_of(eid, configs):
    return configs[eid]["model"]["model_type"]

def case_dir(eid, case):
    return os.path.join(EXP_DIR, eid, case)

def load_results(eid):
    """experiments/l63/<eid>/results.json: {config, s0, s1, total_time_seconds}
    -- s0/s1 already carry the per-case metric dict (X/Y/Z/rmse/r2/crps/
    ensemble_spread/elapsed_seconds), unwrapped from the fm_<case> key that
    the per-case results.json uses."""
    path = os.path.join(EXP_DIR, eid, "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def load_param_rmse(eid, case):
    """Per-parameter RMSE (sigma/rho/beta/c1) for joint_* models, read from
    the per-case results.json (not mirrored into the top-level results.json)."""
    path = os.path.join(case_dir(eid, case), "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return d.get(f"param_rmse_{case}")

def fmt(val):
    return f"{val:.4f}" if isinstance(val, (int, float)) and np.isfinite(val) else "  n/a "

def latest_version_dir_with_metrics(eid, case, stage=1):
    stage_dir = os.path.join(case_dir(eid, case), "outputs", f"stage{stage}")
    if not os.path.isdir(stage_dir):
        return None
    versions = sorted(
        (v for v in os.listdir(stage_dir) if v.startswith("version_")),
        key=lambda v: int(v.split("_")[1]), reverse=True,
    )
    for v in versions:
        vdir = os.path.join(stage_dir, v)
        if os.path.exists(os.path.join(vdir, "metrics.csv")):
            return vdir
    return None

def load_loss_curve(eid, case, stage=1):
    vdir = latest_version_dir_with_metrics(eid, case, stage)
    if vdir is None:
        return None
    train, val = {}, {}
    with open(os.path.join(vdir, "metrics.csv")) as f:
        for row in csv.DictReader(f):
            ep = int(row["epoch"])
            if row.get("train_loss"):
                train[ep] = float(row["train_loss"])
            if row.get("val_loss"):
                val[ep] = float(row["val_loss"])
    epochs = sorted(set(train) | set(val))
    return (np.array(epochs),
            np.array([train.get(e, np.nan) for e in epochs]),
            np.array([val.get(e, np.nan) for e in epochs]))

def has_stage2(eid, case):
    return os.path.isdir(os.path.join(case_dir(eid, case), "outputs", "stage2"))

def load_trajectories(eid, case):
    """(trajectories, truths, members) for a model/case, `members` is None for
    deterministic models (direct_unet, joint_direct_unet) which save no
    per-ensemble-member array."""
    path = os.path.join(case_dir(eid, case), f"trajectories_{case}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return d["trajectories"], d["truths"], d["members"] if "members" in d.files else None

# --- DA baselines (Weak-4DVar, Strong-4DVar, EnKF, ETKF) ---------------------
BASELINE_METHODS = ["Weak-4DVar", "Strong-4DVar", "EnKF", "ETKF"]
BASELINE_COLOR = {
    "Weak-4DVar": "#888888", "Strong-4DVar": "#444444",
    "EnKF": "#1baf7a", "ETKF": "#e34948",
}
BASELINES_JSON_PATH = os.path.join(EXP_DIR, "baselines", "baselines_s0s1.json")
BASELINES_NPZ_PATH = os.path.join(EXP_DIR, "baselines", "baselines_trajectories_s0s1.npz")

def load_baselines_json():
    if not os.path.exists(BASELINES_JSON_PATH):
        return None
    with open(BASELINES_JSON_PATH) as f:
        return json.load(f)

def load_baselines_npz():
    if not os.path.exists(BASELINES_NPZ_PATH):
        return None
    return dict(np.load(BASELINES_NPZ_PATH))

def best_baseline_method(baselines_json, case):
    """(method, overall_rmse_mean) with the lowest overall test RMSE for
    `case`, using the already-aggregated rmse.mean baselines_s0s1.json
    reports per method -- no need to re-derive it from the raw trajectories."""
    if baselines_json is None or case not in baselines_json:
        return None
    scored = [(m, baselines_json[case].get(m, {}).get("rmse", {}).get("mean", float("nan")))
              for m in BASELINE_METHODS]
    scored = [(m, r) for m, r in scored if np.isfinite(r)]
    if not scored:
        return None
    return min(scored, key=lambda mr: mr[1])

def baseline_trajectory(baselines_npz, case, method, idx):
    key = f"{case}_{method.replace('-', '_')}_trajectories"
    if baselines_npz is None or key not in baselines_npz:
        return None
    return baselines_npz[key][idx]

# --- Reference windows, shared across every trajectory plot for a case ------
REF_MODEL_EID = "direct_unet"
SAMPLE_TAGS = ["best", "median", "worst"]
SAMPLE_LABEL = {"best": "Good", "median": "Median", "worst": "Worst"}

def pick_indices_grouped(rmse, k=1):
    order = np.argsort(rmse)
    n = len(order)
    lo = max(0, min(n - k, n // 2 - k // 2))
    return {"best": order[:k], "median": order[lo:lo + k], "worst": order[-k:][::-1]}

def reference_indices(case, k=1):
    """Window indices ranked by REF_MODEL_EID's own per-window RMSE, shared by
    every model's trajectory page for `case` so comparisons land on the same
    windows."""
    data = load_trajectories(REF_MODEL_EID, case)
    if data is None:
        return None
    traj, truth, _ = data
    rmse = np.sqrt(np.mean((traj - truth) ** 2, axis=(1, 2)))
    return pick_indices_grouped(rmse, k=k)

def obs_mask_for(n_steps):
    mask = np.zeros(n_steps, dtype=bool)
    mask[np.arange(0, n_steps, OBS_INTERVAL)] = True
    return mask

# --- Plots -------------------------------------------------------------------

def make_metrics_bar_charts(display, results_by_eid, baselines_json):
    """RMSE / R^2 / CRPS mean, one row per case, one bar per model + the best
    DA baseline for that case."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.suptitle("Mean RMSE / R² / CRPS by experiment", fontsize=14, fontweight="bold")
    for row, case in enumerate(CASES):
        present = [eid for eid in EXP_IDS if results_by_eid.get(eid) and case in results_by_eid[eid]]
        labels = [display(eid) for eid in present]
        colors = [EXP_COLOR[eid] for eid in present]
        best = best_baseline_method(baselines_json, case)
        if best is not None:
            labels = labels + [f"{best[0]}\n(best baseline)"]
            colors = colors + [BASELINE_COLOR[best[0]]]

        for col, (metric, ylabel) in enumerate([("rmse", "Mean RMSE"), ("r2", "Mean R²"), ("crps", "Mean CRPS")]):
            ax = axes[row, col]
            vals = [results_by_eid[eid][case][metric]["mean"] if metric != "r2" else results_by_eid[eid][case][metric]
                    for eid in present]
            if best is not None:
                if metric == "rmse":
                    vals = vals + [best[1]]
                elif metric == "r2":
                    vals = vals + [baselines_json[case][best[0]]["r2"]]
                else:
                    vals = vals + [baselines_json[case][best[0]]["crps"]["mean"]]
            x = np.arange(len(vals))
            heights = [v if np.isfinite(v) else 0.0 for v in vals]
            bars = ax.bar(x, heights, color=colors, width=0.6, edgecolor="black", linewidth=0.5, alpha=0.9)
            for bar, val in zip(bars, vals):
                va = "bottom" if bar.get_height() >= 0 else "top"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}" if np.isfinite(val) else "n/a",
                        ha="center", va=va, fontsize=6.5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=6.5, rotation=35, ha="right")
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(f"{CASE_LABEL[case]} — {ylabel}", fontsize=9.5)
            ax.grid(True, axis="y", alpha=0.3, ls="--")
            clean = [v for v in vals if np.isfinite(v)]
            if clean:
                lo = min(clean + [0.0])
                hi = max(clean + ([0.0] if metric == "r2" else [1.0]))
                pad = 0.08 * max(hi - lo, 1e-6)
                ax.set_ylim(lo - pad, hi + pad)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def make_loss_curves_page(display, curves_by_eid, curves_s2_by_eid, case):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f"{CASE_LABEL[case]} — Training & Validation Loss", fontsize=14, fontweight="bold")
    for ax, key, subtitle in zip(axes, ("train", "val"), ("Train Loss", "Validation Loss")):
        for eid in EXP_IDS:
            color = EXP_COLOR[eid]
            stage1_end = 0
            c = curves_by_eid.get(eid)
            if c is not None:
                epochs, train_arr, val_arr = c
                y = train_arr if key == "train" else val_arr
                ax.plot(epochs, y, "-", color=color, lw=1.6, alpha=0.9, label=display(eid))
                stage1_end = int(epochs[-1]) + 1 if len(epochs) else 0
            c2 = curves_s2_by_eid.get(eid)
            if c2 is not None:
                s2_epochs, s2_train, s2_val = c2
                y2 = s2_train if key == "train" else s2_val
                ax.plot(s2_epochs + stage1_end, y2, ":", color=color, lw=1.6, alpha=0.9,
                        label=f"{display(eid)} (stage 2)")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(f"{subtitle} (log scale)", fontsize=10)
        ax.set_title(subtitle, fontsize=11)
        ax.grid(True, alpha=0.3, ls="--", which="both")
        ax.legend(fontsize=6.5, loc="upper right", ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig

def make_overview_trajectory_page(display, case, tag, idx, baselines_npz, best_baseline, title_prefix=""):
    """All models overlaid on ONE example window -- a quick, at-a-glance
    comparison across every experiment, before drilling into per-model pages."""
    data = load_trajectories(REF_MODEL_EID, case)
    truth = data[1][idx]
    n_steps = truth.shape[0]
    time = np.linspace(0, T_MAX, n_steps)
    mask = obs_mask_for(n_steps)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle(f"{title_prefix}{CASE_LABEL[case]} — all experiments — {SAMPLE_LABEL[tag]} window #{idx}"
                 + (f" (ref.: {best_baseline[0]})" if best_baseline else ""),
                 fontsize=13, fontweight="bold")
    for ci, comp in enumerate(STATE_NAMES):
        ax = axes[ci]
        ax.plot(time, truth[:, ci], "k-", lw=1.8, alpha=0.9, label="Truth")
        if best_baseline is not None:
            btraj = baseline_trajectory(baselines_npz, case, best_baseline[0], idx)
            if btraj is not None:
                ax.plot(time, btraj[:, ci], "--", color=BASELINE_COLOR[best_baseline[0]], lw=1.5, alpha=0.85,
                        label=f"{best_baseline[0]} (best baseline)")
        for eid in EXP_IDS:
            d = load_trajectories(eid, case)
            if d is None:
                continue
            traj = d[0]
            ax.plot(time, traj[idx, :, ci], "-", color=EXP_COLOR[eid], lw=1.2, alpha=0.85, label=display(eid))
        ax.scatter(time[mask], truth[mask, ci], c="black", s=18, zorder=4, edgecolors="white", linewidths=0.8)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel(comp, fontsize=9)
        ax.grid(True, alpha=0.3, ls="--")
        if ci == 2:
            ax.legend(fontsize=6.5, loc="upper right", ncol=1)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig

def make_model_trajectory_page(display, eid, case, ref_indices, baselines_npz, best_baseline, title_prefix=""):
    """Best/median/worst reference windows (ranked by REF_MODEL_EID, shared
    across every model) for one experiment: Truth, best DA baseline, and this
    model's prediction -- with an ensemble-spread band (mean +/- 1 std across
    N_ensemble members) where the model stores one (i.e. not direct_unet)."""
    data = load_trajectories(eid, case)
    if data is None:
        return None
    traj, truth, members = data
    n_steps = truth.shape[1]
    time = np.linspace(0, T_MAX, n_steps)
    mask = obs_mask_for(n_steps)
    rows = [(tag, int(idx)) for tag in SAMPLE_TAGS for idx in ref_indices[tag]]

    fig, axes = plt.subplots(len(rows), 3, figsize=(14, 2.9 * len(rows)), squeeze=False)
    baseline_note = f" vs. best baseline ({best_baseline[0]})" if best_baseline else ""
    fig.suptitle(f"{title_prefix}{CASE_LABEL[case]} — {display(eid)}{baseline_note}",
                 fontsize=13, fontweight="bold", y=0.995)

    for row, (tag, idx) in enumerate(rows):
        btraj = baseline_trajectory(baselines_npz, case, best_baseline[0], idx) if best_baseline else None
        model_rmse = float(np.sqrt(np.mean((traj[idx] - truth[idx]) ** 2)))
        base_rmse = float(np.sqrt(np.mean((btraj - truth[idx]) ** 2))) if btraj is not None else None
        for ci, comp in enumerate(STATE_NAMES):
            ax = axes[row, ci]
            ax.plot(time, truth[idx, :, ci], "k-", lw=1.6, alpha=0.9, label="Truth")
            if btraj is not None:
                ax.plot(time, btraj[:, ci], "--", color=BASELINE_COLOR[best_baseline[0]], lw=1.4, alpha=0.8,
                        label=f"{best_baseline[0]} (best baseline)")
            if members is not None:
                spread = members[idx, :, ci, :].std(axis=-1)
                ax.fill_between(time, traj[idx, :, ci] - spread, traj[idx, :, ci] + spread,
                                 color=EXP_COLOR[eid], alpha=0.18, lw=0)
            ax.plot(time, traj[idx, :, ci], "-", color=EXP_COLOR[eid], lw=1.8, alpha=0.95, label=display(eid))
            ax.scatter(time[mask], truth[idx, mask, ci], c="black", s=18, zorder=4, edgecolors="white", linewidths=0.8)
            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_ylabel(comp, fontsize=9)
            ax.grid(True, alpha=0.3, ls="--")
            if row == 0 and ci == 2:
                ax.legend(fontsize=7, loc="upper right")
            if ci == 1:
                title = f"{SAMPLE_LABEL[tag]} #{idx} — RMSE: {display(eid)}={model_rmse:.3f}"
                if base_rmse is not None:
                    title += f" / {best_baseline[0]}={base_rmse:.3f}"
                ax.set_title(title, fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig
