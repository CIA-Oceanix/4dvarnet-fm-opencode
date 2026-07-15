#!/usr/bin/env python3
"""Generate training synthesis PDF for all T* (S0/S1 randomized-bias, s0_s1 data setup) experiments."""
import os, sys, re, json, csv, argparse
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE, "experiments")
CFG_DIR = os.path.join(BASE, "config", "experiment")
OUTPUT_DIR = os.path.join(BASE, "reports", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COMPONENTS = ["X", "Y", "Z"]

def discover_exp_ids():
    """Find every config/experiment/T<n>_*.yaml, sorted T1,T2,...
    Tie-broken alphabetically by full id so ordering (and therefore color
    assignment) is deterministic regardless of filesystem listing order."""
    found = []
    for fname in os.listdir(CFG_DIR):
        m = re.match(r"^(T)(\d+)_.*\.yaml$", fname)
        if m:
            found.append((m.group(1), int(m.group(2)), fname[:-len(".yaml")]))
    found.sort(key=lambda t: (t[0], t[1], t[2]))
    return [eid for _, _, eid in found]

EXP_IDS = discover_exp_ids()
MODEL_LABEL = {
    "direct_unet": "DirectUNet", "vanilla_cfm": "VanillaCFM",
    "tweedie": "Tweedie", "joint_cfm": "JointCFM",
    "joint_direct_unet": "JointDirectUNet",
}

# Validated CVD-safe categorical palette (fixed hue order, never cycled by
# rank/value — see reports/README or the dataviz skill's palette reference).
# 8 slots covers today's E1/E2/F1/F2 x {cs1,cs2}; a 9th+ experiment repeats
# the cycle rather than inventing an unvalidated hue.
CATEGORICAL_PALETTE = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]
EXP_COLOR = {eid: CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)] for i, eid in enumerate(EXP_IDS)}
VARIANT_STYLE = {"default": "-", "small": "--", "rand": ":", "other": "-."}

def variant_of(eid):
    if "_rand" in eid:
        return "rand"
    if "_small" in eid:
        return "small"
    if "_default" in eid:
        return "default"
    return "other"

def fmt(val):
    return f"{val:.4f}" if isinstance(val, (int, float)) and np.isfinite(val) else "  n/a "

def fmt_ms(mean, std=None):
    """'mean +/- std' for a table cell; falls back to plain mean if std is
    missing/non-finite (e.g. no std stored for this metric, like the
    aggregate fm_* mean or the flat param_rmse_* scalars)."""
    if not (isinstance(mean, (int, float)) and np.isfinite(mean)):
        return "  n/a "
    if isinstance(std, (int, float)) and np.isfinite(std):
        return f"{mean:.4f}±{std:.4f}"
    return f"{mean:.4f}"

def load_config(eid):
    path = os.path.join(CFG_DIR, f"{eid}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)

def trained_on(eid, configs):
    """E/F/G configs are trained on a train_mix of CS1/CS2 corruption cases.
    T* configs have no train_mix field (they instead pick a data_setup)."""
    return configs.get(eid, {}).get("data", {}).get("train_mix", "?")

def trained_on_display(eid, configs):
    """Like trained_on, but falls back to data_setup (e.g. 's0_s1') for
    experiments with no train_mix field, so 'Trained On' columns aren't just '?'."""
    d = configs.get(eid, {}).get("data", {})
    return d.get("train_mix") or d.get("data_setup", "?")

# Short, suggestive model-family abbreviations used by variant_suffix() so plot
# legends/labels read as e.g. "T8 (v_cfm_s_t0)" instead of the raw
# "T8 (s0s1_small_tau0)" config-suffix dump.
MODEL_ABBR = {
    "direct_unet": "unet",
    "joint_direct_unet": "j_unet",
    "vanilla_cfm": "v_cfm",
    "joint_cfm": "j_cfm",
    "tweedie": "tweedie",
}

def variant_suffix(eid, configs):
    """Fallback descriptor built from model_type + size/tau0 flags, e.g.
    'T8_vanilla_cfm_s0s1_small_tau0' -> 'v_cfm_s_t0' (vanilla_cfm, small,
    train_tau_0_only). Small/tau0 are detected from the id itself (same
    substrings variant_of() keys off), so this stays in sync with the
    filenames rather than re-deriving them from nested config lookups."""
    mt = configs.get(eid, {}).get("model", {}).get("model_type", "")
    parts = [MODEL_ABBR.get(mt, mt or "?")]
    if "_small" in eid:
        parts.append("s")
    if "_tau0" in eid:
        parts.append("t0")
    return "_".join(parts)

def display_label(eid, configs):
    """Short plot label that disambiguates same-numbered variants, e.g. 'E1 (cs2)' or 'T8 (v_cfm_s_t0)'."""
    train_mix = trained_on(eid, configs)
    suffix = train_mix if train_mix != "?" else variant_suffix(eid, configs)
    return f"{eid.split('_')[0]} ({suffix})"

def load_results():
    data = {}
    for eid in EXP_IDS:
        rpath = os.path.join(EXP_DIR, eid, "results.json")
        if os.path.exists(rpath):
            with open(rpath) as f:
                data[eid] = json.load(f)
    return data

def latest_version_dir_with_metrics(eid):
    """Newest-to-oldest version dir that actually has a metrics.csv. Some runs'
    highest-numbered version is an eval-only rerun that only wrote tfevents
    (e.g. the T* experiments), so picking strictly the newest dir would silently
    drop loss curves that exist in an earlier version."""
    stage_dir = os.path.join(EXP_DIR, eid, "outputs", "stage1")
    if not os.path.isdir(stage_dir):
        return None
    versions = sorted(
        (v for v in os.listdir(stage_dir) if v.startswith("version_")),
        key=lambda v: int(v.split("_")[1]),
        reverse=True,
    )
    for v in versions:
        vdir = os.path.join(stage_dir, v)
        if os.path.exists(os.path.join(vdir, "metrics.csv")):
            return vdir
    return None

def load_loss_curve(eid):
    """Return (epochs, train_loss, val_loss) from the latest CSV logger run, deduped by epoch."""
    csv_path_dir = latest_version_dir_with_metrics(eid)
    if csv_path_dir is None:
        return None
    csv_path = os.path.join(csv_path_dir, "metrics.csv")
    train, val = {}, {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ep = int(row["epoch"])
            if row.get("train_loss"):
                train[ep] = float(row["train_loss"])
            if row.get("val_loss"):
                val[ep] = float(row["val_loss"])
    epochs = sorted(set(train) | set(val))
    train_arr = np.array([train.get(e, np.nan) for e in epochs])
    val_arr = np.array([val.get(e, np.nan) for e in epochs])
    return np.array(epochs), train_arr, val_arr

def hyperparameter_table(configs):
    lines = [
        "Experiment Hyperparameters",
        "=" * 144,
        f"{'ID':<32} {'Model':<16} {'Channels':<16} {'S1Ep':<6} {'S2Ep':<6} "
        f"{'TimeEmb':<8} {'N_outer':<8} {'SigPrior':<9} "
        f"{'TrainMix':<16} {'Rand':<6} {'ParamNoise':<10}",
        "-" * 144,
    ]
    for eid in EXP_IDS:
        cfg = configs[eid]
        m = cfg["model"]
        mt = m["model_type"]
        # tweedie's fields sit directly under `model:`; direct_unet/vanilla_cfm
        # nest theirs under `model.<model_type>:`. joint_cfm/joint_direct_unet
        # reuse their non-joint backbone's block (hidden_channels, N_outer, ...)
        # and add their own param-conditioning fields under `model.joint_*:`.
        if mt == "tweedie":
            mcfg = m
        elif mt == "joint_cfm":
            mcfg = {**m.get("vanilla_cfm", {}), **m.get("joint_cfm", {})}
        elif mt == "joint_direct_unet":
            mcfg = {**m.get("direct_unet", {}), **m.get("joint_direct_unet", {})}
        else:
            mcfg = m.get(mt, {})
        ch = str(mcfg.get("hidden_channels", "?"))
        s1 = cfg["training"]["stage1"]
        # train.py only ever executes stage 2 for model_type == "tweedie" (see
        # train.py's `if model_type == "tweedie" and epochs_s2 > 0`), so a
        # nonzero stage2.epochs in a direct_unet/vanilla_cfm config is inert —
        # report the *effective* stage2 epoch count, not the declared one.
        s2_epochs = cfg["training"]["stage2"]["epochs"] if mt == "tweedie" else 0
        d = cfg["data"]
        rand = d.get("randomize_params", False)
        lines.append(
            f"{eid:<32} {MODEL_LABEL.get(mt, mt):<16} {ch:<16} {s1['epochs']:<6} {s2_epochs:<6} "
            f"{mcfg.get('time_emb_dim', '-'):<8} "
            f"{mcfg.get('N_outer', '-'):<8} {mcfg.get('sigma_prior', '-'):<9} "
            f"{trained_on_display(eid, configs):<16} {'yes' if rand else 'no':<6} "
            f"{d.get('param_noise', '-') if rand else '-':<10}"
        )
    lines += [
        "-" * 144,
        "",
        "Shared settings (config/lorenz63_default.yaml):",
        "  dt=0.01, T_max=3.0s, obs_interval=20, R_var=0.5, B_var=2.0",
        "  num_windows=2000, spinup_steps=10000, batch_size=32, dropout=0.1",
        "  optimizer=Adam, gradient_weight=0.1 (grad-loss term)",
        "  Stage 2 only ever runs for Tweedie (S2Ep>0 above); DirectUNet/VanillaCFM are stage-1-only.",
        "",
        "T* (data_setup=s0_s1): train/val use randomized param_bias in [0, bias_max] and",
        "  forcing_state_bias=0 (RandomBiasLorenz63Dataset); eval on two held-out test sets:",
        "  S0: param_bias=0.0, forcing_state_bias=0.0 (in-distribution, no bias)",
        "  S1: param_bias=0.15, forcing_state_bias=0.1, fixed (out-of-distribution bias)",
        "",
        "DirectUNet: single UNet1D pass, obs -> state (MSE + gradient loss), no flow matching.",
        "VanillaCFM: conditional flow matching, LinearInterpolant(nu=1.0), Euler sampling",
        "            over N_outer=10 steps, x0 ~ N(0, sigma_prior).",
        "JointCFM: VanillaCFM backbone jointly conditioned on/predicting model parameters",
        "          (param_dim, param_loss_weight); train_tau_0_only restricts training to tau=0.",
        "JointDirectUNet: DirectUNet backbone with the same joint param-conditioning head",
        "                 (param_dim, param_loss_weight) as JointCFM; no tau (single pass).",
        "Tweedie: 2-stage solver -- stage1 fits a Gaussian mean estimator, stage2 fits a",
        "         non-Gaussian residual correction on top of the frozen stage1 estimator.",
    ]
    return lines

def case_scheme(configs, exp_data):
    """Determine the (label, key_suffix) pairs for the two eval cases reported
    in results.json, e.g. [("CS1","cs1"),("CS2","cs2")] for E/F/G experiments
    (result keys fm_cs1/fm_cs2) or [("S0","s0"),("S1","s1")] for T* experiments
    (result keys fm_s0/fm_s1, data_setup=s0_s1). Chosen by majority among the
    experiments that actually have results, so the report adapts to whichever
    family was trained without needing separate code paths."""
    s0s1_count = sum(1 for eid in exp_data
                      if configs.get(eid, {}).get("data", {}).get("data_setup") == "s0_s1")
    if s0s1_count > 0 and s0s1_count >= len(exp_data) - s0s1_count:
        return [("S0", "s0"), ("S1", "s1")]
    return [("CS1", "cs1"), ("CS2", "cs2")]

def status_table(exp_data):
    lines = ["Run Status", "=" * 80]
    for eid in EXP_IDS:
        has_results = eid in exp_data
        has_loss = latest_version_dir_with_metrics(eid) is not None
        status = "trained (results + loss log)" if has_results and has_loss else \
                 "loss log only" if has_loss else "NOT YET TRAINED"
        lines.append(f"  {eid:<36} {status}")
    return lines

LOSS_CURVE_GROUPS = [
    # DirectUNet architecture: default/small x {state-only, jointly param-estimating}.
    ["T1_direct_unet_s0s1", "T2_direct_unet_s0s1_small",
     "T3_joint_direct_unet_s0s1", "T4_joint_direct_unet_s0s1_small"],
    # VanillaCFM/JointCFM trained over the full tau in [0,1]: default/small x {state-only, joint}.
    ["T5_vanilla_cfm_s0s1", "T7_vanilla_cfm_s0s1_small",
     "T9_joint_cfm_s0s1", "T11_joint_cfm_s0s1_small"],
    # Same architectures, restricted to train_tau_0_only=True.
    ["T6_vanilla_cfm_s0s1_tau0", "T8_vanilla_cfm_s0s1_small_tau0",
     "T10_joint_cfm_s0s1_tau0", "T12_joint_cfm_s0s1_small_tau0"],
]

def loss_curve_groups():
    """Aggregate loss curves per LOSS_CURVE_GROUPS (one page per group).
    Ids not present in EXP_IDS are dropped from their group; ids that belong
    to none of the groups (e.g. a future, differently-named series) each get
    their own singleton group, appended after."""
    grouped_ids = {eid for g in LOSS_CURVE_GROUPS for eid in g}
    groups = [[eid for eid in g if eid in EXP_IDS] for g in LOSS_CURVE_GROUPS]
    groups = [g for g in groups if g]
    other = [eid for eid in EXP_IDS if eid not in grouped_ids]
    groups += [[eid] for eid in other]
    return groups

def make_loss_curves_page(pdf, curves, configs, eids, title):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, key, subtitle in zip(axes, ("train", "val"), ("Train Loss", "Validation Loss")):
        for eid in eids:
            if eid not in curves:
                continue
            epochs, train_arr, val_arr = curves[eid]
            y = train_arr if key == "train" else val_arr
            color = EXP_COLOR[eid]
            style = VARIANT_STYLE[variant_of(eid)]
            label = display_label(eid, configs)  # e.g. "E1 (cs2)"
            ax.plot(epochs, y, style, color=color, lw=1.8, alpha=0.9, label=label)
        # ax.axvline(100, color="black", ls="--", lw=1.2, alpha=0.6, label="Stage 1 / Stage 2")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(f"{subtitle} (log scale)", fontsize=10)
        ax.set_title(subtitle, fontsize=11)
        ax.grid(True, alpha=0.3, ls="--", which="both")
        ax.legend(fontsize=7.5, loc="upper right", ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close()
    print(f"  Page: Loss curves ({title})")

def metrics_table(exp_data, configs, scheme):
    label0, label1 = scheme[0][0], scheme[1][0]
    lines = [
        f"Per-Variable RMSE Summary ({label0} / {label1}) — X/Y/Z as mean ± std, Mean as plain mean",
        "=" * 126,
    ]
    for cs_label, case_id in scheme:
        cs_key = f"fm_{case_id}"
        lines += [
            f"--- {cs_label} ---",
            f"{'ID':<32} {'Model':<16} {'Trained On':<20} {'X':<16} {'Y':<16} {'Z':<16} {'Mean':<10}",
            "-" * 126,
        ]
        for eid in EXP_IDS:
            if eid not in exp_data or cs_key not in exp_data[eid]:
                continue
            r = exp_data[eid]
            cs = r[cs_key]
            mt = MODEL_LABEL.get(r.get("model_type", ""), "?")
            train_mix = trained_on_display(eid, configs)
            lines.append(
                f"{eid:<32} {mt:<16} {train_mix:<20} "
                f"{fmt_ms(cs['X']['mean'], cs['X'].get('std')):<16} "
                f"{fmt_ms(cs['Y']['mean'], cs['Y'].get('std')):<16} "
                f"{fmt_ms(cs['Z']['mean'], cs['Z'].get('std')):<16} {fmt(cs['mean']):<10}"
            )
        lines.append("")
    header = f"{'ID':<32} {'Model':<16} {'Trained On':<20} {label0+' mu':<9} {label1+' mu':<9} {'Deg':<8} {'Time(s)':<9}"
    lines += [
        "-" * len(header),
        f"Degradation ({label1} mean / {label0} mean) and training time:",
        "-" * len(header),
        header,
        "-" * len(header),
    ]
    key0, key1 = f"fm_{scheme[0][1]}", f"fm_{scheme[1][1]}"
    for eid in EXP_IDS:
        if eid not in exp_data:
            continue
        r = exp_data[eid]
        mt = MODEL_LABEL.get(r.get("model_type", ""), "?")
        train_mix = trained_on_display(eid, configs)
        c1 = r.get(key0, {}).get("mean", float("nan"))
        c2 = r.get(key1, {}).get("mean", float("nan"))
        deg = r.get("fm_degradation", float("nan"))
        t = r.get("total_time_seconds", 0)
        lines.append(f"{eid:<32} {mt:<16} {train_mix:<20} {fmt(c1):<9} {fmt(c2):<9} {deg:<7.2f}x {t:<9.0f}")
    return lines

PARAM_COMPONENTS = ["sigma", "rho", "beta", "c1"]

def param_rmse_table(exp_data, configs, scheme):
    """Per-parameter RMSE table for joint experiments (JointCFM, JointDirectUNet;
    e.g. T9/T10), which jointly predict model parameters (param_rmse_<case> in
    results.json)."""
    label0, label1 = scheme[0][0], scheme[1][0]
    lines = [
        f"Parameter RMSE Summary ({label0} / {label1}) — Joint Variants",
        "=" * 104,
    ]
    for cs_label, case_id in scheme:
        pr_key = f"param_rmse_{case_id}"
        lines += [
            f"--- {cs_label} ---",
            f"{'ID':<32} {'sigma':<10} {'rho':<10} {'beta':<10} {'c1':<10}",
            "-" * 104,
        ]
        for eid in EXP_IDS:
            if eid not in exp_data or pr_key not in exp_data[eid]:
                continue
            pr = exp_data[eid][pr_key]
            lines.append(
                f"{eid:<32} {fmt(pr.get('sigma')):<10} {fmt(pr.get('rho')):<10} "
                f"{fmt(pr.get('beta')):<10} {fmt(pr.get('c1')):<10}"
            )
        lines.append("")
    return lines

def family_short_label(label):
    """Drop the redundant '(default)' qualifier for the aggregated bar chart's
    tick labels, e.g. 'VanillaCFM (default, tau0)' -> 'VanillaCFM (tau0)',
    'DirectUNet (default)' -> 'DirectUNet'. 'small'/'tau0' qualifiers stay,
    since those actually distinguish the bar from its siblings."""
    return label.replace("default, ", "").replace(" (default)", "")

def _grouped_family_bars(ax, values_by_variant, title, ylabel, fmt_str="{:.3f}"):
    """Grouped bars, one cluster per FAMILIES entry, Default/Joint side by
    side within each cluster (hatch distinguishes variant, color keys the
    family) -- mirrors generate_baseline_report.py's Default-vs-Joint bars."""
    n = len(FAMILIES)
    width = 0.35
    x = np.arange(n)
    for i, variant in enumerate(VARIANTS):
        offset = (i - 0.5) * width
        vals = values_by_variant[variant]
        colors = [FAMILY_COLOR[label] for label in FAMILY_LABELS]
        heights = [v if np.isfinite(v) else 0.0 for v in vals]
        bars = ax.bar(x + offset, heights, width=width, color=colors, edgecolor="black",
                      linewidth=0.6, hatch=VARIANT_HATCH[variant], alpha=0.9)
        for bar, val in zip(bars, vals):
            label = fmt_str.format(val) if np.isfinite(val) else "n/a"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    label, ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(FAMILY_LABELS_SHORT, fontsize=7.5, rotation=25, ha="right")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, ls="--")
    all_vals = values_by_variant["Default"] + values_by_variant["Joint"]
    finite = [v for v in all_vals if np.isfinite(v)]
    if finite:
        ax.set_ylim(0, max(finite) * 1.3)

def make_family_bar_charts(fig, exp_data, configs, scheme):
    """Aggregated Default-vs-Joint view, one cluster per FAMILIES entry --
    mirrors generate_baseline_report.py's grouped bar chart."""
    label0, label1 = scheme[0][0], scheme[1][0]
    key0, key1 = scheme[0][1], scheme[1][1]

    s0_vals = {v: [family_state_mean(exp_data, f, key0, v) for f in FAMILIES] for v in VARIANTS}
    s1_vals = {v: [family_state_mean(exp_data, f, key1, v) for f in FAMILIES] for v in VARIANTS}
    deg_vals = {
        v: [family_state_mean(exp_data, f, key1, v) / family_state_mean(exp_data, f, key0, v)
            if family_state_mean(exp_data, f, key0, v) else float("nan") for f in FAMILIES]
        for v in VARIANTS
    }

    axes = fig.subplots(1, 3)
    fig.suptitle(f"{label0} / {label1} Mean RMSE & Robustness — Default vs. Joint", fontsize=14, fontweight="bold", y=0.98)

    _grouped_family_bars(axes[0], s0_vals, f"{label0} Mean RMSE", "Mean RMSE")
    _grouped_family_bars(axes[1], s1_vals, f"{label1} Mean RMSE", "Mean RMSE")
    _grouped_family_bars(axes[2], deg_vals, f"Degradation ({label1}/{label0})", "Ratio", fmt_str="{:.2f}x")
    axes[2].axhline(1.0, color="gray", ls=":", lw=1, alpha=0.6)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor="white", edgecolor="black", hatch=VARIANT_HATCH[v], label=v) for v in VARIANTS]
    fig.legend(handles=handles, loc="upper right", fontsize=9, bbox_to_anchor=(0.99, 0.96))

def make_cs_bar_charts(fig, exp_data, configs, scheme):
    """Original flat view: one bar per experiment (not aggregated by family)."""
    label0, label1 = scheme[0][0], scheme[1][0]
    key0, key1 = f"fm_{scheme[0][1]}", f"fm_{scheme[1][1]}"
    present = [eid for eid in EXP_IDS if eid in exp_data]
    display = [display_label(eid, configs) for eid in present]
    colors = [EXP_COLOR[eid] for eid in present]

    cs1_vals = [exp_data[eid].get(key0, {}).get("mean", float("nan")) for eid in present]
    cs2_vals = [exp_data[eid].get(key1, {}).get("mean", float("nan")) for eid in present]
    deg_vals = [exp_data[eid].get("fm_degradation", float("nan")) for eid in present]

    axes = fig.subplots(1, 3)
    fig.suptitle(f"{label0} / {label1} Mean RMSE & Robustness", fontsize=14, fontweight="bold", y=0.98)
    titles = [f"{label0} Mean RMSE", f"{label1} Mean RMSE", f"Degradation ({label1}/{label0})"]
    datasets = [cs1_vals, cs2_vals, deg_vals]

    for col, (ax, title, vals) in enumerate(zip(axes, titles, datasets)):
        x = np.arange(len(vals))
        bars = ax.bar(x, vals, color=colors, width=0.55, edgecolor="black", linewidth=0.5, alpha=0.9)
        for bar, val in zip(bars, vals):
            fmt_str = f"{val:.3f}" if col < 2 else f"{val:.2f}x"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt_str,
                    ha="center", va="bottom", fontsize=8)
        if col == 2:
            ax.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(display, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel("Mean RMSE" if col < 2 else "Ratio", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3, ls="--")
        clean = [v for v in vals if np.isfinite(v)]
        if clean:
            ax.set_ylim(0, max(clean) * 1.3)

def make_param_rmse_breakdown(pdf, exp_data, configs, scheme):
    """Bar charts of per-parameter RMSE (sigma, rho, beta, c1), one row per
    case (S0/S1), for the joint experiments (JointCFM, JointDirectUNet) that
    report param_rmse_*."""
    present = [eid for eid in EXP_IDS if eid in exp_data and f"param_rmse_{scheme[0][1]}" in exp_data[eid]]
    if not present:
        print("  Skipped parameter RMSE breakdown (no joint-variant param_rmse data)")
        return
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), squeeze=False)
    label0, label1 = scheme[0][0], scheme[1][0]
    fig.suptitle(f"Parameter RMSE Breakdown ({label0}, {label1}) — Joint Variants", fontsize=14, fontweight="bold")

    for row, (cs_label, case_id) in enumerate(scheme):
        pr_key = f"param_rmse_{case_id}"
        for ci, comp in enumerate(PARAM_COMPONENTS):
            ax = axes[row, ci]
            vals, labels, bar_colors = [], [], []
            for eid in present:
                r = exp_data[eid]
                if pr_key not in r:
                    continue
                vals.append(r[pr_key][comp])
                labels.append(display_label(eid, configs))
                bar_colors.append(EXP_COLOR[eid])
            x = np.arange(len(vals))
            bars = ax.bar(x, vals, color=bar_colors, width=0.55, edgecolor="black", linewidth=0.5, alpha=0.9)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}",
                        ha="center", va="bottom", fontsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
            ax.set_ylabel("RMSE", fontsize=9)
            ax.set_title(f"{cs_label} — {comp}", fontsize=10)
            ax.grid(True, axis="y", alpha=0.3, ls="--")
            if vals:
                ax.set_ylim(0, max(vals) * 1.35)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close()
    print("  Page: Parameter RMSE breakdown (Joint Variants)")

def make_component_breakdown(pdf, exp_data, configs, scheme):
    present = [eid for eid in EXP_IDS if eid in exp_data]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    label0, label1 = scheme[0][0], scheme[1][0]
    fig.suptitle(f"Per-Component RMSE Breakdown ({label0}, {label1})", fontsize=14, fontweight="bold")

    for row, (cs_label, case_id) in enumerate(scheme):
        cs_key = f"fm_{case_id}"
        for ci, comp in enumerate(COMPONENTS):
            ax = axes[row, ci]
            vals, labels, bar_colors = [], [], []
            for eid in present:
                r = exp_data[eid]
                if cs_key not in r:
                    continue
                vals.append(r[cs_key][comp]["mean"])
                labels.append(display_label(eid, configs))
                bar_colors.append(EXP_COLOR[eid])
            x = np.arange(len(vals))
            bars = ax.bar(x, vals, color=bar_colors, width=0.55, edgecolor="black", linewidth=0.5, alpha=0.9)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}",
                        ha="center", va="bottom", fontsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
            ax.set_ylabel("RMSE", fontsize=9)
            ax.set_title(f"{cs_label} — {comp}", fontsize=10)
            ax.grid(True, axis="y", alpha=0.3, ls="--")
            if vals:
                ax.set_ylim(0, max(vals) * 1.35)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close()
    print("  Page: Per-component RMSE breakdown")

def load_trajectories(eid, case):
    tpath = os.path.join(EXP_DIR, eid, f"trajectories_{case}.npz")
    if not os.path.exists(tpath):
        return None
    d = np.load(tpath)
    return d["trajectories"], d["truths"]

# Each family pairs a "Default" (state-only) experiment with its "Joint"
# (state + online param estimation) counterpart, matched by architecture,
# hidden-channel size, and (for VanillaCFM) train_tau_0_only setting.
FAMILIES = [
    {"label": "DirectUNet (default)", "default": "T1_direct_unet_s0s1", "joint": "T3_joint_direct_unet_s0s1"},
    {"label": "DirectUNet (small)", "default": "T2_direct_unet_s0s1_small", "joint": "T4_joint_direct_unet_s0s1_small"},
    {"label": "VanillaCFM (default)", "default": "T5_vanilla_cfm_s0s1", "joint": "T9_joint_cfm_s0s1"},
    {"label": "VanillaCFM (default, tau0)", "default": "T6_vanilla_cfm_s0s1_tau0", "joint": "T10_joint_cfm_s0s1_tau0"},
    {"label": "VanillaCFM (small)", "default": "T7_vanilla_cfm_s0s1_small", "joint": "T11_joint_cfm_s0s1_small"},
    {"label": "VanillaCFM (small, tau0)", "default": "T8_vanilla_cfm_s0s1_small_tau0", "joint": "T12_joint_cfm_s0s1_small_tau0"},
]
FAMILY_LABELS = [f["label"] for f in FAMILIES]
FAMILY_LABELS_SHORT = [family_short_label(label) for label in FAMILY_LABELS]
FAMILY_COLOR = {label: CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)] for i, label in enumerate(FAMILY_LABELS)}
VARIANTS = ["Default", "Joint"]
VARIANT_HATCH = {"Default": "", "Joint": "//"}
SAMPLE_TAGS = ["best", "median", "worst"]
SAMPLE_LABEL = {"best": "Good", "median": "Median", "worst": "Worst"}
TRAJ_VARIANT_STYLE = {"Default": "--", "Joint": ":"}

def family_state_mean(exp_data, family, case_id, variant):
    eid = family["default"] if variant == "Default" else family["joint"]
    entry = exp_data.get(eid, {}).get(f"fm_{case_id}")
    return entry["mean"] if entry else float("nan")

def pick_indices(rmse):
    args = np.argsort(rmse)
    return {"best": args[0], "median": args[len(args) // 2], "worst": args[-1]}

def make_family_trajectory_page(pdf, configs, family, cs_label, case):
    """3x3 grid (good/median/worst x X/Y/Z) comparing the Default and Joint
    members of `family` on the same held-out window, mirroring
    generate_baseline_report.py's make_trajectory_sample_page."""
    d_eid, j_eid = family["default"], family["joint"]
    d_data = load_trajectories(d_eid, case)
    j_data = load_trajectories(j_eid, case)
    if d_data is None and j_data is None:
        return False

    truth_ref = (d_data if d_data is not None else j_data)[1]
    time = np.linspace(0, 3.0, truth_ref.shape[1])  # T_max=3.0s (lorenz63_default)
    obs_mask = np.zeros(truth_ref.shape[1], dtype=bool)
    obs_mask[np.arange(20, truth_ref.shape[1], 20)] = True  # obs_interval=20

    variant_data = {}
    if d_data is not None:
        variant_data["Default"] = (d_eid, d_data[0])
    if j_data is not None:
        variant_data["Joint"] = (j_eid, j_data[0])

    rank_traj = next(iter(variant_data.values()))[1]
    rmse = np.sqrt(np.mean((rank_traj - truth_ref) ** 2, axis=(1, 2)))
    indices = pick_indices(rmse)

    fig, axes = plt.subplots(3, 3, figsize=(14, 9.5))
    fig.suptitle(f"{cs_label} — {family['label']}: Default vs. Joint (good / median / worst reconstructions)",
                 fontsize=13, fontweight="bold", y=0.99)

    for row, tag in enumerate(SAMPLE_TAGS):
        idx = indices[tag]
        rmse_str_parts = []
        for ci, comp in enumerate(COMPONENTS):
            ax = axes[row, ci]
            ax.plot(time, truth_ref[idx, :, ci], "k-", lw=1.5, alpha=0.85, label="Truth")
            for variant, (eid, traj) in variant_data.items():
                ax.plot(time, traj[idx, :, ci], TRAJ_VARIANT_STYLE[variant], color=EXP_COLOR[eid],
                        lw=1.6, alpha=0.9, label=display_label(eid, configs))
            ax.scatter(time[obs_mask], truth_ref[idx, obs_mask, ci], c="gray", s=8, alpha=0.4, zorder=3)
            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_ylabel(comp, fontsize=9)
            ax.grid(True, alpha=0.3, ls="--")
            if row == 0 and ci == 2:
                ax.legend(fontsize=7, loc="upper right")
            if ci == 1:
                for variant, (eid, traj) in variant_data.items():
                    r = float(np.sqrt(np.mean((traj[idx] - truth_ref[idx]) ** 2)))
                    rmse_str_parts.append(f"{variant}={r:.3f}")
                title = f"{SAMPLE_LABEL[tag]}" if not rmse_str_parts else \
                    f"{SAMPLE_LABEL[tag]} — RMSE: {' / '.join(rmse_str_parts)}"
                ax.set_title(title, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close()
    print(f"  Page: Trajectory samples — {cs_label} / {family['label']}")
    return True

def main():
    families = sorted({eid[0] for eid in EXP_IDS})
    families_label = "".join(f"{f}*" for f in families) if families else "T*"
    parser = argparse.ArgumentParser(description=f"Generate {families_label} training synthesis report")
    parser.add_argument("--output", default=os.path.join(
        OUTPUT_DIR, f"synthesis_training_{'_'.join(families) or 'T'}.pdf"))
    args = parser.parse_args()

    if not EXP_IDS:
        print(f"No T* experiment configs found in {CFG_DIR}")
        sys.exit(1)

    configs = {eid: load_config(eid) for eid in EXP_IDS}
    exp_data = load_results()
    curves = {}
    for eid in EXP_IDS:
        c = load_loss_curve(eid)
        if c is not None:
            curves[eid] = c

    print(f"Configs loaded: {len(configs)} / {len(EXP_IDS)}")
    print(f"Results loaded: {len(exp_data)} / {len(EXP_IDS)}")
    print(f"Loss curves loaded: {len(curves)} / {len(EXP_IDS)}")
    for eid in EXP_IDS:
        r = "results" if eid in exp_data else "no-results"
        l = "loss-log" if eid in curves else "no-loss-log"
        print(f"  {eid}: {r}, {l}")

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    scheme = case_scheme(configs, exp_data)
    print(f"Case scheme: {scheme[0][0]} / {scheme[1][0]}")

    with PdfPages(args.output) as pdf:
        # Page 1: Title + hyperparameters + status
        fig1, ax1 = plt.subplots(figsize=(12, 9))
        ax1.axis("off")
        lines = hyperparameter_table(configs) + [""] + status_table(exp_data)
        ax1.text(0.02, 0.98, "\n".join(lines), transform=ax1.transAxes,
                 fontsize=7.3, fontfamily="monospace", verticalalignment="top")
        fig1.suptitle(f"4DVarNet-FM: Training Experiments ({len(EXP_IDS)} runs, {families_label})", fontsize=15, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig1)
        plt.close()
        print("  Page 1: Title + hyperparameters + run status")

        # Page 2+: Loss curves, one page per LOSS_CURVE_GROUPS entry (DirectUNet;
        # VanillaCFM/JointCFM full-tau; VanillaCFM/JointCFM tau0-only)
        if curves:
            for group in loss_curve_groups():
                if not any(eid in curves for eid in group):
                    continue
                title = "Training & Validation Loss — " + " / ".join(group)
                make_loss_curves_page(pdf, curves, configs, group, title)
        else:
            print("  Skipped loss curves (no data)")

        # Page 3: Metrics table
        fig3, ax3 = plt.subplots(figsize=(11, 8))
        ax3.axis("off")
        ax3.text(0.03, 0.97, "\n".join(metrics_table(exp_data, configs, scheme)), transform=ax3.transAxes,
                 fontsize=8.5, fontfamily="monospace", verticalalignment="top")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig3)
        plt.close()
        print("  Page 3: Metrics table")

        # Page 3b: Parameter RMSE table (joint experiments only)
        if any(f"param_rmse_{scheme[0][1]}" in exp_data.get(eid, {}) for eid in EXP_IDS):
            fig3b, ax3b = plt.subplots(figsize=(11, 6))
            ax3b.axis("off")
            ax3b.text(0.03, 0.97, "\n".join(param_rmse_table(exp_data, configs, scheme)),
                      transform=ax3b.transAxes, fontsize=8.5, fontfamily="monospace", verticalalignment="top")
            fig3b.suptitle(f"Parameter RMSE Summary ({scheme[0][0]} / {scheme[1][0]}) — Joint Variants", fontsize=15, fontweight="bold")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig3b)
            plt.close()
            print("  Page 3b: Parameter RMSE table (Joint Variants)")

        # Page 4: CS1/CS2 bar charts (flat, one bar per experiment)
        if exp_data:
            fig4 = plt.figure(figsize=(13, 5))
            make_cs_bar_charts(fig4, exp_data, configs, scheme)
            plt.tight_layout(rect=[0, 0, 1, 0.90])
            pdf.savefig(fig4)
            plt.close()
            print(f"  Page 4: {scheme[0][0]}/{scheme[1][0]} bar charts")

            # Page 4b: Same metrics, aggregated Default vs Joint per family
            fig4b = plt.figure(figsize=(13, 5))
            make_family_bar_charts(fig4b, exp_data, configs, scheme)
            plt.tight_layout(rect=[0, 0, 1, 0.90])
            pdf.savefig(fig4b)
            plt.close()
            print(f"  Page 4b: {scheme[0][0]}/{scheme[1][0]} bar charts (Default vs Joint)")

            # Page 5: Per-component breakdown
            make_component_breakdown(pdf, exp_data, configs, scheme)

            # Page 5b: Parameter RMSE breakdown (Joint Variants)
            make_param_rmse_breakdown(pdf, exp_data, configs, scheme)

            # Page 6+: Trajectory samples (good/median/worst, Default vs Joint) per family per case
            for cs_label, case in scheme:
                for family in FAMILIES:
                    make_family_trajectory_page(pdf, configs, family, cs_label, case)

    print(f"\nReport saved: {args.output}")

if __name__ == "__main__":
    main()