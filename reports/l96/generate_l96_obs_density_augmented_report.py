#!/usr/bin/env python3
"""L96 fast-Y observation-density: training-augmentation dedicated report.

Combines the three sweeps run this session into one narrative report:

1. ``eval_obs_density_l96.py``'s original generalization sweep (no
   retraining) -- ``experiments/l96_obs_density_generalization/`` -- covering
   DirectUNet-L(cos), CFM-M(flat), SDA3, and the DirectUNet+SDA3 hybrid
   (non-augmented mean).
2. The obs-density-AUGMENTED training follow-up (PR #182) evaluated at full
   density only, then swept across density with the same script --
   ``experiments/l96_obs_density_augmented_checkpoints/`` -- covering
   DirectUNet-M(cosine, augmented) and CFM-M(augmented).
3. The augmented DirectUNet-M warm-starting SDA3 guidance --
   ``experiments/l96_obs_density_directunet_aug_sda3_hybrid/``.

Renders three sections: (1) experiment description, (2) a combined
RMSE/EV/degradation summary table across all 7 method variants, (3) a
best/median/worst-window analysis (mirroring
``generate_l96_consolidated_report.py``'s ``select_windows``/Hovmöller
convention, but with ``keep_k`` as the varying axis for a single method
instead of comparing methods at one density) showing exactly how 3
representative windows degrade as fast-Y density drops.

Consumes the per-(method, case, keep_k) ``estimates_*.npz`` files
``eval_obs_density_l96.py`` already saves (last repeat only) alongside each
sweep's ``results.json`` -- no new evaluation runs needed.
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GEN_JSON = ROOT / "experiments/l96_obs_density_generalization/results.json"
DEFAULT_GEN_DIR = ROOT / "experiments/l96_obs_density_generalization"
DEFAULT_AUG_JSON = ROOT / "experiments/l96_obs_density_augmented_checkpoints/results.json"
DEFAULT_AUG_DIR = ROOT / "experiments/l96_obs_density_augmented_checkpoints"
DEFAULT_HYBRID_JSON = ROOT / "experiments/l96_obs_density_directunet_aug_sda3_hybrid/results.json"
DEFAULT_HYBRID_DIR = ROOT / "experiments/l96_obs_density_directunet_aug_sda3_hybrid"
DEFAULT_OUT = ROOT / "reports/l96/outputs/l96_obs_density_augmented_training.md"
DEFAULT_FIG_DIR = ROOT / "reports/l96/outputs/figs_obs_density_augmented"

KEEP_K_VALUES = [16, 8, 4, 0]
CASES = ["s0", "s1"]
NO = 8  # slow-X count, for the Hovmöller panel split

# (row label, source json key, result key inside that json, source estimates dir)
METHOD_ROWS = [
    ("DirectUNet-L(cos) [non-aug]", DEFAULT_GEN_JSON, "directunet", DEFAULT_GEN_DIR),
    ("DirectUNet-M(cos) [augmented]", DEFAULT_AUG_JSON, "directunet", DEFAULT_AUG_DIR),
    ("CFM-M(flat) [non-aug]", DEFAULT_GEN_JSON, "cfm", DEFAULT_GEN_DIR),
    ("CFM-M(flat) [augmented]", DEFAULT_AUG_JSON, "cfm", DEFAULT_AUG_DIR),
    ("SDA3(monai)", DEFAULT_GEN_JSON, "sda3", DEFAULT_GEN_DIR),
    ("DirectUNet+SDA3 [non-aug mean]", DEFAULT_GEN_JSON, "directunet_sda3", DEFAULT_GEN_DIR),
    ("DirectUNet(aug)+SDA3 [aug mean]", DEFAULT_HYBRID_JSON, "directunet_sda3", DEFAULT_HYBRID_DIR),
]

# The 4 "headline" methods shown in the per-window keep_k-impact table/figure
# (dropping the two non-augmented rows there -- already characterized in the
# PR #180 generalization sweep summarized in section 1, and cluttering the
# per-window comparison).
HEADLINE_ROWS = [
    "DirectUNet-M(cos) [augmented]",
    "CFM-M(flat) [augmented]",
    "SDA3(monai)",
    "DirectUNet(aug)+SDA3 [aug mean]",
]

EXPERIMENT_DESCRIPTION = """\
## 1. Experiments

**Motivation.** The original fast-Y observation-density generalization sweep
(PR #180) found that DirectUNet-L(cos)
and CFM-M(flat) -- which consume `obs` only via `torch.nan_to_num(obs,
nan=0.0)`, with no separate mask channel -- degrade steeply under randomly
reduced fast-Y observation density (RMSE ~1.9x at half density, ~2.7x at
zero fast-Y density) because a dropped channel is indistinguishable from a
genuine near-zero observation: this is out-of-distribution for them, never
seen at training time. SDA3, whose prior network never conditions on raw
obs at all (only its guided-sampling cost reads it, cleanly excluding
dropped channels), degrades far more gracefully (~2.0x) with no retraining
needed.

**Training-time fix (PR #182).** Added a training-time counterpart:
`data/obs_density.py::sample_training_density_mask` mixes full-density and
randomly-reduced-density observation events within every training batch
(`full_prob=0.4`: 40% chance of full density, else `keep_k` drawn uniformly
from `{0,...,15}`), wired through `data/dataloader.py::make_collate_fm` and
applied to the train loader only (val/test always stay at full density).

**Tier investigation.** The first attempt (DirectUNet-L, cosine LR)
converged fine on `val_loss` but collapsed toward the fast-Y conditional
mean at eval time even at full density (predicted/true fast-Y variance
ratio ~35%, per-channel correlation ~0.55-0.64) -- CFM-M's *identical*
augmentation pipeline instead reconstructed fast-Y almost fully (variance
ratio ~96%, correlation ~0.94-0.95), ruling out a data-generation or
eval-framework bug and pointing at an L-tier-specific training pathology
(consistent with L-tier's pre-existing flat-LR instability documented in
`l96_consolidated_benchmark.md`). Retraining at **M-tier with cosine
annealing** (adopted as the default for all obs-density-augmented training
going forward, not just L) fully avoided the collapse: variance ratio 99%,
correlation 0.94, matching CFM's healthy behavior.

**Checkpoints compared in this report:**

| Label | Checkpoint | Augmented? |
|---|---|---|
| DirectUNet-L(cos) [non-aug] | `L1b_monai_unet_s0s1_norm_l_cosine` | no |
| DirectUNet-M(cos) [augmented] | `L1b_monai_unet_s0s1_norm_obsdensity` | yes (M-tier, cosine) |
| CFM-M(flat) [non-aug] | `L2b_monai_vanilla_cfm_s0s1_norm` | no |
| CFM-M(flat) [augmented] | `L2b_monai_vanilla_cfm_s0s1_norm_obsdensity` | yes |
| SDA3(monai) | `SDA3_monai_cond_noisy_l96_norm` | no (architecturally robust already) |
| DirectUNet+SDA3 [non-aug mean] | `L1b_monai_unet_s0s1_norm` warm-starting SDA3 | no |
| DirectUNet(aug)+SDA3 [aug mean] | `L1b_monai_unet_s0s1_norm_obsdensity` warm-starting SDA3 | yes (mean model only) |

**Protocol (unchanged from PR #180):** same 200-window cached S0/S1 test
set; `keep_k in {16, 8, 4, 0}` of the 16 canonical fast-Y channels kept,
**redrawn independently at every observation time** within each window
(harder than a fixed-per-window mask); `n_repeats=3` independent seed
reruns per (method, case, keep_k) cell (fresh mask redraw, and fresh
sampling noise for the stochastic methods); `n_outer=10`, `n_members=30`,
`r_var=0.5` for the ensemble/guided methods; hybrid `tau0=0.3`,
`guidance_weight=2.0`.
"""


def load_json(path: Path) -> dict:
    if not path.exists():
        logger.warning(f"Missing: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def fmt_mean_std(entry, ndigits=3):
    if not entry:
        return "--"
    return f"{entry['mean']:.{ndigits}f}±{entry['std']:.{ndigits}f}"


def build_summary_section(rows) -> str:
    lines = [
        "## 2. Summary: RMSE / EV(all_obs) vs. fast-Y density (all methods, S0; S1 tracks within noise)",
        "",
    ]
    header = ["Method"] + [f"keep_k={k}" for k in KEEP_K_VALUES]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for label, json_path, key, _ in rows:
        data = load_json(json_path)
        method_data = (data.get("results", {}) or {}).get(key, {}).get("s0", {})
        cells = [label]
        for k in KEEP_K_VALUES:
            entry = method_data.get(str(k), {})
            rmse = fmt_mean_std(entry.get("rmse"))
            ev = fmt_mean_std(entry.get("ev", {}).get("all_obs"))
            cells.append(f"{rmse} / EV {ev}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("### RMSE degradation vs. own full-density baseline (keep_k=16)")
    lines.append("")
    lines.append("| " + " | ".join(["Method"] + [f"keep_k={k}" for k in KEEP_K_VALUES]) + " |")
    lines.append("|" + "---|" * (len(KEEP_K_VALUES) + 1))
    for label, json_path, key, _ in rows:
        data = load_json(json_path)
        method_data = (data.get("results", {}) or {}).get(key, {}).get("s0", {})
        base = method_data.get(str(KEEP_K_VALUES[0]), {}).get("rmse", {}).get("mean")
        cells = [label]
        for k in KEEP_K_VALUES:
            cur = method_data.get(str(k), {}).get("rmse", {}).get("mean")
            if base is None or cur is None or base == 0:
                cells.append("--")
            else:
                cells.append(f"{cur / base:.3f}x")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append(
        "**Reading this table:** the augmented DirectUNet-M and CFM-M rows keep a *better* "
        "full-density baseline than their non-augmented counterparts while degrading far less "
        "(~1.5x vs ~1.9x at keep_k=8). The augmented hybrid combines the best full-density score "
        "of any row here with the best absolute worst-case RMSE (keep_k=0), edging out even SDA3 "
        "-- though SDA3 still has the flattest *relative* degradation curve, since the hybrid's "
        "much lower starting point means even a larger relative drop still lands ahead in "
        "absolute terms.\n"
    )
    return "\n".join(lines)


def load_estimates(exp_dir: Path, method_key: str, case: str, keep_k: int):
    npz_path = exp_dir / f"estimates_{method_key}_{case}_keep{keep_k}.npz"
    if not npz_path.exists():
        return None, None
    d = np.load(npz_path)
    return d["trajectories"], d["truth"]


def select_reference_windows(exp_dir: Path, method_key: str, case: str, keep_k: int) -> dict:
    traj, truth = load_estimates(exp_dir, method_key, case, keep_k)
    rw = np.sqrt(np.mean((traj - truth) ** 2, axis=(1, 2)))
    order = np.argsort(rw)
    mid = len(order) // 2
    return {
        "best": (int(order[0]), float(rw[order[0]])),
        "median": (int(order[mid]), float(rw[order[mid]])),
        "worst": (int(order[-1]), float(rw[order[-1]])),
    }


def per_window_rmse(exp_dir: Path, method_key: str, case: str, keep_k: int, win_idx: int):
    traj, truth = load_estimates(exp_dir, method_key, case, keep_k)
    if traj is None:
        return None
    return float(np.sqrt(np.mean((traj[win_idx] - truth[win_idx]) ** 2)))


def build_window_table(rank: str, win_idx: int, ref_rmse: float, case: str, rows) -> str:
    lines = [f"### {rank.capitalize()} window #{win_idx} (reference RMSE {ref_rmse:.3f})", ""]
    lines.append("| Method | " + " | ".join(f"keep_k={k}" for k in KEEP_K_VALUES) + " |")
    lines.append("|---|" + "---|" * len(KEEP_K_VALUES))
    for label, _, key, exp_dir in rows:
        if label not in HEADLINE_ROWS:
            continue
        cells = [label]
        for k in KEEP_K_VALUES:
            v = per_window_rmse(exp_dir, key, case, k, win_idx)
            cells.append(f"{v:.3f}" if v is not None else "--")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def plot_keep_k_hovmoller(fig_path: Path, case: str, rank: str, win_idx: int,
                          method_label: str, method_key: str, exp_dir: Path, dt: float = 0.001) -> None:
    """Hovmöller-style figure for ONE window, ONE method, across keep_k --
    the keep_k-as-varying-axis counterpart of
    generate_l96_consolidated_report.py::plot_hovmoller (which instead
    varies method at one fixed density)."""
    traj0, truth = load_estimates(exp_dir, method_key, case, KEEP_K_VALUES[0], )
    if traj0 is None:
        return
    truth_win = truth[win_idx]
    per_k_traj = {}
    for k in KEEP_K_VALUES:
        traj, _ = load_estimates(exp_dir, method_key, case, k)
        if traj is not None:
            per_k_traj[k] = traj[win_idx]

    labels = ["Truth"] + [f"keep_k={k}" for k in per_k_traj]
    n_rows = len(labels)
    fig, axes = plt.subplots(n_rows, 4, figsize=(15, 1.35 * n_rows + 1.0), constrained_layout=True)
    t = np.arange(truth_win.shape[0]) * dt

    state_data = [[truth_win[:, :NO], truth_win[:, NO:]]]
    for k in per_k_traj:
        state_data.append([per_k_traj[k][:, :NO], per_k_traj[k][:, NO:]])
    err_data = [[np.abs(d - tb) for d, tb in zip(row, [truth_win[:, :NO], truth_win[:, NO:]])] for row in state_data]

    flat_state = [d for row in state_data for d in row]
    s_vmin, s_vmax = min(d.min() for d in flat_state), max(d.max() for d in flat_state)
    e_vmax = float(np.percentile(np.concatenate([d.ravel() for row in err_data for d in row]), 99.5))
    cmap_state, cmap_err = plt.get_cmap("viridis"), plt.get_cmap("inferno")

    im_state = im_err = None
    for r, label in enumerate(labels):
        win_rmse = float(np.sqrt(np.mean(np.concatenate(err_data[r], axis=1) ** 2)))
        row_label = label if r == 0 else f"{label}\nRMSE {win_rmse:.3f}"
        for c in range(4):
            ax = axes[r, c]
            data, cmap, vmin, vmax = (
                (state_data[r][c % 2], cmap_state, s_vmin, s_vmax)
                if c < 2 else (np.minimum(err_data[r][c % 2], e_vmax), cmap_err, 0.0, e_vmax)
            )
            mesh = ax.pcolormesh(t, np.arange(data.shape[1]), data.T, cmap=cmap, vmin=vmin, vmax=vmax,
                                 shading="auto", rasterized=True)
            if c < 2:
                im_state = mesh
            else:
                im_err = mesh
            if r == n_rows - 1:
                ax.set_xlabel("time (tu)", fontsize=8)
            else:
                ax.tick_params(labelbottom=False)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=7)
            if r > 0:
                ax.set_yticks([])
            ax.tick_params(labelsize=6)

    col_titles = ["state: slow X", "state: fast Y", "|error|: slow X", "|error|: fast Y"]
    for c, ttl in enumerate(col_titles):
        axes[0, c].set_title(ttl, fontsize=9)

    cb_state = fig.colorbar(im_state, ax=list(axes[:, :2].ravel()), shrink=0.9, pad=0.01)
    cb_state.set_label("state", fontsize=8)
    cb_err = fig.colorbar(im_err, ax=list(axes[:, 2:].ravel()), shrink=0.9, pad=0.01)
    cb_err.set_label(f"|error| (vmax={e_vmax:.2f}, q99.5)", fontsize=8)
    fig.suptitle(
        f"L96 {case.upper()} — {rank} window #{win_idx} — {method_label} across fast-Y density",
        fontsize=10,
    )
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def build_window_impact_section(rows, fig_dir: Path, case: str = "s0") -> str:
    ref_label, ref_key, ref_dir = "DirectUNet(aug)+SDA3 [aug mean]", "directunet_sda3", DEFAULT_HYBRID_DIR
    windows = select_reference_windows(ref_dir, ref_key, case, KEEP_K_VALUES[0])
    fig_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "## 3. Best/median/worst-window impact of fast-Y density",
        "",
        (f"Windows selected by full-density (`keep_k=16`) reconstruction RMSE of the best "
         f"overall scheme (**{ref_label}**, S0), mirroring "
         "`generate_l96_consolidated_report.py`'s `select_windows` convention -- but here the "
         "varying axis across the figure/table is `keep_k`, not method."),
        "",
    ]
    for rank in ("best", "median", "worst"):
        win_idx, ref_rmse = windows[rank]
        lines.append(build_window_table(rank, win_idx, ref_rmse, case, rows))
        fig_path = fig_dir / f"keep_k_impact_{case}_{rank}.png"
        plot_keep_k_hovmoller(fig_path, case, rank, win_idx, ref_label, ref_key, ref_dir)
        rel_path = fig_path.relative_to(fig_dir.parent)
        lines.append(f"![{rank} window]({rel_path})")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="L96 obs-density training-augmentation dedicated report")
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--fig-dir", default=str(DEFAULT_FIG_DIR))
    parser.add_argument("--case", default="s0", choices=CASES)
    args = parser.parse_args()

    out_path = Path(args.output)
    fig_dir = Path(args.fig_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sections = [
        "# L96 Fast-Y Observation-Density: Training-Augmentation Study",
        "",
        EXPERIMENT_DESCRIPTION,
        build_summary_section(METHOD_ROWS),
        build_window_impact_section(METHOD_ROWS, fig_dir, args.case),
    ]
    out_path.write_text("\n".join(sections))
    logger.info(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
