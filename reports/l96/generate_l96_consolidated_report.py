#!/usr/bin/env python3
"""Consolidated L96 benchmark report: full metric tables + Hovmöller reconstructions.

Consumes only cached artifacts of the DA-parity benchmark (Obs30, dws=500,
200 shared test windows):

- DA metric cache  ``experiments/l96_baselines_dws500_s0c_*_obsj2_int100_fw.json``
- DA trajectories  ``experiments/l96_baselines_trajectories_dws500_s0c_*_int100_fw.npz``
- Test dataset     ``experiments/l96_datasets_obsj2_int100_nwin200.pt``
- Neural estimates ``experiments/L*/estimates_{s0,s1}.npz``

Outputs ``reports/l96/outputs/l96_consolidated_benchmark.md`` with RMSE/EV/ES tables
over the all/slow/fast variable groups, a consistency-check section (cached
metrics recomputed from stored arrays), and Hovmöller reconstruction figures
(state + |error| maps, slow/fast blocks) for the worst/median/best test windows
ranked by the best DA scheme (Strong-4DVar per-window RMSE).
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluation.estimate_metrics import evaluate_estimates

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

DA_JSON_CANDIDATES = [
    "experiments/l96_baselines_dws500_s0c_inf2.0_etkf_inf2.0_obsj2_int100_fw.json",
    "experiments/l96_baselines_dws500_inf2.0_etkf_inf2.0_obsj2_int100_fw.json",
    "experiments/l96_baselines_dws500_inf2.0_etkf_inf2.0_obsj2_int100.json",
]
DA_TRAJ_CANDIDATES = [
    "experiments/l96_baselines_trajectories_dws500_s0c_inf2.0_etkf_inf2.0_obsj2_int100_fw.npz",
    "experiments/l96_baselines_trajectories_dws500_inf2.0_etkf_inf2.0_obsj2_int100_fw.npz",
]
DATASET_CANDIDATES = [
    "experiments/l96_datasets_obsj2_int100_nwin200.pt",
]
NEURAL_EXP_DIRS = [
    "L1b_direct_unet_s0s1",
    "L2b_vanilla_cfm_s0s1",
    "L3_vanilla_cfm_s0s1",
    "L4_direct_unet_s0s1_small",
    "L5_vanilla_cfm_s0s1_small_tau0",
    "L6_vanilla_cfm_s0s1_forcing_cond",
]
DEFAULT_FIGURE_METHODS = [
    "Strong-4DVar",
    "EnKF",
    "ETKF",
    "L4_direct_unet_s0s1_small",
    "L2b_vanilla_cfm_s0s1",
]
DA_METHODS = ["ETKF", "EnKF", "Strong-4DVar"]
RANKS = ["worst", "median", "best"]
CASES = ["s0", "s1"]
GROUPS = ("all_obs", "slow", "obs_fast")
NO = 8

SCHEME_DESCRIPTIONS: list[tuple[str, str, str]] = [
    ("Strong-4DVar", "Variational",
     ("Strong-constraint 4D-Var over the dws=500 window (`B_var=2.0`, `R_var=0.5`, `max_iter=10`, "
      "`lr=0.2`, autodiff minimization); assimilates the full window trajectory.")),
    ("EnKF", "Ensemble KF",
     ("Stochastic ensemble Kalman filter, `N_ens=30`, inflation=2.0, no localization; sequential "
      "observation updates.")),
    ("ETKF", "Ensemble KF",
     "Deterministic ensemble square-root filter, `N_ens=30`, inflation=2.0, no localization."),
    ("L1b_direct_unet_s0s1", "Neural (DirectUNet)",
     "Single-pass regression obs → state, hidden [64,128,256]; obs-only conditioning; 200 epochs."),
    ("L2b_vanilla_cfm_s0s1", "Neural (CFM, τ=0)",
     ("Conditional flow matching trained at τ=0 only; sampled with a single Euler step (deterministic, "
      "conditional-mean-like); hidden [64,128,256]; 400 epochs.")),
    ("L3_vanilla_cfm_s0s1", "Neural (CFM, multi-τ)",
     ("Standard multi-τ CFM training; evaluated with a single sample (`N_outer=1`, one Euler step from a "
      "random x₀) — stochastic, no ensemble averaging; hidden [64,128,256]; 400 epochs.")),
    ("L4_direct_unet_s0s1_small", "Neural (DirectUNet)", "As L1b with small backbone [32,64,128]."),
    ("L5_vanilla_cfm_s0s1_small_tau0", "Neural (CFM, τ=0)", "As L2b with small backbone [32,64,128]."),
    ("L6_vanilla_cfm_s0s1_forcing_cond", "Neural (CFM, τ=0)",
     ("As L2b plus corrupted-forcing conditioning (`cond_extra_dim=1`); tests the robustness value of "
      "forcing input.")),
]


def make_obs_j_indices(no: int, j_truth: int, j_obs: int) -> np.ndarray:
    x_idx = list(range(no))
    y_idx = [no + k * j_truth + j for k in range(no) for j in range(j_obs)]
    return np.array(x_idx + y_idx)


def _first_existing(patterns: list[str]) -> Path:
    for p in patterns:
        path = ROOT / p
        if path.exists():
            return path
    raise FileNotFoundError(f"None of the candidates exist: {patterns}")


def short_name(name: str) -> str:
    return name.split("_")[0] if "_" in name else name


def load_truth(dataset_path: Path, obs_idx: np.ndarray) -> dict[str, np.ndarray]:
    ds = torch.load(dataset_path, map_location="cpu", weights_only=False)
    return {
        case: np.stack([w["true_state"][..., obs_idx].numpy() for w in ds[f"test_{case}"]])
        for case in CASES
    }


def load_da_trajectories(path: Path, case: str, method: str, obs_idx: np.ndarray) -> np.ndarray:
    data = np.load(path)
    traj = data[f"{case}_{method.replace('-', '_')}_trajectories"]
    if traj.shape[-1] > len(obs_idx):
        traj = traj[..., obs_idx]
    return traj.astype(np.float64)


def load_neural_trajectories(exp_dir: Path, case: str) -> np.ndarray | None:
    npz_path = exp_dir / f"estimates_{case}.npz"
    if not npz_path.exists():
        return None
    return np.load(npz_path)["trajectories"].astype(np.float64)


def collect_estimates(
    da_traj_path: Path,
    obs_idx: np.ndarray,
    neural_dirs: list[str],
) -> dict[str, dict[str, np.ndarray | None]]:
    est: dict[str, dict[str, np.ndarray | None]] = {}
    for method in DA_METHODS:
        est[method] = {case: load_da_trajectories(da_traj_path, case, method, obs_idx) for case in CASES}
    for dirname in neural_dirs:
        exp_dir = ROOT / "experiments" / dirname
        est[dirname] = {case: load_neural_trajectories(exp_dir, case) for case in CASES}
    return est


def per_window_rmse(traj: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((traj - ref) ** 2, axis=(1, 2)))


def select_windows(traj: np.ndarray, ref: np.ndarray) -> dict[str, tuple[int, float]]:
    rw = per_window_rmse(traj, ref)
    order = np.argsort(rw)
    mid = len(order) // 2
    return {
        "best": (int(order[0]), float(rw[order[0]])),
        "median": (int(order[mid]), float(rw[order[mid]])),
        "worst": (int(order[-1]), float(rw[order[-1]])),
    }


def resolve_figure_methods(names: list[str], available: dict[str, dict[str, np.ndarray | None]]) -> list[str]:
    resolved = []
    for name in names:
        if name not in available:
            raise FileNotFoundError(f"Unknown method '{name}' (not a DA scheme or known experiment dir)")
        missing = [c for c in CASES if available[name][c] is None]
        if missing:
            raise FileNotFoundError(f"Missing estimates for '{name}' cases {missing}")
        resolved.append(name)
    return resolved


def _run_l96_convention_groups(traj: np.ndarray, ref: np.ndarray) -> dict[str, dict[str, float]]:
    """Replicate evaluation/run_l96.py metric conventions.

    RMSE = mean over windows of per-window RMSE; EV = pooled; ES = pooled MAE
    (only valid for deterministic schemes, i.e. Strong-4DVar).
    """
    err_sq = (traj - ref) ** 2
    rmse_dim = np.mean(np.sqrt(np.mean(err_sq, axis=1)), axis=0)
    ev_dim = 1.0 - np.mean(err_sq, axis=(0, 1)) / np.maximum(np.var(ref, axis=(0, 1)), 1e-12)
    es_dim = np.mean(np.abs(traj - ref), axis=(0, 1))

    def grouped(arr: np.ndarray) -> dict[str, float]:
        return {"slow": float(np.mean(arr[:NO])), "obs_fast": float(np.mean(arr[NO:])), "all_obs": float(np.mean(arr))}

    return {"rmse": grouped(rmse_dim), "ev": grouped(ev_dim), "es": grouped(es_dim)}


def check_da_consistency(
    da_json_path: Path,
    est: dict[str, dict[str, np.ndarray | None]],
    truth: dict[str, np.ndarray],
) -> tuple[float, int]:
    with open(da_json_path) as f:
        cached = json.load(f)
    max_diff = 0.0
    n_checked = 0
    for case in CASES:
        for method, metrics in cached.get(case, {}).items():
            conv = _run_l96_convention_groups(est[method][case], truth[case])
            pairs = [("rmse", metrics["groups"], conv["rmse"]), ("ev", metrics["ev"]["groups"], conv["ev"])]
            if method == "Strong-4DVar":
                pairs.append(("es", metrics["es"]["groups"], conv["es"]))
            for _, cache_groups, new_groups in pairs:
                for group in GROUPS:
                    max_diff = max(max_diff, abs(cache_groups[group] - new_groups[group]))
                    n_checked += 1
    return max_diff, n_checked


def check_neural_truth(
    est: dict[str, dict[str, np.ndarray | None]],
    truth: dict[str, np.ndarray],
) -> tuple[float, list[str]]:
    max_diff = 0.0
    problems: list[str] = []
    for name in NEURAL_EXP_DIRS:
        for case in CASES:
            traj = est.get(name, {}).get(case)
            if traj is None:
                continue
            expected = truth[case]
            if traj.shape != expected.shape:
                problems.append(f"{name}/{case}: estimates shape {traj.shape} != truth shape {expected.shape}")
                continue
            stored_truth = np.load(ROOT / "experiments" / name / f"estimates_{case}.npz")["truth"].astype(np.float64)
            max_diff = max(max_diff, float(np.max(np.abs(stored_truth - expected))))
    return max_diff, problems


def collect_metric_values(
    est: dict[str, dict[str, np.ndarray | None]],
    truth: dict[str, np.ndarray],
    row_order: list[str],
) -> dict[str, dict[tuple[str, str], dict[str, float]]]:
    values: dict[str, dict[tuple[str, str], dict[str, float]]] = {"rmse": {}, "ev": {}, "es": {}}
    for row in row_order:
        for case in CASES:
            m = evaluate_estimates(est[row][case], truth[case])
            values["rmse"][(row, case)] = m["groups"]
            values["ev"][(row, case)] = m["ev"]["groups"]
            values["es"][(row, case)] = m["es"]["groups"]
    return values


def fmt_block_table(
    title: str,
    block: dict[tuple[str, str], dict[str, float]],
    row_order: list[str],
    higher_better: bool,
    include_degradation: bool,
) -> str:
    agg = max if higher_better else min
    best = {
        f"{case}_{group}": agg(block[(r, case)][group] for r in row_order)
        for case in CASES
        for group in GROUPS
    }
    header = "| Method | S0 all | S0 slow | S0 fast | S1 all | S1 slow | S1 fast |"
    sep = "|---|---|---|---|---|---|---|"
    if include_degradation:
        header += " S1/S0 |"
        sep += "---|"
    lines = [f"### {title}", "", header, sep]
    for row in row_order:
        cells = []
        for case in CASES:
            for group in GROUPS:
                v = block[(row, case)][group]
                cell = f"{v:.4f}"
                if abs(v - best[f"{case}_{group}"]) < 5e-5:
                    cell = f"**{cell}**"
                cells.append(cell)
        line = f"| {short_name(row)} | " + " | ".join(cells) + " |"
        if include_degradation:
            s0 = block[(row, "s0")]["all_obs"]
            s1 = block[(row, "s1")]["all_obs"]
            line += f" {s1 / s0:.3f} |" if s0 > 0 else " n/a |"
        lines.append(line)
    lines.append("")
    return "\n".join(lines)


def fmt_scheme_table() -> str:
    lines = [
        "| ID | Type | Description |",
        "|---|---|---|",
    ]
    for scheme_id, family, description in SCHEME_DESCRIPTIONS:
        lines.append(f"| {short_name(scheme_id)} | {family} | {description} |")
    lines.append("")
    return "\n".join(lines)


def plot_hovmoller(
    fig_path: Path,
    case: str,
    rank: str,
    win_idx: int,
    sel_rmse: float,
    method_names: list[str],
    est_win: dict[str, np.ndarray],
    truth_win: np.ndarray,
    obs_times: np.ndarray,
    dt: float,
) -> None:
    labels = ["Truth"] + [short_name(m) for m in method_names]
    n_rows = len(labels)
    fig, axes = plt.subplots(n_rows, 4, figsize=(15, 1.35 * n_rows + 1.0), constrained_layout=True)
    t = np.arange(truth_win.shape[0]) * dt

    state_data: list[list[np.ndarray]] = [[truth_win[:, :NO], truth_win[:, NO:]]]
    for m in method_names:
        state_data.append([est_win[m][:, :NO], est_win[m][:, NO:]])
    err_data = [[np.abs(d - truth_block) for d, truth_block in zip(row, [truth_win[:, :NO], truth_win[:, NO:]])] for row in state_data]

    flat_state = [d for row in state_data for d in row]
    s_vmin = min(d.min() for d in flat_state)
    s_vmax = max(d.max() for d in flat_state)
    e_vmax = float(np.percentile(np.concatenate([d.ravel() for row in err_data for d in row]), 99.5))
    cmap_state = plt.get_cmap("viridis")
    cmap_err = plt.get_cmap("inferno")

    im_state = None
    im_err = None
    for r, label in enumerate(labels):
        win_rmse = float(np.sqrt(np.mean(np.concatenate(err_data[r], axis=1) ** 2)))
        row_label = label if r == 0 else f"{label}\nRMSE {win_rmse:.3f}"
        for c in range(4):
            ax = axes[r, c]
            data, cmap, vmin, vmax = (
                (state_data[r][c % 2], cmap_state, s_vmin, s_vmax)
                if c < 2
                else (np.minimum(err_data[r][c % 2], e_vmax), cmap_err, 0.0, e_vmax)
            )
            mesh = ax.pcolormesh(t, np.arange(data.shape[1]), data.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
            if c < 2:
                im_state = mesh
                if r == 0:
                    for ot in obs_times:
                        ax.axvline(ot * dt, color="w", lw=0.5, ls=":", alpha=0.85)
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
    ylabels_fast = [f"Y{j + 1}^{k}" for k in range(1, NO + 1) for j in range(2)]
    for c in (1, 3):
        axes[0, c].set_yticks(np.arange(len(ylabels_fast))[::2])
        axes[0, c].set_yticklabels(ylabels_fast[::2], fontsize=5)

    cb_state = fig.colorbar(im_state, ax=list(axes[:, :2].ravel()), shrink=0.9, pad=0.01)
    cb_state.set_label("state", fontsize=8)
    cb_err = fig.colorbar(im_err, ax=list(axes[:, 2:].ravel()), shrink=0.9, pad=0.01)
    cb_err.set_label(f"|error| (vmax={e_vmax:.2f}, q99.5)", fontsize=8)
    fig.suptitle(
        f"L96 {case.upper()} — {rank} window #{win_idx} (Strong-4DVar window RMSE {sel_rmse:.3f}); dotted lines = obs times",
        fontsize=10,
    )
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_FIGURE_METHODS,
                        help="Methods for the reconstruction figures (DA schemes and/or experiment dir names)")
    parser.add_argument("--ranks", nargs="+", default=RANKS, choices=RANKS)
    parser.add_argument("--out-dir", default="reports/l96/outputs")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    da_json_path = _first_existing(DA_JSON_CANDIDATES)
    da_traj_path = _first_existing(DA_TRAJ_CANDIDATES)
    dataset_path = _first_existing(DATASET_CANDIDATES)
    logger.info("DA json: %s | DA trajs: %s | dataset: %s", da_json_path, da_traj_path, dataset_path)

    obs_idx = make_obs_j_indices(NO, 4, 2)
    truth = load_truth(dataset_path, obs_idx)
    est = collect_estimates(da_traj_path, obs_idx, NEURAL_EXP_DIRS)
    figure_methods = resolve_figure_methods(args.methods, est)
    table_rows = DA_METHODS + NEURAL_EXP_DIRS

    values = collect_metric_values(est, truth, table_rows)
    with open(da_json_path) as f:
        cfg = json.load(f)["config"]

    da_max_diff, n_checked = check_da_consistency(da_json_path, est, truth)
    truth_max_diff, problems = check_neural_truth(est, truth)
    da_ok = da_max_diff <= args.tolerance
    truth_ok = truth_max_diff <= args.tolerance and not problems

    md: list[str] = [
        "# L96 Consolidated Benchmark — DA baselines vs neural models",
        "",
        (
            "Setup: two-scale L96, Obs30 (`obs_interval=100`, `obs_j=2` → 24D observed space), "
            f"dws={cfg.get('da_window_steps', 500)}, 200 shared cached test windows; "
            "S1 = ±20% params + ±10% bias (DA forward model uses biased `*_da`)."
        ),
        "",
        (
            "All table values are recomputed from the stored trajectory arrays via "
            "`evaluation/estimate_metrics.py`; **bold** marks the best value per column."
        ),
        "",
        "## Benchmarked schemes",
        "",
        fmt_scheme_table(),
        (
            "Shared setup: all L-series neural models are trained and evaluated on the identical DA-parity "
            "benchmark (all-5 params ±20% randomized per window; S1 adds a ±10% bias; models operate in the "
            "24D observed subspace with obs-only inputs unless noted). DA baselines receive the same per-window "
            "parameters as the truth generation (S0) or their biased `*_da` counterparts (S1), which is what "
            "makes the DA-vs-neural comparison apples-to-apples."
        ),
        "",
        "## RMSE (pooled, lower is better)",
        "",
        fmt_block_table("RMSE by variable group", values["rmse"], table_rows, False, True),
        (
            "Note on conventions: the DA metric cache stores the **mean of per-window RMSEs** "
            "(evaluation/run_l96.py), while this table uses the **pooled** convention "
            "(`sqrt(mean sq err)` over all windows/timesteps) for every method — the same convention as the "
            "neural evaluation. Pooled RMSE is ≤ mean-of-window RMSE, so DA values here are slightly lower "
            "(more favorable) than in the legacy cache; both orderings agree."
        ),
        "",
        "## Explained Variance (higher is better)",
        "",
        fmt_block_table("EV by variable group", values["ev"], table_rows, True, False),
        "## Energy Score (lower is better)",
        "",
        fmt_block_table("ES by variable group", values["es"], table_rows, False, False),
        (
            "Caveat: EnKF/ETKF cached ES values are computed from their forecast ensembles (proper scoring "
            "rule, N=30); neural models and Strong-4DVar are deterministic, so their ES reduces to a per-dim "
            "MAE proxy (N=1). The two are not strictly comparable — deterministic ES ignores sharpness."
        ),
        "",
        "## Consistency checks",
        "",
        f"- DA cached metrics vs recomputed-from-npz ({n_checked} values): max |Δ| = {da_max_diff:.2e} → "
        + ("PASS" if da_ok else f"FAIL (tolerance {args.tolerance})"),
        f"- Neural stored truth vs dataset true_state[:, obs_var_indices]: max |Δ| = {truth_max_diff:.2e} → "
        + ("PASS" if truth_ok else f"FAIL (tolerance {args.tolerance})"),
    ]
    for problem in problems:
        md.append(f"- WARNING: {problem}")

    md += [
        "",
        "## Reconstruction examples (Hovmöller)",
        "",
        (
            "Windows ranked by per-window pooled 24D RMSE of Strong-4DVar (best DA scheme); "
            "each figure shows rows = Truth/methods and columns = state / |error| maps for the slow X (8D) and "
            "fast Y (16D) blocks. State colors share one scale per figure; error maps share one scale across all "
            "rows/methods (99.5th-percentile cap, noted on the colorbar). Dotted vertical lines on the truth row "
            "mark observation times."
        ),
        "",
    ]
    header = "| Case | Rank | Window | 4DVar win-RMSE | " + " | ".join(short_name(n) for n in figure_methods) + " |"
    md += [header, "|---|---|---|---|" + "---|" * len(figure_methods)]

    for case in CASES:
        sel = select_windows(est["Strong-4DVar"][case], truth[case])
        for rank in args.ranks:
            win_idx, sel_rmse = sel[rank]
            w = torch.load(dataset_path, map_location="cpu", weights_only=False)[f"test_{case}"][win_idx]
            obs_times = np.where(w["obs_mask"].numpy())[0]
            est_win = {name: est[name][case][win_idx] for name in figure_methods}
            truth_win = truth[case][win_idx]
            fig_path = figs_dir / f"l96_hovm_{case}_{rank}.png"
            plot_hovmoller(fig_path, case, rank, win_idx, sel_rmse, figure_methods, est_win, truth_win, obs_times, float(cfg.get("dt", 0.001)))
            logger.info("Figure saved: %s", fig_path)
            cells = [f"{per_window_rmse(est[n][case][win_idx:win_idx + 1], truth_win[None])[0]:.3f}" for n in figure_methods]
            md.append(f"| {case.upper()} | {rank} | {win_idx} | {sel_rmse:.3f} | " + " | ".join(cells) + " |")

    md += [""]
    for case in CASES:
        for rank in args.ranks:
            md += [f"![{case}-{rank}](figs/l96_hovm_{case}_{rank}.png)", ""]

    report_path = out_dir / "l96_consolidated_benchmark.md"
    report_path.write_text("\n".join(md))
    logger.info("Report saved: %s", report_path)

    if not da_ok or not truth_ok or problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
