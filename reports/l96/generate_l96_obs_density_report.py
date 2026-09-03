#!/usr/bin/env python3
"""L96 DA-baseline observation-density report builder.

Compares two observation configurations over the SAME 200-window cached S0/S1
test set (identical dynamics / truth / params; only ``obs`` changes):

* **obsj2** (canonical): all 24D observed (8 slow X + 16 fast Y1,Y2 per node).
* **slow-only obsj0**: only the 8 slow X observed (no fast Y).

Both are evaluated on the **same 24D eval subspace** (slow + first-2-fast), so
the slow / obs_fast / all_obs metric groups are directly comparable — the
"obs_fast" group on the slow-only config reflects fast variables NOT observed by
the DA (a stress test of slow-only observation).

Consumes (all produced by this session's runs with the S1 ``case=2`` corrupted-
forcing fix):
* state-only DA: ``l96_baselines_dws500_*_obsj{2,0}_int100*.json``
* joint  DA:     ``l96_joint_comparison{,_slowobs}.json``

Generates ``reports/l96/outputs/l96_obs_density_da_baselines.md``.
"""
import json
import logging
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.run_l96 import make_obs_j_indices

# Canonical (obsj2) artifacts
CANON_STATE_JSON = ROOT / "experiments/l96_baselines_dws500_s0c_inf2.0_etkf_inf2.0_obsj2_int100_fw.json"
CANON_JOINT_JSON = ROOT / "experiments/l96_joint_comparison.json"
# Slow-only (obsj0) artifacts
SLOW_STATE_JSON = ROOT / "experiments/l96_baselines_dws500_slowobs_inf2.0_etkf_inf2.0_obsj0_s1j2_int100.json"
SLOW_JOINT_JSON = ROOT / "experiments/l96_joint_comparison_slowobs.json"

# State-only DA trajectories for the Hovmöller reconstruction figures (the
# joint comparison npz carries different encodings than the state-only cache,
# so the obsj2/obsj0 figures use the dedicated state-only trajectory files).
CANON_STATE_TRAJ = ROOT / "experiments/l96_baselines_trajectories_dws500_s0c_s1cfix_inf2.0_etkf_inf2.0_obsj2_int100_fw.npz"
SLOW_STATE_TRAJ = ROOT / "experiments/l96_baselines_trajectories_dws500_slowobs_inf2.0_etkf_inf2.0_obsj0_s1j2_int100.npz"
DATASET_PATH = ROOT / "experiments/l96_datasets_obsj2_int100_nwin200.pt"
DEFAULT_FIGS_DIR = ROOT / "reports/l96/outputs/figs_obs_density"

DEFAULT_OUT = ROOT / "reports/l96/outputs/l96_obs_density_da_baselines.md"

PARAM_NAMES = ["F", "c1", "hx", "eps", "w1", "w2", "w3", "w4"]
STATE_ONLY_METHODS = ["Strong-4DVar", "EnKF", "ETKF"]
JOINT_METHODS = ["Joint-EnKF", "Joint-ETKF", "Joint-Strong-4DVar"]
CASES = ["s0", "s1"]
RANKS = ["worst", "median", "best"]
NO = 8


def load_json(path: Path):
    if not path.exists():
        logger.warning("JSON not found: %s", path)
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not read %s: %s", path, e)
        return None


def fmt_num(x, missing="--", ndigits=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return missing
    return f"{x:.{ndigits}f}"


def state_row(state_data, case, method):
    """Returns (mean, slow, obs_fast) from a run_and_cache_baselines state-only JSON."""
    entry = (state_data or {}).get(case.lower(), {}).get(method)
    if not entry:
        return None
    g = entry.get("groups", {})
    return (entry.get("mean"), g.get("slow"), g.get("obs_fast"))


def joint_state_row(joint_data, case, method):
    """Returns (mean, slow, obs_fast, ev_mean, param_mean) from the comparator JSON."""
    entry = (joint_data or {}).get(case, {}).get(method)
    if not entry:
        return None
    sr = entry.get("state_rmse", {})
    ev = entry.get("ev", {})
    pr = entry.get("param_rmse")
    pmean = (sum(pr.values()) / len(pr)) if pr else None
    return (sr.get("mean"), sr.get("slow"), sr.get("obs_fast"), ev.get("mean"), pmean)


def build_state_table(canon, slow):
    lines = []
    lines.append("## State-only DA baselines (S0/S1)")
    lines.append("")
    lines.append("| Case | Method | obsj2 mean | obsj0 mean | Δ mean | obsj2 slow | obsj0 slow | obsj2 obs_fast | obsj0 obs_fast |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for case in ("s0", "S0"):
        pass
    cases = [("S0", "s0"), ("S1", "s1")]
    for case_label, case_key in cases:
        for m in STATE_ONLY_METHODS:
            c = state_row(canon, case_key, m)
            s = state_row(slow, case_key, m)
            if c is None and s is None:
                continue
            delta = (s[0] - c[0]) if (c and s) else None
            lines.append(
                f"| {case_label} | {m} | {fmt_num(c[0]) if c else '--'} | "
                f"{fmt_num(s[0]) if s else '--'} | {fmt_num(delta) if delta is not None else '--'} | "
                f"{fmt_num(c[1]) if c else '--'} | {fmt_num(s[1]) if s else '--'} | "
                f"{fmt_num(c[2]) if c else '--'} | {fmt_num(s[2]) if s else '--'} |"
            )
    lines.append("")
    return lines


def build_joint_state_table(canon, slow):
    lines = []
    lines.append("## Joint state-parameter DA baselines (S0/S1)")
    lines.append("")
    lines.append("| Case | Method | obsj2 mean | obsj0 mean | Δ mean | obsj2 EV | obsj0 EV | obsj2 param* | obsj0 param* |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for case in ("S0", "S1"):
        for m in JOINT_METHODS:
            c = joint_state_row(canon, case, m)
            s = joint_state_row(slow, case, m)
            if c is None and s is None:
                continue
            delta = (s[0] - c[0]) if (c and s) else None
            lines.append(
                f"| {case} | {m} | {fmt_num(c[0]) if c else '--'} | "
                f"{fmt_num(s[0]) if s else '--'} | {fmt_num(delta) if delta is not None else '--'} | "
                f"{fmt_num(c[3]) if c else '--'} | {fmt_num(s[3]) if s else '--'} | "
                f"{fmt_num(c[4]) if c else '--'} | {fmt_num(s[4]) if s else '--'} |"
            )
    lines.append("*param = mean of the (identifiable) per-parameter RMSE "
                 "(8 on S0, 6 on S1 — w3/w4 pinned to the reference prior at J=2).*")
    lines.append("")
    return lines


def build_joint_param_table(canon, slow):
    lines = []
    lines.append("## Joint-DA per-parameter RMSE (S0/S1)")
    lines.append("")
    for case in ("S0", "S1"):
        lines.append(f"### {case} — per-parameter RMSE")
        lines.append("")
        lines.append(f"| Method / config | mean | {' | '.join(PARAM_NAMES)} |")
        lines.append(f"|---|---|{'---|' * len(PARAM_NAMES)}")
        for m in JOINT_METHODS:
            for label, data in (("obsj2", canon), ("obsj0", slow)):
                entry = (data or {}).get(case, {}).get(m)
                pr = (entry or {}).get("param_rmse")
                if not pr:
                    continue
                pmean = sum(pr.values()) / len(pr)
                cells = " | ".join(fmt_num(pr.get(p)) for p in PARAM_NAMES)
                lines.append(f"| {m} / {label} | {pmean:.4f} | {cells} |")
        lines.append("")
    return lines


def load_state_trajs(path: Path, case: str, method: str, obs_idx: np.ndarray) -> np.ndarray:
    """Load a state-only DA trajectory and subsample to the 24D eval subspace."""
    data = np.load(path)
    traj = data[f"{case}_{method.replace('-', '_')}_trajectories"]
    if traj.shape[-1] > len(obs_idx):
        traj = traj[..., obs_idx]
    return traj.astype(np.float64)


def per_window_rmse(traj: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((traj - ref) ** 2, axis=(1, 2)))


def select_windows(ref_traj: np.ndarray, ref: np.ndarray) -> dict[str, tuple[int, float]]:
    """Rank windows by the obsj2 (reference) per-window RMSE so the obsj2 and
    obsj0 reconstructions are shown on the identical windows."""
    rw = per_window_rmse(ref_traj, ref)
    order = np.argsort(rw)
    mid = len(order) // 2
    return {
        "best": (int(order[0]), float(rw[order[0]])),
        "median": (int(order[mid]), float(rw[order[mid]])),
        "worst": (int(order[-1]), float(rw[order[-1]])),
    }


def plot_obsdensity_hovmoller(
    fig_path: Path,
    case: str,
    rank: str,
    win_idx: int,
    sel_rmse: float,
    rows: list[tuple[str, np.ndarray | None]],
    truth_win: np.ndarray,
    obs_times: np.ndarray,
    dt: float,
) -> None:
    """Per-{case,rank} figure: rows = Truth + per-(method,config) reconstructions,
    columns = state slow X / state fast Y / |error| slow X / |error| fast Y.

    ``rows`` is an ordered list of ``(label, traj)`` with the first entry the
    truth (traj=None); subsequent entries are (method·config, traj-in-24D-eval).
    """
    labels = [r[0] for r in rows]
    n_rows = len(labels)
    fig, axes = plt.subplots(n_rows, 4, figsize=(15, 1.35 * n_rows + 1.0), constrained_layout=True)
    t = np.arange(truth_win.shape[0]) * dt

    state_data: list[list[np.ndarray]] = [[truth_win[:, :NO], truth_win[:, NO:]]]
    for _, traj in rows[1:]:
        state_data.append([traj[:, :NO], traj[:, NO:]])
    err_data: list[list[np.ndarray]] = [
        [np.abs(d - truth_block) for d, truth_block in zip(row, [truth_win[:, :NO], truth_win[:, NO:]])]
        for row in state_data
    ]

    flat_state = [d for row in state_data for d in row]
    s_vmin = min(d.min() for d in flat_state)
    s_vmax = max(d.max() for d in flat_state)
    e_vmax = float(np.percentile(np.concatenate([d.ravel() for row in err_data for d in row]), 99.5))
    cmap_state = plt.get_cmap("viridis")
    cmap_err = plt.get_cmap("inferno")

    im_state = None
    im_err = None
    for r in range(n_rows):
        win_rmse = float(np.sqrt(np.mean(np.concatenate(err_data[r], axis=1) ** 2)))
        row_label = labels[r] if r == 0 else f"{labels[r]}\nRMSE {win_rmse:.3f}"
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
        f"L96 {case.upper()} — {rank} window #{win_idx} (obsj2 ETKF window RMSE {sel_rmse:.3f}); "
        "dotted lines = obs times",
        fontsize=10,
    )
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def build_hovmoller_section(
    canon_traj: Path,
    slow_traj: Path,
    dataset_path: Path,
    obs_idx: np.ndarray,
    figs_dir: Path,
    dt: float,
) -> list[str]:
    """Render the Hovmöller reconstruction section comparing obsj2 vs obsj0.

    For each case/rank, one figure whose rows are Truth + {EnKF,ETKF,
    Strong-4DVar}×{obsj2,obsj0} and columns are state slow X / state fast Y /
    |error| slow X / |error| fast Y. Windows are ranked by the obsj2 (reference)
    per-window RMSE so both observation configurations use identical windows.
    """
    if not canon_traj.exists() or not slow_traj.exists() or not dataset_path.exists():
        logger.warning("Missing trajectory/dataset artifacts; skipping Hovmöller figures.")
        return []
    ds = torch.load(dataset_path, map_location="cpu", weights_only=False)
    truth = {c: np.stack([w["true_state"][..., obs_idx].numpy() for w in ds[f"test_{c}"]]) for c in CASES}
    canon = {c: {m: load_state_trajs(canon_traj, c, m, obs_idx) for m in STATE_ONLY_METHODS} for c in CASES}
    slow = {c: {m: load_state_trajs(slow_traj, c, m, obs_idx) for m in STATE_ONLY_METHODS} for c in CASES}
    figs_dir.mkdir(parents=True, exist_ok=True)

    md = [
        "## Reconstruction examples (Hovmöller): obsj2 vs slow-only obsj0",
        "",
        (
            "Per case/rank, state and |error| maps for the slow X (8D) and fast Y (16D) blocks, "
            "with rows = Truth + {EnKF, ETKF, Strong-4DVar} × {obsj2, obsj0}. Windows are ranked "
            "by the **obsj2** (reference) per-window 24D RMSE so both observation configurations "
            "are shown on the identical windows. State colors share one scale per figure; error "
            "maps share one scale across all rows (99.5th-percentile cap). Dotted vertical lines "
            "on the truth row mark observation times. The slow-only obsj0 rows make visible the "
            "degradation concentrated in the **unobserved** obs_fast (fast Y) block."
        ),
        "",
    ]
    header = "| Case | Rank | Window# | obsj2 RMSE* | " + " | ".join(f"{m}·obsj2/{m}·obsj0" for m in STATE_ONLY_METHODS) + " |"
    md += [header, "|---|---|---|---|" + "---|" * len(STATE_ONLY_METHODS)]

    for case in CASES:
        sel = select_windows(canon[case]["ETKF"], truth[case])
        for rank in RANKS:
            win_idx, sel_rmse = sel[rank]
            w = ds[f"test_{case}"][win_idx]
            obs_times = np.where(w["obs_mask"].numpy())[0]
            truth_win = truth[case][win_idx]
            rows: list[tuple[str, np.ndarray | None]] = [("Truth", None)]
            for m in STATE_ONLY_METHODS:
                rows.append((f"{m}·obsj2", canon[case][m][win_idx]))
                rows.append((f"{m}·obsj0", slow[case][m][win_idx]))
            fig_path = figs_dir / f"obsdensity_hovm_{case}_{rank}.png"
            plot_obsdensity_hovmoller(fig_path, case, rank, win_idx, sel_rmse, rows, truth_win, obs_times, dt)
            logger.info("Figure saved: %s", fig_path)
            per_win = {m: (
                f"{per_window_rmse(canon[case][m][win_idx:win_idx + 1], truth_win[None])[0]:.3f}/"
                f"{per_window_rmse(slow[case][m][win_idx:win_idx + 1], truth_win[None])[0]:.3f}"
            ) for m in STATE_ONLY_METHODS}
            rel = f"{figs_dir.name}/obsdensity_hovm_{case}_{rank}.png"
            md.append(f"| {case.upper()} | {rank} | {win_idx} | {sel_rmse:.3f} | " + " | ".join(per_win[m] for m in STATE_ONLY_METHODS) + " |")
            md.append(f"![{case}-{rank}]({rel})")
            md.append("")
    md.append("*obsj2 RMSE = ETKF per-window 24D RMSE (window ranking reference). "
              "Cells are `method·obsj2/method·obsj0` per-window RMSE.*")
    md.append("")
    return md


def write_report(canon_state, canon_joint, slow_state, slow_joint, output_path: Path,
                 hovm_section: list[str]) -> None:
    md = []
    md.append("# L96 DA-Baseline Observation Density: obsj2 (24D) vs slow-only obsj0 (8D)")
    md.append("")
    md.append("**System:** Lorenz-96 two-scale (NO=8, J=4), 200 shared cached S0/S1 test windows, "
              "Obs30 (obs_interval=100). Same dynamics/truth/params; only the observation changes.")
    md.append("")
    md.append("**Configurations:**")
    md.append("* **obsj2 (canonical):** 24D observed = 8 slow X + 16 fast Y1,Y2 per node.")
    md.append("* **slow-only obsj0:** only the **8 slow X** observed (no fast Y); S1 reduced dynamics "
              "kept at J=2 (24D state).")
    md.append("")
    md.append("**Eval subspace:** both are scored on the identical 24D group (slow + first-2-fast), so "
              "the metrics are directly comparable. On obsj0 the `obs_fast` group reflects fast "
              "variables **not observed** by the DA (slow-only stress test).")
    md.append("")
    md.append("**S1 forcings:** all rows use the corrected `case=2` config, i.e. the DA is fed the "
              "**corrupted** forcing `forcing_corrupted` on S1 (and `forcing_true` on S0).")
    md.append("")
    md.append("---")
    md.append("")
    md.extend(build_state_table(canon_state, slow_state))
    md.append("---")
    md.append("")
    md.extend(build_joint_state_table(canon_joint, slow_joint))
    md.append("---")
    md.append("")
    md.extend(build_joint_param_table(canon_joint, slow_joint))
    if hovm_section:
        md.append("---")
        md.append("")
        md.extend(hovm_section)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(md) + "\n")
    logger.info("Report written to %s", output_path)


def main():
    canon_state = load_json(CANON_STATE_JSON)
    canon_joint = load_json(CANON_JOINT_JSON)
    slow_state = load_json(SLOW_STATE_JSON)
    slow_joint = load_json(SLOW_JOINT_JSON)
    obs_idx = make_obs_j_indices(NO, 4, 2)
    dt = 0.001
    figs_dir = DEFAULT_FIGS_DIR
    hovm_section = build_hovmoller_section(
        CANON_STATE_TRAJ, SLOW_STATE_TRAJ, DATASET_PATH, obs_idx, figs_dir, dt
    )
    write_report(canon_state, canon_joint, slow_state, slow_joint, DEFAULT_OUT, hovm_section)


if __name__ == "__main__":
    main()
