#!/usr/bin/env python3
"""Reconstruction figures for the QG S0 DA reference case (lag=5.0d,
noise_frac=0.05): 3 example test windows (best/median/worst by ETKF's
per-window pooled PV-q RMSE) x all 4 DA methods (ETKF/EnKF/Strong-4DVar/
Weak-4DVar), each showing truth | free-forecast | analysis for streamfunction
(psi, both layers) and PV (q, both layers) at a representative day.

Window selection reads `crps_list`... no -- `rmse_list` from an existing
ETKF run's saved JSON (produced by qg_da_sensitivity_scratch.py --method
etkf, or any `run_qg_baselines.run()` call with the same windows/config) so
the 3 examples are picked by real per-window DA skill, not arbitrarily.

Run from the repository root::

    python reports/qg/generate_qg_reconstruction_figs.py \
        --etkf-json qg_repro_validation_lag5_noise05/etkf.json \
        --out-dir reports/qg/outputs/figs
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from data.qg import QGConfig, QGS01Dataset
from data.qg_neural import ensure_truth_cache
from evaluation.run_qg_baselines import run

CACHE_DIR = "/Odyssey/private/rfablet/Python/4dvarnet-fm-qg-100samples/reports/qg/outputs/qg_windows_1000_100_100/cache"
LAG_DAYS = 5.0
NOISE_FRAC = 0.05
CMAP = "RdBu_r"

METHODS = [
    ("ETKF", "etkf", dict(N_ensemble=80, inflation=1.0, loc_radius=6.0)),
    ("EnKF", "enkf", dict(N_ensemble=80, inflation=1.0, loc_radius=6.0)),
    ("Strong-4DVar", "strong4dvar",
     dict(da_window_steps=60, optimizer="lbfgs", fourdvar_max_iter=60,
          fourdvar_lr=1.0, b_var_scale=1.0)),
    ("Weak-4DVar", "weak4dvar",
     dict(da_window_steps=60, optimizer="lbfgs", fourdvar_max_iter=60,
          fourdvar_lr=1.0, b_var_scale=1.0, q_var_scale=0.1)),
]


def _base_cfg(nx=64, seed=20042, num_windows=100):
    return QGConfig(nx=nx, seed=seed, num_windows=num_windows,
                    obs_geometry="random_columns", cols_per_day=4,
                    obs_noise_std_frac=0.01, init_lag_days=1.0)


def select_example_windows(etkf_json_path):
    """best/median/worst window indices by ETKF's per-window pooled q RMSE."""
    with open(etkf_json_path) as f:
        d = json.load(f)
    rmse_list = d["scenarios"]["test_s0"]["rmse_list"]
    order = np.argsort(rmse_list)
    best, median, worst = order[0], order[len(order) // 2], order[-1]
    print(f"Selected windows: best=idx{best} (rmse={rmse_list[best]:.3e}), "
          f"median=idx{median} (rmse={rmse_list[median]:.3e}), "
          f"worst=idx{worst} (rmse={rmse_list[worst]:.3e})")
    return {"best": int(best), "median": int(median), "worst": int(worst)}


def _q_fields(state, ny, nx):
    T = state.shape[0]
    split = ny * nx
    return np.stack([state[:, :split].reshape(T, ny, nx),
                     state[:, split:].reshape(T, ny, nx)], axis=1)


def _symmetric_vmax(field):
    v = float(np.max(np.abs(field)))
    return v if v > 0 else 1.0


def run_one(method_name, method_kw, cfg, window, device, save_dir):
    ds = {"test_s0": [window]}
    payload = run(method_name, cfg, ds=ds, init_lag_days=LAG_DAYS,
                 scenarios=("test_s0",), init="lagged", geometry="random_columns",
                 obs_var="psi", band_half=0.25, device=device, out_path=None,
                 save_traj=save_dir, **method_kw)
    traj_path = payload["scenarios"]["test_s0"]["traj_path"]
    npz = np.load(traj_path)
    return npz["analyses"][0], npz["free_forecast"][0], npz["refs"][0]


def build_figure(window_label, window, cfg, device, out_dir, save_dir):
    from evaluation.run_qg_baselines import _build_dyn
    dyn = _build_dyn(cfg, window, device, psi_state=False)
    truth_inner = dyn.inner if hasattr(dyn, "inner") else dyn

    # ref/free are identical across methods (same window, same shared_init at
    # the same lag -- neither depends on the assimilation method), so take
    # them from the first call and only keep each method's analysis after.
    ref = free = None
    method_trajs = []
    for label, method_name, kw in METHODS:
        analysis, m_free, m_ref = run_one(method_name, kw, cfg, window, device, save_dir)
        if ref is None:
            ref, free = m_ref, m_free
        method_trajs.append((label, analysis))

    labels = ["truth", "free forecast"] + [lbl for lbl, _ in method_trajs]
    trajs = [ref, free] + [a for _, a in method_trajs]

    ny = nx = cfg.nx
    T = ref.shape[0]
    t = min(T - 1, int(T * 0.9) // 30 * 30)

    def psi_of(traj):
        return truth_inner.streamfunctions(
            torch.from_numpy(traj[t]).float().to(device)).detach().cpu().numpy()

    def q_of(traj):
        return _q_fields(traj[t:t + 1], ny, nx)[0]

    fields = [(psi_of(traj), q_of(traj)) for traj in trajs]
    n = len(fields)
    fig, axes = plt.subplots(n, 4, figsize=(16, 3.2 * n))
    col_titles = ["psi1", "psi2", "q1", "q2"]
    for i, (lbl, (psi, q)) in enumerate(zip(labels, fields)):
        panels = [psi[0], psi[1], q[0], q[1]]
        for c, (fld, ctitle) in enumerate(zip(panels, col_titles)):
            ax = axes[i, c] if n > 1 else axes[c]
            vmax = _symmetric_vmax(fld)
            ax.imshow(fld, cmap=CMAP, vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(ctitle)
            if c == 0:
                ax.set_ylabel(lbl, fontsize=10)
    fig.suptitle(f"S0 reconstruction ({window_label} window) -- "
                 f"lag={LAG_DAYS}d, noise={NOISE_FRAC}, step {t}", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, f"qg_s0_reconstruction_{window_label}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etkf-json", required=True)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "reports/qg/outputs/figs"))
    ap.add_argument("--save-traj-dir", default=os.path.join(ROOT, "qg_recon_traj_scratch"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.save_traj_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _base_cfg()
    print(f"Loading test cache (100 windows, nx=64, seed=20042)...", flush=True)
    all_windows = ensure_truth_cache(cfg, 100, CACHE_DIR)

    indices = select_example_windows(args.etkf_json)

    cfg_ref = QGConfig(nx=64, seed=20042, num_windows=100, obs_geometry="random_columns",
                       cols_per_day=4, obs_noise_std_frac=NOISE_FRAC, init_lag_days=1.0)

    for label, idx in indices.items():
        base_w = all_windows[idx]
        ic = QGS01Dataset._generate_obs_ic(cfg_ref, [base_w], [idx])
        w = dict(base_w, **ic[0])
        # save_traj's filename doesn't encode window identity (just scenario/
        # method/obs_var/lag), so a shared dir across windows would silently
        # overwrite each window's trajectories with the next one's -- a
        # per-window subdir keeps all 3 windows' saved trajectories.
        window_save_dir = os.path.join(args.save_traj_dir, label)
        build_figure(label, w, cfg_ref, device, args.out_dir, window_save_dir)


if __name__ == "__main__":
    main()
