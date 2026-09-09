#!/usr/bin/env python3
"""Reconstruction figures for the QG S0 DA reference case (lag=5.0d,
noise_frac=0.05): 3 example test windows (best/median/worst by ETKF's
per-window pooled PV-q RMSE) x all 4 DA methods (ETKF/EnKF/Strong-4DVar/
Weak-4DVar), each showing truth | free-forecast | analysis for streamfunction
(psi, both layers) and PV (q, both layers) at a representative day.

Window selection reads `rmse_list` from an existing ETKF run's saved JSON
(any `run_qg_baselines.run(..., out_path=...)` call with the same
windows/config -- e.g. `reports/qg/outputs/qg_repro_validation/etkf.json`)
so the 3 examples are picked by real per-window DA skill, not arbitrarily.

Run from the repository root::

    python reports/qg/generate_qg_reconstruction_figs.py \
        --etkf-json reports/qg/outputs/qg_repro_validation/etkf.json \
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
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from data.qg import QGConfig, QGS01Dataset
from data.qg_neural import ensure_truth_cache
from evaluation.run_qg_baselines import _build_dyn, run

CACHE_DIR = "/Odyssey/private/rfablet/Python/4dvarnet-fm-qg-100samples/reports/qg/outputs/qg_windows_1000_100_100/cache"
LAG_DAYS = 5.0
NOISE_FRAC = 0.05
CMAP = "RdBu_r"

METHODS = [
    ("ETKF", "etkf", {"N_ensemble": 80, "inflation": 1.0, "loc_radius": 6.0}),
    ("EnKF", "enkf", {"N_ensemble": 80, "inflation": 1.0, "loc_radius": 6.0}),
    ("Strong-4DVar", "strong4dvar",
     {"da_window_steps": 60, "optimizer": "lbfgs", "fourdvar_max_iter": 60,
          "fourdvar_lr": 1.0, "b_var_scale": 1.0}),
    ("Weak-4DVar", "weak4dvar",
     {"da_window_steps": 60, "optimizer": "lbfgs", "fourdvar_max_iter": 60,
          "fourdvar_lr": 1.0, "b_var_scale": 1.0, "q_var_scale": 0.1}),
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


ANIM_METHOD = next(m for m in METHODS if m[1] == "enkf")


def build_animation(window_label, window, cfg, device, out_dir, save_dir,
                    sample_days=1.0):
    """Row 1: obs | wind forcing | truth q1 | EnKF analysis q1 (4 panels, in a
    6-wide grid so it aligns with row 2). Row 2: truth/EnKF pairs for psi1,
    psi2, q2 (6 panels) -- the 3 fields row 1 doesn't already cover.
    Animated over the window, one frame per `sample_days`.

    Row-1 obs-hold logic mirrors `generate_qg_s0s1_figs.py::fig_dacycle` (the
    closest existing prior art -- obs/truth/analysis, no wind panel); the
    wind-curl panel follows `animate_qg_wind.py`'s `dyn.wind_curl_field(...)`
    pattern.
    """
    _label, method_name, method_kw = ANIM_METHOD
    analysis, _free, ref = run_one(method_name, method_kw, cfg, window, device, save_dir)

    dyn = _build_dyn(cfg, window, device, psi_state=False)
    ny = nx = cfg.nx
    split = ny * nx
    days_per = round(86400.0 / dyn.inner.dt)
    T = analysis.shape[0]
    stride = max(1, int(sample_days * days_per))
    steps = list(range(0, T, stride))

    obs = window["obs"].numpy()
    cols = window["obs_columns"].numpy()
    mask = window["obs_mask"].numpy()
    obs_steps = np.where(mask)[0]
    obs_vals = np.abs(obs[np.isfinite(obs)])
    vmax_o = float(obs_vals.max()) if obs_vals.size else 1.0

    truth_q1 = ref[:, :split].reshape(T, ny, nx)
    analysis_q1 = analysis[:, :split].reshape(T, ny, nx)
    truth_q2 = ref[:, split:].reshape(T, ny, nx)
    analysis_q2 = analysis[:, split:].reshape(T, ny, nx)
    vmax_q1 = max(np.nanmax(np.abs(truth_q1)), np.nanmax(np.abs(analysis_q1))) * 0.9
    vmax_q2 = max(np.nanmax(np.abs(truth_q2)), np.nanmax(np.abs(analysis_q2))) * 0.9

    truth_psi = dyn.inner.streamfunctions(
        torch.from_numpy(ref).float().to(device)).detach().cpu().numpy()
    analysis_psi = dyn.inner.streamfunctions(
        torch.from_numpy(analysis).float().to(device)).detach().cpu().numpy()
    vmax_psi1 = max(np.nanmax(np.abs(truth_psi[:, 0])),
                    np.nanmax(np.abs(analysis_psi[:, 0]))) * 0.9
    vmax_psi2 = max(np.nanmax(np.abs(truth_psi[:, 1])),
                    np.nanmax(np.abs(analysis_psi[:, 1]))) * 0.9

    wind_state = window["wind_state_corrupted"].to(device)
    windfields = dyn.inner.wind_curl_field(wind_state).detach().cpu().numpy()
    wind_absmax = float(np.nanmax(np.abs(windfields)))
    vmax_w = wind_absmax * 0.9 or 1.0
    wind_title = "wind-stress curl forcing" if wind_absmax > 0 else \
        "wind-stress curl forcing (none: wind_amp=0 for this window)"

    def panel(ax, field, vmax, title):
        ax.imshow(field, cmap=CMAP, vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    frames = []
    for t in steps:
        fig, axes = plt.subplots(2, 6, figsize=(21, 8.4))

        prior = obs_steps[obs_steps <= t]
        t_obs = int(prior[-1]) if prior.size else (int(obs_steps[0]) if obs_steps.size else t)
        img = np.full((ny, nx), np.nan)
        xc = int(cols[t_obs])
        if 0 <= xc < nx:
            img[:, xc] = obs[t_obs]
        panel(axes[0, 0], img, vmax_o,
              f"raw obs psi1, 1 col/event (day {t_obs / days_per:.2f}, col {xc})")
        panel(axes[0, 1], windfields[t], vmax_w, wind_title)
        panel(axes[0, 2], truth_q1[t], vmax_q1, "truth q1")
        panel(axes[0, 3], analysis_q1[t], vmax_q1, "EnKF analysis q1")
        axes[0, 4].axis("off")
        axes[0, 5].axis("off")

        panel(axes[1, 0], truth_psi[t, 0], vmax_psi1, "truth psi1")
        panel(axes[1, 1], analysis_psi[t, 0], vmax_psi1, "EnKF analysis psi1")
        panel(axes[1, 2], truth_psi[t, 1], vmax_psi2, "truth psi2")
        panel(axes[1, 3], analysis_psi[t, 1], vmax_psi2, "EnKF analysis psi2")
        panel(axes[1, 4], truth_q2[t], vmax_q2, "truth q2")
        panel(axes[1, 5], analysis_q2[t], vmax_q2, "EnKF analysis q2")

        fig.suptitle(f"S0 DA cycle ({window_label} window) -- day {t / days_per:.2f}",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(buf).convert("RGB"))
        plt.close(fig)

    path = os.path.join(out_dir, f"qg_s0_dacycle_{window_label}.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=180, loop=0)
    print(f"wrote {path} ({len(frames)} frames)")
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
    print("Loading test cache (100 windows, nx=64, seed=20042)...", flush=True)
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
        # Reruns EnKF for this window (~20-30s, cheap) rather than threading
        # build_figure's already-computed EnKF trajectory out -- keeps the two
        # builders independent/simple.
        build_animation(label, w, cfg_ref, device, args.out_dir, window_save_dir)


if __name__ == "__main__":
    main()
