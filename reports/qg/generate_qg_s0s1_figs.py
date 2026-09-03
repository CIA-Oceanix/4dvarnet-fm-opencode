#!/usr/bin/env python3
"""Generate illustration figures + DA-cycle animation for the QG S0/S1 report.

Builds the S0 and S1-QG2L (da_nx=32) production datasets (random-column
upper-layer-psi obs, 1% noise), runs a single-window ETKF analysis for each
(reusing the `run_qg_baselines` window machinery), and writes a self-contained
figure set + a GIF animation under ``reports/qg/outputs/figs/``:

- ``qg_s0_obs_days.png`` / ``qg_s1x32_obs_days.png``  — 2x2 panel of observed
  days: the observed upper-psi column samples over the ground-truth field.
- ``qg_s0_obs_hovmoller.png`` / ``qg_s1x32_obs_hovmoller.png`` — full-window
  observation Hovmöller (sparse columns appear as slanted tracks over time).
- ``qg_s0_forcing.png`` / ``qg_s1x32_forcing.png`` — wind-stress-curl forcing
  snapshots + true vs corrupted storm-track.
- ``qg_s0_truth_psi_q.png`` / ``qg_s1x32_truth_psi_q.png`` — ground-truth
  upper-layer streamfunction and both-layer PV at a representative step.
- ``qg_s0_analysis.png`` / ``qg_s1x32_analysis.png`` — reconstruction:
  truth | free-forecast | DA analysis with error fields.
- ``qg_s0_dacycle.gif`` / ``qg_s1x32_dacycle.gif`` — DA-cycle animation over a
  subsampled window: obs-aggregate | truth | DA analysis per frame.

Run from the repository root::

    python reports/qg/generate_qg_s0s1_figs.py                 # production
    python reports/qg/generate_qg_s0s1_figs.py --quick          # nx=32 CPU smoke
"""
import argparse
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from data.qg import QGConfig, make_qg_s0_s1_datasets
from evaluation.baselines import ETKF, _build_qg_col_loc_matrices
from evaluation.run_qg_baselines import (
    _build_dyn,
    _da_nx_for_window,
    _ensemble_from_init,
    _event_columns,
    _make_obs_system,
    _sample_init_state,
    _upsample_to_truth,
)
from models.qg_dynamics import QGDynamics

CMAP = "RdBu_r"
STEPS_PER_DAY_LABEL = "day"


def _symmetric_vmax(field: np.ndarray) -> float:
    return max(float(np.abs(field).max()), 1e-30) * 0.9


def _q_fields(state: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Layer-major flattened q-state (..., 2·ny·nx) -> (..., 2, ny, nx)."""
    lead = state.shape[:-1]
    return state.reshape(*lead, 2, ny, nx)


def _truth_inner(cfg, window, device) -> QGDynamics:
    tp = window["true_params"]
    return QGDynamics(
        nx=cfg.nx, L=cfg.L, dt=cfg.dt, beta=tp["beta"], rd=tp["rd"],
        delta=cfg.delta, U1=tp["U1"], U2=tp.get("U2", cfg.U2), rek=tp["rek"],
        filterfac=cfg.filterfac, wind_amp=window["wind_amp"],
        wind_sigma=cfg.wind_sigma, clip_range=1e-3, dtype=torch.float32,
    ).to(device)


def run_single_window(cfg, scenario, device, N_ensemble, inflation,
                      loc_radius, init_lag_days, band_half, ds=None):
    """Run one ETKF assimilation of window 0 for a scenario.

    Mirrors the per-window loop of `run_qg_baselines.run` (psi-obs): builds the
    DA model + psi obs system, samples the shared lagged init, builds the
    physical-coordinate localization, and calls the production
    `_evaluate_window`. The free forecast is rolled out from the same shared
    init exactly as `run()` does, so the analysis and free-roll match the
    production report. Returns arrays (upsampled to the truth grid) needed for
    the figures plus the raw window.
    """
    from evaluation.run_qg_baselines import _downsample_to_da, _evaluate_window
    if ds is None:
        ds = make_qg_s0_s1_datasets(
            cfg, cache_dir=os.path.join(ROOT, "reports/qg_cache"))
    w = ds[scenario][0]
    dyn = _build_dyn(cfg, w, device, psi_state=False)
    da_nx = _da_nx_for_window(cfg, w)
    nlayers = dyn.state_dim // (dyn.inner.ny * dyn.inner.nx)
    per_layer = cfg.ny * cfg.nx

    obs, r_var, obs_op, _ = _make_obs_system(cfg, w, device, "psi", loc_radius)
    forcing = w["wind_state_corrupted"].to(device)

    # ONE shared lagged init, reused for BOTH the DA ensemble and the free
    # forecast (identical to `run()`), so the comparison is apples-to-apples.
    shared_init, _ = _sample_init_state(cfg, w, init_lag_days, band_half, device)
    if da_nx != cfg.nx:
        shared_init = _downsample_to_da(
            shared_init, da_nx, nlayers, cfg.nx, device)
        lead = _downsample_to_da(
            w["init_lead_truth"].float(), da_nx, nlayers, cfg.nx, device)
    else:
        lead = w["init_lead_truth"].float()
    sigma_raw = float(lead.std(0).mean())
    init_ensemble = _ensemble_from_init(
        shared_init, sigma_raw, N_ensemble, 1.0, device, cfg)

    field_std = float(w["target_state_psi"].std())
    Lx_t = Ly_t = None
    if loc_radius is not None:
        cols_t = _event_columns(cfg, w)
        Lx_t, Ly_t = _build_qg_col_loc_matrices(
            dyn.state_dim, cols_t, 2, cfg.ny, cfg.nx, loc_radius, device,
            state_ny=da_nx, state_nx=da_nx)
    method = ETKF(N_ensemble=N_ensemble, R_var=r_var, inflation=inflation,
                  device=device, dynamics=dyn, obs_operator=obs_op,
                  loc_radius=loc_radius, noise_init_std=field_std,
                  loc_Lx_t=Lx_t, loc_Ly_t=Ly_t)

    res = _evaluate_window(cfg, w, method, device, obs=obs, forcing=forcing,
                           init_ensemble=init_ensemble)

    ref = w["true_state"].numpy()
    traj_da = res.trajectory
    if da_nx != cfg.nx:
        traj_da = _upsample_to_truth(traj_da, da_nx, nlayers, cfg.nx, device)

    free_roll = dyn.rollout_trajectory(
        shared_init, cfg.num_steps - 1, wind_state=forcing)
    free_roll = free_roll.detach().cpu().numpy()
    if da_nx != cfg.nx:
        free_roll = _upsample_to_truth(
            free_roll, da_nx, nlayers, cfg.nx, torch.device("cpu"))

    return {
        "window": w,
        "analysis": traj_da,
        "free": free_roll,
        "ref": ref,
        "truth_inner": _truth_inner(cfg, w, device),
        "per_layer": per_layer,
        "device": device,
    }


def _obs_panel(w, ny, nx, truth_inner, device):
    """Return (selected_steps, days, col_sets, truth_upper_psi)."""
    truth_q = _q_fields(w["true_state"].numpy(), ny, nx)  # (T,2,ny,nx)
    psi = truth_inner.streamfunctions(w["true_state"].to(device))
    psi = psi.detach().cpu().numpy()  # (T,2,ny,nx)
    psi1 = psi[:, 0]  # (T,ny,nx) upper layer
    mask = w["obs_mask"].numpy()
    steps = np.where(mask)[0]
    cols = w["obs_columns"].numpy()  # (T,C)
    days_per = round(86400.0 / truth_inner.dt)
    days = steps / days_per
    obs = w["obs"].numpy()
    col_sets = []
    for t in steps:
        col_sets.append([int(c) for c in cols[t] if 0 <= int(c) < nx])
    return steps, days, col_sets, psi1, truth_q, obs, int(mask.sum())


def fig_obs_days(w, truth_inner, per_scenario, device, out_dir):
    ny = nx = truth_inner.nx
    steps, days, col_sets, psi1, _truth_q, obs, _ = _obs_panel(
        w, ny, nx, truth_inner, device)
    if len(steps) < 4:
        return None
    pick = np.linspace(0, len(steps) - 1, 4).astype(int)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, idx in zip(axes.ravel(), pick):
        t = steps[idx]
        vmax_p = _symmetric_vmax(psi1[t])
        ax.imshow(psi1[t], cmap=CMAP, vmin=-vmax_p, vmax=vmax_p)
        C = w["obs_columns"].shape[1]
        for c in range(C):
            xc = int(w["obs_columns"][t, c])
            if 0 <= xc < nx:
                colvals = obs[t, c * ny: (c + 1) * ny]
                vmax_o = max(np.abs(colvals).max(), 1e-30)
                ax.scatter(np.full(ny, xc), np.arange(ny), c=np.clip(
                    colvals, -vmax_o, vmax_o), cmap=CMAP, s=14,
                    vmin=-vmax_o, vmax=vmax_o, edgecolors="k", linewidths=0.4)
        ax.set_title(f"day {days[idx]:.1f} — obs cols {col_sets[idx]}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{per_scenario}: per-day observed ψ₁ columns over truth",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, f"qg_{per_scenario}_obs_days.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def fig_obs_hovmoller(w, truth_inner, per_scenario, out_dir):
    ny = truth_inner.nx
    obs = w["obs"].numpy()  # (T, C*ny)
    mask = w["obs_mask"].numpy()
    T = obs.shape[0]
    days_per = round(86400.0 / truth_inner.dt)
    days = np.arange(T) / days_per
    img = np.full((T, ny), np.nan)
    steps = np.where(mask)[0]
    cols = w["obs_columns"].numpy()
    for t in steps:
        for c in range(cols.shape[1]):
            xc = int(cols[t, c])
            if 0 <= xc < ny:
                img[t, :] = obs[t, c * ny: (c + 1) * ny]
                break
    fig, ax = plt.subplots(figsize=(9, 4.5))
    vmax = float(np.nanmax(np.abs(img)))
    m = ax.imshow(img.T, aspect="auto", cmap=CMAP, vmin=-vmax, vmax=vmax,
                  extent=[days[0], days[-1], ny, 0])
    ax.set_xlabel("day")
    ax.set_ylabel("meridional y")
    ax.set_title(f"{per_scenario}: observation Hovmöller (ψ₁ column values)")
    fig.colorbar(m, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    path = os.path.join(out_dir, f"qg_{per_scenario}_obs_hovmoller.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def fig_forcing(w, truth_inner, per_scenario, out_dir):
    wind_curl = w["wind_curl"].numpy()  # (T,ny,nx)
    ws_true = w["wind_state_true"].numpy()
    ws_corrupt = w["wind_state_corrupted"].numpy()
    days_per = round(86400.0 / truth_inner.dt)
    steps = np.linspace(0, wind_curl.shape[0] - 1, 3).astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, t in zip(axes.ravel(), steps):
        wf = wind_curl[t]
        vmax = max(np.abs(wf).max(), 1e-15)
        ax.imshow(wf, cmap=CMAP, vmin=-vmax, vmax=vmax)
        ax.set_title(f"wind curl day {t / days_per:.1f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    days = np.arange(ws_true.shape[0]) / days_per
    ax2.plot(days, ws_true[:, 0], label="true amplitude", lw=1.5)
    ax2.plot(days, ws_corrupt[:, 0], label="corrupted amplitude", lw=1.5, ls="--")
    ax2.set_xlabel("day")
    ax2.set_ylabel("wind amplitude A")
    ax2.set_title(f"{per_scenario}: wind-stress-curl amplitude true vs corrupted")
    ax2.legend()
    fig2.tight_layout()
    fig.savefig(os.path.join(out_dir, f"qg_{per_scenario}_forcing.png"), dpi=120)
    fig2.savefig(os.path.join(out_dir, f"qg_{per_scenario}_forcing_amp.png"), dpi=120)
    plt.close(fig)
    plt.close(fig2)
    return os.path.join(out_dir, f"qg_{per_scenario}_forcing.png")


def _days_per(inner):
    return round(86400.0 / inner.dt)


def fig_truth_psi_q(w, truth_inner, per_scenario, out_dir, device):
    ny = nx = truth_inner.nx
    T = w["true_state"].shape[0]
    t = min(int(0.5 * T), T - 1)
    q = _q_fields(w["true_state"].numpy(), ny, nx)[t]
    psi = truth_inner.streamfunctions(
        w["true_state"][t].to(device)).detach().cpu().numpy()
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (fld, ttl) in enumerate(
            ((psi[0], "upper ψ₁"), (psi[1], "lower ψ₂"),
             (q[0], "upper q₁"), (q[1], "lower q₂"))):
        ax = axes.ravel()[ax]
        vmax = _symmetric_vmax(fld)
        ax.imshow(fld, cmap=CMAP, vmin=-vmax, vmax=vmax)
        ax.set_title(ttl)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{per_scenario}: ground-truth streamfunction and PV "
                 f"(day {t / _days_per(truth_inner):.1f})")
    fig.tight_layout()
    path = os.path.join(out_dir, f"qg_{per_scenario}_truth_psi_q.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def fig_analysis(data, per_scenario, out_dir, device):
    ny = nx = data["truth_inner"].nx
    T = data["analysis"].shape[0]
    t = min(T - 1, int(T * 0.9) // 30 * 30)
    analys = _q_fields(data["analysis"], ny, nx)[t]
    free = _q_fields(data["free"], ny, nx)[t]
    ref = _q_fields(data["ref"], ny, nx)[t]
    truth_inner = data["truth_inner"]
    psi_anal = truth_inner.streamfunctions(
        torch.from_numpy(data["analysis"][t]).float().to(device)).detach().cpu().numpy()
    psi_free = truth_inner.streamfunctions(
        torch.from_numpy(data["free"][t]).float().to(device)).detach().cpu().numpy()
    psi_ref = truth_inner.streamfunctions(
        torch.from_numpy(data["ref"][t]).float().to(device)).detach().cpu().numpy()

    def panel(ax, fld, truth, ttl):
        vmax = _symmetric_vmax(fld)
        ax.imshow(fld, cmap=CMAP, vmin=-vmax, vmax=vmax)
        ax.set_title(ttl)
        ax.set_xticks([])
        ax.set_yticks([])

    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    rows = [("truth", psi_ref), ("free forecast", psi_free), ("DA analysis", psi_anal)]
    for r, (lbl, fld) in enumerate(rows):
        for c in range(3):
            ax = axes[r, c]
            if c == 0:
                panel(ax, fld[0], None, f"{lbl} — ψ₁")
            elif c == 1:
                panel(ax, fld[1], None, f"{lbl} — ψ₂")
            else:
                qerr = {"truth": ref, "free forecast": free,
                        "DA analysis": analys}[lbl][0]
                panel(ax, qerr, None, f"{lbl} — q₁")
    fig.suptitle(f"{per_scenario}: DA reconstruction vs truth and free forecast")
    fig.tight_layout()
    path = os.path.join(out_dir, f"qg_{per_scenario}_analysis.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def fig_dacycle(data, per_scenario, out_dir, device, sample_days=2.0):
    ny = nx = data["truth_inner"].nx
    T = data["analysis"].shape[0]
    days_per = round(86400.0 / data["truth_inner"].dt)
    stride = max(1, int(sample_days * days_per))
    steps = list(range(0, T, stride))
    window = data["window"]
    obs = window["obs"].numpy()
    cols = window["obs_columns"].numpy()
    mask = window["obs_mask"].numpy()
    analysis = _q_fields(data["analysis"], ny, nx)
    truth = _q_fields(data["ref"], ny, nx)
    vt = np.nanmax(np.abs(truth[:, 0])) * 0.9
    vmax_q = max(np.nanmax(np.abs(analysis[:, 0])), vt) * 0.9
    frames = []
    for t in steps:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        ax = axes[0]
        img = np.full((ny, nx), np.nan)
        if mask[t]:
            for c in range(cols.shape[1]):
                xc = int(cols[t, c])
                if 0 <= xc < nx:
                    img[:, xc] = obs[t, c * ny:(c + 1) * ny]
        finite = np.isfinite(img)
        vmax_o = float(np.nanmax(np.abs(img[finite]))) if np.any(finite) else 1.0
        ax.imshow(img.T, cmap=CMAP, vmin=-vmax_o, vmax=vmax_o,
                  interpolation="nearest")
        ax.set_title(f"obs aggregate (day {t / days_per:.1f})"
                     + ("  [obs]" if mask[t] else "  [no obs]"))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(truth[t, 0], cmap=CMAP, vmin=-vmax_q, vmax=vmax_q)
        ax.set_title("truth q₁")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(analysis[t, 0], cmap=CMAP, vmin=-vmax_q, vmax=vmax_q)
        ax.set_title("DA analysis q₁")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(buf).convert("RGB"))
        plt.close(fig)
    path = os.path.join(out_dir, f"qg_{per_scenario}_dacycle.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=180, loop=0)
    print(f"wrote {path} ({len(frames)} frames, "
          f"{sum(f.size[0] * f.size[1] for f in frames) * 3 / 1e6:.1f} MB raw)")
    return path


def build_cfg(nx, num_windows, spinup_years, da_nx, cols, seed=7):
    return QGConfig(nx=nx, window_days=30.0, spinup_years=spinup_years,
                    num_windows=num_windows, obs_geometry="random_columns",
                    cols_per_day=cols, obs_noise_std_frac=0.01,
                    obs_field="psi", seed=seed, da_nx=da_nx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "reports/qg/outputs/figs"))
    ap.add_argument("--scenario", choices=["s0", "s1x32", "both"], default="both")
    ap.add_argument("--quick", action="store_true",
                    help="nx=32 CPU smoke (fast correctness check only)")
    ap.add_argument("--num-windows", type=int, default=5)
    ap.add_argument("--spinup-years", type=float, default=2.0)
    ap.add_argument("--ensemble", type=int, default=80)
    ap.add_argument("--inflation", type=float, default=1.0)
    ap.add_argument("--loc-radius", type=float, default=6.0)
    ap.add_argument("--init-lag-days", type=float, default=1.0)
    ap.add_argument("--band", type=float, default=0.25)
    ap.add_argument("--cols-per-day", type=int, default=4)
    args = ap.parse_args()

    nx = 32 if args.quick else 64
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.quick else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    scenarios = (["s0", "s1x32"] if args.scenario == "both"
                 else [args.scenario])
    per_layer_tag = {"s0": "s0", "s1x32": "s1x32"}
    ns = args.num_windows
    ens = args.ensemble
    loc = args.loc_radius
    lag = args.init_lag_days
    if args.quick:
        ns = min(ns, 2)
        ens = 20
        loc = 4.0
        lag = 1.0
    for scen in scenarios:
        da_nx = None if scen == "s0" else 32
        cfg = build_cfg(nx, ns, args.spinup_years, da_nx, args.cols_per_day)
        ds = make_qg_s0_s1_datasets(
            cfg, cache_dir=os.path.join(ROOT, "reports/qg_cache"))
        scen_key = "test_s0" if scen == "s0" else "test_s1"
        if scen_key not in ds:
            print(f"(skip {scen}: no {scen_key} dataset)")
            continue
        t0 = time.time()
        data = run_single_window(cfg, scen_key, device, ens,
                                 args.inflation, loc, lag, args.band, ds=ds)
        print(f"[{scen}] single-window ETKF {time.time() - t0:.1f}s")
        tag = per_layer_tag[scen]
        handle = data["truth_inner"]
        fig_obs_days(data["window"], handle, tag, device, args.out_dir)
        fig_obs_hovmoller(data["window"], handle, tag, args.out_dir)
        fig_forcing(data["window"], handle, tag, args.out_dir)
        fig_truth_psi_q(data["window"], handle, tag, args.out_dir, device)
        fig_analysis(data, tag, args.out_dir, device)
        fig_dacycle(data, tag, args.out_dir, device)

    print("done.")


if __name__ == "__main__":
    main()
