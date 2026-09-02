import argparse
import json
import os
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.qg import QGConfig, make_qg_s0_s1_datasets
from models.qg_dynamics import QGDynamics
from models.qg_interp import spectral_resize_2d

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGDIR = os.path.join(BASE, "reports", "qg", "outputs", "figs")
os.makedirs(FIGDIR, exist_ok=True)


def _resize_state(state, dyn, nlayers, src_n, dst_n, device):
    """Spectral down/upsample a flattened (nlayers*src_n*src_n) state to dst grid."""
    lead = state.shape[:-1]
    x = state.reshape(*lead, nlayers, src_n, src_n)
    y = spectral_resize_2d(x, dst_n, dst_n, device)
    return y.reshape(*lead, nlayers * dst_n * dst_n)


def _build_dyn(cfg, window, device):
    da_params = window["da_params"]
    nx_da = int(window.get("da_nx") or cfg.nx)
    common = {
        "nx": nx_da, "L": cfg.L, "dt": cfg.dt, "beta": da_params["beta"],
        "rd": da_params["rd"], "U1": da_params["U1"], "rek": da_params["rek"],
        "filterfac": cfg.filterfac,
        "wind_amp": window["wind_amp"],
        "wind_sigma": cfg.wind_sigma,
    }
    return QGDynamics(**common, delta=cfg.delta,
                      U2=da_params.get("U2", cfg.U2)).to(device)


def _free_divergence(cfg, dyn, window):
    """Free forecast divergence of the DA model from an exact truth IC.

    For a cross-resolution DA model (S1) the truth IC is spectrally downsampled
    to the DA grid before the roll and the rolled forecast is upsampled back to
    the truth grid for comparison, so the divergence measures the combined
    resolution+param-bias+wind error over the window.
    """
    truth = window["true_state"].to(dyn.device)
    ws = window["wind_state_corrupted"].to(dyn.device)
    ic = truth[0]
    nlayers = dyn.state_dim // (dyn.ny * dyn.nx)
    if dyn.ny != cfg.ny:
        ic = _resize_state(ic, dyn, nlayers, cfg.ny, dyn.ny, dyn.device)
    roll = dyn.rollout_trajectory(ic, cfg.num_steps - 1, wind_state=ws)
    if dyn.ny != cfg.ny:
        roll = _resize_state(roll, dyn, nlayers, dyn.ny, cfg.ny, dyn.device)
    diff = (roll - truth).norm(dim=-1)
    ref = truth.norm(dim=-1)
    return (diff[1:] / ref[1:].clamp_min(1e-12)).mean().item()


def _ke_per_window(cfg, dyn, window):
    vals = []
    truth = window["true_state"].to(dyn.device)
    for k in range(0, cfg.num_steps, 12):
        state = dyn._flatten(truth[k:k + 1]).squeeze(0)
        vals.append(dyn.kinetic_energy(state).item())
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--num-windows", type=int, default=5)
    ap.add_argument("--window-days", type=float, default=60.0)
    ap.add_argument("--spinup-years", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    cfg = QGConfig(nx=args.nx, window_days=args.window_days,
                   spinup_years=args.spinup_years, num_windows=args.num_windows,
                   obs_geometry="alongtrack", seed=7)
    ds = make_qg_s0_s1_datasets(cfg)

    w = ds["test_s0"][0]
    T, ny, nx = cfg.num_steps, cfg.ny, cfg.nx

    summary = {
        "nx": nx,
        "num_windows": len(ds["test_s0"]),
        "window_days": args.window_days,
        "passes_per_window": int(w["obs_mask"].sum()),
        "obs_pts_per_pass": ny,
        "obs_total_per_window": int(w["obs_mask"].sum()) * ny,
        "obs_dense_total": T * ny * nx,
        "coverage_frac": float(w["obs_mask"].sum() * ny) / (T * ny * nx),
        "track_repeat_days": cfg.track_repeat_days,
        "cross_track_spacing_km": cfg.track_advance_pts * (cfg.L / cfg.nx) / 1000.0,
        "obs_noise_frac": cfg.obs_noise_std_frac,
        "param_range": cfg.param_range,
        "s1_param_bias": cfg.s1_param_bias,
        "wind_levels": {},
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    psi0 = w["target_state_psi"][0].reshape(ny, nx).numpy()
    im = axes[0, 0].imshow(psi0, origin="lower", cmap="RdBu_r",
                           aspect="auto", vmin=-1e-3, vmax=1e-3)
    fig.colorbar(im, ax=axes[0, 0], shrink=0.85)
    axes[0, 0].set_title("Upper-layer psi1 snapshot (t=0)")
    tracks = w["track_x_index"][w["obs_mask"]].tolist()
    for xv in tracks:
        axes[0, 0].axvline(xv + 0.5, color="k", lw=0.8, alpha=0.6)
    axes[0, 0].set_xlabel("x index"); axes[0, 0].set_ylabel("y index")

    obs = w["obs"].numpy()
    im = axes[0, 1].imshow(obs, origin="lower", aspect="auto", cmap="RdBu_r")
    fig.colorbar(im, ax=axes[0, 1], shrink=0.85)
    axes[0, 1].set_title("Along-track obs Hovmöller (time vs track y)")
    axes[0, 1].set_xlabel("along-track y"); axes[0, 1].set_ylabel("time step")

    yy, xx = np.mgrid[0:ny, 0:nx]
    for t in w["obs_mask"].nonzero(as_tuple=False).flatten().tolist():
        xv = int(w["track_x_index"][t])
        axes[1, 0].plot(xx[:, xv], yy[:, xv], ".", color="tab:blue",
                        ms=1.5, alpha=0.5)
    axes[1, 0].set_title("QGS01 window coverage (observed columns)")
    axes[1, 0].set_xlim(-1, nx); axes[1, 0].set_ylim(-1, ny)
    axes[1, 0].set_xlabel("x index"); axes[1, 0].set_ylabel("y index")

    kes = []
    levels = []
    rols = {"test_s0": [], "test_s1": []}
    for k in ("test_s0", "test_s1"):
        d = ds[k]
        for i in range(len(d)):
            dyn = _build_dyn(cfg, d[i], device)
            rols[k].append(_free_divergence(cfg, dyn, d[i]))
        if k == "test_s0":
            for i in range(len(d)):
                dyn = _build_dyn(cfg, d[i], device)
                kes.append(_ke_per_window(cfg, dyn, d[i]))
                levels.append(d[i]["wind_amp"])

    max_s0 = max(rols["test_s0"])
    min_s1 = min(rols["test_s1"])
    if max_s0 >= min_s1:
        print(f"WARNING: S0 not well-separated (max_s0={max_s0:.3e} "
              f"min_s1={min_s1:.3e})")
    else:
        print(f"S0 max divergence {max_s0:.3e} < S1 min {min_s1:.3e} "
              f"(separation {min_s1 / max(max_s0, 1e-12):.1f}x)")

    axes[1, 1].bar(range(len(kes)), kes, tick_label=[f"{l:.1e}" for l in levels])
    axes[1, 1].set_title("Window KE by wind_amp level")
    axes[1, 1].set_xlabel("wind_amp"); axes[1, 1].set_ylabel("KE")
    for ax in axes.flat:
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(FIGDIR, "qg_alongtrack_calibration.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Saved {fig_path}")
    plt.close(fig)

    summary["free_divergence"] = {
        "test_s0": rols["test_s0"],
        "test_s1": rols["test_s1"],
    }
    for i in range(len(ds["test_s0"])):
        summary["wind_levels"][str(ds["test_s0"][i]["wind_amp"])] = {
            "ke": round(kes[i], 6) if i < len(kes) else None,
        }

    json_path = os.path.join(FIGDIR, "qg_alongtrack_calibration.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {json_path}")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("wind_levels",)}, indent=2))


if __name__ == "__main__":
    main()
