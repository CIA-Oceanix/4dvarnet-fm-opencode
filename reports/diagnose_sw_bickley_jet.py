#!/usr/bin/env python3
"""Diagnose whether the Bickley-jet initial condition produces 2D eddies.

Tests at 64x64, 128x128, 256x256 and reports:
  - Dominant spatial scale vs Rossby deformation radius
  - 2D energy ratio (v^2 / (u^2 + v^2))
  - Layer-thickness positivity
  - Stability (no NaNs / blow-ups)
"""
import os
import sys
import time
import textwrap

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.shallow_water_dynamics import ShallowWaterDynamics

FIGS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

CMAP = "RdBu_r"

RESOLUTIONS = [64, 128, 256]
RES_OVERRIDES = {
    256: {"dt": 0.02, "friction": 0.005, "spinup_steps": 200, "num_steps": 400},
}
BASE_CFG = dict(
    dt=0.1,
    tau0=0.0,
    f_cor=0.1,
    g1=1.0,
    g2=4.0,
    coupling=0.01,
    friction=0.001,
    viscosity=1e-4,
    bickley_U=0.50,
    bickley_U2=0.30,
    bickley_H_ref=10.0,
    bickley_L_jet_frac=0.15,
    spinup_steps=300,
    num_steps=500,
)


def reshape_field(field_1d, Nx, Ny):
    return field_1d.reshape(Nx, Ny)


def azimuthal_spectrum_2d(field_2d):
    Nx, Ny = field_2d.shape
    f = np.fft.fft2(field_2d)
    ps = np.abs(f) ** 2
    kx = np.fft.fftfreq(Nx) * Nx
    ky = np.fft.fftfreq(Ny) * Ny
    kxx, kyy = np.meshgrid(kx, ky, indexing="ij")
    kr = np.sqrt(kxx**2 + kyy**2).astype(int)
    nbins = max(Nx, Ny) // 2
    power = np.zeros(nbins)
    count = np.zeros(nbins)
    for i in range(Nx):
        for j in range(Ny):
            idx = int(kr[i, j])
            if idx < nbins:
                power[idx] += ps[i, j]
                count[idx] += 1
    mask = count > 0
    k = np.arange(nbins)[mask]
    power_avg = power[mask] / count[mask]
    return k, power_avg


def compute_scales(dyn, traj, Nx, Ny):
    Nxy = Nx * Ny
    v1 = traj[:, 2 * Nxy : 3 * Nxy]
    v1_std = float(v1.std(axis=1).mean())

    u1 = traj[:, Nxy : 2 * Nxy]
    e2d = float((v1**2).mean() / ((u1**2).mean() + (v1**2).mean() + 1e-10))

    h1f = traj[-1, :Nxy].numpy().reshape(Nx, Ny)
    k, ps = azimuthal_spectrum_2d(h1f)
    start = 3
    if len(ps) > start:
        idx = np.argmax(ps[start:]) + start
        peak_k = k[idx]
        peak_scale = Nx / peak_k if peak_k > 0 else 0.0
    else:
        peak_k, peak_scale = 0, 0.0

    c1 = np.sqrt(dyn.g1 * BASE_CFG["bickley_H_ref"])
    c2 = np.sqrt(dyn.g2 * BASE_CFG["bickley_H_ref"])
    Rd1 = c1 / dyn.f_cor
    Rd2 = c2 / dyn.f_cor
    T_f = 2 * np.pi / dyn.f_cor

    return {
        "N": Nx,
        "v1_std": v1_std,
        "2D_ratio": e2d,
        "peak_k": int(peak_k),
        "peak_scale_dx": peak_scale,
        "Rd1": Rd1,
        "Rd2": Rd2,
        "T_f": T_f,
        "stable": True,
    }


def main():
    print("=" * 72)
    print("  Bickley jet eddy diagnostics")
    print("  Figures ->", FIGS_DIR)
    print("=" * 72)

    all_results = []

    for N in RESOLUTIONS:
        print(f"\n--- {N}x{N} ---")
        cfg = {**BASE_CFG, "Nx": N, "Ny": N}
        if N in RES_OVERRIDES:
            cfg.update(RES_OVERRIDES[N])
        dyn = ShallowWaterDynamics(
            Nx=N, Ny=N, dt=cfg["dt"], tau0=cfg["tau0"],
            f_cor=cfg["f_cor"], g1=cfg["g1"], g2=cfg["g2"],
            coupling=cfg["coupling"], friction=cfg["friction"],
            viscosity=cfg["viscosity"],
        )

        t0 = time.time()
        traj, _ = dyn.generate_full_trajectory(
            num_steps=cfg["num_steps"],
            seed=42,
            spinup_steps=cfg["spinup_steps"],
            bickley_jet=True,
            bickley_U=cfg["bickley_U"],
            bickley_U2=cfg["bickley_U2"],
            bickley_H_ref=cfg["bickley_H_ref"],
            bickley_L_jet_frac=cfg["bickley_L_jet_frac"],
        )
        elapsed = time.time() - t0
        print(f"  Generated {cfg['num_steps']} steps (spinup={cfg['spinup_steps']}) in {elapsed:.1f}s")

        stable = traj.isfinite().all().item()
        print(f"  Stable: {stable}")

        Nxy = N * N
        h1 = traj[:, :Nxy]
        h1_pos = h1.min() > 0
        print(f"  h1 positive: {h1_pos.item()} (range {h1.min().item():.3f} to {h1.max().item():.3f})")

        if not stable:
            all_results.append({"N": N, "stable": False})
            continue

        result = compute_scales(dyn, traj, N, Ny=N)
        all_results.append(result)

        Rd1_str = f"Rd1={result['Rd1']:.1f} dx"
        Rd2_str = f"Rd2={result['Rd2']:.1f} dx"
        print(f"  {Rd1_str}, {Rd2_str}")
        print(f"  Peak scale: {result['peak_scale_dx']:.1f} dx (k={result['peak_k']})")
        print(f"  Peak / Rd1: {result['peak_scale_dx'] / result['Rd1']:.2f}")
        print(f"  v1 std: {result['v1_std']:.4f}")
        print(f"  2D ratio: {result['2D_ratio']:.4f}")

        # --- snapshot ---
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        field_names = ["h1", "u1", "v1", "h2", "u2", "v2"]
        for idx, (ax, name) in enumerate(zip(axes.flat, field_names)):
            fld = traj[-1, idx * Nxy : (idx + 1) * Nxy].numpy().reshape(N, N)
            vmax = max(abs(fld.min()), abs(fld.max())) + 1e-10
            norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
            im = ax.imshow(fld, cmap=CMAP, norm=norm, origin="lower")
            ax.set_title(name, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
        plt.suptitle(f"Bickley jet — {N}x{N} (final step)", fontsize=13)
        plt.tight_layout()
        fname = f"bickley_{N}_snapshot.png"
        fig.savefig(os.path.join(FIGS_DIR, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")

        # --- spectrum ---
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for idx, (ax, name) in enumerate(zip(axes.flat, field_names)):
            ps_all = []
            for t in range(len(traj) - 50, len(traj)):
                field = traj[t, idx * Nxy : (idx + 1) * Nxy].numpy().reshape(N, N)
                k, p = azimuthal_spectrum_2d(field)
                ps_all.append(p)
            ps_mean = np.mean(ps_all, axis=0)
            ax.loglog(k / N, ps_mean, "-")
            ax.axvline(1.0 / result["Rd1"], color="C1", ls="--", lw=0.8, label=f"1/Rd1")
            ax.axvline(1.0 / result["Rd2"], color="C2", ls="--", lw=0.8, label=f"1/Rd2")
            ax.set_xlabel("k (cycles / grid unit)", fontsize=8)
            ax.set_ylabel("Power", fontsize=8)
            ax.set_title(f"{name} spectrum", fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, ls="--")
        plt.suptitle(f"Spatial spectra — {N}x{N}", fontsize=13)
        plt.tight_layout()
        fname_spectrum = f"bickley_{N}_spectrum.png"
        fig.savefig(os.path.join(FIGS_DIR, fname_spectrum), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname_spectrum}")

    # --- summary ---
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  {'N':>6}  {'Stable':>7}  {'v1_std':>7}  {'2D_ratio':>9}  "
          f"{'Peak(dx)':>9}  {'Rd1':>7}  {'Rd2':>7}  {'Peak/Rd1':>9}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*9}")
    for r in all_results:
        if r.get("stable", False):
            print(f"  {r['N']:>6}  {'OK':>7}  {r['v1_std']:>7.4f}  {r['2D_ratio']:>9.4f}  "
                  f"{r['peak_scale_dx']:>9.1f}  {r['Rd1']:>7.1f}  {r['Rd2']:>7.1f}  "
                  f"{r['peak_scale_dx']/r['Rd1']:>9.2f}")
        else:
            print(f"  {r['N']:>6}  {'FAIL':>7}  {'N/A':>7}  {'N/A':>9}  "
                  f"{'N/A':>9}  {'N/A':>7}  {'N/A':>7}  {'N/A':>9}")

    eddy_pass = all(r.get("stable", False) and r.get("2D_ratio", 0) > 0.1 and r.get("peak_scale_dx", 0) > 5
                    for r in all_results)
    if eddy_pass:
        print(f"\n  PASS: All resolutions produce 2D eddies at physical scales.")
    else:
        print(f"\n  NOTE: Some resolutions need tuning.")

    print(f"\n  Done. All figures in {FIGS_DIR}/")
    print("=" * 72)


if __name__ == "__main__":
    main()