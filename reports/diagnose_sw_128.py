#!/usr/bin/env python3
"""Shallow water diagnostics at 128x128 with stronger wind forcing.

Tests multiple friction/wind combinations to find a regime that produces
2D eddies via barotropic instability of the forced zonal jet.
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIGS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

CMAP = "RdBu_r"

BASE_CFG = dict(
    Nx=128, Ny=128, dt=0.1,
    f_cor=0.1,
    g1=0.5, g2=2.0,
    coupling=0.01, viscosity=0.001,
)

CONFIGS = [
    dict(label="128x128 r=0.05 t0=0.05", friction=0.05, tau0=0.05, spinup=6000, num=3000),
    dict(label="128x128 r=0.02 t0=0.02", friction=0.02, tau0=0.02, spinup=6000, num=3000),
    dict(label="128x128 r=0.01 t0=0.02", friction=0.01, tau0=0.02, spinup=6000, num=3000),
]


def build_dynamics(cfg):
    from models.shallow_water_dynamics import ShallowWaterDynamics
    return ShallowWaterDynamics(
        Nx=cfg["Nx"], Ny=cfg["Ny"], dt=cfg["dt"],
        tau0=cfg["tau0"], f_cor=cfg["f_cor"],
        g1=cfg["g1"], g2=cfg["g2"],
        coupling=cfg["coupling"], friction=cfg["friction"],
        viscosity=cfg["viscosity"],
    )


def generate_trajectory(dyn, cfg, seed=42):
    nsteps = cfg["num"]
    sp = cfg["spinup"]
    print(f"  {nsteps} steps (spinup={sp}) ...", end=" ", flush=True)
    traj, forcing = dyn.generate_full_trajectory(
        num_steps=nsteps, seed=seed, spinup_steps=sp,
    )
    print(f"done. Finite={np.all(np.isfinite(traj.numpy()))}")
    return traj.numpy(), forcing.numpy()


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


def plot_summary_fig(traj, cfg, tag):
    Nx, Ny = cfg["Nx"], cfg["Ny"]
    Nxy = Nx * Ny
    span = cfg["num"]

    # Final snapshot
    h1 = traj[-1, 0:Nxy].reshape(Nx, Ny)
    u1 = traj[-1, Nxy:2*Nxy].reshape(Nx, Ny)
    h2 = traj[-1, 3*Nxy:4*Nxy].reshape(Nx, Ny)
    u2 = traj[-1, 4*Nxy:5*Nxy].reshape(Nx, Ny)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fields = [h1, u1, traj[-1, 2*Nxy:3*Nxy].reshape(Nx, Ny),
              h2, u2, traj[-1, 5*Nxy:6*Nxy].reshape(Nx, Ny)]
    titles = ["h₁ ocean", "u₁ ocean", "v₁ ocean", "h₂ atmos", "u₂ atmos", "v₂ atmos"]
    for ax, fld, ttl in zip(axes.flat, fields, titles):
        vm = max(abs(fld.min()), abs(fld.max())) + 1e-10
        im = ax.imshow(fld, cmap=CMAP, norm=TwoSlopeNorm(vcenter=0, vmin=-vm, vmax=vm), origin="lower")
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    plt.suptitle(f"Final snapshot — {cfg['label']}", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, f"sw128_{tag}_snap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Zonal mean profile
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    u1_zonal = u1.mean(axis=1)
    u2_zonal = u2.mean(axis=1)
    y = np.arange(Ny)
    axes[0].plot(u1_zonal, y, label="ocean u₁")
    axes[0].plot(u2_zonal, y, label="atmos u₂")
    axes[0].set_xlabel("Zonal mean u"); axes[0].set_ylabel("y (grid)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Power spectrum (mean over last 50 steps)
    for idx, ax, name in [(0, axes[1], "h₁"), (3, axes[2], "h₂")]:
        ps_all = []
        for t in range(span - 50, span):
            field = traj[t, idx*Nxy:(idx+1)*Nxy].reshape(Nx, Ny)
            k, p = azimuthal_spectrum_2d(field)
            ps_all.append(p)
        ps_mean = np.mean(ps_all, axis=0)
        ax.loglog(k / Nx, ps_mean)
        ax.set_title(f"Spectrum {name}", fontsize=10)
        ax.set_xlabel("k (cyc/unit)"); ax.grid(True, alpha=0.3)
    plt.suptitle(f"Zonal mean & spectra — {cfg['label']}", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, f"sw128_{tag}_profile.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved sw128_{tag}_snap.png, sw128_{tag}_profile.png")


def print_table(traj, cfg, tag):
    Nx, Ny = cfg["Nx"], cfg["Ny"]
    Nxy = Nx * Ny
    span = cfg["num"]
    f = cfg["f_cor"]
    g1, g2 = cfg["g1"], cfg["g2"]

    c1, c2 = np.sqrt(g1), np.sqrt(g2)
    Rd1, Rd2 = c1/f, c2/f
    T_f = 2*np.pi/f

    u_balance = cfg["tau0"] / cfg["friction"]

    print(f"\n{'='*60}")
    print(f"  {cfg['label']}")
    print(f"{'='*60}")
    print(f"  g1={g1} g2={g2} f={f} dt={cfg['dt']}")
    print(f"  c1={c1:.3f} c2={c2:.3f}")
    print(f"  Rd1={Rd1:.1f}dx Rd2={Rd2:.1f}dx")
    print(f"  Rd1/L={Rd1/Nx:.3f} Rd2/L={Rd2/Nx:.3f}")
    print(f"  T_f={T_f:.1f} ({T_f/cfg['dt']:.0f} steps)")
    print(f"  u_balance (tau0/r) = {u_balance:.3f}")
    print(f"  u_peak / c1 = {u_balance/c1:.3f}")
    print()
    print(f"  State statistics (spatial std, last 500 steps):")
    for ci, name in enumerate(["h1", "u1", "v1", "h2", "u2", "v2"]):
        fld = traj[-500:, ci*Nxy:(ci+1)*Nxy]
        print(f"    {name:8s}:  mu = {fld.mean():.4f},  sigma = {fld.std():.4f}")

    # Dominant scale
    print(f"\n  Dominant spatial scale (spectrum peak, k>=3):")
    for ci, name in enumerate(["h1", "u1", "v1", "h2", "u2", "v2"]):
        field = traj[-1, ci*Nxy:(ci+1)*Nxy].reshape(Nx, Ny)
        k, p = azimuthal_spectrum_2d(field)
        k_phys = k / Nx
        start = 3
        if len(p) <= start:
            print(f"    {name:8s}:  N/A")
            continue
        idx = np.argmax(p[start:]) + start
        if k_phys[idx] > 0:
            scale = 1.0 / k_phys[idx]
            print(f"    {name:8s}:  {scale:.1f} dx  (k={k_phys[idx]:.3f})")

    # Eddy kinetic energy
    u1_e = traj[-500:, Nxy:2*Nxy]
    v1_e = traj[-500:, 2*Nxy:3*Nxy]
    u2_e = traj[-500:, 4*Nxy:5*Nxy]
    v2_e = traj[-500:, 5*Nxy:6*Nxy]
    eke1 = 0.5 * (u1_e.var() + v1_e.var())
    eke2 = 0.5 * (u2_e.var() + v2_e.var())
    print(f"\n  Eddy kinetic energy (from variance):")
    print(f"    EKE₁ (ocean): {eke1:.4f}")
    print(f"    EKE₂ (atmos): {eke2:.4f}")
    print(f"    Ratio EKE₂/EKE₁: {eke2/eke1:.2f}")


def dominant_2d_vs_1d_ratio(traj, cfg):
    """Check if energy is in 2D structures (k_x > 0, k_y > 0) vs 1D stripes (k_x=0)."""
    Nx, Ny = cfg["Nx"], cfg["Ny"]
    Nxy = Nx * Ny
    span = cfg["num"]

    n_kx0 = 0
    n_2d = 0
    for t in range(span - 50, span):
        for ci in range(6):
            field = traj[t, ci*Nxy:(ci+1)*Nxy].reshape(Nx, Ny)
            f = np.fft.fft2(field)
            ps = np.abs(f)**2
            # Energy at kx=0 (zonal stripes)
            kx0_energy = ps[0, :].sum()
            total_energy = ps.sum()
            if total_energy > 0:
                ratio = kx0_energy / total_energy
                if ratio > 0.9:
                    n_kx0 += 1
                else:
                    n_2d += 1
    return n_kx0, n_2d


def main():
    print("=" * 72)
    print("  SW 128x128 diagnostics — parameter sweep")
    print("=" * 72)

    results = []

    for cfg in CONFIGS:
        full_cfg = {**BASE_CFG, **cfg}
        print(f"\n--- {cfg['label']} ---")
        try:
            dyn = build_dynamics(full_cfg)
            traj, forcing = generate_trajectory(dyn, full_cfg)
            if not np.all(np.isfinite(traj)):
                print("  SKIP: NaN in trajectory")
                continue

            tag = cfg['label'].replace(" ", "_").replace(".", "_")
            plot_summary_fig(traj, full_cfg, tag)
            print_table(traj, full_cfg, tag)

            n1d, n2d = dominant_2d_vs_1d_ratio(traj, full_cfg)
            total = n1d + n2d
            pct_2d = 100 * n2d / total if total > 0 else 0
            print(f"\n  2D structure ratio: {n2d}/{total} ({pct_2d:.0f}%)")
            print(f"  1D structure ratio: {n1d}/{total} ({100-pct_2d:.0f}%)")

            if pct_2d > 10:
                verdict = "PROMISING"
            else:
                verdict = "1D stripes"
            print(f"  Verdict: {verdict}")
            results.append((cfg['label'], verdict, pct_2d))

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    for label, verdict, pct in results:
        print(f"  {label:30s}  {verdict:15s}  ({pct:.0f}% 2D)")
    print("=" * 72)


if __name__ == "__main__":
    main()