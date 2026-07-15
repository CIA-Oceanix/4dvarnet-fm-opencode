#!/usr/bin/env python3
"""SW diagnostics — test forcing at different wavenumbers.

For a zonal jet U(y), barotropic instability growth rate ~ U_max / L_jet.
The single-mode sin(2*pi*y/L) creates a jet with L_jet = L/2 = 128 dx at
256 resolution — the growth rate is too slow for eddies to develop.

Here we test higher-wavenumber forcing: sin(2*n*pi*y/L) for various n.
Each creates a jet with L_jet = L/(2n), which extracts energy faster.

All runs: 256x256, tau0=0.01, friction=0.08 (moderate balance).
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIGS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

CMAP = "RdBu_r"

NX = 256
NY = 256
DT = 0.05
F_COR = 0.1
G1, G2 = 0.5, 2.0
COUPLING = 0.01
FRICTION = 0.08
VISCOSITY = 0.001
TAU0 = 0.01

RD1 = np.sqrt(G1) / F_COR
RD2 = np.sqrt(G2) / F_COR

SPINUP = 6000
NUM = 3000


def build_dynamics(n_modes: int):
    from models.shallow_water_dynamics import ShallowWaterDynamics

    dyn = ShallowWaterDynamics(
        Nx=NX, Ny=NY, dt=DT,
        tau0=TAU0, f_cor=F_COR,
        g1=G1, g2=G2,
        coupling=COUPLING, friction=FRICTION,
        viscosity=VISCOSITY,
    )

    y_coords = torch.arange(NY, dtype=torch.float32) * dyn.dy
    pattern = torch.zeros(NY, dtype=torch.float32)
    for n in range(1, n_modes + 1):
        pattern += (1.0 / n) * torch.sin(2.0 * n * torch.pi * y_coords / dyn.Ly)
    pattern = pattern / pattern.abs().max()
    dyn.wind_pattern = pattern.unsqueeze(0).expand(NX, NY).reshape(-1).contiguous()

    return dyn


def generate_trajectory(dyn, seed=42):
    print(f"  {NUM} steps (spinup={SPINUP}) ...", end=" ", flush=True)
    traj, forcing = dyn.generate_full_trajectory(
        num_steps=NUM, seed=seed, spinup_steps=SPINUP,
    )
    ok = np.all(np.isfinite(traj.numpy()))
    print(f"done. Finite={ok}")
    return traj.numpy() if ok else None, forcing


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
    return np.arange(nbins)[mask] / Nx, power[mask] / count[mask]


def evaluate(traj, tag):
    Nxy = NX * NY

    # Final snapshot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    titles = ["h₁ ocean", "u₁ ocean", "v₁ ocean", "h₂ atmos", "u₂ atmos", "v₂ atmos"]
    for ci, (ax, ttl) in enumerate(zip(axes.flat, titles)):
        fld = traj[-1, ci*Nxy:(ci+1)*Nxy].reshape(NX, NY)
        vm = max(abs(fld.min()), abs(fld.max())) + 1e-10
        im = ax.imshow(fld, cmap=CMAP, norm=TwoSlopeNorm(vcenter=0, vmin=-vm, vmax=vm), origin="lower")
        ax.set_title(ttl, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    plt.suptitle(f"Final snapshot — {tag}", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, f"sw_wind_{tag}_snap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Zonal mean profile & spectrum
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    u1_zonal = traj[-1, Nxy:2*Nxy].reshape(NX, NY).mean(axis=1)
    u2_zonal = traj[-1, 4*Nxy:5*Nxy].reshape(NX, NY).mean(axis=1)
    axes[0].plot(u1_zonal, label="ocean u₁")
    axes[0].plot(u2_zonal, label="atmos u₂")
    axes[0].set_xlabel("y"); axes[0].set_ylabel("u")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Zonal mean u")

    for ci, ax, n, c in [(0, axes[1], "h₁", "C0"), (3, axes[1], "h₂", "C1")]:
        ps_all = [azimuthal_spectrum_2d(traj[t, ci*Nxy:(ci+1)*Nxy].reshape(NX, NY))[1]
                  for t in range(NUM - 50, NUM)]
        k = azimuthal_spectrum_2d(traj[-1, ci*Nxy:(ci+1)*Nxy].reshape(NX, NY))[0]
        axes[1].loglog(k, np.mean(ps_all, axis=0), c=c, label=n)
    axes[1].axvline(RD1/NX, c="C0", ls=":", label=f"Rd₁={RD1:.0f}dx")
    axes[1].axvline(RD2/NX, c="C1", ls=":", label=f"Rd₂={RD2:.0f}dx")
    axes[1].set_xlabel("k (cyc/dx)"); axes[1].grid(True, alpha=0.3)
    axes[1].legend(); axes[1].set_title("Spectrum")

    # 1D vs 2D energy ratio vs k_y
    kx_fft = np.fft.fftfreq(NX)
    for ci, ax in [(0, axes[2])]:
        fld = traj[-1, ci*Nxy:(ci+1)*Nxy].reshape(NX, NY)
        ps = np.abs(np.fft.fft2(fld))**2
        # Energy in kx=0 (zonal stripes) vs total
        energy_kx0 = ps[0, :].sum()
        energy_total = ps.sum()
        ax.set_title(f"kx=0 energy: {100*energy_kx0/energy_total:.0f}%")

    plt.suptitle(f"Profile & spectra — {tag}", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, f"sw_wind_{tag}_profile.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Stats
    print(f"  Stats (last 500 steps):")
    for ci, name in enumerate(["h1", "u1", "v1", "h2", "u2", "v2"]):
        fld = traj[-500:, ci*Nxy:(ci+1)*Nxy]
        print(f"    {name}: mu={fld.mean():.4f} sigma={fld.std():.4f}")

    print(f"  Dominant scale (spectrum peak, k>=3):")
    for ci, name in enumerate(["h1", "u1", "v1", "h2", "u2", "v2"]):
        k, p = azimuthal_spectrum_2d(traj[-1, ci*Nxy:(ci+1)*Nxy].reshape(NX, NY))
        if len(p) > 3:
            idx = np.argmax(p[3:]) + 3
            print(f"    {name}: {1.0/k[idx]:.1f}dx k={k[idx]:.4f}")

    # 2D ratio
    n2d, n1d = 0, 0
    for t in range(NUM - 50, NUM):
        for ci in range(6):
            fld = traj[t, ci*Nxy:(ci+1)*Nxy].reshape(NX, NY)
            ps = np.abs(np.fft.fft2(fld))**2
            r = ps[0, :].sum() / ps.sum()
            n2d += 1 if r < 0.9 else 0
            n1d += 1 if r >= 0.9 else 0
    pct = 100 * n2d / (n2d + n1d) if (n2d+n1d) > 0 else 0
    print(f"  2D/1D: {n2d}/{n1d} ({pct:.0f}% 2D)")

    # EKE
    eke1 = 0.5 * (traj[-500:, Nxy:2*Nxy].var() + traj[-500:, 2*Nxy:3*Nxy].var())
    eke2 = 0.5 * (traj[-500:, 4*Nxy:5*Nxy].var() + traj[-500:, 5*Nxy:6*Nxy].var())
    print(f"  EKE₁={eke1:.4f} EKE₂={eke2:.4f}")

    if pct > 20 and max(traj[-1, Nxy:2*Nxy].std(), traj[-1, 4*Nxy:5*Nxy].std()) < 2.0:
        return "✓ GOOD"
    elif pct > 20:
        return "~ turbulent"
    else:
        return "✗ stripes"


def main():
    print("=" * 72)
    print("  SW wind-wavenumber sweep (256x256)")
    print(f"  tau0={TAU0}, friction={FRICTION}, dt={DT}")
    print(f"  Rd₁={RD1:.1f}dx, Rd₂={RD2:.1f}dx")
    print(f"  spinup={SPINUP} ({SPINUP*DT:.0f} tu), num={NUM} ({NUM*DT:.0f} tu)")
    print("=" * 72)

    results = []

    for n_modes in [1, 2, 4, 8, 16]:
        tag = f"n{n_modes}"
        L_jet = NX / (2 * n_modes)
        growth_rate = TAU0 / FRICTION / L_jet if L_jet > 0 else 0
        print(f"\n--- n={n_modes} (L_jet={L_jet:.0f}dx, γ~{growth_rate:.4f}) ---")
        dyn = build_dynamics(n_modes)
        traj = generate_trajectory(dyn)
        if traj is None:
            print("  SKIP: NaN")
            results.append((n_modes, "NaN"))
            continue

        verdict = evaluate(traj, tag)
        results.append((n_modes, verdict))

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    for n, v in results:
        print(f"  n={n:2d}  {v}")
    print("=" * 72)


if __name__ == "__main__":
    main()