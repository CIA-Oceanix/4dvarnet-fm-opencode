#!/usr/bin/env python3
"""SW at 256x256 with multi-mode wind forcing to produce physical eddies.

The single-mode sin(2*pi*y/L) forcing produces a jet that fills the domain,
which suppresses eddy formation at all resolutions.  Here we add a
higher-wavenumber component (sin(4*pi*y/L)) to produce a narrower jet that
is more susceptible to barotropic instability.

Tests: friction=0.05, tau0=0.05, 256x256.
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

# We create a custom dynamics subclass that adds a sin(4*pi*y/L) component
# to the wind pattern.

CUSTOM_WIND = True  # Use multi-mode wind
Nx, Ny = 256, 256
dt = 0.1
f_cor = 0.1
g1, g2 = 0.5, 2.0
coupling = 0.01
friction = 0.05
viscosity = 0.001
tau0 = 0.05

spinup_steps = 4000
num_steps = 2000

FIGS_TAG = "sw256"


def build_dynamics(with_2mode=True):
    from models.shallow_water_dynamics import ShallowWaterDynamics

    dyn = ShallowWaterDynamics(
        Nx=Nx, Ny=Ny, dt=dt,
        tau0=tau0, f_cor=f_cor,
        g1=g1, g2=g2,
        coupling=coupling, friction=friction,
        viscosity=viscosity,
    )

    if with_2mode:
        # Replace the wind pattern to include sin(4*pi*y/L) component
        y_coords = torch.arange(Ny, dtype=torch.float32) * dyn.dy
        wind_slow = torch.sin(2.0 * torch.pi * y_coords / dyn.Ly)
        wind_fast = torch.sin(4.0 * torch.pi * y_coords / dyn.Ly)
        wind_combined = (wind_slow + 0.7 * wind_fast.unsqueeze(0)) \
            .expand(Nx, Ny).reshape(-1).contiguous()
        # Renormalize so the peak amplitude is still ~1
        wind_combined = wind_combined / wind_combined.abs().max()
        dyn.wind_pattern = wind_combined

    return dyn


def generate_trajectory(dyn, seed=42):
    print(f"  Generating {num_steps} steps (spinup={spinup_steps}) ...")
    traj, forcing = dyn.generate_full_trajectory(
        num_steps=num_steps, seed=seed,
        spinup_steps=spinup_steps,
    )
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
    return np.arange(nbins)[mask] / Nx, power[mask] / count[mask]


def main():
    print("=" * 72)
    print("  SW 256x256 eddy diagnostics")
    print(f"  tau0={tau0}, friction={friction}")
    print(f"  wind: {'multi-mode (sin + 0.7*sin2)' if CUSTOM_WIND else 'single-mode sin'}")
    print(f"  spinup={spinup_steps}, num={num_steps}")
    print(f"  Rd1={np.sqrt(g1)/f_cor:.1f}dx, Rd2={np.sqrt(g2)/f_cor:.1f}dx")
    print("=" * 72)

    Nxy = Nx * Ny

    for use_2mode, label in [(False, "single-mode"), (True, "multi-mode")]:
        print(f"\n--- {label} ---")
        dyn = build_dynamics(with_2mode=use_2mode)
        traj, forcing = generate_trajectory(dyn)
        if not np.all(np.isfinite(traj)):
            print("  SKIP: NaN")
            continue

        tag = f"{FIGS_TAG}_{label.replace('-','_')}"

        # Final snapshot
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fields_list = [traj[-1, ci*Nxy:(ci+1)*Nxy].reshape(Nx, Ny) for ci in range(6)]
        titles = ["h₁ ocean", "u₁ ocean", "v₁ ocean", "h₂ atmos", "u₂ atmos", "v₂ atmos"]
        for ax, fld, ttl in zip(axes.flat, fields_list, titles):
            vm = max(abs(fld.min()), abs(fld.max())) + 1e-10
            im = ax.imshow(fld, cmap=CMAP, norm=TwoSlopeNorm(vcenter=0, vmin=-vm, vmax=vm), origin="lower")
            ax.set_title(ttl, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
        plt.suptitle(f"Final snapshot — {label} (256x256)", fontsize=12)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGS_DIR, f"{tag}_snap.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {tag}_snap.png")

        # Zonal mean profile
        u1_zonal = fields_list[1].mean(axis=1)
        u2_zonal = fields_list[4].mean(axis=1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(u1_zonal, label="ocean u₁")
        axes[0].plot(u2_zonal, label="atmos u₂")
        axes[0].set_xlabel("y (grid)"); axes[0].set_ylabel("Zonal mean u")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[0].set_title("Zonal mean velocity profile")

        # Spectrum overlay
        for ci, ax, n in [(0, axes[1], "h₁"), (3, axes[1], "h₂")]:
            ps_all = []
            for t in range(num_steps - 50, num_steps):
                field = traj[t, ci*Nxy:(ci+1)*Nxy].reshape(Nx, Ny)
                k, p = azimuthal_spectrum_2d(field)
                ps_all.append(p)
            axes[1].loglog(k, np.mean(ps_all, axis=0), label=n)
        axes[1].axvline(np.sqrt(g1)/f_cor / Nx, color="C0", ls=":", label=f"Rd₁={np.sqrt(g1)/f_cor:.0f}dx")
        axes[1].axvline(np.sqrt(g2)/f_cor / Nx, color="C1", ls=":", label=f"Rd₂={np.sqrt(g2)/f_cor:.0f}dx")
        axes[1].set_xlabel("k (cyc/dx)"); axes[1].set_ylabel("Power")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
        axes[1].set_title("Azimuthal power spectrum")
        plt.suptitle(f"Profile & spectra — {label}", fontsize=12)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGS_DIR, f"{tag}_profile.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {tag}_profile.png")

        # Scales table
        print(f"\n  State statistics (spatial std, last 500 steps):")
        for ci, name in enumerate(["h1", "u1", "v1", "h2", "u2", "v2"]):
            fld = traj[-500:, ci*Nxy:(ci+1)*Nxy]
            print(f"    {name:8s}:  mu = {fld.mean():.4f},  sigma = {fld.std():.4f}")

        print(f"  Dominant scale (spectrum peak, k>=3):")
        for ci, name in enumerate(["h1", "u1", "v1", "h2", "u2", "v2"]):
            field = traj[-1, ci*Nxy:(ci+1)*Nxy].reshape(Nx, Ny)
            k, p = azimuthal_spectrum_2d(field)
            start = 3
            if len(p) > start:
                idx = np.argmax(p[start:]) + start
                print(f"    {name:8s}:  {1.0/k[idx]:.1f} dx  (k={k[idx]:.4f})")

        # 2D vs 1D ratio
        n_kx0, n_2d = 0, 0
        for t in range(num_steps - 50, num_steps):
            for ci in range(6):
                field = traj[t, ci*Nxy:(ci+1)*Nxy].reshape(Nx, Ny)
                ps = np.abs(np.fft.fft2(field))**2
                total = ps.sum()
                if total > 0:
                    n_2d += 1 if ps[0, :].sum() / total < 0.9 else 0
                    n_kx0 += 1 if ps[0, :].sum() / total >= 0.9 else 0
        pct_2d = 100 * n_2d / (n_2d + n_kx0) if (n_2d + n_kx0) > 0 else 0
        print(f"  2D/1D: {n_2d}/{n_kx0} ({pct_2d:.0f}% 2D)")

        if pct_2d > 20:
            print(f"  ✓ Eddies detected!")
        else:
            print(f"  ✗ Still 1D stripes")

    print(f"\n{'='*72}")
    print(f"  Done. Figures in {FIGS_DIR}/")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()