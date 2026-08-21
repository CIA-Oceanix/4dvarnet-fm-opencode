#!/usr/bin/env python3
"""Trajectory-quality diagnostics for the two-layer rotating shallow water model.

Produces a complete quality pack into ``--outdir`` (default
``reports/outputs/sw_quality``):

- snapshots.png   : 2x3 field snapshots (h1,u1,v1 / h2,u2,v2) at 4 times
- vorticity.png   : relative vorticity zeta = dv/dx - du/dy for both layers
- spectra.png     : azimuthally-averaged kinetic-energy spectra + Rd markers
- hovmoeller.png  : Hovmoeller diagrams (time vs x) for h1 and h2
- stability.png   : long-rollout stability diagnostics vs time
- animation_h1.gif / animation_zeta1.gif / animation_h2.gif
- sw_quality_report.md : embeds images + physical characteristic scales

Usage:
    python reports/diagnose_sw_quality.py [--Nx 64] [--Ny 64] [--steps 5000]
        [--spinup 1000] [--seed 42] [--frames 150] [--device cpu] [--outdir ...]
        [--quick]
"""
import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = os.path.join(os.path.dirname(__file__), "..")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.shallow_water_dynamics import ShallowWaterDynamics  # noqa: E402

CMAP = "RdBu_r"

STATE_NAMES = ["h1", "u1", "v1", "h2", "u2", "v2"]
BAROTROPIC = {"h1": 0, "u1": 1, "v1": 2, "h2": 3, "u2": 4, "v2": 5}

# Bickley-jet reference depth passed to generate_full_trajectory, set to the
# model's actual resting layer depth (~1.0). The dynamics' thickness clip is
# [0.1, 3.0] and _derivative documents true h1 staying within [0.86, 1.14], so
# an initial depth of 10.0 would be instantly crushed to the 3.0 clip bound,
# freezing the jet and stalling instability. H=1.0 gives the intended
# scale separation: Rd1~7 dx, Rd2~14 dx.
H_REF = 1.0


def _device_tensor(t, device):
    return t.to(device) if device is not None else t


def build_dynamics(Nx, Ny, device):
    return ShallowWaterDynamics(
        Nx=Nx, Ny=Ny, dt=0.1, tau0=0.0, f_cor=0.1,
        g1=0.5, g2=2.0, coupling=0.01, friction=0.001, viscosity=1e-4,
    ).to(device)


def field_arrays(traj, Nx, Ny):
    """Return dict of numpy (T, Nx, Ny) fields for each state component."""
    Nxy = Nx * Ny
    out = {}
    a = traj.reshape(-1, 6 * Nxy)
    for name in STATE_NAMES:
        out[name] = a[:, BAROTROPIC[name] * Nxy:(BAROTROPIC[name] + 1) * Nxy] \
            .reshape(-1, Nx, Ny)
    return out


def vorticity(u, v, Nx, Ny):
    """dv/dx - du/dy via central differences (numpy, periodic)."""
    du_dy = (np.roll(u, -1, axis=-1) - np.roll(u, 1, axis=-1)) / 2.0
    dv_dx = (np.roll(v, -1, axis=-2) - np.roll(v, 1, axis=-2)) / 2.0
    return dv_dx - du_dy


def azimuthal_spectrum_2d(field_2d):
    """Azimuthally averaged power spectrum of a 2D periodic field."""
    Nx, Ny = field_2d.shape
    ps = np.abs(np.fft.fft2(field_2d)) ** 2
    kx = np.fft.fftfreq(Nx) * Nx
    ky = np.fft.fftfreq(Ny) * Ny
    kxx, kyy = np.meshgrid(kx, ky, indexing="ij")
    kr = np.sqrt(kxx ** 2 + kyy ** 2).astype(int)
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
    return k, power[mask] / count[mask]


def characteristic_scales(f_cor=0.1, g1=0.5, g2=2.0, H=H_REF):
    c1 = np.sqrt(g1 * H)
    c2 = np.sqrt(g2 * H)
    return {
        "c1": c1, "c2": c2,
        "Rd1": c1 / f_cor, "Rd2": c2 / f_cor,
        "T_f": 2.0 * np.pi / f_cor,
    }


def pick_times(n, k=4):
    q = np.linspace(0, 1, k)
    return [max(0, int(np.round(i * (n - 1)))) for i in q]


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def plot_snapshots(flds, times, outdir, Nx, Ny):
    rows = ["h1", "u1", "v1", "h2", "u2", "v2"]
    names = [r"$h_1$", r"$u_1$", r"$v_1$", r"$h_2$", r"$u_2$", r"$v_2$"]
    fig, axes = plt.subplots(len(times), 6, figsize=(18, 4 * len(times)))
    axes = np.atleast_2d(axes)
    for r, t in enumerate(times):
        for c, name in enumerate(rows):
            field = flds[name][t]
            vmax = max(abs(field.min()), abs(field.max()))
            norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
            im = axes[r, c].imshow(field, cmap=CMAP, norm=norm, origin="lower")
            if r == 0:
                axes[r, c].set_title(names[c], fontsize=11)
            if c == 0:
                axes[r, c].set_ylabel(f"t={t}", fontsize=10)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            plt.colorbar(im, ax=axes[r, c], fraction=0.046, pad=0.03, shrink=0.8)
    fig.suptitle("Two-layer SW snapshots (Bickley jet, tau0=0)", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "snapshots.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_vorticity(flds, times, outdir, Nx, Ny):
    fig, axes = plt.subplots(len(times), 2, figsize=(10, 4 * len(times)))
    axes = np.atleast_2d(axes)
    for r, t in enumerate(times):
        z1 = vorticity(flds["u1"][t], flds["v1"][t], Nx, Ny)
        z2 = vorticity(flds["u2"][t], flds["v2"][t], Nx, Ny)
        for c, (z, name) in enumerate([(z1, "layer 1"), (z2, "layer 2")]):
            vmax = max(abs(z.min()), abs(z.max()))
            norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
            im = axes[r, c].imshow(z, cmap=CMAP, norm=norm, origin="lower")
            if r == 0:
                axes[r, c].set_title(f"relative vorticity {name} "
                                     r"$\zeta = \partial_x v - \partial_y u$",
                                     fontsize=10)
            if c == 0:
                axes[r, c].set_ylabel(f"t={t}", fontsize=10)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            plt.colorbar(im, ax=axes[r, c], fraction=0.046, pad=0.03, shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "vorticity.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spectra(flds, Nx, Ny, outdir, scales):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (L, u, v) in zip(axes,
                             [("layer 1 (ocean)", "u1", "v1"),
                              ("layer 2 (atmos)", "u2", "v2")]):
        eke = []
        k_arr = None
        half = flds[u].shape[0] // 2
        for t in range(half, flds[u].shape[0]):
            k_arr, pu = azimuthal_spectrum_2d(flds[u][t])
            _, pv = azimuthal_spectrum_2d(flds[v][t])
            eke.append(0.5 * (pu + pv))
        eke_mean = np.mean(eke, axis=0)
        k_phys = k_arr / Nx
        ax.loglog(k_phys, eke_mean, "-", label=f"KE {L}")
        ax.axvline(1.0 / scales["Rd1"] / Nx, color="C1", ls="--", lw=1.0,
                   label=r"$1/Rd_1$")
        ax.axvline(1.0 / scales["Rd2"] / Nx, color="C2", ls="--", lw=1.0,
                   label=r"$1/Rd_2$")
        ax.axvline(0.5, color="k", ls=":", lw=1.0, label="dx Nyquist")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, ls="--")
        ax.set_xlabel("k (cycles / grid unit)", fontsize=9)
        ax.set_ylabel(r"KE spectrum $E(k)$", fontsize=9)
    plt.suptitle("Azimuthally averaged kinetic-energy spectra", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "spectra.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hovmoeller(flds, Nx, Ny, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    T = flds["h1"].shape[0]
    for ax, name in zip(axes, ["h1", "h2"]):
        data = flds[name]
        y0 = Ny // 2
        band = data[:, :, max(0, y0 - 3):min(Ny, y0 + 4)].mean(axis=-1)
        vmax = max(abs(band.min()), abs(band.max()))
        norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
        im = ax.imshow(band, aspect="auto", cmap=CMAP, norm=norm,
                       origin="lower", extent=[0, Nx, T - 0.5, -0.5])
        ax.set_xlabel("x (grid)", fontsize=9)
        ax.set_ylabel("time step", fontsize=9)
        ax.set_title(f"Hovmöller {name} (y-band around jet)", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "hovmoeller.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stability(traj, Nx, Ny, outdir):
    np.seterr(all="ignore")
    a = traj.reshape(-1, 6 * Nx * Ny)
    Nxy = Nx * Ny
    h1 = a[:, 0:Nxy]
    h2 = a[:, 3 * Nxy:4 * Nxy]
    u1 = a[:, Nxy:2 * Nxy]
    v1 = a[:, 2 * Nxy:3 * Nxy]
    u2 = a[:, 4 * Nxy:5 * Nxy]
    v2 = a[:, 5 * Nxy:6 * Nxy]
    ke1 = 0.5 * (u1 ** 2 + v1 ** 2).mean(axis=1)
    ke2 = 0.5 * (u2 ** 2 + v2 ** 2).mean(axis=1)
    mass1 = h1.mean(axis=1)
    mass2 = h2.mean(axis=1)
    total = mass1 + mass2
    t = np.arange(a.shape[0])

    finite = np.isfinite(ke1).all() and np.isfinite(ke2).all() and \
        np.isfinite(h1).all() and np.isfinite(h2).all()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(t, h1.min(axis=1), label="min h1")
    axes[0].plot(t, h1.max(axis=1), label="max h1")
    axes[0].set_ylabel("h1 min/max")
    axes[0].legend()
    axes[0].grid(alpha=0.3, ls="--")
    axes[1].plot(t, ke1, label="KE layer 1")
    axes[1].plot(t, ke2, label="KE layer 2")
    axes[1].set_ylabel("mean KE")
    axes[1].legend()
    axes[1].grid(alpha=0.3, ls="--")
    axes[1].set_yscale("log")
    axes[2].plot(t, total - total[0], label="total mass drift")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("mass - mass(0)")
    axes[2].legend()
    axes[2].grid(alpha=0.3, ls="--")
    axes[0].set_title(f"Stability diagnostics — finite={finite}", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "stability.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {
        "finite": bool(finite),
        "h1_min": float(h1.min()), "h1_max": float(h1.max()),
        "mass_drift": float(total[-1] - total[0]),
        "ke1_last": float(ke1[-1]), "ke2_last": float(ke2[-1]),
    }


def animate_field(flds, field, Nx, Ny, outdir, frames, fname):
    T = flds[field].shape[0]
    idx = np.unique(np.linspace(0, T - 1, frames).astype(int))
    fig, ax = plt.subplots(figsize=(5, 5))
    first = flds[field][idx[0]]
    vmax = max(abs(first.min()), abs(first.max()))
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    im = ax.imshow(first, cmap=CMAP, norm=norm, origin="lower")
    ax.set_title(f"{field}  t=0")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    def update(i):
        fi = flds[field][idx[i]]
        vmax = max(abs(fi.min()), abs(fi.max()))
        im.set_norm(TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
        im.set_data(fi)
        ax.set_title(f"{field}  t={idx[i]}")
        return im,

    anim = FuncAnimation(fig, update, frames=len(idx), interval=50, blit=True)
    anim.save(os.path.join(outdir, fname), writer=PillowWriter(fps=20))
    plt.close(fig)


def write_report(outdir, cfg, scales, stability, out_files):
    lines = []
    lines.append("# SW trajectory-quality diagnostics")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("```")
    for k, v in cfg.items():
        lines.append(f"  {k}: {v}")
    lines.append("```")
    lines.append("")
    lines.append("## Characteristic scales")
    lines.append("")
    lines.append(f"  c1 = sqrt(g1*H) = {scales['c1']:.3f}   "
                 f"c2 = sqrt(g2*H) = {scales['c2']:.3f}")
    lines.append(f"  Rd1 = c1/f = {scales['Rd1']:.1f} dx   "
                 f"Rd2 = c2/f = {scales['Rd2']:.1f} dx")
    lines.append(f"  Rd2/Rd1 = {scales['Rd2'] / scales['Rd1']:.2f}")
    lines.append(f"  inertial period T_f = {scales['T_f']:.1f}")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for f in out_files:
        lines.append(f"![{f}]({f})")
    lines.append("")
    lines.append("## Auto-generated checklist")
    lines.append("")
    stable = stability["finite"]
    lines.append(f"- [{'x' if stable else ' '}] stable over {cfg['steps']} steps "
                 f"(no NaN/blowup): {'PASS' if stable else 'FAIL'}")
    lines.append(f"    - h1 range: {stability['h1_min']:.3f} .. {stability['h1_max']:.3f}")
    lines.append(f"    - total mass drift: {stability['mass_drift']:.3e}")
    lines.append(f"    - KE layer1 last: {stability['ke1_last']:.3e}, "
                 f"layer2 last: {stability['ke2_last']:.3e}")
    lines.append(f"- [{'x' if scales['Rd2'] / scales['Rd1'] >= 1.5 else ' '}] "
                 f"two-layer scale separation (Rd2/Rd1={scales['Rd2'] / scales['Rd1']:.1f})")
    lines.append("- [ ] jet present (meandering) — inspect snapshots.h1 / animation_h1.gif")
    lines.append("- [ ] ring shedding visible — inspect vorticity / animation_zeta1.gif")
    lines.append("- [ ] spectra slope near k^-3 in inertial band — inspect spectra.png")
    lines.append("")
    with open(os.path.join(outdir, "sw_quality_report.md"), "w") as f:
        f.write("\n".join(lines))


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="SW trajectory-quality diagnostics")
    p.add_argument("--Nx", type=int, default=64)
    p.add_argument("--Ny", type=int, default=64)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--spinup", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--device", default="cpu")
    p.add_argument("--outdir", default=os.path.join(
        os.path.dirname(__file__), "outputs", "sw_quality"))
    p.add_argument("--quick", action="store_true",
                   help="small run for tests (32x32, 2000 steps, 60 frames)")
    args = p.parse_args(argv)

    if args.quick:
        Nx, Ny, steps, spinup, frames = 32, 32, 2000, 500, 60
    else:
        Nx, Ny, steps, spinup, frames = args.Nx, args.Ny, args.steps, args.spinup, args.frames

    os.makedirs(args.outdir, exist_ok=True)

    dyn = build_dynamics(Nx, Ny, args.device)
    print(f"Generating {steps} steps (spinup={spinup}) at {Nx}x{Ny} ...")
    traj, _ = dyn.generate_full_trajectory(
        num_steps=steps, seed=args.seed, spinup_steps=spinup,
        bickley_jet=True, bickley_H_ref=H_REF,
    )
    traj = traj.detach().cpu().numpy()

    flds = field_arrays(traj, Nx, Ny)
    scales = characteristic_scales(f_cor=dyn.f_cor, g1=dyn.g1, g2=dyn.g2, H=H_REF)

    times = pick_times(flds["u1"].shape[0], 4)
    plot_snapshots(flds, times, args.outdir, Nx, Ny)
    plot_vorticity(flds, times, args.outdir, Nx, Ny)
    plot_spectra(flds, Nx, Ny, args.outdir, scales)
    plot_hovmoeller(flds, Nx, Ny, args.outdir)
    stability = plot_stability(traj, Nx, Ny, args.outdir)

    animate_field(flds, "h1", Nx, Ny, args.outdir, frames, "animation_h1.gif")
    z1 = vorticity(flds["u1"], flds["v1"], Nx, Ny)
    flds["zeta1"] = z1
    animate_field(flds, "zeta1", Nx, Ny, args.outdir, frames, "animation_zeta1.gif")
    animate_field(flds, "h2", Nx, Ny, args.outdir, frames, "animation_h2.gif")

    out_files = ["snapshots.png", "vorticity.png", "spectra.png",
                 "hovmoeller.png", "stability.png",
                 "animation_h1.gif", "animation_zeta1.gif", "animation_h2.gif"]
    cfg = {
        "system": "shallow_water", "bickley_jet": True, "tau0": 0.0,
        "Nx": Nx, "Ny": Ny, "dt": dyn.dt, "g1": dyn.g1, "g2": dyn.g2,
        "f_cor": dyn.f_cor, "coupling": dyn.coupling_coeff,
        "friction": dyn.friction, "viscosity": dyn.viscosity,
        "steps": steps, "spinup": spinup, "seed": args.seed,
        "H_ref": H_REF,
    }
    write_report(args.outdir, cfg, scales, stability, out_files)

    print(f"Stability finite={stability['finite']} "
          f"h1[{stability['h1_min']:.3f},{stability['h1_max']:.3f}] "
          f"Rd1={scales['Rd1']:.1f} Rd2={scales['Rd2']:.1f} "
          f"Rd2/Rd1={scales['Rd2'] / scales['Rd1']:.2f}")
    print(f"Wrote quality pack to {args.outdir}/")


if __name__ == "__main__":
    main()
