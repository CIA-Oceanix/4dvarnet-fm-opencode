#!/usr/bin/env python3
"""Single continuous 100-day MAOOAM trajectory with daily visualizations.

Unlike report_maooam_comparison.py (which concatenates independent trajectories
causing time jumps), this generates one long continuous run from a single
initial condition on the attractor.

Generates:
  1. psi_a[0] timeseries (100 days)
  2. Physical field snapshots (first and last day)
  3. Phase-space attractor (psi_a[0] vs psi_o[0])
  4. Temporal variability (PSD, autocorrelation)
  5. Ocean streamfunction animation (daily frames, 100 days)
  6. Multi-panel animation (daily frames, 100 days)

Usage:
    python reports/report_maooam_continuous.py [--output-dir reports/outputs/figs/maooam_continuous]
"""

import sys, os, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.animation import FuncAnimation, PillowWriter
from models.maooam_torch import MaooamTorchDynamics
from models.maooam_dynamics import _count_atm_modes, _count_oc_modes

FIGS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "figs", "maooam_continuous")

DT_NONDIM = 0.1
F0 = 1.032e-4
SECONDS_PER_DAY = 86400
STEPS_PER_DAY = int(round(SECONDS_PER_DAY * F0 / DT_NONDIM))  # ~89
DAYS = 100
N_STEPS = DAYS * STEPS_PER_DAY
SPINUP = 10000

CONFIG = "ddv2016"
CONFIG_LABEL = "DDV2016"

DYNAMICS_KWARGS = {
    "device": "cpu", "compile": False,
    "atm_nx": 2, "atm_ny": 2,
    "occ_nx": 2, "occ_ny": 4,
    "dynamic_T": False,
}


def generate_trajectory():
    print(f"Generating {DAYS}-day continuous trajectory (spinup={SPINUP})...")
    dyn = MaooamTorchDynamics(**DYNAMICS_KWARGS)
    total_steps = SPINUP + N_STEPS
    traj, _ = dyn.generate_full_trajectory(num_steps=total_steps, seed=42, spinup_steps=SPINUP)
    traj = traj[SPINUP:]  # (N_STEPS, 36)
    return dyn, traj


def make_days(traj_full, stride):
    n_frames = (traj_full.shape[0] + stride - 1) // stride
    days_arr = np.arange(n_frames).astype(float)
    return days_arr, n_frames


# ── Figure 1: Timeseries ──────────────────────────────────────────────

def plot_timeseries(traj_daily, days, output_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(days, traj_daily[:, 0], "b-", lw=1.0, label=CONFIG_LABEL)
    ax.set_xlabel("time (days)")
    ax.set_ylabel(r"$\psi_a[0]$")
    ax.legend()
    ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))
    ax.set_xlim(0, days[-1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "maooam_continuous_timeseries.png"), dpi=150)
    plt.close()
    print("  saved maooam_continuous_timeseries.png")


# ── Figure 2: Field snapshots (first vs last day) ────────────────────

def plot_snapshots(dyn, traj_daily, output_dir, interp_size=64):
    fields = ["psi_upper", "psi_oc", "T_atm", "T_oc"]
    titles = [r"$\psi_{upper}$", r"$\psi_{oc}$", r"$T_{atm}$", r"$T_{oc}$"]
    labels = ["day 0", f"day {traj_daily.shape[0]-1}"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for row, t in enumerate([0, -1]):
        st = traj_daily[t]
        phys = dyn.spectral_to_physical(st, interp_size=interp_size)
        for col, fname in enumerate(fields):
            ax = axes[row, col]
            data = phys[fname]
            vmax = max(abs(data.min()), abs(data.max())) or 1.0
            ax.imshow(data, cmap="RdBu_r", origin="lower",
                       norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
            title_str = f"{labels[row]} — {titles[col]}" if col == 0 else titles[col]
            ax.set_title(title_str, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "maooam_continuous_snapshots.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved maooam_continuous_snapshots.png")


# ── Figure 3: Phase space ────────────────────────────────────────────

def plot_phase_space(dyn, traj_daily, output_dir):
    natm = _count_atm_modes(DYNAMICS_KWARGS["atm_nx"], DYNAMICS_KWARGS["atm_ny"])
    fig, ax = plt.subplots(figsize=(8, 6))
    x = traj_daily[:, 0]
    y = traj_daily[:, 2 * natm]
    ax.plot(x, y, "k-", alpha=0.5, lw=0.3)
    ax.scatter(x[0], y[0], c="green", s=40, marker="o", zorder=5, label="start")
    ax.scatter(x[-1], y[-1], c="red", s=40, marker="s", zorder=5, label="end")
    ax.set_xlabel(r"$\psi_a[0]$")
    ax.set_ylabel(r"$\psi_o[0]$")
    ax.set_title(f"{CONFIG_LABEL} — {DAYS}-day attractor")
    ax.legend()
    ax.ticklabel_format(style="sci", axis="both", scilimits=(-2, 2))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "maooam_continuous_phase_space.png"), dpi=150)
    plt.close()
    print("  saved maooam_continuous_phase_space.png")


# ── Figure 4: Variability ────────────────────────────────────────────

def plot_variability(traj_daily, days, output_dir):
    dt_day = days[1] - days[0] if len(days) > 1 else 1.0
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = traj_daily[:, 0]
    n = x.shape[0]

    # PSD
    fft = np.fft.rfft(x)
    psd = np.abs(fft) ** 2
    freq = np.fft.rfftfreq(n, d=dt_day)
    axes[0].loglog(freq[1:], psd[1:], "b-", alpha=0.8, label=CONFIG_LABEL)
    axes[0].set_xlabel("frequency (1/day)")
    axes[0].set_ylabel("PSD")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"PSD of $\\psi_a[0]$ ({DAYS} days)")

    # Autocorrelation
    ac = np.correlate(x, x, mode="full")
    ac = ac[ac.size // 2:] / ac.max()
    lags = np.arange(len(ac)) * dt_day
    axes[1].plot(lags[:min(100, len(lags))], ac[:min(100, len(ac))], "b-", alpha=0.8)
    axes[1].set_xlabel("lag (days)")
    axes[1].set_ylabel("autocorrelation")
    axes[1].set_title(f"Autocorrelation of $\\psi_a[0]$ ({DAYS} days)")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].axhline(1/np.e, color="gray", ls="--", lw=0.5, label="1/e")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "maooam_continuous_variability.png"), dpi=150)
    plt.close()
    print("  saved maooam_continuous_variability.png")


# ── Figure 5: Ocean animation ────────────────────────────────────────

def make_ocean_animation(dyn, traj_daily, days, output_dir, interp_size=64, fps=10):
    nframes = traj_daily.shape[0]
    print(f"  ocean animation: {nframes} frames ({days[-1]:.0f} days)...")

    fig, ax = plt.subplots(figsize=(6, 5))
    st = traj_daily[0]
    phys = dyn.spectral_to_physical(st, interp_size=interp_size)
    data = phys["psi_oc"]
    vmax = max(abs(data.min()), abs(data.max())) or 1.0
    im = ax.imshow(data, cmap="RdBu_r", origin="lower",
                   norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
    plt.colorbar(im, ax=ax, shrink=0.8, format="%.1e")
    title = ax.set_title(f"{CONFIG_LABEL} $\\psi_{{oc}}$ — day {days[0]:.0f}")
    ax.set_xticks([]); ax.set_yticks([])

    traj_np = traj_daily

    def update(frame):
        st = traj_np[frame]
        phys = dyn.spectral_to_physical(st, interp_size=interp_size)
        data = phys["psi_oc"]
        im.set_data(data)
        vmax = max(abs(data.min()), abs(data.max())) or 1.0
        im.set_norm(TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
        title.set_text(f"{CONFIG_LABEL} $\\psi_{{oc}}$ — day {days[frame]:.0f}")
        return [im, title]

    anim = FuncAnimation(fig, update, frames=nframes, interval=1000//fps, blit=True)
    outpath = os.path.join(output_dir, "maooam_continuous_psi_oc.gif")
    anim.save(outpath, writer=PillowWriter(fps=fps))
    plt.close()
    print(f"  saved {outpath} ({nframes} frames, {days[-1]:.0f} days)")


# ── Figure 6: Multi-panel animation ──────────────────────────────────

def make_multi_panel_animation(dyn, traj_daily, days, output_dir, interp_size=64, fps=10):
    nframes = traj_daily.shape[0]
    print(f"  multi-panel animation: {nframes} frames ({days[-1]:.0f} days)...")

    traj_np = traj_daily
    natm = _count_atm_modes(DYNAMICS_KWARGS["atm_nx"], DYNAMICS_KWARGS["atm_ny"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"{CONFIG_LABEL} — {DAYS}-day continuous trajectory", fontsize=14)

    ax_ts = axes[0, 0]; ax_ps = axes[0, 1]; ax_pu = axes[0, 2]
    ax_po = axes[1, 0]; ax_ta = axes[1, 1]; ax_to = axes[1, 2]

    # Timeseries
    ts_line, = ax_ts.plot([], [], "b-", lw=1.0)
    ax_ts.set_xlim(0, days[-1])
    ax_ts.set_ylim(traj_np[:, 0].min(), traj_np[:, 0].max())
    ax_ts.set_xlabel("time (days)"); ax_ts.set_ylabel(r"$\psi_a[0]$"); ax_ts.set_title("Timeseries")

    # Phase space
    ps_line, = ax_ps.plot([], [], "k-", lw=0.3, alpha=0.5)
    ax_ps.set_xlim(traj_np[:, 0].min(), traj_np[:, 0].max())
    ax_ps.set_ylim(traj_np[:, 2*natm].min(), traj_np[:, 2*natm].max())
    ax_ps.set_xlabel(r"$\psi_a[0]$"); ax_ps.set_ylabel(r"$\psi_o[0]$"); ax_ps.set_title("Phase space")
    ps_scat, = ax_ps.plot([], [], "ro", markersize=3)

    # Field panels
    phys_0 = dyn.spectral_to_physical(traj_np[0], interp_size=interp_size)
    im_pu = ax_pu.imshow(phys_0["psi_upper"], cmap="RdBu_r", origin="lower")
    ax_pu.set_title(r"$\psi_{upper}$"); ax_pu.set_xticks([]); ax_pu.set_yticks([])

    im_po = ax_po.imshow(phys_0["psi_oc"], cmap="RdBu_r", origin="lower")
    ax_po.set_title(r"$\psi_{oc}$"); ax_po.set_xticks([]); ax_po.set_yticks([])

    im_ta = ax_ta.imshow(phys_0["T_atm"], cmap="RdBu_r", origin="lower")
    ax_ta.set_title(r"$T_{atm}$"); ax_ta.set_xticks([]); ax_ta.set_yticks([])

    im_to = ax_to.imshow(phys_0["T_oc"], cmap="RdBu_r", origin="lower")
    ax_to.set_title(r"$T_{oc}$"); ax_to.set_xticks([]); ax_to.set_yticks([])

    plt.tight_layout()

    def update(frame):
        day = days[frame]
        ts_line.set_data(days[:frame+1], traj_np[:frame+1, 0])
        ps_line.set_data(traj_np[:frame+1, 0], traj_np[:frame+1, 2*natm])
        ps_scat.set_data([traj_np[frame, 0]], [traj_np[frame, 2*natm]])

        phys = dyn.spectral_to_physical(traj_np[frame], interp_size=interp_size)
        for im, key in [(im_pu, "psi_upper"), (im_po, "psi_oc"),
                        (im_ta, "T_atm"), (im_to, "T_oc")]:
            data = phys[key]
            im.set_data(data)
            vmax = max(abs(data.min()), abs(data.max())) or 1.0
            im.set_norm(TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))

        fig.suptitle(f"{CONFIG_LABEL} — day {day:.0f}", fontsize=14)
        return [ts_line, ps_line, ps_scat, im_pu, im_po, im_ta, im_to]

    anim = FuncAnimation(fig, update, frames=nframes, interval=1000//fps, blit=True)
    outpath = os.path.join(output_dir, "maooam_continuous_multi_panel.gif")
    anim.save(outpath, writer=PillowWriter(fps=fps))
    plt.close()
    print(f"  saved {outpath} ({nframes} frames, {days[-1]:.0f} days)")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=FIGS_DIR)
    parser.add_argument("--no-animations", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate trajectory
    dyn, traj_full = generate_trajectory()
    stride = STEPS_PER_DAY
    days, n_frames = make_days(traj_full, stride)
    traj_daily = traj_full[::stride].numpy()

    print(f"Stride: {stride} steps = {days[1]:.1f} days")
    print(f"Trajectory: {traj_full.shape[0]} steps -> {n_frames} daily frames ({days[-1]:.0f} days)")

    print("\n--- Figure 1: Timeseries ---")
    plot_timeseries(traj_daily, days, args.output_dir)

    print("\n--- Figure 2: Field snapshots ---")
    plot_snapshots(dyn, traj_daily, args.output_dir)

    print("\n--- Figure 3: Phase space ---")
    plot_phase_space(dyn, traj_daily, args.output_dir)

    print("\n--- Figure 4: Temporal variability ---")
    plot_variability(traj_daily, days, args.output_dir)

    if not args.no_animations:
        print("\n--- Figure 5: Ocean animation ---")
        make_ocean_animation(dyn, traj_daily, days, args.output_dir)

        print("\n--- Figure 6: Multi-panel animation ---")
        make_multi_panel_animation(dyn, traj_daily, days, args.output_dir)
    else:
        print("\n--- Figures 5-6: Animations (skipped) ---")

    print("\nDone.")


if __name__ == "__main__":
    main()