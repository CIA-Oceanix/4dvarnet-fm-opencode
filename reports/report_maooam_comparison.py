#!/usr/bin/env python3
"""Comparison report for 3 published MAOOAM configurations.

Generates:
  1. Overlaid timeseries (psi_a[0] for all 3 configs)
  2. Physical field snapshots (3x4 grid)
  3. Phase-space attractor projections
  4. Temporal variability analysis (PSD, autocorrelation)
  5. Ocean streamfunction animation (3 GIFs)
  6. qgs-style multi-panel animation (DDV2016 only)

Usage:
    python reports/report_maooam_comparison.py [--data-dir experiments] [--output-dir reports/outputs/figs/maooam_comparison]
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

FIGS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "figs", "maooam_comparison")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments")

CONFIGS = ["ddv2016", "vspd2019", "hamilton2023"]
CONFIG_LABELS = {
    "ddv2016": "DDV2016",
    "vspd2019": "VSPD2019",
    "hamilton2023": "Hamilton2023",
}
CONFIG_COLORS = {"ddv2016": "#1f77b4", "vspd2019": "#ff7f0e", "hamilton2023": "#2ca02c"}

PRESETS = {
    "ddv2016": {"atm_nx": 2, "atm_ny": 2, "occ_nx": 2, "occ_ny": 4, "dynamic_T": False},
    "vspd2019": {"atm_nx": 2, "atm_ny": 2, "occ_nx": 2, "occ_ny": 4, "dynamic_T": False},
    "hamilton2023": {"atm_nx": 2, "atm_ny": 2, "occ_nx": 2, "occ_ny": 4, "dynamic_T": True},
}


def _get_modes(cfg_name):
    p = PRESETS[cfg_name]
    return _count_atm_modes(p["atm_nx"], p["atm_ny"]), _count_oc_modes(p["occ_nx"], p["occ_ny"])


def load_dataset(cfg_name, data_dir):
    path = os.path.join(data_dir, f"maooam_comparison_{cfg_name}.pt")
    d = torch.load(path, weights_only=False)
    windows = d["windows"]
    traj = torch.cat([w["true_state"].cpu() for w in windows])
    return windows, traj, d["state_dim"], d["n_trajs"]


def make_dynamics(cfg_name):
    p = PRESETS[cfg_name]
    return MaooamTorchDynamics(
        device="cpu", compile=False,
        atm_nx=p["atm_nx"], atm_ny=p["atm_ny"],
        occ_nx=p["occ_nx"], occ_ny=p["occ_ny"],
        dynamic_T=p["dynamic_T"],
    )


# ── Figure 1: Overlaid timeseries ─────────────────────────────────────

def plot_timeseries(trajs, output_dir, T=500):
    fig, ax = plt.subplots(figsize=(12, 5))
    time = np.arange(T)
    for cfg in CONFIGS:
        traj = trajs[cfg][:T]
        natm, _ = _get_modes(cfg)
        ax.plot(time, traj[:, 0], label=CONFIG_LABELS[cfg],
                color=CONFIG_COLORS[cfg], alpha=0.8, lw=1.0)
    ax.set_xlabel("time step")
    ax.set_ylabel(r"$\psi_a[0]$")
    ax.legend()
    ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "maooam_comparison_timeseries.png"), dpi=150)
    plt.close()
    print("  saved maooam_comparison_timeseries.png")


# ── Figure 2: Physical field snapshots ────────────────────────────────

def plot_snapshots(trajs, output_dir, interp_size=64):
    fields = ["psi_upper", "psi_oc", "T_atm", "T_oc"]
    titles = [r"$\psi_{upper}$", r"$\psi_{oc}$", r"$T_{atm}$", r"$T_{oc}$"]

    dyns = {}
    snapshots = {}
    for cfg in CONFIGS:
        print(f"  building dynamics for {cfg}...")
        dyns[cfg] = make_dynamics(cfg)
        snapshots[cfg] = trajs[cfg][0].numpy()

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    for row, cfg in enumerate(CONFIGS):
        for col, fname in enumerate(fields):
            ax = axes[row, col]
            phys = dyns[cfg].spectral_to_physical(snapshots[cfg], interp_size=interp_size)
            data = phys[fname]
            vmax = max(abs(data.min()), abs(data.max())) or 1.0
            ax.imshow(data, cmap="RdBu_r", origin="lower",
                       norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
            ax.set_title(f"{CONFIG_LABELS[cfg]} — {titles[col]}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "maooam_comparison_snapshots.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved maooam_comparison_snapshots.png")


# ── Figure 3: Phase-space attractors ──────────────────────────────────

def plot_phase_space(trajs, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for col, cfg in enumerate(CONFIGS):
        ax = axes[col]
        x = trajs[cfg].numpy()
        natm, noc = _get_modes(cfg)
        ax.plot(x[:, 0], x[:, 2*natm], "k-", alpha=0.5, lw=0.3)
        ax.set_title(f"{CONFIG_LABELS[cfg]}\n$\\psi_a[0]$ vs $\\psi_o[0]$")
        ax.set_xlabel(r"$\psi_a[0]$")
        ax.set_ylabel(r"$\psi_o[0]$")
        ax.ticklabel_format(style="sci", axis="both", scilimits=(-2, 2))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "maooam_comparison_phase_space.png"), dpi=150)
    plt.close()
    print("  saved maooam_comparison_phase_space.png")


# ── Figure 4: Temporal variability ────────────────────────────────────

def plot_variability(trajs, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for cfg in CONFIGS:
        x = trajs[cfg].numpy()
        n = x.shape[0]
        # PSD via FFT
        fft = np.fft.rfft(x[:, 0])
        psd = np.abs(fft) ** 2
        freq = np.fft.rfftfreq(n, d=0.1)
        axes[0].loglog(freq[1:], psd[1:], color=CONFIG_COLORS[cfg],
                       label=CONFIG_LABELS[cfg], alpha=0.8)
        # Autocorrelation
        ac = np.correlate(x[:, 0], x[:, 0], mode="full")
        ac = ac[ac.size // 2:] / ac.max()
        axes[1].plot(ac[:200], color=CONFIG_COLORS[cfg],
                     label=CONFIG_LABELS[cfg], alpha=0.8)

    axes[0].set_xlabel("frequency (1/dt)")
    axes[0].set_ylabel("PSD")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Power spectral density of $\\psi_a[0]$")

    axes[1].set_xlabel("lag")
    axes[1].set_ylabel("autocorrelation")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Autocorrelation of $\\psi_a[0]$")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "maooam_comparison_variability.png"), dpi=150)
    plt.close()
    print("  saved maooam_comparison_variability.png")


# ── Figure 5: Ocean streamfunction animations ─────────────────────────

def make_ocean_animations(trajs, output_dir, interp_size=64, fps=10, max_frames=500):
    for cfg in CONFIGS:
        print(f"  building dynamics for {cfg} animation...")
        dyn = make_dynamics(cfg)
        traj = trajs[cfg]
        nframes = min(len(traj), max_frames)

        fig, ax = plt.subplots(figsize=(6, 5))
        st = traj[0].numpy()
        phys = dyn.spectral_to_physical(st, interp_size=interp_size)
        data = phys["psi_oc"]
        vmax = max(abs(data.min()), abs(data.max())) or 1.0
        im = ax.imshow(data, cmap="RdBu_r", origin="lower",
                       norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
        plt.colorbar(im, ax=ax, shrink=0.8, format="%.1e")
        title = ax.set_title(f"{CONFIG_LABELS[cfg]} $\\psi_{{oc}}$ — step 0")
        ax.set_xticks([])
        ax.set_yticks([])

        def make_update(traj_np, im, title):
            def update(frame):
                st = traj_np[frame]
                phys = dyn.spectral_to_physical(st, interp_size=interp_size)
                data = phys["psi_oc"]
                im.set_data(data)
                vmax = max(abs(data.min()), abs(data.max())) or 1.0
                im.set_norm(TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
                title.set_text(f"{CONFIG_LABELS[cfg]} $\\psi_{{oc}}$ — step {frame}")
                return [im, title]
            return update

        anim = FuncAnimation(fig, make_update(traj.numpy(), im, title),
                             frames=nframes, interval=1000//fps, blit=True)
        outpath = os.path.join(output_dir, f"maooam_psi_oc_{cfg}.gif")
        anim.save(outpath, writer=PillowWriter(fps=fps))
        plt.close()
        print(f"  saved {outpath} ({nframes} frames)")


# ── Figure 6: Multi-panel animation (qgs-style, DDV2016 only) ─────────

def make_multi_panel_animation(trajs, output_dir, interp_size=64, fps=10, max_frames=500):
    cfg = "ddv2016"
    print(f"  building multi-panel animation for {cfg}...")
    dyn = make_dynamics(cfg)
    traj = trajs[cfg]
    nframes = min(len(traj), max_frames)
    traj_np = traj.numpy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"{CONFIG_LABELS[cfg]} — Multi-panel", fontsize=14)

    # Axes layout: timeseries, phase, psi_upper, psi_oc, T_atm, T_oc
    ax_ts = axes[0, 0]
    ax_ps = axes[0, 1]
    ax_pu = axes[0, 2]
    ax_po = axes[1, 0]
    ax_ta = axes[1, 1]
    ax_to = axes[1, 2]

    # Timeseries: psi_a[0]
    natm, noc = _get_modes(cfg)
    ts_line, = ax_ts.plot([], [], "b-", lw=1.0)
    ax_ts.set_xlim(0, nframes)
    ax_ts.set_ylim(traj_np[:, 0].min(), traj_np[:, 0].max())
    ax_ts.set_xlabel("step")
    ax_ts.set_ylabel(r"$\psi_a[0]$")
    ax_ts.set_title("Timeseries")

    # Phase space attractor
    ps_line, = ax_ps.plot([], [], "k-", lw=0.3, alpha=0.5)
    ax_ps.set_xlim(traj_np[:, 0].min(), traj_np[:, 0].max())
    ax_ps.set_ylim(traj_np[:, 2*natm].min(), traj_np[:, 2*natm].max())
    ax_ps.set_xlabel(r"$\psi_a[0]$")
    ax_ps.set_ylabel(r"$\psi_o[0]$")
    ax_ps.set_title("Phase space")
    ps_scat, = ax_ps.plot([], [], "ro", markersize=3)

    # Field panels
    st_0 = traj_np[0]
    phys_0 = dyn.spectral_to_physical(st_0, interp_size=interp_size)

    def _init_field(data):
        vmax = max(abs(data.min()), abs(data.max())) or 1.0
        im = plt.imshow(data, cmap="RdBu_r", origin="lower",
                        norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
        return im

    im_pu = ax_pu.imshow(phys_0["psi_upper"], cmap="RdBu_r", origin="lower")
    ax_pu.set_title(r"$\psi_{upper}$")
    ax_pu.set_xticks([])
    ax_pu.set_yticks([])

    im_po = ax_po.imshow(phys_0["psi_oc"], cmap="RdBu_r", origin="lower")
    ax_po.set_title(r"$\psi_{oc}$")
    ax_po.set_xticks([])
    ax_po.set_yticks([])

    im_ta = ax_ta.imshow(phys_0["T_atm"], cmap="RdBu_r", origin="lower")
    ax_ta.set_title(r"$T_{atm}$")
    ax_ta.set_xticks([])
    ax_ta.set_yticks([])

    im_to = ax_to.imshow(phys_0["T_oc"], cmap="RdBu_r", origin="lower")
    ax_to.set_title(r"$T_{oc}$")
    ax_to.set_xticks([])
    ax_to.set_yticks([])

    plt.tight_layout()

    def update(frame):
        ts_line.set_data(np.arange(frame), traj_np[:frame, 0])
        ps_line.set_data(traj_np[:frame, 0], traj_np[:frame, 2*natm])
        ps_scat.set_data([traj_np[frame, 0]], [traj_np[frame, 2*natm]])

        st = traj_np[frame]
        phys = dyn.spectral_to_physical(st, interp_size=interp_size)

        for im, key in [(im_pu, "psi_upper"), (im_po, "psi_oc"),
                        (im_ta, "T_atm"), (im_to, "T_oc")]:
            data = phys[key]
            im.set_data(data)
            vmax = max(abs(data.min()), abs(data.max())) or 1.0
            im.set_norm(TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))

        fig.suptitle(f"{CONFIG_LABELS[cfg]} — step {frame}", fontsize=14)
        return [ts_line, ps_line, ps_scat, im_pu, im_po, im_ta, im_to]

    anim = FuncAnimation(fig, update, frames=nframes, interval=1000//fps, blit=True)
    outpath = os.path.join(output_dir, "maooam_multi_panel_ddv2016.gif")
    anim.save(outpath, writer=PillowWriter(fps=fps))
    plt.close()
    print(f"  saved {outpath} ({nframes} frames)")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=FIGS_DIR)
    parser.add_argument("--no-animations", action="store_true", help="Skip GIF animations")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data_dir = args.data_dir

    print("Loading datasets...")
    trajs = {}
    for cfg in CONFIGS:
        windows, traj, sd, nt = load_dataset(cfg, data_dir)
        trajs[cfg] = traj
        print(f"  {cfg}: state_dim={sd}, {len(windows)} windows, {nt} trajs")

    print("\n--- Figure 1: Overlaid timeseries ---")
    plot_timeseries(trajs, args.output_dir)

    print("\n--- Figure 2: Physical field snapshots ---")
    plot_snapshots(trajs, args.output_dir)

    print("\n--- Figure 3: Phase-space attractors ---")
    plot_phase_space(trajs, args.output_dir)

    print("\n--- Figure 4: Temporal variability ---")
    plot_variability(trajs, args.output_dir)

    if not args.no_animations:
        print("\n--- Figure 5: Ocean streamfunction animations ---")
        make_ocean_animations(trajs, args.output_dir)

        print("\n--- Figure 6: Multi-panel animation (DDV2016) ---")
        make_multi_panel_animation(trajs, args.output_dir)
    else:
        print("\n--- Figures 5-6: Animations (skipped) ---")

    print("\nDone.")


if __name__ == "__main__":
    main()