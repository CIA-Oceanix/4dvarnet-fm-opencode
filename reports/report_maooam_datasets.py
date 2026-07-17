#!/usr/bin/env python3
"""Diagnostic report for MAOOAM datasets (nx=3 and nx=4).

Loads the pre-generated .pt dataset files and produces:
  1. Multi-trajectory timeseries overview
  2. Physical field snapshots (spectral -> physical)
  3. Hovmöller diagrams (time vs mode index)
  4. Phase-space attractor projections
  5. Metrics comparison table (nx3 vs nx4)
  6. Animation of psi_upper field (nx4)

Usage:
    python reports/report_maooam_datasets.py [--output-dir reports/outputs/figs/maooam_datasets]
"""

import sys, os, argparse, warnings, textwrap
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

FIGS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "figs", "maooam_datasets")

DS_NX3 = "experiments/maooam_dataset_nx3.pt"
DS_NX4 = "experiments/maooam_dataset_nx4.pt"


# ── Helpers ──────────────────────────────────────────────────────────

def load_dataset(path):
    """Load a .pt dataset and return windows + metadata."""
    d = torch.load(path, weights_only=False)
    return d["windows"], d["config"], d["state_dim"], d["nx"], d["n_trajs"], d["seeds"]


def get_mode_ranges(state_dim, natm, noc):
    """Return (psi_a, theta_a, psi_o, dT_o) slice objects."""
    return (
        slice(0, natm),
        slice(natm, 2 * natm),
        slice(2 * natm, 2 * natm + noc),
        slice(2 * natm + noc, None),
    )


def compute_metrics(traj, dt=0.1, natm=36, noc=16):
    """Compute the 6 validation metrics on a numpy array."""
    x = traj if isinstance(traj, np.ndarray) else traj.numpy()
    N = x.shape[0]
    metrics = {
        "max_value": float(x.max()),
        "stationarity_ratio": float(
            np.abs(x[:N//4].mean(0) - x[3*N//4:].mean(0)).mean()
            / (x.std(0).mean() + 1e-10)
        ),
        "temporal_autocorrelation": float(np.mean([
            np.corrcoef(x[:-1, i], x[1:, i])[0, 1] for i in range(x.shape[1])
        ])),
        "slow_mode_variance_fraction": float(np.var(x[:, :2*natm]) / (np.var(x) + 1e-10)),
        "var_psi_a": float(np.var(x[:, :natm])),
        "var_theta_a": float(np.var(x[:, natm:2*natm])),
        "var_psi_o": float(np.var(x[:, 2*natm:2*natm+noc])),
        "var_dT_o": float(np.var(x[:, 2*natm+noc:])),
        "amp_psi_a": float(x[:, :natm].max() - x[:, :natm].min()),
        "amp_theta_a": float(x[:, natm:2*natm].max() - x[:, natm:2*natm].min()),
        "amp_psi_o": float(x[:, 2*natm:2*natm+noc].max() - x[:, 2*natm:2*natm+noc].min()),
        "amp_dT_o": float(x[:, 2*natm+noc:].max() - x[:, 2*natm+noc:].min()),
    }
    return metrics


def _get_modes(nx):
    """Return (natm, noc) for a given spectral truncation nx."""
    return _count_atm_modes(nx, nx), _count_oc_modes(nx, nx)


# ── Figure 1: Multi-trajectory timeseries ────────────────────────────

def plot_timeseries(windows_nx3, windows_nx4, nx3, nx4, sd3, sd4, output_dir, T=500):
    """Overlay 10 trajectories per dataset for each variable block."""
    block_labels = [r"$\psi_a$", r"$\theta_a$", r"$\psi_o$", r"$\Delta T_o$"]
    natm3, noc3 = _get_modes(nx3)
    natm4, noc4 = _get_modes(nx4)
    slices3 = get_mode_ranges(sd3, natm3, noc3)
    slices4 = get_mode_ranges(sd4, natm4, noc4)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    time = np.arange(T)

    for row, (windows, ntrajs, slices, nx_label, state_dim) in enumerate([
        (windows_nx3, 10, slices3, "nx=3 (state_dim=60)", sd3),
        (windows_nx4, 10, slices4, "nx=4 (state_dim=104)", sd4),
    ]):
        trajs_stacked = []
        for tr in range(ntrajs):
            seg = torch.cat([windows[tr * 200 + w]["true_state"].cpu() for w in range(200)])
            trajs_stacked.append(seg[:T])
        trajs_stacked = torch.stack(trajs_stacked).numpy()

        for col, slc in enumerate(slices):
            ax = axes[row, col]
            for tr in range(ntrajs):
                data = trajs_stacked[tr, :, slc]
                alpha, lw = (0.15, 0.4) if tr != 0 else (0.9, 1.2)
                for j in range(min(3, data.shape[1])):
                    ax.plot(time, data[:, j], alpha=alpha, lw=lw, color="k")
            ax.set_title(f"{nx_label} — {block_labels[col]}")
            ax.set_xlabel("time step")
            ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "maooam_dataset_timeseries.png"), dpi=150)
    plt.close()
    print("  saved maooam_dataset_timeseries.png")


# ── Figure 2: Physical field snapshots ───────────────────────────────

def plot_snapshots(windows_nx3, windows_nx4, nx3, nx4, output_dir, interp_size=64):
    """Physical field snapshots for 3 windows × 2 datasets."""
    fields_to_plot = ["psi_upper", "psi_oc", "T_atm", "T_oc"]
    field_titles = [r"$\psi_{upper}$", r"$\psi_{oc}$", r"$T_{atm}$", r"$T_{oc}$"]

    # Build dynamics for each resolution
    print("  building dynamics for nx=3...")
    dyn3 = MaooamTorchDynamics(device="cpu", compile=False,
                               atm_nx=nx3, atm_ny=nx3, occ_nx=nx3, occ_ny=nx3)
    print("  building dynamics for nx=4...")
    dyn4 = MaooamTorchDynamics(device="cpu", compile=False,
                               atm_nx=nx4, atm_ny=nx4, occ_nx=nx4, occ_ny=nx4)

    snap3 = []
    for w_idx in [0, 100, 199]:
        snap3.append(windows_nx3[w_idx]["true_state"][0].cpu().numpy())
    snap4 = []
    for w_idx in [0, 100, 199]:
        snap4.append(windows_nx4[w_idx]["true_state"][0].cpu().numpy())

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for row, (state, dyn, nx_label) in enumerate([
        (snap3, dyn3, "nx=3"), (snap4, dyn4, "nx=4"),
    ]):
        for fcol, fname in enumerate(fields_to_plot):
            ax = axes[row, fcol]
            st = state[0]  # first snapshot
            phys = dyn.spectral_to_physical(st, interp_size=interp_size)
            data = phys[fname]
            vmax = max(abs(data.min()), abs(data.max())) or 1.0
            im = ax.imshow(data, cmap="RdBu_r", origin="lower",
                           norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
            ax.set_title(f"{nx_label} — {field_titles[fcol]}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "maooam_dataset_snapshots.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved maooam_dataset_snapshots.png")


# ── Figure 3: Hovmöller ──────────────────────────────────────────────

def plot_hovmoller(windows_nx3, windows_nx4, nx3, nx4, output_dir):
    """Hovmöller: time vs mode index for atmosphere and ocean blocks."""
    natm3, noc3 = _get_modes(nx3)
    natm4, noc4 = _get_modes(nx4)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, (windows, natm, noc, nx_label) in enumerate([
        (windows_nx3, natm3, noc3, "nx=3"),
        (windows_nx4, natm4, noc4, "nx=4"),
    ]):
        traj = torch.cat([windows[w]["true_state"].cpu() for w in range(200)])
        traj_np = traj.numpy()

    # Atmosphere modes (first 2*natm)
        atm = traj_np[:, :2*natm].T
        axes[row, 0].imshow(atm, aspect="auto", cmap="RdBu_r",
                            norm=TwoSlopeNorm(vcenter=0))
        axes[row, 0].set_title(f"{nx_label} — atmosphere modes")
        axes[row, 0].set_xlabel("time step")
        axes[row, 0].set_ylabel("mode index")

        # Ocean modes
        oc = traj_np[:, 2*natm:2*natm+noc].T
        axes[row, 1].imshow(oc, aspect="auto", cmap="RdBu_r",
                            norm=TwoSlopeNorm(vcenter=0))
        axes[row, 1].set_title(f"{nx_label} — ocean modes")
        axes[row, 1].set_xlabel("time step")
        axes[row, 1].set_ylabel("mode index")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "maooam_dataset_hovmoller.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved maooam_dataset_hovmoller.png")


# ── Figure 4: Phase space ────────────────────────────────────────────

def plot_phase_space(windows_nx3, windows_nx4, nx3, nx4, output_dir):
    """2D attractor projections for atmosphere (psi_a[0] vs theta_a[0])
    and ocean (psi_o[0] vs dT_o[0])."""
    natm3, noc3 = _get_modes(nx3)
    natm4, noc4 = _get_modes(nx4)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for row, (windows, natm, noc, nx_label) in enumerate([
        (windows_nx3, natm3, noc3, "nx=3"),
        (windows_nx4, natm4, noc4, "nx=4"),
    ]):
        traj = torch.cat([windows[w]["true_state"].cpu() for w in range(200)])
        x = traj.numpy()

        atm_a = x[:, 0]       # psi_a[0]
        atm_b = x[:, natm]     # theta_a[0]
        oc_a = x[:, 2*natm]             # psi_o[0]
        oc_b = x[:, 2*natm + noc]       # dT_o[0]

        axes[row, 0].plot(atm_a, atm_b, "k-", alpha=0.6, lw=0.5)
        axes[row, 0].set_title(f"{nx_label} — atmosphere attractor\n$\\psi_a[0]$ vs $\\theta_a[0]$")
        axes[row, 0].set_xlabel(r"$\psi_a[0]$")
        axes[row, 0].set_ylabel(r"$\theta_a[0]$")
        axes[row, 0].ticklabel_format(style="sci", axis="both", scilimits=(-2, 2))

        axes[row, 1].plot(oc_a, oc_b, "k-", alpha=0.6, lw=0.5)
        axes[row, 1].set_title(f"{nx_label} — ocean attractor\n$\\psi_o[0]$ vs $\\Delta T_o[0]$")
        axes[row, 1].set_xlabel(r"$\psi_o[0]$")
        axes[row, 1].set_ylabel(r"$\Delta T_o[0]$")
        axes[row, 1].ticklabel_format(style="sci", axis="both", scilimits=(-2, 2))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "maooam_dataset_phase_space.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved maooam_dataset_phase_space.png")


# ── Figure 5: Metrics table (text) ───────────────────────────────────

def save_metrics_table(windows_nx3, windows_nx4, nx3, nx4, sd3, sd4, output_dir):
    """Compute metrics for both datasets and save a formatted text table."""
    natm3, noc3 = _get_modes(nx3)
    natm4, noc4 = _get_modes(nx4)

    traj3 = torch.cat([windows_nx3[w]["true_state"].cpu() for w in range(200)]).numpy()
    traj4 = torch.cat([windows_nx4[w]["true_state"].cpu() for w in range(200)]).numpy()

    m3 = compute_metrics(traj3, natm=natm3, noc=noc3)
    m4 = compute_metrics(traj4, natm=natm4, noc=noc4)

    lines = [
        "=" * 60,
        "MAOOAM Dataset Metrics Comparison",
        "=" * 60,
        f"{'Metric':<35} {'nx=3':>12} {'nx=4':>12}",
        "-" * 60,
    ]
    for key in m3:
        lines.append(f"{key:<35} {m3[key]:>12.6f} {m4[key]:>12.6f}")

    lines += [
        "-" * 60,
        f"state_dim:          {sd3:>12d} {sd4:>12d}",
        f"n_trajs:            10             10",
        f"total windows:      {len(windows_nx3):>12d} {len(windows_nx4):>12d}",
        f"total windows:      {len(windows_nx3):>12d} {len(windows_nx4):>12d}",
        "=" * 60,
    ]

    text = "\n".join(lines)
    outpath = os.path.join(output_dir, "maooam_dataset_metrics.txt")
    with open(outpath, "w") as f:
        f.write(text + "\n")
    print("  saved maooam_dataset_metrics.txt")
    print(text)


# ── Figure 6: Animation of psi_upper field ──────────────────────────

def make_animation(windows_nx4, output_dir, interp_size=64, fps=10):
    """GIF animation of psi_upper field over the full trajectory (nx=4 dataset)."""
    print("  building dynamics for nx=4 animation...")
    dyn = MaooamTorchDynamics(device="cpu", compile=False,
                               atm_nx=4, atm_ny=4, occ_nx=4, occ_ny=4)

    traj = torch.cat([windows_nx4[w]["true_state"].cpu() for w in range(200)])
    nframes = len(traj)

    fig, ax = plt.subplots(figsize=(6, 5))

    # First frame
    st = traj[0].numpy()
    phys = dyn.spectral_to_physical(st, interp_size=interp_size)
    data = phys["psi_upper"]
    vmax = max(abs(data.min()), abs(data.max())) or 1.0
    im = ax.imshow(data, cmap="RdBu_r", origin="lower",
                   norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
    plt.colorbar(im, ax=ax, shrink=0.8, format="%.1e")
    title = ax.set_title(f"$\\psi_{{upper}}$ — step 0")
    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame):
        st = traj[frame].numpy()
        phys = dyn.spectral_to_physical(st, interp_size=interp_size)
        data = phys["psi_upper"]
        im.set_data(data)
        vmax = max(abs(data.min()), abs(data.max())) or 1.0
        im.set_norm(TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
        title.set_text(f"$\\psi_{{upper}}$ — step {frame}")
        return [im, title]

    print(f"  rendering {nframes} frames...")
    anim = FuncAnimation(fig, update, frames=nframes, interval=1000//fps, blit=True)
    outpath = os.path.join(output_dir, "maooam_psi_upper_nx4.gif")
    anim.save(outpath, writer=PillowWriter(fps=fps))
    plt.close()
    print(f"  saved maooam_psi_upper_nx4.gif ({nframes} frames)")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=FIGS_DIR)
    parser.add_argument("--make-animation", action="store_true", default=True,
                        help="Generate psi_upper GIF animation (default: True)")
    parser.add_argument("--no-animation", action="store_false", dest="make_animation")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    output_dir = args.output

    print("Loading datasets...")
    windows3, cfg3, sd3, nx3, nt3, seeds3 = load_dataset(DS_NX3)
    windows4, cfg4, sd4, nx4, nt4, seeds4 = load_dataset(DS_NX4)
    print(f"  nx=3: state_dim={sd3}, {len(windows3)} windows, {nt3} trajectories")
    print(f"  nx=4: state_dim={sd4}, {len(windows4)} windows, {nt4} trajectories")

    print("\n--- Figure 1: Multi-trajectory timeseries ---")
    plot_timeseries(windows3, windows4, nx3, nx4, sd3, sd4, output_dir)

    print("\n--- Figure 2: Physical field snapshots ---")
    plot_snapshots(windows3, windows4, nx3, nx4, output_dir)

    print("\n--- Figure 3: Hovmöller diagrams ---")
    plot_hovmoller(windows3, windows4, nx3, nx4, output_dir)

    print("\n--- Figure 4: Phase space ---")
    plot_phase_space(windows3, windows4, nx3, nx4, output_dir)

    print("\n--- Figure 5: Metrics table ---")
    save_metrics_table(windows3, windows4, nx3, nx4, sd3, sd4, output_dir)

    if args.make_animation:
        print("\n--- Figure 6: Animation ---")
        make_animation(windows4, output_dir)
    else:
        print("\n--- Figure 6: Animation (skipped) ---")

    print("\nDone.")


if __name__ == "__main__":
    main()