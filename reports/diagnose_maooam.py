"""MAOOAM diagnostic validation: 6-metric suite + physical-field visualization.

Usage:
    python reports/diagnose_maooam.py [--seed 42] [--output reports/outputs/figs/maooam]
"""

import sys, os, argparse
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from models.maooam_dynamics import MaooamDynamics


def run_trajectory(dynamics, num_steps=50000, seed=42):
    """Generate a long trajectory."""
    print(f"Generating {num_steps}-step trajectory (seed={seed})...")
    traj, forcing = dynamics.generate_full_trajectory(num_steps, seed=seed, spinup_steps=5000)
    print(f"  shape: {traj.shape}, range: [{traj.min():.4f}, {traj.max():.4f}]")
    return traj


def compute_metrics(traj, dt=0.1, Natm=36, Noc=16):
    """Compute the 6 validation metrics."""
    x = traj.numpy()
    N = x.shape[0]

    # 1. Stability: max absolute value
    max_val = float(x.max())

    # 2. Stationarity: mean/std of first vs last quarter
    q1 = x[:N//4].mean(axis=0)
    q4 = x[3*N//4:].mean(axis=0)
    stationarity = float(np.abs(q1 - q4).mean() / (x.std(axis=0).mean() + 1e-10))

    # 3. Temporal autocorrelation at lag 1
    ac = 0.0
    for i in range(x.shape[1]):
        ac += np.corrcoef(x[:-1, i], x[1:, i])[0, 1]
    ac /= x.shape[1]

    # 4. Fraction of variance in slow modes (atm: psi_a + theta_a)
    slow_var = np.var(x[:, :2*Natm])
    total_var = np.var(x)
    slow_frac = slow_var / total_var

    # 5. Variance per variable block
    var_psi_a = np.var(x[:, :Natm]).item()
    var_theta_a = np.var(x[:, Natm:2*Natm]).item()
    var_psi_o = np.var(x[:, 2*Natm:2*Natm+Noc]).item()
    var_dT_o = np.var(x[:, 2*Natm+Noc:]).item()

    # 6. Peak-to-peak amplitude per block
    amp_psi_a = (x[:, :Natm].max() - x[:, :Natm].min()).item()
    amp_theta_a = (x[:, Natm:2*Natm].max() - x[:, Natm:2*Natm].min()).item()
    amp_psi_o = (x[:, 2*Natm:2*Natm+Noc].max() - x[:, 2*Natm:2*Natm+Noc].min()).item()
    amp_dT_o = (x[:, 2*Natm+Noc:].max() - x[:, 2*Natm+Noc:].min()).item()

    return {
        "max_value": max_val,
        "stationarity_ratio": stationarity,
        "temporal_autocorrelation": ac,
        "slow_mode_variance_fraction": slow_frac,
        "var_psi_a": var_psi_a,
        "var_theta_a": var_theta_a,
        "var_psi_o": var_psi_o,
        "var_dT_o": var_dT_o,
        "amp_psi_a": amp_psi_a,
        "amp_theta_a": amp_theta_a,
        "amp_psi_o": amp_psi_o,
        "amp_dT_o": amp_dT_o,
    }


def plot_timeseries_and_spectra(traj, output_dir, dt=0.1, Natm=36, Noc=16):
    """Plot 4-panel: timeseries + spectra for each variable block."""
    x = traj.numpy()
    labels = ["psi_a (atm barotropic)", "theta_a (atm baroclinic)",
              "psi_o (ocean)", "dT_o (ocean temp)"]
    slices = [slice(0, Natm), slice(Natm, 2*Natm),
              slice(2*Natm, 2*Natm+Noc), slice(2*Natm+Noc, None)]

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    time = np.arange(x.shape[0]) * dt

    for i, (lab, slc) in enumerate(zip(labels, slices)):
        data = x[:, slc]
        # Timeseries (first 3 modes)
        ax = axes[i, 0]
        for j in range(min(3, data.shape[1])):
            ax.plot(time[:2000], data[:2000, j], alpha=0.8, lw=0.8)
        ax.set_title(f"{lab} — timeseries")
        ax.set_xlabel("time")
        ax.set_ylabel("amplitude")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))

        # Spectrum
        ax = axes[i, 1]
        freqs = np.fft.rfftfreq(data.shape[0], d=dt)
        psd = np.mean(np.abs(np.fft.rfft(data, axis=0))**2, axis=1)
        mask = freqs > 0
        ax.loglog(freqs[mask], psd[mask], "k-", lw=0.8)
        ax.loglog(freqs[mask], 1e-2 * freqs[mask]**(-2), "r--", lw=0.8, label="$k^{-2}$")
        ax.loglog(freqs[mask], 1e-1 * freqs[mask]**(-5/3), "b--", lw=0.8, label="$k^{-5/3}$")
        ax.set_title(f"{lab} — spectrum")
        ax.set_xlabel("frequency")
        ax.set_ylabel("PSD")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "maooam_timeseries_spectra.png"), dpi=150)
    plt.close()
    print("  saved maooam_timeseries_spectra.png")


def plot_physical_fields(dynamics, traj, output_dir, interp_size=64):
    """Plot physical-field snapshots at different times, interpolated to finer grid."""
    times = [0, len(traj)//4, len(traj)//2, 3*len(traj)//4, len(traj)-1]
    fields_to_plot = ["psi_upper", "psi_oc", "T_atm", "T_oc"]

    fig, axes = plt.subplots(len(times), len(fields_to_plot), figsize=(16, 3.2*len(times)))

    for i, t in enumerate(times):
        phys = dynamics.spectral_to_physical(traj[t].numpy(), interp_size=interp_size)
        for j, fname in enumerate(fields_to_plot):
            ax = axes[i, j]
            data = phys[fname]
            vmax = max(abs(data.min()), abs(data.max()))
            if vmax < 1e-10:
                vmax = 1.0
            im = ax.imshow(data, cmap="RdBu_r", origin="lower",
                           vmin=-vmax, vmax=vmax)
            plt.colorbar(im, ax=ax, shrink=0.7, format="%.1e")
            if i == 0:
                ax.set_title(fname)
            if j == 0:
                ax.set_ylabel(f"t={t}", fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(f"MAOOAM physical fields ({interp_size}×{interp_size} interpolated)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "maooam_physical_fields.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved maooam_physical_fields.png")


def plot_slow_fast_separation(traj, output_dir, dt=0.1, Natm=36, Noc=16):
    """Plot slow (ocean) vs fast (atmosphere) time series."""
    x = traj.numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    time = np.arange(x.shape[0]) * dt

    # Atmosphere (fast)
    atm = x[:, :2*Natm]
    axes[0, 0].plot(time[:3000], atm[:3000, 0], alpha=0.8, lw=0.6, label="psi_a")
    axes[0, 0].set_title("Atmosphere (fast) — psi_a[0]")
    axes[0, 0].set_xlabel("time")

    axes[0, 1].plot(time[:3000], atm[:3000, Natm], alpha=0.8, lw=0.6, label="theta_a")
    axes[0, 1].set_title("Atmosphere (fast) — theta_a[0]")
    axes[0, 1].set_xlabel("time")

    # Ocean (slow)
    oc = x[:, 2*Natm:]
    axes[1, 0].plot(time[:3000], oc[:3000, 0], alpha=0.8, lw=0.6, label="psi_o")
    axes[1, 0].set_title("Ocean (slow) — psi_o[0]")
    axes[1, 0].set_xlabel("time")

    axes[1, 1].plot(time[:3000], oc[:3000, Noc], alpha=0.8, lw=0.6, label="dT_o")
    axes[1, 1].set_title("Ocean (slow) — dT_o[0]")
    axes[1, 1].set_xlabel("time")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "maooam_slow_fast_separation.png"), dpi=150)
    plt.close()
    print("  saved maooam_slow_fast_separation.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=50000)
    parser.add_argument("--output", type=str, default="reports/outputs/figs/maooam")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    dynamics = MaooamDynamics(dt=0.1, K=5)
    print(f"Dynamics: {dynamics}")

    traj = run_trajectory(dynamics, num_steps=args.num_steps, seed=args.seed)

    print("\n--- Metrics ---")
    m = compute_metrics(traj, dt=0.1, Natm=dynamics.Natm, Noc=dynamics.Npsi_o)
    for k, v in m.items():
        print(f"  {k}: {v:.6f}")

    print("\nGenerating figures...")
    plot_timeseries_and_spectra(traj, args.output, dt=0.1,
                                Natm=dynamics.Natm, Noc=dynamics.Npsi_o)
    plot_physical_fields(dynamics, traj, args.output)
    plot_slow_fast_separation(traj, args.output, dt=0.1,
                              Natm=dynamics.Natm, Noc=dynamics.Npsi_o)

    print("\nDone.")


if __name__ == "__main__":
    main()
