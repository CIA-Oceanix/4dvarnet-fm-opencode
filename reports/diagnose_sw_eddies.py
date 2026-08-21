#!/usr/bin/env python3
"""Diagnose whether the two-layer rotating SW system produces 2D eddies
with reduced friction (0.001) and extended spinup (8000 steps).

Generates:
  - Snapshots at 3 timesteps (mid, late, final)
  - Azimuthally-averaged spatial power spectra
  - Temporal autocorrelation
  - Scale summary table
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIGS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

CMAP = "RdBu_r"

CFG = dict(
    label="Low friction (0.001)",
    Nx=64, Ny=64, dt=0.1,
    tau0=0.01, f_cor=0.1,
    g1=1.0, g2=4.0,
    coupling=0.01, friction=0.001, viscosity=0.001,
    num_steps=4000, spinup_steps=8000,
)


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
    print(f"  Generating {cfg['num_steps']} steps (spinup={cfg['spinup_steps']}) ...")
    traj, forcing = dyn.generate_full_trajectory(
        num_steps=cfg["num_steps"], seed=seed,
        spinup_steps=cfg["spinup_steps"],
    )
    return traj.numpy(), forcing.numpy()


def reshape_field(field_1d, Nx, Ny):
    return field_1d.reshape(Nx, Ny)


def field_snapshot(traj, t_idx, Nx, Ny):
    Nxy = Nx * Ny
    h1 = reshape_field(traj[t_idx, 0:Nxy], Nx, Ny)
    u1 = reshape_field(traj[t_idx, Nxy:2*Nxy], Nx, Ny)
    v1 = reshape_field(traj[t_idx, 2*Nxy:3*Nxy], Nx, Ny)
    h2 = reshape_field(traj[t_idx, 3*Nxy:4*Nxy], Nx, Ny)
    u2 = reshape_field(traj[t_idx, 4*Nxy:5*Nxy], Nx, Ny)
    v2 = reshape_field(traj[t_idx, 5*Nxy:6*Nxy], Nx, Ny)
    return h1, u1, v1, h2, u2, v2


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


def temporal_autocorr(series, max_lag):
    mu = series.mean()
    var = series.var()
    if var < 1e-15:
        return np.ones(max_lag + 1)
    acf = np.ones(max_lag + 1)
    for lag in range(1, max_lag + 1):
        c = ((series[:-lag] - mu) * (series[lag:] - mu)).mean() / var
        acf[lag] = c
    return acf


def compute_e_folding(acf):
    e_thresh = 1.0 / np.e
    for i in range(1, len(acf)):
        if acf[i] < e_thresh:
            return i
    return None


def plot_snapshots_multi(traj, cfg, t_indices, label_suffix=""):
    Nx, Ny = cfg["Nx"], cfg["Ny"]
    nrows = len(t_indices)
    fig, axes = plt.subplots(nrows, 6, figsize=(18, 4 * nrows))
    axes = axes.reshape(nrows, 6)
    row_titles = [f"Step {t} (t={traj[t,0]:.2f})" for t in t_indices]
    col_titles = ["h₁ ocean", "u₁ ocean", "v₁ ocean", "h₂ atmos", "u₂ atmos", "v₂ atmos"]

    for row, (t_idx, rtitle) in enumerate(zip(t_indices, row_titles)):
        h1, u1, v1, h2, u2, v2 = field_snapshot(traj, t_idx, Nx, Ny)
        fields = [h1, u1, v1, h2, u2, v2]
        for col in range(6):
            ax = axes[row, col]
            field = fields[col]
            vmax = max(abs(field.min()), abs(field.max())) + 1e-10
            norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
            im = ax.imshow(field, cmap=CMAP, norm=norm, origin="lower")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(rtitle, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)

    plt.suptitle(f"Snapshots — {cfg['label']} {label_suffix}", fontsize=12, y=1.01)
    plt.tight_layout()
    return fig


def plot_spectra(traj, cfg, label_suffix=""):
    Nx, Ny = cfg["Nx"], cfg["Ny"]
    names = ["h₁ (ocean)", "u₁ (ocean)", "v₁ (ocean)", "h₂ (atmos)", "u₂ (atmos)", "v₂ (atmos)"]
    Nxy = Nx * Ny
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for idx, (ax, name) in enumerate(zip(axes.flat, names)):
        ps_all = []
        for t in range(len(traj) - 50, len(traj)):
            field = traj[t, idx * Nxy : (idx + 1) * Nxy].reshape(Nx, Ny)
            k, p = azimuthal_spectrum_2d(field)
            ps_all.append(p)
        ps_mean = np.mean(ps_all, axis=0)
        L = Nx
        ax.loglog(k / L, ps_mean, "-", label=cfg["label"])
        ax.set_xlabel("Wavenumber (cycles / grid unit)", fontsize=8)
        ax.set_ylabel("Power", fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, ls="--")
    plt.suptitle(f"Spatial power spectra {label_suffix}", fontsize=12)
    plt.tight_layout()
    return fig


def plot_spectra_combined(traj_low, cfg_low, traj_high, cfg_high, label_low="low", label_high="high"):
    Nx, Ny = cfg_low["Nx"], cfg_low["Ny"]
    names = ["h₁ (ocean)", "u₁ (ocean)", "v₁ (ocean)", "h₂ (atmos)", "u₂ (atmos)", "v₂ (atmos)"]
    Nxy = Nx * Ny
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    configs = [(cfg_low, traj_low, "-", label_low), (cfg_high, traj_high, "--", label_high)]

    for idx, (ax, name) in enumerate(zip(axes.flat, names)):
        for ncfg, ntraj, ls, lbl in configs:
            nNx, nNy = ncfg["Nx"], ncfg["Ny"]
            nNxy = nNx * nNy
            ps_all = []
            for t in range(len(ntraj) - 50, len(ntraj)):
                field = ntraj[t, idx * nNxy : (idx + 1) * nNxy].reshape(nNx, nNy)
                k, p = azimuthal_spectrum_2d(field)
                ps_all.append(p)
            ps_mean = np.mean(ps_all, axis=0)
            L = nNx
            ax.loglog(k / L, ps_mean, ls, label=lbl)
        ax.set_xlabel("Wavenumber (cycles / grid unit)", fontsize=8)
        ax.set_ylabel("Power", fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, ls="--")
    plt.suptitle("Spatial power spectra — low vs high friction", fontsize=12)
    plt.tight_layout()
    return fig


def plot_temporal_autocorr(traj, cfg, label_suffix=""):
    Nx, Ny = cfg["Nx"], cfg["Ny"]
    names = ["h₁ (ocean)", "u₁ (ocean)", "v₁ (ocean)", "h₂ (atmos)", "u₂ (atmos)", "v₂ (atmos)"]
    Nxy = Nx * Ny
    max_lag = min(500, len(traj) // 4)

    pts = [(Nx//2, Ny//2), (0, 0), (Nx//4, Ny//4)]
    pt_labels = ["center", "corner", "quarter"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for idx, (ax, name) in enumerate(zip(axes.flat, names)):
        ntime = len(traj)
        for pt, ptl in zip(pts, pt_labels):
            fi = pt[0] * Ny + pt[1]
            series = traj[:, idx * Nxy + fi]
            series_smooth = uniform_filter1d(series, size=5)
            acf = temporal_autocorr(series_smooth, max_lag)
            lag_time = np.arange(max_lag + 1) * cfg["dt"]
            ax.plot(lag_time, acf, lw=0.8 if ptl == "center" else 0.5,
                    alpha=0.8 if ptl == "center" else 0.5, label=ptl)
        ax.axhline(1/np.e, color="gray", ls=":", lw=0.8, label="1/e")
        ax.set_xlabel("Lag (model time)", fontsize=8)
        ax.set_ylabel("Autocorrelation", fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3, ls="--")
        ax.set_ylim(-0.5, 1.05)
    plt.suptitle(f"Temporal autocorrelation {label_suffix}", fontsize=12)
    plt.tight_layout()
    return fig


def compute_scales_table(traj, cfg):
    Nx, Ny = cfg["Nx"], cfg["Ny"]
    Nxy = Nx * Ny
    dt = cfg["dt"]
    f = cfg["f_cor"]
    g1, g2 = cfg["g1"], cfg["g2"]

    c1 = np.sqrt(g1)
    c2 = np.sqrt(g2)
    Rd1 = c1 / f
    Rd2 = c2 / f
    T_f = 2 * np.pi / f
    T_f_steps = T_f / dt

    max_lag = min(2000, len(traj) // 3)
    center_idx = (Nx // 2) * Ny + (Ny // 2)

    lines = []
    lines.append(f"--- {cfg['label']} ---")
    lines.append(f"Parameters: g1={g1}, g2={g2}, f={f}, dt={dt}, friction={cfg['friction']}, spinup={cfg['spinup_steps']}")
    lines.append("")
    lines.append("  Analytical scales:")
    lines.append(f"    Gravity wave speed:       c1 = {c1:.3f}, c2 = {c2:.3f}")
    lines.append(f"    Rossby deformation radius: Rd1 = {Rd1:.1f} dx, Rd2 = {Rd2:.1f} dx")
    lines.append(f"    Rd / Domain:              Rd1/L = {Rd1/Nx:.3f}, Rd2/L = {Rd2/Nx:.3f}")
    lines.append(f"    Inertial period:          T_f = {T_f:.1f} ({T_f_steps:.0f} steps)")
    lines.append(f"    Frictional damping:       1/r = {1.0/cfg['friction']:.1f} tu ({1.0/(cfg['friction']*T_f):.1f} T_f)")
    lines.append("")

    lines.append("  State statistics (spatial std, time-averaged over last 1000 steps):")
    for c_idx, name in enumerate(["h1", "u1", "v1", "h2", "u2", "v2"]):
        fld = traj[-1000:, c_idx * Nxy : (c_idx + 1) * Nxy]
        mu = fld.mean()
        sigma = fld.std()
        lines.append(f"    {name:8s}:  mu = {mu:.4f},  sigma = {sigma:.4f}")
    lines.append("")

    for c_idx, name, label in [
        (0, "h1", "Ocean h1"), (3, "h2", "Atmos h2"),
        (1, "u1", "Ocean u1"), (4, "u2", "Atmos u2"),
    ]:
        fld = traj[:, c_idx * Nxy + center_idx]
        acf = temporal_autocorr(uniform_filter1d(fld, size=5), max_lag)
        tau = compute_e_folding(acf)
        tau_str = f"{tau} steps ({tau*dt:.1f} tu)" if tau else ">max"
        lines.append(f"    {label:12s} e-fold: {tau_str}")

    k_all, ps_all = [], []
    for c_idx in range(6):
        field = traj[-1, c_idx * Nxy : (c_idx + 1) * Nxy].reshape(Nx, Ny)
        k, p = azimuthal_spectrum_2d(field)
        k_all.append(k / Nx)
        ps_all.append(p)

    def dom_scale(k_arr, p_arr):
        start = 3
        if len(p_arr) <= start:
            return 0, 0
        idx = np.argmax(p_arr[start:]) + start
        if k_arr[idx] < 1e-10:
            return 0, 0
        return 1.0 / k_arr[idx] if k_arr[idx] > 0 else 0, k_arr[idx]

    lines.append("")
    lines.append("  Dominant spatial scale (from power spectrum peak, excluding k<3):")
    for c_idx, name in enumerate(["h1", "u1", "v1", "h2", "u2", "v2"]):
        dom = dom_scale(k_all[c_idx], ps_all[c_idx])
        lines.append(f"    {name:8s}:  {dom[0]:.1f} dx  (k={dom[1]:.2f})")

    return "\n".join(lines)


def main():
    print("=" * 72)
    print("  SW Eddy Diagnostics — low friction test")
    print("  Figures ->", FIGS_DIR)
    print("=" * 72)

    print("\n--- Building dynamics ---")
    dyn = build_dynamics(CFG)
    traj, forcing = generate_trajectory(dyn, CFG)

    tot_steps = CFG["spinup_steps"] + CFG["num_steps"]
    print(f"  Total steps generated: {tot_steps}")
    print(f"  Trajectory shape: {traj.shape}")
    print(f"  Trajectory finite: {np.all(np.isfinite(traj))}")

    # Snapshots at 3 timesteps
    t_mid = len(traj) // 2
    t_late = len(traj) - 500
    t_final = -1
    print(f"\n  Plotting snapshots at steps {t_mid}, {t_late}, {t_final} ...")
    fig = plot_snapshots_multi(traj, CFG, [t_mid, t_late, t_final],
                               label_suffix=f"[{CFG['label']}]")
    fig.savefig(os.path.join(FIGS_DIR, "sw_eddy_snapshots.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved sw_eddy_snapshots.png")

    # Compare with the high-friction (old NEW_CFG) trajectory for combined spectra
    print("\n  Re-running high-friction (friction=0.1) for comparison ...")
    HIGH_CFG = dict(
        label="High friction (0.1)",
        Nx=64, Ny=64, dt=0.1,
        tau0=0.01, f_cor=0.1,
g1=1.0, g2=4.0,
        coupling=0.01, friction=0.1, viscosity=0.001,
        num_steps=2000, spinup_steps=500,
    )
    dyn_high = build_dynamics(HIGH_CFG)
    traj_high, _ = generate_trajectory(dyn_high, HIGH_CFG, seed=42)

    fig = plot_spectra_combined(traj, CFG, traj_high, HIGH_CFG,
                                label_low="friction=0.001", label_high="friction=0.1")
    fig.savefig(os.path.join(FIGS_DIR, "sw_eddy_spectra_combined.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved sw_eddy_spectra_combined.png")

    # High-friction snapshot for comparison
    fig = plot_snapshots_multi(traj_high, HIGH_CFG, [-1],
                               label_suffix="[friction=0.1, final step]")
    fig.savefig(os.path.join(FIGS_DIR, "sw_eddy_highfric_snapshot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved sw_eddy_highfric_snapshot.png")

    # Low-friction standalone spectra
    print("\n  Plotting spectra (low friction) ...")
    fig = plot_spectra(traj, CFG, label_suffix=f"[{CFG['label']}]")
    fig.savefig(os.path.join(FIGS_DIR, "sw_eddy_spectra.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved sw_eddy_spectra.png")

    # Temporal autocorrelation
    print("  Plotting autocorrelation ...")
    fig = plot_temporal_autocorr(traj, CFG, label_suffix=f"[{CFG['label']}]")
    fig.savefig(os.path.join(FIGS_DIR, "sw_eddy_autocorr.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved sw_eddy_autocorr.png")

    # Scales table
    print("\n" + "=" * 60)
    table = compute_scales_table(traj, CFG)
    print(table)
    print("=" * 60)

    print(f"\n{'='*72}")
    print(f"  Done. All figures in {FIGS_DIR}/")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()