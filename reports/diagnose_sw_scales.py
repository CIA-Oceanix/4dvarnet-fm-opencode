#!/usr/bin/env python3
"""Diagnose spatial and temporal scales of the two-layer rotating shallow water system.

Generates trajectory snapshots, azimuthally-averaged spatial power spectra,
temporal autocorrelation plots, and prints a scale summary table.

Usage:
    conda run -n fdv python reports/diagnose_sw_scales.py
"""
import os, sys, textwrap
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIGS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

CMAP = "RdBu_r"

# ── Parameter sets ────────────────────────────────────────────────────

OLD_CFG = dict(
    label="Old (grid-scale noise)",
    Nx=64, Ny=64, dt=0.01,
    tau0=0.08, f_cor=0.1,
    g1=0.02, g2=0.01,
    coupling=0.05, friction=0.1, viscosity=0.001,
    num_steps=1000, spinup_steps=500,
)

NEW_CFG = dict(
    label="New (well-resolved eddies)",
    Nx=64, Ny=64, dt=0.1,
    tau0=0.01, f_cor=0.1,
    g1=0.5, g2=2.0,
    coupling=0.01, friction=0.1, viscosity=0.001,
    num_steps=2000, spinup_steps=500,
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
    """Azimuthally-averaged power spectrum of a 2D field.
    
    field_2d: (Nx, Ny) numpy array on a periodic domain.
    
    Returns (k_radial, power) where k_radial is in cycles per domain.
    """
    f = np.fft.fft2(field_2d)
    ps = np.abs(f) ** 2
    Nx, Ny = field_2d.shape
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


def wavelet_power_1d(series):
    """Simple periodogram-based power spectrum of a 1D time series."""
    n = len(series)
    fft = np.fft.rfft(series - series.mean())
    ps = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n)
    return freqs[1:], ps[1:]  # skip DC


def temporal_autocorr(series, max_lag):
    """Auto-correlation function of a 1D time series (Pearson)."""
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
    """Find lag where ACF crosses 1/e, or return None."""
    e_thresh = 1.0 / np.e
    for i in range(1, len(acf)):
        if acf[i] < e_thresh:
            return i
    return None


def plot_snapshots(traj, cfg, t_idx, label_suffix=""):
    """Figure 1: 3×2 grid of snapshot fields."""
    Nx, Ny = cfg["Nx"], cfg["Ny"]
    h1, u1, v1, h2, u2, v2 = field_snapshot(traj, t_idx, Nx, Ny)
    fields = [h1, u1, v1, h2, u2, v2]
    titles = [
        f"Ocean h₁ {label_suffix}", f"Ocean u₁ {label_suffix}", f"Ocean v₁ {label_suffix}",
        f"Atmos h₂ {label_suffix}", f"Atmos u₂ {label_suffix}", f"Atmos v₂ {label_suffix}",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, field, title in zip(axes.flat, fields, titles):
        vmax = max(abs(field.min()), abs(field.max()))
        norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
        im = ax.imshow(field, cmap=CMAP, norm=norm, origin="lower")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (grid)", fontsize=7)
        ax.set_ylabel("y (grid)", fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle(f"Snapshot at t={traj[t_idx, 0]:.3f}", fontsize=12)
    plt.tight_layout()
    return fig


def plot_spectra(traj, cfg, label_suffix="", traj2=None, cfg2=None):
    """Figure 2: Azimuthally-averaged spatial power spectra, old vs new."""
    Nx, Ny = cfg["Nx"], cfg["Ny"]
    names = ["h₁ (ocean)", "u₁ (ocean)", "v₁ (ocean)", "h₂ (atmos)", "u₂ (atmos)", "v₂ (atmos)"]
    time_avg = traj[100:].mean(axis=0)  # average after spinup
    Nxy = Nx * Ny
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    
    configs = [(cfg, traj, "-")]
    label_map = [cfg["label"]]
    if traj2 is not None and cfg2 is not None:
        configs.append((cfg2, traj2, "--"))
        label_map.append(cfg2["label"])
    
    for idx, (ax, name) in enumerate(zip(axes.flat, names)):
        c_idx = idx
        for ncfg, ntraj, ls in configs:
            nNx, nNy = ncfg["Nx"], ncfg["Ny"]
            nNxy = nNx * nNy
            # Average spectrum over last 50 timesteps
            ps_all = []
            for t in range(len(ntraj) - 50, len(ntraj)):
                field = ntraj[t, c_idx * nNxy : (c_idx + 1) * nNxy].reshape(nNx, nNy)
                k, p = azimuthal_spectrum_2d(field)
                ps_all.append(p)
            ps_mean = np.mean(ps_all, axis=0)
            # Convert wavenumber to physical scale k_phys = k / L
            L = nNx
            ax.loglog(k / L, ps_mean, ls, label=ncfg["label"])
        
        ax.set_xlabel("Wavenumber (cycles / grid unit)", fontsize=8)
        ax.set_ylabel("Power", fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, ls="--")
    plt.suptitle(f"Spatial power spectra (azimuthally averaged) {label_suffix}", fontsize=12)
    plt.tight_layout()
    return fig


def plot_temporal_autocorr(traj, cfg, label_suffix="", traj2=None, cfg2=None):
    """Figure 3: Temporal autocorrelation at selected spatial points."""
    Nx, Ny = cfg["Nx"], cfg["Ny"]
    names = ["h₁ (ocean)", "u₁ (ocean)", "v₁ (ocean)", "h₂ (atmos)", "u₂ (atmos)", "v₂ (atmos)"]
    Nxy = Nx * Ny
    max_lag = min(200, len(traj) // 4)
    
    # Pick 3 sample points: center, edge, quarter
    pts = [(Nx//2, Ny//2), (0, 0), (Nx//4, Ny//4)]
    pt_labels = ["center", "corner", "quarter"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    
    configs = [(cfg, traj, "-")]
    if traj2 is not None and cfg2 is not None:
        configs.append((cfg2, traj2, "--"))
    
    for idx, (ax, name) in enumerate(zip(axes.flat, names)):
        c_idx = idx
        for ncfg, ntraj, ls in configs:
            nNx, nNy = ncfg["Nx"], ncfg["Ny"]
            nNxy = nNx * nNy
            ntime = len(ntraj)
            time_axis = np.arange(ntime) * ncfg["dt"]
            for pt, ptl in zip(pts, pt_labels):
                fi = pt[0] * nNy + pt[1]
                series = ntraj[:, c_idx * nNxy + fi]
                series_smooth = uniform_filter1d(series, size=5)
                acf = temporal_autocorr(series_smooth, max_lag)
                lag_time = np.arange(max_lag + 1) * ncfg["dt"]
                ax.plot(lag_time, acf, ls, lw=0.8 if ptl == "center" else 0.5, alpha=0.8 if ptl == "center" else 0.5)
        
        ax.axhline(1/np.e, color="gray", ls=":", lw=0.8, label="1/e")
        ax.set_xlabel("Lag (model time)", fontsize=8)
        ax.set_ylabel("Autocorrelation", fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3, ls="--")
        ax.set_ylim(-0.5, 1.05)
    plt.suptitle(f"Temporal autocorrelation (3 sample points) {label_suffix}", fontsize=12)
    plt.tight_layout()
    return fig


def compute_scales_table(traj, cfg):
    """Compute analytical and empirical scales and return as formatted string."""
    Nx, Ny = cfg["Nx"], cfg["Ny"]
    Nxy = Nx * Ny
    dt = cfg["dt"]
    f = cfg["f_cor"]
    g1, g2 = cfg["g1"], cfg["g2"]
    
    # Analytical
    c1 = np.sqrt(g1)
    c2 = np.sqrt(g2)
    Rd1 = c1 / f
    Rd2 = c2 / f
    T_f = 2 * np.pi / f
    T_f_steps = T_f / dt
    
    # Empirical from trajectory
    h1 = traj[:, 0:Nxy]
    h2 = traj[:, 3*Nxy:4*Nxy]
    u1 = traj[:, Nxy:2*Nxy]
    u2 = traj[:, 4*Nxy:5*Nxy]
    
    # Temporal autocorrelation e-folding (use large lag for slow dynamics)
    max_lag = min(2000, len(traj) // 3)
    center_idx = (Nx // 2) * Ny + (Ny // 2)
    acf_h1 = temporal_autocorr(uniform_filter1d(h1[:, center_idx], size=5), max_lag)
    acf_h2 = temporal_autocorr(uniform_filter1d(h2[:, center_idx], size=5), max_lag)
    acf_u1 = temporal_autocorr(uniform_filter1d(u1[:, center_idx], size=5), max_lag)
    acf_u2 = temporal_autocorr(uniform_filter1d(u2[:, center_idx], size=5), max_lag)
    
    tau_h1 = compute_e_folding(acf_h1)
    tau_h2 = compute_e_folding(acf_h2)
    tau_u1 = compute_e_folding(acf_u1)
    tau_u2 = compute_e_folding(acf_u2)
    
    # Empirical dominant spatial scale from spectrum
    k_all, ps_all = [], []
    for c_idx in range(6):
        field = traj[-1, c_idx * Nxy : (c_idx + 1) * Nxy].reshape(Nx, Ny)
        k, p = azimuthal_spectrum_2d(field)
        k_all.append(k / Nx)
        ps_all.append(p)
    
    # Dominant wavenumber (peak of power spectrum, excluding k=0 and k=1)
    def dom_scale(k_arr, p_arr):
        start = 3
        if len(p_arr) <= start:
            return 0, 0
        idx = np.argmax(p_arr[start:]) + start
        if k_arr[idx] < 1e-10:
            return 0, 0
        return 1.0 / k_arr[idx] if k_arr[idx] > 0 else 0, k_arr[idx]
    
    dom_1 = dom_scale(k_all[0], ps_all[0])
    dom_4 = dom_scale(k_all[3], ps_all[3])
    
    # State statistics
    lines = []
    lines.append(f"--- {cfg['label']} ---")
    lines.append(f"Parameters: g1={g1}, g2={g2}, f={f}, dt={dt}")
    lines.append(f"")
    lines.append(f"  Analytical scales:")
    lines.append(f"    Gravity wave speed:       c₁ = {c1:.3f}, c₂ = {c2:.3f}")
    lines.append(f"    Rossby deformation radius: Rd₁ = {Rd1:.1f} dx, Rd₂ = {Rd2:.1f} dx")
    lines.append(f"    Rd / Domain:              Rd₁/L = {Rd1/Nx:.3f}, Rd₂/L = {Rd2/Nx:.3f}")
    lines.append(f"    Inertial period:           T_f = {T_f:.1f} ({T_f_steps:.0f} steps)")
    lines.append(f"")
    lines.append(f"  State statistics (spatial std, time-averaged):")
    for c_idx, name in enumerate(["h₁", "u₁", "v₁", "h₂", "u₂", "v₂"]):
        fld = traj[:, c_idx * Nxy : (c_idx + 1) * Nxy]
        mu = fld.mean()
        sigma = fld.std()
        lines.append(f"    {name:8s}:  μ = {mu:.4f},  σ = {sigma:.4f}")
    lines.append(f"")
    def fmt_tau(t):
        if t is None:
            return ">max (>max τ)"
        return f"{t} steps ({t*dt:.2f} τ)"
    
    lines.append(f"  Temporal e-folding (autocorrelation 1/e):")
    lines.append(f"    Ocean:      h₁ = {fmt_tau(tau_h1)},  u₁ = {fmt_tau(tau_u1)}")
    lines.append(f"    Atmosphere: h₂ = {fmt_tau(tau_h2)},  u₂ = {fmt_tau(tau_u2)}")
    lines.append(f"")
    lines.append(f"  Dominant spatial scale (from power spectrum peak):")
    lines.append(f"    Ocean h₁:     {dom_1[0]:.1f} dx  (k={dom_1[1]:.2f})")
    lines.append(f"    Atmosphere h₂: {dom_4[0]:.1f} dx  (k={dom_4[1]:.2f})")
    lines.append(f"")
    if dom_1[1] > 0 and dom_4[1] > 0:
        lines.append(f"  → Ocean eddies are {dom_1[0]/dom_4[0]:.1f}× the scale of atmosphere eddies (in grid units)")
    lines.append(f"")
    
    return "\n".join(lines)


def plot_side_by_side(traj_old, cfg_old, traj_new, cfg_new):
    """Figure 4: Old vs new h1 snapshot comparison."""
    Nx, Ny = cfg_old["Nx"], cfg_old["Ny"]
    h1_old = traj_old[-1, 0:Nx*Ny].reshape(Nx, Ny)
    
    nNx, nNy = cfg_new["Nx"], cfg_new["Ny"]
    h1_new = traj_new[-1, 0:nNx*nNy].reshape(nNx, nNy)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Old
    vmax_old = max(abs(h1_old.min()), abs(h1_old.max()))
    im0 = axes[0].imshow(h1_old, cmap=CMAP, norm=TwoSlopeNorm(vcenter=0, vmin=-vmax_old, vmax=vmax_old), origin="lower")
    axes[0].set_title(f"Old config: h₁ (Rd≈{np.sqrt(cfg_old['g1'])/cfg_old['f_cor']:.0f} dx)", fontsize=10)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # New
    vmax_new = max(abs(h1_new.min()), abs(h1_new.max()))
    im1 = axes[1].imshow(h1_new, cmap=CMAP, norm=TwoSlopeNorm(vcenter=0, vmin=-vmax_new, vmax=vmax_new), origin="lower")
    axes[1].set_title(f"New config: h₁ (Rd≈{np.sqrt(cfg_new['g1'])/cfg_new['f_cor']:.0f} dx)", fontsize=10)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.suptitle("Ocean layer thickness h₁ — old vs new parameters", fontsize=12)
    plt.tight_layout()
    return fig


def print_scale_comparison(tables):
    print("=" * 72)
    print("  TWO-LAYER ROTATING SHALLOW WATER — SCALE DIAGNOSTICS")
    print("=" * 72)
    print()
    for tbl in tables:
        print(tbl)
        print("-" * 60)
    print()


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  SW Scale Diagnostics")
    print("  Figures →", FIGS_DIR)
    print("=" * 72)
    print()
    
    all_tables = []
    
    for cfg in [OLD_CFG, NEW_CFG]:
        print(f"\n--- {cfg['label']} ---")
        dyn = build_dynamics(cfg)
        traj, forcing = generate_trajectory(dyn, cfg)
        
        # Snapshot at last timestep
        print("  Plotting snapshots ...")
        fig = plot_snapshots(traj, cfg, t_idx=-1, label_suffix=f"[{cfg['label']}]")
        tag = "old" if "Old" in cfg["label"] else "new"
        fig.savefig(os.path.join(FIGS_DIR, f"sw_snapshots_{tag}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        # Spectra
        print("  Plotting spectra ...")
        fig = plot_spectra(traj, cfg, label_suffix=f"[{cfg['label']}]")
        fig.savefig(os.path.join(FIGS_DIR, f"sw_spectra_{tag}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        # Temporal autocorrelation
        print("  Plotting autocorrelation ...")
        fig = plot_temporal_autocorr(traj, cfg, label_suffix=f"[{cfg['label']}]")
        fig.savefig(os.path.join(FIGS_DIR, f"sw_autocorr_{tag}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        # Compute scales table
        table = compute_scales_table(traj, cfg)
        all_tables.append(table)
        
        print(f"  Done with {cfg['label']}")
    
    # Side-by-side comparison
    print("\nPlotting old-vs-new comparison ...")
    dyn_old = build_dynamics(OLD_CFG)
    traj_old, _ = generate_trajectory(dyn_old, OLD_CFG, seed=42)
    dyn_new = build_dynamics(NEW_CFG)
    traj_new, _ = generate_trajectory(dyn_new, NEW_CFG, seed=42)
    fig = plot_side_by_side(traj_old, OLD_CFG, traj_new, NEW_CFG)
    fig.savefig(os.path.join(FIGS_DIR, "sw_old_vs_new.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Combined spectra for both (overlay old + new in same plot)
    print("Plotting combined spectra ...")
    fig = plot_spectra(traj_old, OLD_CFG, label_suffix="", traj2=traj_new, cfg2=NEW_CFG)
    fig.savefig(os.path.join(FIGS_DIR, "sw_spectra_combined.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Combined autocorrelation
    print("Plotting combined autocorrelation ...")
    fig = plot_temporal_autocorr(traj_old, OLD_CFG, label_suffix="", traj2=traj_new, cfg2=NEW_CFG)
    fig.savefig(os.path.join(FIGS_DIR, "sw_autocorr_combined.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Print scales comparison
    print_scale_comparison(all_tables)
    
    print(f"\n{'='*72}")
    print(f"  All figures saved to {FIGS_DIR}/")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()