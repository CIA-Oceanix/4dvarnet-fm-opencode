import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.qg_dynamics import QGDynamics

NOMINAL = {"U1": 0.05, "U2": 0.0, "rd": 15000.0, "beta": 1.5e-11,
           "delta": 0.25, "rek": 5.787e-7, "dt": 7200.0}

AMPS = [0.0, 1e-11, 3e-11]
LABELS = {0.0: "unforced", 1e-11: "default (1e-11, +37%)",
          3e-11: "strong (3e-11, +290%)"}
STRONG = 3e-11


def _ke_timeseries(dyn: QGDynamics, traj: torch.Tensor, mono: int,
                   device: torch.device) -> np.ndarray:
    vals = []
    for k in range(0, traj.shape[0], mono):
        state = dyn._flatten(traj[k:k + 1]).squeeze(0)
        vals.append(dyn.kinetic_energy(state.to(device)).item())
    return np.asarray(vals)


def _ke_spectrum(dyn: QGDynamics, state: torch.Tensor, device: torch.device
                 ) -> tuple[np.ndarray, np.ndarray]:
    q = dyn._grid(state.to(device))
    if q.dim() == 3:
        q = q.unsqueeze(0)
    qh = torch.fft.rfft2(q, dim=(-2, -1))
    ph = dyn._invert(qh)
    M2 = float(dyn.nx * dyn.ny) ** 2
    H1 = 500.0
    H2 = H1 / dyn.delta
    ke1 = 0.5 * H1 * (dyn.K2 * ph[..., 0, :, :].abs() ** 2) / M2
    ke2 = 0.5 * H2 * (dyn.K2 * ph[..., 1, :, :].abs() ** 2) / M2
    ke = ((ke1 + ke2) / (H1 + H2))[0].cpu().numpy()

    nx, ny = dyn.nx, dyn.ny
    dk = 2.0 * np.pi / dyn.L
    dl = 2.0 * np.pi / dyn.W
    kk = dk * np.arange(nx // 2 + 1)
    ll = np.concatenate([np.arange(0.0, ny // 2), np.arange(-ny // 2, 0.0)])
    k2d = np.broadcast_to(kk[None, :], (ny, nx // 2 + 1))
    l2d = np.broadcast_to(ll[:, None], (ny, nx // 2 + 1))
    kr = np.sqrt(k2d ** 2 + l2d ** 2)

    nbins = 24
    kmax = float(np.sqrt((dk * nx / 2) ** 2 + (dl * ny / 2) ** 2))
    edges = np.linspace(0.0, kmax, nbins + 1)
    mask = kr > 0
    idx = np.searchsorted(edges, kr[mask], side="right") - 1
    idx = np.clip(idx, 0, nbins - 1)
    spec = np.zeros(nbins)
    counts = np.zeros(nbins)
    np.add.at(spec, idx, ke[mask] * kr[mask])
    np.add.at(counts, idx, 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    spec = np.where(counts > 0, spec / np.maximum(counts, 1.0), 0.0)
    return centers, spec


def _wind_work(dyn: QGDynamics, traj: torch.Tensor,
               wind_state: torch.Tensor, mono: int,
               device: torch.device) -> np.ndarray:
    vals = []
    for k in range(0, traj.shape[0], mono):
        psi = dyn.streamfunctions(traj[k:k + 1].to(device))
        curl = dyn.wind_curl_field(wind_state[k:k + 1].to(device))
        vals.append((curl[0] * psi[0, 0]).mean().item())
    return np.asarray(vals)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assess the impact of QG wind forcing (combined figure).")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default="reports/qg/outputs/figs")
    parser.add_argument("--spinup-years", type=float, default=2.0)
    parser.add_argument("--wind-days", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mono-days", type=float, default=5.0)
    parser.add_argument("--anom-days", nargs="+", type=float,
                        default=[30.0, 60.0, 90.0])
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    steps_per_day = round(86400.0 / NOMINAL["dt"])
    spinup_steps = int(args.spinup_years * 365.0 * steps_per_day)
    num_steps = int(args.wind_days * steps_per_day)
    mono = max(1, int(args.mono_days * steps_per_day))
    anom_idx = [int(d * steps_per_day) for d in args.anom_days]
    print(f"device={device} spinup_steps={spinup_steps} num_steps={num_steps}")

    dyns = {amp: QGDynamics(nx=args.nx, wind_amp=amp,
                            wind_seed=args.seed, **NOMINAL).to(device)
            for amp in AMPS}
    t_days = np.arange(0.0, args.wind_days, args.mono_days)

    ke_series = {}
    trajs = {}
    wind_states = {}
    for amp in AMPS:
        t0 = time.time()
        traj, ws = dyns[amp].generate_full_trajectory(
            num_steps=num_steps, seed=args.seed, spinup_steps=spinup_steps)
        trajs[amp] = traj.cpu()
        wind_states[amp] = ws.cpu()
        ke_series[amp] = _ke_timeseries(dyns[amp], trajs[amp], mono, device)
        print(f"amp={amp:.1e} KE start={ke_series[amp][0]:.3e} "
              f"end={ke_series[amp][-1]:.3e} mean={ke_series[amp].mean():.3e} "
              f"({time.time()-t0:.0f}s)")

    spectra = {}
    for amp in AMPS:
        state = dyns[amp]._flatten(trajs[amp][-1:]).squeeze(0)
        spectra[amp] = _ke_spectrum(dyns[amp], state.to(device), device)

    work = _wind_work(dyns[STRONG], trajs[STRONG], wind_states[STRONG],
                      mono, device)

    fig = plt.figure(figsize=(14, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.55, 0.5],
                          left=0.09, right=0.97, top=0.95, bottom=0.06,
                          hspace=0.55, wspace=0.32)

    ax_a = fig.add_subplot(gs[0, 0])
    for amp in AMPS:
        ax_a.semilogy(t_days[:len(ke_series[amp])],
                      np.maximum(ke_series[amp], 1e-12), label=LABELS[amp])
    ax_a.set_xlabel("time in forced window [days]")
    ax_a.set_ylabel("domain-mean KE [m$^2$/s$^2$]")
    ax_a.set_title("(a) kinetic energy time series")
    ax_a.legend(fontsize=7)

    ax_b = fig.add_subplot(gs[0, 1])
    for amp in AMPS:
        centers, spec = spectra[amp]
        m = spec > 0
        ax_b.loglog(centers[m], spec[m], ".-", lw=1.0, ms=4, label=LABELS[amp])
    km = 1.0 / (2.5e5)
    k = np.linspace(km, 2e-4, 2)
    ax_b.plot(k, 1e6 * (k / km) ** -3, "k--", lw=0.8, label=r"$\sim k^{-3}$")
    ax_b.set_xlabel("total wavenumber $k$ [m$^{-1}$]")
    ax_b.set_ylabel("KE spectral density [a.u.]")
    ax_b.set_title("(b) KE spectrum (isotropized)")
    ax_b.legend(fontsize=7)

    import matplotlib.gridspec as mgs
    gs_c = mgs.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, :],
                                       wspace=0.05)
    refq = dyns[0]._grid(trajs[0])[..., 0, :, :].numpy()
    anom_axes = []
    for i, k in enumerate(anom_idx):
        qf = dyns[STRONG]._grid(trajs[STRONG][k:k + 1])[0, 0].numpy()
        anom = qf - refq[k]
        ax = fig.add_subplot(gs_c[i])
        anom_axes.append(ax)
        vmax = max(np.abs(anom).max(), 1e-30)
        im = ax.imshow(anom, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"q1 anomaly day {args.anom_days[i]:.0f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=anom_axes, fraction=0.025, pad=0.02)
    fig.text(0.5, 0.52, "(c) upper-layer PV anomaly  q1_forced − q1_unforced  "
             "(strong 3e-11)", ha="center", fontsize=10)

    ax_d = fig.add_subplot(gs[2, :])
    ax_d.plot(t_days[:len(work)], work, label="wind work (strong 3e-11)")
    ax_d.axhline(float(work.mean()), color="k", ls="--",
                 label=f"time-mean {work.mean():.1e}")
    ax_d.set_xlabel("time in forced window [days]")
    ax_d.set_ylabel(r"$\langle \tau_{curl}\,\psi_1\rangle$ [m$^3$ s$^{-3}$]")
    ax_d.set_title("(d) domain-mean wind work")
    ax_d.legend(fontsize=7)

    out = os.path.join(args.out_dir, "qg_wind_impact.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
