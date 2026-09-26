"""Simulate the two-layer QG ocean under the spectral wind drivers.

Runs the current moving storm ("ou") and the Option B spectral drivers
("spectral_ou", "gyrostat", "gyrostat_surrogate") from one shared, unforced
spin-up, then writes per-driver animations and figures in the style of the
QG S0/S1 report, a cross-driver comparison figure and a markdown report.

Until the QG dynamics hooks land, the spectral PV source enters through a
local `QGDynamics` subclass that overrides `_wind_curl_spectral`.
"""
import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models.gyrostat_driver import get_preset  # noqa: E402
from models.qg_dynamics import QGDynamics  # noqa: E402
from models.qg_wind_modes import (  # noqa: E402
    FourierWindBasis,
    default_time_unit_days,
    generate_spectral_wind,
)

CMAP = "RdBu_r"
DRIVER_LABELS = {
    "ou": "E0: moving storm (current)",
    "spectral_ou": "E0b: spectral OU",
    "gyrostat": "E1: gyrostat",
    "gyrostat_surrogate": "E2: phase-randomized surrogate",
}
DRIVER_COLORS = {"ou": "#555555", "spectral_ou": "#1f77b4",
                 "gyrostat": "#d62728", "gyrostat_surrogate": "#2ca02c"}


class SpectralWindQG(QGDynamics):
    def __init__(self, basis: FourierWindBasis, **kwargs):
        super().__init__(**kwargs)
        self.basis = basis

    def _wind_curl_spectral(self, qh: torch.Tensor, wind_state_t) -> torch.Tensor:
        if wind_state_t is None:
            return torch.zeros(self.ny, qh.shape[-1], device=qh.device, dtype=qh.dtype)
        return self.basis.curl_spectral(wind_state_t).to(qh.dtype)

    def wind_curl_field(self, wind_state: torch.Tensor) -> torch.Tensor:
        return self.basis.curl_field(wind_state)


def _ocean_kwargs(args) -> dict:
    return dict(nx=args.nx, L=args.L, dt=args.dt, beta=1.5e-11, rd=15000.0,
                delta=0.25, U1=0.05, U2=0.0, rek=5.787e-7, filterfac=23.6,
                wind_amp=args.wind_amp, wind_tau_days=args.wind_tau_days,
                wind_sigma=args.wind_sigma, wind_cx=args.wind_cx, wind_cy=args.wind_cy,
                wind_seed=args.wind_seed, dtype=torch.float32)


def _spinup(dyn: QGDynamics, steps: int, seed: int, device) -> torch.Tensor:
    qh = torch.fft.rfft2(dyn._initial_q(1, seed, device), dim=(-2, -1))
    for _ in range(steps):
        qh = dyn._rk4_step(qh, dyn.dt, dyn.U1, dyn.U2, dyn.beta, dyn.rek)
    q = torch.fft.irfft2(qh, s=(dyn.ny, dyn.nx), dim=(-2, -1))
    return dyn._flatten(q).squeeze(0)


def _wind_state(driver: str, args, basis: FourierWindBasis, dyn_ou: QGDynamics,
                n: int) -> torch.Tensor:
    x0 = y0 = 0.5 * args.L
    if driver == "ou":
        lead = int(4 * args.wind_tau_days * args.steps_per_day)
        ws = dyn_ou.generate_wind_state(lead + n, seed=args.wind_seed, x0=x0, y0=y0)
        return ws[lead:]
    return generate_spectral_wind(
        basis, driver, n, args.dt, amp=args.wind_amp, sigma=args.wind_sigma,
        cx=args.wind_cx, cy=args.wind_cy, x0=x0, y0=y0, seed=args.wind_seed,
        tau_days=args.wind_tau_days, preset=args.preset, mapping=args.mapping)


def _simulate(driver: str, args, basis, basis_dev, state0, device) -> dict:
    n = int(args.days * args.steps_per_day)
    dyn_ou = QGDynamics(**_ocean_kwargs(args)).to(device)
    if driver == "ou":
        dyn = dyn_ou
    else:
        dyn = SpectralWindQG(basis_dev, **_ocean_kwargs(args)).to(device)
    ws = _wind_state(driver, args, basis, dyn_ou, n)
    t0 = time.time()
    traj = dyn.rollout_trajectory(state0, n - 1, wind_state=ws.to(device=device, dtype=torch.float32))
    elapsed = time.time() - t0
    stride = int(args.sample_days * args.steps_per_day)
    snaps = traj[::stride]
    q = dyn._grid(snaps).cpu().numpy()
    psi = dyn.streamfunctions(snaps).cpu().numpy()
    curl_all = dyn.wind_curl_field(ws.to(device=device, dtype=torch.float32)).cpu().double()
    amps = basis.project(curl_all).numpy()
    ke = dyn.kinetic_energy(traj).cpu().numpy()
    return {
        "driver": driver, "q": q, "psi": psi,
        "curl": curl_all[::stride].numpy(), "rms_curl": curl_all.pow(2).mean((-2, -1)).sqrt().numpy(),
        "amps": amps, "ke": ke, "days_snap": np.arange(q.shape[0]) * args.sample_days,
        "days": np.arange(n) / args.steps_per_day, "elapsed_s": elapsed,
        "wavevectors": basis.wavevectors,
    }


def _vmax(runs, key, idx=None, pct=99.5):
    vals = [np.abs(r[key] if idx is None else r[key][:, idx]).ravel() for r in runs]
    return float(np.percentile(np.concatenate(vals), pct)) or 1e-30


def _animate(run, scales, args, path):
    frames = []
    for i in range(run["q"].shape[0]):
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
        panels = ((run["psi"][i, 0], "upper ψ₁", scales["psi1"]),
                  (run["q"][i, 0], "upper q₁", scales["q1"]),
                  (run["q"][i, 1], "lower q₂", scales["q2"]),
                  (run["curl"][i], "wind-stress curl", scales["curl"]))
        for ax, (fld, ttl, vm) in zip(axes, panels):
            im = ax.imshow(fld, cmap=CMAP, vmin=-vm, vmax=vm, origin="lower")
            ax.set_title(f"{ttl} — day {run['days_snap'][i]:.0f}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.suptitle(DRIVER_LABELS[run["driver"]], fontsize=11)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB"))
        plt.close(fig)
    frames = [f.resize((int(f.width * args.gif_scale), int(f.height * args.gif_scale)),
                       Image.LANCZOS) for f in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=args.duration_ms, loop=0, optimize=True)


def _fig_forcing(run, scales, path):
    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.8])
    n = run["curl"].shape[0]
    for j, i in enumerate(np.linspace(0, n - 1, 3).astype(int)):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(run["curl"][i], cmap=CMAP, vmin=-scales["curl"], vmax=scales["curl"], origin="lower")
        ax.set_title(f"wind-stress curl, day {run['days_snap'][i]:.0f}")
        ax.set_xticks([])
        ax.set_yticks([])
    ax = fig.add_subplot(gs[1, :])
    for i, (kx, ky) in enumerate(run["wavevectors"][:4]):
        ax.plot(run["days"], run["amps"][:, 2 * i], lw=1.2, label=f"({kx},{ky}) cos")
    ax.plot(run["days"], run["rms_curl"], color="k", lw=1.8, label="domain-rms curl")
    ax.set_xlabel("day")
    ax.set_ylabel("curl amplitude (s⁻²)")
    ax.legend(ncol=5, fontsize=9, loc="upper right")
    ax.set_title("projected Fourier amplitudes and domain-rms curl")
    fig.suptitle(f"{DRIVER_LABELS[run['driver']]}: wind forcing", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fig_truth(run, scales, path):
    i = run["q"].shape[0] // 2
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (fld, ttl, vm) in zip(axes.ravel(), (
            (run["psi"][i, 0], "upper ψ₁", scales["psi1"]),
            (run["psi"][i, 1], "lower ψ₂", scales["psi2"]),
            (run["q"][i, 0], "upper q₁", scales["q1"]),
            (run["q"][i, 1], "lower q₂", scales["q2"]))):
        ax.imshow(fld, cmap=CMAP, vmin=-vm, vmax=vm, origin="lower")
        ax.set_title(ttl)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{DRIVER_LABELS[run['driver']]}: streamfunction and PV "
                 f"(day {run['days_snap'][i]:.0f})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _long_amplitude_stats(args, basis, drivers, years=40.0) -> dict:
    n = int(years * 365 * 4)
    dt = 6 * 3600.0
    out = {}
    for d in drivers:
        if d == "ou":
            dyn = QGDynamics(**{**_ocean_kwargs(args), "dt": dt, "wind_cx": 0.0, "wind_cy": 0.0})
            ws = dyn.generate_wind_state(n, seed=args.wind_seed + 1, x0=0.0, y0=0.0).double()
            a = basis.ricker_amplitudes(ws[:, 0], ws[:, 1], ws[:, 2], args.wind_sigma).numpy()
        else:
            a = generate_spectral_wind(basis, d, n, dt, amp=args.wind_amp, sigma=args.wind_sigma,
                                       cx=0.0, cy=0.0, x0=0.0, y0=0.0, seed=args.wind_seed + 1,
                                       tau_days=args.wind_tau_days, preset=args.preset,
                                       mapping=args.mapping).numpy()
        z = a[:, 0] / a[:, 0].std()
        v = z - z.mean()
        spec = np.fft.rfft(v, 2 * n)
        acf = np.fft.irfft(spec * np.conj(spec))[:n]
        acf = acf / acf[0]
        lag_days = np.arange(n) * dt / 86400.0
        efold = float(lag_days[np.argmax(acf < 1 / np.e)])
        out[d] = {"z": z, "acf": acf[: int(90 * 4)], "lag_days": lag_days[: int(90 * 4)],
                  "efold_days": efold, "kurtosis": float((v ** 4).mean() / (v ** 2).mean() ** 2),
                  "pair_rms": np.sqrt(0.5 * (a ** 2).reshape(n, -1, 2).sum(-1).mean(0))}
    return out


def _fig_compare(runs, stats, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for r in runs:
        c, lab = DRIVER_COLORS[r["driver"]], DRIVER_LABELS[r["driver"]]
        axes[0, 0].plot(r["days"], r["rms_curl"], color=c, lw=1.3, label=lab)
        axes[0, 1].plot(r["days"], r["ke"], color=c, lw=1.3, label=lab)
    axes[0, 0].set_title("domain-rms wind-stress curl")
    axes[0, 0].set_xlabel("day")
    axes[0, 0].set_ylabel("s⁻²")
    axes[0, 1].set_title("kinetic energy (layer-weighted, from the same spun-up state)")
    axes[0, 1].set_xlabel("day")
    axes[0, 1].set_ylabel("m² s⁻²")
    axes[0, 0].legend(fontsize=9)
    bins = np.linspace(-4, 4, 61)
    for d, s in stats.items():
        c, lab = DRIVER_COLORS[d], DRIVER_LABELS[d]
        axes[1, 0].hist(s["z"], bins=bins, density=True, histtype="step", lw=1.6, color=c,
                        label=f"{lab} (kurtosis {s['kurtosis']:.2f})")
        axes[1, 1].plot(s["lag_days"], s["acf"], color=c, lw=1.5,
                        label=f"{lab} (e-folding {s['efold_days']:.1f} d)")
    xs = np.linspace(-4, 4, 200)
    axes[1, 0].plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), "k:", lw=1, label="Gaussian")
    axes[1, 0].set_title("distribution of the (1,0) cos amplitude (40-year series, standardized)")
    axes[1, 0].legend(fontsize=8, loc="upper left")
    axes[1, 1].axhline(1 / np.e, color="k", ls=":", lw=1)
    axes[1, 1].set_title("autocorrelation of the (1,0) cos amplitude (40-year series)")
    axes[1, 1].set_xlabel("lag (days)")
    axes[1, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _write_report(args, runs, stats, files, path):
    spec = get_preset(args.preset)
    unit = default_time_unit_days(args.preset, args.wind_tau_days)
    rel = os.path.relpath(args.out_dir, os.path.dirname(path))
    lines = [
        "# QG ocean under spectral (Option B) wind forcing — first simulations",
        "",
        "Generated by `reports/qg/simulate_qg_spectral_wind.py`. Four runs start from one",
        f"shared, unforced {args.spinup_years:g}-year spin-up of the two-layer QG ocean (nominal",
        f"benchmark parameters, nx={args.nx}), then run {args.days:g} days forced by:",
        "",
        "| run | driver | forcing |",
        "|---|---|---|",
        "| E0 | `ou` | the current moving Mexican-hat storm (OU amplitude, 15-day memory; the wind series "
        "starts after a 4-memory-time lead-in, since the OU amplitude is initialized at zero) |",
        f"| E0b | `spectral_ou` | {files['n_amp']} Fourier amplitudes (`|k| ≤ {args.kmax}`), independent OU, "
        "same per-mode variance and memory as the storm |",
        f"| E1 | `gyrostat` | the same amplitudes driven by the `{args.preset}` gyrostat preset "
        f"({spec.n_modes} modes; `l63ring4` is four Lorenz-63 gyrostats on a ring, energy-conserving "
        "coupling 0.5) |",
        "| E2 | `gyrostat_surrogate` | multivariate phase-randomized surrogate of E1's gyrostat series "
        "(same auto- and cross-spectra, Gaussian) |",
        "",
        "All spectral drivers share the storm's spatial spectrum (variance per mode, analytic from the",
        "Mexican-hat Fourier transform) and its drift "
        f"(cx={args.wind_cx} m/s, cy={args.wind_cy} m/s, applied as a rotation",
        f"of each cos/sin pair). Wind amplitude level {args.wind_amp:g} s⁻².",
        "",
        "## Calibration",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| gyrostat preset | `{spec.name}`, {spec.n_modes} modes |",
        f"| e-folding of the x modes (gyrostat units) | {spec.efold_units} |",
        f"| time unit | {unit:.1f} days per gyrostat unit (matches the {args.wind_tau_days:g}-day storm memory) |",
    ]
    lines += ["", "Amplitude statistics from 40-year amplitude-only series (no ocean):", "",
              "| run | e-folding of the (1,0) cos amplitude (days) | kurtosis | rms per wavevector / storm target |",
              "|---|---|---|---|"]
    target = files["target"]
    for d, s in stats.items():
        ratio = ", ".join(f"{x:.2f}" for x in s["pair_rms"] / target)
        lines.append(f"| {DRIVER_LABELS[d]} | {s['efold_days']:.1f} | {s['kurtosis']:.2f} | {ratio} |")
    lines += ["", "## Ocean response over the simulated window", "",
              "| run | mean KE, last half (m² s⁻²) | max KE (m² s⁻²) | rms curl (s⁻²) | GPU time (s) |",
              "|---|---|---|---|---|"]
    for r in runs:
        half = r["ke"][len(r["ke"]) // 2:]
        lines.append(f"| {DRIVER_LABELS[r['driver']]} | {half.mean():.3e} | {r['ke'].max():.3e} | "
                     f"{r['rms_curl'].mean():.2e} | {r['elapsed_s']:.0f} |")
    lines += ["", "A single realization per driver over "
              f"{args.days:g} days: these runs show the forcing and the response, they do not",
              "support statistical conclusions. The response analysis (long forced runs, ensembles,",
              "surrogate comparison) is a later work package.", "",
              "Two structural differences are visible even in one realization, because they",
              "follow from how each driver is built rather than from sampling:", "",
              "- **Intermittent vs steady energy input.** The current storm is one coherent blob",
              "  whose signed amplitude crosses zero, so its domain-rms curl repeatedly drops to",
              "  about 0. The spectral drivers sum 12 partly independent amplitudes and keep a",
              "  steady rms curl at the same mean variance. E0 vs E0b isolates this effect.",
              "- **Regimes and long memory.** The gyrostat amplitude is bimodal and keeps an",
              "  autocorrelation of about 0.3 at 60–90 days. Its surrogate shares the memory but",
              "  is Gaussian. E1 vs E2 isolates the bimodality.", "",
              "## Cross-driver comparison", "", f"![comparison]({rel}/{files['compare']})", ""]
    for r in runs:
        d = r["driver"]
        lines += [f"## {DRIVER_LABELS[d]}", "",
                  "| figure | |", "|---|---|",
                  f"| animation (ψ₁, q₁, q₂, wind curl) | ![]({rel}/{files[d]['gif']}) |",
                  f"| wind forcing | ![]({rel}/{files[d]['forcing']}) |",
                  f"| streamfunction and PV | ![]({rel}/{files[d]['truth']}) |", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--L", type=float, default=1e6)
    p.add_argument("--dt", type=float, default=7200.0)
    p.add_argument("--spinup-years", type=float, default=2.0)
    p.add_argument("--days", type=float, default=120.0)
    p.add_argument("--sample-days", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wind-seed", type=int, default=7)
    p.add_argument("--wind-amp", type=float, default=1e-11)
    p.add_argument("--wind-tau-days", type=float, default=15.0)
    p.add_argument("--wind-sigma", type=float, default=250000.0)
    p.add_argument("--wind-cx", type=float, default=0.5)
    p.add_argument("--wind-cy", type=float, default=0.03)
    p.add_argument("--kmax", type=int, default=2)
    p.add_argument("--preset", default="l63ring4",
                   help="gyrostat preset; its modes must match the basis amplitudes "
                        "(l63ring4 has 12, which is |k| <= 2)")
    p.add_argument("--mapping", default=None,
                   help="mode-to-amplitude mapping; defaults to the preset's first mapping")
    p.add_argument("--drivers", default="ou,spectral_ou,gyrostat,gyrostat_surrogate")
    p.add_argument("--duration-ms", type=int, default=160)
    p.add_argument("--gif-scale", type=float, default=0.5)
    p.add_argument("--out-dir", default="reports/qg/outputs/figs")
    p.add_argument("--report", default="reports/qg/outputs/qg_spectral_wind_report.md")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    args.steps_per_day = round(86400.0 / args.dt)
    DRIVER_LABELS["gyrostat"] = f"E1: gyrostat ({args.preset})"
    drivers = args.drivers.split(",")
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    basis = FourierWindBasis(nx=args.nx, L=args.L, kmax=args.kmax)
    basis_dev = FourierWindBasis(nx=args.nx, L=args.L, kmax=args.kmax,
                                 dtype=torch.float32, device=device)
    t0 = time.time()
    state0 = _spinup(QGDynamics(**_ocean_kwargs(args)).to(device),
                     int(args.spinup_years * 365 * args.steps_per_day), args.seed, device)
    print(f"spin-up {args.spinup_years}y in {time.time() - t0:.0f}s")

    runs = []
    for d in drivers:
        runs.append(_simulate(d, args, basis, basis_dev, state0, device))
        print(f"{d}: {args.days}d in {runs[-1]['elapsed_s']:.0f}s, "
              f"mean KE {runs[-1]['ke'].mean():.3e}")

    scales = {"psi1": _vmax(runs, "psi", 0), "psi2": _vmax(runs, "psi", 1),
              "q1": _vmax(runs, "q", 0), "q2": _vmax(runs, "q", 1), "curl": _vmax(runs, "curl")}
    files = {"n_amp": basis.n_amp, "target": (args.wind_amp * basis.ricker_mode_std(args.wind_sigma)).numpy()[::2]}
    for r in runs:
        d = r["driver"]
        files[d] = {"gif": f"qg_specwind_{d}_animation.gif",
                    "forcing": f"qg_specwind_{d}_forcing.png",
                    "truth": f"qg_specwind_{d}_truth_psi_q.png"}
        _animate(r, scales, args, os.path.join(args.out_dir, files[d]["gif"]))
        _fig_forcing(r, scales, os.path.join(args.out_dir, files[d]["forcing"]))
        _fig_truth(r, scales, os.path.join(args.out_dir, files[d]["truth"]))
        print(f"{d}: figures written")

    stats = _long_amplitude_stats(args, basis, drivers)
    files["compare"] = "qg_specwind_comparison.png"
    _fig_compare(runs, stats, os.path.join(args.out_dir, files["compare"]))
    _write_report(args, runs, stats, files, args.report)
    summary = {d: {"efold_days": s["efold_days"], "kurtosis": s["kurtosis"]} for d, s in stats.items()}
    print(json.dumps(summary, indent=1))
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
