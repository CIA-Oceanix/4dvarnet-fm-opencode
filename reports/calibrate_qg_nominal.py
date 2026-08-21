import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.qg_dynamics import QGDynamics

PRESETS: dict[str, dict] = {
    "A_pyqg_default": {"U1": 0.025, "U2": 0.0, "rd": 15000.0,
                       "beta": 1.5e-11, "delta": 0.25, "rek": 5.787e-7,
                       "spinup_steps": 13140},
    "B_strong_shear": {"U1": 0.05, "U2": 0.0, "rd": 15000.0,
                       "beta": 1.5e-11, "delta": 0.25, "rek": 5.787e-7,
                       "spinup_steps": 6570},
    "C_weak_beta": {"U1": 0.025, "U2": 0.0, "rd": 15000.0,
                    "beta": 0.5e-11, "delta": 0.25, "rek": 5.787e-7,
                    "spinup_steps": 6570},
}


def shell_averaged_spectrum(dyn: QGDynamics, state: torch.Tensor) -> np.ndarray:
    q = dyn._grid(state.unsqueeze(0))
    qh = torch.fft.rfft2(q, dim=(-2, -1))
    ph = dyn._invert(qh)
    M2 = float(dyn.nx * dyn.ny) ** 2
    del1 = dyn.delta / (dyn.delta + 1.0)
    del2 = 1.0 / (dyn.delta + 1.0)
    ke_spec = del1 * dyn.K2 * ph[0, 0].abs() ** 2 / M2 \
        + del2 * dyn.K2 * ph[0, 1].abs() ** 2 / M2
    kv = torch.sqrt(dyn.K2).cpu().numpy()
    spec = ke_spec.detach().cpu().numpy()
    nbins = min(dyn.ny // 2, 32)
    edges = np.linspace(0.0, kv.max(), nbins + 1)
    idx = np.digitize(kv.ravel(), edges) - 1
    out = np.zeros(nbins)
    for b in range(nbins):
        mask = idx == b
        if mask.any():
            out[b] = spec.ravel()[mask].sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, out


def run_config(name: str, params: dict, nx: int, device: torch.device,
               out_dir: str, record_every_days: float = 5.0,
               record_days: float = 30.0) -> dict:
    dt = 7200.0
    spinup = params.pop("spinup_steps")
    rec_every = max(1, int(record_every_days * 86400.0 / dt))
    n_rec = int(record_days * 86400.0 / dt / rec_every)
    dyn = QGDynamics(nx=nx, **{k: v for k, v in params.items()
                               if k not in ("spinup_steps",)}).to(device)
    q0 = dyn._initial_q(1, seed=42, device=device)
    qh = torch.fft.rfft2(q0, dim=(-2, -1))

    ke_hist, t_hist = [], []
    mon_every = max(1, spinup // 200)
    t0 = time.time()
    for step in range(spinup):
        qh = dyn._rk4_step(qh, dt, dyn.U1, dyn.U2, dyn.beta, dyn.rek)
        if step % mon_every == 0:
            qm = torch.fft.irfft2(qh, s=(dyn.ny, dyn.nx), dim=(-2, -1))
            ke_hist.append(dyn.kinetic_energy(dyn._flatten(qm)).item())
            t_hist.append(step * dt / 86400.0)
    spin_time = time.time() - t0

    snaps = []
    for i in range(n_rec):
        if i > 0:
            for _ in range(rec_every):
                qh = dyn._rk4_step(qh, dt, dyn.U1, dyn.U2, dyn.beta, dyn.rek)
        snaps.append(torch.fft.irfft2(qh, s=(dyn.ny, dyn.nx), dim=(-2, -1)))
    state = dyn._flatten(snaps[-1]).squeeze(0)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    psi = dyn.streamfunctions(state).cpu().numpy()
    qf = dyn._grid(state).cpu().numpy()
    for ax, fld, ttl in zip(axes.flat,
                            [psi[0], psi[1], qf[0], qf[1]],
                            [r"$\psi_1$", r"$\psi_2$", r"$q_1$", r"$q_2$"]):
        im = ax.imshow(fld, cmap="RdBu_r", aspect="equal")
        ax.set_title(ttl)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"{name}: snapshots after {spinup} steps "
                 f"({spinup * dt / 86400:.0f} d)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"qg_snapshots_{name}.png"), dpi=110)
    plt.close(fig)

    centers, spec = shell_averaged_spectrum(dyn, state)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(centers[1:], spec[1:] + 1e-30, marker="o", ms=3)
    kd = 1.0 / dyn.rd
    ax.axvline(kd, color="k", ls="--", label=f"$k_d$ ({kd:.2e})")
    k_ny = np.pi / (dyn.L / dyn.nx)
    ax.axvline(k_ny, color="gray", ls=":", label="Nyquist")
    ax.set_xlabel("wavenumber [1/m]")
    ax.set_ylabel("KE spectral density")
    ax.set_title(f"{name}: shell-averaged KE spectrum")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"qg_spectrum_{name}.png"), dpi=110)
    plt.close(fig)

    return {"name": name, "ke_hist": ke_hist, "t_hist": t_hist,
            "spinup_seconds": spin_time,
            "ke_final": ke_hist[-1], "params": dict(params)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default="reports/outputs/figs")
    parser.add_argument("--presets", nargs="+", default=list(PRESETS))
    parser.add_argument("--with-pyqg", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for name in args.presets:
        params = dict(PRESETS[name])
        res = run_config(name, params, args.nx, device, args.out_dir)
        results.append(res)
        print(f"{name}: spinup {res['spinup_seconds']:.1f}s, "
              f"final KE {res['ke_final']:.3e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for res in results:
        ax.semilogy(res["t_hist"], np.maximum(res["ke_hist"], 1e-12),
                    label=res["name"])
    ax.set_xlabel("time [days]")
    ax.set_ylabel("domain-mean KE [m$^2$/s$^2$]")
    ax.set_title("QG spinup: kinetic energy growth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "qg_spinup_ke.png"), dpi=110)
    plt.close(fig)

    if args.with_pyqg:
        import pyqg
        curves = {}
        for name in args.presets:
            p = PRESETS[name]
            np.random.seed(42)
            m = pyqg.QGModel(nx=args.nx, beta=p["beta"], rd=p["rd"],
                             delta=p["delta"], U1=p["U1"], U2=p["U2"],
                             rek=p["rek"], dt=7200.0,
                             tmax=(p["spinup_steps"] + 10) * 7200.0,
                             twrite=10 ** 9, tavestart=1e18, log_level=0)
            ts, kes = [], []
            target = p["spinup_steps"]
            while m.tc < target:
                m._step_forward()
                if m.tc % max(1, target // 100) == 0:
                    ts.append(m.t / 86400.0)
                    kes.append(m._calc_ke())
            curves[name] = (ts, kes)
            print(f"pyqg {name}: final KE {kes[-1]:.3e}")
        for res in results:
            ts, kes = curves[res["name"]]
            ax.semilogy(ts, np.maximum(kes, 1e-12), "--",
                        label=f"pyqg {res['name']}")
        ax.legend()
        fig.savefig(os.path.join(args.out_dir, "qg_spinup_ke.png"), dpi=110)
        plt.close(fig)

    summary = {r["name"]: {"ke_final": r["ke_final"],
                           "spinup_seconds": r["spinup_seconds"],
                           "params": r["params"]} for r in results}
    with open(os.path.join(args.out_dir, "qg_calibration_summary.json"),
              "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
