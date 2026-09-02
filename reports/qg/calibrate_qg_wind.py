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

NOMINAL = {"U1": 0.05, "U2": 0.0, "rd": 15000.0, "beta": 1.5e-11,
           "delta": 0.25, "rek": 5.787e-7, "dt": 7200.0}

AMPS = [0.0, 3e-12, 1e-11, 2e-11, 3e-11]


def run_level(amp: float, nx: int, device: torch.device, seed: int,
              spinup_steps: int, wind_days: float, tau_days: float,
              mono_days: float) -> dict:
    dyn = QGDynamics(nx=nx, wind_amp=amp, wind_tau_days=tau_days,
                     wind_seed=seed, **NOMINAL).to(device)
    steps_per_day = round(86400.0 / dyn.dt)
    num_steps = int(wind_days * steps_per_day)
    mono = max(1, int(mono_days * steps_per_day))

    t0 = time.time()
    traj, _series = dyn.generate_full_trajectory(
        num_steps=num_steps, seed=seed, spinup_steps=spinup_steps)
    traj = traj.cpu()
    q = dyn._grid(traj)                     # (T, 2, ny, nx)
    ke_time = []
    for k in range(0, num_steps, mono):
        state = dyn._flatten(q[k:k + 1]).squeeze(0)
        ke_time.append(dyn.kinetic_energy(state.to(device)).item())
    walk_time = time.time() - t0

    ke_arr = np.asarray(ke_time)
    return {
        "wind_amp": amp,
        "spindown_sec": walk_time,
        "ke_start": float(ke_arr[0]),
        "ke_end": float(ke_arr[-1]),
        "ke_mean": float(ke_arr.mean()),
        "ke_hist": ke_arr.tolist(),
        "ke_std": float(ke_arr.std()),
        "t_days": (np.arange(len(ke_arr)) * mono / steps_per_day).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate QG wind forcing amplitude vs equilibrium KE.")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default="reports/qg/outputs/figs")
    parser.add_argument("--spinup-years", type=float, default=2.0)
    parser.add_argument("--wind-days", type=float, default=180.0)
    parser.add_argument("--tau-days", type=float, default=15.0)
    parser.add_argument("--mono-days", type=float, default=5.0)
    parser.add_argument("--amps", nargs="+", default=AMPS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    steps_per_day = round(86400.0 / NOMINAL["dt"])
    spinup_steps = int(args.spinup_years * 365.0 * steps_per_day)
    print(f"device={device} spinup_steps={spinup_steps}")

    results = []
    for amp in args.amps:
        amp = float(amp)
        res = run_level(amp, args.nx, device, args.seed, spinup_steps,
                        args.wind_days, args.tau_days, args.mono_days)
        results.append(res)
        print(f"wind_amp={amp:9.1e}  KE start={res['ke_start']:.3e} "
              f"end={res['ke_end']:.3e} mean={res['ke_mean']:.3e} "
              f"(run {res['spindown_sec']:.0f}s)")

    baseline = next((r for r in results if r["wind_amp"] == 0.0), None)
    fig, ax = plt.subplots(figsize=(9, 5))
    for res in results:
        ax.semilogy(res["t_days"], np.maximum(res["ke_hist"], 1e-12),
                    label=f"wind_amp={res['wind_amp']:.1e}")
    if baseline is not None:
        ax.axhline(baseline["ke_mean"], color="k", ls="--",
                   label=f"unforced mean KE={baseline['ke_mean']:.2e}")
    ax.set_xlabel("time in forced window [days]")
    ax.set_ylabel("domain-mean KE [m$^2$/s$^2$]")
    ax.set_title("QG wind-forcing calibration: KE response")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "qg_wind_ke.png"), dpi=110)
    plt.close(fig)

    for res in results:
        res.pop("ke_hist", None)
        res.pop("t_days", None)
    with open(os.path.join(args.out_dir, "qg_wind_calibration.json"),
              "w") as f:
        json.dump({"spinup_steps": spinup_steps, "wind_days": args.wind_days,
                   "tau_days": args.tau_days, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
