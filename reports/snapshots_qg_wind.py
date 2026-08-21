import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.qg_dynamics import QGDynamics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Static snapshots of wind-forced two-layer QG flow.")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--spinup-years", type=float, default=2.0)
    parser.add_argument("--snapshot-days", type=float, default=90.0)
    parser.add_argument("--day", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wind-amp", type=float, default=1e-11)
    parser.add_argument("--wind-tau-days", type=float, default=5.0)
    parser.add_argument("--wind-cx", type=float, default=0.5)
    parser.add_argument("--wind-cy", type=float, default=0.03)
    parser.add_argument("--out-dir", default="reports/outputs/figs")
    args = parser.parse_args()

    dyn = QGDynamics(nx=args.nx, dtype=torch.float32,
                     beta=1.5e-11, rd=15000.0, delta=0.25,
                     U1=0.05, U2=0.0, rek=5.787e-7,
                     wind_amp=args.wind_amp,
                     wind_tau_days=args.wind_tau_days,
                     wind_cx=args.wind_cx, wind_cy=args.wind_cy).to("cuda")

    steps_per_day = round(86400.0 / dyn.dt)
    spinup_steps = int(args.spinup_years * 365.0 * steps_per_day)
    num_steps = int(args.snapshot_days * steps_per_day)
    snap = int(args.day * steps_per_day)

    traj, wind_state = dyn.generate_full_trajectory(num_steps=num_steps,
                                                    seed=args.seed,
                                                    spinup_steps=spinup_steps)
    traj = traj.cpu()
    wind_state = wind_state.cpu()
    q = dyn._grid(traj)                      # (T, 2, ny, nx)
    state = dyn._flatten(q[snap:snap + 1]).squeeze(0)
    psi = dyn.streamfunctions(state.to("cuda")).cpu().numpy()
    qn = q[snap].numpy()

    windfield = dyn.wind_curl_field(
        wind_state[snap:snap + 1].to(dyn.device)).squeeze(0).cpu().numpy()
    a_t = float(wind_state[snap, 0])

    qmax = max(np.abs(qn).max(), 1e-30)
    pmax = max(np.abs(psi).max(), 1e-30)
    wmax = max(float(np.abs(windfield).max()), 1e-15)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, fld, ttl, vm in (
            (axes[0], qn[0], f"upper q1 (day {args.day:.0f})", qmax),
            (axes[1], qn[1], f"lower q2 (day {args.day:.0f})", qmax),
            (axes[2], windfield,
             f"wind curl day {args.day:.0f} (A={a_t:.1e})", wmax)):
        ax.imshow(fld, cmap="RdBu_r", vmin=-vm, vmax=vm)
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    fig.savefig(os.path.join(args.out_dir, "qg_wind_snapshots.png"), dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, fld, ttl, vm in (
            (axes[0], psi[0], "upper psi1", pmax),
            (axes[1], psi[1], "lower psi2", pmax)):
        ax.imshow(fld, cmap="RdBu_r", vmin=-vm, vmax=vm)
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "qg_wind_streamfunctions.png"),
                dpi=110)
    plt.close(fig)

    print(f"wrote qg_wind_snapshots.png + qg_wind_streamfunctions.png "
          f"(day {args.day:.0f}, wind_amp={args.wind_amp:.1e})")


if __name__ == "__main__":
    main()
