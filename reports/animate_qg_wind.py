import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from models.qg_dynamics import QGDynamics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animate wind-forced two-layer QG flow (equilibrated).")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--spinup-years", type=float, default=2.0)
    parser.add_argument("--days", type=float, default=120.0)
    parser.add_argument("--sample-days", type=float, default=1.0)
    parser.add_argument("--duration-ms", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wind-amp", type=float, default=1e-11)
    parser.add_argument("--wind-tau-days", type=float, default=5.0)
    parser.add_argument("--wind-cx", type=float, default=0.5)
    parser.add_argument("--wind-cy", type=float, default=0.03)
    parser.add_argument("--out", type=str,
                        default="reports/outputs/figs/qg_wind_animation.gif")
    args = parser.parse_args()

    dyn = QGDynamics(nx=args.nx, dtype=torch.float32,
                     beta=1.5e-11, rd=15000.0, delta=0.25,
                     U1=0.05, U2=0.0, rek=5.787e-7,
                     wind_amp=args.wind_amp,
                     wind_tau_days=args.wind_tau_days,
                     wind_cx=args.wind_cx, wind_cy=args.wind_cy).to("cuda")

    steps_per_day = round(86400.0 / dyn.dt)
    spinup_steps = int(args.spinup_years * 365.0 * steps_per_day)
    num_steps = int(args.days * steps_per_day)

    t0 = time.time()
    traj, wind_state = dyn.generate_full_trajectory(num_steps=num_steps,
                                                    seed=args.seed,
                                                    spinup_steps=spinup_steps)
    traj = traj.cpu()
    wind_state = wind_state.cpu()
    q = dyn._grid(traj)                      # (T, 2, ny, nx)
    print(f"spinup {args.spinup_years}y + {args.days}d ({num_steps} steps) in "
          f"{time.time()-t0:.1f}s; wind_amp={args.wind_amp:.1e} "
          f"(amp std {wind_state[:, 0].std().item():.1e})")

    stride = int(args.sample_days * steps_per_day)
    q1 = q[::stride, 0].numpy()
    q2 = q[::stride, 1].numpy()
    days = np.arange(len(q1)) * args.sample_days
    windfields = dyn.wind_curl_field(
        wind_state[::stride].to(dyn.device)).cpu().numpy()
    print(f"{len(q1)} frames, q1 scale (±{np.abs(q1).max():.2e}), "
          f"q2 scale (±{np.abs(q2).max():.2e}), "
          f"wind-curl field scale (±{np.abs(windfields).max():.2e})")

    vmax = max(np.abs(q1).max(), np.abs(q2).max()) * 0.9
    wmax = max(float(np.abs(windfields).max()), 1e-15)
    frames = []
    for i in range(len(q1)):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
        for ax, field, title, vm in (
                (axes[0], q1[i], "upper layer q1", vmax),
                (axes[1], q2[i], "lower layer q2", vmax),
                (axes[2], windfields[i], "wind curl field", wmax)):
            im = ax.imshow(field, cmap="RdBu_r", vmin=-vm, vmax=vm)
            ax.set_title(f"{title} — day {days[i]:.0f}")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(buf).convert("RGB"))
        plt.close(fig)

    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=args.duration_ms, loop=0)
    print(f"wrote {args.out} ({len(frames)} frames, "
          f"{sum(f.size[0] * f.size[1] for f in frames) * 3 / 1e6:.1f} MB raw)")


if __name__ == "__main__":
    main()
