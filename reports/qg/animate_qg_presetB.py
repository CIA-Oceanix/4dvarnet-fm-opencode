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
        description="Animate preset-B two-layer QG flow (equilibrated).")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--spinup-years", type=float, default=2.0)
    parser.add_argument("--days", type=float, default=120.0)
    parser.add_argument("--sample-days", type=float, default=2.0)
    parser.add_argument("--duration-ms", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str,
                        default="reports/qg/outputs/figs/qg_presetB_animation.gif")
    args = parser.parse_args()

    dyn = QGDynamics(nx=args.nx, dtype=torch.float32,
                     beta=1.5e-11, rd=15000.0, delta=0.25,
                     U1=0.05, U2=0.0, rek=5.787e-7).to("cuda")

    steps_per_day = round(86400.0 / dyn.dt)
    spinup_steps = int(args.spinup_years * 365.0 * steps_per_day)
    num_steps = int(args.days * steps_per_day)

    t0 = time.time()
    traj, _ = dyn.generate_full_trajectory(num_steps=num_steps, seed=args.seed,
                                           spinup_steps=spinup_steps)
    traj = traj.cpu()
    q = dyn._grid(traj)                      # (T, 2, ny, nx)
    print(f"spinup {args.spinup_years}y + {args.days}d ({num_steps} steps) in "
          f"{time.time()-t0:.1f}s")

    stride = int(args.sample_days * steps_per_day)
    q1 = q[::stride, 0].numpy()
    q2 = q[::stride, 1].numpy()
    days = np.arange(len(q1)) * args.sample_days
    print(f"{len(q1)} frames, q1 scale (±{np.abs(q1).max():.2e}), "
          f"q2 scale (±{np.abs(q2).max():.2e})")

    vmax = max(np.abs(q1).max(), np.abs(q2).max()) * 0.9
    frames = []
    for i in range(len(q1)):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
        for ax, field, title in ((axes[0], q1[i], "upper layer q1"),
                                 (axes[1], q2[i], "lower layer q2")):
            im = ax.imshow(field, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
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
