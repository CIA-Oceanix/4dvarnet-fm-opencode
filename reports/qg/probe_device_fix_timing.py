"""Timing probe for the GPU device-placement fix in QGS01Dataset._generate_truth.

Measures per-window wall-clock on a dedicated GPU (matching the sbatch
--gres=gpu:a40:1 target hardware) to extrapolate total generation time for
the 1000/100/100 train/val/test dataset targets.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data.qg import QGConfig, QGS01Dataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if torch.cuda.is_available():
        print(f"gpu_name={torch.cuda.get_device_name(0)}", flush=True)

    cfg = QGConfig(nx=64, dt=7200.0)
    n = 20
    t0 = time.time()
    per_window = []
    for i in range(n):
        tw0 = time.time()
        QGS01Dataset._generate_truth(cfg, 1, device=device)
        dt = time.time() - tw0
        per_window.append(dt)
        print(f"window {i}: {dt:.2f}s", flush=True)
    total = time.time() - t0
    import numpy as np
    arr = np.array(per_window)
    print(f"n={n} total={total:.1f}s mean={arr.mean():.2f}s std={arr.std():.2f}s "
          f"min={arr.min():.2f}s max={arr.max():.2f}s", flush=True)

    for label, count in [("train", 1000), ("val", 100), ("test", 100)]:
        est_h = count * arr.mean() / 3600.0
        print(f"estimate[{label}] n={count}: {est_h:.2f}h (mean-based)", flush=True)
    total_h = 1200 * arr.mean() / 3600.0
    print(f"estimate[total 1000+100+100]: {total_h:.2f}h serial", flush=True)


if __name__ == "__main__":
    main()
