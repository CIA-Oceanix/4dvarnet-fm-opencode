#!/usr/bin/env python3
"""Generate MAOOAM dataset: multiple trajectories × DA windows.

Creates one dynamics instance and reuses it across all trajectories,
avoiding repeated qgs tensor extraction (~4 min per instance).

Usage:
    python batch/gen_maooam_dataset.py --nx 4 --n-trajs 10
    python batch/gen_maooam_dataset.py --nx 3 --output experiments/maooam_nx3.pt
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.maooam import MaooamConfig, MaooamDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MAOOAM dataset with multiple trajectories (single dynamics instance)"
    )
    parser.add_argument("--nx", type=int, required=True,
                        help="Spectral truncation (atm/occ nx = ny = nx)")
    parser.add_argument("--n-trajs", type=int, default=10,
                        help="Number of independent trajectories (default: 10)")
    parser.add_argument("--seed-start", type=int, default=42,
                        help="Starting seed (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .pt path (default: experiments/maooam_dataset_nx{nx}.pt)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu (default: cuda)")
    parser.add_argument("--num-windows", type=int, default=200,
                        help="Windows per trajectory (default: 200)")
    parser.add_argument("--spinup-steps", type=int, default=5000,
                        help="Spinup steps before collecting (default: 5000)")
    return parser.parse_args()


def _make_cfg(nx, device, num_windows, spinup_steps, seed):
    return MaooamConfig(
        device=device, compile=True,
        atm_nx=nx, atm_ny=nx,
        occ_nx=nx, occ_ny=nx,
        num_windows=num_windows,
        spinup_steps=spinup_steps,
        seed=seed,
    )


def main():
    args = parse_args()
    output = args.output or f"experiments/maooam_dataset_nx{args.nx}.pt"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    seeds = list(range(args.seed_start, args.seed_start + args.n_trajs))
    base_cfg = _make_cfg(args.nx, args.device, args.num_windows, args.spinup_steps, seeds[0])

    print(f"MAOOAM dataset generation (shared dynamics)")
    print(f"  nx=ny       = {args.nx}")
    print(f"  state_dim   = {base_cfg.state_dim}")
    print(f"  n_trajs     = {args.n_trajs}")
    print(f"  seeds       = {seeds[0]}..{seeds[-1]}")
    print(f"  num_windows = {args.num_windows}")
    print(f"  spinup      = {args.spinup_steps} steps")
    print(f"  device      = {args.device}")
    print(f"  output      = {output}")
    print()

    # Create dynamics ONCE — qgs tensor extraction happens here (~4 min)
    print("Building dynamics (qgs tensor extraction)...")
    t0 = time.time()
    from models.maooam_torch import MaooamTorchDynamics
    dynamics = MaooamTorchDynamics(
        device=args.device, compile=True,
        atm_nx=args.nx, atm_ny=args.nx,
        occ_nx=args.nx, occ_ny=args.nx,
    )
    print(f"  dynamics ready in {time.time() - t0:.1f}s")
    print()

    all_windows = []
    t_total = time.time()

    for i, seed in enumerate(seeds):
        cfg_i = _make_cfg(args.nx, args.device, args.num_windows, args.spinup_steps, seed)

        print(f"[{i + 1}/{args.n_trajs}] Generating trajectory (seed={seed})...")
        t0 = time.time()

        ds = MaooamDataset(cfg_i, scenario="S0", dynamics=dynamics)
        all_windows.extend(ds.windows)

        elapsed = time.time() - t0
        print(f"  -> {len(ds)} windows in {elapsed:.1f}s")

    total_time = time.time() - t_total
    print(f"\nTotal: {len(all_windows)} windows in {total_time:.1f}s")

    torch.save({
        "windows": all_windows,
        "config": base_cfg.__dict__,
        "state_dim": base_cfg.state_dim,
        "nx": args.nx,
        "n_trajs": args.n_trajs,
        "seeds": seeds,
    }, output)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()