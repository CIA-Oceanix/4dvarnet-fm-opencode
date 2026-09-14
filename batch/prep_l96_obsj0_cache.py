#!/usr/bin/env python3
"""Build a slow-only-observation L96 S0/S1 test cache from the canonical obsj2 cache.

Re-observes each stored window's full 40D ``true_state`` with the slow-only
observation indices ``(0..7)`` (obs_j=0), reusing each window's ``obs_seed`` so
the observation noise is reproducible. The dynamics trajectories (true_state),
per-window parameters, forcings and obs_seed are UNCHANGED — only ``obs`` and
``obs_mask`` are regenerated as the 8D slow-only subset.

This lets DA baselines (state-only and state+param) run under slow-only
observation while comparing directly against the existing obsj2 benchmark on the
identity same true state-param dataset.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.lorenz96 import _generate_observations
from evaluation.run_l96 import make_obs_j_indices

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE, "experiments")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=None,
                        help="Source cache (default: l96_datasets_obsj2_int100_nwin200.pt)")
    parser.add_argument("--obs-interval", type=int, default=100)
    parser.add_argument("--r-var", type=float, default=0.5)
    parser.add_argument("--num-test-windows", type=int, default=200)
    parser.add_argument("--dest-obs-j", type=int, default=0,
                        help="Fast vars observed in the destination (0 = slow-only)")
    parser.add_argument("--dest", type=str, default=None,
                        help="Destination cache path (default: auto-name with dest-obs-j)")
    args = parser.parse_args()

    src = args.src or os.path.join(EXP_DIR, "l96_datasets_obsj2_int100_nwin200.pt")
    if args.dest:
        dest = args.dest
    else:
        dest = os.path.join(
            EXP_DIR, f"l96_datasets_obsj{args.dest_obs_j}_int{args.obs_interval}_nwin{args.num_test_windows}.pt")

    if not os.path.exists(src):
        raise SystemExit(f"Source cache not found: {src}")

    obs_indices = make_obs_j_indices(8, 4, args.dest_obs_j)
    print(f"Source: {src}")
    print(f"Dest:   {dest}")
    print(f"Re-observing with obs_j={args.dest_obs_j}, obs_var_indices={list(obs_indices) if obs_indices is not None else 'all'}")

    datasets = torch.load(src, weights_only=False, map_location="cpu")
    for key in ("test_s0", "test_s1"):
        for w in datasets[key]:
            w["obs"], w["obs_mask"] = _generate_observations(
                w["true_state"], args.obs_interval, args.r_var, w["obs_seed"],
                obs_var_indices=obs_indices)
    torch.save(datasets, dest)

    for key in ("test_s0", "test_s1"):
        obs_dim = datasets[key][0]["obs"].shape[-1]
        print(f"  {key}: {len(datasets[key])} windows, obs dim = {obs_dim}")
    print(f"Saved slow-only cache to {dest}")


if __name__ == "__main__":
    main()
