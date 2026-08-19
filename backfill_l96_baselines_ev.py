#!/usr/bin/env python3
"""One-off CPU script to back-compute pooled explained variance (EV) for existing
L96 S0/S1 DA baseline caches that were generated before EV was stored.

Reads the analysis trajectories from the combined .npz and the reference truth from
the cached S0/S1 dataset, then computes the same pooled EV as `evaluate_baseline`
in `evaluation/run_l96.py` and writes an `ev` entry into each method's JSON cache.

Usage:
    python backfill_l96_baselines_ev.py [--baselines PATH] [--trajectories PATH] [--dataset PATH]
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.run_l96 import fmt_ev, make_obs_j_indices

BASE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(BASE, "experiments")

DEFAULT_BASELINES = os.path.join(EXP_DIR, "l96_baselines_dws500_inf2.0_etkf_inf2.0_obsj2.json")
DEFAULT_TRAJ = os.path.join(EXP_DIR, "l96_baselines_trajectories_dws500_inf2.0_etkf_inf2.0_obsj2.npz")
DEFAULT_DATASET = os.path.join(EXP_DIR, "l96_datasets_obsj2_nwin200.pt")

NO = 8
J_TRUTH = 4
OBS_J = 2


def compute_ev(analysis, ref, obs_var_indices):
    """Pooled per-dimension EV, matching evaluate_baseline's formula.

    analysis: (nwin, T, D_analysis); ref: (nwin, T, D_ref_truth).
    Both are subsampled to obs_var_indices (24D) before pooling.
    """
    if obs_var_indices is not None:
        ref_sub = ref[..., obs_var_indices]
        analysis_sub = analysis
        if analysis_sub.shape[-1] > len(obs_var_indices):
            analysis_sub = analysis_sub[..., obs_var_indices]
    else:
        ref_sub = ref
        analysis_sub = analysis
        if analysis_sub.shape[-1] != ref_sub.shape[-1]:
            ref_sub = ref_sub[..., :analysis_sub.shape[-1]]
    sq_err = (analysis_sub - ref_sub) ** 2
    pooled_mse = sq_err.mean(axis=(0, 1))
    pooled_var = ref_sub.var(axis=(0, 1))
    pooled_var = np.maximum(pooled_var, 1e-12)
    return 1.0 - pooled_mse / pooled_var


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", default=DEFAULT_BASELINES)
    parser.add_argument("--trajectories", default=DEFAULT_TRAJ)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    args = parser.parse_args()

    obs_var_indices = make_obs_j_indices(NO, J_TRUTH, OBS_J)

    with open(args.baselines) as f:
        cache = json.load(f)
    traj = np.load(args.trajectories)
    ds = torch.load(args.dataset, weights_only=False)
    truth = {case: torch.stack([ds[f"test_{case}"][i]["true_state"] for i in range(len(ds[f"test_{case}"]))]).numpy()
             for case in ("s0", "s1")}

    updated = 0
    for case in ("s0", "s1"):
        if case not in cache:
            print(f"  ! {case}: no cached entry, skipping")
            continue
        nwin = traj[f"{case}_EnKF_trajectories"].shape[0]
        ref = truth[case]
        for name in list(cache[case].keys()):
            if name == "config":
                continue
            key = f"{case}_{name.replace('-', '_').replace(' ', '_')}_trajectories"
            if key not in traj.files:
                print(f"  ! {case}/{name}: no trajectory, skipping")
                continue
            analysis = traj[key]
            ev = compute_ev(analysis, ref, obs_var_indices)
            cache[case][name]["ev"] = fmt_ev(ev, NO=NO, obs_j=OBS_J)
            ev_groups = cache[case][name]["ev"]["groups"]
            print(f"  {case}/{name}: nwin={nwin} ev all_obs={ev_groups['all_obs']:.4f} "
                  f"(slow={ev_groups['slow']:.4f} obs_fast={ev_groups['obs_fast']:.4f})")
            updated += 1

    with open(args.baselines, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"\nBackfilled EV into {updated} method entries: {args.baselines}")


if __name__ == "__main__":
    main()
