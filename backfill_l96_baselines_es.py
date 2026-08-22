#!/usr/bin/env python3
"""One-off CPU script to backfill Energy Score (ES) for existing L96 S0/S1 DA
baseline caches that were generated before ES was stored, mirroring the EV
backfill in ``backfill_l96_baselines_ev.py``.

Reads the analysis trajectories from the combined .npz and the reference truth
from the cached S0/S1 dataset, then computes the same pooled RMSE/EV/ES via the
scheme-agnostic ``evaluate_estimates`` (Energy Score for a deterministic N=1
reconstruction reduces to the per-dimension mean absolute error) and writes an
``es`` entry into each method's JSON cache. RMSE/EV are refreshed from the same
trajectories so all three metrics are consistent.

Usage:
    python backfill_l96_baselines_es.py [--baselines PATH] [--trajectories PATH] [--dataset PATH]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.estimate_metrics import evaluate_estimates
from evaluation.run_l96 import make_obs_j_indices

BASE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(BASE, "experiments")

DEFAULT_BASELINES = os.path.join(EXP_DIR, "l96_baselines_dws500_s0c_inf2.0_etkf_inf2.0_obsj2_int100_fw.json")
DEFAULT_TRAJ = os.path.join(EXP_DIR, "l96_baselines_trajectories_dws500_s0c_inf2.0_etkf_inf2.0_obsj2_int100_fw.npz")
DEFAULT_DATASET = os.path.join(EXP_DIR, "l96_datasets_obsj2_int100_nwin200.pt")

NO = 8
J_TRUTH = 4
OBS_J = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", default=DEFAULT_BASELINES)
    parser.add_argument("--trajectories", default=DEFAULT_TRAJ)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dry-run", action="store_true", help="Do not write the cache")
    args = parser.parse_args()

    obs_var_indices = make_obs_j_indices(NO, J_TRUTH, OBS_J)

    with open(args.baselines) as f:
        cache = json.load(f)
    traj = np.load(args.trajectories)
    ds = torch.load(args.dataset, weights_only=False)
    truth = {
        case: torch.stack([ds[f"test_{case}"][i]["true_state"] for i in range(len(ds[f"test_{case}"]))]).numpy()
        for case in ("s0", "s1")
    }

    updated = 0
    for case in ("s0", "s1"):
        if case not in cache:
            print(f"  ! {case}: no cached entry, skipping")
            continue
        ref = truth[case]
        ref_sub = ref[..., obs_var_indices] if ref.shape[-1] > len(obs_var_indices) else ref
        for name in list(cache[case].keys()):
            if name == "config":
                continue
            key = f"{case}_{name.replace('-', '_').replace(' ', '_')}_trajectories"
            if key not in traj.files:
                print(f"  ! {case}/{name}: no trajectory, skipping")
                continue
            analysis = traj[key]
            analysis_sub = analysis[..., obs_var_indices] if analysis.shape[-1] > len(obs_var_indices) else analysis
            if analysis_sub.shape[-1] != ref_sub.shape[-1]:
                ref_use = ref_sub[..., :analysis_sub.shape[-1]]
            else:
                ref_use = ref_sub
            m = evaluate_estimates(analysis_sub, ref_use)
            cache[case][name]["es"] = {"groups": m["es"]["groups"]}
            print(f"  {case}/{name}: es all_obs={m['es']['groups']['all_obs']:.4f} "
                  f"(slow={m['es']['groups']['slow']:.4f} obs_fast={m['es']['groups']['obs_fast']:.4f})")
            updated += 1

    if updated and not args.dry_run:
        with open(args.baselines, "w") as f:
            json.dump(cache, f, indent=2)
    print(f"\nBackfilled ES into {updated} method entries: {args.baselines}")


if __name__ == "__main__":
    main()
