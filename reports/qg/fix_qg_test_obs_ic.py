"""Correct the obs/IC protocol of the pre-generated 100-window TEST split
without re-running its (already-correct, expensive) truth generation.

`generate_qg_window_chunk.py` generated the 1000/100/100 train/val/test
truth windows using QGConfig defaults for obs_geometry/cols_per_day/
obs_noise_std_frac/init_lag_days, not the established S0 reference-case
settings (random_columns, cols_per_day=4, obs_noise_std_frac=0.01,
init_lag_days=1.0). Truth generation (the ~2-year-spinup rollout) doesn't
depend on any of those fields, so `QGS01Dataset._generate_obs_ic` can
recompute correct obs/IC directly from the existing truth cache -- no GPU,
no rollout, seconds not hours. Train/val don't need this: their (equally
mismatched) cached obs/IC are unused -- obs/IC are meant to be generated
on the fly downstream for those splits.

Writes three files under `--out-dir`:
  - `truth_only/test.pt`      -- the expensive fields, extracted once, reusable
                                  for any future obs/IC protocol correction.
  - `obs_ic/test_reference.pt` -- the corrected obs/IC fields alone.
  - `cache/qg_truth_<hash>.pt` -- truth+corrected-obs/IC merged, at the exact
                                  path `make_qg_s0_s1_datasets(ref_cfg, ...,
                                  cache_dir=...)` looks up, so
                                  `run_qg_baselines.py --cache-dir ...` (or
                                  any other existing caller) works unchanged.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data.qg import QGConfig, QGS01Dataset, _truth_cache_path
from generate_qg_window_chunk import SPLIT_SEED_BASE, SPLIT_SIZE  # noqa: E402

TRUTH_FIELDS = ("true_state", "target_state_psi", "target_state_q", "wind_curl",
                "wind_state_true", "true_params", "wind_seed", "wind_amp",
                "init_lead_truth")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old-cache", required=True,
                    help="Existing (wrong obs/IC config) test-split combined "
                         "cache file, e.g. .../cache/qg_truth_<hash>.pt")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    n = SPLIT_SIZE["test"]
    seed = SPLIT_SEED_BASE["test"]
    indices = list(range(n))

    old_windows = torch.load(args.old_cache, map_location="cpu")
    if len(old_windows) != n:
        raise RuntimeError(f"expected {n} windows in {args.old_cache}, "
                           f"found {len(old_windows)}")

    truth_only = [{k: w[k] for k in TRUTH_FIELDS} for w in old_windows]
    truth_dir = os.path.join(args.out_dir, "truth_only")
    os.makedirs(truth_dir, exist_ok=True)
    truth_path = os.path.join(truth_dir, "test.pt")
    torch.save(truth_only, truth_path)
    print(f"wrote truth-only: {truth_path}")

    ref_cfg = QGConfig(nx=64, dt=7200.0, seed=seed, num_windows=n,
                       obs_geometry="random_columns", cols_per_day=4,
                       obs_noise_std_frac=0.01, init_lag_days=1.0)
    obs_ic = QGS01Dataset._generate_obs_ic(ref_cfg, truth_only, indices)
    obs_ic_dir = os.path.join(args.out_dir, "obs_ic")
    os.makedirs(obs_ic_dir, exist_ok=True)
    obs_ic_path = os.path.join(obs_ic_dir, "test_reference.pt")
    torch.save(obs_ic, obs_ic_path)
    print(f"wrote obs/IC: {obs_ic_path}")

    merged = [{**t, **o} for t, o in zip(truth_only, obs_ic)]
    cache_dir = os.path.join(args.out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = _truth_cache_path(ref_cfg, n, cache_dir)
    torch.save(merged, cache_path)
    print(f"wrote corrected combined cache: {cache_path}")


if __name__ == "__main__":
    main()
