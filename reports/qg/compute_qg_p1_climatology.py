"""Climatological references of the QG P1 scores, from the 100 forced test windows (S0, true state).

- `crps_clim`: CRPS of the climatological forecast per field, E|X - X'| / 2 over pooled
  space-time values (20000 values sampled per window), the denominator of the CRPS skill score;
- `mean_std`: climatological mean and standard deviation per field, which standardise the
  flow-matching scores (the `--fm-clim` input of the QG scorers).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from data.qg_datasets import SPECS
from evaluation.run_qg_specwind_da import FIELDS, _layers, build_cfg, s0_windows


def half_gini(x: np.ndarray) -> float:
    s = np.sort(x)
    n = s.size
    return float(np.sum((2 * np.arange(1, n + 1) - n - 1) * s) / n ** 2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="experiments/qg_datasets")
    p.add_argument("--out", default="reports/qg/outputs/qg_p1/qg_p1_climatology.json")
    args = p.parse_args()
    spec = SPECS["qg_specwind_gyrostat_v1"]
    cfg = build_cfg(spec, cols_per_day=3, obs_noise_std_frac=0.05, init_lag_days=5.0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    samples = {k: [] for k in FIELDS}
    acc = {k: [0.0, 0.0, 0] for k in FIELDS}
    for start in range(0, 100, 10):
        ws, _ = s0_windows(spec, "test", args.root, list(range(start, start + 10)), cfg)
        for w in ws:
            t = _layers(w["true_state"].cpu().numpy(), w["true_params"], cfg, dev)
            for k in FIELDS:
                v = t[k].astype(np.float64)
                samples[k].append(rng.choice(v.ravel(), 20000, replace=False))
                acc[k][0] += v.sum()
                acc[k][1] += (v ** 2).sum()
                acc[k][2] += v.size
    mean_std = {}
    for k, (s1, s2, n) in acc.items():
        m = s1 / n
        mean_std[k] = {"mean": m, "std": float(np.sqrt(s2 / n - m ** 2))}
    out = {"crps_clim": {k: half_gini(np.concatenate(samples[k])) for k in FIELDS}, "mean_std": mean_std}
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
