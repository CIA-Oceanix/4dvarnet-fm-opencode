#!/usr/bin/env python3
"""Precompute per-channel z-score stats for the L63 normalization pipeline.

Mirrors precompute_l96_norm_stats.py's convention: compute mean/std once
over a canonical train split and reuse it everywhere `data.normalize: true`
is set, so comparisons aren't contaminated by different-seed stat estimates.
L63 has no partial-observation subspace (unlike L96's 8 slow + 16 fast
split) -- all 3 state channels (X, Y, Z) are observed, so a single 3-channel
stats dict covers both `states` and `obs` (see data/normalization.py,
data/dataloader.py's make_collate_fm).

Uses the S0 (unbiased) train split only, like L96's script -- S1's biases
are modest additive shifts (see config/lorenz63.yaml's cases.s1), not a
different observed subspace, so a single shared stats file is reused for
both cases' training exactly as L96 does.

Usage:
    python precompute_l63_norm_stats.py [--output experiments/l63_norm_stats.pt]
"""
import argparse
import logging

import hydra
import torch

from data.build import build_datasets
from data.normalization import compute_channel_stats, save_norm_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Precompute L63 per-channel norm stats")
    parser.add_argument("--train-case", default="s0", choices=["s0", "s1"])
    parser.add_argument("--output", default="experiments/l63_norm_stats.pt")
    args = parser.parse_args()

    with hydra.initialize(config_path="config", version_base="1.3"):
        cfg = hydra.compose("lorenz63")

    datasets, test_keys, base_cfg, system, obs_var_indices = build_datasets(cfg, train_case=args.train_case)
    assert system == "lorenz63", f"expected lorenz63, got {system}"
    train = datasets["train"]

    logger.info(f"Collecting true_state from {len(train)} windows (train_case={args.train_case})...")
    states = torch.stack([train[i]["true_state"] for i in range(len(train))])
    logger.info(f"States tensor: {tuple(states.shape)}")

    stats = compute_channel_stats(states)
    logger.info(f"mean={stats['mean'].tolist()}")
    logger.info(f"std={stats['std'].tolist()}")

    save_norm_stats(args.output, stats)
    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
