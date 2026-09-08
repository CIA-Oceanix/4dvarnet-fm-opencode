#!/usr/bin/env python3
"""Precompute global per-layer psi normalization stats + derived q_loss_weight
for the QG neural baseline (Q1/Q2).

Loads the production 1000-window S0 train truth cache (same seed/split
convention as `train_qg_neural.py`'s defaults), and computes:

* Global per-layer (psi1, psi2) mean/std of the daily-mean streamfunction,
  pooled over all windows/days/grid cells -- via the shared
  `data.normalization.compute_channel_stats` (the same z-score utility the
  L96 normalization ablation uses).
* Per-window per-layer psi std (the existing `window_scales` quantity) so we
  can check the window-to-window energy dynamic range at nx=64 against the
  ~30->1e4 range documented at nx=8 in PLAN.md, which was the reason
  per-window normalization was originally chosen over a global scalar.
* Global pooled variance of the raw (unnormalized) PV-q target, and a
  derived `q_loss_weight = 1 / Var(q)` so the auxiliary PV loss contributes
  comparably to the z-scored psi loss (which has target variance ~1) even
  though q itself is left in raw physical units.

Usage:
    python precompute_qg_norm_stats.py \
        --cache-dir /path/to/qg_windows_1000_100_100/cache \
        --output experiments/qg_psi_norm_stats.pt
"""
import argparse
import logging

import torch

from data.normalization import compute_channel_stats, save_norm_stats
from data.qg import QGConfig
from data.qg_neural import (
    ensure_truth_cache,
    layer_split,
    psi_daily,
    q_daily,
    window_scales,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description="Precompute QG psi norm stats + q_loss_weight")
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--train-seed", type=int, default=42)
    ap.add_argument("--num-train", type=int, default=1000)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output", default="experiments/qg_psi_norm_stats.pt")
    args = ap.parse_args()

    cfg = QGConfig(nx=args.nx, seed=args.train_seed, num_windows=args.num_train)
    split = layer_split(cfg)

    logger.info(f"Loading {args.num_train} train windows (nx={args.nx}, seed={args.train_seed})...")
    windows = ensure_truth_cache(cfg, args.num_train, args.cache_dir)
    logger.info(f"Loaded {len(windows)} windows")

    psi1_chunks, psi2_chunks = [], []
    q1_chunks, q2_chunks = [], []
    win_psi1_std, win_psi2_std = [], []
    for i, w in enumerate(windows):
        ps = psi_daily(w, cfg)
        qs = q_daily(w, cfg)
        psi1_chunks.append(ps[:, :split].reshape(-1))
        psi2_chunks.append(ps[:, split:].reshape(-1))
        q1_chunks.append(qs[:, :split].reshape(-1))
        q2_chunks.append(qs[:, split:].reshape(-1))
        sc = window_scales(w, cfg)
        win_psi1_std.append(sc.psi1)
        win_psi2_std.append(sc.psi2)
        if (i + 1) % 100 == 0:
            logger.info(f"  processed {i + 1}/{len(windows)} windows")

    psi1 = torch.cat(psi1_chunks)
    psi2 = torch.cat(psi2_chunks)
    q1 = torch.cat(q1_chunks)
    q2 = torch.cat(q2_chunks)

    psi_layers = torch.stack([psi1, psi2], dim=-1)  # (M, 2) channel = layer
    psi_stats = compute_channel_stats(psi_layers)

    q_var = torch.cat([q1, q2]).var().item()
    q_loss_weight = 1.0 / q_var

    win_psi1_std = torch.tensor(win_psi1_std)
    win_psi2_std = torch.tensor(win_psi2_std)

    logger.info(f"Global psi1: mean={psi_stats['mean'][0]:.4e} std={psi_stats['std'][0]:.4e}")
    logger.info(f"Global psi2: mean={psi_stats['mean'][1]:.4e} std={psi_stats['std'][1]:.4e}")
    logger.info(f"Global pooled q var={q_var:.4e} -> q_loss_weight=1/var={q_loss_weight:.4e}")
    logger.info("Per-window psi1 std range: "
                f"min={win_psi1_std.min():.4e} p10={win_psi1_std.quantile(0.1):.4e} "
                f"median={win_psi1_std.median():.4e} p90={win_psi1_std.quantile(0.9):.4e} "
                f"max={win_psi1_std.max():.4e}  (max/min={win_psi1_std.max() / win_psi1_std.min():.1f}x)")
    logger.info("Per-window psi2 std range: "
                f"min={win_psi2_std.min():.4e} p10={win_psi2_std.quantile(0.1):.4e} "
                f"median={win_psi2_std.median():.4e} p90={win_psi2_std.quantile(0.9):.4e} "
                f"max={win_psi2_std.max():.4e}  (max/min={win_psi2_std.max() / win_psi2_std.min():.1f}x)")

    save_norm_stats(args.output, psi_stats)
    extra = {
        "q_var": q_var,
        "q_loss_weight": q_loss_weight,
        "win_psi1_std": win_psi1_std,
        "win_psi2_std": win_psi2_std,
        "nx": args.nx,
        "train_seed": args.train_seed,
        "num_train": args.num_train,
    }
    torch.save(extra, args.output.replace(".pt", "_extra.pt"))
    logger.info(f"Saved psi norm stats to {args.output} "
                f"(+ diagnostics to {args.output.replace('.pt', '_extra.pt')})")


if __name__ == "__main__":
    main()
