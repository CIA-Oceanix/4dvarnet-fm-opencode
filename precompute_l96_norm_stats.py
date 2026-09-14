#!/usr/bin/env python3
"""Precompute per-channel z-score stats for the L96 normalization ablation.

Generates a fresh L96 train split (train/val are never cached to disk --
only test_s0/test_s1 are) using the same ``lorenz96_default`` data config
shared by L1b/L3/FDV1, and computes per-channel mean/std over the 24D
observed subspace (8 slow + 16 fast). The result is saved once and reused
by every model/phase of the ablation, so comparisons aren't contaminated
by different-seed stat estimates across models.

Usage:
    python precompute_l96_norm_stats.py [--output experiments/l96_norm_stats_obsj2.pt]
"""
import argparse
import logging

import hydra
import torch

from data.lorenz96 import Lorenz96Config, make_l96_s0_s1_trainval
from data.normalization import compute_channel_stats, save_norm_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Precompute L96 per-channel norm stats")
    parser.add_argument("--num-train-windows", type=int, default=1000,
                        help="Train windows to generate (matches L1b/L3's num_train_windows)")
    parser.add_argument("--output", default="experiments/l96_norm_stats_obsj2.pt")
    args = parser.parse_args()

    with hydra.initialize(config_path="config", version_base="1.3"):
        cfg = hydra.compose("experiment/L1b_direct_unet_s0s1")
    dc = cfg.data

    obs_j = dc.get("obs_j", 2)
    NO, J = dc.get("NO", 8), dc.get("J", 4)
    obs_var_indices = tuple(list(range(NO)) + [NO + k * J + j for k in range(NO) for j in range(obs_j)])
    logger.info(f"Observed subspace ({len(obs_var_indices)} dims): {obs_var_indices}")

    base_cfg = Lorenz96Config(
        case=dc.get("case", 1), dt=dc.dt, T_max=dc.T_max,
        obs_interval=dc.obs_interval, R_var=dc.R_var, B_var=dc.B_var,
        param_bias=dc.get("param_bias", 0.0),
        num_windows=dc.num_windows, window_spacing=dc.window_spacing,
        spinup_steps=dc.spinup_steps, seed=dc.get("seed", 42),
        NO=NO, J=J,
        h=dc.get("h", 1.0), hx=dc.get("hx", 1.0), eps=dc.get("eps", 0.1),
        F_true=dc.get("F_true", 8.0), F_da=dc.get("F_da", 8.0),
        gamma=dc.get("gamma", 0.05), W_L_bar=dc.get("W_L_bar", 0.0),
        c1=dc.get("c1", 1.0), c2=dc.get("c2", 0.1),
        sigma_0=dc.get("sigma_0", 0.08), sigma_L=dc.get("sigma_L", 0.20),
        tau_eta=dc.get("tau_eta", 5.0),
        sigma_eta=dc.get("sigma_eta", 0.7071067811865476),
        forcing_state_bias=dc.get("forcing_state_bias", 0.0),
        forcing_coupling=dc.get("forcing_coupling", "linear"),
        coupling_exponent_truth=dc.get("coupling_exponent_truth", 1.6),
        coupling_exponent_da=dc.get("coupling_exponent_da", 1.0),
        fast_weights=list(dc.get("fast_weights", [1.0, 1.0, 0.1, 0.1])),
        randomize=dict(dc.get("randomize", {})),
        obs_var_indices=obs_var_indices,
    )

    logger.info(f"Generating {args.num_train_windows} train windows (this is the only split we need)...")
    datasets = make_l96_s0_s1_trainval(
        base_cfg,
        num_train_windows=args.num_train_windows,
        num_val_windows=1,
        num_test_windows=1,
        param_noise=dc.get("test_param_noise", 0.2),
        bias_range=(0.0, dc.get("bias_max", 0.2)),
    )

    train = datasets["train"]
    logger.info(f"Collecting true_state from {len(train)} windows...")
    states = torch.stack([train[i]["true_state"][:, obs_var_indices] for i in range(len(train))])
    logger.info(f"States tensor: {tuple(states.shape)}")

    stats = compute_channel_stats(states)
    logger.info(f"mean: {stats['mean'].tolist()}")
    logger.info(f"std:  {stats['std'].tolist()}")

    save_norm_stats(args.output, stats)
    logger.info(f"Saved stats to {args.output}")


if __name__ == "__main__":
    main()
