#!/usr/bin/env python3
"""Evaluate the trained Q1 DirectUNet checkpoint against a redrawn
lag=5.0d/noise_frac=0.05 test set (the DA baselines' reference-case obs/IC
protocol) -- NOT what the checkpoint was trained on (train_qg_neural.py's
defaults were lag=1.0d/noise=0.01), so this is a train/test-distribution-
mismatch diagnostic, not a fair like-for-like retrain. Reuses the cached S0
truth trajectories (obs_noise_std_frac/init_lag_days don't affect truth
generation, only obs/IC sampling -- see data/qg.py's `_generate_obs_ic`
docstring) and redraws obs/init-state at the new lag/noise, exactly as
qg_da_s1_scratch.py does for the DA baselines.

Usage:
    python eval_qg_q1_lag5_noise05.py --ckpt experiments/Q1_direct_unet_s0_monai2d/stage1_best.pt
"""
import argparse
import json

import numpy as np
import torch

from data.normalization import load_norm_stats
from data.qg import QGConfig, QGS01Dataset, _truth_cache_path
from data.qg_neural import psi_daily, psi_to_q, q_daily
from train_qg_neural import estimate_windows, layer_summary, pooled_metrics

CACHE_DIR = "/Odyssey/private/rfablet/Python/4dvarnet-fm-qg-100samples/reports/qg/outputs/qg_windows_1000_100_100/cache"
# Matches the production test cache's key exactly (see qg_da_s1_scratch.py).
CACHE_KW = dict(nx=64, seed=20_042, num_windows=100, obs_geometry="random_columns",
                cols_per_day=4, obs_noise_std_frac=0.01, init_lag_days=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num-test", type=int, default=100)
    ap.add_argument("--lag-days", type=float, default=5.0)
    ap.add_argument("--noise-frac", type=float, default=0.05)
    ap.add_argument("--norm-stats-path", default="experiments/qg_psi_norm_stats.pt")
    ap.add_argument("--out", default="reports/qg/outputs/qg_repro_validation/"
                                     "q1_direct_unet_lag5_noise05.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_cfg = QGConfig(**CACHE_KW)
    path = _truth_cache_path(cache_cfg, CACHE_KW["num_windows"], CACHE_DIR)
    print(f"Loading cached truth from {path} ...", flush=True)
    base = torch.load(path, map_location="cpu")[:args.num_test]
    print(f"Loaded {len(base)} base truth windows.", flush=True)

    eval_cfg = QGConfig(**{**CACHE_KW, "obs_noise_std_frac": args.noise_frac,
                           "init_lag_days": args.lag_days})
    s0_raw = QGS01Dataset(eval_cfg, "test_s0", base_windows=base).windows
    ic = QGS01Dataset._generate_obs_ic(eval_cfg, s0_raw, list(range(len(s0_raw))))
    windows = [dict(w, **entry) for w, entry in zip(s0_raw, ic)]
    print(f"Redrew obs/init-state at lag={args.lag_days}d, "
          f"noise_frac={args.noise_frac}.", flush=True)

    norm = load_norm_stats(args.norm_stats_path)
    from models.monai_unet_qg2d import MonaiDirectUNetQG
    model = MonaiDirectUNetQG(ny=eval_cfg.ny, nx=eval_cfg.nx, nlayers=2, param_dim=0,
                              cond_extra_dim=0, hidden_channels=[64, 128, 256])
    loaded = torch.load(args.ckpt, map_location="cpu")
    state_dict = loaded["state_dict"] if isinstance(loaded, dict) and "state_dict" in loaded else loaded
    state_dict = {(k[6:] if k.startswith("model.") else k): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Loaded {args.ckpt}", flush=True)

    est_psi, est_rd = estimate_windows(model, windows, eval_cfg, "direct_unet", device, norm=norm)
    truth_psi = np.stack([psi_daily(w, eval_cfg).numpy() for w in windows])
    truth_q = np.stack([q_daily(w, eval_cfg).numpy() for w in windows])
    est_q = np.stack([
        psi_to_q(torch.tensor(est_psi[i], dtype=torch.float32), float(est_rd[i]),
                 eval_cfg, device=device).cpu().numpy()
        for i in range(len(windows))
    ])
    rmse_psi, ev_psi = pooled_metrics(est_psi, truth_psi)
    rmse_q, ev_q = pooled_metrics(est_q, truth_q)
    summ_psi = layer_summary(rmse_psi, ev_psi, eval_cfg)
    summ_q = layer_summary(rmse_q, ev_q, eval_cfg)

    print(f"\n=== Q1 @ lag={args.lag_days}d, noise={args.noise_frac} (N={len(windows)}) ===")
    print(f"  PSI  pooled RMSE {summ_psi['pooled_rmse']:.6e}  EV {summ_psi['pooled_ev']:.4f}"
          f"   layer1 EV {summ_psi['layer1']['ev']:.4f}   layer2 EV {summ_psi['layer2']['ev']:.4f}")
    print(f"  PV-q pooled RMSE {summ_q['pooled_rmse']:.6e}  EV {summ_q['pooled_ev']:.4f}"
          f"   layer1 EV {summ_q['layer1']['ev']:.4f}   layer2 EV {summ_q['layer2']['ev']:.4f}")

    result = {
        "model_type": "direct_unet",
        "ckpt": args.ckpt,
        "eval_config": {"lag_days": args.lag_days, "noise_frac": args.noise_frac,
                        "num_windows": len(windows)},
        # This checkpoint was TRAINED at train_qg_neural.py's old defaults
        # (lag=1.0d, noise_frac=0.01) -- this eval redraws obs/init-state at
        # the DA baselines' reference-case setting for a closer (but still
        # train/test-distribution-mismatched, not a matched retrain) look.
        "train_test_mismatch_caveat": (
            "Checkpoint trained at lag=1.0d/noise=0.01 (train_qg_neural.py "
            "defaults); this eval redraws obs at lag=5.0d/noise=0.05 to "
            "match the DA baselines' reference case, but the model was NOT "
            "retrained for this distribution -- not a fully fair comparison."
        ),
        "psi": summ_psi,
        "q": summ_q,
    }
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
