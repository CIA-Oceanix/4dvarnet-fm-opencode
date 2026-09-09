#!/usr/bin/env python3
"""S0-only (tau0, guidance_weight) grid sweep for FDV1(monai)-warm-started
SDA3(monai), mirroring sweep_sda_directunet_hybrid_l96.py's methodology but
for a raw-obs mean model (FDV1 has no data.normalize=true) warm-starting a
normalized SDA prior -- the mean estimate is computed on a separate raw
dataloader, then normalized before use as the warm start (see
eval_sda_mean_hybrid_l96.py's --no-mean-normalized path).
"""
import itertools
import json
import logging

import torch

from data.normalization import denormalize, load_norm_stats, normalize
from evaluation.estimate_metrics import evaluate_estimates
from evaluation.neural_inference import BatchDict, load_model, prepare_dataset
from evaluation.sda_sampler import sda_guided_sample
from models.fourdvarnet import FourDVarNetSolver

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TAU0_GRID = [0.1, 0.2, 0.3, 0.4, 0.5]
GW_GRID = [0.5, 1.0, 2.0, 5.0]
MEAN_CKPT = "experiments/FDV1_unrolled_monai_unet_l96/checkpoints/stage1_best.ckpt"
MEAN_CFG = "experiments/FDV1_unrolled_monai_unet_l96/resolved_config.yaml"
MEAN_N_OUTER = 10
SDA_CKPT = "experiments/SDA3_monai_cond_noisy_l96_norm/checkpoints/stage1_best.ckpt"
SDA_CFG = "experiments/SDA3_monai_cond_noisy_l96_norm/resolved_config.yaml"
NORM_STATS_PATH = "experiments/l96_norm_stats_obsj2.pt"
DATASET_PATH = "experiments/l96_datasets_obsj2_int100_nwin200.pt"
N_OUTER = 10
R_VAR = 0.5
BATCH_SIZE = 16
SEED = 0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_stats = load_norm_stats(NORM_STATS_PATH)

    logger.info(f"Loading mean model: {MEAN_CKPT}")
    mean_model, _ = load_model(MEAN_CKPT, MEAN_CFG, device=device)
    logger.info(f"Loading SDA: {SDA_CKPT}")
    sda_model, sda_cfg = load_model(SDA_CKPT, SDA_CFG, device=device)
    assert isinstance(mean_model, FourDVarNetSolver)

    from models.sda import ConditionalPriorCFM
    is_conditioned = isinstance(sda_model, ConditionalPriorCFM)
    _, norm_dataloaders, obs_var_indices = prepare_dataset(
        sda_cfg, DATASET_PATH, 200, 100, obs_j=2, is_joint=is_conditioned,
        norm_stats=norm_stats, batch_size=BATCH_SIZE,
    )
    _, raw_dataloaders, _ = prepare_dataset(
        sda_cfg, DATASET_PATH, 200, 100, obs_j=2, is_joint=is_conditioned,
        norm_stats=None, batch_size=BATCH_SIZE,
    )

    # Precompute FDV1's raw-space mean estimate + its normalized version once
    # (identical every grid point -- only tau0/guidance_weight vary).
    mean_ests_norm, truths = [], []
    with torch.no_grad():
        for raw_batch in raw_dataloaders["s0"]:
            raw_batch = {k: v.to(device) if v is not None else v for k, v in raw_batch.items()}
            mean_est_raw = mean_model.sample(BatchDict(raw_batch), N_outer=MEAN_N_OUTER)
            mean_ests_norm.append(normalize(mean_est_raw, norm_stats))
    for norm_batch in norm_dataloaders["s0"]:
        truths.append(norm_batch["true_state"])

    results = []
    for tau0, gw in itertools.product(TAU0_GRID, GW_GRID):
        preds, truth_list = [], []
        torch.manual_seed(SEED)
        for i, norm_batch in enumerate(norm_dataloaders["s0"]):
            norm_batch = {k: v.to(device) if v is not None else v for k, v in norm_batch.items()}
            batch_obj = BatchDict(norm_batch)
            pred, _ = sda_guided_sample(
                sda_model, batch_obj, R_var=R_VAR, N_outer=N_OUTER,
                guidance_weight=gw, n_members=1,
                mean_estimate=mean_ests_norm[i], tau0=tau0,
            )
            preds.append(pred.detach().float().cpu())
            truth_list.append(norm_batch["true_state"].detach().cpu())
        trajectories = torch.cat(preds, dim=0).numpy()
        truth = torch.cat(truth_list, dim=0)
        d_pred = trajectories.shape[-1]
        if truth.shape[-1] > d_pred:
            truth = truth[..., list(obs_var_indices)] if len(obs_var_indices) == d_pred else truth[..., :d_pred]
        truth = truth.numpy()
        trajectories = denormalize(trajectories, norm_stats)
        m = evaluate_estimates(trajectories, truth)
        rmse, ev = m["rmse"], m["ev"]["groups"]["all_obs"]
        logger.info(f"tau0={tau0} gw={gw}: RMSE={rmse:.4f} EV={ev:.4f}")
        results.append({"tau0": tau0, "guidance_weight": gw, "rmse": rmse, "ev": ev})

    results.sort(key=lambda r: r["rmse"])
    with open("sda_fdv1_hybrid_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info(f"Best: {results[0]}")


if __name__ == "__main__":
    main()
