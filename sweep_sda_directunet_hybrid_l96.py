#!/usr/bin/env python3
"""S0-only (tau0, guidance_weight) grid sweep for the DirectUNet-M-warm-started
SDA hybrids (SDA1/SDA2/SDA3), mirroring the FDV1+SDA hybrids' sweep
methodology (CHANGELOG: tau0 in {0,0.3,0.5,0.7,0.8} x guidance_weight in
{0,1,2,5,10,40,100}, S0-only, n_members=1, RMSE-only ranking).

Loads each mean/SDA model pair once and reuses them across the whole grid
(unlike calling eval_sda_directunet_hybrid_l96.py once per combo, which would
reload both models from scratch every time).
"""
import itertools
import json
import logging

import torch

from data.normalization import denormalize, load_norm_stats
from evaluation.estimate_metrics import evaluate_estimates
from evaluation.neural_inference import BatchDict, load_model, prepare_dataset
from evaluation.sda_sampler import sda_guided_sample
from models.sda import ConditionalPriorCFM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TAU0_GRID = [0.0, 0.3, 0.5, 0.7, 0.8]
GW_GRID = [0.0, 1.0, 2.0, 5.0, 10.0, 40.0, 100.0]
MEAN_CKPT = "experiments/L1b_monai_unet_s0s1_norm/checkpoints/stage1_best.ckpt"
MEAN_CFG = "experiments/L1b_monai_unet_s0s1_norm/resolved_config.yaml"
SDA_VARIANTS = {
    "SDA1": ("experiments/SDA1_monai_prior_l96_norm/checkpoints/stage1_best.ckpt",
             "experiments/SDA1_monai_prior_l96_norm/resolved_config.yaml"),
    "SDA2": ("experiments/SDA2_monai_cond_mixed_l96_norm/checkpoints/stage1_best.ckpt",
             "experiments/SDA2_monai_cond_mixed_l96_norm/resolved_config.yaml"),
    "SDA3": ("experiments/SDA3_monai_cond_noisy_l96_norm/checkpoints/stage1_best.ckpt",
             "experiments/SDA3_monai_cond_noisy_l96_norm/resolved_config.yaml"),
}
NORM_STATS_PATH = "experiments/l96_norm_stats_obsj2.pt"
DATASET_PATH = "experiments/l96_datasets_obsj2_int100_nwin200.pt"
N_OUTER = 10
R_VAR = 0.5
BATCH_SIZE = 16
SEED = 0


def run_one(mean_model, sda_model, dataloader, device, tau0, guidance_weight, obs_var_indices, norm_stats):
    preds, truths = [], []
    torch.manual_seed(SEED)
    for batch in dataloader:
        batch = {k: v.to(device) if v is not None else v for k, v in batch.items()}
        batch_obj = BatchDict(batch)
        with torch.no_grad():
            mean_est = mean_model(batch_obj)
        pred, _ = sda_guided_sample(
            sda_model, batch_obj, R_var=R_VAR, N_outer=N_OUTER,
            guidance_weight=guidance_weight, n_members=1,
            mean_estimate=mean_est, tau0=tau0,
        )
        preds.append(pred.detach().float().cpu())
        truths.append(batch["true_state"].detach().cpu())
    trajectories = torch.cat(preds, dim=0).numpy()
    truth = torch.cat(truths, dim=0)
    d_pred = trajectories.shape[-1]
    if truth.shape[-1] > d_pred:
        if obs_var_indices is not None and len(obs_var_indices) == d_pred:
            truth = truth[..., list(obs_var_indices)]
        else:
            truth = truth[..., :d_pred]
    truth = truth.numpy()
    trajectories = denormalize(trajectories, norm_stats)
    return evaluate_estimates(trajectories, truth)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_stats = load_norm_stats(NORM_STATS_PATH)
    logger.info(f"Loading mean-estimate model: {MEAN_CKPT}")
    mean_model, _ = load_model(MEAN_CKPT, MEAN_CFG, device=device)

    results = {}
    for sda_name, (sda_ckpt, sda_cfg_path) in SDA_VARIANTS.items():
        logger.info(f"=== {sda_name} ===")
        logger.info(f"Loading SDA: {sda_ckpt}")
        sda_model, sda_cfg = load_model(sda_ckpt, sda_cfg_path, device=device)
        is_conditioned = isinstance(sda_model, ConditionalPriorCFM)
        dataset, dataloaders, obs_var_indices = prepare_dataset(
            sda_cfg, DATASET_PATH, 200, 100, obs_j=2, is_joint=is_conditioned,
            norm_stats=norm_stats, batch_size=BATCH_SIZE,
        )
        dl = dataloaders["s0"]

        variant_results = []
        for tau0, gw in itertools.product(TAU0_GRID, GW_GRID):
            if tau0 == 0.0 and gw != GW_GRID[0]:
                # tau0=0 (no warm start) makes guidance_weight the only knob
                # that matters for the pure-noise-start baseline; skip the
                # redundant repeats across the rest of the tau0=0 row except
                # one representative point already covered by SDA's own
                # already-established guidance_weight=40 pure-noise result.
                if gw not in (40.0,):
                    continue
            m = run_one(mean_model, sda_model, dl, device, tau0, gw, obs_var_indices, norm_stats)
            rmse = m["rmse"]
            ev = m["ev"]["groups"]["all_obs"]
            logger.info(f"{sda_name} tau0={tau0} gw={gw}: RMSE={rmse:.4f} EV={ev:.4f}")
            variant_results.append({"tau0": tau0, "guidance_weight": gw, "rmse": rmse, "ev": ev})

        variant_results.sort(key=lambda r: r["rmse"])
        results[sda_name] = variant_results
        logger.info(f"--- {sda_name} best: {variant_results[0]} ---")

    with open("sda_directunet_hybrid_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info("Sweep complete. Best per variant:")
    for sda_name, variant_results in results.items():
        logger.info(f"  {sda_name}: {variant_results[0]}")


if __name__ == "__main__":
    main()
