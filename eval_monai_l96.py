#!/usr/bin/env python3
"""Inference for a trained MonaiDirectUNet checkpoint on the cached L96 test
dataset. Mirrors eval_neural_l96.py exactly (same estimate/metrics pipeline,
same neural_eval.json schema) but bypasses evaluation.neural_inference.
load_model()'s checkpoint-shape-inference (hardcoded to models.unet.UNet1D's
state_dict key names -- unet.enc_out, unet.downs, unet.cond_encoder.proj --
none of which exist in MonaiDirectUNet's state dict) since we already know
the exact architecture from the training config; we just build the model
directly and load its state dict, then reuse the same generic evaluation
codepath.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from evaluation.estimate_metrics import evaluate_estimates, save_estimates
from evaluation.neural_inference import prepare_dataset, run_inference
from models.monai_unet_adapter import MonaiDirectUNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_monai_direct_unet(checkpoint_path: str, config_path: str, device):
    cfg = OmegaConf.load(config_path)
    mc = cfg.model
    mdu = mc.monai_direct_unet
    model = MonaiDirectUNet(
        state_dim=mc.state_dim,
        hidden_channels=list(mdu.hidden_channels),
        dropout=mdu.get("dropout", 0.1),
        param_dim=mc.get("param_dim", 0),
        cond_extra_dim=mdu.get("cond_extra_dim", 0),
        num_res_blocks=mdu.get("num_res_blocks", 2),
        norm_num_groups=mdu.get("norm_num_groups", 32),
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    new_sd = {(k[6:] if k.startswith("model.") else k): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(new_sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.to(device)
    model.eval()
    return model, cfg


def main():
    parser = argparse.ArgumentParser(description="Run MonaiDirectUNet on L96 S0/S1 test dataset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", help="Path to cached test dataset .pt (optional)")
    parser.add_argument("--num-windows", type=int, default=200)
    parser.add_argument("--obs-interval", type=int, default=100)
    parser.add_argument("--obs-j", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cases", nargs="+", default=["s0", "s1"], choices=["s0", "s1"])
    parser.add_argument("--output", default="neural_eval_results.json")
    args = parser.parse_args()

    device = torch.device(args.device)

    logger.info(f"Loading model: {args.checkpoint}")
    model, cfg = load_monai_direct_unet(args.checkpoint, args.config, device)
    logger.info(f"Model: {type(model).__name__}, state_dim={model.state_dim}")

    dataset_path = args.dataset
    if not dataset_path:
        ckpt_dir = Path(args.checkpoint).parent
        exp_dir = ckpt_dir.parent
        candidates = sorted(list(ckpt_dir.glob("l96_datasets_obsj*.pt"))
                            + list(exp_dir.glob("l96_datasets_obsj*.pt")))
        if candidates:
            dataset_path = str(candidates[0])
            logger.info(f"Auto-detected dataset: {dataset_path}")

    # cfg loaded from the experiment yaml lacks data.system_config -- patch in
    # the minimal fields prepare_dataset() reads (NO/J/obs_j) from the base
    # lorenz96_default config the experiment yaml inherits at train time.
    if "system_config" not in cfg.data:
        OmegaConf.set_struct(cfg, False)
        cfg.data.system_config = {"NO": cfg.data.get("NO", 8), "J": cfg.data.get("J", 4)}
        cfg.data.obs_j = cfg.data.get("obs_j", args.obs_j)
        OmegaConf.set_struct(cfg, True)

    # The model was TRAINED on z-score normalized obs (data.normalize: true).
    # Eval must feed it the same normalized obs and denormalize its
    # predictions back to raw physical units before scoring against truth --
    # mirroring eval_neural_l96.py's --normalize-stats path exactly. Skipping
    # this (as an earlier version of this script did) feeds raw-scale obs to
    # weights tuned for normalized-scale obs, producing meaningless RMSE.
    norm_stats = None
    if cfg.data.get("normalize", False):
        from data.normalization import load_norm_stats
        # args.checkpoint is <exp_dir>/checkpoints/stage1.pt -- parents[2] of
        # its resolved path is the shared "experiments" directory itself.
        norm_stats_path = cfg.data.get(
            "norm_stats_path",
            str(Path(args.checkpoint).resolve().parents[2] / "l96_norm_stats_obsj2.pt"),
        )
        norm_stats = load_norm_stats(norm_stats_path)
        logger.info(f"data.normalize=True: loaded per-channel stats from {norm_stats_path}")

    dataset, dataloaders, obs_var_indices = prepare_dataset(
        cfg, dataset_path, args.num_windows, args.obs_interval, obs_j=args.obs_j,
        norm_stats=norm_stats,
    )
    logger.info(f"Dataset: {len(dataset)} windows, batch={args.batch_size}")
    logger.info(f"obs_var_indices ({len(obs_var_indices)} dims): {list(obs_var_indices)}")

    torch.manual_seed(args.seed)
    logger.info(f"Running inference (step 1): cases={args.cases} seed={args.seed}")
    estimates = run_inference(model, dataloaders, device, obs_var_indices)

    if norm_stats is not None:
        from data.normalization import denormalize
        for est in estimates.values():
            est["trajectories"] = denormalize(est["trajectories"], norm_stats)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {}
    estimates_paths = {}
    for case in args.cases:
        est = estimates[case]
        npz_path = output_path.parent / f"estimates_{case}.npz"
        save_estimates(str(npz_path), est["trajectories"], est["truth"])
        estimates_paths[case] = str(npz_path)
        metrics[case] = evaluate_estimates(est["trajectories"], est["truth"])
        logger.info(f"Saved estimates: {npz_path}")

    metrics["degradation"] = (
        float(metrics["s1"]["rmse"] / metrics["s0"]["rmse"])
        if "s0" in metrics and "s1" in metrics and metrics["s0"]["rmse"] > 0
        else float("nan")
    )

    output = {
        "checkpoint": args.checkpoint,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "dataset": {"path": dataset_path, "num_windows": args.num_windows, "obs_interval": args.obs_interval},
        "sampling": {"n_members": 1, "n_outer": 1, "seed": args.seed, "cases": list(args.cases)},
        "estimates": estimates_paths,
        "metrics": metrics,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=float)

    logger.info(f"\n{'='*70}")
    logger.info(f"Results saved to: {output_path}")
    for case in args.cases:
        m = metrics[case]
        logger.info(f"[{case.upper()}] RMSE: {m['rmse']:.6f} | "
                    f"slow: {m['groups']['slow']:.6f} | obs_fast: {m['groups']['obs_fast']:.6f} | "
                    f"EV(all): {m['ev']['groups']['all_obs']:.6f} | ES(all): {m['es']['groups']['all_obs']:.6f}")
    logger.info(f"[DEGRADATION] S1/S0 RMSE: {metrics['degradation']:.6f}")
    logger.info(f"{'='*70}\n")
    return metrics


if __name__ == "__main__":
    main()
