#!/usr/bin/env python3
"""Inference for trained neural models on the cached L96 test dataset.

Two-step design (scheme-agnostic benchmarking):
  Step 1 (here): run the model on the S0/S1 test splits and save the state
  estimates to per-case ``.npz`` files (plus the reference truth). No metrics
  are computed here.
  Step 2: a generic evaluator (``evaluation/estimate_metrics.py``) loads any
  stored ``.npz`` (neural or DA) and computes RMSE/EV/ES identically.

This script does step 1 and then runs the generic evaluator on its own outputs
so a quick console summary + ``neural_eval.json`` are produced.
"""
import argparse
import json
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

from evaluation.estimate_metrics import evaluate_estimates, save_estimates
from evaluation.neural_inference import load_model, prepare_dataset, run_inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run a neural model on L96 S0/S1 test dataset")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--config", help="Path to config.yaml (optional)")
    parser.add_argument("--dataset", help="Path to cached test dataset .pt (optional)")
    parser.add_argument("--num-windows", type=int, default=200, help="Number of test windows")
    parser.add_argument("--obs-interval", type=int, default=100, help="Observation interval")
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--output", default="neural_eval_results.json", help="Output JSON")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    logger.info(f"Loading model: {args.checkpoint}")
    model, cfg = load_model(args.checkpoint, args.config, device=device)
    logger.info(f"Model: {type(model).__name__}, state_dim={model.state_dim}")

    # Prepare dataset
    dataset_path = args.dataset
    if not dataset_path:
        # Auto-detect the cached DA-baseline dataset in the experiments dir
        ckpt_dir = Path(args.checkpoint).parent
        exp_dir = ckpt_dir.parent
        candidates = sorted(list(ckpt_dir.glob("l96_datasets_obsj*.pt"))
                            + list(exp_dir.glob("l96_datasets_obsj*.pt")))
        if candidates:
            dataset_path = str(candidates[0])
            logger.info(f"Auto-detected dataset: {dataset_path}")

    dataset, dataloaders = prepare_dataset(cfg, dataset_path, args.num_windows, args.obs_interval)
    logger.info(f"Dataset: {len(dataset)} windows, batch={args.batch_size}")

    # Step 1: inference -> estimates
    logger.info("Running inference (step 1)...")
    estimates = run_inference(model, dataloaders, device)

    # Save per-case .npz estimates + truth, and compute generic metrics (step 2)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for case in ("s0", "s1"):
        est = estimates[case]
        npz_path = output_path.parent / f"estimates_{case}.npz"
        save_estimates(str(npz_path), est["trajectories"], est["truth"])
        metrics[case] = evaluate_estimates(est["trajectories"], est["truth"])
        logger.info(f"Saved estimates: {npz_path}")

    metrics["degradation"] = (
        float(metrics["s1"]["rmse"] / metrics["s0"]["rmse"]) if metrics["s0"]["rmse"] > 0 else float("nan")
    )

    output = {
        "checkpoint": args.checkpoint,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "dataset": {
            "path": dataset_path,
            "num_windows": args.num_windows,
            "obs_interval": args.obs_interval,
        },
        "estimates": {
            "s0": str(output_path.parent / "estimates_s0.npz"),
            "s1": str(output_path.parent / "estimates_s1.npz"),
        },
        "metrics": metrics,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=float)

    logger.info(f"\n{'='*70}")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"{'='*70}")
    for case in ("s0", "s1"):
        m = metrics[case]
        logger.info(f"[{case.upper()}] RMSE: {m['rmse']:.6f} | "
                    f"slow: {m['groups']['slow']:.6f} | obs_fast: {m['groups']['obs_fast']:.6f} | "
                    f"EV(all): {m['ev']['groups']['all_obs']:.6f} | ES(all): {m['es']['groups']['all_obs']:.6f}")
    logger.info(f"[DEGRADATION] S1/S0 RMSE: {metrics['degradation']:.6f}")
    logger.info(f"{'='*70}\n")

    return metrics


if __name__ == "__main__":
    main()
