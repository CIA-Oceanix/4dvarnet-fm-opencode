#!/usr/bin/env python3
"""CLI script to evaluate trained neural models on cached L96 test dataset."""
import argparse
import json
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

from evaluation.neural_inference import load_model, prepare_dataset, evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate neural model on L96 test dataset")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--config", help="Path to config.yaml (optional)")
    parser.add_argument("--dataset", help="Path to cached test dataset .pt (optional)")
    parser.add_argument("--num-windows", type=int, default=200, help="Number of test windows")
    parser.add_argument("--obs-interval", type=int, default=100, help="Observation interval")
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic mode (DirectUNet only)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", default="neural_eval_results.json", help="Output JSON")
    parser.add_argument("--exp-dir", default="experiments", help="Experiment directory")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load model
    logger.info(f"Loading model: {args.checkpoint}")
    model, cfg = load_model(args.checkpoint, args.config, device=device)
    logger.info(f"Model: {type(model).__name__}, state_dim={model.state_dim}")
    
    # Prepare dataset
    dataset_path = args.dataset
    if not dataset_path:
        ckpt_dir = Path(args.checkpoint).parent
        candidates = list(ckpt_dir.glob("l96_datasets_obsj*.pt"))
        if candidates:
            dataset_path = str(candidates[0])
            logger.info(f"Auto-detected dataset: {dataset_path}")
    
    dataset, dataloader = prepare_dataset(cfg, dataset_path, args.num_windows, args.obs_interval)
    logger.info(f"Dataset: {len(dataset)} windows, batch={args.batch_size}")
    
    # Evaluate
    logger.info("Evaluating...")
    results = evaluate_model(model, dataloader, device)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output = {
        "checkpoint": args.checkpoint,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "dataset": {
            "path": dataset_path,
            "num_windows": args.num_windows,
            "obs_interval": args.obs_interval,
        },
        "metrics": results,
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"{'='*60}")
    logger.info(f"RMSE:          {results['rmse']:.6f}")
    logger.info(f"RMSE (slow):   {results['rmse_slow']:.6f}")
    logger.info(f"RMSE (obs):    {results['rmse_obs_fast']:.6f}")
    logger.info(f"EV:            {results['ev']:.6f}")
    logger.info(f"ES:            {results['es']:.6f}")
    logger.info(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    main()
