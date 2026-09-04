#!/usr/bin/env python3
"""Inference for a trained FourDVarNetSolver (FDV1, unrolled 4DVarNet-style
solver) on the cached L96 test dataset.

Same two-step, scheme-agnostic design as ``eval_neural_l96.py``/``eval_sda_l96.py``:
  Step 1 (here): run ``model.sample(batch)`` on the S0/S1 test splits and save
  the state estimates to per-case ``.npz`` files (plus the reference truth).
  No metrics are computed here.
  Step 2: the generic evaluator (``evaluation/estimate_metrics.py``) loads any
  stored ``.npz`` (neural, DA, or SDA) and computes RMSE/EV/ES identically.

Unlike SDA, ``FourDVarNetSolver`` is fully deterministic (zero-init, no
guidance term, no random sampling of any kind) -- state estimation is a
single ``model.sample(batch)`` call, no guidance/ensemble flags needed.
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
    parser = argparse.ArgumentParser(description="Run FDV1 on the L96 S0/S1 test dataset")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .ckpt")
    parser.add_argument("--config", help="Path to config.yaml (recommended: recovers the fdv block)")
    parser.add_argument("--dataset", help="Path to cached test dataset .pt (optional)")
    parser.add_argument("--num-windows", type=int, default=200, help="Number of test windows")
    parser.add_argument("--obs-interval", type=int, default=100, help="Observation interval")
    parser.add_argument("--obs-j", type=int, default=2, help="Fast vars observed per slow node (default: 2)")
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--n-outer", type=int, default=None,
                        help="Unroll iterations (default: the trained model's own N_outer)")
    parser.add_argument("--cases", nargs="+", default=["s0", "s1"], choices=["s0", "s1"],
                        help="Which test cases to evaluate")
    parser.add_argument("--output", default="fdv1_eval_results.json", help="Output JSON")
    args = parser.parse_args()

    device = torch.device(args.device)

    logger.info(f"Loading model: {args.checkpoint}")
    model, cfg = load_model(args.checkpoint, args.config, device=device)
    n_outer = args.n_outer if args.n_outer is not None else model.N_outer
    logger.info(f"Model: {type(model).__name__}, state_dim={model.state_dim}, "
                f"update_input={model.update_input}, N_outer={n_outer}")

    dataset_path = args.dataset
    if not dataset_path:
        ckpt_dir = Path(args.checkpoint).parent
        exp_dir = ckpt_dir.parent
        candidates = sorted(list(ckpt_dir.glob("l96_datasets_obsj*.pt"))
                            + list(exp_dir.glob("l96_datasets_obsj*.pt")))
        if candidates:
            dataset_path = str(candidates[0])
            logger.info(f"Auto-detected dataset: {dataset_path}")

    dataset, dataloaders, obs_var_indices = prepare_dataset(
        cfg, dataset_path, args.num_windows, args.obs_interval, obs_j=args.obs_j,
    )
    logger.info(f"Dataset: {len(dataset)} windows, batch={args.batch_size}")
    logger.info(f"obs_var_indices ({len(obs_var_indices)} dims): {list(obs_var_indices)}")

    logger.info(f"Running inference (step 1): cases={args.cases} n_outer={n_outer}")
    estimates = run_inference(model, dataloaders, device, obs_var_indices, n_outer=n_outer)

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
        "dataset": {
            "path": dataset_path,
            "num_windows": args.num_windows,
            "obs_interval": args.obs_interval,
        },
        "sampling": {"n_outer": n_outer, "cases": list(args.cases)},
        "estimates": estimates_paths,
        "metrics": metrics,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=float)

    logger.info(f"\n{'='*70}")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"{'='*70}")
    for case in args.cases:
        m = metrics[case]
        logger.info(f"[{case.upper()}] RMSE: {m['rmse']:.6f} | "
                    f"slow: {m['groups']['slow']:.6f} | obs_fast: {m['groups']['obs_fast']:.6f} | "
                    f"EV(all): {m['ev']['groups']['all_obs']:.6f} | ES(all): {m['es']['groups']['all_obs']:.6f}")
    if "s0" in metrics and "s1" in metrics:
        logger.info(f"[DEGRADATION] S1/S0 RMSE: {metrics['degradation']:.6f}")
    logger.info(f"{'='*70}\n")

    return metrics


if __name__ == "__main__":
    main()
