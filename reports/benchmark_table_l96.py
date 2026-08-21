#!/usr/bin/env python3
"""Benchmark table generator: DA baselines + neural models comparison."""
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from tabulate import tabulate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DA baseline cache patterns
DA_CACHE_PATTERNS = [
    "l96_baselines_dws500_inf2.0_etkf_inf2.0_obsj2_int100.json",  # S0c/S1c default
    "l96_baselines_dws500_obsj2_int100.json",
]

# Neural model checkpoint patterns
NEURAL_CKPT_PATTERNS = [
    "L1_direct_unet_s0s1/checkpoints/*.pt",
    "L2_vanilla_cfm_s0s1/checkpoints/*.pt",
]


def load_da_baseline(cache_path: str) -> dict:
    """Load DA baseline results."""
    with open(cache_path, "r") as f:
        data = json.load(f)
    
    # Extract S0 and S1 results
    results = {}
    for case, methods in data.get("baselines", {}).items():
        for method, metrics in methods.items():
            key = f"{case}_{method}"
            results[key] = {
                "rmse": metrics.get("rmse", []),
                "ev": metrics.get("ev", []),
                "es": metrics.get("es", []),
            }
    
    return results


def load_neural_results(checkpoint_path: str) -> dict:
    """Load neural model evaluation results."""
    results = {}
    
    if checkpoint_path.endswith(".json"):
        # Already evaluated
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
        results = {
            "model": data.get("checkpoint", "unknown"),
            "rmse": data.get("metrics", {}).get("rmse", 0),
            "rmse_slow": data.get("metrics", {}).get("rmse_slow", [0]),
            "rmse_obs": data.get("metrics", {}).get("rmse_obs_fast", [0]),
            "ev": data.get("metrics", {}).get("ev", 0),
            "es": data.get("metrics", {}).get("es", 0),
        }
    else:
        # Load checkpoint and evaluate
        from evaluation.neural_inference import load_model, prepare_dataset, evaluate_model
        import torch
        
        logger.info(f"Evaluating: {checkpoint_path}")
        model, cfg = load_model(checkpoint_path)
        device = next(model.parameters()).device
        
        # Find dataset
        ckpt_dir = Path(checkpoint_path).parent
        dataset_path = list(ckpt_dir.glob("l96_datasets_obsj2_int100_nwin200.pt"))[0]
        
        dataset, dataloader = prepare_dataset(cfg, dataset_path, num_test_windows=200, obs_interval=100)
        results = evaluate_model(model, dataloader, device)
        results["model"] = checkpoint_path
    
    return results


def find_all_results() -> tuple:
    """Find all DA and neural results."""
    da_results = {}
    neural_results = {}
    
    # Load DA baselines
    for pattern in DA_CACHE_PATTERNS:
        cache_path = Path(pattern)
        if cache_path.exists():
            logger.info(f"Loading DA baseline: {cache_path}")
            da_results.update(load_da_baseline(str(cache_path)))
    
    # Load neural models
    for pattern in NEURAL_CKPT_PATTERNS:
        pattern_path = Path(pattern)
        for ckpt_path in pattern_path.glob("**/*.pt"):
            logger.info(f"Loading neural: {ckpt_path}")
            neural_results[ckpt_path] = load_neural_results(str(ckpt_path))
    
    return da_results, neural_results


def generate_comparison_table(da_results: dict, neural_results: dict) -> str:
    """Generate formatted comparison table."""
    rows = []
    
    # DA baselines header
    rows.append(["Method", "Case", "RMSE", "RMSE (slow)", "RMSE (obs)", "EV", "ES", "Type"])
    
    # DA baselines
    for key, metrics in da_results.items():
        case, method = key.rsplit("_", 1)
        rmse = metrics["rmse"][0] if isinstance(metrics["rmse"], list) else metrics["rmse"]
        rmse_slow = metrics["rmse_slow"][0] if isinstance(metrics["rmse_slow"], list) else metrics["rmse_slow"]
        rmse_obs = metrics["rmse_obs_fast"][0] if isinstance(metrics["rmse_obs_fast"], list) else metrics["rmse_obs_fast"]
        ev = metrics["ev"][0] if isinstance(metrics["ev"], list) else metrics["ev"]
        es = metrics["es"][0] if isinstance(metrics["es"], list) else metrics["es"]
        
        rows.append([f"{method[:12]:<12}", f"{case[:6]:<6}", f"{rmse:.4f}", f"{rmse_slow:.4f}", 
                     f"{rmse_obs:.4f}", f"{ev:.4f}", f"{es:.4f}", "DA"])
    
    # Neural models
    for ckpt, metrics in neural_results.items():
        model_name = Path(ckpt).stem
        rmse = metrics.get("rmse", 0)
        rmse_slow = metrics.get("rmse_slow", [0])
        rmse_obs = metrics.get("rmse_obs", [0])
        ev = metrics.get("ev", 0)
        es = metrics.get("es", 0)
        
        rows.append([f"{model_name:<12}", "S0/S1", f"{rmse:.4f}", 
                     f"{rmse_slow[0]:.4f}", f"{rmse_obs[0]:.4f}", 
                     f"{ev:.4f}", f"{es:.4f}", "Neural"])
    
    table = tabulate(rows, headers="firstrow", tablefmt="grid")
    return table


def generate_summary(da_results: dict, neural_results: dict) -> str:
    """Generate summary statistics."""
    da_rmse = [m["rmse"][0] for m in da_results.values() if isinstance(m["rmse"], list)]
    neural_rmse = [m.get("rmse", 0) for m in neural_results.values()]
    
    da_avg = np.mean(da_rmse) if da_rmse else 0
    neural_avg = np.mean(neural_rmse) if neural_rmse else 0
    
    summary = f"""
{'='*70}
L96 NEURAL MODEL BENCHMARK SUMMARY
{'='*70}

DA Baselines (Obs30, dws=500):
  Methods: {len(da_results)}
  Avg RMSE: {da_avg:.4f}
  Best: {min(da_rmse):.4f}

Neural Models:
  Models: {len(neural_results)}
  Avg RMSE: {neural_avg:.4f}
  Best: {min(neural_rmse):.4f}

Delta (Neural - DA): {neural_avg - da_avg:+.4f}
{'='*70}
"""
    return summary


def main():
    """Generate benchmark table."""
    output_dir = Path("reports/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading results...")
    da_results, neural_results = find_all_results()
    
    if not da_results:
        logger.warning("No DA baseline results found")
    if not neural_results:
        logger.warning("No neural model results found")
    
    logger.info("Generating table...")
    table = generate_comparison_table(da_results, neural_results)
    summary = generate_summary(da_results, neural_results)
    
    # Save table
    table_path = output_dir / "neural_benchmark_table.md"
    with open(table_path, "w") as f:
        f.write(summary)
        f.write(table)
        f.write("\n")
    
    logger.info(f"Table saved: {table_path}")
    print(table)
    print(summary)


if __name__ == "__main__":
    main()
