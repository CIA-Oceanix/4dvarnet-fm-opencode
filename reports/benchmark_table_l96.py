#!/usr/bin/env python3
"""Benchmark table generator: DA baselines + neural models comparison."""
import json
import logging
from pathlib import Path

import numpy as np
from tabulate import tabulate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DA baseline cache patterns
DA_CACHE_PATTERNS = [
    "experiments/l96_baselines_dws500_inf2.0_etkf_inf2.0_obsj2_int100.json",  # S0c/S1c default
    "experiments/l96_baselines_dws500_obsj2_int100.json",
]

# Neural model evaluation results (neural_eval.json) and checkpoint patterns
NEURAL_CKPT_PATTERNS = [
    "experiments/L1_direct_unet_s0s1/checkpoints/*.pt",
    "experiments/L2_vanilla_cfm_s0s1/checkpoints/*.pt",
]

# Pre-computed neural eval results (preferred over re-evaluation)
NEURAL_JSON_PATTERNS = [
    "experiments/L1_direct_unet_s0s1/neural_eval.json",
    "experiments/L2_vanilla_cfm_s0s1/neural_eval.json",
]


def load_da_baseline(cache_path: str) -> dict:
    """Load DA baseline results.

    The cache schema is ``{"config":..., "s0": {method: {...}}, "s1": {...}}``
    with per-method ``mean`` (RMSE), ``groups`` (slow/obs_fast/all_obs) and an
    ``ev`` entry with its own ``groups``.
    """
    with open(cache_path, "r") as f:
        data = json.load(f)

    results = {}
    for case in ("s0", "s1"):
        methods = data.get(case, {})
        for method, metrics in methods.items():
            key = f"{case}_{method}"
            ev_groups = metrics.get("ev", {})
            if isinstance(ev_groups, dict):
                ev_groups = ev_groups.get("groups", {})
            results[key] = {
                "rmse": metrics.get("mean", 0.0),
                "rmse_slow": metrics.get("groups", {}).get("slow", 0.0),
                "rmse_obs_fast": metrics.get("groups", {}).get("obs_fast", 0.0),
                "ev": ev_groups.get("all_obs", 0.0) if isinstance(ev_groups, dict) else 0.0,
                "es": 0.0,  # ES not recorded in DA caches
            }
    return results


def load_neural_results(checkpoint_path: str) -> dict:
    """Load neural model evaluation results.

    ``neural_eval.json`` (from ``eval_neural_l96.py``) has
    ``metrics = {"s0": {...}, "s1": {...}, "degradation": ...}`` with per-case
    ``rmse``, ``groups``, ``ev.groups`` and ``es.groups``.
    """
    results = {}

    if checkpoint_path.endswith(".json"):
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
        metrics = data.get("metrics", {})
        results = {"model": data.get("checkpoint", "unknown")}
        for case in ("s0", "s1"):
            m = metrics.get(case, {})
            g = m.get("groups", {})
            ev_g = m.get("ev", {}).get("groups", {})
            es_g = m.get("es", {}).get("groups", {})
            results[f"{case}_rmse"] = m.get("rmse", 0.0)
            results[f"{case}_slow"] = g.get("slow", 0.0)
            results[f"{case}_obs_fast"] = g.get("obs_fast", 0.0)
            results[f"{case}_ev"] = ev_g.get("all_obs", 0.0)
            results[f"{case}_es"] = es_g.get("all_obs", 0.0)
        results["degradation"] = metrics.get("degradation", 0.0)
    else:
        # Load checkpoint and run the two-step inference + generic evaluation
        from evaluation.estimate_metrics import evaluate_estimates
        from evaluation.neural_inference import load_model, prepare_dataset, run_inference

        logger.info(f"Evaluating: {checkpoint_path}")
        model, cfg = load_model(checkpoint_path)
        device = next(model.parameters()).device

        # Find dataset
        ckpt_dir = Path(checkpoint_path).parent
        dataset_candidates = sorted(list(ckpt_dir.glob("l96_datasets_obsj*.pt"))
                                    + list(ckpt_dir.parent.glob("l96_datasets_obsj*.pt")))
        if not dataset_candidates:
            raise FileNotFoundError(f"No cached L96 dataset found near {checkpoint_path}")
        dataset_path = str(dataset_candidates[0])

        _, dataloaders, obs_var_indices = prepare_dataset(
            cfg, dataset_path, num_test_windows=200, obs_interval=100, obs_j=2
        )
        estimates = run_inference(model, dataloaders, device, obs_var_indices)
        for case in ("s0", "s1"):
            m = evaluate_estimates(estimates[case]["trajectories"], estimates[case]["truth"])
            results[f"{case}_rmse"] = m["rmse"]
            results[f"{case}_slow"] = m["groups"]["slow"]
            results[f"{case}_obs_fast"] = m["groups"]["obs_fast"]
            results[f"{case}_ev"] = m["ev"]["groups"]["all_obs"]
            results[f"{case}_es"] = m["es"]["groups"]["all_obs"]
        results["degradation"] = results["s1_rmse"] / results["s0_rmse"] if results["s0_rmse"] > 0 else float("nan")
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
    
    # Load neural models: prefer pre-computed neural_eval.json results
    for pattern in NEURAL_JSON_PATTERNS:
        json_path = Path(pattern)
        if json_path.exists():
            logger.info(f"Loading neural results: {json_path}")
            neural_results[json_path] = load_neural_results(str(json_path))

    for pattern in NEURAL_CKPT_PATTERNS:
        pattern_path = Path(pattern)
        for ckpt_path in pattern_path.glob("**/*.pt"):
            # Skip checkpoints that already have a neural_eval.json
            if ckpt_path.parent.parent.joinpath("neural_eval.json").exists():
                continue
            logger.info(f"Loading neural: {ckpt_path}")
            neural_results[ckpt_path] = load_neural_results(str(ckpt_path))

    return da_results, neural_results


def _neural_model_name(path):
    """Derive the experiment/model label from a results path."""
    p = Path(path)
    # experiments/<EXP>/neural_eval.json  OR  experiments/<EXP>/checkpoints/<file>.pt
    if p.name == "neural_eval.json":
        return p.parent.name
    return p.parent.parent.name


def generate_comparison_table(da_results: dict, neural_results: dict) -> str:
    """Generate formatted comparison table (one row per method x case)."""
    rows = []

    # DA baselines header
    rows.append(["Method", "Case", "RMSE", "RMSE (slow)", "RMSE (obs)", "EV", "ES", "Type"])

    # DA baselines
    for key, metrics in sorted(da_results.items()):
        case, method = key.rsplit("_", 1)
        rows.append([f"{method[:12]:<12}", case.upper(), f"{metrics['rmse']:.4f}",
                     f"{metrics['rmse_slow']:.4f}", f"{metrics['rmse_obs_fast']:.4f}",
                     f"{metrics['ev']:.4f}", f"{metrics['es']:.4f}", "DA"])

    # Neural models (one row per model, echo S0 and S1)
    for ckpt, metrics in neural_results.items():
        model_name = _neural_model_name(ckpt)[:14]
        for case in ("s0", "s1"):
            rows.append([f"{model_name:<14}", case.upper(),
                         f"{metrics[f'{case}_rmse']:.4f}", f"{metrics[f'{case}_slow']:.4f}",
                         f"{metrics[f'{case}_obs_fast']:.4f}", f"{metrics[f'{case}_ev']:.4f}",
                         f"{metrics[f'{case}_es']:.4f}", "Neural"])
        rows.append(["", "S1/S0", f"{metrics['degradation']:.4f}", "", "", "", "", "Neural"])

    table = tabulate(rows, headers="firstrow", tablefmt="grid")
    return table


def generate_summary(da_results: dict, neural_results: dict) -> str:
    """Generate summary statistics."""
    da_rmse = [m["rmse"] for m in da_results.values()]
    neural_rmse = [metrics[f"{case}_rmse"] for metrics in neural_results.values() for case in ("s0", "s1")]

    da_avg = np.mean(da_rmse) if da_rmse else 0
    neural_avg = np.mean(neural_rmse) if neural_rmse else 0

    summary = f"""
{'='*70}
L96 NEURAL MODEL BENCHMARK SUMMARY
{'='*70}

DA Baselines (Obs30, dws=500):
  Cases x Methods: {len(da_results)}
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
