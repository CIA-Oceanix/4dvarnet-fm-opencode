#!/usr/bin/env python3
"""Inference for a trained SDA (score-based DA) prior on the cached L96 test dataset.

Same two-step, scheme-agnostic design as ``eval_neural_l96.py``:
  Step 1 (here): run the observation-guided sampler
  (``evaluation/sda_sampler.sda_guided_sample``) on the S0/S1 test splits and
  save the state estimates to per-case ``.npz`` files (plus the reference
  truth). No metrics are computed here.
  Step 2: the generic evaluator (``evaluation/estimate_metrics.py``) loads any
  stored ``.npz`` (neural, DA, or SDA) and computes RMSE/EV/ES identically.

Unlike every other L96 CFM variant, ``UnconditionalPriorCFM.sample()`` is not
conditioned on ``obs`` at all (see ``models/sda.py``), so state estimation
here goes through ``sda_guided_sample`` instead of a plain
``model.sample(batch)`` call -- ``--r-var``/``--guidance-weight`` control that
guidance term. ``r_var`` defaults to the same observation-noise variance
(``data.R_var``) the DA baselines and the guidance cost both assume; unlike
``r_var``, ``guidance_weight`` is a free DPS-style step-size knob with no
principled default (see ``evaluation/sda_sampler.py``) -- pick it via a small
sweep on S0 before trusting S1 numbers.

``--guidance-obs-j`` (distinct from ``--obs-j``, which stays pinned to the
checkpoint's native/eval dimension) restricts what the guidance cost is
allowed to see, mirroring the DA baselines' obs_j/eval_j decoupling
(``evaluation/run_l96.py``) without any retraining or new dataset cache: the
model still generates a full checkpoint-native-D sample and is scored on the
same subspace as always, but only the first ``NO + guidance_obs_j*NO``
channels of the existing cached observation (slow vars first, then the
canonical ordering's per-node fast vars) enter the guidance term. Pass 0 for
slow-only observation density.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from evaluation.estimate_metrics import (
    evaluate_ensemble_estimates,
    evaluate_estimates,
    save_estimates,
)
from evaluation.neural_inference import load_model, prepare_dataset, run_inference
from models.sda import ConditionalPriorCFM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def guidance_obs_indices(guidance_obs_j: int | None, canonical_obs_j: int, NO: int = 8):
    """Map a reduced fast-vars-per-node density onto index positions within
    the canonical (``canonical_obs_j``-built) 24D observed-subspace ordering.

    Mirrors ``evaluation/run_l96.py::make_obs_j_indices``'s (X-vars first,
    then per-node-interleaved Y-vars) layout, but computes positions *within*
    that already-built 24D array rather than the original 40D truth space.
    Returns ``None`` (no restriction) when ``guidance_obs_j`` is unset or
    already covers everything the canonical array has.
    """
    if guidance_obs_j is None or guidance_obs_j >= canonical_obs_j:
        return None
    slow = list(range(NO))
    fast = [NO + canonical_obs_j * k + j for k in range(NO) for j in range(guidance_obs_j)]
    return slow + fast


def main():
    parser = argparse.ArgumentParser(description="Run the SDA prior on the L96 S0/S1 test dataset")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .ckpt")
    parser.add_argument("--config", help="Path to config.yaml (recommended: recovers sda_prior block)")
    parser.add_argument("--dataset", help="Path to cached test dataset .pt (optional)")
    parser.add_argument("--num-windows", type=int, default=200, help="Number of test windows")
    parser.add_argument("--obs-interval", type=int, default=100, help="Observation interval")
    parser.add_argument("--obs-j", type=int, default=2, help="Fast vars observed per slow node (default: 2)")
    parser.add_argument("--guidance-obs-j", type=int, default=None,
                        help="Restrict the guidance cost to this many fast vars/node "
                             "(0 = slow-only), independent of --obs-j (default: unrestricted)")
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--n-members", type=int, default=1,
                        help="Number of stochastic members to sample (1 = single guided sample)")
    parser.add_argument("--n-outer", type=int, default=10,
                        help="Euler integration steps for the guided sampler (NFE per sample)")
    parser.add_argument("--r-var", type=float, default=0.5,
                        help="Observation-noise variance for the guidance cost (default matches data.R_var)")
    parser.add_argument("--guidance-weight", type=float, default=1.0,
                        help="DPS-style normalized-gradient step size (see evaluation/sda_sampler.py)")
    parser.add_argument("--seed", type=int, default=0, help="Torch seed before sampling")
    parser.add_argument("--cases", nargs="+", default=["s0", "s1"], choices=["s0", "s1"],
                        help="Which test cases to evaluate")
    parser.add_argument("--output", default="sda_eval_results.json", help="Output JSON")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    logger.info(f"Loading model: {args.checkpoint}")
    model, cfg = load_model(args.checkpoint, args.config, device=device)
    logger.info(f"Model: {type(model).__name__}, state_dim={model.state_dim}, "
                f"sigma_prior={model.sigma_prior}")

    # Prepare dataset
    dataset_path = args.dataset
    if not dataset_path:
        ckpt_dir = Path(args.checkpoint).parent
        exp_dir = ckpt_dir.parent
        candidates = sorted(list(ckpt_dir.glob("l96_datasets_obsj*.pt"))
                            + list(exp_dir.glob("l96_datasets_obsj*.pt")))
        if candidates:
            dataset_path = str(candidates[0])
            logger.info(f"Auto-detected dataset: {dataset_path}")

    is_conditioned = isinstance(model, ConditionalPriorCFM)
    dataset, dataloaders, obs_var_indices = prepare_dataset(
        cfg, dataset_path, args.num_windows, args.obs_interval,
        obs_j=args.obs_j, is_joint=is_conditioned,
    )
    logger.info(f"Dataset: {len(dataset)} windows, batch={args.batch_size}")
    logger.info(f"obs_var_indices ({len(obs_var_indices)} dims): {list(obs_var_indices)}")

    guide_indices = guidance_obs_indices(args.guidance_obs_j, args.obs_j)

    # Step 1: guided inference -> estimates
    torch.manual_seed(args.seed)
    logger.info(
        f"Running guided inference (step 1): cases={args.cases} n_members={args.n_members} "
        f"n_outer={args.n_outer} r_var={args.r_var} guidance_weight={args.guidance_weight} "
        f"guidance_obs_j={args.guidance_obs_j} ({len(guide_indices) if guide_indices else 'all'} dims) "
        f"seed={args.seed}"
    )
    estimates = run_inference(
        model, dataloaders, device, obs_var_indices,
        n_members=args.n_members, n_outer=args.n_outer,
        r_var=args.r_var, guidance_weight=args.guidance_weight,
        obs_indices=guide_indices,
    )

    # Save per-case .npz estimates + truth, and compute generic metrics (step 2)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {}
    estimates_paths = {}
    for case in args.cases:
        est = estimates[case]
        npz_path = output_path.parent / f"estimates_{case}.npz"
        save_estimates(str(npz_path), est["trajectories"], est["truth"])
        estimates_paths[case] = str(npz_path)
        if "members" in est:
            members_path = output_path.parent / f"members_{case}.npz"
            np.savez_compressed(members_path, members=est["members"], truth=est["truth"])
            estimates_paths[f"{case}_members"] = str(members_path)
            metrics[case] = evaluate_ensemble_estimates(est["members"], est["truth"])
            logger.info(f"Saved estimates: {npz_path} + members: {members_path}")
        else:
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
        "sampling": {
            "n_members": args.n_members,
            "n_outer": args.n_outer,
            "r_var": args.r_var,
            "guidance_weight": args.guidance_weight,
            "guidance_obs_j": args.guidance_obs_j,
            "guidance_obs_indices": guide_indices,
            "seed": args.seed,
            "cases": list(args.cases),
        },
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
        extra = ""
        if "ensemble" in m:
            es = m["ensemble"]["es"]["groups"]["all_obs"]
            sp = m["ensemble"]["spread"]["groups"]["all_obs"]
            extra = f" | ESens: {es:.6f} | spread: {sp:.6f}"
        logger.info(f"[{case.upper()}] RMSE: {m['rmse']:.6f} | "
                    f"slow: {m['groups']['slow']:.6f} | obs_fast: {m['groups']['obs_fast']:.6f} | "
                    f"EV(all): {m['ev']['groups']['all_obs']:.6f} | ES(all): {m['es']['groups']['all_obs']:.6f}"
                    f"{extra}")
    if "s0" in metrics and "s1" in metrics:
        logger.info(f"[DEGRADATION] S1/S0 RMSE: {metrics['degradation']:.6f}")
    logger.info(f"{'='*70}\n")

    return metrics


if __name__ == "__main__":
    main()
