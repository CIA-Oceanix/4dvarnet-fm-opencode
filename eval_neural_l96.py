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

import numpy as np
import torch
from omegaconf import OmegaConf

from evaluation.estimate_metrics import (
    save_members_or_scores,
    evaluate_ensemble_estimates,
    evaluate_estimates,
    save_estimates,
)
from evaluation.neural_inference import load_model, prepare_dataset, run_inference
from models.vanilla_cfm import DEFAULT_STEP_POWER

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run a neural model on L96 S0/S1 test dataset")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--config", help="Path to config.yaml (optional)")
    parser.add_argument("--dataset", help="Path to cached test dataset .pt (optional)")
    parser.add_argument("--num-windows", type=int, default=200, help="Number of test windows")
    parser.add_argument("--obs-interval", type=int, default=100, help="Observation interval")
    parser.add_argument("--obs-j", type=int, default=2, help="Fast vars observed per slow node (default: 2)")
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--train-tau0-only", action="store_true",
                        help="Load with train_tau_0_only=True (tau=0-trained CFM checkpoints)")
    parser.add_argument("--n-members", type=int, default=1,
                        help="Number of stochastic members to sample (CFM; 1 = legacy single sample)")
    parser.add_argument("--n-outer", type=int, default=1,
                        help="Euler integration steps for CFM sampling")
    parser.add_argument("--step-power", type=float, default=None,
                        help="CFM Euler grid tau_k = 1-(1-k/N)^p (VanillaCFM/PredictStateCFM). "
                             "Default: the models' DEFAULT_STEP_POWER (0.5, early-fine, since "
                             "2026-09-24); 1.0 reproduces the pre-2026-09-24 uniform grid.")
    parser.add_argument("--seed", type=int, default=0, help="Torch seed before sampling")
    parser.add_argument("--cases", nargs="+", default=["s0", "s1"], choices=["s0", "s1"],
                        help="Which test cases to evaluate")
    parser.add_argument("--output", default="neural_eval_results.json", help="Output JSON")
    parser.add_argument("--sigma-prior", type=float, default=None,
                        help="override the flow's prior std at SAMPLING time (calibration probe)")
    parser.add_argument("--members-file", default="full", choices=["full", "scores"],
                        help="full: members_<case>.npz (GBs); scores: per-window rmse/crps/spread "
                             "only (scores_<case>.npz, KBs)")
    parser.add_argument("--normalize-stats", default=None,
                        help="Path to a per-channel norm stats .pt (mean/std). When given, "
                             "obs is z-score normalized before each model call and predictions "
                             "are denormalized back to raw physical units before scoring. "
                             "Omitting this flag is a true no-op (identical to not passing it).")
    parser.add_argument("--trueprior-phi-source-s1", default="true", choices=["true", "biased"],
                        help="Only meaningful for update_input='subgrad+state+trueprior' "
                             "(a true no-op for every other model): 'true' (default) feeds "
                             "the S1 window's own genuinely-true forcing/params into the ODE "
                             "prior Phi, matching how the 'perfect-model' checkpoints were "
                             "trained; 'biased' feeds the DA-biased forcing/params instead, "
                             "stress-testing how the trained solver degrades when Phi's own "
                             "inputs are wrong (the S0-true/S1-noisy evaluation plan). S0 "
                             "always uses the true values regardless of this flag.")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    logger.info(f"Loading model: {args.checkpoint}")
    overrides = {"train_tau_0_only": True} if args.train_tau0_only else None
    model, cfg = load_model(args.checkpoint, args.config, device=device, overrides=overrides)
    if args.sigma_prior is not None:
        if not hasattr(model, "sigma_prior"):
            raise ValueError(f"--sigma-prior given but {type(model).__name__} has no sigma_prior")
        logger.info(f"sigma_prior override: {model.sigma_prior} -> {args.sigma_prior} (sampling only)")
        model.sigma_prior = args.sigma_prior
    logger.info(f"Model: {type(model).__name__}, state_dim={model.state_dim}")
    if hasattr(model, "train_tau_0_only"):
        logger.info(f"train_tau_0_only={model.train_tau_0_only}")
    # subgrad+state+trueprior: state_dim IS the full physical state (e.g. 40D
    # for NO=8,J=4), not the 24D observed subspace every other config uses --
    # model.obs_var_indices (set only for this mode) bridges the two. Needs
    # the window's true forcing/params fed to the ODE prior at inference too.
    needs_true_forcing = getattr(model, "obs_var_indices", None) is not None
    if needs_true_forcing:
        logger.info(
            f"Detected subgrad+state+trueprior (full_state_target): state_dim="
            f"{model.state_dim}, scoring restricted to obs_var_indices "
            f"({len(model.obs_var_indices)} dims); S1 Phi source="
            f"{args.trueprior_phi_source_s1}"
        )

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

    # Keyed off the CONFIG, not off whether --normalize-stats was passed. The
    # previous form silently skipped normalization when the flag was omitted
    # for a model trained with data.normalize=true, feeding raw-scale obs to
    # weights tuned for normalized-scale obs -- which does not fail, it just
    # produces meaningless RMSE (the same bug eval_monai_l96.py's comment
    # records having been bitten by).
    norm_stats = None
    if cfg.data.get("normalize", False) or args.normalize_stats:
        from data.normalization import load_norm_stats
        from evaluation.archive import resolve_norm_stats
        norm_stats_path = args.normalize_stats or resolve_norm_stats(cfg, required=True)
        norm_stats = load_norm_stats(str(norm_stats_path))
        logger.info(f"Loaded normalize-stats from {norm_stats_path}: "
                    f"mean/std shape {tuple(norm_stats['mean'].shape)}")

    dataset, dataloaders, obs_var_indices = prepare_dataset(
        cfg, dataset_path, args.num_windows, args.obs_interval,
        obs_j=args.obs_j, norm_stats=norm_stats, batch_size=args.batch_size,
        needs_true_forcing=needs_true_forcing,
    )
    if needs_true_forcing:
        # model.obs_var_indices is the SAME formula (data.NO/J/obs_j) as
        # prepare_dataset's own resolved obs_var_indices when obs_j<J -- assert
        # rather than silently trusting two independently-derived index lists
        # to agree, since a mismatch here would silently score the wrong
        # channels against each other.
        assert tuple(model.obs_var_indices) == tuple(obs_var_indices), (
            f"model.obs_var_indices {tuple(model.obs_var_indices)} != "
            f"prepare_dataset's obs_var_indices {tuple(obs_var_indices)}"
        )
    logger.info(f"Dataset: {len(dataset)} windows, batch={args.batch_size}")
    logger.info(f"obs_var_indices ({len(obs_var_indices)} dims): {list(obs_var_indices)}")

    # Step 1: inference -> estimates
    torch.manual_seed(args.seed)
    logger.info(
        f"Running inference (step 1): cases={args.cases} n_members={args.n_members} "
        f"n_outer={args.n_outer} seed={args.seed}"
    )
    estimates = run_inference(
        model, dataloaders, device, obs_var_indices,
        n_members=args.n_members, n_outer=args.n_outer,
        trueprior_phi_source_s1=args.trueprior_phi_source_s1,
        step_power=args.step_power,
    )

    if norm_stats is not None:
        # obs (and, for training, states) were fed to the model normalized;
        # predictions come back in normalized space and must be denormalized
        # to raw physical units before scoring (truth is already raw, since
        # collate_eval never touches true_state).
        from data.normalization import denormalize
        for est in estimates.values():
            est["trajectories"] = denormalize(est["trajectories"], norm_stats)
            if "members" in est:
                # members: (W, T, D, M) -- channel dim D is second-to-last, not
                # last, so swap it into the last axis for the per-channel
                # broadcast then swap back.
                m = np.swapaxes(est["members"], -1, -2)
                m = denormalize(m, norm_stats)
                est["members"] = np.swapaxes(m, -1, -2)

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
            members_path = save_members_or_scores(output_path.parent, case, est["members"],
                                                  est["truth"], args.members_file)
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
            "step_power": args.step_power if args.step_power is not None else DEFAULT_STEP_POWER,
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
    logger.info(f"[DEGRADATION] S1/S0 RMSE: {metrics['degradation']:.6f}")
    logger.info(f"{'='*70}\n")

    return metrics


if __name__ == "__main__":
    main()
