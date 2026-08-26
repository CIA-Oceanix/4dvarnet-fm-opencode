#!/usr/bin/env python3
"""Inference + evaluation for L96 joint state-parameter neural models.

Port of ``eval_neural_l96.py`` for joint models: loads a JointCFM /
JointDirectUNet checkpoint, runs it on the cached S0/S1 test dataset (the same
one the joint DA baselines use), and reports per-case state RMSE/EV/ES (pooled,
per-group) plus the 8-param RMSE vector (F, c1, hx, eps, w1..w4). Output goes
to a ``joint_neural_eval.json`` mirroring the DA comparator's conventions so a
report script can merge joint-neural and joint-DA rows apples-to-apples.
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
    nrmse_param,
    trajectory_forecast_skill,
)
from evaluation.metrics import param_rmse
from evaluation.neural_inference import (
    L96_JOINT_PARAM_NAMES,
    load_model,
    prepare_dataset,
    run_inference,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PD = len(L96_JOINT_PARAM_NAMES)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a joint L96 neural model on S0/S1")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--config", help="Path to config.yaml (optional)")
    parser.add_argument("--dataset", help="Path to cached test dataset .pt (optional)")
    parser.add_argument("--num-windows", type=int, default=200)
    parser.add_argument("--obs-interval", type=int, default=100)
    parser.add_argument("--obs-j", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-tau0-only", action="store_true",
                        help="Load with train_tau_0_only=True (tau=0-trained joint CFM)")
    parser.add_argument("--n-outer", type=int, default=None,
                        help="Euler integration steps for joint CFM sampling")
    parser.add_argument("--n-members", type=int, default=1,
                        help="Number of ensemble members to sample")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for torch")
    parser.add_argument("--cases", nargs="+", default=["s0", "s1"], choices=["s0", "s1"])
    parser.add_argument("--output", default="joint_neural_eval.json", help="Output JSON")
    parser.add_argument("--n-compare-steps", type=int, default=300,
                        help="Forecast horizon (steps) for the parameter-sensitivity "
                             "trajectory metric (true vs estimated params, same x0/forcing)")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    overrides = {"train_tau_0_only": True} if args.train_tau0_only else None
    model, cfg = load_model(args.checkpoint, args.config, device=device, overrides=overrides)
    logger.info(f"Model: {type(model).__name__}, state_dim={model.state_dim}, param_dim={model.param_dim}")

    dataset_path = args.dataset
    if not dataset_path:
        ckpt_dir = Path(args.checkpoint).parent
        exp_dir = ckpt_dir.parent
        candidates = sorted(list(ckpt_dir.glob("l96_datasets_obsj*.pt"))
                            + list(exp_dir.glob("l96_datasets_obsj*.pt")))
        if candidates:
            dataset_path = str(candidates[0])
            logger.info(f"Auto-detected dataset: {dataset_path}")

    _, dataloaders, obs_var_indices = prepare_dataset(
        cfg, dataset_path, args.num_windows, args.obs_interval,
        obs_j=args.obs_j, is_joint=True, batch_size=args.batch_size,
    )
    logger.info(f"Dataset: {len(dataloaders['s0'].dataset)} windows, batch={args.batch_size}")
    logger.info(f"obs_var_indices ({len(obs_var_indices)} dims): {list(obs_var_indices)}")

    # Truth L96 dynamics (full J=4) used for the parameter-sensitivity forecast
    # metric: both the true-param and estimated-param rollouts use the SAME
    # dynamics so the divergence isolates the effect of parameter error.
    from models.lorenz96_dynamics import Lorenz96Dynamics
    truth_dyn = Lorenz96Dynamics(
        dt=0.001, coupling_exponent=1.6, NO=8, J=4,
        h=1.0, hx=1.0, eps=0.1, fast_weights=[1.0, 1.0, 0.1, 0.1],
    )

    torch.manual_seed(args.seed)
    n_outer = args.n_outer if args.n_outer is not None else getattr(model, "N_outer", 10)
    logger.info(f"Running joint inference (step 1): cases={args.cases} n_members={args.n_members} n_outer={n_outer}")
    estimates = run_inference(model, dataloaders, device, obs_var_indices,
                              n_members=args.n_members, n_outer=n_outer)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = {}
    estimates_paths = {}
    for case in args.cases:
        est = estimates[case]
        if args.n_members > 1:
            npz_path = output_path.parent / f"joint_estimates_{case}_ens{args.n_members}.npz"
            np.savez_compressed(npz_path,
                                members=est["members"],
                                truth=est["truth"],
                                params_pred=est["params_pred"],
                                params_true=est["params_true"],
                                x0=est["x0"],
                                forcing_true=est["forcing_true"])
            sm = evaluate_ensemble_estimates(est["members"], est["truth"])
        else:
            npz_path = output_path.parent / f"joint_estimates_{case}.npz"
            np.savez_compressed(npz_path,
                                trajectories=est["trajectories"],
                                truth=est["truth"],
                                params_pred=est["params_pred"],
                                params_true=est["params_true"],
                                x0=est["x0"],
                                forcing_true=est["forcing_true"])
            sm = evaluate_estimates(est["trajectories"], est["truth"])
        estimates_paths[case] = str(npz_path)
        prmse = param_rmse(est["params_pred"], est["params_true"])
        nrmse = nrmse_param(est["params_pred"], est["params_true"])
        traj_skill = trajectory_forecast_skill(
            truth_dyn,
            est["x0"], est["forcing_true"],
            est["params_true"], est["params_pred"],
            n_steps=args.n_compare_steps,
            obs_var_indices=obs_var_indices,
        )
        metrics[case] = {
            "rmse": sm["rmse"],
            "groups": sm["groups"],
            "ev": sm["ev"],
            "es": sm["es"],
            "param_rmse": {nm: float(prmse[i]) for i, nm in enumerate(L96_JOINT_PARAM_NAMES)},
            "param_rmse_mean": float(np.mean(prmse)),
            "nrmse_param": {
                nm: float(nrmse["per_param"][i]) for i, nm in enumerate(L96_JOINT_PARAM_NAMES)
            },
            "nrmse_param_mean": float(nrmse["mean"]),
            "traj_forecast": {
                "n_steps": traj_skill["n_steps"],
                "rmse": traj_skill["rmse"],
                "ev": traj_skill["ev"],
            },
        }
        logger.info(f"Saved: {npz_path}")

    metrics["degradation"] = (
        float(metrics["s1"]["rmse"] / metrics["s0"]["rmse"])
        if "s0" in metrics and "s1" in metrics and metrics["s0"]["rmse"] > 0
        else float("nan")
    )

    output = {
        "checkpoint": args.checkpoint,
        "model_type": type(model).__name__,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "param_names": list(L96_JOINT_PARAM_NAMES),
        "dataset": {
            "path": dataset_path,
            "num_windows": args.num_windows,
            "obs_interval": args.obs_interval,
        },
        "sampling": {"n_members": args.n_members, "n_outer": n_outer, "seed": args.seed, "cases": list(args.cases)},
        "estimates": estimates_paths,
        "metrics": metrics,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=float)

    logger.info("\n" + "=" * 70)
    logger.info(f"Results saved to: {output_path}")
    for case in args.cases:
        m = metrics[case]
        logger.info(f"[{case.upper()}] RMSE: {m['rmse']:.6f} | "
                    f"slow: {m['groups']['slow']:.6f} | obs_fast: {m['groups']['obs_fast']:.6f} | "
                    f"EV(all): {m['ev']['groups']['all_obs']:.6f} | ES(all): {m['es']['groups']['all_obs']:.6f} | "
                    f"paramRMSE: {m['param_rmse_mean']:.6f} | "
                    f"nrmse: {m['nrmse_param_mean']:.4f} | "
                    f"trajEV(all): {m['traj_forecast']['ev']['groups']['all_obs']:.4f}")
    logger.info(f"[DEGRADATION] S1/S0 RMSE: {metrics['degradation']:.6f}")
    logger.info("=" * 70)

    return metrics


if __name__ == "__main__":
    main()
