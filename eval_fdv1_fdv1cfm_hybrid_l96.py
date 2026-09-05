#!/usr/bin/env python3
"""FDV1-warm-started FDV1-CFM sampling on the cached L96 test dataset.

Combines two already-trained, frozen models with no retraining:
  - FDV1 (``models/fourdvarnet.py::FourDVarNetSolver``) provides a per-window
    deterministic mean estimate.
  - FDV1-CFM (``models/fourdvarnet.py::FourDVarNetPredictStateCFM``) is
    sampled via its own ``sample()``, but instead of starting the tau=0 noise
    from ``N(0, sigma_prior^2)``, it is recentered on FDV1's estimate:
    ``x_0 = FDV1_estimate + N(0, sigma_prior^2)`` (``sample()``'s
    ``mean_estimate`` param). Unlike the FDV1+SDA hybrid
    (``eval_sda_fdv1_hybrid_l96.py``), no Euler steps are skipped -- the full
    ``N_outer``-step trajectory still runs from tau=0, just anchored on a
    better starting point than pure noise.

This is a genuinely two-model script and cannot reuse the shared single-model
``evaluation/neural_inference.run_inference``/``_run_case_inference`` dispatch,
so it drives its own (small) inference loop, reusing everything else:
``load_model``/``prepare_dataset`` for loading, ``evaluation/estimate_metrics``
for scoring, the same two-step (estimates -> generic evaluator) design as
every other L96 eval script this session.
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
from evaluation.neural_inference import BatchDict, load_model, prepare_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _run_case(fdv1_model, fdv1cfm_model, dataloader, device, obs_var_indices, n_outer, n_members, tau0):
    pred_means, members_list, truths = [], [], []
    for batch in dataloader:
        batch = {k: v.to(device) if v is not None else v for k, v in batch.items()}
        batch_obj = BatchDict(batch)
        with torch.no_grad():
            mean_est = fdv1_model.sample(batch_obj)
            members = torch.stack(
                [fdv1cfm_model.sample(batch_obj, N_outer=n_outer, mean_estimate=mean_est, tau0=tau0)
                 for _ in range(n_members)],
                dim=-1,
            )
        pred = members.detach().float().cpu()
        if n_members == 1:
            pred_means.append(pred[..., 0])
        else:
            members_list.append(pred)
            pred_means.append(pred.mean(dim=-1))
        truths.append(batch["true_state"].detach().cpu())

    trajectories = torch.cat(pred_means, dim=0).numpy()
    truth = torch.cat(truths, dim=0)
    d_pred = trajectories.shape[-1]
    if truth.shape[-1] > d_pred:
        if obs_var_indices is not None and len(obs_var_indices) == d_pred:
            truth = truth[..., list(obs_var_indices)]
        else:
            truth = truth[..., :d_pred]
    truth = truth.numpy()

    out = {"trajectories": trajectories, "truth": truth}
    if n_members > 1:
        out["members"] = torch.cat(members_list, dim=0).numpy().astype(np.float32)
    return out


def main():
    parser = argparse.ArgumentParser(description="Run FDV1-warm-started FDV1-CFM sampling on L96 S0/S1")
    parser.add_argument("--fdv1-checkpoint", required=True, help="Path to FDV1 checkpoint .ckpt")
    parser.add_argument("--fdv1-config", help="Path to FDV1 config.yaml (optional)")
    parser.add_argument("--fdv1cfm-checkpoint", required=True, help="Path to FDV1-CFM checkpoint .ckpt")
    parser.add_argument("--fdv1cfm-config", help="Path to FDV1-CFM config.yaml (optional)")
    parser.add_argument("--dataset", help="Path to cached test dataset .pt (optional)")
    parser.add_argument("--num-windows", type=int, default=200, help="Number of test windows")
    parser.add_argument("--obs-interval", type=int, default=100, help="Observation interval")
    parser.add_argument("--obs-j", type=int, default=2, help="Fast vars observed per slow node (default: 2)")
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--n-members", type=int, default=1,
                        help="Number of stochastic members to sample (1 = single warm-started sample)")
    parser.add_argument("--n-outer", type=int, default=10, help="Euler integration steps for FDV1-CFM sampling")
    parser.add_argument("--tau0", type=float, default=0.5,
                        help="Warm-start point in [0,1); 0.0 = pure noise start (no warm start)")
    parser.add_argument("--seed", type=int, default=0, help="Torch seed before sampling")
    parser.add_argument("--cases", nargs="+", default=["s0", "s1"], choices=["s0", "s1"])
    parser.add_argument("--output", default="fdv1_fdv1cfm_hybrid_eval_results.json", help="Output JSON")
    args = parser.parse_args()

    device = torch.device(args.device)

    logger.info(f"Loading FDV1: {args.fdv1_checkpoint}")
    fdv1_model, _ = load_model(args.fdv1_checkpoint, args.fdv1_config, device=device)
    logger.info(f"Loading FDV1-CFM: {args.fdv1cfm_checkpoint}")
    fdv1cfm_model, fdv1cfm_cfg = load_model(args.fdv1cfm_checkpoint, args.fdv1cfm_config, device=device)
    logger.info(f"FDV1={type(fdv1_model).__name__} state_dim={fdv1_model.state_dim} | "
                f"FDV1-CFM={type(fdv1cfm_model).__name__} state_dim={fdv1cfm_model.state_dim}")

    dataset_path = args.dataset
    if not dataset_path:
        ckpt_dir = Path(args.fdv1cfm_checkpoint).parent
        exp_dir = ckpt_dir.parent
        candidates = sorted(list(ckpt_dir.glob("l96_datasets_obsj*.pt"))
                            + list(exp_dir.glob("l96_datasets_obsj*.pt")))
        if candidates:
            dataset_path = str(candidates[0])
            logger.info(f"Auto-detected dataset: {dataset_path}")

    dataset, dataloaders, obs_var_indices = prepare_dataset(
        fdv1cfm_cfg, dataset_path, args.num_windows, args.obs_interval, obs_j=args.obs_j,
    )
    logger.info(f"Dataset: {len(dataset)} windows, batch={args.batch_size}")

    torch.manual_seed(args.seed)
    logger.info(
        f"Running hybrid inference: cases={args.cases} n_members={args.n_members} "
        f"n_outer={args.n_outer} tau0={args.tau0} seed={args.seed}"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {}
    estimates_paths = {}
    for case in args.cases:
        est = _run_case(fdv1_model, fdv1cfm_model, dataloaders[case], device, obs_var_indices,
                        args.n_outer, args.n_members, args.tau0)
        npz_path = output_path.parent / f"estimates_{case}.npz"
        save_estimates(str(npz_path), est["trajectories"], est["truth"])
        estimates_paths[case] = str(npz_path)
        if "members" in est:
            members_path = output_path.parent / f"members_{case}.npz"
            np.savez_compressed(members_path, members=est["members"], truth=est["truth"])
            estimates_paths[f"{case}_members"] = str(members_path)
            metrics[case] = evaluate_ensemble_estimates(est["members"], est["truth"])
        else:
            metrics[case] = evaluate_estimates(est["trajectories"], est["truth"])
        logger.info(f"Saved estimates: {npz_path}")

    metrics["degradation"] = (
        float(metrics["s1"]["rmse"] / metrics["s0"]["rmse"])
        if "s0" in metrics and "s1" in metrics and metrics["s0"]["rmse"] > 0
        else float("nan")
    )

    output = {
        "fdv1_checkpoint": args.fdv1_checkpoint,
        "fdv1cfm_checkpoint": args.fdv1cfm_checkpoint,
        "dataset": {"path": dataset_path, "num_windows": args.num_windows, "obs_interval": args.obs_interval},
        "sampling": {
            "n_members": args.n_members, "n_outer": args.n_outer, "tau0": args.tau0,
            "seed": args.seed, "cases": list(args.cases),
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
