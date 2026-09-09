#!/usr/bin/env python3
"""Mean-estimate-warm-started SDA guided sampling on the cached L96 test dataset.

Generalizes ``eval_sda_directunet_hybrid_l96.py`` (from the
``4dvarnet-fm-l96-eval-config-persist`` worktree) to any deterministic
mean-estimate model, not just DirectUNet/MonaiDirectUNet -- in particular
``FourDVarNetSolver`` (FDV1, ``unet_backbone=monai``), which is called via
``.sample(batch_obj, N_outer=...)`` rather than a plain forward pass. Same
"SDEdit"-style warm-start mechanism as ``eval_sda_fdv1_hybrid_l96.py``
(``evaluation/sda_sampler.py``'s ``mean_estimate``/``tau0``): the guided-
sampling trajectory starts at ``interpolant.mix(noise, mean_est, tau0)``
instead of pure noise, i.e. SDA is used to sample the *anomaly* around the
mean model's point estimate rather than starting the whole state from
scratch. ``guided_obs_cost``/the Tweedie x_hat_1 machinery are unchanged;
only the trajectory's starting point differs.

Normalization: when both the mean-estimate model and the SDA prior share the
same per-channel z-score convention (``experiments/l96_norm_stats_obsj2.pt``),
pass ``--normalize-stats``: obs is normalized before both model calls, the
mean estimate is used as-is (already in the SDA prior's normalized space --
no denormalize/renormalize round-trip needed for the warm start itself), and
only the final guided sample is denormalized before scoring against raw
truth. ``--r-var`` is not rescaled for the same reason as ``eval_sda_l96.py``:
the guidance step normalizes its own gradient by its norm, which exactly
cancels any positive R_var scale factor.

Same two-step, scheme-agnostic design as every other L96 eval script this
session: no metrics computed here, just estimates + truth saved to .npz,
then scored generically by ``evaluation/estimate_metrics``.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from evaluation.estimate_metrics import (
    evaluate_ensemble_estimates,
    evaluate_estimates,
    save_estimates,
)
from evaluation.neural_inference import BatchDict, load_model, prepare_dataset
from evaluation.sda_sampler import sda_guided_sample
from models.fourdvarnet import FourDVarNetPredictStateCFM, FourDVarNetSolver

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _mean_estimate(mean_model, batch_obj, n_outer: int):
    """Dispatch the mean-estimate model call by type: FourDVarNetSolver/
    FourDVarNetPredictStateCFM need ``.sample(batch_obj, N_outer=...)`` (their
    own unrolled/multi-step convention); DirectUNet/MonaiDirectUNet and other
    single-forward-pass models are just called directly."""
    if isinstance(mean_model, (FourDVarNetSolver, FourDVarNetPredictStateCFM)):
        return mean_model.sample(batch_obj, N_outer=n_outer)
    return mean_model(batch_obj)


def _run_case(mean_model, sda_model, dataloader, mean_dataloader, device, obs_var_indices,
              r_var, n_outer, mean_n_outer, guidance_weight, n_members, tau0, norm_stats):
    pred_means, members_list, truths = [], [], []
    mean_iter = iter(mean_dataloader) if mean_dataloader is not None else None
    for batch in dataloader:
        batch = {k: v.to(device) if v is not None else v for k, v in batch.items()}
        batch_obj = BatchDict(batch)
        with torch.no_grad():
            if mean_iter is not None:
                # Mean model expects raw (un-normalized) obs -- a different
                # convention from the SDA prior's normalized space (e.g. FDV1
                # was trained without data.normalize=true) -- so it gets its
                # own separately-prepared, raw batch (same windows, same
                # order, shuffle=False on both loaders), and its raw-space
                # output is normalized before being handed to SDA as the warm
                # start (SDA's guidance/interpolant machinery only ever
                # operates in its own, normalized, training space).
                mean_batch = next(mean_iter)
                mean_batch = {k: v.to(device) if v is not None else v for k, v in mean_batch.items()}
                mean_est_raw = _mean_estimate(mean_model, BatchDict(mean_batch), mean_n_outer)
                from data.normalization import normalize
                mean_est = normalize(mean_est_raw, norm_stats)
            else:
                mean_est = _mean_estimate(mean_model, batch_obj, mean_n_outer)
        pred, _ = sda_guided_sample(
            sda_model, batch_obj, R_var=r_var, N_outer=n_outer,
            guidance_weight=guidance_weight, n_members=n_members,
            mean_estimate=mean_est, tau0=tau0,
        )
        pred = pred.detach().float().cpu()
        if n_members == 1:
            pred_means.append(pred)
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
    parser = argparse.ArgumentParser(description="Run mean-model-warm-started SDA guided sampling on L96 S0/S1")
    parser.add_argument("--mean-checkpoint", required=True, help="Path to the mean-estimate model checkpoint")
    parser.add_argument("--mean-config", required=True, help="Path to the mean-estimate model's config.yaml")
    parser.add_argument("--mean-n-outer", type=int, default=10,
                        help="N_outer for the mean model's own .sample() call (FourDVarNetSolver only; "
                             "ignored for single-forward-pass models like DirectUNet)")
    parser.add_argument("--sda-checkpoint", required=True, help="Path to SDA (SDA1/SDA2/SDA3) checkpoint .ckpt")
    parser.add_argument("--sda-config", required=True, help="Path to SDA config.yaml")
    parser.add_argument("--dataset", help="Path to cached test dataset .pt (optional)")
    parser.add_argument("--num-windows", type=int, default=200, help="Number of test windows")
    parser.add_argument("--obs-interval", type=int, default=100, help="Observation interval")
    parser.add_argument("--obs-j", type=int, default=2, help="Fast vars observed per slow node (default: 2)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--n-members", type=int, default=1,
                        help="Number of stochastic members to sample (1 = single guided sample)")
    parser.add_argument("--n-outer", type=int, default=10,
                        help="Total Euler schedule length N_outer (warm start runs fewer steps: N_outer - step0)")
    parser.add_argument("--tau0", type=float, default=0.3,
                        help="Warm-start point in [0,1); 0.0 = pure noise start (no warm start)")
    parser.add_argument("--r-var", type=float, default=0.5, help="Observation-noise variance for the guidance cost")
    parser.add_argument("--guidance-weight", type=float, default=2.0,
                        help="DPS-style normalized-gradient step size (see evaluation/sda_sampler.py). "
                             "Default 2.0/tau0=0.3 carries over the DirectUNet+SDA hybrids' sweep result "
                             "-- NOT yet re-swept for this FDV1 mean estimate specifically.")
    parser.add_argument("--seed", type=int, default=0, help="Torch seed before sampling")
    parser.add_argument("--cases", nargs="+", default=["s0", "s1"], choices=["s0", "s1"])
    parser.add_argument("--output", default="sda_mean_hybrid_eval_results.json", help="Output JSON")
    parser.add_argument("--normalize-stats", default=None,
                        help="Path to a per-channel norm stats .pt (mean/std) for the SDA prior's "
                             "space. When given, obs is normalized before the SDA guidance call; the "
                             "final guided sample is denormalized before scoring. Omitting this flag "
                             "is a true no-op (also implies --no-mean-normalized, since there is then "
                             "no normalized space for the mean model to mismatch).")
    parser.add_argument("--mean-normalized", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether the mean-estimate model was ALSO trained with data.normalize=true "
                             "(shares the SDA prior's normalized space, e.g. DirectUNet-M(monai,norm)). "
                             "Pass --no-mean-normalized when it was trained on raw obs instead (e.g. "
                             "FDV1(monai), which has no data.normalize field at all) -- the mean model "
                             "then gets its own separately-prepared raw batch, and its raw-space output "
                             "is normalized (via --normalize-stats) before being used as the SDA warm "
                             "start. Getting this wrong silently feeds the mean model obs in the wrong "
                             "scale, producing a garbage estimate that poisons the whole hybrid "
                             "(confirmed: RMSE 1.36/EV 0.35 with FDV1 fed normalized obs it was never "
                             "trained on, vs a sane result once fixed).")
    args = parser.parse_args()

    device = torch.device(args.device)

    logger.info(f"Loading mean-estimate model: {args.mean_checkpoint}")
    mean_model, _ = load_model(args.mean_checkpoint, args.mean_config, device=device)
    logger.info(f"Loading SDA: {args.sda_checkpoint}")
    sda_model, sda_cfg = load_model(args.sda_checkpoint, args.sda_config, device=device)
    logger.info(f"Mean model={type(mean_model).__name__} state_dim={mean_model.state_dim} | "
                f"SDA={type(sda_model).__name__} state_dim={sda_model.state_dim}")

    dataset_path = args.dataset
    if not dataset_path:
        ckpt_dir = Path(args.sda_checkpoint).parent
        exp_dir = ckpt_dir.parent
        candidates = sorted(list(ckpt_dir.glob("l96_datasets_obsj*.pt"))
                            + list(exp_dir.glob("l96_datasets_obsj*.pt")))
        if candidates:
            dataset_path = str(candidates[0])
            logger.info(f"Auto-detected dataset: {dataset_path}")

    norm_stats = None
    if args.normalize_stats:
        from data.normalization import load_norm_stats
        norm_stats = load_norm_stats(args.normalize_stats)
        logger.info(f"Loaded normalize-stats from {args.normalize_stats}: "
                    f"mean/std shape {tuple(norm_stats['mean'].shape)}")

    from models.sda import ConditionalPriorCFM
    is_conditioned = isinstance(sda_model, ConditionalPriorCFM)
    dataset, dataloaders, obs_var_indices = prepare_dataset(
        sda_cfg, dataset_path, args.num_windows, args.obs_interval,
        obs_j=args.obs_j, is_joint=is_conditioned, norm_stats=norm_stats,
        batch_size=args.batch_size,
    )
    logger.info(f"Dataset: {len(dataset)} windows, batch={args.batch_size}")

    mean_dataloaders = None
    if norm_stats is not None and not args.mean_normalized:
        logger.info("Mean model uses raw (un-normalized) obs -- preparing a separate raw dataloader")
        _, mean_dataloaders, _ = prepare_dataset(
            sda_cfg, dataset_path, args.num_windows, args.obs_interval,
            obs_j=args.obs_j, is_joint=is_conditioned, norm_stats=None,
            batch_size=args.batch_size,
        )

    torch.manual_seed(args.seed)
    logger.info(
        f"Running hybrid inference: cases={args.cases} n_members={args.n_members} "
        f"n_outer={args.n_outer} mean_n_outer={args.mean_n_outer} tau0={args.tau0} r_var={args.r_var} "
        f"guidance_weight={args.guidance_weight} seed={args.seed}"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {}
    estimates_paths = {}
    for case in args.cases:
        est = _run_case(mean_model, sda_model, dataloaders[case],
                        mean_dataloaders[case] if mean_dataloaders is not None else None,
                        device, obs_var_indices,
                        args.r_var, args.n_outer, args.mean_n_outer, args.guidance_weight,
                        args.n_members, args.tau0, norm_stats)
        if norm_stats is not None:
            from data.normalization import denormalize
            est["trajectories"] = denormalize(est["trajectories"], norm_stats)
            if "members" in est:
                m = np.swapaxes(est["members"], -1, -2)
                m = denormalize(m, norm_stats)
                est["members"] = np.swapaxes(m, -1, -2)
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
        "mean_checkpoint": args.mean_checkpoint,
        "sda_checkpoint": args.sda_checkpoint,
        "dataset": {"path": dataset_path, "num_windows": args.num_windows, "obs_interval": args.obs_interval},
        "sampling": {
            "n_members": args.n_members, "n_outer": args.n_outer, "mean_n_outer": args.mean_n_outer,
            "tau0": args.tau0, "r_var": args.r_var, "guidance_weight": args.guidance_weight,
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
