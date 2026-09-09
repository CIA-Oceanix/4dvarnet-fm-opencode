#!/usr/bin/env python3
"""Fast-Y observation-density generalization study for the 4 best-of-subcategory
L96 monai-backbone schemes: DirectUNet-L(cos), CFM-M(flat), SDA3, DirectUNet+SDA3.

No retraining: each checkpoint is evaluated on the same cached S0/S1 test set
it was benchmarked on (``l96_consolidated_benchmark.md``), but with the
fast-Y observation density randomly reduced at inference time -- of the 16
canonical fast-Y channels (2 per slow node), only ``keep_k`` are kept,
**redrawn independently at every observation time** within each window (the
harder OOD test vs. a fixed-per-window mask). The 8 slow-X channels always
stay fully observed. ``keep_k=16`` is the full-density sanity check and must
reproduce each scheme's canonical benchmark numbers exactly (see
``evaluation/obs_density.py`` for the masking mechanics and the important
caveat: DirectUNet/CFM never see a mask channel, only ``obs`` itself -- a
dropped channel is indistinguishable from a genuine near-zero observation for
them, unlike SDA's guidance cost which excludes it cleanly).

Each (method, case, keep_k) cell is run ``--n-repeats`` times with a fresh
seed (the mask redraw -- and for stochastic samplers, the sampling noise --
differ per repeat); the reported RMSE/EV are the mean +/- std across those
repeats, not across windows (see ``evaluation/estimate_metrics.py``'s
``per_window_rmse_ev`` for that finer-grained decomposition, not used here).

Same two-step, scheme-agnostic design as the other L96 eval scripts: estimates
+ truth are saved to per-(method, case, keep_k, repeat) ``.npz`` files, scored
here inline via ``evaluate_estimates`` (the generic evaluator), and the full
sweep is written to a single results JSON for
``reports/l96/generate_l96_obs_density_generalization_report.py`` to render
(distinct from the pre-existing ``generate_l96_obs_density_report.py``, which
covers the unrelated DA-baseline slow-only-vs-obsj2 study).
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from eval_monai_l96 import load_monai_direct_unet
from evaluation.estimate_metrics import evaluate_estimates, save_estimates
from evaluation.neural_inference import (
    BatchDict,
    load_model,
    prepare_dataset,
    run_inference,
)
from evaluation.obs_density import (
    NUM_FAST,
    apply_density_mask_to_obs,
    fast_channel_keep_mask,
)
from evaluation.sda_sampler import sda_guided_sample
from models.sda import ConditionalPriorCFM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_KEEP_K = (16, 8, 4, 0)


def _mean_std(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": len(values)}


def _hybrid_estimates(mean_model, sda_model, dataloader, device, obs_var_indices,
                       r_var, n_outer, guidance_weight, n_members, tau0, keep_k):
    """DirectUNet-warm-started SDA3 guided sampling, one (case, keep_k, seed) cell.

    Mirrors ``eval_sda_directunet_hybrid_l96.py::_run_case``, with the density
    mask applied to (a) the mean model's obs input directly (it consumes obs
    the same ambiguous zero-imputation way as plain DirectUNet) and (b) the
    SDA guidance cost via ``obs_channel_mask`` (clean, no imputation).
    """
    pred_means, truths = [], []
    for batch in dataloader:
        batch = {k: v.to(device) if v is not None else v for k, v in batch.items()}
        obs_channel_mask = None
        mean_obs = batch["obs"]
        if keep_k < NUM_FAST:
            Bb, Tb, _ = batch["obs"].shape
            obs_channel_mask = fast_channel_keep_mask(Bb, Tb, keep_k, device=device)
            mean_obs = apply_density_mask_to_obs(batch["obs"], obs_channel_mask)

        mean_batch_obj = BatchDict({**batch, "obs": mean_obs})
        with torch.no_grad():
            mean_est = mean_model(mean_batch_obj)

        sda_batch_obj = BatchDict(batch)  # unmasked: obs_channel_mask restricts the cost instead
        pred, _ = sda_guided_sample(
            sda_model, sda_batch_obj, R_var=r_var, N_outer=n_outer,
            guidance_weight=guidance_weight, n_members=n_members,
            mean_estimate=mean_est, tau0=tau0, obs_channel_mask=obs_channel_mask,
        )
        pred = pred.detach().float().cpu()
        pred_means.append(pred.mean(dim=-1) if n_members > 1 else pred)
        truths.append(batch["true_state"].detach().cpu())

    trajectories = torch.cat(pred_means, dim=0).numpy()
    truth = torch.cat(truths, dim=0)
    d_pred = trajectories.shape[-1]
    if truth.shape[-1] > d_pred:
        if obs_var_indices is not None and len(obs_var_indices) == d_pred:
            truth = truth[..., list(obs_var_indices)]
        else:
            truth = truth[..., :d_pred]
    return trajectories, truth.numpy()


def _run_sweep(name, run_one_fn, cases, keep_k_values, n_repeats, seed_base, output_dir):
    """Shared (case, keep_k, repeat) loop: calls ``run_one_fn(case, keep_k, seed)
    -> (trajectories, truth)``, scores each repeat, and aggregates mean+/-std.
    """
    results = {}
    for case in cases:
        results[case] = {}
        for keep_k in keep_k_values:
            per_repeat = {"rmse": [], "ev_all_obs": [], "ev_slow": [], "ev_obs_fast": []}
            last_est = None
            for r in range(n_repeats):
                seed = seed_base + r
                torch.manual_seed(seed)
                trajectories, truth = run_one_fn(case, keep_k, seed)
                m = evaluate_estimates(trajectories, truth)
                per_repeat["rmse"].append(m["rmse"])
                per_repeat["ev_all_obs"].append(m["ev"]["groups"]["all_obs"])
                per_repeat["ev_slow"].append(m["ev"]["groups"]["slow"])
                per_repeat["ev_obs_fast"].append(m["ev"]["groups"]["obs_fast"])
                last_est = (trajectories, truth)
                logger.info(f"[{name}][{case}][keep_k={keep_k}][seed={seed}] "
                            f"RMSE={m['rmse']:.4f} EV={m['ev']['groups']['all_obs']:.4f}")
            if output_dir is not None and last_est is not None:
                npz_path = output_dir / f"estimates_{name}_{case}_keep{keep_k}.npz"
                save_estimates(str(npz_path), last_est[0], last_est[1])
            results[case][str(keep_k)] = {
                "rmse": _mean_std(per_repeat["rmse"]),
                "ev": {
                    "all_obs": _mean_std(per_repeat["ev_all_obs"]),
                    "slow": _mean_std(per_repeat["ev_slow"]),
                    "obs_fast": _mean_std(per_repeat["ev_obs_fast"]),
                },
            }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fast-Y observation-density generalization sweep for the 4 "
                     "best-of-subcategory L96 monai schemes")
    parser.add_argument("--directunet-checkpoint",
                        default="experiments/L1b_monai_unet_s0s1_norm_l_cosine/checkpoints/stage1.pt")
    parser.add_argument("--directunet-config",
                        default="experiments/L1b_monai_unet_s0s1_norm_l_cosine/resolved_config.yaml")
    parser.add_argument("--cfm-checkpoint",
                        default="experiments/L2b_monai_vanilla_cfm_s0s1_norm/checkpoints/stage1_best.ckpt")
    parser.add_argument("--cfm-config",
                        default="experiments/L2b_monai_vanilla_cfm_s0s1_norm/resolved_config.yaml")
    parser.add_argument("--sda-checkpoint",
                        default="experiments/SDA3_monai_cond_noisy_l96_norm/checkpoints/stage1_best.ckpt")
    parser.add_argument("--sda-config",
                        default="experiments/SDA3_monai_cond_noisy_l96_norm/resolved_config.yaml")
    parser.add_argument("--hybrid-mean-checkpoint",
                        default="experiments/L1b_monai_unet_s0s1_norm/checkpoints/stage1_best.ckpt")
    parser.add_argument("--hybrid-mean-config",
                        default="experiments/L1b_monai_unet_s0s1_norm/resolved_config.yaml")
    parser.add_argument("--dataset", default="experiments/l96_datasets_obsj2_int100_nwin200.pt")
    parser.add_argument("--normalize-stats", default="experiments/l96_norm_stats_obsj2.pt")
    parser.add_argument("--num-windows", type=int, default=200)
    parser.add_argument("--obs-interval", type=int, default=100)
    parser.add_argument("--obs-j", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--keep-k", type=int, nargs="+", default=list(DEFAULT_KEEP_K),
                        help="Fast-Y channels to keep per obs time out of 16 (default: 16 8 4 0)")
    parser.add_argument("--n-repeats", type=int, default=3,
                        help="Independent seed reruns per (method, case, keep_k) cell")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; repeat r uses seed+r")
    parser.add_argument("--n-outer", type=int, default=10, help="Euler steps for CFM/SDA/hybrid sampling")
    parser.add_argument("--n-members", type=int, default=30, help="Ensemble members for CFM/SDA/hybrid")
    parser.add_argument("--r-var", type=float, default=0.5)
    parser.add_argument("--sda-guidance-weight", type=float, default=40.0)
    parser.add_argument("--hybrid-guidance-weight", type=float, default=2.0)
    parser.add_argument("--hybrid-tau0", type=float, default=0.3)
    parser.add_argument("--cases", nargs="+", default=["s0", "s1"], choices=["s0", "s1"])
    parser.add_argument("--methods", nargs="+",
                        default=["directunet", "cfm", "sda3", "directunet_sda3"],
                        choices=["directunet", "cfm", "sda3", "directunet_sda3"])
    parser.add_argument("--output", default="experiments/l96_obs_density_generalization/results.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from data.normalization import denormalize, load_norm_stats
    norm_stats = load_norm_stats(args.normalize_stats) if args.normalize_stats else None
    if norm_stats is not None:
        logger.info(f"Loaded normalize-stats from {args.normalize_stats}")

    all_results = {}
    sda_model, sda_cfg = None, None

    if "directunet" in args.methods:
        logger.info(f"Loading DirectUNet-L(cos): {args.directunet_checkpoint}")
        model, cfg = load_monai_direct_unet(args.directunet_checkpoint, args.directunet_config, device)
        # Mirrors eval_monai_l96.py::main()'s patch: the experiment yaml may
        # lack data.system_config (prepare_dataset needs NO/J/obs_j to resolve
        # obs_var_indices when no explicit --dataset cache short-circuits it).
        if "system_config" not in cfg.data:
            OmegaConf.set_struct(cfg, False)
            cfg.data.system_config = {"NO": cfg.data.get("NO", 8), "J": cfg.data.get("J", 4)}
            cfg.data.obs_j = cfg.data.get("obs_j", args.obs_j)
            OmegaConf.set_struct(cfg, True)
        _, dataloaders, obs_var_indices = prepare_dataset(
            cfg, args.dataset, args.num_windows, args.obs_interval, obs_j=args.obs_j,
            norm_stats=norm_stats, batch_size=args.batch_size,
        )

        def _run_directunet(case, keep_k, seed):
            est = run_inference(model, {case: dataloaders[case]}, device, obs_var_indices,
                                obs_density_keep_k=keep_k)[case]
            traj, truth = est["trajectories"], est["truth"]
            if norm_stats is not None:
                traj = denormalize(traj, norm_stats)
            return traj, truth

        all_results["directunet"] = _run_sweep(
            "directunet", _run_directunet, args.cases, args.keep_k, args.n_repeats, args.seed,
            output_path.parent,
        )

    if "cfm" in args.methods:
        logger.info(f"Loading CFM-M(flat): {args.cfm_checkpoint}")
        model, cfg = load_model(args.cfm_checkpoint, args.cfm_config, device=device)
        _, dataloaders, obs_var_indices = prepare_dataset(
            cfg, args.dataset, args.num_windows, args.obs_interval, obs_j=args.obs_j,
            norm_stats=norm_stats, batch_size=args.batch_size,
        )

        def _run_cfm(case, keep_k, seed):
            est = run_inference(model, {case: dataloaders[case]}, device, obs_var_indices,
                                n_members=args.n_members, n_outer=1,
                                obs_density_keep_k=keep_k)[case]
            traj, truth = est["trajectories"], est["truth"]
            if norm_stats is not None:
                traj = denormalize(traj, norm_stats)
            return traj, truth

        all_results["cfm"] = _run_sweep(
            "cfm", _run_cfm, args.cases, args.keep_k, args.n_repeats, args.seed, output_path.parent,
        )

    if "sda3" in args.methods:
        logger.info(f"Loading SDA3: {args.sda_checkpoint}")
        sda_model, sda_cfg = load_model(args.sda_checkpoint, args.sda_config, device=device)
        is_conditioned = isinstance(sda_model, ConditionalPriorCFM)
        _, dataloaders, obs_var_indices = prepare_dataset(
            sda_cfg, args.dataset, args.num_windows, args.obs_interval, obs_j=args.obs_j,
            is_joint=is_conditioned, norm_stats=norm_stats, batch_size=args.batch_size,
        )

        def _run_sda3(case, keep_k, seed):
            est = run_inference(sda_model, {case: dataloaders[case]}, device, obs_var_indices,
                                n_members=args.n_members, n_outer=args.n_outer,
                                r_var=args.r_var, guidance_weight=args.sda_guidance_weight,
                                obs_density_keep_k=keep_k)[case]
            traj, truth = est["trajectories"], est["truth"]
            if norm_stats is not None:
                traj = denormalize(traj, norm_stats)
            return traj, truth

        all_results["sda3"] = _run_sweep(
            "sda3", _run_sda3, args.cases, args.keep_k, args.n_repeats, args.seed, output_path.parent,
        )

    if "directunet_sda3" in args.methods:
        logger.info(f"Loading DirectUNet+SDA3 hybrid mean: {args.hybrid_mean_checkpoint}")
        mean_model, _ = load_model(args.hybrid_mean_checkpoint, args.hybrid_mean_config, device=device)
        if sda_model is None:
            # Not already loaded by the "sda3" branch above (e.g. --methods
            # directunet_sda3 alone) -- load the same SDA3 checkpoint here.
            logger.info(f"Loading DirectUNet+SDA3 hybrid SDA3: {args.sda_checkpoint}")
            sda_model, sda_cfg = load_model(args.sda_checkpoint, args.sda_config, device=device)
        is_conditioned = isinstance(sda_model, ConditionalPriorCFM)
        _, dataloaders, obs_var_indices = prepare_dataset(
            sda_cfg, args.dataset, args.num_windows, args.obs_interval, obs_j=args.obs_j,
            is_joint=is_conditioned, norm_stats=norm_stats, batch_size=args.batch_size,
        )

        def _run_hybrid(case, keep_k, seed):
            traj, truth = _hybrid_estimates(
                mean_model, sda_model, dataloaders[case], device, obs_var_indices,
                args.r_var, args.n_outer, args.hybrid_guidance_weight, args.n_members,
                args.hybrid_tau0, keep_k,
            )
            if norm_stats is not None:
                traj = denormalize(traj, norm_stats)
            return traj, truth

        all_results["directunet_sda3"] = _run_sweep(
            "directunet_sda3", _run_hybrid, args.cases, args.keep_k, args.n_repeats, args.seed,
            output_path.parent,
        )

    output = {
        "keep_k_values": list(args.keep_k),
        "n_repeats": args.n_repeats,
        "seed_base": args.seed,
        "num_fast_channels": NUM_FAST,
        "cases": list(args.cases),
        "methods": list(args.methods),
        "sampling": {
            "n_outer": args.n_outer, "n_members": args.n_members, "r_var": args.r_var,
            "sda_guidance_weight": args.sda_guidance_weight,
            "hybrid_guidance_weight": args.hybrid_guidance_weight,
            "hybrid_tau0": args.hybrid_tau0,
        },
        "results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    logger.info(f"Results saved to: {output_path}")
    return output


if __name__ == "__main__":
    main()
