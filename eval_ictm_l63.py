#!/usr/bin/env python3
"""ICTM (Iterative Corrupted Trajectory Matching) evaluation for Lorenz-63.

Zhang et al., "Flow Priors for Linear Inverse Problems via Iterative
Corrupted Trajectory Matching" (NeurIPS 2024, ``reports/FlowPrior.pdf``) --
see ``evaluation/ictm_sampler.py`` for the algorithm itself and how it
relates to ``evaluation/sda_sampler.py``'s SDA sampler.

Unlike ``eval_sda_l63.py`` (which guides a purely unconditional/params-only
``models/sda.py`` prior that never sees ``obs`` as a network input), this
script applies ICTM's local-MAP refinement directly on top of an
already-obs-conditioned flow model's own Euler trajectory -- any
``vanilla_cfm``/``monai_vanilla_cfm`` config works (state_dim/sigma_prior/
forward(x, batch, tau) is all ``ictm_map_sample`` needs). This departs from
the paper's strict "unconditional prior + inference-time guidance" setting
(the data term and the "prior" term both end up conditioned on obs to some
degree here, since the model already reads obs), but is a well-defined
ablation: does K-step local-MAP refinement improve on the model's own
one-shot Euler sample?

ICTM is a deterministic MAP point estimate (no ensemble by construction,
unlike SDA), so this harness reports RMSE/R2 only; ``n_members>1`` stacks
independent MAP solves (different noise draws) purely as a spread sanity
check, not the primary metric.

No retraining: reuses whatever checkpoint ``train.py --config-name
models/<name>`` already wrote to ``experiments/l63/<name>/{s0,s1}/
checkpoints/stage1.pt``.

Usage:
    python eval_ictm_l63.py --config-name models/monai_vanilla_cfm_splus
    python eval_ictm_l63.py model.ictm.lam=5.0 model.ictm.K=2

ICTM hyperparameters (``model.ictm.*`` in config, all optional, no sweep
done yet -- first-pass defaults only): ``N_outer`` (50), ``K`` (1), ``lam``
(1.0), ``step_size`` (1e-2), ``n_members`` (1).
"""
import os
import sys
import json
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_float32_matmul_precision('medium')

from data.build import build_datasets
from train import model_factory, _make_eval_batch, EXP_DIR
from evaluation.metrics import rmse, energy_score
from evaluation.ictm_sampler import ictm_map_sample
from eval_sda_l63 import _rmse_entry

DEFAULT_SOURCE_MODEL = "monai_vanilla_cfm_splus"


def evaluate_ictm(model, dataset, device, N_outer, K, lam, step_size, n_members,
                  param_names, param_dim, norm_stats=None):
    rmse_list, es_list, ens_var_list = [], [], []
    all_sq_err, all_ref = [], []
    for i in range(len(dataset)):
        w = dataset[i]
        batch = _make_eval_batch(w, device, param_names=param_names, param_dim=param_dim,
                                 norm_stats=norm_stats)
        x1, _ = ictm_map_sample(model, batch, N_outer=N_outer, K=K, lam=lam,
                                step_size=step_size, n_members=n_members)
        if norm_stats is not None:
            from data.normalization import denormalize
            x1 = denormalize(x1, norm_stats)
        if n_members == 1:
            ensemble = x1.detach().cpu().numpy()  # (B=1,T,D) read as (M=1,T,D)
        else:
            ensemble = x1[0].permute(2, 0, 1).detach().cpu().numpy()  # (B=1,T,D,M) -> (M,T,D)
        truth = w["true_state"].numpy()
        pred = ensemble.mean(axis=0)
        rmse_list.append(rmse(pred, truth))
        es_list.append(energy_score(ensemble, truth))
        all_sq_err.append((pred - truth) ** 2)
        all_ref.append(truth)
        ens_var_list.append(ensemble.var(axis=0))

    all_rmse = np.stack(rmse_list, axis=0)
    mean_rmse, std_rmse = np.mean(all_rmse, axis=0), np.std(all_rmse, axis=0)
    pooled_mse = np.mean(np.concatenate(all_sq_err, axis=0), axis=0)
    pooled_var = np.maximum(np.var(np.concatenate(all_ref, axis=0), axis=0), 1e-12)
    r2 = 1.0 - pooled_mse / pooled_var
    all_es = np.stack(es_list, axis=0)
    crps_mean, crps_std = np.mean(all_es, axis=0), np.std(all_es, axis=0)
    ensemble_spread = np.sqrt(np.mean(np.concatenate(ens_var_list, axis=0), axis=0))
    return mean_rmse, std_rmse, r2, crps_mean, crps_std, ensemble_spread


def save_ictm_trajectories(model, dataset, device, N_outer, K, lam, step_size, n_members,
                           param_names, param_dim, save_path, norm_stats=None):
    trajs, truths, members_list = [], [], []
    for i in range(len(dataset)):
        w = dataset[i]
        batch = _make_eval_batch(w, device, param_names=param_names, param_dim=param_dim,
                                 norm_stats=norm_stats)
        x1, _ = ictm_map_sample(model, batch, N_outer=N_outer, K=K, lam=lam,
                                step_size=step_size, n_members=n_members)
        if norm_stats is not None:
            from data.normalization import denormalize
            x1 = denormalize(x1, norm_stats)
        if n_members == 1:
            stacked = x1.permute(1, 2, 0).detach().cpu().numpy()  # (T,D,M=1)
        else:
            stacked = x1[0].detach().cpu().numpy()  # (T,D,M)
        trajs.append(stacked.mean(axis=-1))
        truths.append(w["true_state"].numpy())
        members_list.append(stacked)
    np.savez_compressed(save_path, trajectories=np.stack(trajs, axis=0),
                        truths=np.stack(truths, axis=0), members=np.stack(members_list, axis=0))


@hydra.main(config_path="config", config_name=f"models/{DEFAULT_SOURCE_MODEL}", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from hydra.core.hydra_config import HydraConfig
    config_name = HydraConfig.get().job.config_name
    SOURCE_MODEL = config_name.replace("models/", "") if config_name.startswith("models/") else cfg.model.model_type
    EXPERIMENT_ID = f"ictm_{SOURCE_MODEL}"

    exp_dir = os.path.join(EXP_DIR, "l63", EXPERIMENT_ID)
    os.makedirs(exp_dir, exist_ok=True)
    combined_results_path = os.path.join(exp_dir, "results.json")

    dc = cfg.data
    ic = cfg.model.get("ictm", {}) or {}
    param_names = tuple(dc.get("param_names", ["sigma", "rho", "beta", "c1"]))
    param_dim = cfg.model.get("param_dim", 0)
    state_names = cfg.data.get("state_names", ["X", "Y", "Z"])
    N_outer = int(ic.get("N_outer", 50))
    K = int(ic.get("K", 1))
    lam = float(ic.get("lam", 1.0))
    step_size = float(ic.get("step_size", 1e-2))
    n_members = int(ic.get("n_members", 1))

    norm_stats = None
    if dc.get("normalize", False):
        from data.normalization import load_norm_stats
        norm_stats_path = dc.get("norm_stats_path", os.path.join(EXP_DIR, "l63_norm_stats.pt"))
        norm_stats = load_norm_stats(norm_stats_path)
        print(f"  data.normalize=True: loaded per-channel stats from {norm_stats_path}")

    combined = {}
    total_time = 0.0
    for case in ["s0", "s1"]:
        print(f"\n{'=' * 60}\n  Case: {case}  (source model: {SOURCE_MODEL})\n{'=' * 60}")
        t0 = time.time()

        datasets, test_keys, base_cfg, system, obs_var_indices = build_datasets(cfg, train_case=case)
        model = model_factory(cfg, device)
        ckpt_path = os.path.join(EXP_DIR, "l63", SOURCE_MODEL, case, "checkpoints", "stage1.pt")
        print(f"  Loading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()

        test_key = f"test_{case}"
        dataset = datasets[test_key]
        # ictm_map_sample runs its K-step inner refinement under
        # torch.enable_grad() internally (double backward for the Hutchinson
        # trace term), so wrapping the outer loop in no_grad here is safe --
        # same pattern eval_sda_l63.py uses for sda_guided_sample.
        with torch.no_grad():
            m, s, r2, crps_mean, crps_std, ens_spread = evaluate_ictm(
                model, dataset, device, N_outer, K, lam, step_size, n_members,
                param_names, param_dim, norm_stats=norm_stats)
        eval_elapsed = time.time() - t0
        print(f"  {case}: rmse={np.mean(m):.4f}  r2={np.mean(r2):.4f}  "
              f"crps={np.mean(crps_mean):.4f}  spread={np.mean(ens_spread):.4f}  "
              f"({eval_elapsed:.0f}s)")

        case_dir = os.path.join(exp_dir, case)
        os.makedirs(case_dir, exist_ok=True)
        with torch.no_grad():
            save_ictm_trajectories(model, dataset, device, N_outer, K, lam, step_size, n_members,
                                   param_names, param_dim,
                                   os.path.join(case_dir, f"trajectories_{case}.npz"),
                                   norm_stats=norm_stats)

        entry = _rmse_entry(state_names, m, s, r2, crps_mean, crps_std, ens_spread, eval_elapsed)
        with open(os.path.join(case_dir, "results.json"), "w") as f:
            json.dump({"experiment_id": f"{EXPERIMENT_ID}_{case}", "model_type": "ictm",
                      f"fm_{case}": entry, "eval_time_seconds": eval_elapsed}, f, indent=2)

        combined[case] = entry
        total_time += eval_elapsed

    combined["config"] = {
        "model_type": "ictm",
        "source_model": SOURCE_MODEL,
        "N_outer": N_outer,
        "K": K,
        "lam": lam,
        "step_size": step_size,
        "n_members": n_members,
    }
    combined["total_time_seconds"] = total_time
    with open(combined_results_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved combined results to {combined_results_path}")


if __name__ == "__main__":
    main()
