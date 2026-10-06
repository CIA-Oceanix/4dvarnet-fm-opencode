#!/usr/bin/env python3
"""
Score-based data assimilation for Lorenz-63, using the master-branch SDA
stack (models/sda.py + evaluation/sda_sampler.py) instead of a bespoke
L63-only sampler.

Follows Rozet & Louppe, "Score-based Data Assimilation" (NeurIPS 2023), as
formalized in Sec. 3.4 / Eq. (12)-(14) of docs/preprint_4dvarnet_fm_2025.pdf:
a purely unconditional flow-matching prior p(x_1) -- models.sda.
UnconditionalPriorCFM, which never reads obs/forcing/params, trained via
`python train.py --config-name models/sda_prior` exactly like any other
stage-1-only CFM -- is turned into a state estimator p(x_1|y) at *inference*
time only, by nudging its Euler integration toward the observations at every
step (DPS/Pi-GDM-style normalized-gradient guidance on an L2 observation
cost; see evaluation/sda_sampler.py::sda_guided_sample for the derivation).

This is the same model class and sampler the L96 SDA benchmark uses
(config/experiment/SDA1_prior_l96.yaml, eval_sda_l96.py) -- only the
dataset/harness plumbing here is L63-specific (data.build.build_datasets,
train.py's model_factory/_make_eval_batch, and the experiments/l63/<model>/
results.json schema every other L63 model writes), since master has no L63
pipeline of its own.

No retraining: reuses the checkpoint already written by
`python train.py --config-name models/sda_prior` under
experiments/l63/sda_prior/{s0,s1}/checkpoints/stage1.pt.

Usage:
    python train.py --config-name models/sda_prior   # train the prior once
    python eval_sda_l63.py                            # then run guided DA
    python eval_sda_l63.py model.N_ensemble=20 model.sda_prior.guidance_weight=0.5

Any prior config (--config-name models/monai_sda_prior, monai_sda_prior_cond,
monai_sda_prior_cond_noisy) is supported. Hybrid mode -- the L96
FDV1+SDA / DirectUNet+SDA "SDEdit" warm start (evaluation/sda_sampler.py's
mean_estimate/tau0) -- warm-starts the guided trajectory from a frozen mean
estimator already trained via train.py:
    python eval_sda_l63.py --config-name models/monai_sda_prior_cond_noisy \
        +hybrid.mean_model=monai_fdv_subgrad +hybrid.tau0=0.3
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
from evaluation.sda_sampler import sda_guided_sample

DEFAULT_SOURCE_MODEL = "sda_prior"


def _experiment_id(source_model: str, mean_model: str = None) -> str:
    """Keeps the established "score_based_cfm" experiment dir (and its
    report row) for the plain UNet1D prior; any other prior config name
    (e.g. "monai_sda_prior") gets its own dir so it can't silently clobber
    that one's results.json/checkpoints. A warm-start mean model
    (hybrid mode) is prefixed so the hybrid never overwrites the plain
    guided-prior results either."""
    base = "score_based_cfm" if source_model == DEFAULT_SOURCE_MODEL else f"score_based_cfm_{source_model}"
    return base if mean_model is None else f"hybrid_{mean_model}+{source_model}"


def _prior_block_name(model_type: str) -> str:
    return "monai_sda_prior" if model_type.startswith("monai_sda_prior") else "sda_prior"


def load_mean_model(mean_model: str, case: str, device):
    """Loads a frozen deterministic mean estimator (e.g. monai_fdv_subgrad)
    from experiments/l63/<mean_model>/<case>/ using the resolved_config.yaml
    train.py persisted next to its checkpoint, so its architecture never
    has to be re-specified here."""
    case_dir = os.path.join(EXP_DIR, "l63", mean_model, case)
    mean_cfg = OmegaConf.load(os.path.join(case_dir, "resolved_config.yaml"))
    model = model_factory(mean_cfg, device)
    model.load_state_dict(torch.load(os.path.join(case_dir, "checkpoints", "stage1.pt"), map_location=device))
    model.to(device)
    model.eval()
    model.mean_param_dim = int(mean_cfg.model.get("param_dim", 0))
    return model


def _guided_members(model, batch, n_ensemble, n_outer, r_var, guidance_weight,
                    mean_model=None, tau0=0.0, mean_batch=None, mean_norm_stats=None):
    mean_estimate = None
    if mean_model is not None:
        with torch.no_grad():
            mean_estimate = mean_model.sample(mean_batch) if hasattr(mean_model, "sample") else mean_model(mean_batch)
            if mean_norm_stats is not None:
                # mean_model was trained under data.normalize: true, so its
                # own state-space output is z-scored -- denormalize back to
                # the SDA prior's native raw physical units (no monai_sda_prior*
                # config trains normalized) before it anchors the warm-started
                # trajectory; interpolant.mix/sda_guided_sample downstream have
                # no other way to know the two models disagree on scale.
                from data.normalization import denormalize
                mean_estimate = denormalize(mean_estimate, mean_norm_stats)
    members, _ = sda_guided_sample(
        model, batch, R_var=r_var, N_outer=n_outer,
        guidance_weight=guidance_weight, n_members=n_ensemble,
        mean_estimate=mean_estimate, tau0=tau0,
    )
    return members


def evaluate_sda_guided(model, dataset, device, n_ensemble, n_outer, r_var,
                        guidance_weight, param_names, param_dim,
                        mean_model=None, tau0=0.0, norm_stats=None):
    """Mirrors `train.evaluate_model`, but draws samples via the
    observation-guided sampler (`sda_guided_sample`) instead of the model's
    own unconditional `model.sample(batch)`.

    The SDA prior (`model`) always operates in raw physical units -- none of
    the `monai_sda_prior*` configs train with `data.normalize: true` -- so
    its own `batch`/guided-sampling output must stay raw throughout, never
    normalized or denormalized (`guided_obs_cost` compares `x_hat_1`, in the
    prior's raw scale, against `batch.obs`, which must match). `norm_stats`,
    when given, is used ONLY for the optional hybrid mean model: it was
    trained under `data.normalize: true`, so its own input obs is normalized
    and its raw-state output is denormalized (see `_guided_members`) back to
    the prior's native scale before anchoring the warm-started trajectory."""
    rmse_list, es_list, ens_var_list = [], [], []
    all_sq_err, all_ref = [], []
    for i in range(len(dataset)):
        w = dataset[i]
        batch = _make_eval_batch(w, device, param_names=param_names, param_dim=param_dim)
        mean_batch = None if mean_model is None else _make_eval_batch(
            w, device, param_names=param_names, param_dim=mean_model.mean_param_dim,
            norm_stats=norm_stats)
        members = _guided_members(model, batch, n_ensemble, n_outer, r_var,
                                  guidance_weight, mean_model, tau0, mean_batch,
                                  mean_norm_stats=norm_stats)
        # sda_guided_sample returns (B,T,D) when n_ensemble==1, (B,T,D,M)
        # otherwise (its own documented contract) -- pre-existing bug, only
        # ever exercised at n_ensemble==1 (real runs always use N_ensemble
        # > 1), fixed here rather than worked around.
        if n_ensemble == 1:
            members_mtd = members  # (B=1,T,D), already (M=1,T,D)
        else:
            members_mtd = members[0].permute(2, 0, 1)  # (B=1,T,D,M) -> (M,T,D)
        ensemble = members_mtd.detach().cpu().numpy()
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


def save_sda_trajectories(model, dataset, device, n_ensemble, n_outer, r_var,
                          guidance_weight, param_names, param_dim, save_path,
                          mean_model=None, tau0=0.0, norm_stats=None):
    trajs, truths, members_list = [], [], []
    for i in range(len(dataset)):
        w = dataset[i]
        batch = _make_eval_batch(w, device, param_names=param_names, param_dim=param_dim)
        mean_batch = None if mean_model is None else _make_eval_batch(
            w, device, param_names=param_names, param_dim=mean_model.mean_param_dim,
            norm_stats=norm_stats)
        members = _guided_members(model, batch, n_ensemble, n_outer, r_var,
                                  guidance_weight, mean_model, tau0, mean_batch,
                                  mean_norm_stats=norm_stats)
        if n_ensemble == 1:
            members_mtd = members  # (B=1,T,D), already (M=1,T,D)
        else:
            members_mtd = members[0].permute(2, 0, 1)  # (B=1,T,D,M) -> (M,T,D)
        ensemble = members_mtd.detach().cpu().numpy()
        stacked = np.transpose(ensemble, (1, 2, 0))  # (T,D,M), matches save_trajectories elsewhere
        trajs.append(stacked.mean(axis=-1))
        truths.append(w["true_state"].numpy())
        members_list.append(stacked)
    np.savez_compressed(save_path, trajectories=np.stack(trajs, axis=0),
                        truths=np.stack(truths, axis=0), members=np.stack(members_list, axis=0))


def _rmse_entry(state_names, m, s, r2, crps_mean, crps_std, ens_spread, elapsed_seconds):
    d = {}
    for i, nm in enumerate(state_names):
        d[nm] = {
            "rmse": {"mean": float(m[i]), "std": float(s[i])},
            "r2": float(r2[i]),
            "crps": {"mean": float(crps_mean[i]), "std": float(crps_std[i])},
            "ensemble_spread": float(ens_spread[i]),
        }
    d["rmse"] = {"mean": float(np.mean(m)), "std": float(np.mean(s))}
    d["r2"] = float(np.mean(r2))
    d["crps"] = {"mean": float(np.mean(crps_mean)), "std": float(np.mean(crps_std))}
    d["ensemble_spread"] = float(np.mean(ens_spread))
    d["elapsed_seconds"] = float(elapsed_seconds)
    return d


@hydra.main(config_path="config", config_name=f"models/{DEFAULT_SOURCE_MODEL}", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Derived from the actually-composed config, not the CLI-default
    # constant above -- so `--config-name models/monai_sda_prior` (or any
    # future prior model_type) picks up its own sub-block/checkpoint dir/
    # results dir instead of silently reading "sda_prior"'s.
    from hydra.core.hydra_config import HydraConfig
    config_name = HydraConfig.get().job.config_name
    SOURCE_MODEL = config_name.replace("models/", "") if config_name.startswith("models/") else cfg.model.model_type
    prior_block = _prior_block_name(cfg.model.model_type)
    hybrid = cfg.get("hybrid", None) or {}
    MEAN_MODEL = hybrid.get("mean_model", None)
    TAU0 = float(hybrid.get("tau0", 0.3)) if MEAN_MODEL else 0.0
    EXPERIMENT_ID = _experiment_id(SOURCE_MODEL, MEAN_MODEL)

    exp_dir = os.path.join(EXP_DIR, "l63", EXPERIMENT_ID)
    os.makedirs(exp_dir, exist_ok=True)
    combined_results_path = os.path.join(exp_dir, "results.json")

    dc = cfg.data
    sp = cfg.model[prior_block]
    param_names = tuple(dc.get("param_names", ["sigma", "rho", "beta", "c1"]))
    param_dim = cfg.model.get("param_dim", 0)
    state_names = cfg.data.get("state_names", ["X", "Y", "Z"])
    N_ensemble = cfg.model.get("N_ensemble", 50)
    N_outer = sp.N_outer
    guidance_weight = sp.get("guidance_weight", 1.0)
    R_var = dc.R_var

    norm_stats = None
    if dc.get("normalize", False):
        from data.normalization import load_norm_stats
        norm_stats_path = dc.get("norm_stats_path", os.path.join(EXP_DIR, "l63_norm_stats.pt"))
        norm_stats = load_norm_stats(norm_stats_path)
        print(f"  data.normalize=True: loaded per-channel stats from {norm_stats_path}")

    combined = {}
    total_time = 0.0
    for case in ["s0", "s1"]:
        print(f"\n{'=' * 60}\n  Case: {case}  (source prior: {SOURCE_MODEL})\n{'=' * 60}")
        t0 = time.time()

        datasets, test_keys, base_cfg, system, obs_var_indices = build_datasets(cfg, train_case=case)
        model = model_factory(cfg, device)
        ckpt_path = os.path.join(EXP_DIR, "l63", SOURCE_MODEL, case, "checkpoints", "stage1.pt")
        print(f"  Loading unconditional prior checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()
        mean_model = load_mean_model(MEAN_MODEL, case, device) if MEAN_MODEL else None

        test_key = f"test_{case}"
        dataset = datasets[test_key]
        # sda_guided_sample runs its guided steps under torch.enable_grad()
        # internally (it needs autograd for the observation-cost gradient),
        # so wrapping the outer loop in no_grad here is safe -- same pattern
        # evaluation/neural_inference.py::_run_case_inference uses for L96.
        with torch.no_grad():
            m, s, r2, crps_mean, crps_std, ens_spread = evaluate_sda_guided(
                model, dataset, device, N_ensemble, N_outer, R_var, guidance_weight,
                param_names, param_dim, mean_model=mean_model, tau0=TAU0,
                norm_stats=norm_stats)
        eval_elapsed = time.time() - t0
        print(f"  {case}: rmse={np.mean(m):.4f}  r2={np.mean(r2):.4f}  "
              f"crps={np.mean(crps_mean):.4f}  spread={np.mean(ens_spread):.4f}  "
              f"({eval_elapsed:.0f}s)")

        case_dir = os.path.join(exp_dir, case)
        os.makedirs(case_dir, exist_ok=True)
        with torch.no_grad():
            save_sda_trajectories(model, dataset, device, N_ensemble, N_outer, R_var,
                                  guidance_weight, param_names, param_dim,
                                  os.path.join(case_dir, f"trajectories_{case}.npz"),
                                  mean_model=mean_model, tau0=TAU0, norm_stats=norm_stats)

        entry = _rmse_entry(state_names, m, s, r2, crps_mean, crps_std, ens_spread, eval_elapsed)
        with open(os.path.join(case_dir, "results.json"), "w") as f:
            json.dump({"experiment_id": f"{EXPERIMENT_ID}_{case}", "model_type": "score_based_cfm",
                      f"fm_{case}": entry, "eval_time_seconds": eval_elapsed}, f, indent=2)

        combined[case] = entry
        total_time += eval_elapsed

    combined["config"] = {
        "hidden_channels": list(sp.hidden_channels),
        "N_outer": N_outer,
        "N_ensemble": N_ensemble,
        "model_type": "score_based_cfm",
        "prior_model": SOURCE_MODEL,
        "R_var": R_var,
        "sigma_prior": sp.sigma_prior,
        "guidance_weight": guidance_weight,
        "mean_model": MEAN_MODEL,
        "tau0": TAU0,
    }
    combined["total_time_seconds"] = total_time
    with open(combined_results_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved combined results to {combined_results_path}")


if __name__ == "__main__":
    main()
