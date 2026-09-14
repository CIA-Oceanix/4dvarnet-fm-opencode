#!/usr/bin/env python3
"""
Score-based data assimilation using the unconditional VanillaCFM as prior.

Implements Section 3.4 ("Revisiting score-based DA") of the 4DVarNet-FM
preprint (docs/preprint_4dvarnet_fm_2025.pdf), following Rozet & Louppe
(2023): given a pre-trained model of E[x1|x_tau] that never sees the
observations y (an "unconditional" flow-matching prior on state, still
allowed to condition on forcing/params like the other CFM baselines --
here `config/models/vanilla_cfm_beta0.5_unconditional.yaml`, use_obs=False),
we correct that prior mean at every Euler step with a linear-Gaussian
Bayes update against y (Eq. 13-14), then advance the flow with the
corrected conditional expectation via the existing conditional-expectation
ODE (Eq. 7/8, `LinearInterpolant.compute_drift`).

Reuses the already-trained checkpoints under
experiments/l63/vanilla_cfm_beta0.5_unconditional/{s0,s1}/checkpoints/ --
no retraining. Writes results.json in the same schema as the sibling
experiments/l63/<model>/ directories so it drops straight into the same
comparison tables.

Usage:
    python score_based_inference.py
    python score_based_inference.py model.N_ensemble=20
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
from evaluation.metrics import rmse, energy_score, param_rmse as _param_rmse_fn

SOURCE_MODEL = "vanilla_cfm_beta0.5_unconditional"
EXPERIMENT_ID = "score_based_cfm"


def score_based_sample(model, batch, N_outer, R_var, sigma_prior=1.0):
    """Sample x1 ~ p(x1|y) by guiding the unconditional prior E[x1|x_tau]
    with observations y at each Euler step (Eq. 12-14 of the preprint).

    r_tau^2 = alpha_tau^2 * sigma_prior^2 / beta_tau^2 is the (linear-
    Gaussian) conditional variance of x1 given x_tau alone (Eq. 6's
    background precision, inverted); the Kalman gain r_tau^2/(r_tau^2+R_var)
    interpolates between "trust y" (tau->0, r_tau^2->inf) and "trust the
    unconditional prior mean" (tau->1, r_tau^2->0), matching the intuition
    that x_tau becomes fully informative about x1 as tau->1.
    """
    obs = batch.obs
    B, T, D = obs.shape
    device = obs.device
    interpolant = model.interpolant
    y = torch.nan_to_num(obs, nan=0.0)
    mask = batch.obs_mask
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1).expand(B, T, D)
    mask = mask.float()

    x = torch.randn_like(obs) * sigma_prior
    dt = 1.0 / N_outer
    for step in range(N_outer):
        tau = torch.full((B,), step / N_outer, device=device)
        v_prior = model.forward(x, batch, tau)
        a = interpolant.alpha(tau).view(B, 1, 1)
        b = interpolant.beta(tau).clamp(min=1e-3).view(B, 1, 1)
        mu = x + (1.0 - tau).view(B, 1, 1) * v_prior
        r2 = (a ** 2) * (sigma_prior ** 2) / (b ** 2)
        gain = r2 / (r2 + R_var)
        mu_guided = mu + gain * mask * (y - mu)
        drift = interpolant.compute_drift(x, mu_guided, tau)
        x = x + dt * drift
    return x


def evaluate_score_based(model, dataset, device, N_ensemble, N_outer, R_var,
                         sigma_prior, param_names, param_dim):
    """Mirrors `train.evaluate_model`, but draws samples via
    `score_based_sample` instead of `model.sample(batch)`."""
    rmse_list, es_list, ens_var_list = [], [], []
    all_sq_err, all_ref = [], []
    for i in range(len(dataset)):
        w = dataset[i]
        batch = _make_eval_batch(w, device, param_names=param_names, param_dim=param_dim)
        member_preds = []
        for _ in range(N_ensemble):
            pred = score_based_sample(model, batch, N_outer, R_var, sigma_prior)
            member_preds.append(pred.detach().cpu().numpy()[0])
        truth = w["true_state"].numpy()
        ensemble = np.stack(member_preds, axis=0)  # (N_ensemble, T, D)
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


def save_trajectories(model, dataset, device, N_ensemble, N_outer, R_var,
                      sigma_prior, param_names, param_dim, save_path):
    trajs, truths, members = [], [], []
    for i in range(len(dataset)):
        w = dataset[i]
        batch = _make_eval_batch(w, device, param_names=param_names, param_dim=param_dim)
        member_preds = []
        for _ in range(N_ensemble):
            pred = score_based_sample(model, batch, N_outer, R_var, sigma_prior)
            member_preds.append(pred.detach().cpu().numpy()[0])
        truth = w["true_state"].numpy()
        stacked = np.stack(member_preds, axis=-1)  # (T, D, M)
        trajs.append(stacked.mean(axis=-1))
        truths.append(truth)
        members.append(stacked)
    np.savez_compressed(save_path, trajectories=np.stack(trajs, axis=0),
                        truths=np.stack(truths, axis=0), members=np.stack(members, axis=0))


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


@hydra.main(config_path="config", config_name=f"models/{SOURCE_MODEL}", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exp_dir = os.path.join(EXP_DIR, "l63", EXPERIMENT_ID)
    os.makedirs(exp_dir, exist_ok=True)
    combined_results_path = os.path.join(exp_dir, "results.json")

    dc = cfg.data
    vc = cfg.model.vanilla_cfm
    param_names = tuple(dc.get("param_names", ["sigma", "rho", "beta", "c1"]))
    param_dim = cfg.model.get("param_dim", 4)
    state_names = cfg.data.get("state_names", ["X", "Y", "Z"])
    N_ensemble = cfg.model.get("N_ensemble", 50)
    N_outer = vc.N_outer
    sigma_prior = vc.sigma_prior
    R_var = dc.R_var

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

        test_key = f"test_{case}"
        dataset = datasets[test_key]
        with torch.no_grad():
            m, s, r2, crps_mean, crps_std, ens_spread = evaluate_score_based(
                model, dataset, device, N_ensemble, N_outer, R_var, sigma_prior,
                param_names, param_dim)
        eval_elapsed = time.time() - t0
        print(f"  {case}: rmse={np.mean(m):.4f}  r2={np.mean(r2):.4f}  "
              f"crps={np.mean(crps_mean):.4f}  spread={np.mean(ens_spread):.4f}  "
              f"({eval_elapsed:.0f}s)")

        case_dir = os.path.join(exp_dir, case)
        os.makedirs(case_dir, exist_ok=True)
        with torch.no_grad():
            save_trajectories(model, dataset, device, N_ensemble, N_outer, R_var,
                              sigma_prior, param_names, param_dim,
                              os.path.join(case_dir, f"trajectories_{case}.npz"))

        entry = _rmse_entry(state_names, m, s, r2, crps_mean, crps_std, ens_spread, eval_elapsed)
        with open(os.path.join(case_dir, "results.json"), "w") as f:
            json.dump({"experiment_id": f"{EXPERIMENT_ID}_{case}", "model_type": "score_based_cfm",
                      f"fm_{case}": entry, "eval_time_seconds": eval_elapsed}, f, indent=2)

        combined[case] = entry
        total_time += eval_elapsed

    combined["config"] = {
        "hidden_channels": list(vc.hidden_channels),
        "N_outer": N_outer,
        "N_ensemble": N_ensemble,
        "model_type": "score_based_cfm",
        "prior_model": SOURCE_MODEL,
        "R_var": R_var,
        "sigma_prior": sigma_prior,
    }
    combined["total_time_seconds"] = total_time
    with open(combined_results_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved combined results to {combined_results_path}")


if __name__ == "__main__":
    main()
