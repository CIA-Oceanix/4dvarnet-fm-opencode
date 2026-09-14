#!/usr/bin/env python3
"""
4DVarNet-FM: Training entry point with Hydra config management.
Supports TweedieSolver, DirectUNet, and VanillaCFM models.

Usage:
    python train.py                                               # defaults (TweedieSolver)
    python train.py --config-name models/E1_direct_unet_default  # model preset

The on-disk dataset cache (dataset_cache/) can be pre-warmed independently
of a training run with:
    python generate_dataset.py --config-name models/<name>
"""
import os
import sys
import json
import time
import logging
from collections import namedtuple
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_float32_matmul_precision('medium')

logger = logging.getLogger(__name__)

from data.dataloader import FlowMatchingDataset, collate_fm
from data.build import build_datasets
from torch.utils.data import DataLoader
from models.solver import TweedieSolver
from models.direct_unet import DirectUNet
from models.vanilla_cfm import VanillaCFM
from training.pipeline import create_trainer, train_stage, load_best_checkpoint
from training.lightning_module import LitModel
from evaluation.metrics import rmse, param_rmse, energy_score

BASE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(BASE, "experiments")


def make_experiment_dataloaders(datasets, batch_size=32, num_workers=4,
                                base_cfg=None, with_params=False):
    kw = dict(batch_size=batch_size, collate_fn=collate_fm,
              num_workers=num_workers, pin_memory=True)
    obs_cfg = {"obs_interval": base_cfg.obs_interval, "R_var": base_cfg.R_var} if base_cfg else {}
    return {
        "train": DataLoader(FlowMatchingDataset(datasets["train"], with_params=True, **obs_cfg), shuffle=True, **kw),
        "val": DataLoader(FlowMatchingDataset(datasets["val"], with_params=True, **obs_cfg), shuffle=False, **kw),
    }


def make_l96_dataloaders(datasets, batch_size=32, with_params=False,
                         obs_interval=100, R_var=0.5, param_names=("F",),
                         obs_var_indices=None):
    kw = dict(batch_size=batch_size, collate_fn=collate_fm,
              num_workers=4, pin_memory=True)
    fm_kw = dict(obs_interval=obs_interval, R_var=R_var,
                 with_params=with_params, param_names=list(param_names),
                 obs_var_indices=obs_var_indices)
    return {
        "train": DataLoader(FlowMatchingDataset(datasets["train"], **fm_kw),
                            shuffle=True, **kw),
        "val": DataLoader(FlowMatchingDataset(datasets["val"], **fm_kw),
                          shuffle=False, **kw),
    }


def model_factory(cfg: DictConfig, device: torch.device):
    model_type = cfg.model.get("model_type", "tweedie")
    # What's fed as conditioning input to the ML models (direct_unet/vanilla_cfm/
    # tweedie_cfm/predict_state_cfm); ignored by the legacy "tweedie" solver and
    # by joint_cfm/joint_direct_unet, whose conditioning is fixed by design.
    use_obs = cfg.model.get("use_obs", True)
    use_forcing = cfg.model.get("use_forcing", True)
    use_params = cfg.model.get("use_params", True)
    if model_type == "tweedie":
        model = TweedieSolver(
            state_dim=cfg.model.state_dim,
            hidden_channels=cfg.model.hidden_channels,
            time_emb_dim=cfg.model.time_emb_dim,
            use_obs=cfg.model.use_obs,
            use_energy=cfg.model.use_energy,
            nu=cfg.model.nu,
            K_inner=cfg.model.K_inner,
            N_outer=cfg.model.N_outer,
            dropout=cfg.model.dropout,
        )
    elif model_type == "direct_unet":
        dc = cfg.model.direct_unet
        param_dim = cfg.model.get("param_dim", 4)
        model = DirectUNet(
            state_dim=cfg.model.state_dim,
            hidden_channels=dc.hidden_channels,
            dropout=dc.dropout,
            param_dim=param_dim,
            use_obs=use_obs, use_forcing=use_forcing, use_params=use_params,
        )
    elif model_type == "vanilla_cfm":
        vc = cfg.model.vanilla_cfm
        param_dim = cfg.model.get("param_dim", 4)
        model = VanillaCFM(
            state_dim=cfg.model.state_dim,
            hidden_channels=vc.hidden_channels,
            time_emb_dim=vc.time_emb_dim,
            N_outer=vc.N_outer,
            sigma_prior=vc.sigma_prior,
            dropout=vc.dropout,
            train_tau_0_only=vc.get("train_tau_0_only", False),
            param_dim=param_dim,
            use_obs=use_obs, use_forcing=use_forcing, use_params=use_params,
            tau_sampling=vc.get("tau_sampling", "uniform"),
            logit_normal_loc=vc.get("logit_normal_loc", 0.0),
            logit_normal_scale=vc.get("logit_normal_scale", 1.0),
            beta_alpha=vc.get("beta_alpha", 2.5),
            beta_beta=vc.get("beta_beta", 1.0),
        )
    elif model_type == "joint_cfm":
        from models.vanilla_cfm import JointCFM
        jc = cfg.model.joint_cfm
        vc = cfg.model.vanilla_cfm
        model = JointCFM(
            state_dim=cfg.model.state_dim,
            param_dim=jc.param_dim,
            hidden_channels=vc.hidden_channels,
            time_emb_dim=vc.time_emb_dim,
            N_outer=vc.N_outer,
            sigma_prior=vc.sigma_prior,
            dropout=vc.dropout,
            param_loss_weight=jc.param_loss_weight,
            param_flow_channels=jc.get("param_flow_channels", None),
            train_tau_0_only=jc.train_tau_0_only,
            tau_sampling=jc.get("tau_sampling", "uniform"),
            logit_normal_loc=jc.get("logit_normal_loc", 0.0),
            logit_normal_scale=jc.get("logit_normal_scale", 1.0),
            beta_alpha=jc.get("beta_alpha", 2.5),
            beta_beta=jc.get("beta_beta", 1.0),
        )
    elif model_type == "joint_direct_unet":
        from models.direct_unet import JointDirectUNet
        jdu = cfg.model.joint_direct_unet
        dc = cfg.model.direct_unet
        model = JointDirectUNet(
            state_dim=cfg.model.state_dim,
            param_dim=jdu.param_dim,
            hidden_channels=dc.hidden_channels,
            dropout=dc.dropout,
            param_loss_weight=jdu.param_loss_weight,
            param_head_channels=jdu.get("param_head_channels", None),
        )
    elif model_type == "predict_state_cfm":
        from models.vanilla_cfm import PredictStateCFM
        psc = cfg.model.predict_state_cfm
        param_dim = cfg.model.get("param_dim", 4)
        model = PredictStateCFM(
            state_dim=cfg.model.state_dim,
            hidden_channels=psc.hidden_channels,
            time_emb_dim=psc.time_emb_dim,
            N_outer=psc.N_outer,
            sigma_prior=psc.sigma_prior,
            dropout=psc.dropout,
            train_tau_0_only=psc.get("train_tau_0_only", False),
            param_dim=param_dim,
            use_obs=use_obs, use_forcing=use_forcing, use_params=use_params,
            tau_sampling=psc.get("tau_sampling", "uniform"),
            logit_normal_loc=psc.get("logit_normal_loc", 0.0),
            logit_normal_scale=psc.get("logit_normal_scale", 1.0),
            beta_alpha=psc.get("beta_alpha", 2.5),
            beta_beta=psc.get("beta_beta", 1.0),
        )
    elif model_type == "tweedie_cfm":
        from models.vanilla_cfm import TweedieCFM
        tc = cfg.model.tweedie_cfm
        param_dim = cfg.model.get("param_dim", 4)
        model = TweedieCFM(
            state_dim=cfg.model.state_dim,
            hidden_channels=tc.hidden_channels,
            time_emb_dim=tc.time_emb_dim,
            K_inner=tc.K_inner,
            N_outer=tc.N_outer,
            sigma_prior=tc.sigma_prior,
            dropout=tc.dropout,
            train_tau_0_only=tc.train_tau_0_only,
            param_dim=param_dim,
            use_obs=use_obs, use_forcing=use_forcing, use_params=use_params,
            tau_sampling=tc.get("tau_sampling", "uniform"),
            logit_normal_loc=tc.get("logit_normal_loc", 0.0),
            logit_normal_scale=tc.get("logit_normal_scale", 1.0),
            beta_alpha=tc.get("beta_alpha", 2.5),
            beta_beta=tc.get("beta_beta", 1.0),
        )
    elif model_type == "joint_tweedie_cfm":
        from models.vanilla_cfm import JointTweedieCFM
        jtc = cfg.model.joint_tweedie_cfm
        tc = cfg.model.tweedie_cfm
        model = JointTweedieCFM(
            state_dim=cfg.model.state_dim,
            param_dim=jtc.param_dim,
            hidden_channels=tc.hidden_channels,
            time_emb_dim=tc.time_emb_dim,
            K_inner=tc.K_inner,
            N_outer=tc.N_outer,
            sigma_prior=tc.sigma_prior,
            dropout=tc.dropout,
            param_loss_weight=jtc.param_loss_weight,
            param_flow_channels=jtc.get("param_flow_channels", None),
            train_tau_0_only=jtc.train_tau_0_only,
            tau_sampling=jtc.get("tau_sampling", "uniform"),
            logit_normal_loc=jtc.get("logit_normal_loc", 0.0),
            logit_normal_scale=jtc.get("logit_normal_scale", 1.0),
            beta_alpha=jtc.get("beta_alpha", 2.5),
            beta_beta=jtc.get("beta_beta", 1.0),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model.to(device)


def _make_eval_batch(w, device, param_names=("sigma", "rho", "beta", "c1"),
                     param_dim=4):
    from data.dataloader import FlowMatchingBatch
    states = w["true_state"].unsqueeze(0).to(device)
    obs = w["obs"].unsqueeze(0).to(device)
    mask = w["obs_mask"].unsqueeze(0).to(device)
    forcing = w["forcing_corrupted"].unsqueeze(0).to(device)
    if param_dim == 0:
        return FlowMatchingBatch(states, obs, mask, forcing)
    params = torch.tensor([[w.get(nm, 0.0) for nm in param_names]],
                          dtype=torch.float32, device=device)
    true_params = torch.tensor([[w.get(f"true_{nm}", w.get(nm, 0.0)) for nm in param_names]],
                               dtype=torch.float32, device=device)
    return FlowMatchingBatch(states, obs, mask, forcing, params=params, true_params=true_params)


# Model types whose prediction is stochastic (each call draws fresh noise),
# so repeated calls on the same input form a genuine ensemble. Deterministic
# types (direct_unet, joint_direct_unet) always return the same output, so
# N_ensemble is a no-op for them.
STOCHASTIC_MODEL_TYPES = {"tweedie", "vanilla_cfm", "joint_cfm", "predict_state_cfm", "tweedie_cfm", "joint_tweedie_cfm"}


def _predict_once(model, batch, model_type, return_params=False):
    """Single forward/sample pass. Returns (pred, params_or_None)."""
    if model_type == "tweedie":
        return model(batch.obs), None
    elif model_type == "direct_unet":
        return model(batch), None
    elif model_type in ("joint_cfm", "joint_direct_unet", "joint_tweedie_cfm"):
        pred, params = model.sample(batch, return_params=True)
        return pred, params
    elif model_type in ("vanilla_cfm", "predict_state_cfm", "tweedie_cfm"):
        return model.sample(batch), None
    raise ValueError(f"Unknown model_type: {model_type}")


EvalMetrics = namedtuple(
    "EvalMetrics",
    ["mean", "std", "r2", "crps_mean", "crps_std", "ensemble_spread", "param_rmse"],
)


def evaluate_model(model, dataset, device, model_type="tweedie", return_params=False,
                   param_names=("sigma", "rho", "beta", "c1"), param_dim=4,
                   obs_var_indices=None, N_ensemble=1):
    """Evaluate `model` over every window in `dataset`.

    Mirrors `evaluation.run.evaluate_baseline`/`fmt_rmse`'s metrics so ML and
    baseline results are directly comparable: per-window RMSE mean/std, R^2
    (1 - MSE/Var) pooled over all windows/timesteps together, and CRPS
    (Energy Score -- reduces to MAE for a deterministic single-member
    "ensemble") mean/std across windows. `ensemble_spread` (RMS ensemble
    std, pooled like R^2) is populated only when multiple stochastic draws
    are actually taken.
    """
    n_draws = N_ensemble if (model_type in STOCHASTIC_MODEL_TYPES and N_ensemble > 1) else 1
    rmse_list = []
    es_list = []
    ens_var_list = []
    all_sq_err = []
    all_ref = []
    param_list = []
    true_param_list = []
    for i in range(len(dataset)):
        w = dataset[i]
        batch = _make_eval_batch(w, device, param_names=param_names, param_dim=param_dim)
        member_preds, member_params = [], []
        for _ in range(n_draws):
            pred, params = _predict_once(model, batch, model_type, return_params=return_params)
            member_preds.append(pred.detach().cpu().numpy()[0])
            if params is not None:
                member_params.append(params.detach().cpu().numpy()[0])
        truth = w["true_state"].numpy()
        if obs_var_indices is not None and member_preds[0].shape[-1] != truth.shape[-1]:
            truth = truth[..., obs_var_indices]
        ensemble = np.stack(member_preds, axis=0)  # (n_draws, T, D)
        pred = ensemble.mean(axis=0)
        rmse_list.append(rmse(pred, truth))
        es_list.append(energy_score(ensemble, truth))
        all_sq_err.append((pred - truth) ** 2)
        all_ref.append(truth)
        if n_draws > 1:
            ens_var_list.append(ensemble.var(axis=0))
        if member_params:
            param_list.append(np.mean(member_params, axis=0))
            tp = [w.get(f"true_{nm}", w.get(nm, 0.0)) for nm in param_names]
            true_param_list.append(np.array(tp))

    all_rmse = np.stack(rmse_list, axis=0)
    mean_rmse, std_rmse = np.mean(all_rmse, axis=0), np.std(all_rmse, axis=0)

    pooled_mse = np.mean(np.concatenate(all_sq_err, axis=0), axis=0)
    pooled_var = np.maximum(np.var(np.concatenate(all_ref, axis=0), axis=0), 1e-12)
    r2 = 1.0 - pooled_mse / pooled_var

    all_es = np.stack(es_list, axis=0)
    crps_mean, crps_std = np.mean(all_es, axis=0), np.std(all_es, axis=0)

    ensemble_spread = (np.sqrt(np.mean(np.concatenate(ens_var_list, axis=0), axis=0))
                       if ens_var_list else None)

    prmse = None
    if return_params and len(param_list) > 0:
        pred_params = np.stack(param_list, axis=0)
        true_params = np.stack(true_param_list, axis=0)
        prmse = param_rmse(pred_params, true_params)

    return EvalMetrics(mean_rmse, std_rmse, r2, crps_mean, crps_std, ensemble_spread, prmse)


def _per_group_rmse(mean_rmse, obs_var_indices, NO=8, J=4, obs_j=2):
    groups = {}
    groups["all_obs"] = float(np.mean(mean_rmse))
    groups["slow"] = float(np.mean(mean_rmse[:NO]))
    if obs_j < J:
        groups["obs_fast"] = float(np.mean(mean_rmse[NO:]))
    else:
        groups["obs_fast"] = float(np.mean(mean_rmse[NO:]))
    return groups


def save_trajectories(model, dataset, device, model_type, save_path,
                      param_names=("sigma", "rho", "beta", "c1"), param_dim=4,
                      obs_var_indices=None, N_ensemble=1):
    n_draws = N_ensemble if (model_type in STOCHASTIC_MODEL_TYPES and N_ensemble > 1) else 1
    trajs, truths, members = [], [], []
    for i in range(len(dataset)):
        w = dataset[i]
        batch = _make_eval_batch(w, device, param_names=param_names, param_dim=param_dim)
        member_preds = []
        for _ in range(n_draws):
            pred, _ = _predict_once(model, batch, model_type)
            member_preds.append(pred.detach().cpu().numpy()[0])
        truth = w["true_state"].numpy()
        if obs_var_indices is not None and member_preds[0].shape[-1] != truth.shape[-1]:
            truth = truth[..., obs_var_indices]
        stacked = np.stack(member_preds, axis=-1)  # (T, D, M)
        trajs.append(stacked.mean(axis=-1))
        truths.append(truth)
        if n_draws > 1:
            members.append(stacked)
    save_kwargs = dict(trajectories=np.stack(trajs, axis=0), truths=np.stack(truths, axis=0))
    if members:
        save_kwargs["members"] = np.stack(members, axis=0)  # (W, T, D, M)
    np.savez_compressed(save_path, **save_kwargs)


def run_experiment(cfg: DictConfig, device: torch.device, case: str = None):
    """Train + evaluate one experiment. `case` ('s0'/'s1'/None) picks which
    Lorenz-63 regime to train+eval on (diagonal: train s0 -> eval s0, train
    s1 -> eval s1); it's ignored for systems other than lorenz63."""
    model_type = cfg.model.get("model_type", "tweedie")
    model_base_id = cfg.get("experiment_id", f"{model_type}_custom")
    from hydra.core.hydra_config import HydraConfig
    hcfg = HydraConfig.get()
    if hcfg and hcfg.job.config_name and hcfg.job.config_name.startswith("models/"):
        model_base_id = hcfg.job.config_name.replace("models/", "")

    # Lorenz-63 runs lay out as experiments/l63/<model>/<s0|s1>/, with a
    # combined experiments/l63/<model>/results.json (baseline-schema, see
    # `combined_results_path` below) merging both cases. Other systems keep
    # the flat experiments/<exp_id>/ layout.
    system_hint = cfg.data.get("system", "lorenz63")
    combined_results_path = None
    if system_hint == "lorenz63" and case is not None:
        exp_id = f"{model_base_id}_{case}"
        exp_dir = os.path.join(EXP_DIR, "l63", model_base_id, case)
        combined_results_path = os.path.join(EXP_DIR, "l63", model_base_id, "results.json")
    else:
        exp_id = model_base_id if case is None else f"{model_base_id}_{case}"
        exp_dir = os.path.join(EXP_DIR, exp_id)

    os.makedirs(exp_dir, exist_ok=True)
    results_path = os.path.join(exp_dir, "results.json")

    if os.path.exists(results_path):
        print(f"  Results exist at {results_path}, skipping.")
        return

    # Data
    dc = cfg.data
    param_names = tuple(dc.get("param_names", ["sigma", "rho", "beta", "c1"]))
    datasets, test_keys, base_cfg, system, obs_var_indices = build_datasets(cfg, train_case=case)
    if system == "lorenz96":
        loaders = make_l96_dataloaders(
            datasets, batch_size=cfg.training.batch_size,
            obs_interval=dc.obs_interval, R_var=dc.R_var,
            param_names=param_names,
            with_params=(model_type in ("joint_cfm", "joint_direct_unet", "joint_tweedie_cfm")),
            obs_var_indices=obs_var_indices,
        )
    else:
        loaders = make_experiment_dataloaders(
            datasets, batch_size=cfg.training.batch_size,
            num_workers=4, base_cfg=base_cfg,
            with_params=(model_type in ("joint_cfm", "joint_direct_unet", "joint_tweedie_cfm")),
        )

    print(f"  Train: {len(loaders['train'].dataset)}, Val: {len(loaders['val'].dataset)}")

    # Model
    print(f"  Creating model (type={model_type})...")
    model = model_factory(cfg, device)
    param_dim = cfg.model.get("param_dim", 4)

    # Train
    total_t0 = time.time()
    orig_cwd = os.getcwd()
    os.chdir(exp_dir)
    try:
        epochs_s1 = cfg.training.stage1.get("max_epochs", 0)

        stage2_cfg = cfg.training.get("stage2", {})
        epochs_s2 = stage2_cfg.get("max_epochs", 0)
        train_time = 0.0
        epochs_trained_s1 = epochs_trained_s2 = None

        if epochs_s1 > 0:
            t0 = time.time()
            if model_type == "tweedie":
                model = train_stage(model, loaders, cfg, stage=1, device=device)
            else:
                stage_cfg = cfg.training.stage1
                lit = LitModel(model, model_type=model_type, stage=1,
                               lr=stage_cfg.lr, gradient_clip_val=stage_cfg.gradient_clip_val)
                trainer = create_trainer(cfg, 1, max_epochs=epochs_s1)
                trainer.fit(lit, loaders["train"], loaders["val"])
                epochs_trained_s1 = trainer.current_epoch + 1
                load_best_checkpoint(lit, trainer)
                path = cfg.paths.checkpoint_stage1
                torch.save(lit.model.state_dict(), path)
            train_time += time.time() - t0
            print(f"    Stage 1 done in {train_time:.1f}s"
                  + (f" ({epochs_trained_s1}/{epochs_s1} epochs)" if epochs_trained_s1 else ""))

        if model_type == "tweedie" and epochs_s2 > 0:
            t0 = time.time()
            model = train_stage(model, loaders, cfg, stage=2, device=device)
            train_time += time.time() - t0
            print(f"    Stage 2 done in {time.time()-t0:.1f}s")
        elif model_type in ("tweedie_cfm", "joint_tweedie_cfm") and epochs_s2 > 0:
            t0 = time.time()
            stage_cfg = cfg.training.stage2
            lit = LitModel(model, model_type=model_type, stage=2,
                           lr=stage_cfg.lr, gradient_clip_val=stage_cfg.gradient_clip_val)
            trainer = create_trainer(cfg, 2, max_epochs=epochs_s2)
            trainer.fit(lit, loaders["train"], loaders["val"])
            epochs_trained_s2 = trainer.current_epoch + 1
            load_best_checkpoint(lit, trainer)
            path = cfg.paths.checkpoint_stage2
            torch.save(lit.model.state_dict(), path)
            train_time += time.time() - t0
            print(f"    Stage 2 done in {time.time()-t0:.1f}s ({epochs_trained_s2}/{epochs_s2} epochs)")
            print(f"    Stage 2 done in {time.time()-t0:.1f}s")
    finally:
        os.chdir(orig_cwd)
    total_t = time.time() - total_t0

    # Evaluate
    model.to(device)
    model.eval()
    t0 = time.time()
    results_metrics = {}
    eval_elapsed = {}
    param_metrics = {}
    is_joint = model_type in ("joint_cfm", "joint_direct_unet", "joint_tweedie_cfm")
    NO = dc.get("NO", 8)
    J = dc.get("J", 4)
    obs_j_local = dc.get("obs_j", 2)
    N_ensemble = cfg.model.get("N_ensemble", 1)
    for key in test_keys:
        if key not in datasets:
            continue
        t_case0 = time.time()
        if is_joint:
            metrics = evaluate_model(model, datasets[key], device, model_type,
                                     return_params=True, param_names=param_names,
                                     param_dim=param_dim, obs_var_indices=obs_var_indices,
                                     N_ensemble=N_ensemble)
            param_metrics[key] = metrics.param_rmse
        else:
            metrics = evaluate_model(model, datasets[key], device, model_type,
                                     param_names=param_names, param_dim=param_dim,
                                     obs_var_indices=obs_var_indices, N_ensemble=N_ensemble)
        eval_elapsed[key] = time.time() - t_case0
        results_metrics[key] = metrics
    eval_t = time.time() - t0

    # Save trajectories
    for key in test_keys:
        if key in datasets:
            case = key.replace("test_", "")
            save_trajectories(model, datasets[key], device, model_type,
                              os.path.join(exp_dir, f"trajectories_{case}.npz"),
                              param_names=param_names, param_dim=param_dim,
                              obs_var_indices=obs_var_indices, N_ensemble=N_ensemble)

    state_names = cfg.data.get("state_names", ["X", "Y", "Z"])

    def _rmse_entry(metrics, elapsed_seconds):
        """Same schema as `evaluation.run.fmt_rmse` (baselines), so ML and
        baseline results.json entries are directly comparable."""
        m, s, r2, crps_mean, crps_std, ens_spread = (
            metrics.mean, metrics.std, metrics.r2,
            metrics.crps_mean, metrics.crps_std, metrics.ensemble_spread,
        )
        d = {}
        for i, nm in enumerate(state_names):
            d[nm] = {
                "rmse": {"mean": float(m[i]), "std": float(s[i])},
                "r2": float(r2[i]),
                "crps": {"mean": float(crps_mean[i]), "std": float(crps_std[i])},
            }
            if ens_spread is not None:
                d[nm]["ensemble_spread"] = float(ens_spread[i])
        d["rmse"] = {"mean": float(np.mean(m)), "std": float(np.mean(s))}
        d["r2"] = float(np.mean(r2))
        d["crps"] = {"mean": float(np.mean(crps_mean)), "std": float(np.mean(crps_std))}
        if ens_spread is not None:
            d["ensemble_spread"] = float(np.mean(ens_spread))
        d["elapsed_seconds"] = float(elapsed_seconds)
        if obs_var_indices is not None:
            d["groups"] = _per_group_rmse(m, obs_var_indices, NO=NO, J=J, obs_j=obs_j_local)
        return d

    def _param_entry(p):
        return {nm: float(p[i]) for i, nm in enumerate(param_names)}

    s0 = results_metrics.get("test_s0")
    s1 = results_metrics.get("test_s1")

    hc_src = (cfg.model.direct_unet if model_type in ("direct_unet", "joint_direct_unet")
              else cfg.model.get("vanilla_cfm") if model_type in ("vanilla_cfm", "joint_cfm")
              else cfg.model.get("tweedie_cfm") if model_type in ("tweedie_cfm", "joint_tweedie_cfm")
              else cfg.model.get(model_type, cfg.model))
    result = {
        "experiment_id": exp_id,
        "model_type": model_type,
        "config": {
            "hidden_channels": list(hc_src.hidden_channels) if hc_src is not None and "hidden_channels" in hc_src else list(cfg.model.get("hidden_channels", [])),
            "epochs": epochs_s1 + (epochs_s2 if model_type in ("tweedie", "tweedie_cfm", "joint_tweedie_cfm") else 0),
            "epochs_trained": (epochs_trained_s1 or 0) + (epochs_trained_s2 or 0) or None,
            "N_ensemble": N_ensemble if model_type in STOCHASTIC_MODEL_TYPES else 1,
        },
        "total_time_seconds": total_t,
        "train_time_seconds": train_time,
        "eval_time_seconds": eval_t,
    }
    if s0:
        result["fm_s0"] = _rmse_entry(s0, eval_elapsed["test_s0"])
    if s1:
        result["fm_s1"] = _rmse_entry(s1, eval_elapsed["test_s1"])
    if s0 and s1:
        result["fm_degradation"] = float(np.mean(s1.mean) / (np.mean(s0.mean) + 1e-10))
    if is_joint:
        if "test_s0" in param_metrics:
            result["param_rmse_s0"] = _param_entry(param_metrics["test_s0"])
        if "test_s1" in param_metrics:
            result["param_rmse_s1"] = _param_entry(param_metrics["test_s1"])

    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    if combined_results_path is not None:
        combined = {}
        if os.path.exists(combined_results_path):
            with open(combined_results_path) as f:
                combined = json.load(f)
        combined["config"] = result["config"] | {"model_type": model_type}
        if "fm_s0" in result:
            combined["s0"] = result["fm_s0"]
        if "fm_s1" in result:
            combined["s1"] = result["fm_s1"]
        combined["total_time_seconds"] = combined.get("total_time_seconds", 0.0) + total_t
        with open(combined_results_path, "w") as f:
            json.dump(combined, f, indent=2)

    def _fmt_rmse(m):
        parts = [f"{nm}={m[i]:.4f}" for i, nm in enumerate(state_names)]
        return " ".join(parts) + f"  mean={np.mean(m):.4f}"

    print("\n  ── Results ─────────────────────────────────")
    if s0:
        m0 = s0.mean
        groups0 = _per_group_rmse(m0, obs_var_indices, NO=NO, J=J, obs_j=obs_j_local) if obs_var_indices else {}
        print(f"  S0: {_fmt_rmse(m0)}  r2={np.mean(s0.r2):.4f}  crps={np.mean(s0.crps_mean):.4f}")
        if groups0:
            print(f"       slow={groups0['slow']:.4f}  obs_fast={groups0['obs_fast']:.4f}  all_obs={groups0['all_obs']:.4f}")
    if s1:
        m1 = s1.mean
        groups1 = _per_group_rmse(m1, obs_var_indices, NO=NO, J=J, obs_j=obs_j_local) if obs_var_indices else {}
        print(f"  S1: {_fmt_rmse(m1)}  r2={np.mean(s1.r2):.4f}  crps={np.mean(s1.crps_mean):.4f}")
        if groups1:
            print(f"       slow={groups1['slow']:.4f}  obs_fast={groups1['obs_fast']:.4f}  all_obs={groups1['all_obs']:.4f}")
    if is_joint:
        for k in ["test_s0", "test_s1"]:
            if k in param_metrics:
                p = param_metrics[k]
                parts = " ".join(f"{nm}={p[i]:.4f}" for i, nm in enumerate(param_names))
                print(f"  {k} param RMSE: {parts}")
    print(f"  Total: {total_t:.0f}s")


@hydra.main(config_path="config", config_name="lorenz63", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({dev_name})")

    system = cfg.data.get("system", "lorenz63")
    # Lorenz-63 always runs both regimes: train on s0 -> eval on s0, then
    # train on s1 -> eval on s1 (two independent experiments per config).
    cases = ["s0", "s1"] if system == "lorenz63" else [None]
    for case in cases:
        print(f"\n{'=' * 60}\n  Case: {case or system}\n{'=' * 60}")
        run_experiment(cfg, device, case=case)


if __name__ == "__main__":
    main()
