#!/usr/bin/env python3
"""
4DVarNet-FM: Training entry point with Hydra config management.
Supports TweedieSolver, DirectUNet, and VanillaCFM models.

Usage:
    python train.py                                               # defaults (TweedieSolver)
    python train.py --config-name experiment/E1_direct_unet_default  # experiment preset
"""
import os, sys, json, time
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_float32_matmul_precision('medium')

from data.lorenz63 import Lorenz63Config, make_mixed_datasets, make_s0_s1_trainval, Lorenz63Dataset
from data.random_param_dataset import RandomParamLorenz63Dataset
from data.dataloader import FlowMatchingDataset, ConcatFMDataset, collate_fm
from torch.utils.data import DataLoader
from models.solver import TweedieSolver
from models.direct_unet import DirectUNet
from models.vanilla_cfm import VanillaCFM
from training.pipeline import run_2stage_pipeline, create_trainer, train_stage
from training.lightning_module import LitModel
from evaluation.metrics import rmse, param_rmse

BASE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(BASE, "experiments")


def make_experiment_dataloaders(datasets, batch_size=32, train_mix="cs1+cs2",
                                num_workers=4, randomize_params=False, param_noise=0.2,
                                base_cfg=None, num_train_windows=1000,
                                data_setup="legacy", with_params=False):
    kw = dict(batch_size=batch_size, collate_fn=collate_fm,
              num_workers=num_workers, pin_memory=True)
    if data_setup == "s0_s1":
        obs_cfg = {"obs_interval": base_cfg.obs_interval, "R_var": base_cfg.R_var} if base_cfg else {}
        return {
            "train": DataLoader(FlowMatchingDataset(datasets["train"], with_params=True, **obs_cfg), shuffle=True, **kw),
            "val": DataLoader(FlowMatchingDataset(datasets["val"], with_params=True, **obs_cfg), shuffle=False, **kw),
        }
    if randomize_params and base_cfg is not None:
        train_cs1 = RandomParamLorenz63Dataset(
            Lorenz63Config(**{**base_cfg.__dict__, "case": 1, "param_bias": 0.0,
                              "seed": 42, "num_windows": num_train_windows}), param_noise=param_noise)
        train_cs2 = RandomParamLorenz63Dataset(
            Lorenz63Config(**{**base_cfg.__dict__, "case": 2, "param_bias": 0.15,
                              "forcing_state_bias": 0.15, "forcing_coupling": "quartic",
                              "seed": 42, "num_windows": num_train_windows}), param_noise=param_noise)
        train_sources = {"cs1_rand+cs2_rand": [train_cs1, train_cs2]}
    else:
        train_sources = {
            "cs1+cs2": [datasets["train_cs1"], datasets["train_cs2"]],
            "cs1_only": [datasets["train_cs1"]],
            "cs2_only": [datasets["train_cs2"]],
        }
    sources = train_sources.get(train_mix, next(iter(train_sources.values())))
    return {
        "train": DataLoader(ConcatFMDataset(sources), shuffle=True, **kw),
        "val": DataLoader(
            ConcatFMDataset([datasets["val_cs1"], datasets["val_cs2"]]),
            shuffle=False, **kw),
    }


def model_factory(cfg: DictConfig, device: torch.device):
    model_type = cfg.model.get("model_type", "tweedie")
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
        model = DirectUNet(
            state_dim=cfg.model.state_dim,
            hidden_channels=dc.hidden_channels,
            dropout=dc.dropout,
        )
    elif model_type == "vanilla_cfm":
        vc = cfg.model.vanilla_cfm
        model = VanillaCFM(
            state_dim=cfg.model.state_dim,
            hidden_channels=vc.hidden_channels,
            time_emb_dim=vc.time_emb_dim,
            N_outer=vc.N_outer,
            sigma_prior=vc.sigma_prior,
            dropout=vc.dropout,
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
            train_tau_0_only=jc.train_tau_0_only,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model.to(device)


def _make_eval_batch(w, device):
    from data.dataloader import FlowMatchingBatch
    states = w["true_state"].unsqueeze(0).to(device)
    obs = w["obs"].unsqueeze(0).to(device)
    mask = w["obs_mask"].unsqueeze(0).to(device)
    forcing = w["forcing_corrupted"].unsqueeze(0).to(device)
    params = torch.tensor([[w.get("sigma", 0.0), w.get("rho", 0.0),
                            w.get("beta", 0.0), w.get("c1", 1.0)]], dtype=torch.float32, device=device)
    true_params = torch.tensor([[w.get("true_sigma", w.get("sigma", 0.0)),
                                  w.get("true_rho", w.get("rho", 0.0)),
                                  w.get("true_beta", w.get("beta", 0.0)),
                                  w.get("true_c1", w.get("c1", 1.0))]], dtype=torch.float32, device=device)
    return FlowMatchingBatch(states, obs, mask, forcing, params=params, true_params=true_params)


def evaluate_model(model, dataset, device, model_type="tweedie", return_params=False):
    rmse_list = []
    param_list = []
    true_param_list = []
    for i in range(len(dataset)):
        w = dataset[i]
        batch = _make_eval_batch(w, device)
        if model_type == "tweedie":
            pred = model(batch.obs).detach().cpu().numpy()[0]
        elif model_type == "direct_unet":
            pred = model(batch).detach().cpu().numpy()[0]
        elif model_type == "vanilla_cfm":
            pred = model.sample(batch).detach().cpu().numpy()[0]
        elif model_type == "joint_cfm":
            pred, params = model.sample(batch, return_params=True)
            pred = pred.detach().cpu().numpy()[0]
            param_list.append(params.detach().cpu().numpy()[0])
            tp = [w.get("true_sigma", w.get("sigma")),
                  w.get("true_rho", w.get("rho")),
                  w.get("true_beta", w.get("beta")),
                  w.get("true_c1", w.get("c1", 1.0))]
            true_param_list.append(np.array(tp))
        truth = w["true_state"].numpy()
        rmse_list.append(rmse(pred, truth))
    all_rmse = np.stack(rmse_list, axis=0)
    out = (np.mean(all_rmse, axis=0), np.std(all_rmse, axis=0))
    if return_params and len(param_list) > 0:
        pred_params = np.stack(param_list, axis=0)
        true_params = np.stack(true_param_list, axis=0)
        prmse = param_rmse(pred_params, true_params)
        return out + (prmse,)
    return out


def save_trajectories(model, dataset, device, model_type, save_path):
    trajs, truths = [], []
    for i in range(len(dataset)):
        w = dataset[i]
        batch = _make_eval_batch(w, device)
        if model_type == "tweedie":
            pred = model(batch.obs).detach().cpu().numpy()[0]
        elif model_type == "direct_unet":
            pred = model(batch).detach().cpu().numpy()[0]
        elif model_type == "vanilla_cfm":
            pred = model.sample(batch).detach().cpu().numpy()[0]
        elif model_type == "joint_cfm":
            pred = model.sample(batch).detach().cpu().numpy()[0]
        trajs.append(pred)
        truths.append(w["true_state"].numpy())
    np.savez_compressed(save_path,
                        trajectories=np.stack(trajs, axis=0),
                        truths=np.stack(truths, axis=0))


@hydra.main(config_path="config", config_name="lorenz63_default", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({dev_name})")

    model_type = cfg.model.get("model_type", "tweedie")
    train_mix = cfg.data.get("train_mix", "cs1+cs2")
    randomize_params = cfg.data.get("randomize_params", False)
    param_noise = cfg.data.get("param_noise", 0.2)
    data_setup = cfg.data.get("data_setup", "legacy")
    exp_id = cfg.get("experiment_id", f"{model_type}_custom")
    from hydra.core.hydra_config import HydraConfig
    hcfg = HydraConfig.get()
    if hcfg and hcfg.job.config_name and hcfg.job.config_name.startswith("experiment/"):
        exp_id = hcfg.job.config_name.replace("experiment/", "")

    exp_dir = os.path.join(EXP_DIR, exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    results_path = os.path.join(exp_dir, "results.json")
    trajs_path = os.path.join(exp_dir, "trajectories.npz")

    if os.path.exists(results_path):
        print(f"  Results exist at {results_path}, skipping.")
        return

    # Data
    dc = cfg.data
    base_cfg = Lorenz63Config(
        dt=dc.dt, T_max=dc.T_max, obs_interval=dc.obs_interval,
        R_var=dc.R_var, B_var=dc.B_var,
        num_windows=dc.num_windows, window_spacing=dc.window_spacing,
        spinup_steps=dc.spinup_steps, seed=dc.get("seed", 42),
        sigma_true=dc.sigma_true, rho_true=dc.rho_true, beta_true=dc.beta_true,
        gamma=dc.gamma, W_L_bar=dc.W_L_bar, c1=dc.c1, c2=dc.c2,
        sigma_0=dc.sigma_0, sigma_L=dc.sigma_L,
        tau_eta=dc.tau_eta, sigma_eta=dc.sigma_eta,
        param_bias=dc.get("param_bias", 0.0),
        forcing_state_bias=dc.get("forcing_state_bias", 0.0),
        forcing_coupling=dc.get("forcing_coupling", "linear"),
    )
    if data_setup == "s0_s1":
        bias_max = dc.get("bias_max", 0.2)
        datasets = make_s0_s1_trainval(
            base_cfg,
            num_train_windows=dc.get("num_train_windows", 1000),
            num_val_windows=dc.get("num_val_windows", 100),
            num_test_windows=dc.get("num_test_windows", 200),
            param_noise=dc.get("test_param_noise", 0.2),
            bias_range=(0.0, bias_max),
        )
        test_keys = ["test_s0", "test_s1"]
    else:
        datasets = make_mixed_datasets(
            base_cfg,
            num_train_windows=dc.get("num_train_windows", 1000),
            num_val_windows=dc.get("num_val_windows", 100),
            num_test_windows=dc.get("num_test_windows", 200),
            include_randparam_test=dc.get("test_randparam", True),
            param_noise=dc.get("test_param_noise", 0.2),
        )
        test_keys = ["test_cs1", "test_cs2", "test_cs3", "test_cs4"]
    loaders = make_experiment_dataloaders(
        datasets, batch_size=cfg.training.batch_size,
        train_mix=train_mix, num_workers=4,
        randomize_params=randomize_params, param_noise=param_noise,
        base_cfg=base_cfg,
        num_train_windows=dc.get("num_train_windows", 1000),
        data_setup=data_setup,
        with_params=(model_type == "joint_cfm"),
    )

    print(f"  Train: {len(loaders['train'].dataset)}, Val: {len(loaders['val'].dataset)}")

    # Model
    print(f"  Creating model (type={model_type})...")
    model = model_factory(cfg, device)

    # Train
    total_t0 = time.time()
    orig_cwd = os.getcwd()
    os.chdir(exp_dir)
    try:
        epochs_s1 = cfg.training.stage1.epochs
        epochs_s2 = cfg.training.stage2.epochs
        train_time = 0.0

        if epochs_s1 > 0:
            t0 = time.time()
            if model_type == "tweedie":
                model = train_stage(model, loaders, cfg, stage=1, device=device)
            else:
                stage_cfg = cfg.training.stage1
                lit = LitModel(model, model_type=model_type, stage=1,
                               lr=stage_cfg.lr, gradient_clip_val=stage_cfg.gradient_clip_val,
                               use_gradient_loss=cfg.training.loss.use_gradient,
                               gradient_weight=cfg.training.loss.gradient_weight)
                trainer = create_trainer(cfg, 1)
                trainer.fit(lit, loaders["train"], loaders["val"])
                path = cfg.paths.checkpoint_stage1
                torch.save(lit.model.state_dict(), path)
            train_time += time.time() - t0
            print(f"    Stage 1 done in {train_time:.1f}s")

        if model_type == "tweedie" and epochs_s2 > 0:
            t0 = time.time()
            model = train_stage(model, loaders, cfg, stage=2, device=device)
            train_time += time.time() - t0
            print(f"    Stage 2 done in {time.time()-t0:.1f}s")
    finally:
        os.chdir(orig_cwd)
    total_t = time.time() - total_t0

    # Evaluate
    model.to(device)
    model.eval()
    t0 = time.time()
    results_metrics = {}
    param_metrics = {}
    is_joint = model_type == "joint_cfm"
    for key in test_keys:
        if key not in datasets:
            continue
        if is_joint:
            m, s, prmse = evaluate_model(model, datasets[key], device, model_type, return_params=True)
            results_metrics[key] = (m, s)
            param_metrics[key] = prmse
        else:
            m, s = evaluate_model(model, datasets[key], device, model_type)
            results_metrics[key] = (m, s)
    eval_t = time.time() - t0

    # Save trajectories
    for key in test_keys:
        if key in datasets:
            case = key.replace("test_", "")
            save_trajectories(model, datasets[key], device, model_type,
                              os.path.join(exp_dir, f"trajectories_{case}.npz"))

    def _rmse_entry(m, s):
        return {
            "X": {"mean": float(m[0]), "std": float(s[0])},
            "Y": {"mean": float(m[1]), "std": float(s[1])},
            "Z": {"mean": float(m[2]), "std": float(s[2])},
            "mean": float(np.mean(m)),
        }

    def _param_entry(p):
        return {
            "sigma": float(p[0]), "rho": float(p[1]),
            "beta": float(p[2]), "c1": float(p[3]),
        }

    s0 = results_metrics.get("test_s0")
    s1 = results_metrics.get("test_s1")
    cs1 = results_metrics.get("test_cs1")
    cs2 = results_metrics.get("test_cs2")
    cs3 = results_metrics.get("test_cs3")
    cs4 = results_metrics.get("test_cs4")

    hc_src = (cfg.model.direct_unet if model_type == "direct_unet"
              else cfg.model.get("vanilla_cfm") if model_type in ("vanilla_cfm", "joint_cfm")
              else cfg.model)
    result = {
        "experiment_id": exp_id,
        "model_type": model_type,
        "config": {
            "hidden_channels": list(hc_src.hidden_channels) if hc_src is not None and "hidden_channels" in hc_src else list(cfg.model.hidden_channels),
            "epochs": epochs_s1 + (epochs_s2 if model_type == "tweedie" else 0),
            "train_mix": train_mix,
            "randomize_params": randomize_params,
            "data_setup": data_setup,
        },
        "total_time_seconds": total_t,
        "train_time_seconds": train_time,
        "eval_time_seconds": eval_t,
    }
    if s0:
        result["fm_s0"] = _rmse_entry(*s0)
    if s1:
        result["fm_s1"] = _rmse_entry(*s1)
    if cs1:
        result["fm_cs1"] = _rmse_entry(*cs1)
    if cs2:
        result["fm_cs2"] = _rmse_entry(*cs2)
    if cs3:
        result["fm_cs3"] = _rmse_entry(*cs3)
    if cs4:
        result["fm_cs4"] = _rmse_entry(*cs4)
    if s0 and s1:
        result["fm_degradation"] = float(np.mean(s1[0]) / (np.mean(s0[0]) + 1e-10))
    if cs1 and cs2:
        result["fm_degradation_cs1cs2"] = float(np.mean(cs2[0]) / (np.mean(cs1[0]) + 1e-10))
    if cs3 and cs4:
        result["fm_degradation_cs3cs4"] = float(np.mean(cs4[0]) / (np.mean(cs3[0]) + 1e-10))
    if is_joint:
        if "test_s0" in param_metrics:
            result["param_rmse_s0"] = _param_entry(param_metrics["test_s0"])
        if "test_s1" in param_metrics:
            result["param_rmse_s1"] = _param_entry(param_metrics["test_s1"])

    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  ── Results ─────────────────────────────────")
    if s0:
        m0, _ = s0
        print(f"  S0: X={m0[0]:.4f} Y={m0[1]:.4f} Z={m0[2]:.4f}  mean={np.mean(m0):.4f}")
    if s1:
        m1, _ = s1
        print(f"  S1: X={m1[0]:.4f} Y={m1[1]:.4f} Z={m1[2]:.4f}  mean={np.mean(m1):.4f}")
    if cs1:
        m1, s1 = cs1
        print(f"  CS1: X={m1[0]:.4f} Y={m1[1]:.4f} Z={m1[2]:.4f}  mean={np.mean(m1):.4f}")
    if cs2:
        m2, s2 = cs2
        print(f"  CS2: X={m2[0]:.4f} Y={m2[1]:.4f} Z={m2[2]:.4f}  mean={np.mean(m2):.4f}")
    if is_joint:
        for k in ["test_s0", "test_s1"]:
            if k in param_metrics:
                p = param_metrics[k]
                print(f"  {k} param RMSE: s={p[0]:.4f} r={p[1]:.4f} b={p[2]:.4f} c1={p[3]:.4f}")
    print(f"  Total: {total_t:.0f}s")


if __name__ == "__main__":
    main()
