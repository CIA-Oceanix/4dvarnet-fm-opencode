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

from data.lorenz63 import Lorenz63Config, make_mixed_datasets, Lorenz63Dataset
from data.random_param_dataset import RandomParamLorenz63Dataset
from data.dataloader import FlowMatchingDataset, ConcatFMDataset, collate_fm
from torch.utils.data import DataLoader
from models.solver import TweedieSolver
from models.direct_unet import DirectUNet
from models.vanilla_cfm import VanillaCFM
from training.pipeline import run_2stage_pipeline, create_trainer, train_stage
from training.lightning_module import LitModel
from evaluation.metrics import rmse

BASE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(BASE, "experiments")


def make_experiment_dataloaders(datasets, batch_size=32, train_mix="cs1+cs2",
                                num_workers=4, randomize_params=False, param_noise=0.2,
                                base_cfg=None, num_train_windows=1000):
    kw = dict(batch_size=batch_size, collate_fn=collate_fm,
              num_workers=num_workers, pin_memory=True)
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
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model.to(device)


def evaluate_model(model, dataset, device, model_type="tweedie"):
    rmse_list = []
    for i in range(len(dataset)):
        w = dataset[i]
        obs = w["obs"].unsqueeze(0).to(device)
        if model_type == "tweedie":
            pred = model(obs).detach().cpu().numpy()[0]
        elif model_type == "direct_unet":
            pred = model(obs).detach().cpu().numpy()[0]
        elif model_type == "vanilla_cfm":
            pred = model.sample(obs).detach().cpu().numpy()[0]
        truth = w["true_state"].numpy()
        rmse_list.append(rmse(pred, truth))
    all_rmse = np.stack(rmse_list, axis=0)
    return np.mean(all_rmse, axis=0), np.std(all_rmse, axis=0)


def save_trajectories(model, dataset, device, model_type, save_path):
    trajs, truths = [], []
    for i in range(len(dataset)):
        w = dataset[i]
        obs = w["obs"].unsqueeze(0).to(device)
        if model_type == "tweedie":
            pred = model(obs).detach().cpu().numpy()[0]
        elif model_type == "direct_unet":
            pred = model(obs).detach().cpu().numpy()[0]
        elif model_type == "vanilla_cfm":
            pred = model.sample(obs).detach().cpu().numpy()[0]
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
    datasets = make_mixed_datasets(
        base_cfg,
        num_train_windows=dc.get("num_train_windows", 1000),
        num_val_windows=dc.get("num_val_windows", 100),
        num_test_windows=dc.get("num_test_windows", 200),
    )
    loaders = make_experiment_dataloaders(
        datasets, batch_size=cfg.training.batch_size,
        train_mix=train_mix, num_workers=4,
        randomize_params=randomize_params, param_noise=param_noise,
        base_cfg=base_cfg,
        num_train_windows=dc.get("num_train_windows", 1000),
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
    m1, s1 = evaluate_model(model, datasets["test_cs1"], device, model_type)
    m2, s2 = evaluate_model(model, datasets["test_cs2"], device, model_type)
    eval_t = time.time() - t0

    # Save trajectories
    save_trajectories(model, datasets["test_cs1"], device, model_type,
                      os.path.join(exp_dir, "trajectories_cs1.npz"))
    save_trajectories(model, datasets["test_cs2"], device, model_type,
                      os.path.join(exp_dir, "trajectories_cs2.npz"))

    deg = float(np.mean(m2) / (np.mean(m1) + 1e-10))
    result = {
        "experiment_id": exp_id,
        "model_type": model_type,
        "config": {
            "hidden_channels": list(cfg.model.direct_unet.hidden_channels if model_type == "direct_unet"
                                      else cfg.model.vanilla_cfm.hidden_channels if model_type == "vanilla_cfm"
                                      else cfg.model.hidden_channels),
            "epochs": epochs_s1 + (epochs_s2 if model_type == "tweedie" else 0),
            "train_mix": train_mix,
            "randomize_params": randomize_params,
        },
        "total_time_seconds": total_t,
        "train_time_seconds": train_time,
        "eval_time_seconds": eval_t,
        "fm_cs1": {
            "X": {"mean": float(m1[0]), "std": float(s1[0])},
            "Y": {"mean": float(m1[1]), "std": float(s1[1])},
            "Z": {"mean": float(m1[2]), "std": float(s1[2])},
            "mean": float(np.mean(m1)),
        },
        "fm_cs2": {
            "X": {"mean": float(m2[0]), "std": float(s2[0])},
            "Y": {"mean": float(m2[1]), "std": float(s2[1])},
            "Z": {"mean": float(m2[2]), "std": float(s2[2])},
            "mean": float(np.mean(m2)),
        },
        "fm_degradation": deg,
    }
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  ── Results ─────────────────────────────────")
    print(f"  CS1: X={m1[0]:.4f} Y={m1[1]:.4f} Z={m1[2]:.4f}  mean={np.mean(m1):.4f}")
    print(f"  CS2: X={m2[0]:.4f} Y={m2[1]:.4f} Z={m2[2]:.4f}  mean={np.mean(m2):.4f}")
    print(f"  Degradation: {deg:.2f}x  |  Total: {total_t:.0f}s")


if __name__ == "__main__":
    main()
