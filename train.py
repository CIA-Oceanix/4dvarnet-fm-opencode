#!/usr/bin/env python3
"""
4DVarNet-FM: Training entry point with Hydra config management.
Usage:
    python train.py                            # defaults
    python train.py model.hidden_channels='[32,64,128]'  # override
    python train.py --config-name experiment/B1_small_unet  # experiment preset
"""
import os
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.lorenz63 import Lorenz63Config, make_mixed_datasets
from run_experiments import make_experiment_dataloaders
from models.solver import TweedieSolver
from training.pipeline import run_2stage_pipeline
from evaluation.metrics import rmse
import numpy as np


def evaluate_model(model, dataset, device):
    rmse_list = []
    for i in range(len(dataset)):
        w = dataset[i]
        obs = w["obs"].unsqueeze(0).to(device)
        pred = model(obs).detach().cpu().numpy()[0]
        truth = w["true_state"].numpy()
        rmse_list.append(rmse(pred, truth))
    all_rmse = np.stack(rmse_list, axis=0)
    return np.mean(all_rmse, axis=0), np.std(all_rmse, axis=0)


@hydra.main(config_path="config", config_name="lorenz63_default", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
    datasets = make_mixed_datasets(base_cfg)
    loaders = make_experiment_dataloaders(
        datasets, batch_size=cfg.training.batch_size,
        train_mix=cfg.get("train_mix", "cs1+cs2"),
        num_workers=4,
    )

    # Model
    print("Creating model...")
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
    ).to(device)

    # Train
    print("Training...")
    model = run_2stage_pipeline(model, loaders, cfg, device)

    # Evaluate
    print("Evaluating...")
    m1, s1 = evaluate_model(model, datasets["test_cs1"], device)
    m2, s2 = evaluate_model(model, datasets["test_cs2"], device)
    deg = float(np.mean(m2) / (np.mean(m1) + 1e-10))

    print(f"\n  CS1: X={m1[0]:.4f} Y={m1[1]:.4f} Z={m1[2]:.4f}  mean={np.mean(m1):.4f}")
    print(f"  CS2: X={m2[0]:.4f} Y={m2[1]:.4f} Z={m2[2]:.4f}  mean={np.mean(m2):.4f}")
    print(f"  Degradation: {deg:.2f}x")


if __name__ == "__main__":
    main()
