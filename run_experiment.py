#!/usr/bin/env python3
import torch
import numpy as np
from data.lorenz63 import Lorenz63Config, make_datasets
from data.dataloader import make_dataloaders
from models.solver import TweedieSolver
from training.stage1 import train_stage1
from training.stage2 import train_stage2
from evaluation.baselines import Weak4DVar, Strong4DVar, EnKF
from evaluation.experiment import run_baseline
from evaluation.metrics import print_metrics_table


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    cfg = Lorenz63Config(
        case=1,
        dt=0.01,
        T_max=5.0,
        obs_interval=20,
        R_var=0.5,
        B_var=2.0,
        Q_var=0.05,
        param_bias=0.05,
        num_windows=2000,
        window_spacing=2000,
        spinup_steps=10000,
        seed=42,
    )

    print("\n" + "=" * 60)
    print("STEP 1: Generating datasets (train/val/test)")
    print("=" * 60)
    datasets = make_datasets(cfg)
    loaders = make_dataloaders(datasets, batch_size=32)
    print(f"  Train windows: {len(datasets['train'])}")
    print(f"  Val windows:   {len(datasets['val'])}")
    print(f"  Test CS1:      {len(datasets['test_cs1'])}")
    print(f"  Test CS2:      {len(datasets['test_cs2'])}")

    print("\n" + "=" * 60)
    print("STEP 2: Training 4DVarNet-FM (Stage 1: Mean Estimator)")
    print("=" * 60)
    model = TweedieSolver(
        state_dim=3,
        hidden_channels=[64, 128, 256],
        time_emb_dim=64,
        use_obs=True,
        use_energy=True,
        nu=1.0,
        K_inner=5,
        N_outer=10,
        dropout=0.1,
    ).to(device)

    model = train_stage1(
        model, loaders["train"], loaders["val"],
        epochs=200, lr=1e-3, device=device,
    )

    print("\n" + "=" * 60)
    print("STEP 3: Training 4DVarNet-FM (Stage 2: Full Solver)")
    print("=" * 60)
    model = train_stage2(
        model, loaders["train"], loaders["val"],
        epochs=400, lr=1e-3, device=device,
    )

    print("\n" + "=" * 60)
    print("STEP 4: Running baselines on Case Study 1 (noise-free)")
    print("=" * 60)
    ds_cs1 = datasets["test_cs1"]
    cfg_cs1 = Lorenz63Config(case=1, param_bias=0.0, seed=123)

    w4d = Weak4DVar(dt=0.01, device=device)
    s4d = Strong4DVar(dt=0.01, device=device)
    enkf = EnKF(dt=0.01, device=device)

    cs1_results = {}
    w = ds_cs1[0]
    cs1_results["Weak-4DVar"] = run_baseline(w4d, ds_cs1, cfg_cs1, device)
    cs1_results["Strong-4DVar"] = run_baseline(s4d, ds_cs1, cfg_cs1, device)
    cs1_results["EnKF"] = run_baseline(enkf, ds_cs1, cfg_cs1, device)

    print_metrics_table(cs1_results, "CASE STUDY 1: Noise-free forcings & parameters")

    print("\n" + "=" * 60)
    print("STEP 5: Running baselines on Case Study 2 (noisy)")
    print("=" * 60)
    ds_cs2 = datasets["test_cs2"]
    cfg_cs2 = Lorenz63Config(case=2, param_bias=0.05, seed=123)

    cs2_results = {}
    cs2_results["Weak-4DVar"] = run_baseline(w4d, ds_cs2, cfg_cs2, device)
    cs2_results["Strong-4DVar"] = run_baseline(s4d, ds_cs2, cfg_cs2, device)
    cs2_results["EnKF"] = run_baseline(enkf, ds_cs2, cfg_cs2, device)

    print_metrics_table(cs2_results, "CASE STUDY 2: Noisy forcings & biased parameters")

    print("\n" + "=" * 60)
    print("DEGRADATION SUMMARY")
    print("=" * 60)
    print(f"{'Method':<20} {'CS1 RMSE':<12} {'CS2 RMSE':<12} {'Degradation':<12}")
    print(f"{'-' * 56}")
    for name in cs1_results:
        r1 = np.mean(cs1_results[name].rmse)
        r2 = np.mean(cs2_results[name].rmse)
        deg = r2 / (r1 + 1e-10)
        print(f"{name:<20} {r1:<12.4f} {r2:<12.4f} {deg:<12.2f}x")

    print("\nDone.")


if __name__ == "__main__":
    main()
