#!/usr/bin/env python3
import os
import torch
import numpy as np
from data.lorenz63 import Lorenz63Config, make_mixed_datasets
from data.dataloader import make_dataloaders
from models.solver import TweedieSolver
from training.stage1 import train_stage1
from training.stage2 import train_stage2
from evaluation.baselines import Weak4DVar, Strong4DVar, EnKF
from evaluation.metrics import rmse


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


def evaluate_baseline(method, dataset, cfg, device):
    sig, rho, bet = cfg.da_params
    rmse_list = []
    all_sq_err = []
    all_ref = []
    for i in range(len(dataset)):
        w = dataset[i]
        obs = w["obs"]
        mask = w["obs_mask"]
        truth = w["true_state"]
        if cfg.use_corrupted_forcing:
            force = w["forcing_corrupted"]
        else:
            force = w["forcing_true"]
        result = method.assimilate(obs, mask, force, truth, sigma=sig, rho=rho, beta=bet)
        analysis = result.trajectory
        ref = truth.numpy()
        all_sq_err.append((analysis - ref) ** 2)
        all_ref.append(ref)
        rmse_list.append(result.rmse)
    all_rmse = np.stack(rmse_list, axis=0)
    all_sq_err = np.concatenate(all_sq_err, axis=0)
    all_ref = np.concatenate(all_ref, axis=0)
    pooled_mse = np.mean(all_sq_err, axis=0)
    pooled_var = np.var(all_ref, axis=0)
    pooled_var = np.maximum(pooled_var, 1e-12)
    expvar = 1 - pooled_mse / pooled_var
    expvar_stats = (expvar, np.zeros_like(expvar))
    return (np.mean(all_rmse, axis=0), np.std(all_rmse, axis=0)), expvar_stats


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    cfg = Lorenz63Config(
        case=1,
        param_bias=0.15,
        forcing_state_bias=0.15,
        forcing_coupling="quartic",
        num_windows=2000,
        spinup_steps=10000,
        seed=42,
    )

    print("\n" + "=" * 60)
    print("STEP 1: Generating datasets (train/val/test)")
    print("=" * 60)
    datasets = make_mixed_datasets(cfg)
    print(f"  Train CS1:     {len(datasets['train_cs1'])} windows")
    print(f"  Train CS2:     {len(datasets['train_cs2'])} windows")
    print(f"  Val CS1:       {len(datasets['val_cs1'])} windows")
    print(f"  Val CS2:       {len(datasets['val_cs2'])} windows")
    print(f"  Test CS1:      {len(datasets['test_cs1'])} windows")
    print(f"  Test CS2:      {len(datasets['test_cs2'])} windows")

    loaders = make_dataloaders(datasets, batch_size=32)

    print("\n" + "=" * 60)
    print("STEP 2: Creating/loading 4DVarNet-FM model")
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

    if os.path.exists("checkpoint_stage1.pt") and os.path.exists("checkpoint_stage2.pt"):
        print("  Found checkpoint_stage2.pt, loading model...")
        model.load_state_dict(torch.load("checkpoint_stage2.pt", map_location=device))
    else:
        print("\n" + "=" * 60)
        print("STEP 3: Training 4DVarNet-FM (Stage 1: Mean Estimator)")
        print("=" * 60)
        model = train_stage1(
            model, loaders["train"], loaders["val"],
            epochs=200, lr=1e-3, device=device,
        )
        torch.save(model.state_dict(), "checkpoint_stage1.pt")
        print(f"  Saved checkpoint_stage1.pt")

        print("\n" + "=" * 60)
        print("STEP 4: Training 4DVarNet-FM (Stage 2: Full Solver)")
        print("=" * 60)
        model = train_stage2(
            model, loaders["train"], loaders["val"],
            epochs=400, lr=1e-3, device=device,
        )
        torch.save(model.state_dict(), "checkpoint_stage2.pt")
        print(f"  Saved checkpoint_stage2.pt")

    print("\n" + "=" * 60)
    print("STEP 5: Evaluating 4DVarNet-FM")
    print("=" * 60)
    fm_cs1_mean, fm_cs1_std = evaluate_model(model, datasets["test_cs1"], device)
    fm_cs2_mean, fm_cs2_std = evaluate_model(model, datasets["test_cs2"], device)
    print(f"  4DVarNet-FM CS1: X={fm_cs1_mean[0]:.4f}+-{fm_cs1_std[0]:.4f} "
          f"Y={fm_cs1_mean[1]:.4f}+-{fm_cs1_std[1]:.4f} "
          f"Z={fm_cs1_mean[2]:.4f}+-{fm_cs1_std[2]:.4f}")
    print(f"  4DVarNet-FM CS2: X={fm_cs2_mean[0]:.4f}+-{fm_cs2_std[0]:.4f} "
          f"Y={fm_cs2_mean[1]:.4f}+-{fm_cs2_std[1]:.4f} "
          f"Z={fm_cs2_mean[2]:.4f}+-{fm_cs2_std[2]:.4f}")

    print("\n" + "=" * 60)
    print("STEP 6: Evaluating baselines on Case Study 1")
    print("=" * 60)
    cfg_cs1 = Lorenz63Config(case=1, param_bias=0.0, seed=123)
    ds_cs1 = datasets["test_cs1"]

    w4d = Weak4DVar(dt=0.01, device=device)
    s4d = Strong4DVar(dt=0.01, device=device)
    enkf = EnKF(dt=0.01, device=device)

    bl_cs1 = {}
    bl_cs1["Weak-4DVar"] = evaluate_baseline(w4d, ds_cs1, cfg_cs1, device)[0]
    bl_cs1["Strong-4DVar"] = evaluate_baseline(s4d, ds_cs1, cfg_cs1, device)[0]
    bl_cs1["EnKF"] = evaluate_baseline(enkf, ds_cs1, cfg_cs1, device)[0]

    print("\n" + "=" * 60)
    print("STEP 7: Evaluating baselines on Case Study 2")
    print("=" * 60)
    cfg_cs2 = Lorenz63Config(
        case=2, param_bias=0.15, forcing_state_bias=0.15,
        forcing_coupling="quartic", seed=124,
    )
    ds_cs2 = datasets["test_cs2"]

    bl_cs2 = {}
    bl_cs2["Weak-4DVar"] = evaluate_baseline(w4d, ds_cs2, cfg_cs2, device)[0]
    bl_cs2["Strong-4DVar"] = evaluate_baseline(s4d, ds_cs2, cfg_cs2, device)[0]
    bl_cs2["EnKF"] = evaluate_baseline(enkf, ds_cs2, cfg_cs2, device)[0]

    print("\n" + "=" * 60)
    print("DEGRADATION ANALYSIS (200 windows each)")
    print("=" * 60)
    print(f"{'Method':<20} {'CS1 RMSE':<36} {'CS2 RMSE':<36} {'Degradation':<12}")
    print("-" * 104)
    for name in ["Weak-4DVar", "Strong-4DVar", "EnKF", "4DVarNet-FM"]:
        if name == "4DVarNet-FM":
            m1, s1 = fm_cs1_mean, fm_cs1_std
            m2, s2 = fm_cs2_mean, fm_cs2_std
        else:
            m1, s1 = bl_cs1[name]
            m2, s2 = bl_cs2[name]
        cs1_str = f"{m1[0]:.3f}+-{s1[0]:.3f} {m1[1]:.3f}+-{s1[1]:.3f} {m1[2]:.3f}+-{s1[2]:.3f}"
        cs2_str = f"{m2[0]:.3f}+-{s2[0]:.3f} {m2[1]:.3f}+-{s2[1]:.3f} {m2[2]:.3f}+-{s2[2]:.3f}"
        deg = np.mean(m2) / (np.mean(m1) + 1e-10)
        print(f"{name:<20} {cs1_str:<36} {cs2_str:<36} {deg:.2f}x")

    print("=" * 60)
    print("\nDone.")


if __name__ == "__main__":
    main()
