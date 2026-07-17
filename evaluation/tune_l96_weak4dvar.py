#!/usr/bin/env python3
"""Tune Weak-4DVar opt_steps/lr/da_window_steps on L96 S0+S1 (b2 config)."""
import os, sys, json, time, itertools
import torch
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.lorenz96 import Lorenz96Config, RandomParamLorenz96Dataset, RandomBiasLorenz96Dataset
from models.lorenz96_dynamics import Lorenz96Dynamics
from evaluation.baselines import Weak4DVar
from evaluation.run_l96 import evaluate_baseline

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE, "experiments")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Device: {device}")

    dt = 0.001
    T_max = 3.0
    obs_interval = 200
    num_windows = 3

    base_cfg = Lorenz96Config(
        dt=dt, T_max=T_max, obs_interval=obs_interval,
        R_var=0.5, B_var=2.0,
        num_windows=num_windows, window_spacing=num_windows,
        spinup_steps=5000, seed=42,
        NO=8, J=4, h=1.0, hx=1.0, eps=0.1,
        F_true=8.0, F_da=8.0,
        gamma=0.05, W_L_bar=0.0, c1=1.0, c2=0.1,
        sigma_0=0.08, sigma_L=0.20,
        tau_eta=5.0, sigma_eta=np.sqrt(0.5),
        param_bias=0.0, forcing_state_bias=0.0,
    )

    t0 = time.time()
    truth_dynamics = Lorenz96Dynamics(dt=dt, coupling_exponent=1.6)
    # S0: correct dynamics
    ds_s0 = RandomParamLorenz96Dataset(base_cfg, param_noise=0.2, dynamics=truth_dynamics)
    # S1: same dataset (b2 — single-scale mismatch)
    ds_s1 = RandomBiasLorenz96Dataset(base_cfg, param_noise=0.2, dynamics=truth_dynamics)
    print(f"  Dataset gen: {time.time()-t0:.1f}s")

    # Dynamics pool: correct multi-scale + single-scale
    ms_dynamics = Lorenz96Dynamics(dt=dt, coupling_exponent=1.6)
    ss_dynamics = Lorenz96Dynamics(dt=dt, NO=40, J=0, h=0.0, hx=0.0, coupling_exponent=1.0)

    dws_list = [100, 250, 500]
    opt_steps_list = [500, 1000, 2000]
    lr_list = [0.05, 0.1, 0.2]

    grid = list(itertools.product(opt_steps_list, lr_list, dws_list))
    results = []

    # Test S0 first (multi-scale dynamics)
    print(f"\n--- S0 (multi-scale dynamics) ---")
    for opt_steps, lr, dws in grid:
        print(f"  opt_steps={opt_steps}, lr={lr}, dws={dws} ...", end=" ", flush=True)
        method = Weak4DVar(dt=dt, da_window_steps=dws, device=device,
                           coupling_exponent=1.6, dynamics=ms_dynamics,
                           opt_steps=opt_steps, lr=lr)
        t1 = time.time()
        (m, s), (ev_m, ev_s) = evaluate_baseline(method, ds_s0, base_cfg, device, return_trajs=False, batch_size=3)
        mu = float(np.mean(m))
        elapsed = time.time() - t1
        print(f"  mu={mu:.4f} [{elapsed:.1f}s]")
        results.append({
            "case": "S0", "opt_steps": opt_steps, "lr": lr, "da_window_steps": dws,
            "rmse_mean": mu, "rmse_per_var": m.tolist(), "rmse_std": s.tolist(),
            "time_s": round(elapsed, 1),
        })

    # Test S1 (single-scale dynamics)
    print(f"\n--- S1 (single-scale dynamics) ---")
    for opt_steps, lr, dws in grid:
        print(f"  opt_steps={opt_steps}, lr={lr}, dws={dws} ...", end=" ", flush=True)
        method = Weak4DVar(dt=dt, da_window_steps=dws, device=device,
                           coupling_exponent=1.0, dynamics=ss_dynamics,
                           opt_steps=opt_steps, lr=lr)
        t1 = time.time()
        (m, s), (ev_m, ev_s) = evaluate_baseline(method, ds_s1, base_cfg, device, return_trajs=False, batch_size=3)
        mu = float(np.mean(m))
        elapsed = time.time() - t1
        print(f"  mu={mu:.4f} [{elapsed:.1f}s]")
        results.append({
            "case": "S1", "opt_steps": opt_steps, "lr": lr, "da_window_steps": dws,
            "rmse_mean": mu, "rmse_per_var": m.tolist(), "rmse_std": s.tolist(),
            "time_s": round(elapsed, 1),
        })

    # Best per case
    best_s0 = min([r for r in results if r["case"] == "S0"], key=lambda r: r["rmse_mean"])
    best_s1 = min([r for r in results if r["case"] == "S1"], key=lambda r: r["rmse_mean"])
    print(f"\n=== Best S0: opt_steps={best_s0['opt_steps']}, lr={best_s0['lr']}, dws={best_s0['da_window_steps']} => RMSE={best_s0['rmse_mean']:.4f} ===")
    print(f"=== Best S1: opt_steps={best_s1['opt_steps']}, lr={best_s1['lr']}, dws={best_s1['da_window_steps']} => RMSE={best_s1['rmse_mean']:.4f} ===")

    # Print table
    print(f"\n{'Case':<5} {'opt_steps':<10} {'lr':<6} {'dws':<6} {'RMSE':<8} {'Time':<8}")
    print("-" * 45)
    for r in results:
        print(f"{r['case']:<5} {r['opt_steps']:<10} {r['lr']:<6} {r['da_window_steps']:<6} {r['rmse_mean']:<8.4f} {r['time_s']:<8.1f}")

    out_path = os.path.join(EXP_DIR, "l96_weak4dvar_tune.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "best_s0": best_s0, "best_s1": best_s1}, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()