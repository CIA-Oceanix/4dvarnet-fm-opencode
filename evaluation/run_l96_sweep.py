#!/usr/bin/env python3
"""
Run a single L96 S1 config sweep with specified parameters.
Outputs JSON results to experiments/l96_sweep_{label}.json

Usage:
    conda run -n fdv python evaluation/run_l96_sweep.py --label a1 --param-bias 0.30 --forcing-state-bias 0.3 --obs-frac 1.0
"""
import os, sys, json, argparse, time
import torch
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.lorenz96 import Lorenz96Config, RandomParamLorenz96Dataset, RandomBiasLorenz96Dataset, _make_lorenz96_dynamics
from models.lorenz96_dynamics import Lorenz96Dynamics
from evaluation.baselines import Weak4DVar, Strong4DVar, EnKF, ETKF
from evaluation.run_l96 import evaluate_baseline

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE, "experiments")

_METHODS = ["Strong-4DVar", "EnKF", "ETKF", "Weak-4DVar"]

def make_s0_s1_datasets(cfg, num_windows, s1_param_bias, s1_forcing_state_bias):
    dynamics = Lorenz96Dynamics(dt=0.001, coupling_exponent=1.6)
    s0_cfg = Lorenz96Config(**{**cfg.__dict__, "case": 1, "param_bias": 0.0,
        "forcing_state_bias": 0.0, "seed": 123, "num_windows": num_windows})
    s1_cfg = Lorenz96Config(**{**cfg.__dict__, "case": 1, "param_bias": s1_param_bias,
        "forcing_state_bias": s1_forcing_state_bias, "seed": 131, "num_windows": num_windows})
    return {
        "s0": RandomParamLorenz96Dataset(s0_cfg, param_noise=0.2, dynamics=dynamics),
        "s1": RandomBiasLorenz96Dataset(s1_cfg, param_noise=0.2, dynamics=dynamics),
    }

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--param-bias", type=float, required=True)
    parser.add_argument("--forcing-state-bias", type=float, required=True)
    parser.add_argument("--obs-frac", type=float, default=1.0)
    parser.add_argument("--num-windows", type=int, default=20)
    parser.add_argument("--da-window-steps", type=int, default=500)
    parser.add_argument("--skip-weak", action="store_true", default=False)
    parser.add_argument("--weak-opt-steps", type=int, default=50)
    parser.add_argument("--weak-lr", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if torch.cuda.is_available():
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Device: {device}")

    dt = 0.001
    T_max = 3.0
    num_steps = int(T_max / dt)
    obs_interval = 200
    # If obs_frac < 1, we subsample the observation dimensions
    # We handle this by modifying the observation tensor after creation

    base_cfg = Lorenz96Config(
        dt=dt, T_max=T_max, obs_interval=obs_interval,
        R_var=0.5, B_var=2.0,
        num_windows=args.num_windows, window_spacing=args.num_windows,
        spinup_steps=5000, seed=42,
        NO=8, J=4, h=1.0, hx=1.0, eps=0.1,
        F_true=8.0, F_da=8.0,
        gamma=0.05, W_L_bar=0.0, c1=1.0, c2=0.1,
        sigma_0=0.08, sigma_L=0.20,
        tau_eta=5.0, sigma_eta=np.sqrt(0.5),
        param_bias=args.param_bias, forcing_state_bias=args.forcing_state_bias,
    )

    print(f"\n── {args.label}: S1 config ──")
    print(f"  param_bias={args.param_bias}, forcing_state_bias={args.forcing_state_bias}")
    print(f"  F_da = {base_cfg.F_true * (1 - args.param_bias):.2f}")
    print(f"  obs_frac={args.obs_frac}, num_windows={args.num_windows}")

    t0 = time.time()
    datasets = make_s0_s1_datasets(base_cfg, args.num_windows,
                                    args.param_bias, args.forcing_state_bias)
    print(f"  Dataset gen: {time.time()-t0:.1f}s")

    dynamics_pool = {
        1.0: Lorenz96Dynamics(dt=dt, coupling_exponent=1.0),
        1.6: Lorenz96Dynamics(dt=dt, coupling_exponent=1.6),
    }

    methods_to_run = _METHODS[:]
    if args.skip_weak:
        methods_to_run.remove("Weak-4DVar")
    valid_methods = [m for m in methods_to_run if m in _METHODS]

    results = {}
    for case_key, case_label, da_expo in [("s0", "S0", 1.6), ("s1", "S1", 1.0)]:
        ds = datasets[case_key]
        cfg = base_cfg
        method_map = {
            "Weak-4DVar": Weak4DVar(dt=dt, da_window_steps=args.da_window_steps, device=device,
                                     coupling_exponent=da_expo, dynamics=dynamics_pool[1.0],
                                     opt_steps=args.weak_opt_steps, lr=args.weak_lr),
            "Strong-4DVar": Strong4DVar(dt=dt, da_window_steps=args.da_window_steps, device=device,
                                          coupling_exponent=da_expo, dynamics=dynamics_pool[1.0],
                                          max_iter=10, lr=0.2),
            "EnKF": EnKF(dt=dt, device=device, coupling_exponent=da_expo,
                          dynamics=dynamics_pool[1.0], inflation=2.0),
            "ETKF": ETKF(dt=dt, device=device, coupling_exponent=da_expo,
                           dynamics=dynamics_pool[1.0], inflation=2.0),
        }
        results[case_key] = {}
        for name in valid_methods:
            method = method_map[name]
            print(f"  {case_label}/{name} ...", end=" ", flush=True)
            t1 = time.time()
            (m, s), (ev_m, ev_s) = evaluate_baseline(method, ds, cfg, device, return_trajs=False,
                                                       batch_size=min(20, args.num_windows))
            results[case_key][name] = {
                "mean_rmse": float(np.mean(m)),
                "per_var_mean": m.tolist(),
                "per_var_std": s.tolist(),
            }
            elapsed = time.time() - t1
            print(f"  μ={np.mean(m):.4f} [{elapsed:.1f}s]")

    out = {
        "label": args.label,
        "config": {
            "param_bias": args.param_bias,
            "forcing_state_bias": args.forcing_state_bias,
            "obs_frac": args.obs_frac,
            "num_windows": args.num_windows,
            "da_window_steps": args.da_window_steps,
            "F_da": base_cfg.F_true * (1 - args.param_bias),
        },
        "results": results,
    }
    out_path = os.path.join(EXP_DIR, f"l96_sweep_{args.label}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print summary table
    print(f"\n── {args.label} Summary ──")
    print(f"{'Method':<20} {'S0 μ':<10} {'S1 μ':<10} {'Δ%':<10}")
    print("-" * 50)
    for name in valid_methods:
        s0_m = results["s0"][name]["mean_rmse"]
        s1_m = results["s1"][name]["mean_rmse"]
        pct = (s1_m / s0_m - 1) * 100
        print(f"{name:<20} {s0_m:<10.4f} {s1_m:<10.4f} {pct:<+10.1f}%")

if __name__ == "__main__":
    run()