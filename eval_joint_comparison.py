#!/usr/bin/env python3
"""Joint state-parameter estimation: S0/S1 benchmark with vanilla vs joint DA."""
import os, sys, json, argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.lorenz63 import Lorenz63Config, make_mixed_datasets
from evaluation.baselines import (
    Weak4DVar, Strong4DVar, EnKF, ETKF,
    JointWeak4DVar, JointStrong4DVar, JointEnKF, JointETKF,
)
from evaluation.metrics import rmse, param_rmse
from evaluation.run import evaluate_baseline

BASE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(BASE, "experiments")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--enkf-inflation", type=float, default=2.0)
    parser.add_argument("--etkf-inflation", type=float, default=2.0)
    parser.add_argument("--da-window-steps", type=int, default=50)
    parser.add_argument("--r-var", type=float, default=0.5)
    parser.add_argument("--obs-interval", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    base_cfg = Lorenz63Config(
        dt=0.01, T_max=3.0, obs_interval=args.obs_interval,
        R_var=args.r_var, B_var=2.0,
        num_windows=2000, window_spacing=2000,
        spinup_steps=10000, seed=42,
        sigma_true=10.0, rho_true=28.0, beta_true=2.6666666666666665,
        gamma=0.05, W_L_bar=0.0, c1=1.0, c2=0.1,
        sigma_0=0.08, sigma_L=0.20,
        tau_eta=5.0, sigma_eta=0.7071067811865476,
        param_bias=0.0, forcing_state_bias=0.0,
    )
    print(f"Config: a_truth=1.6, R_var={args.r_var}, obs_interval={args.obs_interval}, dws={args.da_window_steps}")

    datasets = make_mixed_datasets(base_cfg, num_test_windows=10,
                                   include_s1_test=True, param_noise=0.2)

    # Per-case coupling exponents matching evaluate_all.py convention
    cases = [
        ("S0", "test_s0", 1.6),
        ("S1", "test_s1", 1.0),
    ]

    method_factories = {
        "Weak-4DVar":       lambda ce: Weak4DVar(dt=0.01, da_window_steps=args.da_window_steps, device=device, coupling_exponent=ce, opt_steps=200, Q_var=1.0),
        "Joint-Weak-4DVar": lambda ce: JointWeak4DVar(dt=0.01, da_window_steps=args.da_window_steps, device=device, coupling_exponent=ce, opt_steps=200, Q_var=1.0, P_var=1.0),
        "Strong-4DVar":     lambda ce: Strong4DVar(dt=0.01, da_window_steps=args.da_window_steps, device=device, coupling_exponent=ce, max_iter=50),
        "Joint-Strong-4DVar": lambda ce: JointStrong4DVar(dt=0.01, da_window_steps=args.da_window_steps, device=device, coupling_exponent=ce, max_iter=50, P_var=1.0),
        "EnKF":             lambda ce: EnKF(dt=0.01, device=device, coupling_exponent=ce, N_ensemble=30, inflation=args.enkf_inflation),
        "Joint-EnKF":       lambda ce: JointEnKF(dt=0.01, device=device, coupling_exponent=ce, N_ensemble=30, inflation=args.enkf_inflation),
        "ETKF":             lambda ce: ETKF(dt=0.01, device=device, coupling_exponent=ce, N_ensemble=30, inflation=args.etkf_inflation),
        "Joint-ETKF":       lambda ce: JointETKF(dt=0.01, device=device, coupling_exponent=ce, N_ensemble=30, inflation=args.etkf_inflation),
    }

    cfg_s0 = Lorenz63Config(param_bias=0.0, forcing_state_bias=0.0, T_max=3.0, seed=123)
    cfg_s1 = Lorenz63Config(param_bias=0.15, forcing_state_bias=0.1, T_max=3.0, seed=131)
    cfg_map = {"S0": cfg_s0, "S1": cfg_s1}

    results = {}
    for label, ds_key, coupling_exponent in cases:
        if ds_key not in datasets:
            continue
        ds = datasets[ds_key]
        cfg = cfg_map[label]
        print(f"\n{'=' * 80}")
        print(f"  {label} (coupling_exponent={coupling_exponent})")
        print(f"{'=' * 80}")

        case_results = {}
        for method_name, factory in method_factories.items():
            method = factory(coupling_exponent)
            stats, bl_results = evaluate_baseline(
                method, ds, cfg, device, return_trajs=True, batch_size=200)
            mean_rmse = stats[0]

            entry = {
                "state_rmse": {
                    "X": float(mean_rmse[0]),
                    "Y": float(mean_rmse[1]),
                    "Z": float(mean_rmse[2]),
                    "mean": float(np.mean(mean_rmse)),
                },
            }

            is_joint = method_name.startswith("Joint-")
            if is_joint and bl_results[0].params is not None:
                all_pred_params = np.stack([r.params for r in bl_results], axis=0)
                num_steps_i = all_pred_params.shape[1]
                all_true_params = np.stack([
                    np.array([w["true_sigma"], w["true_rho"], w["true_beta"], w["true_c1"]])
                    for w in ds
                ], axis=0)
                all_true_params = np.repeat(
                    all_true_params[:, np.newaxis, :], num_steps_i, axis=1)
                prmse = param_rmse(all_pred_params.reshape(-1, 4), all_true_params.reshape(-1, 4))
                entry["param_rmse"] = {
                    "sigma": float(prmse[0]),
                    "rho": float(prmse[1]),
                    "beta": float(prmse[2]),
                    "c1": float(prmse[3]),
                }
            else:
                entry["param_rmse"] = None

            case_results[method_name] = entry
            print(f"  {method_name:<20} state: X={mean_rmse[0]:.4f} Y={mean_rmse[1]:.4f} Z={mean_rmse[2]:.4f} mean={np.mean(mean_rmse):.4f}", end="")
            if entry["param_rmse"]:
                print(f"  | params: s={prmse[0]:.4f} r={prmse[1]:.4f} b={prmse[2]:.4f} c1={prmse[3]:.4f}")
            else:
                print()

        results[label] = case_results

    # Build comparison table
    print(f"\n{'=' * 120}")
    print(f"  {'Case':<6} {'Method':<20} {'State RMSE (X/Y/Z = mean)':<40} {'Param RMSE (s/r/b/c1)':<40} {'Ratio':<10}")
    print(f"  {'-' * 118}")

    for label, ds_key, coupling_exponent in cases:
        if ds_key not in datasets:
            continue
        case_results = results[label]
        base_names = ["Weak-4DVar", "Strong-4DVar", "EnKF", "ETKF"]
        vanilla_states = {}
        for bn in base_names:
            if bn in case_results:
                vanilla_states[bn] = case_results[bn]["state_rmse"]["mean"]

        for method_name, entry in case_results.items():
            s = entry["state_rmse"]
            state_str = f"{s['X']:.4f}/{s['Y']:.4f}/{s['Z']:.4f} = {s['mean']:.4f}"
            if entry["param_rmse"]:
                p = entry["param_rmse"]
                param_str = f"{p['sigma']:.4f}/{p['rho']:.4f}/{p['beta']:.4f}/{p['c1']:.4f}"
            else:
                param_str = "N/A"

            if method_name.startswith("Joint-"):
                vn = method_name.replace("Joint-", "")
                v_mean = vanilla_states.get(vn, 0)
                ratio = s["mean"] / v_mean if v_mean > 0 else 0
                ratio_str = f"{ratio:.4f}"
            else:
                ratio_str = "N/A"

            print(f"  {label:<6} {method_name:<20} {state_str:<40} {param_str:<40} {ratio_str:<10}")
    print(f"  {'-' * 118}")

    # Save
    out_path = os.path.join(EXP_DIR, "joint_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved joint comparison to {out_path}")


if __name__ == "__main__":
    main()
