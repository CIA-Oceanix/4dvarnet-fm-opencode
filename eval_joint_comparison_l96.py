#!/usr/bin/env python3
"""L96 joint state-parameter estimation: S0/S1 benchmark with vanilla vs joint DA.

Runs the same cached test datasets used by DA baselines and the neural models,
then compares vanilla EnKF/ETKF/Strong-4DVar against their joint counterparts
that also estimate L96 parameters (F, c1, hx, eps + fast_weights, h fixed).
"""
import os
import sys
import json
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.lorenz96 import Lorenz96Config, make_l96_s0_s1_trainval
from evaluation.baselines import (
    EnKF, ETKF, Strong4DVar,
    JointEnKFL96, JointETKFL96, JointStrong4DVarL96,
)
from evaluation.metrics import param_rmse
from evaluation.run_l96 import evaluate_baseline, make_obs_j_indices

BASE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(BASE, "experiments")

_L96_ESTIMATED = ["F", "c1", "hx", "eps", "w1", "w2", "w3", "w4"]
_REF_FAST_WEIGHTS = [1.0, 1.0, 0.1, 0.1]  # w3/w4 default on S1 (J=2), matching JointETKFL96._ref_fw


def _true_param_vector(w, J):
    F = w["true_F"]
    c1 = w["true_c1"]
    hx = w["true_hx"]
    eps = w["true_eps"]
    fw = list(w["true_fast_weights"])
    if J < len(fw):
        fw = fw[:J]
    vec = [F, c1, hx, eps] + list(fw)
    while len(vec) < len(_L96_ESTIMATED):
        vec.append(_REF_FAST_WEIGHTS[len(vec) - 4])
    return np.array(vec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--enkf-inflation", type=float, default=2.0)
    parser.add_argument("--etkf-inflation", type=float, default=1.6)
    parser.add_argument("--da-window-steps", type=int, default=500)
    parser.add_argument("--obs-interval", type=int, default=100)
    parser.add_argument("--obs-j", type=int, default=2)
    parser.add_argument("--num-test-windows", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated subset of methods to run (default: all). "
                             "e.g. --methods EnKF,Joint-EnKF")
    parser.add_argument("--cases", type=str, default=None,
                        help="Comma-separated subset of cases to run (default: all). "
                             "e.g. --cases S0 to run only the S0 case.")
    parser.add_argument("--regenerate-data", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    NO = 8
    J_truth = 4
    obs_var_indices = make_obs_j_indices(NO, J_truth, args.obs_j)
    s1_state_dim = NO + NO * args.obs_j
    s1_obs_indices = list(range(s1_state_dim))

    base_cfg = Lorenz96Config(
        case=1, dt=0.001, T_max=3.0, obs_interval=args.obs_interval,
        R_var=0.5, B_var=2.0, seed=42, NO=NO, J=J_truth,
        h=1.0, hx=1.0, eps=0.1, F_true=8.0, F_da=8.0,
        obs_var_indices=obs_var_indices,
        fast_weights=[1.0, 1.0, 0.1, 0.1],
        randomize={"fast_weights": {"randomized": False, "biased": True, "bias": 0.0}},
    )
    print(f"Config: obs_j={args.obs_j}, obs_interval={args.obs_interval}, dws={args.da_window_steps}")

    # Reuse cached test datasets (same as DA baselines); generate if missing.
    ds_cache = os.path.join(EXP_DIR, f"l96_datasets_obsj{args.obs_j}_int{args.obs_interval}_nwin{args.num_test_windows}.pt")
    if os.path.exists(ds_cache) and not args.regenerate_data:
        print(f"  Reusing cached datasets from {ds_cache}")
        datasets = torch.load(ds_cache, weights_only=False, map_location="cpu")
    else:
        print("  Generating test datasets (may take a while)...")
        datasets = make_l96_s0_s1_trainval(
            base_cfg, num_train_windows=2, num_val_windows=2,
            num_test_windows=args.num_test_windows,
            param_noise=0.2, bias_range=(0.0, 0.2),
        )
        torch.save(datasets, ds_cache)
    print(f"  test_s0: {len(datasets['test_s0'])} windows, test_s1: {len(datasets['test_s1'])} windows")

    cases = [("S0", "test_s0", 1.6, 4), ("S1", "test_s1", 1.0, args.obs_j)]
    if args.cases:
        keep_cases = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in cases if c[0] in keep_cases]
        missing_cases = keep_cases - {c[0] for c in cases}
        if missing_cases:
            raise SystemExit(f"Unknown case(s): {sorted(missing_cases)}")

    # Joint-EnKF / Joint-ETKF tuned with state-consistent settings; Joint-ETKF
    # uses param_noise=0.03 + etkf_ridge=0.05 (matches the ETKF benchmark), and
    # Joint-EnKF mirrors the same ridge/noise for apples-to-apples S1 stability.
    method_factories = {
        "EnKF": lambda dyn, op, J: EnKF(dt=0.001, device=device, coupling_exponent=1.6,
                                         dynamics=dyn, obs_operator=op, NO=NO, J=J,
                                         N_ensemble=30, inflation=args.enkf_inflation),
        "Joint-EnKF": lambda dyn, op, J: JointEnKFL96(dt=0.001, device=device, coupling_exponent=1.6,
                                                       dynamics=dyn, obs_operator=op, NO=NO, J=J,
                                                       N_ensemble=30, inflation=args.enkf_inflation,
                                                       param_noise=0.03, etkf_ridge=0.05),
        "ETKF": lambda dyn, op, J: ETKF(dt=0.001, device=device, coupling_exponent=1.6,
                                         dynamics=dyn, obs_operator=op, NO=NO, J=J,
                                         N_ensemble=30, inflation=args.etkf_inflation),
        "Joint-ETKF": lambda dyn, op, J: JointETKFL96(dt=0.001, device=device, coupling_exponent=1.6,
                                                       dynamics=dyn, obs_operator=op, NO=NO, J=J,
                                                       N_ensemble=30, inflation=args.etkf_inflation,
                                                       param_noise=0.03, etkf_ridge=0.05),
        "Strong-4DVar": lambda dyn, op, J: Strong4DVar(dt=0.001, da_window_steps=args.da_window_steps,
                                                        device=device, coupling_exponent=1.6,
                                                        dynamics=dyn, obs_operator=op, max_iter=10, lr=0.2),
        "Joint-Strong-4DVar": lambda dyn, op, J: JointStrong4DVarL96(dt=0.001, da_window_steps=args.da_window_steps,
                                                                     device=device, coupling_exponent=1.6,
                                                                     dynamics=dyn, obs_operator=op, max_iter=10, lr=0.2,
                                                                     J=J),
    }
    if args.methods:
        keep = {m.strip() for m in args.methods.split(",") if m.strip()}
        missing = keep - set(method_factories)
        if missing:
            raise SystemExit(f"Unknown method(s): {sorted(missing)}")
        method_factories = {k: v for k, v in method_factories.items() if k in keep}

    from models.lorenz96_dynamics import Lorenz96Dynamics
    from evaluation.baselines import ObsOperator
    s0_dyn = Lorenz96Dynamics(dt=0.001, coupling_exponent=1.6)
    s0_obs_op = ObsOperator(NO + NO * J_truth, obs_var_indices)
    s1_dyn = Lorenz96Dynamics(dt=0.001, NO=NO, J=args.obs_j, h=1.0, hx=1.0, eps=0.1,
                              coupling_exponent=1.0)
    s1_obs_op = ObsOperator(s1_state_dim, s1_obs_indices)

    cfg_s0 = Lorenz96Config(param_bias=0.0, forcing_state_bias=0.0, T_max=3.0, seed=123,
                             obs_interval=args.obs_interval, obs_var_indices=obs_var_indices)
    cfg_s1 = Lorenz96Config(param_bias=0.15, forcing_state_bias=0.1, T_max=3.0, seed=131,
                             obs_interval=args.obs_interval, obs_var_indices=obs_var_indices)
    cfg_map = {"S0": cfg_s0, "S1": cfg_s1}

    results = {}
    for label, ds_key, coupling_exponent, J in cases:
        if ds_key not in datasets:
            continue
        ds = datasets[ds_key]
        cfg = cfg_map[label]
        print(f"\n{'=' * 90}")
        print(f"  {label} (coupling_exponent={coupling_exponent}, DA-J={J})")
        print(f"{'=' * 90}")

        if label == "S0":
            dyn, op = s0_dyn, s0_obs_op
        else:
            dyn, op = s1_dyn, s1_obs_op
        # Per-window true params for the joint filter initial guess.
        da_J = J_truth if label == "S0" else J

        case_results = {}
        for method_name, factory in method_factories.items():
            method = factory(dyn, op, J)
            (rmse_stats, expvar_stats, es_stats), bl_results = evaluate_baseline(
                method, ds, cfg, device, return_trajs=True, batch_size=args.batch_size, da_J=da_J)
            print(f"  [{method_name}] finite windows: {len(bl_results)}/{len(ds)}")
            mean_rmse = rmse_stats[0]
            ev_arr = expvar_stats[0]
            es_arr = es_stats[0]

            entry = {
                "state_rmse": {
                    "slow": float(np.mean(mean_rmse[:NO])),
                    "obs_fast": float(np.mean(mean_rmse[NO:])),
                    "mean": float(np.mean(mean_rmse)),
                },
                "ev": {
                    "slow": float(np.mean(ev_arr[:NO])),
                    "obs_fast": float(np.mean(ev_arr[NO:])),
                    "mean": float(np.mean(ev_arr)),
                },
                "es": {
                    "slow": float(np.mean(es_arr[:NO])) if es_arr is not None else None,
                    "obs_fast": float(np.mean(es_arr[NO:])) if es_arr is not None else None,
                    "mean": float(np.mean(es_arr)) if es_arr is not None else None,
                },
            }

            is_joint = method_name.startswith("Joint-")
            if is_joint and bl_results[0].params is not None:
                all_pred = np.stack([r.params for r in bl_results], axis=0)  # (W, T, 8)
                T_i = all_pred.shape[1]
                all_true = np.stack([_true_param_vector(w, J) for w in ds], axis=0)
                all_true = np.repeat(all_true[:, np.newaxis, :], T_i, axis=1)
                prmse = param_rmse(all_pred.reshape(-1, len(_L96_ESTIMATED)),
                                   all_true.reshape(-1, len(_L96_ESTIMATED)))
                entry["param_rmse"] = {k: float(prmse[i]) for i, k in enumerate(_L96_ESTIMATED)}
            else:
                entry["param_rmse"] = None

            case_results[method_name] = entry
            print(f"  {method_name:<20} rmse={(entry['state_rmse']['mean']):.4f} "
                  f"ev={(entry['ev']['mean']):.4f}", end="")
            if entry["param_rmse"]:
                print(f"  | param mean={np.mean(list(entry['param_rmse'].values())):.4f}")
            else:
                print()

        results[label] = case_results

    out_path = os.path.join(EXP_DIR, "l96_joint_comparison.json")
    # Merge into any existing results (e.g. when re-running only a method subset),
    # preserving entries for methods/cases not run this invocation.
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    for label, case_results in results.items():
        merged = dict(existing.get(label, {}))
        merged.update(case_results)
        existing[label] = merged
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    print(f"\nSaved L96 joint comparison to {out_path}")


if __name__ == "__main__":
    main()
