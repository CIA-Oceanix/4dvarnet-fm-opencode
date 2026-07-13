#!/usr/bin/env python3
"""
Run a single L96 S1 config sweep with support for mismatched dynamics and rectangular observation operator.
Outputs JSON results to experiments/l96_sweep_{label}.json
"""
import os, sys, json, argparse, time
import torch
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.lorenz96 import Lorenz96Config, RandomParamLorenz96Dataset, RandomBiasLorenz96Dataset
from models.lorenz96_dynamics import Lorenz96Dynamics
from evaluation.baselines import Weak4DVar, Strong4DVar, EnKF, ETKF, ObsOperator
from evaluation.run_l96 import evaluate_baseline

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE, "experiments")

_METHODS = ["EnKF", "ETKF", "Strong-4DVar", "Weak-4DVar"]


def make_obs_j_indices(NO, J_truth, J_obs):
    if J_obs is None or J_obs >= J_truth:
        return None
    X_idx = list(range(NO))
    Y_idx = []
    for k in range(NO):
        for j in range(J_obs):
            Y_idx.append(NO + k * J_truth + j)
    return X_idx + Y_idx


def make_s0_s1_datasets(cfg, num_windows, s1_param_bias, s1_forcing_state_bias,
                         obs_var_indices, s1_seed=131):
    s0_cfg = Lorenz96Config(**{**cfg.__dict__, "case": 1, "param_bias": 0.0,
        "forcing_state_bias": 0.0, "seed": 123, "num_windows": num_windows,
        "obs_var_indices": obs_var_indices})
    s1_cfg = Lorenz96Config(**{**cfg.__dict__, "case": 1, "param_bias": s1_param_bias,
        "forcing_state_bias": s1_forcing_state_bias, "seed": s1_seed, "num_windows": num_windows,
        "obs_var_indices": obs_var_indices})
    dynamics = Lorenz96Dynamics(dt=cfg.dt, NO=cfg.NO, J=cfg.J,
                                h=cfg.h, hx=cfg.hx, eps=cfg.eps,
                                coupling_exponent=1.6)
    return {
        "s0": RandomParamLorenz96Dataset(s0_cfg, param_noise=0.2, dynamics=dynamics),
        "s1": RandomBiasLorenz96Dataset(s1_cfg, param_noise=0.2, dynamics=dynamics),
    }


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--param-bias", type=float, default=0.0)
    parser.add_argument("--forcing-state-bias", type=float, default=0.0)
    parser.add_argument("--num-windows", type=int, default=20)
    parser.add_argument("--da-window-steps", type=int, default=500)
    parser.add_argument("--skip-weak", action="store_true", default=False)
    parser.add_argument("--skip-strong", action="store_true", default=False)
    parser.add_argument("--s1-single-scale", action="store_true", default=False,
                        help="Use single-scale L96 (NO=40,J=0) for S1 DA dynamics")
    parser.add_argument("--s1-no-inflation", action="store_true", default=False,
                        help="No inflation for S1 EnKF/ETKF")
    parser.add_argument("--obs-j", type=int, default=None,
                        help="Number of fast vars to observe per slow node (default: all J_truth)")
    parser.add_argument("--s1-j", type=int, default=None,
                        help="Number of fast vars in S1 DA model (default: same as truth J)")
    parser.add_argument("--obs-interval", type=int, default=200,
                        help="Observation interval in steps (default: 200)")
    parser.add_argument("--ensemble-size", type=int, default=30,
                        help="Ensemble size for EnKF/ETKF (default: 30)")
    parser.add_argument("--inflation", type=float, default=2.0,
                        help="Inflation factor for EnKF/ETKF (default: 2.0)")
    parser.add_argument("--loc-radius", type=float, default=None,
                        help="R-localization radius in nodes (default: none)")
    parser.add_argument("--etkf-loc-mode", type=str, default="square_root",
                        choices=["square_root", "per_member"],
                        help="ETKF localization update mode (default: square_root)")
    parser.add_argument("--truth-fast-weights-unobserved", type=float, default=None,
                        help="Weight for unobserved fast vars in truth slow dynamics (default: 1.0 = equal weights)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if torch.cuda.is_available():
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Device: {device}")

    dt = 0.001
    J_truth = 4
    NO = 8

    obs_indices = make_obs_j_indices(NO, J_truth, args.obs_j)
    obs_dim = len(obs_indices) if obs_indices is not None else NO * (1 + J_truth)
    s1_J = args.s1_j if args.s1_j is not None else J_truth
    s1_state_dim = NO + NO * s1_J
    s1_obs_j = args.obs_j if args.obs_j is not None else J_truth
    s1_obs_j = min(s1_J, s1_obs_j)
    s1_obs_indices = list(range(NO + NO * s1_obs_j))
    s1_obs_dim = len(s1_obs_indices)

    if args.truth_fast_weights_unobserved is not None:
        obs_j = args.obs_j if args.obs_j is not None else J_truth
        truth_fast_weights = [1.0] * obs_j + [args.truth_fast_weights_unobserved] * (J_truth - obs_j)
    else:
        truth_fast_weights = None

    base_cfg = Lorenz96Config(
        dt=dt, T_max=3.0, obs_interval=args.obs_interval,
        R_var=0.5, B_var=2.0,
        num_windows=args.num_windows, window_spacing=args.num_windows,
        spinup_steps=5000, seed=42,
        NO=NO, J=J_truth, h=1.0, hx=1.0, eps=0.1,
        F_true=8.0, F_da=8.0,
        gamma=0.05, W_L_bar=0.0, c1=1.0, c2=0.1,
        sigma_0=0.08, sigma_L=0.20,
        tau_eta=5.0, sigma_eta=np.sqrt(0.5),
        param_bias=args.param_bias, forcing_state_bias=args.forcing_state_bias,
        obs_var_indices=obs_indices,
        fast_weights=truth_fast_weights,
    )

    print(f"\n── {args.label}: Config ──")
    print(f"  Truth: NO={NO}, J={J_truth}, dim={NO + NO * J_truth}")
    print(f"  Observed fast vars per node: {'all' if args.obs_j is None else args.obs_j} → obs_dim={obs_dim}")
    print(f"  S1 DA J: {s1_J} → state_dim={NO + NO * s1_J}, obs_dim={s1_obs_dim}")
    print(f"  param_bias={args.param_bias}, forcing_state_bias={args.forcing_state_bias}")
    print(f"  F_da = {base_cfg.F_true * (1 - args.param_bias):.2f}")
    print(f"  num_windows={args.num_windows}")
    print(f"  ensemble_size={args.ensemble_size}, inflation={args.inflation}, loc_radius={args.loc_radius}, etkf_loc_mode={args.etkf_loc_mode}")
    if truth_fast_weights is not None:
        print(f"  truth_fast_weights={truth_fast_weights}")
    labels = []
    if args.s1_single_scale: labels.append("single-scale DA")
    if args.s1_no_inflation: labels.append("no inflation")
    if labels: print(f"  S1 special: {', '.join(labels)}")

    t0 = time.time()
    datasets = make_s0_s1_datasets(base_cfg, args.num_windows,
                                    args.param_bias, args.forcing_state_bias,
                                    obs_indices)
    print(f"  Dataset gen: {time.time()-t0:.1f}s")

    s0_obs_op = ObsOperator(NO + NO * J_truth, obs_indices)
    s1_obs_op = ObsOperator(NO + NO * s1_J, s1_obs_indices)

    dynamics_pool = {
        1.0: Lorenz96Dynamics(dt=dt, coupling_exponent=1.0, fast_weights=truth_fast_weights),
        1.6: Lorenz96Dynamics(dt=dt, coupling_exponent=1.6, fast_weights=truth_fast_weights),
    }
    s1_dynamics = Lorenz96Dynamics(dt=dt, NO=40, J=0, h=0.0, hx=0.0,
                                   coupling_exponent=1.0) if args.s1_single_scale else None
    if not args.s1_single_scale and s1_J != J_truth:
        s1_dynamics = Lorenz96Dynamics(dt=dt, NO=NO, J=s1_J, h=1.0, hx=1.0, eps=0.1,
                                       coupling_exponent=1.0)

    methods_to_run = _METHODS[:]
    if args.skip_weak:
        methods_to_run.remove("Weak-4DVar")
    if args.skip_strong:
        methods_to_run.remove("Strong-4DVar")

    results = {}
    for case_key, case_label, da_expo, obs_op, j_val in [
        ("s0", "S0", 1.6, s0_obs_op, J_truth), ("s1", "S1", 1.0, s1_obs_op, s1_J)]:
        ds = datasets[case_key]
        inf = args.inflation
        dyn = s1_dynamics if (s1_dynamics is not None and case_key == "s1") else dynamics_pool[1.0]
        method_map = {
            "Weak-4DVar": Weak4DVar(dt=dt, da_window_steps=args.da_window_steps, device=device,
                                     coupling_exponent=da_expo, dynamics=dyn, obs_operator=obs_op),
            "Strong-4DVar": Strong4DVar(dt=dt, da_window_steps=args.da_window_steps, device=device,
                                          coupling_exponent=da_expo, dynamics=dyn,
                                          max_iter=10, lr=0.2, obs_operator=obs_op),
            "EnKF": EnKF(dt=dt, device=device, coupling_exponent=da_expo,
                          dynamics=dyn, inflation=inf, obs_operator=obs_op,
                          N_ensemble=args.ensemble_size,
                          loc_radius=args.loc_radius, NO=8, J=j_val),
            "ETKF": ETKF(dt=dt, device=device, coupling_exponent=da_expo,
                           dynamics=dyn, inflation=inf, obs_operator=obs_op,
                           N_ensemble=args.ensemble_size,
                           loc_radius=args.loc_radius, NO=8, J=j_val,
                           loc_mode=args.etkf_loc_mode),
        }
        if case_key == "s1" and s1_obs_dim < obs_dim:
            for w in ds.windows:
                w["obs"] = w["obs"][:, :s1_obs_dim]
        eval_cfg = Lorenz96Config(**{**base_cfg.__dict__,
            "obs_var_indices": list(range(s1_state_dim)) if case_key == "s1" else base_cfg.obs_var_indices})
        results[case_key] = {}
        for name in methods_to_run:
            method = method_map[name]
            print(f"  {case_label}/{name} ...", end=" ", flush=True)
            t1 = time.time()
            m, s = evaluate_baseline(method, ds, cfg=eval_cfg, device=device,
                                      return_trajs=False,
                                      batch_size=min(20, args.num_windows))
            m_common = m[:s1_state_dim] if case_key == "s0" else m
            s_common = s[:s1_state_dim] if case_key == "s0" else s
            results[case_key][name] = {
                "mean_rmse": float(np.mean(m_common)),
                "per_var_mean": m_common.tolist(),
                "per_var_std": s_common.tolist(),
            }
            elapsed = time.time() - t1
            print(f"  mu={np.mean(m_common):.4f} [{elapsed:.1f}s]")

    out = {
        "label": args.label,
        "config": {
            "param_bias": args.param_bias,
            "forcing_state_bias": args.forcing_state_bias,
            "num_windows": args.num_windows,
            "da_window_steps": args.da_window_steps,
            "F_da": base_cfg.F_true * (1 - args.param_bias),
            "s1_single_scale": args.s1_single_scale,
            "s1_no_inflation": args.s1_no_inflation,
            "ensemble_size": args.ensemble_size,
            "inflation": args.inflation,
            "loc_radius": args.loc_radius,
            "etkf_loc_mode": args.etkf_loc_mode,
            "J_truth": J_truth,
            "s1_J": s1_J,
            "obs_J": args.obs_j,
            "obs_dim": obs_dim,
            "s1_state_dim": s1_state_dim,
            "obs_interval": args.obs_interval,
            "obs_indices": obs_indices,
            "truth_fast_weights": truth_fast_weights,
        },
        "results": results,
    }
    out_path = os.path.join(EXP_DIR, f"l96_sweep_{args.label}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    print(f"\n── {args.label} Summary ──")
    print(f"{'Method':<20} {'S0 mu':<10} {'S1 mu':<10} {'Delta%':<10}")
    print("-" * 50)
    for name in methods_to_run:
        if name not in results["s0"] or name not in results["s1"]:
            continue
        s0_m = results["s0"][name]["mean_rmse"]
        s1_m = results["s1"][name]["mean_rmse"]
        pct = (s1_m / s0_m - 1) * 100
        print(f"{name:<20} {s0_m:<10.4f} {s1_m:<10.4f} {pct:<+10.1f}%")

if __name__ == "__main__":
    run()