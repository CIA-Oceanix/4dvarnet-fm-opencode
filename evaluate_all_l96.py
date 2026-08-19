#!/usr/bin/env python3
"""Lorenz96 S0/S1 baseline evaluation — data generation + DA baselines + RMSE table."""
import os, sys, json, argparse, time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.lorenz96 import Lorenz96Config, make_l96_s0_s1_trainval, _generate_observations
from evaluation.run_l96 import run_and_cache_baselines, _BASELINE_METHODS, _BASELINE_CASES, make_obs_j_indices

BASE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(BASE, "experiments")


def run_baselines(datasets, device, da_window_steps=None,
                  enkf_inflation=None, etkf_inflation=None, suffix="",
                  weak_config=None, strong_config=None, exclude_methods=None,
                  obs_j=2, obs_interval=200):
    print("\n── Running L96 Baselines ──")
    enkf_config = {"inflation": enkf_inflation} if enkf_inflation else None
    etkf_config = {"inflation": etkf_inflation} if etkf_inflation else None
    results = run_and_cache_baselines(datasets, device, batch_size=200,
                                       da_window_steps=da_window_steps,
                                       enkf_config=enkf_config,
                                       etkf_config=etkf_config,
                                       suffix=suffix,
                                       weak_config=weak_config,
                                       strong_config=strong_config,
                                       exclude_methods=exclude_methods,
                                       obs_j=obs_j,
                                       obs_interval=obs_interval)
    return results


def build_table(baseline_results, active_methods):
    rows = []
    for case_name, _, _, _, label, _ in _BASELINE_CASES:
        row = {"Case": label}
        for method in active_methods:
            bl = baseline_results.get(case_name, {}).get(method, {})
            row[f"{method}"] = bl.get("mean", float("nan"))
            groups = bl.get("groups", {})
            row[f"{method}_slow"] = groups.get("slow", float("nan"))
            row[f"{method}_obs_fast"] = groups.get("obs_fast", float("nan"))
        rows.append(row)
    return rows


def print_table(rows, headers):
    widths = {k: max(len(k), max(len(f"{r.get(k, ''):.4f}") if isinstance(r.get(k), (int,float)) else len(str(r.get(k, ''))) for r in rows)) for k in headers}
    line = " | ".join(f"{h:<{widths[h]}}" for h in headers)
    sep = "-+-".join("-" * widths[h] for h in headers)
    print(line)
    print(sep)
    for r in rows:
        vals = []
        for h in headers:
            v = r.get(h, "")
            if isinstance(v, float):
                vals.append(f"{v:<{widths[h]}.4f}")
            else:
                vals.append(f"{v:<{widths[h]}}")
        print(" | ".join(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--enkf-inflation", type=float, default=2.0)
    parser.add_argument("--etkf-inflation", type=float, default=2.0)
    parser.add_argument("--da-window-steps", type=int, default=500)
    parser.add_argument("--num-test-windows", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--t-max", type=float, default=3.0, help="Trajectory length in time units")
    parser.add_argument("--r-var", type=float, default=0.5)
    parser.add_argument("--obs-interval", type=int, default=200)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--skip-weak", action="store_true", default=False)
    parser.add_argument("--skip-strong", action="store_true", default=False)
    parser.add_argument("--randomize-params", type=str, default=None,
                        help="Comma-separated list of params to randomize (e.g. 'F' or 'F,c1,h,hx,eps'). "
                             "Default: all 5 params randomized.")
    parser.add_argument("--obs-j", type=int, default=2,
                        help="Number of fast vars observed per slow node (default: 2)")
    parser.add_argument("--regenerate-data", action="store_true", default=False,
                        help="Force dataset regeneration, ignoring cached .pt file")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if torch.cuda.is_available():
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Device: {device}")

    base_cfg = Lorenz96Config(
        dt=0.001, T_max=args.t_max, obs_interval=args.obs_interval,
        R_var=args.r_var, B_var=2.0,
        num_windows=2000, window_spacing=2000,
        spinup_steps=10000, seed=42,
        NO=8, J=4, h=1.0, hx=1.0, eps=0.1,
        F_true=8.0, F_da=8.0,
        gamma=0.05, W_L_bar=0.0, c1=1.0, c2=0.1,
        sigma_0=0.08, sigma_L=0.20,
        tau_eta=5.0, sigma_eta=np.sqrt(0.5),
        param_bias=0.0, forcing_state_bias=0.0,
        fast_weights=[1.0, 1.0, 0.1, 0.1],
        obs_var_indices=make_obs_j_indices(8, 4, args.obs_j),
    )
    obs_var_indices = base_cfg.obs_var_indices
    obs_dim = len(obs_var_indices) if obs_var_indices is not None else 40
    print(f"Config: NO={base_cfg.NO} J={base_cfg.J} F_true={base_cfg.F_true}")
    print(f"  obs_j={args.obs_j} obs_dim={obs_dim}")
    print(f"  R_var={args.r_var} obs_interval={args.obs_interval} dws={args.da_window_steps}")
    print(f"  enkf_inflation={args.enkf_inflation} etkf_inflation={args.etkf_inflation}")
    if obs_var_indices is not None:
        print(f"  obs_var_indices={list(obs_var_indices)}")

    exclude = []
    if args.skip_weak:
        exclude.append("Weak-4DVar")
    if args.skip_strong:
        exclude.append("Strong-4DVar")
    active_methods = [m for m in _BASELINE_METHODS if m not in exclude]
    if exclude:
        print(f"  Skipping: {', '.join(exclude)}")

    randomize_params = None
    if args.randomize_params:
        randomize_params = [p.strip() for p in args.randomize_params.split(",")]
        print(f"  Randomizing only: {randomize_params}")
    else:
        print(f"  Randomizing all params: F, c1, h, hx, eps")

    print("\n── Generating L96 S0/S1 datasets ──")
    ds_cache = os.path.join(EXP_DIR, f"l96_datasets_obsj{args.obs_j}_int{args.obs_interval}_nwin{args.num_test_windows}.pt")
    ref_cache = os.path.join(EXP_DIR, f"l96_datasets_obsj{args.obs_j}_nwin{args.num_test_windows}.pt")
    t0 = time.time()
    if os.path.exists(ds_cache) and not args.regenerate_data:
        print(f"  Loading cached datasets ({ds_cache})...")
        datasets = torch.load(ds_cache, weights_only=False)
    elif os.path.exists(ref_cache) and not args.regenerate_data:
        print(f"  Reusing trajectories from {ref_cache} and re-observing at obs_interval={args.obs_interval}...")
        datasets = torch.load(ref_cache, weights_only=False)
        for key in ("test_s0", "test_s1"):
            for w in datasets[key]:
                w["obs"], w["obs_mask"] = _generate_observations(
                    w["true_state"], args.obs_interval, args.r_var, w["obs_seed"],
                    obs_var_indices=obs_var_indices)
        torch.save(datasets, ds_cache)
    else:
        datasets = make_l96_s0_s1_trainval(
            base_cfg, num_train_windows=2, num_val_windows=2,
            num_test_windows=args.num_test_windows,
            param_noise=0.2, bias_range=(0.0, 0.2),
            randomize_params=randomize_params,
        )
        torch.save(datasets, ds_cache)
    print(f"  test_s0: {len(datasets['test_s0'])} windows")
    print(f"  test_s1: {len(datasets['test_s1'])} windows")
    print(f"  Dataset prep: {time.time() - t0:.1f}s")

    baseline_results = run_baselines(datasets, device,
                                      da_window_steps=args.da_window_steps,
                                      enkf_inflation=args.enkf_inflation,
                                      etkf_inflation=args.etkf_inflation,
                                      suffix=args.suffix,
                                      weak_config={"opt_steps": 50, "lr": 0.1},
                                      strong_config={"max_iter": 10, "lr": 0.2},
                                      exclude_methods=exclude,
                                      obs_j=args.obs_j,
                                      obs_interval=args.obs_interval)

    print("\n── L96 S0/S1 Comparison Table ──")
    headers = ["Case"]
    for m in active_methods:
        headers.extend([m, f"{m}_slow", f"{m}_obs_fast"])
    rows = build_table(baseline_results, active_methods)
    print_table(rows, headers)

    combined = {"baselines": baseline_results}
    suffix_tag = f"{args.da_window_steps}{args.suffix}" if args.suffix else str(args.da_window_steps)
    out_path = os.path.join(EXP_DIR, f"evaluate_all_l96_{suffix_tag}.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()