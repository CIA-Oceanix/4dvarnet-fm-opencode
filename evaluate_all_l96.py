#!/usr/bin/env python3
"""Lorenz96 S0/S1 baseline evaluation — data generation + DA baselines + RMSE table."""
import os, sys, json, argparse, time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.lorenz96 import Lorenz96Config, make_l96_s0_s1_datasets
from evaluation.run_l96 import run_and_cache_baselines, _BASELINE_METHODS, _BASELINE_CASES

BASE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(BASE, "experiments")


def run_baselines(datasets, device, da_window_steps=None,
                  enkf_inflation=None, etkf_inflation=None, suffix="",
                  weak_config=None, strong_config=None):
    print("\n── Running L96 Baselines ──")
    enkf_config = {"inflation": enkf_inflation} if enkf_inflation else None
    etkf_config = {"inflation": etkf_inflation} if etkf_inflation else None
    results = run_and_cache_baselines(datasets, device, batch_size=200,
                                       da_window_steps=da_window_steps,
                                       enkf_config=enkf_config,
                                       etkf_config=etkf_config,
                                       suffix=suffix,
                                       weak_config=weak_config,
                                       strong_config=strong_config)
    return results


def build_table(baseline_results):
    rows = []
    for case_name, _, _, _, label, _ in _BASELINE_CASES:
        row = {"Case": label}
        for method in _BASELINE_METHODS:
            bl = baseline_results.get(case_name, {}).get(method, {})
            row[f"{method}"] = bl.get("mean", float("nan"))
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
    )
    print(f"Config: NO={base_cfg.NO} J={base_cfg.J} F_true={base_cfg.F_true}")
    print(f"  R_var={args.r_var} obs_interval={args.obs_interval} dws={args.da_window_steps}")
    print(f"  enkf_inflation={args.enkf_inflation} etkf_inflation={args.etkf_inflation}")

    print("\n── Generating L96 S0/S1 datasets ──")
    t0 = time.time()
    datasets = make_l96_s0_s1_datasets(base_cfg, num_test_windows=args.num_test_windows)
    print(f"  test_s0: {len(datasets['test_s0'])} windows")
    print(f"  test_s1: {len(datasets['test_s1'])} windows")
    print(f"  Dataset generation: {time.time() - t0:.1f}s")

    baseline_results = run_baselines(datasets, device,
                                      da_window_steps=args.da_window_steps,
                                      enkf_inflation=args.enkf_inflation,
                                      etkf_inflation=args.etkf_inflation,
                                      suffix=args.suffix,
                                      weak_config={"opt_steps": 50, "lr": 0.1},
                                      strong_config={"max_iter": 10, "lr": 0.2})

    print("\n── L96 S0/S1 Comparison Table ──")
    headers = ["Case"] + _BASELINE_METHODS
    rows = build_table(baseline_results)
    print_table(rows, headers)

    combined = {"baselines": baseline_results}
    out_path = os.path.join(EXP_DIR, "evaluate_all_l96.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()