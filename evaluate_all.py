#!/usr/bin/env python3
"""Unified evaluation: run baselines on S0/S1 benchmark, produce RMSE table."""
import os, sys, json, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.lorenz63 import Lorenz63Config, make_mixed_datasets
from evaluation.run import run_and_cache_baselines, _BASELINE_METHODS, _BASELINE_CASES, fmt_rmse

BASE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(BASE, "experiments")


def run_baselines(datasets, device, da_window_steps=None,
                  enkf_inflation=None, etkf_inflation=None, suffix=""):
    print("\n── Running Baselines ──")
    enkf_config = {"inflation": enkf_inflation} if enkf_inflation else None
    etkf_config = {"inflation": etkf_inflation} if etkf_inflation else None
    results = run_and_cache_baselines(datasets, device, batch_size=200,
                                      da_window_steps=da_window_steps,
                                      enkf_config=enkf_config,
                                      etkf_config=etkf_config,
                                      suffix=suffix)
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
    parser.add_argument("--enkf-inflation", type=float, default=None)
    parser.add_argument("--etkf-inflation", type=float, default=None)
    parser.add_argument("--da-window-steps", type=int, default=50)
    parser.add_argument("--r-var", type=float, default=0.5)
    parser.add_argument("--obs-interval", type=int, default=20)
    parser.add_argument("--suffix", type=str, default="")
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

    baseline_results = run_baselines(datasets, device,
                                     da_window_steps=args.da_window_steps,
                                     enkf_inflation=args.enkf_inflation,
                                     etkf_inflation=args.etkf_inflation,
                                     suffix=args.suffix)

    print("\n── Comparison Table ──")
    headers = ["Case"] + _BASELINE_METHODS
    rows = build_table(baseline_results)
    print_table(rows, headers)

    combined = {"baselines": baseline_results}
    out_path = os.path.join(EXP_DIR, "evaluate_all.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved comparison to {out_path}")


if __name__ == "__main__":
    main()
