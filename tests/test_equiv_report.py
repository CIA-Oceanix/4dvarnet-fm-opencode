"""
Compare refactored baseline RMSEs against the published S0/S1 report.
Uses the exact config from reports/outputs/s0_s1_synthesis.md:
  dws=50, 15 obs per window (obs_interval=20), R_var=0.5,
  EnKF/ETKF inflation=2.0, Weak/Strong opt_steps=150, lr=0.01
"""
import sys, os, json
os.environ['TRITON_CACHE_DIR'] = '/tmp/triton_cache'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from data.lorenz63 import Lorenz63Config, make_mixed_datasets
from evaluation.run import run_and_cache_baselines, EXP_DIR

device = torch.device("cuda")
print(f"Device: {device}")

cfg = Lorenz63Config(dt=0.01, T_max=3.0, obs_interval=20, R_var=0.5, B_var=2.0,
                     num_windows=1, window_spacing=1, spinup_steps=10000, seed=42,
                     param_bias=0.0, forcing_state_bias=0.0)

datasets = make_mixed_datasets(cfg, include_s1_test=True)
print(f"Datasets: { {k: len(v) for k, v in datasets.items()} }")

result = run_and_cache_baselines(
    datasets, device, batch_size=1, da_window_steps=50,
    weak_config={"opt_steps": 150, "lr": 0.01},
    strong_config={"max_iter": 50, "lr": 0.5},
    enkf_config={"N_ensemble": 30, "inflation": 2.0},
    etkf_config={"N_ensemble": 30, "inflation": 2.0},
    suffix="_equiv_test",
)

print()
print("=" * 60)
print("NEW RESULTS (refactored code with dynamics)")
print("=" * 60)
for case in ["s0", "s1"]:
    print(f"\n--- {case.upper()} ---")
    for method in ["Weak-4DVar", "Strong-4DVar", "EnKF", "ETKF"]:
        v = result[case][method]
        print(f"  {method:<15} X={v['X']['mean']:.4f} Y={v['Y']['mean']:.4f} Z={v['Z']['mean']:.4f} mean={v['mean']:.4f}")

# Report values from s0_s1_synthesis.md
REPORT = {
    "s0": {
        "Weak-4DVar": 0.64,
        "Strong-4DVar": 0.73,
        "EnKF": 0.78,
        "ETKF": 0.77,
    },
    "s1": {
        "Weak-4DVar": 1.64,
        "Strong-4DVar": 2.14,
        "EnKF": 2.27,
        "ETKF": 2.28,
    },
}

print()
print("=" * 60)
print("COMPARISON: new vs report (diff)")
print("=" * 60)
max_diff = 0.0
all_pass = True
for case in ["s0", "s1"]:
    print(f"\n--- {case.upper()} ---")
    for method in ["Weak-4DVar", "Strong-4DVar", "EnKF", "ETKF"]:
        new_mean = result[case][method]["mean"]
        ref_mean = REPORT[case][method]
        diff = abs(new_mean - ref_mean)
        max_diff = max(max_diff, diff)
        tol = 0.05
        ok = diff < tol
        if not ok:
            all_pass = False
        print(f"  {method:<15} new={new_mean:.4f}  ref={ref_mean:.4f}  diff={diff:.4f}  {'PASS' if ok else 'FAIL'}")

print(f"\nMax diff: {max_diff:.4f}")
print(f"OVERALL: {'ALL PASS' if all_pass else 'SOME FAIL'}")