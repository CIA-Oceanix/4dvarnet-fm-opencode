#!/usr/bin/env python3
"""
Run DA baselines with the NaN-fixed code and save results with '_nancheck' suffix.
Compares against the old baselines_dws300.json and generates a PDF report.
"""
import os, sys, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.lorenz63 import Lorenz63Config, make_mixed_datasets
from evaluation.run import run_and_cache_baselines, EXP_DIR

# Match defaults from config/lorenz63_default.yaml
base_cfg = Lorenz63Config(
    dt=0.01, T_max=3.0, obs_interval=20, R_var=0.5, B_var=2.0,
    num_windows=2000, window_spacing=2000, spinup_steps=10000, seed=42,
    param_bias=0.0, forcing_state_bias=0.0, forcing_coupling="linear",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Device: {device} ({dev_name})")

# Delete old cached datasets so they regenerate with NaN obs
ds_cache = os.path.join(EXP_DIR, "datasets.pt")
if os.path.exists(ds_cache):
    os.remove(ds_cache)
    print(f"Deleted old {ds_cache}")

t0 = time.time()
datasets = make_mixed_datasets(base_cfg)
print(f"Datasets generated in {time.time()-t0:.1f}s")
total_test = sum(len(datasets[k]) for k in datasets if k.startswith("test_"))
print(f"  test_s0={len(datasets.get('test_s0',[]))}, test_s1={len(datasets.get('test_s1',[]))}")

# Run baselines with suffix="_nancheck"
run_and_cache_baselines(
    datasets, device,
    batch_size=128, da_window_steps=300,
    weak_config={"opt_steps": 150, "lr": 0.02},
    strong_config={"max_iter": 40, "lr": 0.1},
    enkf_config={"N_ensemble": 30, "inflation": 1.0},
    etkf_config={"N_ensemble": 30, "inflation": 1.0},
    suffix="_nancheck",
)

# Compare with old results
old_path = os.path.join(EXP_DIR, "baselines_dws300.json")
new_path = os.path.join(EXP_DIR, "baselines_dws300_nancheck.json")

print("\n============================================")
print("  Comparison: old vs new baseline RMSE")
print("============================================")
if os.path.exists(old_path):
    with open(old_path) as f:
        old = json.load(f)
else:
    old = {}
with open(new_path) as f:
    new = json.load(f)

header = f"{'Case/Method':<30} {'Old mean':>10} {'New mean':>10} {'Δ':>10}"
print(header)
print("-" * len(header))
for case in ("s0", "s1"):
    for method in ("Weak-4DVar", "Strong-4DVar", "EnKF", "ETKF"):
        om = old.get(case, {}).get(method, {}).get("mean", None)
        nm = new.get(case, {}).get(method, {}).get("mean", None)
        if om is not None and nm is not None:
            delta = nm - om
            print(f"{case}/{method:<25} {om:>10.4f} {nm:>10.4f} {delta:>+10.4f}")
        elif nm is not None:
            print(f"{case}/{method:<25} {'N/A':>10} {nm:>10.4f} {'N/A':>10}")
print("============================================")

# Generate PDF report
report_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "outputs")
os.makedirs(report_dir, exist_ok=True)
import subprocess
report_script = os.path.join(os.path.dirname(__file__), "..", "reports", "generate_baseline_report.py")
pdf_path = os.path.join(report_dir, "synthesis_nancheck.pdf")
traj_path = os.path.join(EXP_DIR, "baselines_trajectories_dws300_nancheck.npz")
result = subprocess.run([
    sys.executable, report_script,
    "--json", new_path,
    "--trajs", traj_path,
    "--output", pdf_path,
], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(f"Report generation stderr:\n{result.stderr}")
else:
    print(f"Report saved to {pdf_path}")