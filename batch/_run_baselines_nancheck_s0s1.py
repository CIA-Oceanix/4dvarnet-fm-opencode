import os, sys, time, json
import torch
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.lorenz63 import Lorenz63Config, make_s0_s1_trainval
from evaluation.run import run_and_cache_baselines

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

base_cfg = Lorenz63Config()

datasets = make_s0_s1_trainval(
    base_cfg,
    num_train_windows=1,
    num_val_windows=1,
    num_test_windows=10,
    param_noise=0.2,
    bias_range=(0.0, 0.2),
)
print(f"test_s0: {len(datasets['test_s0'])} windows")
print(f"test_s1: {len(datasets['test_s1'])} windows")

ds = datasets["test_s0"]
w = ds[0]
has_obs = "obs" in w
print(f"Window has obs: {has_obs}")
print(f"obs shape: {w['obs'].shape}")
print(f"obs contains NaN: {torch.isnan(w['obs']).any().item()}")
print(f"obs contains inf: {torch.isinf(w['obs']).any().item()}")

results = run_and_cache_baselines(
    datasets, device,
    batch_size=10, da_window_steps=50,
    weak_config={"opt_steps": 150, "lr": 0.02},
    strong_config={"max_iter": 40, "lr": 0.1},
    enkf_config={"N_ensemble": 30, "inflation": 2.0},
    etkf_config={"N_ensemble": 30, "inflation": 2.0},
    suffix="_nancheck_s0s1",
)

all_finite = True
for case in ["s0", "s1"]:
    if case not in results:
        continue
    for method in ["Weak-4DVar", "Strong-4DVar", "EnKF", "ETKF"]:
        if method not in results[case]:
            continue
        m = results[case][method]["mean"]
        for coord in ["X", "Y", "Z"]:
            if not np.isfinite(results[case][method][coord]["mean"]):
                print(f"  {case}/{method}/{coord}: NaN!")
                all_finite = False
        print(f"  {case}/{method:<20} mean={m:.4f}")

if all_finite:
    print("\nAll results finite!")
else:
    print("\nNaN detected!")