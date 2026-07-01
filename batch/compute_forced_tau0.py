#!/usr/bin/env python3
"""Compute forced τ=0 single-step evaluation for all F1/F2/F3 models."""

import os, sys, json, time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.lorenz63 import Lorenz63Config, Lorenz63Dataset
from models.vanilla_cfm import VanillaCFM
from evaluation.metrics import rmse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

base = dict(
    dt=0.01, T_max=3.0, obs_interval=20,
    R_var=0.5, B_var=2.0,
    num_windows=200, window_spacing=2000,
    spinup_steps=10000, seed=42,
    sigma_true=10.0, rho_true=28.0, beta_true=2.6666666666666665,
    gamma=0.05, W_L_bar=0.0, c1=1.0, c2=0.1,
    sigma_0=0.08, sigma_L=0.20,
    tau_eta=5.0, sigma_eta=0.7071067811865476,
    param_bias=0.0, forcing_state_bias=0.0, forcing_coupling="linear",
)

print("Creating test datasets...")
t0 = time.time()
test_cs1_cfg = Lorenz63Config(**{**base, "case": 1, "param_bias": 0.0, "seed": 123, "num_windows": 200})
test_cs2_cfg = Lorenz63Config(**{**base, "case": 2, "param_bias": 0.15, "forcing_state_bias": 0.15, "seed": 124, "num_windows": 200})
ds_cs1 = Lorenz63Dataset(test_cs1_cfg)
ds_cs2 = Lorenz63Dataset(test_cs2_cfg)
print(f"  Done in {time.time()-t0:.1f}s")

def make_model(hidden, train_tau_0_only=False):
    return VanillaCFM(
        state_dim=3, hidden_channels=hidden, time_emb_dim=64,
        N_outer=10, sigma_prior=0.5, dropout=0.1,
        train_tau_0_only=train_tau_0_only,
    ).to(DEVICE)

def evaluate_tau0(model, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    results = {}
    for label, ds in [("cs1", ds_cs1), ("cs2", ds_cs2)]:
        rmse_list = []
        for i in range(len(ds)):
            w = ds[i]
            obs = w["obs"].unsqueeze(0).to(DEVICE)
            x = torch.randn_like(obs) * model.sigma_prior
            v = model.forward(x, obs, torch.zeros(1, device=DEVICE))
            pred = (x + v).detach().cpu().numpy()[0]
            rmse_list.append(rmse(pred, w["true_state"].numpy()))
        all_r = np.stack(rmse_list, axis=0)
        mean_xyz = np.mean(all_r, axis=0)
        results[label] = {
            "X": float(mean_xyz[0]), "Y": float(mean_xyz[1]), "Z": float(mean_xyz[2]),
            "overall_mean": float(np.mean(all_r)),
        }
    return results

configs = [
    ("F1_vanilla_cfm_default", "experiments/F1_vanilla_cfm_default/checkpoints/stage1.pt", [64,128,256]),
    ("F2_vanilla_cfm_small",   "experiments/F2_vanilla_cfm_small/checkpoints/stage1.pt",   [32,64,128]),
    ("F3_vanilla_cfm_rand",    "experiments/F3_vanilla_cfm_rand/checkpoints/stage1.pt",    [32,64,128]),
]

results = {}
for name, ckpt_rel, hidden in configs:
    ckpt_path = os.path.join(BASE, ckpt_rel)
    print(f"\n{name} -> forced τ=0 ...")
    model = make_model(hidden, train_tau_0_only=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    t0 = time.time()
    r = evaluate_tau0(model, seed=42)
    t = time.time() - t0
    results[name] = r
    cs2_val = r['cs2']['overall_mean']
    print(f"  CS1={r['cs1']['overall_mean']:.6f}  CS2={cs2_val:.6f}  ({t:.1f}s)")

out_path = os.path.join(BASE, "experiments/forced_tau0_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
