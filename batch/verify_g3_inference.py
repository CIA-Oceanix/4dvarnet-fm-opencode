#!/usr/bin/env python3
"""Verify G3 (τ=0 CFM, randparam) inference with multiple x0 seeds,
   and compare F3 evaluated with τ=0 single-step sampling."""

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

# ── Config matching lorenz63_default ───────────────────────────────────
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

# Only create test datasets (skip train/val for speed)
print("Creating test datasets...")
t0 = time.time()
test_cs1_cfg = Lorenz63Config(**{**base, "case": 1, "param_bias": 0.0, "seed": 123, "num_windows": 200})
test_cs2_cfg = Lorenz63Config(**{**base, "case": 2, "param_bias": 0.15, "forcing_state_bias": 0.15, "seed": 124, "num_windows": 200})
ds_cs1 = Lorenz63Dataset(test_cs1_cfg)
ds_cs2 = Lorenz63Dataset(test_cs2_cfg)
print(f"  Created in {time.time()-t0:.1f}s  CS1={len(ds_cs1)} windows, CS2={len(ds_cs2)} windows")

# ── Model factory ──────────────────────────────────────────────────────
def make_model(train_tau_0_only: bool) -> VanillaCFM:
    return VanillaCFM(
        state_dim=3,
        hidden_channels=[32, 64, 128],
        time_emb_dim=64,
        N_outer=10,
        sigma_prior=0.5,
        dropout=0.1,
        train_tau_0_only=train_tau_0_only,
    ).to(DEVICE)

# ── Evaluation ─────────────────────────────────────────────────────────
def evaluate_model(model, dataset, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rmse_list = []
    for i in range(len(dataset)):
        w = dataset[i]
        obs = w["obs"].unsqueeze(0).to(DEVICE)
        pred = model.sample(obs).detach().cpu().numpy()[0]
        truth = w["true_state"].numpy()
        rmse_list.append(rmse(pred, truth))
    all_rmse = np.stack(rmse_list, axis=0)
    return np.mean(all_rmse, axis=0), np.std(all_rmse, axis=0)

def fmt(mean, std):
    return {
        "X": {"mean": float(mean[0]), "std": float(std[0])},
        "Y": {"mean": float(mean[1]), "std": float(std[1])},
        "Z": {"mean": float(mean[2]), "std": float(std[2])},
        "mean": float(np.mean(mean)),
    }

# ── G3: τ=0 CFM ────────────────────────────────────────────────────────
print("\n=== G3 (τ=0 CFM, randparam) ===")
g3_model = make_model(train_tau_0_only=True)
g3_ckpt = os.path.join(BASE, "experiments/G3_vanilla_cfm_t0_rand/checkpoints/stage1.pt")
g3_model.load_state_dict(torch.load(g3_ckpt, map_location=DEVICE))
g3_model.eval()

g3_results = {"experiment": "G3_vanilla_cfm_t0_rand", "train_tau_0_only": True}
for seed in [42, 123, 999]:
    t0 = time.time()
    m_cs1, s_cs1 = evaluate_model(g3_model, ds_cs1, seed)
    m_cs2, s_cs2 = evaluate_model(g3_model, ds_cs2, seed)
    t = time.time() - t0
    key = f"seed_{seed}"
    g3_results[key] = {"cs1": fmt(m_cs1, s_cs1), "cs2": fmt(m_cs2, s_cs2)}
    print(f"  seed={seed:3d}:  CS1={np.mean(m_cs1):.6f}  CS2={np.mean(m_cs2):.6f}  ({t:.1f}s)")

g3_results["original"] = {"cs1": {"mean": 0.03197464346885681}, "cs2": {"mean": 0.03201243281364441}}

# ── F3: vanilla CFM, forced τ=0 single step ────────────────────────────
print("\n=== F3 (vanilla CFM) → forced τ=0 single step ===")
f3_model = make_model(train_tau_0_only=False)
f3_ckpt = os.path.join(BASE, "experiments/F3_vanilla_cfm_rand/checkpoints/stage1.pt")
f3_model.load_state_dict(torch.load(f3_ckpt, map_location=DEVICE))
f3_model.eval()

f3_results = {"experiment": "F3_vanilla_cfm_rand", "inference_mode": "forced_tau0_single_step"}
for seed in [42, 123, 999]:
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    m_cs1_list, m_cs2_list = [], []
    for i in range(len(ds_cs1)):
        w_cs1 = ds_cs1[i]; w_cs2 = ds_cs2[i]
        obs1 = w_cs1["obs"].unsqueeze(0).to(DEVICE)
        obs2 = w_cs2["obs"].unsqueeze(0).to(DEVICE)
        # τ=0 single step
        x1 = torch.randn_like(obs1) * f3_model.sigma_prior
        v1 = f3_model.forward(x1, obs1, torch.zeros(1, device=DEVICE))
        x2 = torch.randn_like(obs2) * f3_model.sigma_prior
        v2 = f3_model.forward(x2, obs2, torch.zeros(1, device=DEVICE))
        m_cs1_list.append(rmse((x1+v1).detach().cpu().numpy()[0], w_cs1["true_state"].numpy()))
        m_cs2_list.append(rmse((x2+v2).detach().cpu().numpy()[0], w_cs2["true_state"].numpy()))
    all1 = np.stack(m_cs1_list, axis=0)
    all2 = np.stack(m_cs2_list, axis=0)
    m_cs1, s_cs1 = np.mean(all1, axis=0), np.std(all1, axis=0)
    m_cs2, s_cs2 = np.mean(all2, axis=0), np.std(all2, axis=0)
    key = f"seed_{seed}_tau0"
    f3_results[key] = {"cs1": fmt(m_cs1, s_cs1), "cs2": fmt(m_cs2, s_cs2)}
    print(f"  {key}:  CS1={np.mean(m_cs1):.6f}  CS2={np.mean(m_cs2):.6f}  ({time.time()-t0:.1f}s)")

# F3 reference: full 10-step at seed=42
print("\n=== F3 reference: full 10-step sampling ===")
f3_ref = make_model(train_tau_0_only=False)
f3_ref.load_state_dict(torch.load(f3_ckpt, map_location=DEVICE))
f3_ref.eval()
t0 = time.time()
m_cs1, s_cs1 = evaluate_model(f3_ref, ds_cs1, 42)
m_cs2, s_cs2 = evaluate_model(f3_ref, ds_cs2, 42)
f3_results["reference_10step_seed42"] = {"cs1": fmt(m_cs1, s_cs1), "cs2": fmt(m_cs2, s_cs2)}
print(f"  10-step: CS1={np.mean(m_cs1):.6f}  CS2={np.mean(m_cs2):.6f}  ({time.time()-t0:.1f}s)")

# ── Report ──────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print(f"{'Model':<30} {'Seed':<8} {'CS1':<12} {'CS2':<12}")
print("=" * 62)
for label, results in [("G3 (τ=0 trained, τ=0 eval)", g3_results),
                        ("F3 (multi-τ trained, τ=0 eval)", f3_results)]:
    for key, vals in results.items():
        if key.startswith("seed_"):
            seed_label = key.replace("seed_", "").replace("_tau0", "")
            print(f"{label:<30} {seed_label:<8} {vals['cs1']['mean']:<12.6f} {vals['cs2']['mean']:<12.6f}")
    print()
print(f"{'F3 (full 10-step)':<30} {'42':<8} {f3_results['reference_10step_seed42']['cs1']['mean']:<12.6f} {f3_results['reference_10step_seed42']['cs2']['mean']:<12.6f}")
print(f"{'G3 original':<30} {'?':<8} {0.0319746:<12.6f} {0.0320124:<12.6f}")

output = {"g3": g3_results, "f3": f3_results}
save_path = os.path.join(BASE, "experiments/verify_g3_inference.json")
with open(save_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {save_path}")
