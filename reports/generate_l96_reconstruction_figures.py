#!/usr/bin/env python3
"""
Generate S0/S1 reconstruction vs truth trajectory figures for the report.
Uses ETKF (N=30, inf=1.1, no loc) on the default weighted coupling config.

Usage:
    conda run -n fdv python reports/generate_l96_reconstruction_figures.py
"""
import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.lorenz96 import Lorenz96Config, RandomParamLorenz96Dataset, RandomBiasLorenz96Dataset
from models.lorenz96_dynamics import Lorenz96Dynamics
from evaluation.baselines import ETKF, ObsOperator

FIGS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

DT = 0.001
T_MAX = 3.0
NUM_STEPS = int(T_MAX / DT)
J_TRUTH = 4
NO = 8
OBS_J = 2
S1_J = 2
INF = 1.1
N_ENS = 30
NUM_WINDOWS = 1
WINDOW = 0
SEED = 42
TRUTH_FAST_WEIGHTS = [1.0, 1.0, 0.1, 0.1]


def make_obs_j_indices(NO, J_truth, J_obs):
    if J_obs is None or J_obs >= J_truth:
        return None
    X_idx = list(range(NO))
    Y_idx = []
    for k in range(NO):
        for j in range(J_obs):
            Y_idx.append(NO + k * J_truth + j)
    return X_idx + Y_idx


obs_indices = make_obs_j_indices(NO, J_TRUTH, OBS_J)
obs_dim = len(obs_indices)
s1_state_dim = NO + NO * S1_J

base_cfg = Lorenz96Config(
    dt=DT, T_max=T_MAX, obs_interval=5, R_var=0.5, B_var=2.0,
    num_windows=NUM_WINDOWS, window_spacing=NUM_WINDOWS,
    spinup_steps=5000, seed=SEED,
    NO=NO, J=J_TRUTH, h=1.0, hx=1.0, eps=0.1,
    F_true=8.0, F_da=8.0,
    gamma=0.05, W_L_bar=0.0, c1=1.0, c2=0.1,
    sigma_0=0.08, sigma_L=0.20,
    tau_eta=5.0, sigma_eta=np.sqrt(0.5),
    param_bias=0.0, forcing_state_bias=0.0,
    obs_var_indices=obs_indices,
    fast_weights=TRUTH_FAST_WEIGHTS,
)

dynamics_truth = Lorenz96Dynamics(dt=DT, coupling_exponent=1.6, fast_weights=TRUTH_FAST_WEIGHTS)

s0_ds = RandomParamLorenz96Dataset(base_cfg, param_noise=0.2, dynamics=dynamics_truth)
s1_backend_cfg = Lorenz96Config(**{**base_cfg.__dict__, "case": 1, "seed": 131, "num_windows": NUM_WINDOWS})
s1_ds = RandomBiasLorenz96Dataset(s1_backend_cfg, param_noise=0.2, dynamics=dynamics_truth)

s0_obs_op = ObsOperator(NO + NO * J_TRUTH, obs_indices)
s1_obs_op = ObsOperator(NO + NO * S1_J, None)

s1_dynamics = Lorenz96Dynamics(dt=DT, NO=NO, J=S1_J, h=1.0, hx=1.0, eps=0.1, coupling_exponent=1.0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

etkf_s0 = ETKF(
    dt=DT, device=device, coupling_exponent=1.6, dynamics=dynamics_truth,
    inflation=INF, obs_operator=s0_obs_op, N_ensemble=N_ENS, NO=8, J=J_TRUTH,
    loc_radius=None,
)
etkf_s1 = ETKF(
    dt=DT, device=device, coupling_exponent=1.0, dynamics=s1_dynamics,
    inflation=INF, obs_operator=s1_obs_op, N_ensemble=N_ENS, NO=8, J=S1_J,
    loc_radius=None,
)

print("Running S0 ETKF...")
w0 = s0_ds[WINDOW]
obs_s0 = w0["obs"].to(device)
mask_s0 = w0["obs_mask"].to(device)
force_s0 = w0["forcing_true"].to(device)
result_s0 = etkf_s0.assimilate(obs_s0, mask_s0, force_s0, w0["true_state"], F=base_cfg.F_da)

print("Running S1 ETKF...")
w1 = s1_ds[WINDOW]
obs_s1 = w1["obs"].to(device)
mask_s1 = w1["obs_mask"].to(device)
force_s1 = w1["forcing_true"].to(device)
result_s1 = etkf_s1.assimilate(obs_s1, mask_s1, force_s1, w1["true_state"], F=base_cfg.F_da)

N = NUM_STEPS
truth_40 = w0["true_state"].numpy()
truth_24 = truth_40[..., obs_indices]
analysis_s0 = result_s0.trajectory
analysis_s1 = result_s1.trajectory

time = np.linspace(0, T_MAX, N)

var_groups = [
    ("Slow X[0]", 0, "X₁"),
    ("Fast Y1[0]", 8, "Y₁¹ (observed)"),
    ("Fast Y2[0]", 16, "Y₂¹ (observed)"),
]

print("Plotting...")
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.28, wspace=0.30)

for col, (vname, v_idx, vlabel) in enumerate(var_groups):
    ax_s0 = fig.add_subplot(gs[0, col])
    ax_s1 = fig.add_subplot(gs[1, col])

    truth_var = truth_24[:, v_idx]

    rmse_s0 = np.sqrt(np.mean((analysis_s0[:, v_idx] - truth_var) ** 2))
    rmse_s1 = np.sqrt(np.mean((analysis_s1[:, v_idx] - truth_var) ** 2))

    ax_s0.plot(time, truth_var, "k-", lw=1.0, alpha=0.8, label="Truth")
    ax_s0.plot(time, analysis_s0[:, v_idx], "-", lw=1.0, alpha=0.85, label=f"S0 (RMSE={rmse_s0:.3f})")
    ax_s0.set_title(f"S0 — {vlabel}", fontsize=10)
    ax_s0.set_ylabel("Value", fontsize=9)
    ax_s0.legend(fontsize=7, loc="upper right")
    ax_s0.grid(True, alpha=0.3, ls="--")
    ax_s0.set_xlim(0, T_MAX)

    ax_s1.plot(time, truth_var, "k-", lw=1.0, alpha=0.8, label="Truth")
    ax_s1.plot(time, analysis_s1[:, v_idx], "-", lw=1.0, alpha=0.85, label=f"S1 (RMSE={rmse_s1:.3f})")
    ax_s1.set_title(f"S1 — {vlabel}", fontsize=10)
    ax_s1.set_xlabel("Time (model units)", fontsize=9)
    ax_s1.set_ylabel("Value", fontsize=9)
    ax_s1.legend(fontsize=7, loc="upper right")
    ax_s1.grid(True, alpha=0.3, ls="--")
    ax_s1.set_xlim(0, T_MAX)

fig.suptitle("L96 S0 vs S1 reconstruction — ETKF (N=30, inf=1.1, truth fast weights=[1,1,0.1,0.1])",
             fontsize=11, y=0.98)

out_path = os.path.join(FIGS_DIR, "l96_reconstruction_comparison.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")

s0_rmse_all = np.sqrt(np.mean((analysis_s0 - truth_24) ** 2, axis=0))
s1_rmse_all = np.sqrt(np.mean((analysis_s1 - truth_24) ** 2, axis=0))
print(f"\nS0 mean RMSE (24 vars): {np.mean(s0_rmse_all):.4f}")
print(f"S1 mean RMSE (24 vars): {np.mean(s1_rmse_all):.4f}")

var = np.var(truth_24, axis=0)
s0_ev = 1.0 - np.mean(s0_rmse_all ** 2) / np.mean(var)
s1_ev = 1.0 - np.mean(s1_rmse_all ** 2) / np.mean(var)
print(f"S0 EV: {s0_ev:.4f}")
print(f"S1 EV: {s1_ev:.4f}")