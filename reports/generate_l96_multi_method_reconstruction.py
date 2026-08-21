#!/usr/bin/env python3
"""
S0/S1 multi-method reconstruction figure using cached dataset_ens200.pt.
Loads window 0 from the cached dataset, runs ETKF / EnKF / Strong-4DVar,
and produces an 8-row × 3-column grid:

   Rows 0-3: S0 (Truth+obs | ETKF | EnKF | Strong-4DVar)
   Rows 4-7: S1 (Truth+obs | ETKF | EnKF | Strong-4DVar)
   Columns:  X₁ | Y₁¹ | Y₂¹

Usage:
    conda run -n fdv python reports/generate_l96_multi_method_reconstruction.py
"""

import os
import sys
import time
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evaluation.baselines import ETKF, EnKF, Strong4DVar, ObsOperator
from models.lorenz96_dynamics import Lorenz96Dynamics

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
WINDOW = 0
SEED = 42
TRUTH_FAST_WEIGHTS = [1.0, 1.0, 0.1, 0.1]
CACHED_DATASET = os.path.join(os.path.dirname(__file__), "..", "experiments", "dataset_ens200.pt")

STRONG4DVAR_KWARGS = dict(max_iter=10, lr=0.2, da_window_steps=500)


def make_obs_j_indices(NO, J_truth, J_obs):
    if J_obs is None or J_obs >= J_truth:
        return None
    X_idx = list(range(NO))
    Y_idx = []
    for k in range(NO):
        for j in range(J_obs):
            Y_idx.append(NO + k * J_truth + j)
    return X_idx + Y_idx


obs_indices = make_obs_j_indices(NO, J_TRUTH, OBS_J)  # 24 indices
obs_to_flat = np.arange(24)  # remap obs_indices order → flat 24D (first 24 of truth_40)
for k in range(8):
    obs_to_flat[8 + 2 * k] = 8 + k   # Y1[k] in obs → flat Y1[0..7] at 8+k
    obs_to_flat[9 + 2 * k] = 16 + k  # Y2[k] in obs → flat Y2[0..7] at 16+k
s0_state_dim = NO + NO * J_TRUTH  # 40
s1_state_dim = NO + NO * S1_J     # 24

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("Loading cached dataset...")
data = torch.load(CACHED_DATASET, map_location="cpu")

w0 = data["s0"][WINDOW]
w1 = data["s1"][WINDOW]

truth_s0_40 = w0["true_state"].numpy()
truth_s1_40 = w1["true_state"].numpy()
obs_s0_24 = w0["obs"]
obs_s1_24 = w1["obs"]
mask_s0 = w0["obs_mask"]
mask_s1 = w1["obs_mask"]
force_s0 = w0["forcing_true"]
force_s1 = w1["forcing_true"]

# Remap observations from obs_indices ordering to flat 24D (first 24 of truth_40)
# Remap observations to flat 24D ordering for plotting; DA methods still use obs_indices order
obs_s0_flat = obs_s0_24[..., obs_to_flat]
obs_s1_flat = obs_s1_24[..., obs_to_flat]

truth_s0_24 = truth_s0_40[..., :24]
truth_s1_24 = truth_s1_40[..., :24]

dynamics_truth = Lorenz96Dynamics(dt=DT, coupling_exponent=1.6, fast_weights=TRUTH_FAST_WEIGHTS)
s1_dynamics = Lorenz96Dynamics(dt=DT, NO=NO, J=S1_J, h=1.0, hx=1.0, eps=0.1, coupling_exponent=1.0)

obs_op_s0 = ObsOperator(s0_state_dim, obs_indices)
obs_op_s1 = ObsOperator(s1_state_dim, None)

# ── Instantiate methods ──────────────────────────────────────────────
etkf_s0 = ETKF(N_ensemble=N_ENS, R_var=0.5, inflation=INF, dt=DT, device=device,
               coupling_exponent=1.6, dynamics=dynamics_truth,
               obs_operator=obs_op_s0, loc_radius=None, NO=NO, J=J_TRUTH)
enKF_s0 = EnKF(N_ensemble=N_ENS, R_var=0.5, inflation=INF, dt=DT, device=device,
               coupling_exponent=1.6, dynamics=dynamics_truth,
               obs_operator=obs_op_s0, loc_radius=None, NO=NO, J=J_TRUTH)
s4dvar_s0 = Strong4DVar(R_var=0.5, dt=DT, device=device,
                        coupling_exponent=1.6, dynamics=dynamics_truth,
                        obs_operator=obs_op_s0, **STRONG4DVAR_KWARGS)

etkf_s1 = ETKF(N_ensemble=N_ENS, R_var=0.5, inflation=INF, dt=DT, device=device,
               coupling_exponent=1.0, dynamics=s1_dynamics,
               obs_operator=obs_op_s1, loc_radius=None, NO=NO, J=S1_J)
enKF_s1 = EnKF(N_ensemble=N_ENS, R_var=0.5, inflation=INF, dt=DT, device=device,
               coupling_exponent=1.0, dynamics=s1_dynamics,
               obs_operator=obs_op_s1, loc_radius=None, NO=NO, J=S1_J)
s4dvar_s1 = Strong4DVar(R_var=0.5, dt=DT, device=device,
                        coupling_exponent=1.0, dynamics=s1_dynamics,
                        obs_operator=obs_op_s1, **STRONG4DVAR_KWARGS)

F_da_s0 = 8.0
F_da_s1 = 8.0

# ── Run all methods ──────────────────────────────────────────────────
def run_method(method, label, case, obs, mask, force, truth_40, truth_24, obs_indices):
    print(f"  Running {label} ({case})...", end=" ", flush=True)
    t0 = time.time()
    result = method.assimilate(
        obs.to(device), mask.to(device), force.to(device), truth_40.to(device),
        F=F_da_s0 if case == "S0" else F_da_s1)
    elapsed = time.time() - t0
    analysis = result.trajectory
    if analysis.shape[-1] != truth_24.shape[-1]:
        analysis = analysis[..., :truth_24.shape[-1]]
    rmse = np.sqrt(np.mean((analysis - truth_24) ** 2, axis=0))
    print(f"{elapsed:.1f}s  mean RMSE={np.mean(rmse):.4f}")
    return analysis, rmse


print("\nS0 methods:")
obs_s0_d = obs_s0_24.to(device)
mask_s0_d = mask_s0.to(device)
force_s0_d = force_s0.to(device)
truth_s0_40_d = w0["true_state"].to(device)

an_s0_truth = truth_s0_24
an_s0_etkf, _ = run_method(etkf_s0, "ETKF", "S0", obs_s0_d, mask_s0_d, force_s0_d, truth_s0_40_d, truth_s0_24, obs_indices)
an_s0_enkf, _ = run_method(enKF_s0, "EnKF", "S0", obs_s0_d, mask_s0_d, force_s0_d, truth_s0_40_d, truth_s0_24, obs_indices)
an_s0_4dvar, _ = run_method(s4dvar_s0, "4DVar", "S0", obs_s0_d, mask_s0_d, force_s0_d, truth_s0_40_d, truth_s0_24, obs_indices)

print("\nS1 methods:")
obs_s1_d = obs_s1_24.to(device)
mask_s1_d = mask_s1.to(device)
force_s1_d = force_s1.to(device)
truth_s1_40_d = w1["true_state"].to(device)

an_s1_truth = truth_s1_24
an_s1_etkf, _ = run_method(etkf_s1, "ETKF", "S1", obs_s1_d, mask_s1_d, force_s1_d, truth_s1_40_d, truth_s1_24, obs_indices)
an_s1_enkf, _ = run_method(enKF_s1, "EnKF", "S1", obs_s1_d, mask_s1_d, force_s1_d, truth_s1_40_d, truth_s1_24, obs_indices)
an_s1_4dvar, _ = run_method(s4dvar_s1, "4DVar", "S1", obs_s1_d, mask_s1_d, force_s1_d, truth_s1_40_d, truth_s1_24, obs_indices)

# ── Build observations for overlay ───────────────────────────────────
obs_times = np.where(mask_s0.numpy())[0]  # same obs pattern for S0/S1
obs_time_axis = obs_times * DT

time = np.linspace(0, T_MAX, NUM_STEPS)

var_groups = [
    (0, "X₁"),
    (8, "Y₁¹ (observed)"),
    (16, "Y₂¹ (observed)"),
]

# ── Plot ─────────────────────────────────────────────────────────────
print("\nPlotting...")
fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(8, 3, figure=fig, hspace=0.35, wspace=0.28)

methods_s0 = [
    ("Truth + obs", an_s0_truth),
    ("ETKF",        an_s0_etkf),
    ("EnKF",        an_s0_enkf),
    ("Strong-4DVar", an_s0_4dvar),
]
methods_s1 = [
    ("Truth + obs", an_s1_truth),
    ("ETKF",        an_s1_etkf),
    ("EnKF",        an_s1_enkf),
    ("Strong-4DVar", an_s1_4dvar),
]

colors = ["black", "#1f77b4", "#d62728", "#2ca02c"]
obs_color_s0 = "#ff7f0e"
obs_color_s1 = "#9467bd"

for row, (label, traj) in enumerate(methods_s0):
    for col, (v_idx, vlabel) in enumerate(var_groups):
        ax = fig.add_subplot(gs[row, col])
        truth_var = an_s0_truth[:, v_idx]
        rmse = np.sqrt(np.mean((traj[:, v_idx] - truth_var) ** 2))
        ax.plot(time, truth_var, "k-", lw=0.8, alpha=0.7, label="Truth")
        if row == 0:
            obs_var = obs_s0_flat[obs_times, v_idx].numpy()
            ax.scatter(obs_time_axis, obs_var, s=6, c=obs_color_s0, alpha=0.6, label="Obs", zorder=3)
        else:
            ax.plot(time, traj[:, v_idx], "-", lw=1.0, alpha=0.85,
                    color=colors[row], label=f"{label} (RMSE={rmse:.3f})")
        if row == 0:
            ax.set_title(f"S0 — {vlabel}", fontsize=10, fontweight="bold")
        else:
            ax.set_title(f"S0 {label} — {vlabel}", fontsize=9)
        if col == 0:
            ax.set_ylabel("Value", fontsize=8)
        if row == 7 or row == 3:
            ax.set_xlabel("Time (tu)", fontsize=8)
        if row == 0:
            ax.legend(fontsize=6, loc="upper right", ncol=2)
        elif row <= 3:
            ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.2, ls="--")
        ax.set_xlim(0, T_MAX)

for row, (label, traj) in enumerate(methods_s1):
    for col, (v_idx, vlabel) in enumerate(var_groups):
        ax = fig.add_subplot(gs[4 + row, col])
        truth_var = an_s1_truth[:, v_idx]
        rmse = np.sqrt(np.mean((traj[:, v_idx] - truth_var) ** 2))
        ax.plot(time, truth_var, "k-", lw=0.8, alpha=0.7, label="Truth")
        if row == 0:
            obs_var = obs_s1_flat[obs_times, v_idx].numpy()
            ax.scatter(obs_time_axis, obs_var, s=6, c=obs_color_s1, alpha=0.6, label="Obs", zorder=3)
        else:
            ax.plot(time, traj[:, v_idx], "-", lw=1.0, alpha=0.85,
                    color=colors[row], label=f"{label} (RMSE={rmse:.3f})")
        if row == 0:
            ax.set_title(f"S1 — {vlabel}", fontsize=10, fontweight="bold")
        else:
            ax.set_title(f"S1 {label} — {vlabel}", fontsize=9)
        if col == 0:
            ax.set_ylabel("Value", fontsize=8)
        if row == 3:
            ax.set_xlabel("Time (tu)", fontsize=8)
        if row == 0:
            ax.legend(fontsize=6, loc="upper right", ncol=2)
        elif row <= 3:
            ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.2, ls="--")
        ax.set_xlim(0, T_MAX)

fig.suptitle(
    "L96 S0 vs S1 — All DA Methods (cached dataset, window 0, "
    f"N={N_ENS}, inf={INF}, truth fast weights={TRUTH_FAST_WEIGHTS})",
    fontsize=11, y=0.99)

out_path = os.path.join(FIGS_DIR, "l96_multi_method_reconstruction.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")

# ── Print summary table ──────────────────────────────────────────────
results_summary = [
    ("S0", "ETKF",        an_s0_etkf,  an_s0_truth),
    ("S0", "EnKF",        an_s0_enkf,  an_s0_truth),
    ("S0", "Strong-4DVar", an_s0_4dvar, an_s0_truth),
    ("S1", "ETKF",        an_s1_etkf,  an_s1_truth),
    ("S1", "EnKF",        an_s1_enkf,  an_s1_truth),
    ("S1", "Strong-4DVar", an_s1_4dvar, an_s1_truth),
]
print("\n" + "=" * 72)
print(f"{'Case':<6} {'Method':<15} {'X₁ RMSE':>9} {'Y₁¹ RMSE':>9} {'Y₂¹ RMSE':>9} {'Mean RMSE':>9} {'Mean EV':>9}")
print("=" * 72)
for case, label, an, tr in results_summary:
    r1 = np.sqrt(np.mean((an[:, 0] - tr[:, 0]) ** 2))
    r2 = np.sqrt(np.mean((an[:, 8] - tr[:, 8]) ** 2))
    r3 = np.sqrt(np.mean((an[:, 16] - tr[:, 16]) ** 2))
    mean_rmse = np.mean(np.sqrt(np.mean((an - tr) ** 2, axis=0)))
    ev = 1.0 - np.mean((an - tr) ** 2) / np.mean(np.var(tr, axis=0))
    print(f"{case:<6} {label:<15} {r1:>9.4f} {r2:>9.4f} {r3:>9.4f} {mean_rmse:>9.4f} {ev:>9.4f}")
print("=" * 72)