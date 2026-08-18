#!/usr/bin/env python3
"""
Generate L96 trajectory + observation figures as PNGs for the report annex.

Usage:
    conda run -n fdv python reports/generate_l96_trajectory_figures.py
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.lorenz96_dynamics import Lorenz96Dynamics

FIGS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

DT = 0.001
T_MAX = 3.0
NUM_STEPS = int(T_MAX / DT)
OBS_INTERVAL = 200
R_VAR = 0.5
NO, J = 8, 4
STATE_DIM = NO + NO * J

COUPLING_EXPONENT = 1.6
CMAP = "viridis"

dyn = Lorenz96Dynamics(dt=DT, coupling_exponent=COUPLING_EXPONENT)

print("Generating L96 trajectory...")
true_traj, W_L_true = dyn.generate_full_trajectory(
    num_steps=NUM_STEPS, seed=123, F=8.0,
    spinup_steps=10000, coupling_exponent=COUPLING_EXPONENT,
)
true_traj = true_traj.numpy()

time = np.linspace(0, T_MAX, NUM_STEPS)
obs_indices = np.arange(0, NUM_STEPS, OBS_INTERVAL)
obs_mask = np.zeros(NUM_STEPS, dtype=bool)
obs_mask[obs_indices] = True

rng = np.random.RandomState(124)
noise = rng.randn(len(obs_indices), STATE_DIM) * np.sqrt(R_VAR)
obs_values = np.full((NUM_STEPS, STATE_DIM), np.nan)
obs_values[obs_indices] = true_traj[obs_indices] + noise

slow_idx = np.arange(NO)
fast_idx = np.arange(NO, STATE_DIM)

print("  Field: {:.3f}±{:.3f}".format(true_traj.mean(), true_traj.std()))
print(f"  15 observation times at {obs_indices}")

# ── Figure 1: 2D field heatmap + observation markers ──
print("Plotting figure 1: l96_trajectory_field.png")
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

vmin, vmax = true_traj.min(), true_traj.max()
norm = Normalize(vmin=vmin, vmax=vmax)

# Panel A: Full field
im = axes[0].imshow(true_traj.T, aspect="auto", cmap=CMAP, norm=norm,
                     extent=[0, T_MAX, STATE_DIM - 0.5, -0.5])
for t_idx in obs_indices:
    axes[0].axvline(t_idx * DT, color="white", ls=":", lw=0.6, alpha=0.4)
axes[0].set_ylabel("State dimension", fontsize=10)
axes[0].set_title(f"True L96 state (NO={NO}, J={J}, F=8.0)", fontsize=11)
cb = plt.colorbar(im, ax=axes[0], fraction=0.02, pad=0.03)
cb.set_label("State value", fontsize=9)

# Panel B: Observed values (NaN shown as white)
obs_disp = np.full_like(true_traj, np.nan)
obs_disp[obs_indices] = obs_values[obs_indices]
im2 = axes[1].imshow(obs_disp.T, aspect="auto", cmap=CMAP, norm=norm,
                      extent=[0, T_MAX, STATE_DIM - 0.5, -0.5])
for t_idx in obs_indices:
    axes[1].axvline(t_idx * DT, color="white", ls=":", lw=0.6, alpha=0.4)
axes[1].set_ylabel("State dimension", fontsize=10)
axes[1].set_title(f"Observations (R_var={R_VAR}, every {OBS_INTERVAL} steps = {OBS_INTERVAL*DT:.3f} tu)", fontsize=11)
cb2 = plt.colorbar(im2, ax=axes[1], fraction=0.02, pad=0.03)
cb2.set_label("Observed value", fontsize=9)

# Panel C: Slow variables line plot
slow_true = true_traj[:, :NO]
colors_slow = plt.cm.tab10(np.linspace(0, 1, NO))
for i in range(NO):
    axes[2].plot(time, slow_true[:, i], color=colors_slow[i], lw=1.0, alpha=0.8,
                 label=f"X{i+1}" if i < 6 else None)
obs_y = obs_values[obs_indices]
for i in range(min(NO, 4)):
    axes[2].scatter(time[obs_mask], obs_y[:, i], color=colors_slow[i],
                    s=6, alpha=0.5, zorder=3)
axes[2].set_xlabel("Time (model units)", fontsize=10)
axes[2].set_ylabel("Slow variable value", fontsize=10)
axes[2].set_title("Slow variables (X1..X8) with observations", fontsize=11)
axes[2].legend(fontsize=7, ncol=2, loc="upper right")
axes[2].grid(True, alpha=0.3, ls="--")

plt.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "l96_trajectory_field.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved")

# ── Figure 2: Selected slow variable trajectories with obs ──
print("Plotting figure 2: l96_slow_vars_trajectories.png")
fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=False)
slow_highlight = [0, 1, 2, 3, 4, 5, 6, 7]
for idx, ax in zip(slow_highlight, axes.flat):
    ax.plot(time, slow_true[:, idx], "k-", lw=1.2, alpha=0.8, label="Truth")
    ax.scatter(time[obs_mask], obs_y[:, idx], c="C1", s=12, alpha=0.6,
               zorder=3, label="Obs")
    ax.set_title(f"X{idx+1}", fontsize=10)
    ax.set_xlabel("Time (model units)", fontsize=8)
    ax.set_ylabel("Value", fontsize=8)
    ax.grid(True, alpha=0.3, ls="--")
    ax.legend(fontsize=7, loc="upper right")
plt.suptitle("L96 slow variables — truth and observations (R_var=0.5, 15 obs/window)",
             fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "l96_slow_vars_trajectories.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved")

# ── Figure 3: Hovmöller diagram for selected slow vars ──
print("Plotting figure 3: l96_hovmoller_fast.png")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fast_traj = true_traj[:, NO:].reshape(NUM_STEPS, NO, J)

# Left: slow variables
im_s = axes[0].imshow(slow_true.T, aspect="auto", cmap=CMAP,
                       extent=[0, T_MAX, NO - 0.5, -0.5])
axes[0].set_ylabel("Slow variable index", fontsize=10)
axes[0].set_xlabel("Time (model units)", fontsize=10)
axes[0].set_title("Slow variables X1..X8", fontsize=11)
cb = plt.colorbar(im_s, ax=axes[0], fraction=0.02, pad=0.03)

# Right: fast variables per slow node (show 1st fast var per node)
Y_first = fast_traj[:, :, 0].T  # shape (NO, T)
im_f = axes[1].imshow(Y_first, aspect="auto", cmap=CMAP,
                       extent=[0, T_MAX, NO - 0.5, -0.5])
axes[1].set_ylabel("Slow node index", fontsize=10)
axes[1].set_xlabel("Time (model units)", fontsize=10)
axes[1].set_title(f"Fast variables Y (1st of J={J} per slow node)", fontsize=11)
cb = plt.colorbar(im_f, ax=axes[1], fraction=0.02, pad=0.03)

plt.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "l96_hovmoller_fast.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved")

print(f"\nAll figures saved to {FIGS_DIR}/")