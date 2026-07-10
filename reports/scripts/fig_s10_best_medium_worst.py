import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP = "experiments/S10_vanilla_cfm_s0s1_small_tau0"
OUT = "reports/outputs/figs/s10_best_medium_worst.png"
FMTS = ['.png', '.pdf']

s0 = np.load(f"{EXP}/trajectories_s0.npz")
s1 = np.load(f"{EXP}/trajectories_s1.npz")

names = {"S0": s0, "S1": s1}

# Per-window RMSE (mean across X,Y,Z)
rmse_s0 = np.sqrt(np.mean((s0["trajectories"] - s0["truths"])**2, axis=(1, 2)))
rmse_s1 = np.sqrt(np.mean((s1["trajectories"] - s1["truths"])**2, axis=(1, 2)))

def pick_indices(rmse):
    args = np.argsort(rmse)
    best = args[0]
    worst = args[-1]
    median_idx = np.argmin(np.abs(np.argsort(rmse) - len(rmse)//2))
    medium = args[median_idx]
    return best, medium, worst

labels = {"best": "Best", "medium": "Medium", "worst": "Worst"}
colors = {"best": "#2ca02c", "medium": "#ff7f0e", "worst": "#d62728"}

for scenario, data in names.items():
    traj, truth = data["trajectories"], data["truths"]
    rmse = np.sqrt(np.mean((traj - truth)**2, axis=(1, 2)))
    best_idx, medium_idx, worst_idx = pick_indices(rmse)

    fig, axes = plt.subplots(3, 3, figsize=(18, 10), sharex=True)
    comp_labels = ["X", "Y", "Z"]
    t = np.arange(300)

    for col, (case, idx) in enumerate([("best", best_idx), ("medium", medium_idx), ("worst", worst_idx)]):
        c = colors[case]
        lbl = labels[case]
        for row in range(3):
            ax = axes[row, col]
            ax.plot(t, truth[idx, :, row], 'k-', linewidth=1.5, label='Truth')
            ax.plot(t, traj[idx, :, row], '-', color=c, linewidth=1.5, label=lbl)
            ax.set_ylabel(comp_labels[row], fontsize=11)
            if row == 2:
                ax.set_xlabel("Time step", fontsize=11)
            if row == 0:
                ax.set_title(f"{lbl} (RMSE={rmse[idx]:.3f})", fontsize=12, color=c)
            ax.grid(True, alpha=0.3)

    fig.suptitle(f"S10 Small VanillaCFM τ=0 — {scenario} reconstruction samples", fontsize=14, y=1.01)
    plt.tight_layout()
    for ext in FMTS:
        path = OUT.replace(".png", f"_{scenario.lower()}{ext}")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved {path}")
    plt.close(fig)

# Combined figure: S0 + S1 side-by-side
fig, axes = plt.subplots(3, 6, figsize=(24, 10), sharex=True)
t = np.arange(300)

for sc_idx, (scenario, data) in enumerate(names.items()):
    traj, truth = data["trajectories"], data["truths"]
    rmse = np.sqrt(np.mean((traj - truth)**2, axis=(1, 2)))
    best_idx, medium_idx, worst_idx = pick_indices(rmse)

    for col_off, (case, idx) in enumerate([("best", best_idx), ("medium", medium_idx), ("worst", worst_idx)]):
        col = col_off + sc_idx * 3
        c = colors[case]
        lbl = labels[case]
        for row in range(3):
            ax = axes[row, col]
            ax.plot(t, truth[idx, :, row], 'k-', linewidth=1.2, label='Truth')
            ax.plot(t, traj[idx, :, row], '-', color=c, linewidth=1.2, label=lbl)
            if row == 2:
                ax.set_xlabel("Time step", fontsize=10)
            if col > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(comp_labels[row], fontsize=11)
            if row == 0:
                ax.set_title(f"{scenario} {lbl}\nRMSE={rmse[idx]:.3f}", fontsize=11, color=c)
            ax.grid(True, alpha=0.3)

fig.suptitle("S10 Small VanillaCFM τ=0 — Best / Medium / Worst reconstructions", fontsize=14, y=1.01)
plt.tight_layout()
for ext in FMTS:
    path = OUT.replace(".png", f"_combined{ext}")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved {path}")
plt.close(fig)

print(f"\nBest/Medium/Worst indices per scenario:")
for scenario, data in names.items():
    rmse = np.sqrt(np.mean((data["trajectories"] - data["truths"])**2, axis=(1, 2)))
    b, m, w = pick_indices(rmse)
    print(f"  {scenario}: best={b} (RMSE={rmse[b]:.4f}), medium={m} (RMSE={rmse[m]:.4f}), worst={w} (RMSE={rmse[w]:.4f})")