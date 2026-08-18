"""
Compare full-window RMSE vs trimmed RMSE (skip first/last 50 steps)
for S0/S1 baselines from the saved job 42642 trajectories.

Computes RMSE per-window then averages (matching the report methodology).
Usage: python tests/compare_rmse_slices.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

EXP_DIR = "experiments"
TRAJ_PATH = os.path.join(EXP_DIR, "baselines_trajectories_dws50_s0s1_inf2.0_etkf_inf2.0.npz")
SLICE_START, SLICE_END = 50, 250  # skip first 50, last 50 of 300-step window

data = np.load(TRAJ_PATH)

cases_methods = [
    ("s0", "Weak-4DVar"), ("s0", "Strong-4DVar"), ("s0", "EnKF"), ("s0", "ETKF"),
    ("s1", "Weak-4DVar"), ("s1", "Strong-4DVar"), ("s1", "EnKF"), ("s1", "ETKF"),
]


def _key(case, method):
    return f"{case}_{method.replace('-', '_').replace(' ', '_')}"


def rmse_per_window(traj, truth, sl=None):
    if sl:
        traj, truth = traj[:, sl], truth[:, sl]
    axis = 1 if traj.ndim == 3 else 0
    return np.sqrt(np.mean((traj - truth) ** 2, axis=axis))


print("=" * 96)
print(f"{'Case/Method':<22} {'Full (0-300) mean':<26} {'Trim (50-250) mean':<26} {'Diff mean':<10} {'Diff X/Y/Z':<14}")
print("=" * 96)

for case, method in cases_methods:
    k = _key(case, method)
    traj = data[f"{k}_trajectories"]
    truth = data[f"{k}_truths"]

    r_full = rmse_per_window(traj, truth)
    m_full = np.mean(r_full, axis=0)

    r_trim = rmse_per_window(traj, truth, slice(SLICE_START, SLICE_END))
    m_trim = np.mean(r_trim, axis=0)

    label = f"{case.upper()}/{method}"
    d = m_trim - m_full
    print(f"{label:<22} "
          f"X={m_full[0]:.4f} Y={m_full[1]:.4f} Z={m_full[2]:.4f}  m={np.mean(m_full):.4f}  "
          f"X={m_trim[0]:.4f} Y={m_trim[1]:.4f} Z={m_trim[2]:.4f}  m={np.mean(m_trim):.4f}  "
          f"{np.mean(d):+.4f}   "
          f"{d[0]:+.4f}/{d[1]:+.4f}/{d[2]:+.4f}")

print("=" * 96)
data.close()