"""Weak-4DVar (L96Weak4DVar) sweep on the L96 benchmark DA setup.

Swaps L96Weak4DVar (whitened controls, gradient clipping, NaN reset, LBFGS) into
run_and_cache_baselines' Weak-4DVar slot, so the case dynamics, observation
operator, per-window params, DA fast weights and 500-step sub-windows are the
benchmark's. ``strong_ref`` runs the benchmark Strong4DVar on the same windows
instead. Regular grid only: L96Weak4DVar's observation cost does not mask
missing channels.

usage: weak4dvar_sweep_l96.py <val_regular|test_regular> <n_windows> <weak|strong|strong_ref> <q_var_scale> <out.json> [max_iter]
(``weak`` / ``strong``: L96Weak4DVar in that mode, LBFGS ``max_iter`` iterations per sub-window, default 10;
``strong_ref``: the benchmark Strong4DVar)
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
import evaluation.run_l96 as R  # noqa: E402
from evaluation.baselines import L96Weak4DVar  # noqa: E402

SHARED = "/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments"
SETS = {"val_regular": "l96_valset_regular_w50.pt", "test_regular": "l96_datasets_obsj2_int100_nwin200.pt"}
dset, n_win, kind, q_scale, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4]), sys.argv[5]
max_iter = int(sys.argv[6]) if len(sys.argv) > 6 else 10
INSTANCES = []


def weak_factory(dt, da_window_steps, device, coupling_exponent, dynamics, obs_operator, **cfg):
    m = L96Weak4DVar(dt=dt, dynamics=dynamics, obs_operator=obs_operator, da_window_steps=da_window_steps,
                     device=device, mode=kind, optimizer="lbfgs", max_iter=max_iter, lr=0.2,
                     q_var_scale=q_scale, b_var_scale=1.0, r_var=0.5)
    INSTANCES.append(m)
    return m


R.Weak4DVar = weak_factory
method = "Strong-4DVar" if kind == "strong_ref" else "Weak-4DVar"
datasets = torch.load(os.path.join(SHARED, SETS[dset]), weights_only=False)
datasets = {k: v for k, v in datasets.items() if k in ("test_s0", "test_s1")}
for v in datasets.values():
    v.windows = v.windows[:n_win]
suffix = f"_w4dsweep_{dset}_w{n_win}_{kind}_q{q_scale:g}_it{max_iter}"
t0 = time.time()
R.run_and_cache_baselines(datasets, torch.device("cuda"), batch_size=200, da_window_steps=500, suffix=suffix,
                          exclude_methods=[m for m in ("ETKF", "EnKF", "Weak-4DVar", "Strong-4DVar") if m != method],
                          obs_j=2, obs_interval=100, fw_randomized=True, da_fast_weights=True)
paths = glob.glob(os.path.join(R.EXP_DIR, f"l96_baselines_trajectories_dws500{suffix}_*.npz"))
assert len(paths) == 1, paths
z = np.load(paths[0])
idx = np.array(R.make_obs_j_indices(8, 4, 2))
key = method.replace("-", "_")
out = {"set": dset, "n_windows": n_win, "method": method, "mode": kind, "max_iter": max_iter,
       "q_var_scale": q_scale if kind == "weak" else None,
       "elapsed_s": time.time() - t0, "n_resets": sum(m.n_resets for m in INSTANCES),
       "n_subwindows": sum(m.n_subwindows for m in INSTANCES), "cases": {}}
for c in ("s0", "s1"):
    tr = np.stack([w["true_state"].numpy()[:, idx] for w in datasets[f"test_{c}"]]).astype(np.float64)
    t = z[f"{c}_{key}_trajectories"].astype(np.float64)
    t = t[..., idx] if t.shape[-1] > len(idx) else t
    se = (t - tr) ** 2
    out["cases"][c] = {"rmse": float(np.sqrt(se.mean(1)).mean()),
                       "rmse_slow": float(np.sqrt(se[..., :8].mean(1)).mean()),
                       "rmse_fast": float(np.sqrt(se[..., 8:].mean(1)).mean())}
json.dump(out, open(out_path, "w"), indent=1)
print(json.dumps(out))
for f in glob.glob(os.path.join(R.EXP_DIR, f"*{suffix}_*")) + glob.glob(os.path.join(R.EXP_DIR, f"*{suffix}.json")):
    os.remove(f)
