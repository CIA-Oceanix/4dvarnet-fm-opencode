"""Filter/smoother identity and ensemble-size check of the L96 ETKS.

The ETKS returns only its smoothed ensemble; its internal ETKF pass is captured
here (``ETKS._smooth_result`` wrapped), so filter and smoother share one
ensemble realisation. At N members, benchmark
inflation (1.15 / 2.5), `correct` retro-inflation, full window. For each case:
RMSE (report convention), pooled spread/RMSE, and the filter/smoother identity
terms MSE_F - MSE_S, var_F - var_S, E|m_F - m_S|^2 (pooled, 24D observed space).

usage (from a master checkout):
  etks_ensemble_check.py <val_regular|val_rlayout|test_regular|test_rlayout> <N> <out.json> [lambda case]
(``lambda case``: one inflation for one case, s0 or s1; default both cases at 1.15 / 2.5)
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
import evaluation.baselines as BL  # noqa: E402
from evaluation.run_l96 import EXP_DIR, make_fast_ring_fill, make_obs_j_indices, run_and_cache_baselines  # noqa: E402

SHARED = "/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments"
SETS = {"val_regular": "l96_valset_regular_w50.pt", "val_rlayout": "l96_valset_rlayout_n10-100_k4-16_w50.pt",
        "test_regular": "l96_datasets_obsj2_int100_nwin200.pt", "test_rlayout": "l96_testset_rlayout_n10-100_k4-16_w200_d1.pt"}
valset, N, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
LAM, ONLY = (float(sys.argv[4]), sys.argv[5]) if len(sys.argv) > 5 else (None, None)
FILTER = []
_orig = BL.ETKS._smooth_result


def _capture(self, result, *a, **k):
    FILTER.append((result.trajectory.copy(), np.asarray(result.ensemble_variance).copy()))
    return _orig(self, result, *a, **k)


BL.ETKS._smooth_result = _capture
INF = {"s0": 1.15, "s1": 2.5} if LAM is None else {"s0": LAM, "s1": LAM}
CASES = ("s0", "s1") if ONLY is None else (ONLY,)

datasets = torch.load(os.path.join(SHARED, SETS[valset]), weights_only=False)
datasets = {k: v for k, v in datasets.items() if k in tuple(f"test_{c}" for c in CASES)}
if os.environ.get("N_WIN"):
    for v in datasets.values():
        v.windows = v.windows[:int(os.environ["N_WIN"])]
idx = np.array(make_obs_j_indices(8, 4, 2))
sel = lambda a: a[..., idx] if a.shape[-1] > len(idx) else a  # noqa: E731
cfg = {"inflation": INF, "N_ensemble": N}
kw = dict(obs_j=2, obs_interval=100, fw_randomized=True, da_fast_weights=True)
if valset.endswith("rlayout"):
    cfg.update(init_fill=make_fast_ring_fill(8, 4, 2), R_var=0.5)
    kw["obs_interval"] = None
suffix = f"_etksN_{valset}_N{N}" + ("" if LAM is None else f"_{ONLY}_lam{LAM:.2f}") + (f"_w{os.environ['N_WIN']}" if os.environ.get("N_WIN") else "")
run_and_cache_baselines(datasets, torch.device("cuda"), batch_size=200, da_window_steps=500, suffix=suffix,
                        etks_config=dict(cfg), exclude_methods=["Weak-4DVar", "Strong-4DVar", "EnKF", "ETKF"], **kw)
paths = glob.glob(os.path.join(EXP_DIR, f"l96_baselines_trajectories_dws500{suffix}_*obsj2*.npz"))
if paths:
    assert len(paths) == 1, paths
    z = dict(np.load(paths[0]))
else:
    z = {}
    for c in CASES:
        per = glob.glob(os.path.join(EXP_DIR, f"l96_baselines_trajs_dws500{suffix}_*_{c}_ETKS.npz"))
        assert len(per) == 1, per
        z.update({f"{c}_ETKS_{k}": v for k, v in np.load(per[0]).items()})
nw = len(datasets[f"test_{CASES[0]}"])
PER_WINDOW = {}
out = {"valset": valset, "N": N, "inflation": INF, "n_windows": nw, "cases": {}}
assert len(FILTER) == len(CASES) * nw, (len(FILTER), nw)
for ci, c in enumerate(CASES):
    tr = np.stack([w["true_state"].numpy()[:, idx] for w in datasets[f"test_{c}"]]).astype(np.float64)
    ft = np.stack([f[0] for f in FILTER[ci * nw:(ci + 1) * nw]])
    fv = np.stack([f[1] for f in FILTER[ci * nw:(ci + 1) * nw]])
    r = {}
    for m, (tt, vv) in {"ETKF": (ft, fv), "ETKS": (z[f"{c}_ETKS_trajectories"], z[f"{c}_ETKS_ensemble_variance"])}.items():
        t = sel(np.asarray(tt)).astype(np.float64)
        v = np.clip(sel(np.asarray(vv)), 0, None).astype(np.float64)
        se = (t - tr) ** 2
        r[m] = {"t": t, "mse": float(se.mean()), "var": float(v.mean()), "rmse": float(np.sqrt(se.mean(1)).mean())}
        PER_WINDOW[f"{c}_{m}_rmse"] = np.sqrt(se.mean(1))
        PER_WINDOW[f"{c}_{m}_spread"] = np.sqrt(v).mean(1)
    out["cases"][c] = {
        "rmse_F": r["ETKF"]["rmse"], "rmse_S": r["ETKS"]["rmse"],
        "pooled_F": float(np.sqrt(r["ETKF"]["var"] / r["ETKF"]["mse"])),
        "pooled_S": float(np.sqrt(r["ETKS"]["var"] / r["ETKS"]["mse"])),
        "mse_gap": r["ETKF"]["mse"] - r["ETKS"]["mse"], "var_gap": r["ETKF"]["var"] - r["ETKS"]["var"],
        "mean_shift_sq": float(((r["ETKF"]["t"] - r["ETKS"]["t"]) ** 2).mean()),
    }
for c in CASES:
    PER_WINDOW[f"{c}_ETKS_crps"] = sel(np.asarray(z[f"{c}_ETKS_crps"]))
np.savez_compressed(out_path.replace(".json", "_per_window.npz"), **PER_WINDOW)
json.dump(out, open(out_path, "w"), indent=1)
print(json.dumps(out))
for f in glob.glob(os.path.join(EXP_DIR, f"*{suffix}_*")):
    os.remove(f)
