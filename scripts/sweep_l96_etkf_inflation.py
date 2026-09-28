"""ETKF inflation sweep on the L96 validation windows (after #291).

One (validation set, inflation) cell per call, both cases. The DA settings copy
the benchmark runs: the regular grid as ``evaluate_all_l96.py`` in
``batch/run_l96_da_crps.sbatch`` (dws 500, per-window fast weights), the random
layout as ``eval_da_random_layout_l96.py`` (fast-ring initial fill, R 0.5,
obs_interval None). Writes RMSE (report convention), analysis-ensemble CRPS and
pooled spread/RMSE per case to ``--out``; the trajectory caches are deleted.

  python scripts/sweep_l96_etkf_inflation.py --valset regular --inflation 1.2 --out x.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evaluation.estimate_metrics import _groups_from_per_window  # noqa: E402
from evaluation.run_l96 import EXP_DIR, make_fast_ring_fill, make_obs_j_indices, run_and_cache_baselines  # noqa: E402

SHARED = "/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments"
VALSETS = {"regular": "l96_valset_regular_w50.pt", "rlayout": "l96_valset_rlayout_n10-100_k4-16_w50.pt"}
N_MEMBERS = 30


def cell_metrics(traj: np.ndarray, var: np.ndarray, crps: np.ndarray, truth: np.ndarray, idx: np.ndarray) -> dict:
    sel = lambda a: a[..., idx] if a.shape[-1] > len(idx) else a  # noqa: E731
    t, v, c = sel(traj).astype(np.float64), np.clip(sel(var), 0, None).astype(np.float64), sel(crps)
    se = (t - truth) ** 2
    pooled = float(np.sqrt(v.mean() / se.mean()))
    return {
        "rmse": float(np.sqrt(se.mean(1)).mean()),
        "crps": float(_groups_from_per_window(c)["all_obs"].mean()),
        "pooled": pooled,
        "per_window": float(np.sqrt(v).mean(1).mean() / np.sqrt(se.mean(1)).mean()),
        "unexplained": float(1 - pooled ** 2 * (N_MEMBERS + 1) / N_MEMBERS),
        "pooled_slow": float(np.sqrt(v[..., :8].mean() / se[..., :8].mean())),
        "pooled_fast": float(np.sqrt(v[..., 8:].mean() / se[..., 8:].mean())),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--valset", choices=list(VALSETS), required=True)
    p.add_argument("--inflation", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-windows", type=int, default=None, help="first N windows only (smoke tests)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = torch.load(os.path.join(SHARED, VALSETS[args.valset]), weights_only=False)
    if args.n_windows:
        for v in datasets.values():
            v.windows = v.windows[:args.n_windows]
    idx = np.array(make_obs_j_indices(8, 4, 2))
    suffix = f"_valsweep_{args.valset}_lam{args.inflation:g}" + (f"_w{args.n_windows}" if args.n_windows else "")
    cfg = {"inflation": args.inflation}
    kw = dict(obs_j=2, obs_interval=100, fw_randomized=True, da_fast_weights=True)
    if args.valset == "rlayout":
        cfg.update(init_fill=make_fast_ring_fill(8, 4, 2), R_var=0.5)
        kw["obs_interval"] = None
    run_and_cache_baselines(datasets, device, batch_size=200, da_window_steps=500, etkf_config=cfg,
                            suffix=suffix, exclude_methods=["Weak-4DVar", "Strong-4DVar", "EnKF"], **kw)

    paths = glob.glob(os.path.join(EXP_DIR, f"l96_baselines_trajectories_dws500{suffix}*.npz"))
    assert len(paths) == 1, paths
    z = np.load(paths[0])
    out = {"valset": args.valset, "inflation": args.inflation, "n_windows": len(datasets["test_s0"]), "cases": {}}
    for c in ("s0", "s1"):
        truth = np.stack([w["true_state"].numpy()[:, idx] for w in datasets[f"test_{c}"]]).astype(np.float64)
        out["cases"][c] = cell_metrics(z[f"{c}_ETKF_trajectories"], z[f"{c}_ETKF_ensemble_variance"],
                                       z[f"{c}_ETKF_crps"], truth, idx)
    z.close()
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out))
    for f in glob.glob(os.path.join(EXP_DIR, f"*{suffix}*")):
        os.remove(f)


if __name__ == "__main__":
    main()
