"""Pooled calibration of an L96 ensemble eval, members kept in memory.

Runs an eval script (eval_neural_l96 / eval_sda_l96 /
eval_sda_directunet_hybrid_l96) with its members-saving hook replaced by one
writing pooled spread/RMSE statistics (pooled_<case>.json next to --output):
E[var] vs E[squared error of the ensemble mean] over windows x time x
channels, per group and per time bin, plus the benchmark per-window ratio for
comparison. No members_*.npz is written.

usage (from the repo root, PYTHONPATH=.):
  python reports/l96/probe_pooled_calibration.py eval_neural_l96 --checkpoint ... --output out/eval.json

Outputs of the 2026-09-28 run: reports/l96/outputs/pooled_calibration/.
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np

NO = 8


def pooled_stats(members: np.ndarray, truth: np.ndarray) -> dict:
    W, T, D, M = members.shape
    se = np.zeros((W, D))
    var = np.zeros((W, D))
    std_tm = np.zeros((W, D))
    tb = [0, 100, 300, 1000, 2000, 2900, 3000]
    se_t = np.zeros(len(tb) - 1)
    var_t = np.zeros(len(tb) - 1)
    for w in range(W):
        m = members[w].astype(np.float64)
        mu = m.mean(-1)
        e2 = (mu - truth[w]) ** 2
        v = m.var(-1, ddof=1)
        se[w], var[w], std_tm[w] = e2.mean(0), v.mean(0), np.sqrt(v).mean(0)
        for i, (a, b) in enumerate(zip(tb[:-1], tb[1:])):
            se_t[i] += e2[a:b].mean() / W
            var_t[i] += v[a:b].mean() / W
    cal = M / (M + 1)
    out = {"n_members": M}
    for g, sl in (("all_obs", slice(None)), ("slow", slice(0, NO)), ("obs_fast", slice(NO, None))):
        s, v, st = se[:, sl], var[:, sl], std_tm[:, sl]
        pooled = float(np.sqrt(v.mean() / s.mean()))
        out[g] = {
            "rmse_pooled": float(np.sqrt(s.mean())),
            "rmse_report": float(np.sqrt(s).mean()),
            "per_window_report": float(st.mean() / np.sqrt(s).mean()),
            "pooled_lb_timemean_std": float(np.sqrt((st ** 2).mean() / s.mean())),
            "pooled": pooled,
            "unexplained": float(1 - pooled ** 2 / cal),
        }
    out["by_time"] = {f"{a}-{b}": float(np.sqrt(var_t[i] / se_t[i])) for i, (a, b) in enumerate(zip(tb[:-1], tb[1:]))}
    return out


def hook(out_dir, case, members, truth, mode="full"):
    path = Path(out_dir) / f"pooled_{case}.json"
    stats = pooled_stats(members, truth)
    path.write_text(json.dumps(stats, indent=1))
    print(f"POOLED {case}: " + json.dumps(stats["all_obs"]), flush=True)
    return str(path)


if __name__ == "__main__":
    mod_name = sys.argv[1]
    sys.argv = [mod_name + ".py"] + sys.argv[2:]
    mod = importlib.import_module(mod_name)
    mod.save_members_or_scores = hook
    mod.members_store.members_dir = lambda out, keep=False, mode="full": Path(out)
    mod.main()
