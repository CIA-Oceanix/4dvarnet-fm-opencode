"""Gaussian FM score of the P1 L96 rows under a scalar variance rescaling c * var.

For each tau, reports the score at c = 1, the optimal c* on a log grid
[1/8, 8], the score at c*, and the score at the Silverman-KDE variance
inflation (to separate KDE shape effects from its extra spread).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reports.l96.generate_p1_l96_benchmark import DA_SOURCES, make_obs_j_indices  # noqa: E402
from reports.l96.probe_fm_score import (  # noqa: E402
    DA_DIR, ENSEMBLES, HERE, SCALE_TAUS, SCALES, accumulate_scales, finish_scales, norm_stats,
)

TAUS = SCALE_TAUS
accumulate = accumulate_scales
finish = finish_scales


def main(case, name):
    mu, sd = norm_stats()
    acc = np.zeros((len(TAUS), len(SCALES) + 1, 24))
    mse, spread2, n = np.zeros(24), np.zeros(24), 0
    if name in ENSEMBLES:
        z = np.load(HERE / ENSEMBLES[name] / f"members_{case}.npz")
        mem, tr = z["members"], z["truth"]
        for c0 in range(0, tr.shape[0], 20):
            m = (mem[c0:c0 + 20].astype(np.float64) - mu[:, None]) / sd[:, None]
            t = (tr[c0:c0 + 20].astype(np.float64) - mu) / sd
            err, var = t - m.mean(-1), m.var(-1, ddof=1)
            accumulate(err, var, acc)
            mse += (err ** 2).sum((0, 1))
            spread2 += var.sum((0, 1))
            n += t.shape[0] * t.shape[1]
    else:
        fn, key, _ = DA_SOURCES[name]
        z = np.load(DA_DIR / fn)
        idx = make_obs_j_indices()
        sel = (lambda a: a[..., idx] if a.shape[-1] > 24 else a)  # noqa: E731
        tr = np.load(HERE / ENSEMBLES["VanillaCFM-M"] / f"members_{case}.npz")["truth"].astype(np.float64)
        mean = sel(z[f"{case}_{key}_trajectories"].astype(np.float64))
        var = np.clip(sel(z[f"{case}_{key}_ensemble_variance"].astype(np.float64)), 0, None)
        err, var = (tr - mean) / sd, var / sd ** 2
        accumulate(err, var, acc)
        mse, spread2, n = (err ** 2).sum((0, 1)), var.sum((0, 1)), err.shape[0] * err.shape[1]
    assert abs(SCALES[20] - 1) < 1e-9
    return finish(acc, n, mse, spread2)


if __name__ == "__main__":
    case, name, out = sys.argv[1], sys.argv[2], sys.argv[3]
    Path(out).write_text(json.dumps(main(case, name), indent=1))
