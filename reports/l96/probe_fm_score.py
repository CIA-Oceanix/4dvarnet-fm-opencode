"""FM-operator score of the P1 L96 rows at a few flow times tau.

Scores (normalised units, pooled over windows x time x channels):
  ens   - empirical ensemble operator, univariate (channel x time) blocks
  kde   - Silverman-smoothed ensemble, univariate blocks
  snap  - empirical ensemble, 24-channel snapshot blocks (reports weight ESS)
  gauss - per-element Gaussian N(mean, var), closed form
  point - point estimate (DirectUNet), score = squared error at every tau

DA rows only have mean + per-time variance, so they get ``gauss`` only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.fm_score import ensemble_score, gaussian_score, silverman_h2  # noqa: E402
from reports.l96.generate_p1_l96_benchmark import DA_SOURCES, make_obs_j_indices  # noqa: E402

ROOT = Path("/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments")
HERE = ROOT / "l96/report_inputs/p1_benchmark/here"
DA_DIR = HERE / "da_current_2026-09-29"
TAUS = (0.0, 0.25, 0.5, 0.75, 0.95)
GROUPS = {"slow": slice(0, 8), "fast": slice(8, 24), "all": slice(0, 24)}
ENSEMBLES = {
    "VanillaCFM-M": "A1_vanillacfm_monaiM_l96/ens30_no20",
    "PredictStateCFM-M": "A2_predictstatecfm_monaiM_l96/ens30_no20",
    "SDA1-M": "B4_sda1_monaiM_l96/ens30_gw20",
    "SDA3-fix-M": "A3_sda3fix_monaiM_l96_seed1/ens30_gw20",
}
POINTS = {"DirectUNet-M": "P1_directunet_monaiM_noaug_l96/ens1_no1"}
DA = ("ETKF", "ETKS")


def norm_stats() -> tuple[np.ndarray, np.ndarray]:
    d = torch.load(ROOT / "l96_norm_stats_obsj2.pt", weights_only=False)
    return np.asarray(d["mean"], dtype=np.float64), np.asarray(d["std"], dtype=np.float64)


def pooled(a: np.ndarray) -> dict[str, float]:
    return {g: float(a[..., s].mean()) for g, s in GROUPS.items()}


def noise(chunk: int, shape: tuple[int, ...], k: int) -> np.ndarray:
    return np.random.default_rng(1000 + chunk).standard_normal(shape + (k,))


def score_ensemble(path: Path, mu: np.ndarray, sd: np.ndarray, k: int, chunk: int,
                   ref_truth: np.ndarray) -> dict:
    z = np.load(path)
    members, truth = z["members"], z["truth"]
    assert np.allclose(truth, ref_truth, atol=1e-4), path
    W = truth.shape[0]
    sums = {v: {tau: np.zeros(24) for tau in TAUS} for v in ("ens", "kde", "snap", "gauss")}
    ess_snap = {tau: 0.0 for tau in TAUS}
    n = 0
    for c0 in range(0, W, chunk):
        m = (members[c0:c0 + chunk].astype(np.float64) - mu[:, None]) / sd[:, None]
        t = (truth[c0:c0 + chunk].astype(np.float64) - mu) / sd
        x0 = noise(c0, t.shape, k)
        var = m.var(axis=-1, ddof=1)
        err = t - m.mean(axis=-1)
        h2 = silverman_h2(m)[..., None]
        for tau in TAUS:
            sums["gauss"][tau] += gaussian_score(err, var, tau).sum(axis=(0, 1))
            if tau == 0.0:
                for v in ("ens", "kde", "snap"):
                    sums[v][tau] += (err ** 2).sum(axis=(0, 1))
                ess_snap[tau] += m.shape[-1] * t.shape[0] * t.shape[1]
                continue
            s, _ = ensemble_score(m, t, tau, x0)
            sums["ens"][tau] += s.sum(axis=(0, 1))
            s, _ = ensemble_score(m, t, tau, x0, h2=h2)
            sums["kde"][tau] += s.sum(axis=(0, 1))
            s, e = ensemble_score(m, t, tau, x0, block_axes=(2,))
            sums["snap"][tau] += s.sum(axis=(0, 1))
            ess_snap[tau] += e[..., 0].sum()
        n += t.shape[0] * t.shape[1]
        print(f"  {path.parent.name}/{path.name} windows {c0 + t.shape[0]}/{W}", flush=True)
    out = {v: {str(tau): pooled(sums[v][tau] / n) for tau in TAUS} for v in sums}
    out["snap_ess"] = {str(tau): ess_snap[tau] / n for tau in TAUS}
    return out


def gauss_rows(mean: np.ndarray, var: np.ndarray, truth: np.ndarray, mu: np.ndarray,
               sd: np.ndarray) -> dict:
    err = (truth - mean) / sd
    v = np.clip(var, 0, None) / sd ** 2
    return {"gauss": {str(tau): pooled(gaussian_score(err, v, tau).mean(axis=(0, 1)))
                      for tau in TAUS}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--cases", nargs="+", default=["s0", "s1"])
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    mu, sd = norm_stats()
    idx = make_obs_j_indices()
    results: dict = json.loads(args.out.read_text()) if args.out.exists() else {}
    for case in args.cases:
        truth = np.load(HERE / ENSEMBLES["VanillaCFM-M"] / f"members_{case}.npz")["truth"]
        truth = truth.astype(np.float64)
        res = results.setdefault(case, {})
        for name, rel in POINTS.items():
            if args.only and name not in args.only:
                continue
            z = np.load(HERE / rel / f"estimates_{case}.npz")
            assert np.allclose(z["truth"], truth, atol=1e-4)
            est = z["trajectories"].astype(np.float64)
            res[name] = gauss_rows(est, np.zeros_like(est), truth, mu, sd)
            res[name]["rmse"] = float(np.sqrt(((est - truth) ** 2).mean(axis=1)).mean())
        for name in DA:
            if args.only and name not in args.only:
                continue
            fn, key, _ = DA_SOURCES[name]
            z = np.load(DA_DIR / fn)
            sel = (lambda a: a[..., idx] if a.shape[-1] > 24 else a)  # noqa: E731
            mean = sel(z[f"{case}_{key}_trajectories"].astype(np.float64))
            var = sel(z[f"{case}_{key}_ensemble_variance"].astype(np.float64))
            res[name] = gauss_rows(mean, var, truth, mu, sd)
            res[name]["rmse"] = float(np.sqrt(((mean - truth) ** 2).mean(axis=1)).mean())
        for name, rel in ENSEMBLES.items():
            if args.only and name not in args.only:
                continue
            print(f"{case} {name}", flush=True)
            res[name] = score_ensemble(HERE / rel / f"members_{case}.npz", mu, sd, args.k,
                                       args.chunk, truth)
            args.out.write_text(json.dumps(results, indent=1))
        args.out.write_text(json.dumps(results, indent=1))
    args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
