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


SCALE_TAUS = (0.25, 0.5, 0.75, 0.95)
SCALES = np.exp(np.linspace(np.log(0.125), np.log(8.0), 41))
KDE_C = 1 + (1.06 * 30 ** -0.2) ** 2


def accumulate_scales(err: np.ndarray, var: np.ndarray, acc: np.ndarray) -> None:
    for i, tau in enumerate(SCALE_TAUS):
        for j, c in enumerate(list(SCALES) + [KDE_C]):
            acc[i, j] += gaussian_score(err, c * var, tau).sum(axis=(0, 1))


def finish_scales(acc: np.ndarray, n: int, mse: np.ndarray, spread2: np.ndarray) -> dict:
    out = {}
    for g, s in GROUPS.items():
        a = acc[..., s].sum(-1) / (n * (s.stop - s.start))
        row = {"pooled_spread_skill": float(np.sqrt(spread2[s].sum() / mse[s].sum()))}
        for i, tau in enumerate(SCALE_TAUS):
            k = int(a[i, :-1].argmin())
            row[str(tau)] = {"c1": float(a[i, 20]), "c_star": float(SCALES[k]),
                             "at_c_star": float(a[i, k]), "kde_var": float(a[i, -1])}
        out[g] = row
    return out


GROUP_LIST = tuple(GROUPS)
VARIANTS = ("gauss", "ens", "kde", "snap")


def window_scale_sums(err: np.ndarray, var: np.ndarray, w2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-window group means of the Gaussian FMS at c * var (time-mean), z and sigma^2-weighted.

    Returns two (w, n_groups, n_scale_taus, n_scales + 1) arrays; the last scale is the KDE inflation.
    """
    cs = list(SCALES) + [KDE_C]
    z = np.zeros((err.shape[0], len(GROUP_LIST), len(SCALE_TAUS), len(cs)))
    ph = np.zeros_like(z)
    for i, tau in enumerate(SCALE_TAUS):
        for j, c in enumerate(cs):
            g = gaussian_score(err, c * var, tau).mean(axis=1)
            for k, name in enumerate(GROUP_LIST):
                sl = GROUPS[name]
                z[:, k, i, j] = g[:, sl].mean(axis=1)
                ph[:, k, i, j] = (g[:, sl] * w2[sl]).mean(axis=1)
    return z, ph


def window_members(members: np.ndarray, truth: np.ndarray, mu: np.ndarray, sd: np.ndarray,
                   k: int = 4, chunk: int = 10, label: str = "") -> dict[str, np.ndarray]:
    """Per-window arrays of every FMS variant (time-mean, z-units) and the sweep sums."""
    W = truth.shape[0]
    w2 = sd ** 2
    out = {f"fms_{v}": np.zeros((W, 24, len(TAUS))) for v in VARIANTS}
    out.update(se=np.zeros((W, 24)), var=np.zeros((W, 24)), snap_ess=np.zeros((W, len(TAUS))),
               scale_z=np.zeros((W, len(GROUP_LIST), len(SCALE_TAUS), len(SCALES) + 1)))
    out["scale_phys"] = np.zeros_like(out["scale_z"])
    for c0 in range(0, W, chunk):
        sl = slice(c0, min(c0 + chunk, W))
        m = (members[sl].astype(np.float64) - mu[:, None]) / sd[:, None]
        t = (truth[sl].astype(np.float64) - mu) / sd
        x0 = noise(c0, t.shape, k)
        var = m.var(axis=-1, ddof=1)
        err = t - m.mean(axis=-1)
        h2 = silverman_h2(m)[..., None]
        out["se"][sl] = (err ** 2).mean(axis=1)
        out["var"][sl] = var.mean(axis=1)
        out["scale_z"][sl], out["scale_phys"][sl] = window_scale_sums(err, var, w2)
        for i, tau in enumerate(TAUS):
            out["fms_gauss"][sl, :, i] = gaussian_score(err, var, tau).mean(axis=1)
            if tau == 0.0:
                for v in ("ens", "kde", "snap"):
                    out[f"fms_{v}"][sl, :, i] = out["se"][sl]
                out["snap_ess"][sl, i] = m.shape[-1]
                continue
            s, _ = ensemble_score(m, t, tau, x0)
            out["fms_ens"][sl, :, i] = s.mean(axis=1)
            s, _ = ensemble_score(m, t, tau, x0, h2=h2)
            out["fms_kde"][sl, :, i] = s.mean(axis=1)
            s, e = ensemble_score(m, t, tau, x0, block_axes=(2,))
            out["fms_snap"][sl, :, i] = s.mean(axis=1)
            out["snap_ess"][sl, i] = e[..., 0].mean(axis=1)
        print(f"  {label} windows {sl.stop}/{W}", flush=True)
    out["rmse_phys"] = np.sqrt(out["se"]) * sd
    return out


def window_gaussian(err: np.ndarray, var: np.ndarray, sd: np.ndarray) -> dict[str, np.ndarray]:
    """Per-window arrays for a Gaussian forecast given z-unit err/var (W, T, 24); var = 0 for a point."""
    out = {"fms_gauss": np.stack([gaussian_score(err, var, tau).mean(axis=1) for tau in TAUS], axis=-1),
           "se": (err ** 2).mean(axis=1), "var": var.mean(axis=1)}
    out["scale_z"], out["scale_phys"] = window_scale_sums(err, var, sd ** 2)
    out["rmse_phys"] = np.sqrt(out["se"]) * sd
    return out


def pooled_from_windows(arr: dict[str, np.ndarray]) -> dict:
    """The pooled JSON of the earlier probes, derived from the per-window arrays."""
    out = {}
    for v in VARIANTS:
        if f"fms_{v}" in arr:
            out[v] = {str(tau): pooled(arr[f"fms_{v}"][:, :, i].mean(axis=0)) for i, tau in enumerate(TAUS)}
    if "snap_ess" in arr:
        out["snap_ess"] = {str(tau): float(arr["snap_ess"][:, i].mean()) for i, tau in enumerate(TAUS)}
    scale = {}
    for k, g in enumerate(GROUP_LIST):
        sl = GROUPS[g]
        a = arr["scale_z"][:, k].mean(axis=0)
        row = {"pooled_spread_skill": float(np.sqrt(arr["var"][:, sl].sum() / arr["se"][:, sl].sum()))}
        for i, tau in enumerate(SCALE_TAUS):
            j = int(a[i, :-1].argmin())
            row[str(tau)] = {"c1": float(a[i, 20]), "c_star": float(SCALES[j]),
                             "at_c_star": float(a[i, j]), "kde_var": float(a[i, -1])}
        scale[g] = row
    out["scale"] = scale
    out["rmse"] = float(arr["rmse_phys"].mean())
    return out


def score_members(members: np.ndarray, truth: np.ndarray, mu: np.ndarray, sd: np.ndarray,
                  k: int = 4, chunk: int = 10, label: str = "") -> dict:
    """All FMS variants plus the variance-rescaling sweep, members (W, T, 24, M) in memory (pooled)."""
    return pooled_from_windows(window_members(members, truth, mu, sd, k, chunk, label))


def score_ensemble(path: Path, mu: np.ndarray, sd: np.ndarray, k: int, chunk: int,
                   ref_truth: np.ndarray) -> dict:
    z = np.load(path)
    members, truth = z["members"], z["truth"]
    assert np.allclose(truth, ref_truth, atol=1e-4), path
    return score_members(members, truth, mu, sd, k, chunk, f"{path.parent.name}/{path.name}")


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
