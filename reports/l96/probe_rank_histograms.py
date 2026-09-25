"""Rank histograms of L96 ensembles (C4 of the P1 paper: does a missing Psi_NG show up as a
wrong posterior *shape*, beyond the spread that spread/RMSE measures?).

Inputs are ``members`` (W, T, D, M) + ``truth`` (W, T, D) npz files -- the layout of the
neural ``members_<case>.npz`` and of ``evaluate_all_l96.py --save-members``. For every
(window, time, coordinate) the truth's rank among the M members (ties broken at random)
falls in one of M + 1 bins; a calibrated ensemble gives a flat histogram.

Reported per scheme and group (all_obs / slow / obs_fast):
  hist              normalized histogram (M + 1 bins)
  ri                reliability index sum_k |h_k - 1/(M+1)| (0 = flat)
  extreme_ratio     (h_0 + h_M) / (2/(M+1)): > 1 under-dispersed (U shape), < 1 over-dispersed
  rank_bias         mean normalized rank - 0.5 (> 0: truth above the ensemble)
  spread_over_rmse  per-window-pooled spread / RMSE of the ensemble mean
  ri_shape, extreme_ratio_shape, rank_bias_shape
                    the same after rescaling every member around the ensemble mean by one
                    scalar per group, chosen so the second moment is calibrated
                    (spread^2 * (M+1)/M = MSE). What remains is *shape* (plus any mean bias).
  ri_debiased, extreme_ratio_debiased, rank_bias_debiased
                    the same after ALSO removing each channel's mean error (ensemble mean
                    minus truth, averaged over windows and time) and re-calibrating the spread
                    to the de-biased MSE. A first-moment (model-error) bias shows up as a
                    ramp that survives the spread correction but not this one; what survives
                    this one is higher-moment shape.

Streams one window at a time; writes only a JSON (and optionally a PNG).
"""
import argparse
import json
from pathlib import Path

import numpy as np

NO = 8
GROUPS = {"all_obs": slice(None), "slow": slice(0, NO), "obs_fast": slice(NO, None)}


def ranks(members: np.ndarray, truth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    below = (members < truth[..., None]).sum(-1)
    ties = (members == truth[..., None]).sum(-1)
    return below + (rng.random(below.shape) * (ties + 1)).astype(np.int64)


def group_scale(path: Path, sl: slice, bias=None) -> float:
    z = np.load(path, mmap_mode="r")
    mem, tru = z["members"], z["truth"]
    M = mem.shape[-1]
    se = var = n = 0.0
    for w in range(mem.shape[0]):
        m = np.asarray(mem[w][:, sl], dtype=np.float64)
        t = np.asarray(tru[w][:, sl], dtype=np.float64)
        mu = m.mean(-1) - (0.0 if bias is None else bias)
        se += ((mu - t) ** 2).sum()
        var += m.var(-1, ddof=1).sum()
        n += t.size
    return float(np.sqrt((se / n) / ((var / n) * (M + 1) / M)))


def channel_bias(path: Path, sl: slice) -> np.ndarray:
    z = np.load(path, mmap_mode="r")
    mem, tru = z["members"], z["truth"]
    acc, n = 0.0, 0
    for w in range(mem.shape[0]):
        m = np.asarray(mem[w][:, sl], dtype=np.float64)
        t = np.asarray(tru[w][:, sl], dtype=np.float64)
        acc = acc + (m.mean(-1) - t).sum(0)
        n += t.shape[0]
    return acc / n


def summarize(counts: np.ndarray, rank_sum: float, n: int, M: int) -> dict:
    h = counts / counts.sum()
    flat = 1.0 / (M + 1)
    return {"hist": h.round(6).tolist(), "ri": float(np.abs(h - flat).sum()),
            "extreme_ratio": float((h[0] + h[-1]) / (2 * flat)),
            "rank_bias": float(rank_sum / (n * M) - 0.5)}


def analyse(label: str, path: Path, seed: int) -> dict:
    z = np.load(path, mmap_mode="r")
    mem, tru = z["members"], z["truth"]
    W, T, D, M = mem.shape
    out = {"label": label, "path": str(path), "windows": W, "members": M}
    for g, sl in GROUPS.items():
        scale = group_scale(path, sl)
        bias = channel_bias(path, sl)
        scale_db = group_scale(path, sl, bias)
        rng = np.random.default_rng(seed)
        c_raw, c_shape, c_db = np.zeros(M + 1), np.zeros(M + 1), np.zeros(M + 1)
        s_raw = s_shape = s_db = 0.0
        se = var = n = 0.0
        for w in range(W):
            m = np.asarray(mem[w][:, sl], dtype=np.float64)
            t = np.asarray(tru[w][:, sl], dtype=np.float64)
            mu = m.mean(-1, keepdims=True)
            r = ranks(m, t, rng)
            c_raw += np.bincount(r.ravel(), minlength=M + 1)
            s_raw += r.sum()
            r2 = ranks(mu + scale * (m - mu), t, rng)
            c_shape += np.bincount(r2.ravel(), minlength=M + 1)
            s_shape += r2.sum()
            r3 = ranks(mu - bias[..., None] + scale_db * (m - mu), t, rng)
            c_db += np.bincount(r3.ravel(), minlength=M + 1)
            s_db += r3.sum()
            se += ((mu[..., 0] - t) ** 2).sum()
            var += m.var(-1, ddof=1).sum()
            n += t.size
        raw, shp = summarize(c_raw, s_raw, int(n), M), summarize(c_shape, s_shape, int(n), M)
        db = summarize(c_db, s_db, int(n), M)
        out[g] = {**raw, "spread_over_rmse": float(np.sqrt(var / n) / np.sqrt(se / n)),
                  "shape_scale": scale, "ri_shape": shp["ri"],
                  "extreme_ratio_shape": shp["extreme_ratio"], "rank_bias_shape": shp["rank_bias"],
                  "hist_shape": shp["hist"],
                  "mean_bias_rms": float(np.sqrt((bias ** 2).mean())), "debias_scale": scale_db,
                  "ri_debiased": db["ri"], "extreme_ratio_debiased": db["extreme_ratio"],
                  "rank_bias_debiased": db["rank_bias"], "hist_debiased": db["hist"]}
    return out


def plot(results: list, png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.4 * rows), squeeze=False)
    for ax, r in zip(axes.ravel(), results):
        g = r["all_obs"]
        k = np.arange(len(g["hist"]))
        ax.bar(k, g["hist"], width=1.0, color="#4c72b0", alpha=0.8, label="raw")
        ax.step(k, g["hist_shape"], where="mid", color="#c44e52", label="spread-corrected")
        ax.step(k, g["hist_debiased"], where="mid", color="#55a868", label="+ de-biased")
        ax.axhline(1 / len(k), color="k", lw=0.8, ls="--")
        ax.set_title(f"{r['label']}\nRI {g['ri']:.2f} -> {g['ri_shape']:.2f} -> {g['ri_debiased']:.2f}", fontsize=8)
        ax.set_xticks([])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(png, dpi=110)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", action="append", required=True, help="LABEL=path/to/members.npz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--png", default=None)
    a = p.parse_args()
    results = []
    for item in a.input:
        label, path = item.split("=", 1)
        results.append(analyse(label, Path(path), a.seed))
        g = results[-1]["all_obs"]
        print(f"{label:34s} RI {g['ri']:.3f} extreme {g['extreme_ratio']:.2f} bias {g['rank_bias']:+.3f} "
              f"sp/rmse {g['spread_over_rmse']:.2f} | shape RI {g['ri_shape']:.3f} "
              f"extreme {g['extreme_ratio_shape']:.2f} bias {g['rank_bias_shape']:+.3f} | debiased RI "
              f"{g['ri_debiased']:.3f} extreme {g['extreme_ratio_debiased']:.2f}", flush=True)
    Path(a.out).write_text(json.dumps(results, indent=1))
    if a.png:
        plot(results, Path(a.png))


if __name__ == "__main__":
    main()
