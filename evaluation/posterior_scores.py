"""Per-window ensemble scores and paired window-bootstrap comparisons.

Scores one window at a time from an ensemble ``members`` of shape ``(M, T, D)``
and the truth ``(T, D)``, both in physical units. Per-window fields:

- ``rmse``: ensemble-mean RMSE, mean over channels of the time RMSE (the
  convention of ``evaluation.metrics.rmse`` and of the FDV2->SDA probe);
- ``mse`` / ``var``: mean over (t, d) of the squared ensemble-mean error / of the
  unbiased member variance (the pooled spread-skill ingredients);
- ``ssr``: sqrt((M+1)/M) * sqrt(var / mse), the spread-skill ratio with RMS
  spread (1 for a calibrated exchangeable ensemble);
- ``spread_std``: mean over (t, d) of the member standard deviation (the
  probe's "spread", kept for comparison with ``sda_warm.json``);
- ``crps``: ``evaluation.metrics.crps`` averaged over channels, i.e.
  E|X-y| - 0.5 E|X-X'| with the 1/M^2 pair normalisation (self-pairs
  included), identical to the probe's estimator;
- ``crps_fair``: the same with the unbiased 1/(M(M-1)) pair normalisation;
- ``energy_score``: multivariate energy score of the whole window as one
  T*D vector, (1/M) sum_i ||x_i - y|| - 1/(2M^2) sum_ij ||x_i - x_j||, divided
  by sqrt(T*D) so that it is on the scale of an RMS error;
- ``rank_counts``: histogram (M+1 bins) of the rank of the truth among the
  members over all (t, d), rank = number of members strictly below the truth;
- ``reliability``: sum_k |f_k - 1/(M+1)| with f_k the rank-histogram
  frequencies (0 for a flat histogram, 2M/(M+1) when every rank is extreme);
- ``cov50`` / ``cov90``: fraction of (t, d) whose truth lies in the 50% / 90%
  central interval of the members, with quantiles interpolated at the Weibull
  plotting positions p*(M+1) of the order statistics, so that an exchangeable
  (calibrated) ensemble has expected coverage exactly 0.5 / 0.9 at any M >= 19
  (numpy's default linear positions give 0.47 / 0.87 at M = 30).

``evaluation.metrics.energy_score`` is the time-averaged per-channel CRPS, which
equals ``crps`` here, so the multivariate form above is reported instead.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from evaluation.bootstrap import Stat, mean_stat, paired_ci

MEAN_METRICS = ("rmse", "crps", "crps_fair", "energy_score", "spread_std", "cov50", "cov90", "ssr_window")


def reliability_index(rank_counts: np.ndarray) -> float:
    counts = np.asarray(rank_counts, dtype=float)
    f = counts / counts.sum()
    return float(np.abs(f - 1.0 / counts.size).sum())


def weibull_quantiles(members: torch.Tensor, levels: Sequence[float]) -> list[torch.Tensor]:
    """Member-axis quantiles at plotting positions p*(M+1) (1-indexed order
    statistics), linearly interpolated and clamped to the extreme members."""
    M = members.shape[0]
    xs = torch.sort(members, 0).values
    out = []
    for p in levels:
        pos = min(max(p * (M + 1) - 1.0, 0.0), M - 1.0)
        i = int(pos)
        j = min(i + 1, M - 1)
        out.append(xs[i] + (pos - i) * (xs[j] - xs[i]))
    return out


def inflate(members: torch.Tensor, factor: float) -> torch.Tensor:
    """Scale member deviations about the ensemble mean (member axis 0)."""
    mean = members.mean(0, keepdim=True)
    return mean + factor * (members - mean)


@torch.no_grad()
def window_scores(members: torch.Tensor, truth: torch.Tensor) -> dict:
    x = members.double()
    y = truth.double().to(x.device)
    M, T, D = x.shape
    err = x.mean(0) - y
    mse = float((err ** 2).mean())
    rmse = float((err ** 2).mean(0).sqrt().mean())
    abs_err = (x - y).abs().mean(0)
    if M > 1:
        var = float(x.var(0, unbiased=True).mean())
        spread_std = float(x.std(0, unbiased=True).mean())
        coeff = (2 * torch.arange(1, M + 1, device=x.device, dtype=x.dtype) - M - 1).view(M, 1, 1)
        pair_sum = 2.0 * (coeff * torch.sort(x, 0).values).sum(0)
        crps = float((abs_err - 0.5 * pair_sum / (M * M)).mean())
        crps_fair = float((abs_err - 0.5 * pair_sum / (M * (M - 1))).mean())
        flat = x.reshape(M, -1)
        es_pair = float(torch.cdist(flat, flat).sum()) / (2 * M * M)
        lo90, lo50, hi50, hi90 = weibull_quantiles(x, (0.05, 0.25, 0.75, 0.95))
        cov50 = float(((y >= lo50) & (y <= hi50)).double().mean())
        cov90 = float(((y >= lo90) & (y <= hi90)).double().mean())
    else:
        var = spread_std = es_pair = 0.0
        crps = crps_fair = float(abs_err.mean())
        cov50 = cov90 = 0.0
    es = (float((x - y).reshape(M, -1).norm(dim=1).mean()) - es_pair) / (T * D) ** 0.5
    ranks = (x < y).sum(0).flatten()
    counts = torch.bincount(ranks, minlength=M + 1).cpu().numpy()
    ssr = float(np.sqrt((M + 1) / M) * np.sqrt(var / mse)) if mse > 0 else float("nan")
    return {"rmse": rmse, "mse": mse, "var": var, "ssr_window": ssr, "spread_std": spread_std,
            "crps": crps, "crps_fair": crps_fair, "energy_score": es,
            "rank_counts": counts.astype(int).tolist(), "reliability": reliability_index(counts),
            "cov50": cov50, "cov90": cov90, "n_members": M}


def stack_per_window(per_window: Sequence[dict]) -> dict:
    keys = [k for k in per_window[0] if k != "n_members"]
    return {k: np.asarray([w[k] for w in per_window], dtype=float) for k in keys}


def pooled_ssr(var: np.ndarray, mse: np.ndarray, n_members: int) -> float:
    return float(np.sqrt((n_members + 1) / n_members) * np.sqrt(var.sum() / mse.sum()))


def metric_stat(per_window: Sequence[dict], metric: str) -> Stat:
    """Window-index statistic for ``evaluation.bootstrap``: means of per-window
    values, the pooled SSR (ratio of summed variance and MSE) or the reliability
    index of the summed rank histogram."""
    s = stack_per_window(per_window)
    M = int(per_window[0]["n_members"])
    if metric in MEAN_METRICS:
        return mean_stat(s[metric])
    if metric == "ssr":
        return lambda i: pooled_ssr(s["var"][i], s["mse"][i], M)
    if metric == "reliability":
        return lambda i: reliability_index(s["rank_counts"][i].sum(0))
    raise ValueError(f"unknown metric {metric!r}")


def aggregate(per_window: Sequence[dict]) -> dict:
    s = stack_per_window(per_window)
    M = int(per_window[0]["n_members"])
    out = {k: float(s[k].mean()) for k in MEAN_METRICS}
    out["ssr"] = pooled_ssr(s["var"], s["mse"], M) if M > 1 else 0.0
    counts = s["rank_counts"].sum(0)
    out["rank_counts"] = counts.astype(int).tolist()
    out["reliability"] = reliability_index(counts)
    out["spread_std_over_rmse"] = out["spread_std"] / out["rmse"]
    out["n_windows"] = len(per_window)
    out["n_members"] = M
    return out


def paired_bootstrap(per_window_a: Sequence[dict], per_window_b: Sequence[dict], metric: str,
                     n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """Paired window bootstrap of ``metric(a) - metric(b)`` (same resampled
    windows for both rows; percentile interval)."""
    if len(per_window_a) != len(per_window_b):
        raise ValueError("paired bootstrap needs the same windows for both rows")
    stat_a, stat_b = metric_stat(per_window_a, metric), metric_stat(per_window_b, metric)
    diff, lo, hi = paired_ci(stat_a, stat_b, len(per_window_a), n_boot, seed, level)
    ref = stat_b(np.arange(len(per_window_b)))
    return {"metric": metric, "diff": diff, "lo": lo, "hi": hi,
            "rel_diff": diff / ref if ref else float("nan"), "n_boot": n_boot, "seed": seed}
