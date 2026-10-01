"""Window bootstrap for scores averaged over test windows.

The resampling unit is the window (time steps and channels within a window are
correlated). A statistic is a function of an index array of resampled windows,
so any averaging convention (mean of per-window RMSE, sqrt of mean MSE, ratio
of sums, re-fitted argmin) is bootstrapped as reported.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

Stat = Callable[[np.ndarray], float]


def resample_indices(n_windows: int, n_boot: int = 2000, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_windows, size=(n_boot, n_windows))


def bootstrap_ci(stat: Stat, n_windows: int, n_boot: int = 2000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float, float]:
    """(value on all windows, lower, upper) percentile interval."""
    reps = np.array([stat(i) for i in resample_indices(n_windows, n_boot, seed)])
    q = (1 - level) / 2
    return float(stat(np.arange(n_windows))), float(np.quantile(reps, q)), float(np.quantile(reps, 1 - q))


def paired_ci(stat_a: Stat, stat_b: Stat, n_windows: int, n_boot: int = 2000, seed: int = 0,
              level: float = 0.95) -> tuple[float, float, float]:
    """Interval of stat_a - stat_b with the same resampled windows for both."""
    def diff(i: np.ndarray) -> float:
        return stat_a(i) - stat_b(i)
    return bootstrap_ci(diff, n_windows, n_boot, seed, level)


def mean_stat(per_window: np.ndarray) -> Stat:
    """Mean over windows of a per-window value (extra axes averaged first)."""
    v = per_window.reshape(per_window.shape[0], -1).mean(axis=1)
    return lambda i: float(v[i].mean())


def sqrt_mean_stat(per_window: np.ndarray) -> Stat:
    v = per_window.reshape(per_window.shape[0], -1).mean(axis=1)
    return lambda i: float(np.sqrt(v[i].mean()))


def ratio_stat(num: np.ndarray, den: np.ndarray, sqrt: bool = False) -> Stat:
    """Ratio of resampled totals (sqrt for a spread/skill ratio)."""
    a = num.reshape(num.shape[0], -1).sum(axis=1)
    b = den.reshape(den.shape[0], -1).sum(axis=1)
    if sqrt:
        return lambda i: float(np.sqrt(a[i].sum() / b[i].sum()))
    return lambda i: float(a[i].sum() / b[i].sum())
