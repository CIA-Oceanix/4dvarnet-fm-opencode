import numpy as np

from evaluation.bootstrap import bootstrap_ci, mean_stat, paired_ci, ratio_stat, sqrt_mean_stat
from evaluation.fm_score import gaussian_score
from reports.l96.probe_fm_score import TAUS, pooled_from_windows, window_gaussian


def test_mean_interval_matches_analytic() -> None:
    x = np.random.default_rng(0).normal(1.0, 2.0, size=400)
    v, lo, hi = bootstrap_ci(mean_stat(x), len(x), n_boot=4000)
    half = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    assert abs(v - x.mean()) < 1e-12
    assert abs((hi - lo) / 2 - half) < 0.1 * half


def test_paired_interval_detects_shift_hidden_by_window_variance() -> None:
    rng = np.random.default_rng(1)
    difficulty = rng.lognormal(0.0, 1.0, size=200)
    a = difficulty * 1.05 + 0.01 * rng.normal(size=200)
    b = difficulty
    _, lo_a, hi_a = bootstrap_ci(mean_stat(a), 200)
    _, lo_b, hi_b = bootstrap_ci(mean_stat(b), 200)
    assert lo_a < hi_b and lo_b < hi_a
    d, lo, hi = paired_ci(mean_stat(a), mean_stat(b), 200)
    assert d > 0 and lo > 0


def test_ratio_and_sqrt_statistics() -> None:
    num = np.full((10, 3), 4.0)
    den = np.full((10, 3), 1.0)
    assert ratio_stat(num, den, sqrt=True)(np.arange(10)) == 2.0
    assert sqrt_mean_stat(np.full((10, 3), 9.0))(np.arange(10)) == 3.0


def test_window_gaussian_is_linear_in_windows() -> None:
    rng = np.random.default_rng(2)
    err = rng.normal(size=(6, 50, 24))
    var = rng.uniform(0.1, 2.0, size=(6, 50, 24))
    arrays = window_gaussian(err, var, np.ones(24))
    pooled = pooled_from_windows(arrays)
    for i, tau in enumerate(TAUS):
        assert abs(pooled["gauss"][str(tau)]["all"] - gaussian_score(err, var, tau).mean()) < 1e-12
