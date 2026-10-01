import numpy as np
import pytest

from evaluation.fm_score import ensemble_score, gaussian_score, mixture_operator, snr


@pytest.mark.parametrize("tau", [0.25, 0.5, 0.75, 0.95])
def test_gaussian_score_minimised_at_calibrated_variance(tau: float) -> None:
    mse = 0.7
    grid = np.linspace(0.05, 3.0, 2000)
    scores = gaussian_score(np.sqrt(mse), grid, tau)
    assert abs(grid[scores.argmin()] - mse) < 5e-3


def test_gaussian_score_tau0_is_squared_error() -> None:
    err = np.array([0.3, -1.2])
    np.testing.assert_allclose(gaussian_score(err, np.array([0.5, 2.0]), 0.0), err ** 2)


def test_point_mass_score_is_constant() -> None:
    err = np.array([0.4])
    for tau in (0.0, 0.5, 0.95):
        np.testing.assert_allclose(gaussian_score(err, np.zeros(1), tau), err ** 2)


def test_gaussian_mixture_single_component_matches_closed_form() -> None:
    rng = np.random.default_rng(0)
    tau, m, v = 0.6, 0.3, 0.8
    x_tau = rng.normal(size=1000)
    psi, _ = mixture_operator(x_tau, np.full((1000, 1), m), tau, h2=v)
    k = np.sqrt(snr(tau)) * v / (1 + snr(tau) * v)
    np.testing.assert_allclose(psi, m + k * (x_tau / (1 - tau) - np.sqrt(snr(tau)) * m))


def test_monte_carlo_score_matches_gaussian_closed_form() -> None:
    rng = np.random.default_rng(1)
    tau, v = 0.5, 0.6
    truth = rng.normal(size=4000)
    mean = truth + rng.normal(scale=0.5, size=truth.shape)
    x0 = rng.normal(size=truth.shape + (64,))
    sc, _ = ensemble_score(mean[:, None], truth, tau, x0, h2=v)
    expected = gaussian_score(truth - mean, np.full(truth.shape, v), tau)
    assert abs(sc.mean() - expected.mean()) < 0.01 * expected.mean()


def test_true_posterior_scores_best() -> None:
    rng = np.random.default_rng(2)
    n, members, tau = 3000, 200, 0.75
    mu = rng.normal(size=n)
    truth = mu + rng.normal(size=n)
    x0 = rng.normal(size=(n, 4))
    calibrated = mu[:, None] + rng.normal(size=(n, members))
    narrow = mu[:, None] + 0.3 * rng.normal(size=(n, members))
    good, _ = ensemble_score(calibrated, truth, tau, x0)
    bad, _ = ensemble_score(narrow, truth, tau, x0)
    assert good.mean() < bad.mean()
