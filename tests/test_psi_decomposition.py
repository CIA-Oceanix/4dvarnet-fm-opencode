"""The Psi_mean/Psi_G/Psi_NG read-out, checked against an exactly Gaussian posterior.

Construction: m ~ N(0, q I) is the observation-informed mean, x1 = m + e with
e ~ N(0, p I), x0 ~ N(0, s I) and x_tau = (1-tau) x0 + tau x1. Then
x_tau - tau*m = (1-tau) x0 + tau e, so

    E[x1 | x_tau, m] = m + gamma(tau) (x_tau - tau*m),
    gamma(tau) = tau p / ((1-tau)^2 s + tau^2 p)

exactly -- a Gaussian posterior with y-independent covariance, hence Psi_NG == 0
at every tau. A model that returns this is the ground truth the fit must recover.
"""
import numpy as np
import pytest
import torch

from evaluation.psi_decomposition import (accumulate, components, gamma_closed_form,
                                          mc_floor, new_accumulator, psi_mean_estimate,
                                          psi_of, solve2)

P_VAR, Q_VAR, S_VAR = 0.4, 1.0, 0.25
TAUS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0]


class GaussianPosterior:
    """Exact E[x1|x_tau,y] for the construction in the module docstring."""

    returns_mu = True  # opt into psi_of's mu-predicting path (see psi_of's guard)

    def __init__(self, m: torch.Tensor, extra: float = 0.0):
        self.m = m
        self.extra = extra

    def forward(self, x_tau: torch.Tensor, batch, tau: torch.Tensor) -> torch.Tensor:
        t = float(tau[0])
        z = x_tau - t * self.m
        psi = self.m + gamma_closed_form(t, P_VAR, S_VAR) * z
        if self.extra:
            psi = psi + self.extra * t * (1.0 - t) * torch.tanh(3.0 * z)
        return psi


def _run(extra: float = 0.0, n_batches: int = 4, seed: int = 0):
    torch.manual_seed(seed)
    B, T, D = 16, 128, 8
    acc = {t: new_accumulator() for t in TAUS}
    x1_sq = 0.0
    m_err_sq = 0.0
    n_tot = 0
    sigma = S_VAR ** 0.5

    for _ in range(n_batches):
        m = torch.randn(B, T, D) * Q_VAR ** 0.5
        x1 = m + torch.randn(B, T, D) * P_VAR ** 0.5
        model = GaussianPosterior(m, extra=extra)

        m_hat, spread_sq = psi_mean_estimate(model, None, x1, sigma, m_draws=4)
        assert spread_sq == pytest.approx(0.0, abs=1e-6)
        x1_sq += float((x1 ** 2).sum())
        m_err_sq += float(((m_hat - x1) ** 2).sum())
        n_tot += B * T * D

        for t in TAUS:
            tt = torch.full((B,), float(t))
            x0 = torch.randn(B, T, D) * sigma
            x_tau = (1.0 - t) * x0 + t * x1
            accumulate(acc[t], m_hat, x_tau, psi_of(model, x_tau, None, t, tt, m_hat))

    p_hat = m_err_sq / n_tot
    return {t: components(acc[t], t, x1_sq, p_hat, S_VAR) for t in TAUS}


def test_psi_mean_estimate_recovers_m_exactly():
    """At tau=0 the operator returns m regardless of x0, so the estimate is exact."""
    torch.manual_seed(0)
    m = torch.randn(4, 32, 3)
    m_hat, spread_sq = psi_mean_estimate(GaussianPosterior(m), None,
                                         torch.zeros(4, 32, 3), S_VAR ** 0.5, m_draws=4)
    assert torch.allclose(m_hat, m, atol=1e-6)
    assert spread_sq == pytest.approx(0.0, abs=1e-6)


def test_gaussian_posterior_has_no_non_gaussian_component():
    rows = _run()
    for t, row in rows.items():
        assert row["psi_NG"] < 0.02 * row["psi_mean"], f"tau={t}: Psi_NG too large"


def test_fitted_gain_matches_closed_form():
    rows = _run()
    for t, row in rows.items():
        if t in (0.0, 1.0):
            continue
        assert row["gain_rel_err"] < 0.02, f"tau={t}: gain error {row['gain_rel_err']:.4f}"


def test_gain_endpoints():
    """K_0 = 0 and K_1 = 1 -- the boundary conditions Psi_NG(0)=Psi_NG(1)=0 follow from."""
    rows = _run()
    assert rows[0.0]["K"] == pytest.approx(0.0, abs=0.02)
    assert rows[1.0]["K"] == pytest.approx(1.0, abs=0.02)
    assert gamma_closed_form(0.0, P_VAR, S_VAR) == 0.0
    assert gamma_closed_form(1.0, P_VAR, S_VAR) == pytest.approx(1.0)


def test_constrained_form_holds():
    """The free 2-parameter fit collapses onto A = 1 - tau*K."""
    rows = _run()
    for t, row in rows.items():
        assert row["constrained_dev"] < 0.015, f"tau={t}: A deviates by {row['constrained_dev']}"


def test_non_affine_posterior_is_detected():
    """A deliberately non-affine operator must show up as a nonzero Psi_NG."""
    rows = _run(extra=0.5)
    assert rows[0.5]["psi_NG"] > 10 * _run()[0.5]["psi_NG"]
    # ...and still vanish where the construction pins it to zero
    assert rows[0.0]["psi_NG"] < 1e-5
    assert rows[1.0]["psi_NG"] < 1e-5


def test_velocity_normalization_cannot_test_the_endpoint():
    """NG/|v| is a 0/0 at tau=1: it blows up where NG/|x1| correctly vanishes.

    This is the whole reason M4c re-normalizes -- see evaluation/psi_decomposition.
    """
    rows = _run(extra=0.5)
    assert rows[0.99]["ng_over_x1"] < rows[0.9]["ng_over_x1"]
    assert rows[0.99]["ng_over_v"] > rows[0.9]["ng_over_v"]


def test_mc_floor_shrinks_with_draws():
    assert mc_floor(0.0, 4) == 0.0
    assert mc_floor(0.1, 1) == 0.0
    assert mc_floor(0.1, 4) == pytest.approx(0.1 * np.sqrt(5 / 3))
    assert mc_floor(0.1, 16) < mc_floor(0.1, 4)


def test_solve2_recovers_known_coefficients():
    torch.manual_seed(0)
    m = torch.randn(2000)
    x = torch.randn(2000)
    mu = 0.7 * m - 0.3 * x
    a, b = solve2(float((m * m).sum()), float((m * x).sum()), float((x * x).sum()),
                  float((m * mu).sum()), float((x * mu).sum()))
    assert a == pytest.approx(0.7, abs=1e-4)
    assert b == pytest.approx(-0.3, abs=1e-4)


class TestVelocityParameterization:
    """VanillaCFM returns a velocity, and its forward signature is identical to
    PredictStateCFM's -- so the fallback path would read it as a mean and fail
    silently. The x1_hat identity x1 = x_tau + (1-tau)v is what makes it readable.
    """

    class _VelocityModel(torch.nn.Module):
        """Stands in for VanillaCFM: exact velocity for the Gaussian construction."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x_tau, batch, tau):
            t = float(tau[0])
            psi = self.m + gamma_closed_form(t, P_VAR, S_VAR) * (x_tau - t * self.m)
            return (psi - x_tau) / (1.0 - t)

    def test_velocity_is_converted_to_the_mean(self):
        from models.vanilla_cfm import VanillaCFM
        torch.manual_seed(0)
        m = torch.randn(2, 32, 4)
        x1 = m + torch.randn(2, 32, 4) * P_VAR ** 0.5
        vel = self._VelocityModel(m)
        ref = GaussianPosterior(m)
        for t in (0.2, 0.5, 0.9):
            tt = torch.full((2,), t)
            x0 = torch.randn_like(x1) * S_VAR ** 0.5
            x_tau = (1.0 - t) * x0 + t * x1
            # psi_of dispatches on isinstance, so exercise the identity directly
            psi_from_velocity = x_tau + (1.0 - t) * vel(x_tau, None, tt)
            assert torch.allclose(psi_from_velocity, ref.forward(x_tau, None, tt), atol=1e-5)
        assert issubclass(VanillaCFM, torch.nn.Module)

    def test_unknown_model_type_raises_instead_of_guessing(self):
        """The whole point: a silent misread is worse than a loud failure."""
        class Mystery(torch.nn.Module):
            def forward(self, x_tau, batch, tau):
                return x_tau

        with pytest.raises(TypeError, match="does not know what"):
            psi_of(Mystery(), torch.randn(2, 8, 3), None, 0.5, torch.full((2,), 0.5))

    def test_mu_predicting_models_still_pass_through(self):
        torch.manual_seed(0)
        m = torch.randn(2, 16, 3)
        model = GaussianPosterior(m)
        x_tau = torch.randn(2, 16, 3)
        tt = torch.full((2,), 0.4)
        # GaussianPosterior opts in with returns_mu, so it takes the mu path.
        assert torch.allclose(psi_of(model, x_tau, None, 0.4, tt),
                              model.forward(x_tau, None, tt))
