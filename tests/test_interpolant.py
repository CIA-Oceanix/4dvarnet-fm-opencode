"""Tests for LinearInterpolant."""
import torch
from models.interpolant import LinearInterpolant


def test_alpha_beta_values():
    interp = LinearInterpolant()
    tau = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(interp.alpha(tau), torch.tensor([1.0, 0.5, 0.0]))
    assert torch.allclose(interp.beta(tau), torch.tensor([0.0, 0.5, 1.0]))


def test_alpha_dot_beta_dot():
    interp = LinearInterpolant()
    tau = torch.tensor([0.0, 0.3, 0.7, 1.0])
    assert torch.allclose(interp.alpha_dot(tau), torch.full_like(tau, -1.0))
    assert torch.allclose(interp.beta_dot(tau), torch.full_like(tau, 1.0))


def test_mix_shape():
    interp = LinearInterpolant()
    x0 = torch.randn(4, 50, 3)
    x1 = torch.randn(4, 50, 3)
    tau = torch.tensor(0.5)
    out = interp.mix(x0, x1, tau)
    assert out.shape == (4, 50, 3)


def test_mix_values():
    interp = LinearInterpolant()
    x0 = torch.randn(4, 50, 3)
    x1 = torch.randn(4, 50, 3)
    tau0 = torch.tensor(0.0)
    tau1 = torch.tensor(1.0)
    assert torch.allclose(interp.mix(x0, x1, tau0), x0)
    assert torch.allclose(interp.mix(x0, x1, tau1), x1)


def test_gain_matrix():
    interp = LinearInterpolant(nu=1.0)
    tau = torch.tensor([0.0, 0.5, 1.0])
    K = interp.gain_matrix(tau)
    assert K[0].item() == 0.0
    assert K[2].item() == 1.0
    assert 0.0 < K[1].item() < 1.0


def test_ng_prefactor():
    interp = LinearInterpolant()
    tau = torch.linspace(0.0, 1.0, 10)
    result = interp.ng_prefactor(tau)
    expected = tau * (1.0 - tau)
    assert torch.allclose(result, expected)


def test_sample_tau():
    interp = LinearInterpolant()
    samples = interp.sample_tau((1000,))
    assert samples.shape == (1000,)
    assert samples.min() >= 0.0
    assert samples.max() < 1.0


def test_score_matches_closed_form_gaussian_mixture():
    """Validates LinearInterpolant.score's Tweedie-formula derivation against
    an exactly-solvable case: x0 ~ N(0, s0^2 I), x1 ~ N(m1, s1^2 I),
    independent, isotropic. Both v_theta = E[x1-x0|x_tau] and the true
    marginal score of x_tau are available in closed form via the standard
    jointly-Gaussian conditional-mean formula, with no trained network
    involved -- an end-to-end check of the algebra, not just the code path.
    """
    interp = LinearInterpolant()
    torch.manual_seed(0)
    D = 5
    m1 = torch.randn(D) * 2.0
    s0, s1 = 0.7, 1.3
    for tau_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
        tau = torch.tensor(tau_val)
        a = interp.alpha(tau)   # noise coeff
        b = interp.beta(tau)    # data coeff
        var_tau = a ** 2 * s0 ** 2 + b ** 2 * s1 ** 2
        mean_tau = b * m1

        x_tau = mean_tau + torch.randn(D) * var_tau.sqrt() * 0.3  # any point, need not be on-distribution

        # Jointly-Gaussian conditional means (E[x0|x_tau], E[x1|x_tau]).
        x0_hat_true = (a * s0 ** 2 / var_tau) * (x_tau - mean_tau)
        x1_hat_true = m1 + (b * s1 ** 2 / var_tau) * (x_tau - mean_tau)
        v = x1_hat_true - x0_hat_true

        true_score = -(x_tau - mean_tau) / var_tau
        got_score = interp.score(x_tau, v, tau, sigma=s0)
        assert torch.allclose(got_score, true_score, atol=1e-5), \
            f"score mismatch at tau={tau_val}"


def test_score_at_tau0_ignores_v():
    """At tau=0, score must reduce to -x_tau/sigma^2 regardless of v --
    algebraic equivalent of ICTM's separate t=0 closed-form prior term,
    since x_0 ~ N(0, sigma^2 I) has a known density independent of the
    (not-yet-meaningfully-evaluated) velocity field."""
    interp = LinearInterpolant()
    torch.manual_seed(1)
    x0 = torch.randn(4, 10, 3)
    tau0 = torch.zeros(4)
    sigma = 0.7
    expected = -x0 / sigma ** 2
    for _ in range(3):
        v_arbitrary = torch.randn(4, 10, 3) * 5.0
        got = interp.score(x0, v_arbitrary, tau0, sigma=sigma)
        assert torch.allclose(got, expected, atol=1e-5)


def test_compute_drift():
    interp = LinearInterpolant()
    x = torch.randn(4, 50, 3)
    x_cond = torch.randn(4, 50, 3)
    tau = torch.tensor(0.5)
    drift = interp.compute_drift(x, x_cond, tau)
    assert drift.shape == (4, 50, 3)
