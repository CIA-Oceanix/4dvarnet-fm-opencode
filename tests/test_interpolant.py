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


def test_compute_drift():
    interp = LinearInterpolant()
    x = torch.randn(4, 50, 3)
    x_cond = torch.randn(4, 50, 3)
    tau = torch.tensor(0.5)
    drift = interp.compute_drift(x, x_cond, tau)
    assert drift.shape == (4, 50, 3)
