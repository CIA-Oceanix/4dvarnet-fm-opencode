"""Tests for TweedieSolver."""
import torch
from models.solver import TweedieSolver


def test_solver_creation():
    model = TweedieSolver(state_dim=3, hidden_channels=[4, 8], K_inner=2, N_outer=2)
    assert model.state_dim == 3
    assert model.K_inner == 2
    assert model.N_outer == 2


def test_estimate_mean_shape():
    model = TweedieSolver(state_dim=3, hidden_channels=[4, 8], K_inner=2, N_outer=2)
    obs = torch.randn(2, 50, 3)
    out = model.estimate_mean(obs)
    assert out.shape == (2, 50, 3)


def test_estimate_mean_deterministic():
    model = TweedieSolver(state_dim=3, hidden_channels=[4, 8], K_inner=2, N_outer=2)
    model.eval()
    obs = torch.randn(2, 50, 3)
    with torch.no_grad():
        out1 = model.estimate_mean(obs)
        out2 = model.estimate_mean(obs)
    torch.testing.assert_close(out1, out2)


def test_energy_terms_shape():
    model = TweedieSolver(state_dim=3, hidden_channels=[4, 8], K_inner=2, N_outer=2)
    B, D, T = 2, 3, 50
    x = torch.randn(B, D, T)
    obs = torch.randn(2, 50, 3)
    x_tau = torch.randn(B, D, T)
    tau = torch.full((B,), 0.5)
    terms = model.energy_terms(x, obs, x_tau, tau)
    assert len(terms) == 3
    for term in terms:
        assert term.shape == (B, D, T)


def test_energy_terms_defaults():
    model = TweedieSolver(state_dim=3, hidden_channels=[4, 8], K_inner=2, N_outer=2)
    B, D, T = 2, 3, 50
    x = torch.randn(B, D, T)
    obs = torch.randn(2, 50, 3)
    x_tau = torch.randn(B, D, T)
    tau = torch.full((B,), 0.5)
    terms = model.energy_terms(x, obs, x_tau, tau, obs_operator=None, prior_operator=None)
    assert torch.allclose(terms[0], torch.zeros(B, D, T))
    assert torch.allclose(terms[1], torch.zeros(B, D, T))


def test_forward_shape():
    model = TweedieSolver(state_dim=3, hidden_channels=[4, 8], K_inner=2, N_outer=2)
    obs = torch.randn(2, 50, 3)
    torch.manual_seed(42)
    out = model(obs)
    assert out.shape == (2, 50, 3)


def test_forward_reproducibility():
    torch.manual_seed(42)
    obs = torch.randn(2, 50, 3)
    torch.manual_seed(42)
    model1 = TweedieSolver(state_dim=3, hidden_channels=[4, 8], K_inner=2, N_outer=2)
    out1 = model1(obs)
    torch.manual_seed(42)
    model2 = TweedieSolver(state_dim=3, hidden_channels=[4, 8], K_inner=2, N_outer=2)
    out2 = model2(obs)
    torch.testing.assert_close(out1, out2)
