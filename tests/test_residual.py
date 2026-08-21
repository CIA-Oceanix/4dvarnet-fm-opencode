"""Tests for MeanEstimatorCell and IterativeUpdateCell."""
import torch
from models.residual import MeanEstimatorCell, IterativeUpdateCell


def test_mean_estimator_shape():
    B, D, L = 2, 3, 50
    cell = MeanEstimatorCell(state_dim=D, hidden_channels=[4, 8], time_emb_dim=8, use_obs=True)
    x = torch.randn(B, D, L)
    obs = torch.randn(B, D, L)
    tau = torch.rand(B)
    out = cell(x, obs, tau)
    assert out.shape == (B, D, L)


def test_mean_estimator_forward():
    cell = MeanEstimatorCell(state_dim=3, hidden_channels=[4, 8], time_emb_dim=8, use_obs=True)
    x = torch.randn(2, 3, 50)
    obs = torch.randn(2, 3, 50)
    tau = torch.rand(2)
    out = cell(x, obs, tau)
    assert out is not None
    assert not torch.isnan(out).any()


def test_iterative_update_shape():
    B, D, L = 2, 3, 50
    cell = IterativeUpdateCell(state_dim=D, hidden_channels=[4, 8], time_emb_dim=8, use_obs=True, use_energy=True)
    x = torch.randn(B, D, L)
    obs = torch.randn(B, D, L)
    x_tau = torch.randn(B, D, L)
    tau = torch.rand(B)
    out = cell(x, obs, x_tau, tau)
    assert out.shape == (B, D, L)


def test_iterative_update_with_energy():
    B, D, L = 2, 3, 50
    cell = IterativeUpdateCell(state_dim=D, hidden_channels=[4, 8], time_emb_dim=8, use_obs=True, use_energy=True)
    x = torch.randn(B, D, L)
    obs = torch.randn(B, D, L)
    x_tau = torch.randn(B, D, L)
    tau = torch.rand(B)
    y_diff = torch.randn(B, D, L)
    phi_diff = torch.randn(B, D, L)
    bg_diff = torch.randn(B, D, L)
    out = cell(x, obs, x_tau, tau, y_diff=y_diff, phi_diff=phi_diff, bg_diff=bg_diff)
    assert out.shape == (B, D, L)


def test_iterative_update_no_energy():
    B, D, L = 2, 3, 50
    cell = IterativeUpdateCell(state_dim=D, hidden_channels=[4, 8], time_emb_dim=8, use_obs=True, use_energy=False)
    x = torch.randn(B, D, L)
    obs = torch.randn(B, D, L)
    x_tau = torch.randn(B, D, L)
    tau = torch.rand(B)
    out = cell(x, obs, x_tau, tau)
    assert out.shape == (B, D, L)


def test_iterative_update_energy_zeros():
    B, D, L = 2, 3, 50
    cell = IterativeUpdateCell(state_dim=D, hidden_channels=[4, 8], time_emb_dim=8, use_obs=True, use_energy=True)
    x = torch.randn(B, D, L)
    obs = torch.randn(B, D, L)
    x_tau = torch.randn(B, D, L)
    tau = torch.rand(B)
    y_diff = torch.zeros(B, D, L)
    phi_diff = torch.zeros(B, D, L)
    bg_diff = torch.zeros(B, D, L)
    out = cell(x, obs, x_tau, tau, y_diff=y_diff, phi_diff=phi_diff, bg_diff=bg_diff)
    assert out.shape == (B, D, L)
    assert not torch.isnan(out).any()
