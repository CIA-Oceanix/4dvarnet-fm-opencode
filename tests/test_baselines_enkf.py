"""
Unit tests for Ensemble Kalman Filter (EnKF) baseline.

Tests cover:
- Initialization
- Ensemble creation and spread
- Forecast and analysis steps
- Assimilation execution
- Ensemble variance tracking
- Mean tracking truth
- Ensemble collapse prevention
"""
import pytest
import torch
import numpy as np
from evaluation.baselines import EnKF, BaselineResult


def test_enkf_initialization(device):
    """EnKF object should initialize with default parameters."""
    enkf = EnKF(
        N_ensemble=30,
        R_var=0.5,
        inflation=1.0,
        dt=0.01,
        device=device,
    )
    
    assert enkf.N_ensemble == 30
    assert enkf.R_var == 0.5
    assert enkf.inflation == 1.0
    assert enkf.dt == 0.01
    assert enkf.device == device


def test_enkf_ensemble_creation(cs1_dataset, device):
    """EnKF should create N_ensemble members with positive spread."""
    torch.manual_seed(42)
    enkf = EnKF(N_ensemble=30, dt=0.01, device=device)
    
    window = cs1_dataset[0]
    obs_init = window["obs"][0]
    
    # Create initial ensemble (mimicking assimilate method)
    ensemble = obs_init.clone().unsqueeze(0).repeat(enkf.N_ensemble, 1)
    ensemble += torch.randn((enkf.N_ensemble, 3), device=device) * 1.5
    
    assert ensemble.shape == (30, 3), f"Expected shape (30, 3), got {ensemble.shape}"
    
    # Check spread
    spread = torch.std(ensemble, dim=0)
    assert torch.all(spread > 0), "Ensemble spread should be positive for all dimensions"


def test_enkf_forecast_step(device):
    """Ensemble spread should increase during forecast without observations."""
    torch.manual_seed(42)
    enkf = EnKF(N_ensemble=30, dt=0.01, device=device)
    
    # Initial ensemble
    ensemble = torch.tensor([[1.0, 1.0, 20.0]], device=device).repeat(30, 1)
    ensemble += torch.randn((30, 3), device=device) * 0.5
    
    initial_spread = torch.std(ensemble, dim=0).mean()
    
    # Forecast for 50 steps without observations
    forcing = torch.zeros(50, device=device)
    for t in range(50):
        W = forcing[t]
        Xe, Ye, Ze = ensemble[:, 0], ensemble[:, 1], ensemble[:, 2]
        dX = 10.0 * (Ye - Xe) + 1.0 * W
        dY = Xe * (28.0 - Ze) - Ye
        dZ = Xe * Ye - (8/3) * Ze
        ensemble[:, 0] += dX * enkf.dt
        ensemble[:, 1] += dY * enkf.dt
        ensemble[:, 2] += dZ * enkf.dt
    
    final_spread = torch.std(ensemble, dim=0).mean()
    
    # Spread should increase without observations
    assert final_spread > initial_spread * 1.5, \
        f"Spread should grow during forecast: {initial_spread:.4f} -> {final_spread:.4f}"


def test_enkf_analysis_step(device):
    """Kalman gain should reduce ensemble spread at observation times."""
    torch.manual_seed(42)
    enkf = EnKF(N_ensemble=30, R_var=0.5, dt=0.01, device=device)
    
    # Create ensemble with some spread
    ensemble = torch.tensor([[1.0, 1.0, 20.0]], device=device).repeat(30, 1)
    ensemble += torch.randn((30, 3), device=device) * 2.0
    
    initial_spread = torch.std(ensemble, dim=0).mean()
    
    # Apply Kalman update with an observation
    y_obs = torch.tensor([1.0, 1.0, 20.0], device=device)
    mean_e = torch.mean(ensemble, dim=0)
    A = ensemble - mean_e
    P_b = (A.T @ A) / (enkf.N_ensemble - 1)
    R = torch.eye(3, device=device) * enkf.R_var
    K = P_b @ torch.inverse(P_b + R)
    
    for n in range(enkf.N_ensemble):
        perturbed = y_obs + torch.randn(3, device=device) * np.sqrt(enkf.R_var)
        ensemble[n] += K @ (perturbed - ensemble[n])
    
    final_spread = torch.std(ensemble, dim=0).mean()
    
    # Analysis should reduce spread
    assert final_spread < initial_spread, \
        f"Analysis should reduce spread: {initial_spread:.4f} -> {final_spread:.4f}"


@pytest.mark.slow
def test_enkf_assimilation_runs(cs1_dataset, cs1_config, device):
    """EnKF assimilation should complete without errors."""
    torch.manual_seed(42)
    enkf = EnKF(
        N_ensemble=30,
        R_var=0.5,
        inflation=1.0,
        dt=cs1_config.dt,
        device=device,
    )
    
    window = cs1_dataset[0]
    result = enkf.assimilate(
        observations=window["obs"],
        obs_mask=window["obs_mask"],
        forcing=window["forcing_true"],
        true_state=window["true_state"],
        sigma=cs1_config.sigma_true,
        rho=cs1_config.rho_true,
        beta=cs1_config.beta_true,
        c1=cs1_config.c1,
    )
    
    assert isinstance(result, BaselineResult), "Should return BaselineResult"
    assert result.trajectory.shape == window["true_state"].shape, "Trajectory shape mismatch"
    assert result.rmse.shape == (3,), "RMSE should have 3 components"
    assert result.ensemble_variance is not None, "EnKF should return ensemble variance"
    assert not np.isnan(result.rmse).any(), "RMSE should not contain NaNs"


def test_enkf_ensemble_variance(cs1_dataset, cs1_config, device, l63_dynamics):
    """EnKF should output ensemble variance at all time steps."""
    torch.manual_seed(42)
    enkf = EnKF(
        N_ensemble=20,
        R_var=0.5,
        dt=cs1_config.dt,
        device=device,
        dynamics=l63_dynamics,
    )
    
    window = cs1_dataset[0]
    result = enkf.assimilate(
        observations=window["obs"],
        obs_mask=window["obs_mask"],
        forcing=window["forcing_true"],
        true_state=window["true_state"],
        sigma=cs1_config.sigma_true,
        rho=cs1_config.rho_true,
        beta=cs1_config.beta_true,
        c1=cs1_config.c1,
    )
    
    assert result.ensemble_variance is not None, "Ensemble variance should exist"
    assert result.ensemble_variance.shape == window["true_state"].shape, \
        f"Variance shape should match true_state: {window['true_state'].shape}"
    assert np.all(result.ensemble_variance >= 0), "Variance should be non-negative"


@pytest.mark.slow
def test_enkf_mean_tracks_truth(cs1_dataset, cs1_config, device):
    """EnKF ensemble mean should track truth with RMSE < 0.5 for CS1."""
    torch.manual_seed(42)
    enkf = EnKF(
        N_ensemble=30,
        R_var=0.5,
        inflation=1.0,
        dt=cs1_config.dt,
        device=device,
    )
    
    window = cs1_dataset[0]
    result = enkf.assimilate(
        observations=window["obs"],
        obs_mask=window["obs_mask"],
        forcing=window["forcing_true"],
        true_state=window["true_state"],
        sigma=cs1_config.sigma_true,
        rho=cs1_config.rho_true,
        beta=cs1_config.beta_true,
        c1=cs1_config.c1,
    )
    
    mean_rmse = np.mean(result.rmse)
    assert mean_rmse < 0.5, \
        f"EnKF should track truth well on CS1 (RMSE < 0.5), got {mean_rmse:.4f}"


def test_enkf_no_collapse(cs1_dataset, cs1_config, device, l63_dynamics):
    """Ensemble spread should not collapse to zero."""
    torch.manual_seed(42)
    enkf = EnKF(
        N_ensemble=25,
        R_var=0.5,
        inflation=1.0,
        dt=cs1_config.dt,
        device=device,
        dynamics=l63_dynamics,
    )
    
    window = cs1_dataset[0]
    result = enkf.assimilate(
        observations=window["obs"],
        obs_mask=window["obs_mask"],
        forcing=window["forcing_true"],
        true_state=window["true_state"],
        sigma=cs1_config.sigma_true,
        rho=cs1_config.rho_true,
        beta=cs1_config.beta_true,
        c1=cs1_config.c1,
    )
    
    # Check that variance stays above threshold throughout
    min_variance = np.min(result.ensemble_variance)
    mean_variance = np.mean(result.ensemble_variance)
    
    assert min_variance > 1e-6, f"Ensemble collapsed: min variance = {min_variance:.2e}"
    assert mean_variance > 0.01, f"Ensemble spread too small: mean variance = {mean_variance:.4f}"
