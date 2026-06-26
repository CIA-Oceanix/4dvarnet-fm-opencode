"""
Unit tests for Lorenz-63 trajectory generation and dataset creation.

Tests cover:
- Trajectory shape and reproducibility
- Bounded dynamics
- Forcing corruption
- Observation sparsity and noise
- Dataset structure
"""
import pytest
import torch
import numpy as np
from data.lorenz63 import (
    Lorenz63Config,
    Lorenz63Dataset,
    generate_long_trajectory,
    generate_corrupted_forcing,
    generate_observations,
)


def test_trajectory_shape(simple_config, device):
    """Verify trajectory has shape (num_steps, 4) with [X, Y, Z, W_L]."""
    num_steps = simple_config.num_steps
    traj = generate_long_trajectory(
        num_steps=num_steps,
        dt=simple_config.dt,
        seed=simple_config.seed,
        sigma=simple_config.sigma_true,
        rho=simple_config.rho_true,
        beta=simple_config.beta_true,
        gamma=simple_config.gamma,
        W_L_bar=simple_config.W_L_bar,
        c1=simple_config.c1,
        c2=simple_config.c2,
        sigma_0=simple_config.sigma_0,
        sigma_L=simple_config.sigma_L,
        device=device,
    )
    assert traj.shape == (num_steps, 4), f"Expected shape ({num_steps}, 4), got {traj.shape}"
    assert traj.device == device, f"Expected device {device}, got {traj.device}"


def test_trajectory_reproducibility(simple_config, device):
    """Same seed should produce identical trajectories."""
    traj1 = generate_long_trajectory(
        num_steps=simple_config.num_steps,
        dt=simple_config.dt,
        seed=42,
        sigma=simple_config.sigma_true,
        rho=simple_config.rho_true,
        beta=simple_config.beta_true,
        gamma=simple_config.gamma,
        W_L_bar=simple_config.W_L_bar,
        c1=simple_config.c1,
        c2=simple_config.c2,
        sigma_0=simple_config.sigma_0,
        sigma_L=simple_config.sigma_L,
        device=device,
    )
    traj2 = generate_long_trajectory(
        num_steps=simple_config.num_steps,
        dt=simple_config.dt,
        seed=42,
        sigma=simple_config.sigma_true,
        rho=simple_config.rho_true,
        beta=simple_config.beta_true,
        gamma=simple_config.gamma,
        W_L_bar=simple_config.W_L_bar,
        c1=simple_config.c1,
        c2=simple_config.c2,
        sigma_0=simple_config.sigma_0,
        sigma_L=simple_config.sigma_L,
        device=device,
    )
    torch.testing.assert_close(traj1, traj2, rtol=0.0, atol=0.0)


def test_trajectory_bounded(simple_config, device):
    """Lorenz-63 states should remain in physical bounds: X∈[-25,25], Y∈[-35,35], Z∈[0,55]."""
    traj = generate_long_trajectory(
        num_steps=simple_config.num_steps,
        dt=simple_config.dt,
        seed=simple_config.seed,
        sigma=simple_config.sigma_true,
        rho=simple_config.rho_true,
        beta=simple_config.beta_true,
        gamma=simple_config.gamma,
        W_L_bar=simple_config.W_L_bar,
        c1=simple_config.c1,
        c2=simple_config.c2,
        sigma_0=simple_config.sigma_0,
        sigma_L=simple_config.sigma_L,
        device=device,
    )
    X, Y, Z = traj[:, 0], traj[:, 1], traj[:, 2]
    assert torch.all(X >= -25) and torch.all(X <= 25), f"X out of bounds: [{X.min():.2f}, {X.max():.2f}]"
    assert torch.all(Y >= -35) and torch.all(Y <= 35), f"Y out of bounds: [{Y.min():.2f}, {Y.max():.2f}]"
    assert torch.all(Z >= 0) and torch.all(Z <= 55), f"Z out of bounds: [{Z.min():.2f}, {Z.max():.2f}]"


def test_forcing_corruption_cs2(cs2_dataset):
    """CS2 should have corrupted forcing different from true forcing."""
    window = cs2_dataset[0]
    W_L_true = window["forcing_true"]
    W_L_corrupted = window["forcing_corrupted"]
    
    # They should be different
    assert not torch.allclose(W_L_true, W_L_corrupted, rtol=1e-3), \
        "Corrupted forcing should differ from true forcing in CS2"
    
    # Difference should be reasonable (not too large)
    diff = torch.abs(W_L_corrupted - W_L_true)
    assert diff.mean() > 0.01, "Corruption should have non-trivial magnitude"
    assert diff.mean() < 5.0, "Corruption should not be excessively large"


def test_forcing_ou_properties(cs2_config, device):
    """Ornstein-Uhlenbeck process should have appropriate temporal correlation."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate dummy true forcing
    num_steps = cs2_config.num_steps
    W_L_true = torch.zeros(num_steps, device=device)
    
    X_stub = torch.zeros(num_steps, device=device)
    W_L_corrupted = generate_corrupted_forcing(
        W_L_true=W_L_true, X=X_stub,
        num_steps=num_steps,
        dt=cs2_config.dt,
        tau_eta=cs2_config.tau_eta,
        sigma_eta=cs2_config.sigma_eta,
        seed=42,
        device=device,
    )
    
    eta = W_L_corrupted - W_L_true
    eta_np = eta.cpu().numpy()
    
    # Compute lag-1 autocorrelation
    mean_eta = np.mean(eta_np)
    var_eta = np.var(eta_np)
    
    if var_eta > 1e-6:
        autocorr = np.corrcoef(eta_np[:-1], eta_np[1:])[0, 1]
        # OU process should have positive autocorrelation
        assert autocorr > 0.5, f"OU process should be temporally correlated (got {autocorr:.3f})"


def test_observations_sparsity(cs1_dataset, cs1_config):
    """Observations should be sparse according to obs_interval."""
    window = cs1_dataset[0]
    obs_mask = window["obs_mask"]
    num_steps = cs1_config.num_steps
    
    expected_obs = num_steps // cs1_config.obs_interval
    actual_obs = obs_mask.sum().item()
    
    # Allow ±1 tolerance for boundary effects
    assert abs(actual_obs - expected_obs) <= 1, \
        f"Expected ~{expected_obs} observations, got {actual_obs}"


def test_observations_noise(cs1_dataset, cs1_config):
    """Observation noise should have variance approximately equal to R_var."""
    window = cs1_dataset[0]
    true_state = window["true_state"]
    noisy_obs = window["obs"]
    obs_mask = window["obs_mask"]
    
    # Extract observed values
    obs_indices = torch.where(obs_mask)[0]
    noise = noisy_obs[obs_indices] - true_state[obs_indices]
    
    # Compute variance per dimension
    noise_var = torch.var(noise, dim=0)
    expected_var = cs1_config.R_var
    
    # Allow ±30% tolerance (stochastic process)
    for i, var in enumerate(noise_var):
        assert expected_var * 0.7 <= var <= expected_var * 1.3, \
            f"Noise variance dim {i}: expected ~{expected_var:.2f}, got {var:.2f}"


def test_cs1_vs_cs2_configs(cs1_config, cs2_config):
    """CS1 and CS2 should have different case settings."""
    assert cs1_config.case == 1, "CS1 should have case=1"
    assert cs2_config.case == 2, "CS2 should have case=2"
    assert cs1_config.param_bias == 0.0, "CS1 should have no parameter bias"
    assert cs2_config.param_bias > 0.0, "CS2 should have positive parameter bias"
    assert not cs1_config.use_corrupted_forcing, "CS1 should not use corrupted forcing"
    assert cs2_config.use_corrupted_forcing, "CS2 should use corrupted forcing"


def test_dataset_length(cs1_dataset, cs1_config):
    """Dataset length should equal num_windows."""
    assert len(cs1_dataset) == cs1_config.num_windows, \
        f"Expected {cs1_config.num_windows} windows, got {len(cs1_dataset)}"


def test_dataset_getitem_structure(cs1_dataset):
    """Dataset items should contain all required keys."""
    window = cs1_dataset[0]
    required_keys = {"true_state", "obs", "obs_mask", "forcing_true", "forcing_corrupted"}
    
    assert isinstance(window, dict), "Window should be a dictionary"
    assert required_keys.issubset(window.keys()), \
        f"Missing keys: {required_keys - window.keys()}"
    
    # Check shapes
    assert window["true_state"].shape[1] == 3, "true_state should have 3 dimensions (X, Y, Z)"
    assert window["obs"].shape == window["true_state"].shape, "obs should match true_state shape"
    assert window["obs_mask"].shape[0] == window["true_state"].shape[0], "obs_mask length mismatch"
