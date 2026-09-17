"""
Unit tests for Strong 4D-Var data assimilation baseline.

Tests cover:
- Initialization
- Forward model (exact dynamics)
- Assimilation execution
- Performance comparison with Weak 4D-Var
- Degradation in CS2
- Exact dynamics (no q term)
"""
import pytest
import torch
import numpy as np
from evaluation.baselines import Strong4DVar, Weak4DVar, BaselineResult


def test_strong4dvar_initialization(device):
    """Strong4DVar object should initialize with default parameters."""
    strong = Strong4DVar(
        da_window_steps=100,
        B_var=2.0,
        R_var=0.5,
        max_iter=40,
        lr=0.1,
        dt=0.01,
        device=device,
    )
    
    assert strong.da_window_steps == 100
    assert strong.B_var == 2.0
    assert strong.R_var == 0.5
    assert strong.max_iter == 40
    assert strong.lr == 0.1
    assert strong.dt == 0.01
    assert strong.device == device


def test_strong4dvar_forward_model(device, l63_dynamics):
    """Test _forward_strong method produces valid trajectories."""
    torch.manual_seed(42)
    strong = Strong4DVar(da_window_steps=50, dt=0.01, device=device, dynamics=l63_dynamics)
    
    x0 = torch.tensor([1.0, 1.0, 20.0], device=device)
    forcing = torch.zeros(100, device=device)
    
    traj = strong._forward_strong(
        x0=x0,
        steps=50,
        start_idx=0,
        forcing=forcing,
        sigma=10.0,
        rho=28.0,
        beta=8/3,
        c1=1.0,
    )
    
    assert traj.shape == (50, 3), f"Expected shape (50, 3), got {traj.shape}"
    assert torch.allclose(traj[0], x0, rtol=1e-5), "First state should match x0"
    assert not torch.isnan(traj).any(), "Trajectory should not contain NaNs"
    assert not torch.isinf(traj).any(), "Trajectory should not contain Infs"


@pytest.mark.slow
def test_strong4dvar_assimilation_runs(cs1_dataset, cs1_config, device):
    """Strong4DVar assimilation should complete without errors."""
    torch.manual_seed(42)
    strong = Strong4DVar(
        da_window_steps=500,
        B_var=2.0,
        R_var=0.5,
        max_iter=20,  # Reduced for speed
        lr=0.1,
        dt=cs1_config.dt,
        device=device,
    )
    
    window = cs1_dataset[0]
    result = strong.assimilate(
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
    assert not np.isnan(result.rmse).any(), "RMSE should not contain NaNs"


@pytest.mark.slow
def test_strong4dvar_better_than_weak_cs1(cs1_dataset, cs1_config, device):
    """Strong 4D-Var should outperform Weak 4D-Var on CS1 (perfect model)."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    window = cs1_dataset[0]
    
    # Configure both methods identically
    weak = Weak4DVar(
        da_window_steps=500,
        B_var=2.0,
        R_var=0.5,
        Q_var=0.05,
        lr=0.02,
        opt_steps=80,
        dt=cs1_config.dt,
        device=device,
    )
    
    strong = Strong4DVar(
        da_window_steps=500,
        B_var=2.0,
        R_var=0.5,
        max_iter=30,
        lr=0.1,
        dt=cs1_config.dt,
        device=device,
    )
    
    result_weak = weak.assimilate(
        observations=window["obs"],
        obs_mask=window["obs_mask"],
        forcing=window["forcing_true"],
        true_state=window["true_state"],
        sigma=cs1_config.sigma_true,
        rho=cs1_config.rho_true,
        beta=cs1_config.beta_true,
        c1=cs1_config.c1,
    )
    
    result_strong = strong.assimilate(
        observations=window["obs"],
        obs_mask=window["obs_mask"],
        forcing=window["forcing_true"],
        true_state=window["true_state"],
        sigma=cs1_config.sigma_true,
        rho=cs1_config.rho_true,
        beta=cs1_config.beta_true,
        c1=cs1_config.c1,
    )
    
    # Strong should be better or comparable
    mean_rmse_weak = np.mean(result_weak.rmse)
    mean_rmse_strong = np.mean(result_strong.rmse)
    
    # Allow 20% tolerance (optimization stochasticity)
    assert mean_rmse_strong <= mean_rmse_weak * 1.2, \
        f"Strong RMSE ({mean_rmse_strong:.4f}) should be ≤ Weak RMSE ({mean_rmse_weak:.4f})"


@pytest.mark.slow
def test_strong4dvar_degrades_cs2(cs2_dataset, cs2_config, device):
    """Strong 4D-Var should have higher RMSE on CS2 than CS1 due to model error."""
    torch.manual_seed(123)
    
    from data.lorenz63 import Lorenz63Config, Lorenz63Dataset
    
    # Create CS1 comparison
    cs1_cfg = Lorenz63Config(
        case=1,
        param_bias=0.0,
        seed=123,
        num_windows=3,
        T_max=5.0,
        dt=0.01,
        obs_interval=20,
        R_var=0.5,
        spinup_steps=5000,
    )
    cs1_ds = Lorenz63Dataset(cs1_cfg)
    
    strong = Strong4DVar(
        da_window_steps=500,
        B_var=2.0,
        R_var=0.5,
        max_iter=20,
        lr=0.1,
        dt=0.01,
        device=device,
    )
    
    # CS1 result
    window_cs1 = cs1_ds[0]
    result_cs1 = strong.assimilate(
        observations=window_cs1["obs"],
        obs_mask=window_cs1["obs_mask"],
        forcing=window_cs1["forcing_true"],
        true_state=window_cs1["true_state"],
        sigma=cs1_cfg.sigma_true,
        rho=cs1_cfg.rho_true,
        beta=cs1_cfg.beta_true,
        c1=cs1_cfg.c1,
    )
    
    # CS2 result
    window_cs2 = cs2_dataset[0]
    result_cs2 = strong.assimilate(
        observations=window_cs2["obs"],
        obs_mask=window_cs2["obs_mask"],
        forcing=window_cs2["forcing_corrupted"],
        true_state=window_cs2["true_state"],
        sigma=cs2_config.da_params[0],
        rho=cs2_config.da_params[1],
        beta=cs2_config.da_params[2],
        c1=cs2_config.c1,
    )
    
    mean_rmse_cs1 = np.mean(result_cs1.rmse)
    mean_rmse_cs2 = np.mean(result_cs2.rmse)
    
    # CS2 should have higher error
    assert mean_rmse_cs2 > mean_rmse_cs1 * 1.2, \
        f"CS2 RMSE ({mean_rmse_cs2:.4f}) should be significantly higher than CS1 ({mean_rmse_cs1:.4f})"


def test_strong4dvar_dynamics_exact(device, l63_dynamics):
    """Strong 4D-Var uses exact dynamics (no model error term q)."""
    torch.manual_seed(42)
    strong = Strong4DVar(da_window_steps=50, dt=0.01, device=device, dynamics=l63_dynamics)
    
    # Forward model should not have a q parameter
    x0 = torch.tensor([1.0, 1.0, 20.0], device=device, requires_grad=True)
    forcing = torch.zeros(100, device=device)
    
    traj = strong._forward_strong(
        x0=x0,
        steps=50,
        start_idx=0,
        forcing=forcing,
        sigma=10.0,
        rho=28.0,
        beta=8/3,
        c1=1.0,
    )
    
    # Compute gradient to verify x0 influences trajectory
    loss = traj.sum()
    loss.backward()
    
    assert x0.grad is not None, "x0 should have gradient"
    assert torch.abs(x0.grad).sum() > 0, "x0 gradient should be non-zero"
    
    # The forward model should be deterministic given x0
    traj2 = strong._forward_strong(
        x0=x0.detach(),
        steps=50,
        start_idx=0,
        forcing=forcing,
        sigma=10.0,
        rho=28.0,
        beta=8/3,
        c1=1.0,
    )
    torch.testing.assert_close(traj.detach(), traj2, rtol=1e-6, atol=1e-6)
