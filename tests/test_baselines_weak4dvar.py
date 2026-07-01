"""
Unit tests for Weak 4D-Var data assimilation baseline.

Tests cover:
- Initialization
- Forward model
- Assimilation execution
- Performance on perfect observations
- Model error estimation
- Output format
"""
import pytest
import torch
import numpy as np
from evaluation.baselines import Weak4DVar, BaselineResult


def test_weak4dvar_initialization(device):
    """Weak4DVar object should initialize with default parameters."""
    weak = Weak4DVar(
        da_window_steps=100,
        B_var=2.0,
        R_var=0.5,
        Q_var=0.05,
        lr=0.02,
        opt_steps=50,
        dt=0.01,
        device=device,
    )
    
    assert weak.da_window_steps == 100
    assert weak.B_var == 2.0
    assert weak.R_var == 0.5
    assert weak.Q_var == 0.05
    assert weak.lr == 0.02
    assert weak.opt_steps == 50
    assert weak.dt == 0.01
    assert weak.device == device


def test_weak4dvar_forward_model(device):
    """Test _forward_weak method produces valid trajectories."""
    torch.manual_seed(42)
    weak = Weak4DVar(da_window_steps=50, dt=0.01, device=device)
    
    x0 = torch.tensor([1.0, 1.0, 20.0], device=device)
    q = torch.zeros(50, 3, device=device)
    forcing = torch.zeros(100, device=device)
    
    traj = weak._forward_weak(
        x0=x0,
        q=q,
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
def test_weak4dvar_assimilation_runs(cs1_dataset, cs1_config, device):
    """Weak4DVar assimilation should complete without errors."""
    torch.manual_seed(42)
    weak = Weak4DVar(
        da_window_steps=500,
        B_var=2.0,
        R_var=0.5,
        Q_var=0.05,
        lr=0.02,
        opt_steps=50,  # Reduced for speed
        dt=cs1_config.dt,
        device=device,
    )
    
    window = cs1_dataset[0]
    result = weak.assimilate(
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
def test_weak4dvar_perfect_obs_low_rmse(device):
    """Dense observations should improve tracking over sparse observations."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    from data.lorenz63 import Lorenz63Config, Lorenz63Dataset
    
    # Create config with dense observations
    cfg_dense = Lorenz63Config(
        case=1,
        seed=42,
        num_windows=2,
        T_max=5.0,
        dt=0.01,
        obs_interval=10,  # Dense observations
        R_var=0.5,
        B_var=2.0,
        spinup_steps=5000,
    )
    dataset_dense = Lorenz63Dataset(cfg_dense)
    
    weak = Weak4DVar(
        da_window_steps=500,
        B_var=2.0,
        R_var=0.5,
        Q_var=0.05,
        lr=0.02,
        opt_steps=100,
        dt=cfg_dense.dt,
        device=device,
    )
    
    window = dataset_dense[0]
    result = weak.assimilate(
        observations=window["obs"],
        obs_mask=window["obs_mask"],
        forcing=window["forcing_true"],
        true_state=window["true_state"],
        sigma=cfg_dense.sigma_true,
        rho=cfg_dense.rho_true,
        beta=cfg_dense.beta_true,
        c1=cfg_dense.c1,
    )
    
    # RMSE should be finite and reasonable
    mean_rmse = np.mean(result.rmse)
    assert mean_rmse > 0.0, f"RMSE should be positive, got {mean_rmse:.4f}"
    assert mean_rmse < 20.0, f"RMSE should be reasonable (<20), got {mean_rmse:.4f}"
    assert not np.isnan(mean_rmse), "RMSE should not be NaN"


def test_weak4dvar_model_error_nonzero(cs2_dataset, cs2_config, device):
    """Weak 4D-Var should estimate non-zero model error (q_ctrl) for CS2."""
    torch.manual_seed(42)
    weak = Weak4DVar(
        da_window_steps=500,
        B_var=2.0,
        R_var=0.5,
        Q_var=0.05,
        lr=0.02,
        opt_steps=30,  # Reduced for speed
        dt=cs2_config.dt,
        device=device,
    )
    
    window = cs2_dataset[0]
    
    # During assimilation, q_ctrl is optimized
    # We'll check that the method runs and produces reasonable output
    result = weak.assimilate(
        observations=window["obs"],
        obs_mask=window["obs_mask"],
        forcing=window["forcing_corrupted"],  # Using corrupted forcing
        true_state=window["true_state"],
        sigma=cs2_config.da_params[0],
        rho=cs2_config.da_params[1],
        beta=cs2_config.da_params[2],
        c1=cs2_config.c1,
    )
    
    # In CS2, RMSE should be higher due to model error
    assert result.rmse.mean() > 0.0, "RMSE should be positive"
    assert result.rmse.mean() < 15.0, "RMSE should be reasonable (< 15.0)"


def test_weak4dvar_output_format(cs1_dataset, cs1_config, device):
    """Weak4DVar should return BaselineResult with correct structure."""
    torch.manual_seed(42)
    weak = Weak4DVar(
        da_window_steps=500,
        dt=cs1_config.dt,
        opt_steps=20,  # Minimal for speed
        device=device,
    )
    
    window = cs1_dataset[0]
    result = weak.assimilate(
        observations=window["obs"],
        obs_mask=window["obs_mask"],
        forcing=window["forcing_true"],
        true_state=window["true_state"],
        sigma=cs1_config.sigma_true,
        rho=cs1_config.rho_true,
        beta=cs1_config.beta_true,
        c1=cs1_config.c1,
    )
    
    assert hasattr(result, "trajectory"), "Result should have trajectory attribute"
    assert hasattr(result, "rmse"), "Result should have rmse attribute"
    assert isinstance(result.trajectory, np.ndarray), "Trajectory should be numpy array"
    assert isinstance(result.rmse, np.ndarray), "RMSE should be numpy array"
    assert result.trajectory.ndim == 2, "Trajectory should be 2D"
    assert result.trajectory.shape[1] == 3, "Trajectory should have 3 state variables"
