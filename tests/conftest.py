"""
Pytest fixtures for Lorenz-63 DA testing.

Provides reusable fixtures for configs, datasets, and devices.
"""
import pytest
import torch
from data.lorenz63 import Lorenz63Config, Lorenz63Dataset


@pytest.fixture
def device():
    """CPU device for testing (ensures reproducibility)."""
    return torch.device("cpu")


@pytest.fixture
def simple_config():
    """Small, fast config for unit tests."""
    return Lorenz63Config(
        case=1,
        seed=42,
        num_windows=5,
        T_max=1.0,
        dt=0.01,
        obs_interval=10,
        spinup_steps=1000,
        R_var=0.5,
        B_var=2.0,
    )


@pytest.fixture
def cs1_config():
    """Case Study 1: noise-free forcing, correct params."""
    return Lorenz63Config(
        case=1,
        param_bias=0.0,
        seed=123,
        num_windows=5,
        T_max=5.0,
        dt=0.01,
        obs_interval=20,
        R_var=0.5,
        B_var=2.0,
        spinup_steps=10000,
    )


@pytest.fixture
def cs2_config():
    """Case Study 2: noisy forcing, biased params."""
    return Lorenz63Config(
        case=2,
        param_bias=0.05,
        seed=123,
        num_windows=5,
        T_max=5.0,
        dt=0.01,
        obs_interval=20,
        R_var=0.5,
        B_var=2.0,
        spinup_steps=10000,
    )


@pytest.fixture
def l63_dynamics(cs1_config):
    """Dynamics for the L63 DA baselines.

    EnKF/ETKF/Strong4DVar/Weak4DVar take an explicit ``dynamics=`` since the
    2026-07 case-study refactor. It defaults to ``None``, so a baseline built
    without it raises ``AttributeError: 'NoneType' object has no attribute
    'step'`` on the first ``assimilate()`` call.
    """
    from models.lorenz63_dynamics import Lorenz63Dynamics
    return Lorenz63Dynamics(
        dt=cs1_config.dt,
        coupling_exponent=cs1_config.coupling_exponent_da,
        c1=cs1_config.c1,
        c2=cs1_config.c2,
        gamma=cs1_config.gamma,
        W_L_bar=cs1_config.W_L_bar,
        sigma_0=cs1_config.sigma_0,
        sigma_L=cs1_config.sigma_L,
    )


@pytest.fixture
def cs1_dataset(cs1_config):
    """Dataset instance for Case Study 1."""
    return Lorenz63Dataset(cs1_config)


@pytest.fixture
def cs2_dataset(cs2_config):
    """Dataset instance for Case Study 2."""
    return Lorenz63Dataset(cs2_config)


@pytest.fixture
def tiny_config():
    """Ultra-small config for quick sanity checks."""
    return Lorenz63Config(
        case=1,
        seed=42,
        num_windows=2,
        T_max=0.5,
        dt=0.01,
        obs_interval=5,
        spinup_steps=500,
    )
