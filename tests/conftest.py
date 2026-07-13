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

# ── Shallow Water fixtures ──────────────────────────────────────────

@pytest.fixture
def sw_config():
    """Small SW config for fast unit tests (8x8 grid)."""
    from data.shallow_water import ShallowWaterConfig
    return ShallowWaterConfig(Nx=8, Ny=8, K=3, window_steps=3, num_windows=2, seed=42)

@pytest.fixture
def sw_dynamics(sw_config):
    """SW dynamics instance."""
    from models.shallow_water_dynamics import ShallowWaterDynamics
    return ShallowWaterDynamics(
        Nx=sw_config.Nx, Ny=sw_config.Ny, dt=sw_config.dt,
        K=sw_config.K, tau0=sw_config.tau0, f_cor=sw_config.f_cor,
        g1=sw_config.g1, g2=sw_config.g2, coupling=sw_config.coupling,
    )

@pytest.fixture
def sw_s0_dataset(sw_config):
    """S0 SW dataset."""
    from data.shallow_water import ShallowWaterDataset
    return ShallowWaterDataset(sw_config, scenario="S0")

@pytest.fixture
def sw_s1_dataset(sw_config):
    """S1 SW dataset (perturbed forcing)."""
    from data.shallow_water import ShallowWaterDataset
    return ShallowWaterDataset(sw_config, scenario="S1")
