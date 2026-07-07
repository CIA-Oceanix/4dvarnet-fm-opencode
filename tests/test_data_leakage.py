"""
Sanity tests for the NaN-obs data leakage fix.

Verifies:
- Observations at unobserved time steps are NaN
- Observations at observed steps have correct noise statistics (~R_var)
- DA baselines (Weak-4DVar, Strong-4DVar) produce finite, reasonable RMSE
  when running on datasets where obs at unobserved steps is NaN
"""
import pytest
import torch
import numpy as np
from data.lorenz63 import Lorenz63Config, make_mixed_datasets
from evaluation.baselines import Weak4DVar, Strong4DVar


@pytest.fixture
def nanos_test_config():
    return Lorenz63Config(
        case=1,
        seed=42,
        num_windows=3,
        T_max=3.0,
        dt=0.01,
        obs_interval=20,
        R_var=0.5,
        B_var=2.0,
        spinup_steps=5000,
    )


def test_nan_at_unobserved(nanos_test_config, device):
    ds = make_mixed_datasets(nanos_test_config, num_test_windows=2)
    dataset = ds["test_s0"]
    for i in range(len(dataset)):
        w = dataset[i]
        obs = w["obs"]
        mask = w["obs_mask"]
        assert torch.isnan(obs[~mask]).all(), \
            f"Window {i}: obs at unobserved steps should be NaN"
        assert not torch.isnan(obs[mask]).any(), \
            f"Window {i}: obs at observed steps should not be NaN"
        # Also verify that unobserved steps of obs_mask are False
        assert (~mask).sum() > 0, \
            f"Window {i}: should have unobserved steps"


def test_obs_noise_statistics(nanos_test_config, device):
    ds = make_mixed_datasets(nanos_test_config, num_test_windows=5)
    dataset = ds["test_s0"]
    noise_samples = []
    for i in range(len(dataset)):
        w = dataset[i]
        mask = w["obs_mask"]
        obs = w["obs"]
        truth = w["true_state"]
        noise = (obs[mask] - truth[mask]).numpy()
        noise_samples.append(noise)
    all_noise = np.concatenate(noise_samples, axis=0)
    empirical_var = np.var(all_noise, axis=0)
    expected_var = nanos_test_config.R_var
    for d in range(3):
        assert abs(empirical_var[d] - expected_var) < 0.5, \
            f"Obs noise variance for dim {d}: {empirical_var[d]:.3f} (expected ~{expected_var})"


@pytest.mark.slow
def test_weak4dvar_with_nan_obs(nanos_test_config, device):
    ds = make_mixed_datasets(nanos_test_config, num_test_windows=3)
    dataset = ds["test_s0"]
    weak = Weak4DVar(da_window_steps=300, B_var=2.0, R_var=0.5,
                     Q_var=0.05, lr=0.02, opt_steps=50, dt=0.01, device=device)
    rmse_list = []
    for i in range(len(dataset)):
        w = dataset[i]
        obs = w["obs"].to(device)
        mask = w["obs_mask"].to(device)
        truth = w["true_state"]
        force = w["forcing_corrupted"].to(device)
        sigma = w.get("sigma", 10.0)
        rho = w.get("rho", 28.0)
        beta = w.get("beta", 8/3)
        result = weak.assimilate(obs, mask, force, truth,
                                 sigma=sigma, rho=rho, beta=beta, c1=1.0)
        rmse_list.append(result.rmse)
    all_rmse = np.stack(rmse_list, axis=0)
    mean_rmse = np.mean(all_rmse, axis=0)
    assert np.all(np.isfinite(mean_rmse)), \
        f"Weak-4DVar RMSE contains non-finite values: {mean_rmse}"
    assert np.all(mean_rmse > 0.0), \
        f"Weak-4DVar RMSE should be positive, got {mean_rmse}"
    assert np.mean(mean_rmse) < 30.0, \
        f"Weak-4DVar mean RMSE suspiciously high: {np.mean(mean_rmse):.3f}"


@pytest.mark.slow
def test_strong4dvar_with_nan_obs(nanos_test_config, device):
    ds = make_mixed_datasets(nanos_test_config, num_test_windows=3)
    dataset = ds["test_s0"]
    strong = Strong4DVar(da_window_steps=300, B_var=2.0, R_var=0.5,
                         max_iter=80, lr=0.1, dt=0.01, device=device)
    rmse_list = []
    for i in range(len(dataset)):
        w = dataset[i]
        obs = w["obs"].to(device)
        mask = w["obs_mask"].to(device)
        truth = w["true_state"]
        force = w["forcing_corrupted"].to(device)
        sigma = w.get("sigma", 10.0)
        rho = w.get("rho", 28.0)
        beta = w.get("beta", 8/3)
        result = strong.assimilate(obs, mask, force, truth,
                                   sigma=sigma, rho=rho, beta=beta, c1=1.0)
        rmse_list.append(result.rmse)
    all_rmse = np.stack(rmse_list, axis=0)
    mean_rmse = np.mean(all_rmse, axis=0)
    assert np.all(np.isfinite(mean_rmse)), \
        f"Strong-4DVar RMSE contains non-finite values: {mean_rmse}"
    assert np.all(mean_rmse > 0.0), \
        f"Strong-4DVar RMSE should be positive, got {mean_rmse}"
    assert np.mean(mean_rmse) < 30.0, \
        f"Strong-4DVar mean RMSE suspiciously high: {np.mean(mean_rmse):.3f}"