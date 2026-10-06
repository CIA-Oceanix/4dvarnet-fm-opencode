import torch

from data.dataloader import FlowMatchingDataset


def _window():
    return {
        "true_state": torch.zeros(4, 3), "obs": torch.zeros(4, 3),
        "obs_mask": torch.ones(4, dtype=torch.bool), "forcing_corrupted": torch.zeros(4),
        "sigma": 8.5, "rho": 23.8, "beta": 3.07, "c1": 1.0,
        "true_sigma": 10.0, "true_rho": 28.0, "true_beta": 2.667, "true_c1": 1.0,
    }


def test_l63_noisy_da_bias_resamples_between_true_and_scaled_da():
    ds = FlowMatchingDataset([_window()], with_params=True, noisy_da_bias=True, noisy_da_max=1.5)
    a, b = ds[0][4:8], ds[0][4:8]
    assert a != b
    assert a[3] == 1.0
    for got, true, da in zip(a[:3], (10.0, 28.0, 2.667), (8.5, 23.8, 3.07)):
        lo, hi = sorted((true, true + 1.5 * (da - true)))
        assert lo - 1e-6 <= got <= hi + 1e-6


def test_l63_default_params_unchanged_without_noisy_flag():
    ds = FlowMatchingDataset([_window()], with_params=True)
    assert ds[0][4:8] == (8.5, 23.8, 3.07, 1.0)
    assert ds[0][8:12] == (10.0, 28.0, 2.667, 1.0)
