import torch
import pytest
from data.random_param_dataset import RandomParamLorenz63Dataset
from data.lorenz63 import Lorenz63Config


class TestRandomParamDataset:
    @pytest.fixture
    def tiny_cfg(self):
        return Lorenz63Config(
            case=1, seed=42, num_windows=3, T_max=0.5, dt=0.01,
            obs_interval=10, spinup_steps=500, R_var=0.5, B_var=2.0,
            sigma_true=10.0, rho_true=28.0, beta_true=8 / 3,
            gamma=0.05, W_L_bar=0.0, c1=1.0, c2=0.1,
            sigma_0=0.08, sigma_L=0.20,
        )

    def test_length(self, tiny_cfg):
        ds = RandomParamLorenz63Dataset(tiny_cfg, param_noise=0.2)
        assert len(ds) == tiny_cfg.num_windows

    def test_getitem_keys(self, tiny_cfg):
        ds = RandomParamLorenz63Dataset(tiny_cfg, param_noise=0.2)
        item = ds[0]
        expected_keys = {"true_state", "obs", "obs_mask", "forcing_true", "forcing_corrupted", "sigma", "rho", "beta"}
        assert set(item.keys()) == expected_keys, f"Got keys: {set(item.keys())}"

    def test_tensor_shapes(self, tiny_cfg):
        ds = RandomParamLorenz63Dataset(tiny_cfg, param_noise=0.2)
        item = ds[0]
        T = tiny_cfg.num_steps
        assert item["true_state"].shape == (T, 3)
        assert item["obs"].shape == (T, 3)
        assert item["obs_mask"].shape == (T,)
        assert item["forcing_true"].shape == (T,)
        assert item["forcing_corrupted"].shape == (T,)

    def test_params_vary_across_windows(self, tiny_cfg):
        ds = RandomParamLorenz63Dataset(tiny_cfg, param_noise=0.2)
        windows = [ds[i] for i in range(len(ds))]
        forcing_vals = [w["forcing_true"].mean().item() for w in windows]
        assert len(set(round(v, 4) for v in forcing_vals)) > 1, \
            "Forcing should vary across windows due to randomized params"

    def test_deterministic_with_seed(self, tiny_cfg):
        ds1 = RandomParamLorenz63Dataset(tiny_cfg, param_noise=0.2)
        ds2 = RandomParamLorenz63Dataset(tiny_cfg, param_noise=0.2)
        for i in range(len(ds1)):
            for key in ["true_state", "obs"]:
                assert torch.allclose(ds1[i][key], ds2[i][key]), \
                    f"Mismatch at window {i}, key {key}"

    def test_cs2_corrupted_forcing(self, tiny_cfg):
        cfg = Lorenz63Config(**{**tiny_cfg.__dict__, "case": 2, "param_bias": 0.15,
                                "forcing_state_bias": 0.15, "forcing_coupling": "quartic"})
        ds = RandomParamLorenz63Dataset(cfg, param_noise=0.2)
        item = ds[0]
        assert not torch.allclose(item["forcing_true"], item["forcing_corrupted"]), \
            "CS2 should have corrupted forcing"
