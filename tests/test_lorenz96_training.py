"""Smoke tests for L96 (two-scale Lorenz-96) training infrastructure."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conf.schema import DataConfig
from data.lorenz96 import (
    Lorenz96Config,
    RandomParamLorenz96Dataset,
    RandomBiasLorenz96Dataset,
    make_l96_s0_s1_trainval,
    _make_lorenz96_dynamics,
)
from data.dataloader import FlowMatchingBatch
from models.direct_unet import DirectUNet
from models.vanilla_cfm import VanillaCFM

from train import _make_eval_batch


@pytest.fixture
def tiny_l96_cfg():
    return Lorenz96Config(
        T_max=0.1, dt=0.001, obs_interval=20,
        num_windows=2, spinup_steps=500, seed=42, param_bias=0.0,
    )


@pytest.fixture
def tiny_l96_dataset(tiny_l96_cfg):
    dyn = _make_lorenz96_dynamics(tiny_l96_cfg)
    return RandomParamLorenz96Dataset(
        tiny_l96_cfg, param_noise=0.0, dynamics=dyn)


def test_to_lorenz96_config():
    dc = DataConfig(system="lorenz96", dt=0.001, T_max=3.0, obs_interval=200,
                    NO=8, J=4, F_true=8.0, coupling_exponent_truth=1.6)
    cfg = dc.to_lorenz96_config()
    assert cfg.dt == 0.001
    assert cfg.NO == 8
    assert cfg.J == 4
    assert cfg.F_true == 8.0
    assert cfg.coupling_exponent_truth == 1.6
    assert cfg.state_dim == 40


def test_l96_dataset_keys(tiny_l96_dataset):
    w = tiny_l96_dataset[0]
    assert "true_state" in w
    assert "obs" in w
    assert "obs_mask" in w
    assert "forcing_true" in w
    assert "forcing_corrupted" in w
    assert "F" in w
    assert w["true_state"].shape[-1] == 40


def test_l96_direct_unet_forward(tiny_l96_dataset):
    w = tiny_l96_dataset[0]
    model = DirectUNet(state_dim=40, param_dim=1, hidden_channels=[8, 16])
    model.eval()
    obs = w["obs"].unsqueeze(0)
    mask = w["obs_mask"].unsqueeze(0)
    forcing = w["forcing_corrupted"].unsqueeze(0)
    params = torch.tensor([[w["F"]]], dtype=torch.float32)
    batch = FlowMatchingBatch(w["true_state"].unsqueeze(0), obs, mask,
                              forcing, params=params)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (1, w["true_state"].shape[0], 40)


def test_l96_vanilla_cfm_sample(tiny_l96_dataset):
    w = tiny_l96_dataset[0]
    model = VanillaCFM(state_dim=40, param_dim=1, hidden_channels=[8, 16])
    model.eval()
    obs = w["obs"].unsqueeze(0)
    mask = w["obs_mask"].unsqueeze(0)
    forcing = w["forcing_corrupted"].unsqueeze(0)
    params = torch.tensor([[w["F"]]], dtype=torch.float32)
    batch = FlowMatchingBatch(w["true_state"].unsqueeze(0), obs, mask,
                              forcing, params=params)
    with torch.no_grad():
        out = model.sample(batch)
    assert out.shape == (1, w["true_state"].shape[0], 40)


def test_l96_make_eval_batch(tiny_l96_dataset):
    w = tiny_l96_dataset[0]
    device = torch.device("cpu")
    batch = _make_eval_batch(w, device, param_names=("F",))
    assert batch.obs.shape == (1, w["true_state"].shape[0], 40)
    assert batch.params is not None
    assert batch.params.shape == (1, 1)
    assert abs(float(batch.params[0, 0]) - w["F"]) < 1e-6


def test_l96_param_dim0_eval_batch(tiny_l96_dataset):
    w = tiny_l96_dataset[0]
    device = torch.device("cpu")
    batch = _make_eval_batch(w, device, param_dim=0)
    assert batch.obs.shape == (1, w["true_state"].shape[0], 40)
    assert batch.params is None


def test_l96_all5_params_randomized(tiny_l96_cfg):
    dyn = _make_lorenz96_dynamics(tiny_l96_cfg)
    ds = RandomParamLorenz96Dataset(tiny_l96_cfg, param_noise=0.2, dynamics=dyn)
    w = ds[0]
    for k in ("F", "c1", "h", "hx", "eps"):
        assert k in w, f"missing {k}"
        assert f"true_{k}" in w
    assert 0.8 * tiny_l96_cfg.F_true <= w["F"] <= 1.2 * tiny_l96_cfg.F_true
    assert 0.8 * tiny_l96_cfg.c1 <= w["c1"] <= 1.2 * tiny_l96_cfg.c1
    assert 0.8 * tiny_l96_cfg.h <= w["h"] <= 1.2 * tiny_l96_cfg.h
    assert 0.8 * tiny_l96_cfg.hx <= w["hx"] <= 1.2 * tiny_l96_cfg.hx
    assert 0.8 * tiny_l96_cfg.eps <= w["eps"] <= 1.2 * tiny_l96_cfg.eps


def test_l96_s1_da_params_biased(tiny_l96_cfg):
    bias_cfg = Lorenz96Config(**{**tiny_l96_cfg.__dict__, "param_bias": 0.1})
    dyn = _make_lorenz96_dynamics(bias_cfg)
    ds = RandomBiasLorenz96Dataset(bias_cfg, param_noise=0.2, dynamics=dyn,
                                   bias_mode="fixed")
    w = ds[0]
    for k in ("F", "c1", "h", "hx", "eps"):
        assert f"{k}_da" in w, f"missing {k}_da"
    assert abs(w["param_bias"] - 0.1) < 1e-6
    da_mult = w["F_da"] / w["F"]
    assert abs(da_mult - 1.1) < 1e-6


def test_make_l96_s0_s1_trainval(tiny_l96_cfg):
    ds = make_l96_s0_s1_trainval(
        tiny_l96_cfg, num_train_windows=3, num_val_windows=2,
        num_test_windows=2, param_noise=0.2, bias_range=(0.0, 0.2))
    assert set(ds.keys()) == {"train", "val", "test_s0", "test_s1"}
    assert len(ds["train"]) == 3
    assert len(ds["val"]) == 2
    assert len(ds["test_s0"]) == 2
    assert len(ds["test_s1"]) == 2
    assert "F_da" in ds["test_s1"][0]  # biased test set has *_da params
    assert "F_da" not in ds["test_s0"][0]  # clean test set has no *_da params


def test_l96_direct_unet_param_dim0(tiny_l96_dataset):
    w = tiny_l96_dataset[0]
    model = DirectUNet(state_dim=40, param_dim=0, hidden_channels=[8, 16])
    model.eval()
    obs = w["obs"].unsqueeze(0)
    mask = w["obs_mask"].unsqueeze(0)
    forcing = w["forcing_corrupted"].unsqueeze(0)
    assert model.obs_dim == 41
    batch = FlowMatchingBatch(w["true_state"].unsqueeze(0), obs, mask, forcing)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (1, w["true_state"].shape[0], 40)


def test_l96_vanilla_cfm_param_dim0(tiny_l96_dataset):
    w = tiny_l96_dataset[0]
    model = VanillaCFM(state_dim=40, param_dim=0, hidden_channels=[8, 16],
                       train_tau_0_only=True)
    model.eval()
    obs = w["obs"].unsqueeze(0)
    mask = w["obs_mask"].unsqueeze(0)
    forcing = w["forcing_corrupted"].unsqueeze(0)
    assert model.obs_dim == 41
    batch = FlowMatchingBatch(w["true_state"].unsqueeze(0), obs, mask, forcing)
    with torch.no_grad():
        out = model.sample(batch)
    assert out.shape == (1, w["true_state"].shape[0], 40)
