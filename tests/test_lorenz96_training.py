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
