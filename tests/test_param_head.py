import os

import numpy as np
import pytest
import torch

from data.dataloader import _l96_biased_param_vector, _l96_true_param_vector
from data.lorenz96 import Lorenz96Config, RandomBiasLorenz96Dataset, _make_lorenz96_dynamics
from evaluation.run_l96 import make_obs_j_indices
from models.param_head import StateParamHead, StateParamModel

PARAM_NAMES = ("F", "c1", "hx", "eps", "w1", "w2", "w3", "w4")
SD, PD = 24, 8
REF = [8.0, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 0.1]

L1B_CKPT = "experiments/L1b_direct_unet_s0s1/checkpoints/stage1_best.ckpt"


@pytest.fixture
def bias_cfg():
    obs_var_indices = make_obs_j_indices(8, 4, 2)
    return Lorenz96Config(
        T_max=0.1, dt=0.001, obs_interval=20,
        num_windows=3, spinup_steps=500, seed=42, param_bias=0.1,
        obs_var_indices=obs_var_indices,
    )


@pytest.fixture
def bias_dataset(bias_cfg):
    dyn = _make_lorenz96_dynamics(bias_cfg)
    return RandomBiasLorenz96Dataset(bias_cfg, param_noise=0.2, dynamics=dyn,
                                     randomize_params=None)


def _batch(w, use_biased=False):
    obs_idx = make_obs_j_indices(8, 4, 2)
    states = w["true_state"][:, obs_idx].unsqueeze(0)
    obs = w["obs"].unsqueeze(0)
    mask = w["obs_mask"].unsqueeze(0)
    forcing = w["forcing_corrupted"].unsqueeze(0)
    if use_biased:
        params = torch.tensor([_l96_biased_param_vector(w)])
    else:
        params = torch.tensor([[float(w[nm]) for nm in PARAM_NAMES]])
    true_params = torch.tensor([[float(w[f"true_{nm}"]) for nm in PARAM_NAMES]])
    from data.dataloader import FlowMatchingBatch
    return FlowMatchingBatch(states, obs, mask, forcing,
                             params=params, true_params=true_params)


def test_biased_param_vector_uses_da(bias_dataset):
    w = bias_dataset[0]
    vec = _l96_biased_param_vector(w)
    assert len(vec) == 8
    for i, nm in enumerate(PARAM_NAMES[:4]):
        assert abs(vec[i] - float(w[f"{nm}_da"])) < 1e-6
    fw_da = list(w["fast_weights_da"])
    assert abs(vec[4] - fw_da[0]) < 1e-6
    assert abs(vec[7] - fw_da[3]) < 1e-6


def test_biased_vec_differs_from_true(bias_dataset):
    w = bias_dataset[0]
    biased = _l96_biased_param_vector(w)
    true = list(_l96_biased_param_vector(w))
    for i in range(4):
        true[i] = float(w[f"true_{PARAM_NAMES[i]}"])
    assert not np.allclose(biased[:5], true[:5])


def test_true_param_vector_list_form_matches_window_param_vector():
    from evaluation.neural_inference import _window_param_vector
    list_w = {"true_F": 8.0, "true_c1": 1.0, "true_hx": 1.0, "true_eps": 0.1,
              "true_fast_weights": [1.08, 0.97, 0.12, 0.11]}
    flat_w = {"true_F": 8.0, "true_c1": 1.0, "true_hx": 1.0, "true_eps": 0.1,
              "true_w1": 1.08, "true_w2": 0.97, "true_w3": 0.12, "true_w4": 0.11}
    expected = [8.0, 1.0, 1.0, 0.1, 1.08, 0.97, 0.12, 0.11]
    assert _l96_true_param_vector(list_w) == tuple(expected)
    assert _l96_true_param_vector(flat_w) == tuple(expected)
    assert _l96_true_param_vector(list_w) == tuple(_window_param_vector(list_w, prefix="true_"))
    assert _l96_true_param_vector(flat_w) == tuple(_window_param_vector(flat_w, prefix="true_"))


def test_state_param_head_shapes():
    w = None
    from models.unet import UNet1D
    model = StateParamHead(state_dim=SD, param_dim=PD, hidden_channels=[8, 16],
                           param_ref=REF)
    class _B:
        pass
    batch = _B()
    batch.obs = torch.zeros(2, 10, SD)
    batch.forcing = torch.zeros(2, 10)
    batch.params = torch.randn(2, PD)
    x_hat = torch.zeros(2, 10, SD)
    out = model(batch, x_hat)
    assert out.shape == (2, PD)
    assert torch.isfinite(out).all()
    batch.true_params = torch.randn(2, PD)
    loss = model.compute_loss(batch, x_hat)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_state_param_head_no_oracle(bias_dataset):
    w = bias_dataset[0]
    b = _batch(w, use_biased=True)
    model = StateParamHead(state_dim=SD, param_dim=PD, hidden_channels=[8, 16],
                           param_ref=REF)
    x_hat = torch.zeros(1, b.T, SD)
    out1 = model(b, x_hat)
    b2 = _batch(w, use_biased=True)
    b2.params = torch.randn_like(b2.params)
    out2 = model(b2, x_hat)
    assert not torch.allclose(out1, out2)
    assert torch.isfinite(out1).all()


def test_state_param_head_deriv_augment_shape():
    w = None
    base = StateParamHead(state_dim=SD, param_dim=PD, hidden_channels=[8, 16],
                          param_ref=REF, augment_derivatives=False)
    aug = StateParamHead(state_dim=SD, param_dim=PD, hidden_channels=[8, 16],
                         param_ref=REF, augment_derivatives=True)
    assert aug.blocks[0].conv1.in_channels == base.blocks[0].conv1.in_channels + SD
    b = _B()
    b.obs = torch.zeros(2, 10, SD)
    b.forcing = torch.zeros(2, 10)
    b.params = torch.randn(2, PD)
    b.true_params = torch.randn(2, PD)
    x_hat = torch.randn(2, 10, SD)
    out = aug(b, x_hat)
    assert out.shape == (2, PD)
    assert torch.isfinite(out).all()
    loss = aug.compute_loss(b, x_hat)
    assert torch.isfinite(loss)


def test_resample_bias_draws_vary_around_true(bias_dataset):
    from data.dataloader import FlowMatchingDataset
    obs_idx = make_obs_j_indices(8, 4, 2)
    ds = FlowMatchingDataset(
        bias_dataset, T_max=0.1, obs_interval=20,
        with_params=True, param_names=list(PARAM_NAMES),
        obs_var_indices=obs_idx, use_biased_params=True,
        resample_bias_draws=True, bias_max=0.2,
    )
    w = bias_dataset[0]
    true = [float(w[f"true_{nm}"]) for nm in PARAM_NAMES]
    vals = [[] for _ in range(8)]
    for _ in range(50):
        item = ds[0]
        p = [float(x) for x in item[4:4 + 8]]
        for i in range(8):
            vals[i].append(p[i])
    for i in range(8):
        assert len(set(round(v, 3) for v in vals[i])) > 5, f"param {i} not varying"
        assert all(v / true[i] >= 1.0 - 1e-6 for v in vals[i]), f"param {i} not positive-only"
        assert abs(float(np.mean(vals[i])) / true[i] - 1.1) < 0.05, f"param {i} mean not ~1.1x"
        assert all(abs(v / true[i] - 1.0) <= 0.2 + 1e-6 for v in vals[i])


def _B():
    class _B:
        pass
    return _B()


def test_state_param_model_frozen_encoder_optional():
    if not os.path.exists(L1B_CKPT):
        pytest.skip("L1b checkpoint not available (need master worktree copy)")
    model = StateParamModel(
        state_dim=SD, param_dim=PD,
        state_checkpoint=L1B_CKPT,
        state_hidden_channels=[64, 128, 256],
        param_head_channels=[8, 16],
        param_ref=REF,
        device=torch.device("cpu"),
    )
    assert all(not p.requires_grad for p in model.state_encoder.parameters())
    assert all(p.requires_grad for p in model.param_head.parameters())
    w = None
    from data.lorenz96 import RandomParamLorenz96Dataset
    obs_idx = make_obs_j_indices(8, 4, 2)
    cfg = Lorenz96Config(T_max=0.1, dt=0.001, obs_interval=20, num_windows=1,
                         spinup_steps=500, seed=42, obs_var_indices=obs_idx)
    dyn = _make_lorenz96_dynamics(cfg)
    ds = RandomParamLorenz96Dataset(cfg, param_noise=0.2, dynamics=dyn)
    b = _batch(ds[0], use_biased=True)
    loss = model.compute_loss(b)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or p.grad.norm() == 0
               for p in model.state_encoder.parameters())
    assert any(p.grad is not None and p.grad.norm() > 0
               for p in model.param_head.parameters())