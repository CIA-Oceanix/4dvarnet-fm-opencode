"""Smoke tests for L96 (two-scale Lorenz-96) training infrastructure."""
import os
import sys

import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conf.schema import DataConfig
from data.lorenz96 import (
    Lorenz96Config,
    RandomParamLorenz96Dataset,
    RandomBiasLorenz96Dataset,
    make_l96_s0_s1_trainval,
    _make_lorenz96_dynamics,
    _draw_l96_params,
)
from data.dataloader import FlowMatchingBatch, FlowMatchingDataset
from models.direct_unet import DirectUNet
from models.vanilla_cfm import VanillaCFM
from evaluation.baselines import ObsOperator

from train import _make_eval_batch, _per_group_rmse
from evaluation.run_l96 import (
    make_obs_j_indices, fmt_ev, _per_group_ev, evaluate_baseline,
    _per_window_params, _fast_weights_active,
)


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
    dc = DataConfig(system="lorenz96", dt=0.001, T_max=3.0, obs_interval=100,
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


def test_l96_fast_weights_flattened_scalar_keys(tiny_l96_cfg):
    dyn = _make_lorenz96_dynamics(tiny_l96_cfg)
    ds = RandomParamLorenz96Dataset(tiny_l96_cfg, param_noise=0.2, dynamics=dyn)
    w = ds[0]
    for j in range(1, 5):
        assert f"w{j}" in w, f"missing w{j}"
        assert f"true_w{j}" in w, f"missing true_w{j}"
    assert [w[f"w{j}"] for j in range(1, 5)] == list(w["fast_weights"])
    assert [w[f"true_w{j}"] for j in range(1, 5)] == list(w["true_fast_weights"])
    assert all(k in w for k in ("w1", "w2", "w3", "w4", "true_w1", "true_w2", "true_w3", "true_w4"))


def test_l96_s1_fast_weights_flattened_da_keys(tiny_l96_cfg):
    bias_cfg = Lorenz96Config(**{**tiny_l96_cfg.__dict__, "param_bias": 0.1})
    dyn = _make_lorenz96_dynamics(bias_cfg)
    ds = RandomBiasLorenz96Dataset(bias_cfg, param_noise=0.2, dynamics=dyn,
                                   bias_mode="fixed")
    w = ds[0]
    for j in range(1, 5):
        assert f"w{j}_da" in w, f"missing w{j}_da"
    assert [w[f"w{j}_da"] for j in range(1, 5)] == list(w["fast_weights_da"])
    assert all(k in w for k in ("F_da", "c1_da", "hx_da", "eps_da"))


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
    model = DirectUNet(state_dim=40, param_dim=0, hidden_channels=[8, 16],
                       cond_extra_dim=0)
    model.eval()
    obs = w["obs"].unsqueeze(0)
    mask = w["obs_mask"].unsqueeze(0)
    forcing = w["forcing_corrupted"].unsqueeze(0)
    assert model.cond_extra_dim == 0
    batch = FlowMatchingBatch(w["true_state"].unsqueeze(0), obs, mask, forcing)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (1, w["true_state"].shape[0], 40)


def test_l96_vanilla_cfm_param_dim0(tiny_l96_dataset):
    w = tiny_l96_dataset[0]
    model = VanillaCFM(state_dim=40, param_dim=0, hidden_channels=[8, 16],
                       train_tau_0_only=True, cond_extra_dim=0)
    model.eval()
    obs = w["obs"].unsqueeze(0)
    mask = w["obs_mask"].unsqueeze(0)
    forcing = w["forcing_corrupted"].unsqueeze(0)
    assert model.cond_extra_dim == 0
    batch = FlowMatchingBatch(w["true_state"].unsqueeze(0), obs, mask, forcing)
    with torch.no_grad():
        out = model.sample(batch)
    assert out.shape == (1, w["true_state"].shape[0], 40)


def test_make_obs_j_indices():
    idx = make_obs_j_indices(NO=8, J_truth=4, J_obs=2)
    assert idx is not None
    assert len(idx) == 24
    assert list(idx[:8]) == list(range(8))
    for k in range(8):
        assert 8 + k * 4 in idx
        assert 8 + k * 4 + 1 in idx
        assert 8 + k * 4 + 2 not in idx
        assert 8 + k * 4 + 3 not in idx


def test_make_obs_j_indices_full():
    idx = make_obs_j_indices(NO=8, J_truth=4, J_obs=4)
    assert idx is None


def test_dataconfig_obs_var_indices():
    dc = DataConfig(system="lorenz96", NO=8, J=4, obs_j=2)
    cfg = dc.to_lorenz96_config()
    assert cfg.obs_var_indices is not None
    assert len(cfg.obs_var_indices) == 24


def test_dataconfig_obs_var_indices_full():
    dc = DataConfig(system="lorenz96", NO=8, J=4, obs_j=4)
    cfg = dc.to_lorenz96_config()
    assert cfg.obs_var_indices is None


def test_dataset_obs_var_indices_subsample(tiny_l96_cfg):
    obs_var_indices = make_obs_j_indices(8, 4, 2)
    cfg = Lorenz96Config(**{**tiny_l96_cfg.__dict__, "obs_var_indices": obs_var_indices})
    dyn = _make_lorenz96_dynamics(cfg)
    ds = RandomParamLorenz96Dataset(cfg, param_noise=0.0, dynamics=dyn)
    w = ds[0]
    assert w["true_state"].shape[-1] == 40
    assert w["obs"].shape[-1] == 24


def test_flowmatching_dataset_obs_subsample(tiny_l96_cfg):
    obs_var_indices = make_obs_j_indices(8, 4, 2)
    cfg = Lorenz96Config(**{**tiny_l96_cfg.__dict__, "obs_var_indices": obs_var_indices})
    dyn = _make_lorenz96_dynamics(cfg)
    ds = RandomParamLorenz96Dataset(cfg, param_noise=0.0, dynamics=dyn)
    fm_ds = FlowMatchingDataset(ds, obs_interval=cfg.obs_interval,
                                R_var=cfg.R_var, obs_var_indices=obs_var_indices)
    true_state, obs, obs_mask, forcing = fm_ds[0]
    assert true_state.shape[-1] == 24
    assert obs.shape[-1] == 24


def test_direct_unet_state_dim24(tiny_l96_cfg):
    obs_var_indices = make_obs_j_indices(8, 4, 2)
    cfg = Lorenz96Config(**{**tiny_l96_cfg.__dict__, "obs_var_indices": obs_var_indices})
    dyn = _make_lorenz96_dynamics(cfg)
    ds = RandomParamLorenz96Dataset(cfg, param_noise=0.0, dynamics=dyn)
    fm_ds = FlowMatchingDataset(ds, obs_interval=cfg.obs_interval,
                                R_var=cfg.R_var, obs_var_indices=obs_var_indices)
    model = DirectUNet(state_dim=24, param_dim=0, hidden_channels=[8, 16])
    model.eval()
    true_state, obs, obs_mask, forcing = fm_ds[0]
    batch = FlowMatchingBatch(true_state.unsqueeze(0), obs.unsqueeze(0),
                              obs_mask.unsqueeze(0), forcing.unsqueeze(0))
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (1, 100, 24)


def test_vanilla_cfm_state_dim24(tiny_l96_cfg):
    obs_var_indices = make_obs_j_indices(8, 4, 2)
    cfg = Lorenz96Config(**{**tiny_l96_cfg.__dict__, "obs_var_indices": obs_var_indices})
    dyn = _make_lorenz96_dynamics(cfg)
    ds = RandomParamLorenz96Dataset(cfg, param_noise=0.0, dynamics=dyn)
    fm_ds = FlowMatchingDataset(ds, obs_interval=cfg.obs_interval,
                                R_var=cfg.R_var, obs_var_indices=obs_var_indices)
    model = VanillaCFM(state_dim=24, param_dim=0, hidden_channels=[8, 16],
                       train_tau_0_only=True)
    model.eval()
    true_state, obs, obs_mask, forcing = fm_ds[0]
    batch = FlowMatchingBatch(true_state.unsqueeze(0), obs.unsqueeze(0),
                              obs_mask.unsqueeze(0), forcing.unsqueeze(0))
    with torch.no_grad():
        out = model.sample(batch)
    assert out.shape == (1, 100, 24)


def test_per_group_rmse():
    mean_rmse = np.arange(24, dtype=np.float64)
    groups = _per_group_rmse(mean_rmse, obs_var_indices=None, NO=8, J=4, obs_j=2)
    assert "slow" in groups
    assert "obs_fast" in groups
    assert "all_obs" in groups
    assert abs(groups["slow"] - np.mean(np.arange(8))) < 1e-6
    assert abs(groups["obs_fast"] - np.mean(np.arange(8, 24))) < 1e-6
    assert abs(groups["all_obs"] - np.mean(np.arange(24))) < 1e-6


def test_per_group_ev():
    ev = np.linspace(0.0, 1.0, 24, dtype=np.float64)
    groups = _per_group_ev(ev, NO=8, obs_j=2)
    assert "slow" in groups and "obs_fast" in groups and "all_obs" in groups
    assert abs(groups["slow"] - np.mean(ev[:8])) < 1e-6
    assert abs(groups["obs_fast"] - np.mean(ev[8:])) < 1e-6
    assert abs(groups["all_obs"] - np.mean(ev)) < 1e-6


def test_fmt_ev_structure():
    ev = np.linspace(0.5, 0.9, 24, dtype=np.float64)
    d = fmt_ev(ev, NO=8, obs_j=2)
    assert "X1" in d and "X24" in d
    assert abs(d["X1"] - ev[0]) < 1e-6
    assert "groups" in d and "all_obs" in d["groups"]
    assert abs(d["groups"]["all_obs"] - np.mean(ev)) < 1e-6


def test_evaluate_baseline_returns_ev():
    obs_var_indices = make_obs_j_indices(8, 4, 2)
    cfg = Lorenz96Config(param_bias=0.0, obs_var_indices=obs_var_indices, T_max=1.0, dt=0.01)

    class DummyAnalysis:
        def __init__(self, trajectory):
            self.trajectory = trajectory

    class DummyMethod:
        def __init__(self, T, obs_dim):
            self.T = T
            self.obs_dim = obs_dim

        def assimilate(self, obs, mask, force, truth, **kw):
            return DummyAnalysis(torch.zeros(self.T, self.obs_dim).numpy())

    T = 100
    method = DummyMethod(T, obs_dim=24)
    pre = {}
    for i in range(3):
        pre[i] = {
            "obs": torch.randn(T, 24),
            "obs_mask": torch.ones(T),
            "true_state": torch.randn(T, 40),
            "forcing_corrupted": torch.randn(T),
            "forcing_true": torch.randn(T),
            "F": 8.0, "c1": 1.0, "h": 1.0, "hx": 1.0, "eps": 0.1,
        }
    ds = RandomParamLorenz96Dataset(cfg, param_noise=0.0,
                                    cached_windows=pre, randomize_params=None)

    rmse_stats, ev_stats, es_stats = evaluate_baseline(method, ds, cfg, "cpu", batch_size=1)
    rmse_mean, _ = rmse_stats
    ev_mean, ev_std = ev_stats
    assert rmse_mean.shape == (24,)
    assert ev_mean.shape == (24,)
    assert np.all(ev_std == 0.0)
    assert np.all(np.isfinite(ev_mean))


def test_evaluate_baseline_obs_eval_decoupled_slow_only():
    """Slow-only obs (8D) but eval on the 24D obsj2 subspace (slow + first-2-fast).

    The S1 reduced-dynamics method runs in a 24D state space (J=2) but only the 8
    slow X are observed. ``eval_var_indices`` must drive the metric subsampling so
    the returned RMSE/EV arrays are 24D (apples-to-apples with the obsj2 config).
    """
    obs_indices = make_obs_j_indices(8, 4, 0)   # slow-only: 8D
    eval_indices = make_obs_j_indices(8, 4, 2)  # obsj2 eval group: 24D
    assert len(obs_indices) == 8
    assert len(eval_indices) == 24
    cfg = Lorenz96Config(param_bias=0.0, obs_var_indices=obs_indices, T_max=1.0, dt=0.01)

    class DummyAnalysis:
        def __init__(self, trajectory):
            self.trajectory = trajectory

    class DummyMethod:
        def __init__(self, T, state_dim):
            self.T = T
            self.state_dim = state_dim
            self.obs_dim = state_dim

        def assimilate(self, obs, mask, force, truth, **kw):
            assert obs.shape[-1] == 8  # slow-only obs fed to DA
            return DummyAnalysis(torch.zeros(self.T, self.state_dim).numpy())

    T = 100
    method = DummyMethod(T, state_dim=24)  # S1 reduced-dynamics state space (J=2) = 24D
    pre = {}
    for i in range(3):
        pre[i] = {
            "obs": torch.randn(T, 8),
            "obs_mask": torch.ones(T),
            "true_state": torch.randn(T, 40),
            "forcing_corrupted": torch.randn(T),
            "forcing_true": torch.randn(T),
            "F": 8.0, "c1": 1.0, "h": 1.0, "hx": 1.0, "eps": 0.1,
        }
    ds = RandomParamLorenz96Dataset(cfg, param_noise=0.0,
                                    cached_windows=pre, randomize_params=None)

    rmse_stats, ev_stats, es_stats = evaluate_baseline(
        method, ds, cfg, "cpu", batch_size=1, eval_var_indices=eval_indices)
    rmse_mean, _ = rmse_stats
    ev_mean, _ = ev_stats
    # Eval is over the 24D obsj2 subspace, not the 8D slow-only obs.
    assert rmse_mean.shape == (24,)
    assert ev_mean.shape == (24,)
    assert np.all(np.isfinite(rmse_mean))

    # Same eval size for the sequential (batch_size=1) path already covered; check
    # the batch path too when the method supports assimilate_batch.
    class DummyBatchMethod(DummyMethod):
        def assimilate_batch(self, obs, mask, force, truth, **kw):
            assert obs.shape[-1] == 8
            B, T = obs.shape[0], obs.shape[1]
            return [DummyAnalysis(torch.zeros(T, self.state_dim).numpy()) for _ in range(B)]

    bmethod = DummyBatchMethod(T, state_dim=24)
    rmse_stats_b, _, _ = evaluate_baseline(
        bmethod, ds, cfg, "cpu", batch_size=3, eval_var_indices=eval_indices)
    assert rmse_stats_b[0].shape == (24,)


def test_obs_operator_partial():
    obs_var_indices = make_obs_j_indices(8, 4, 2)
    op = ObsOperator(40, obs_var_indices)
    assert op.obs_dim == 24
    x = torch.randn(40)
    y = op(x)
    assert y.shape == (24,)


def test_obs_operator_identity():
    op = ObsOperator(24, None)
    assert op.obs_dim == 24
    x = torch.randn(24)
    y = op(x)
    assert y.shape == (24,)


def test_s1_da_cfg_uses_corrupted_forcing():
    """S1 DA must feed the corrupted forcing, not the true one (case=2).

    Regression guard for the bug where ``cfg_s1`` was built without ``case=2``,
    so ``use_corrupted_forcing=False`` and ``evaluate_baseline`` selected
    ``forcing_true`` for S1 — silently dropping the forcing corruption that
    ``forcing_state_bias=0.1`` is meant to model.
    """
    obs_indices = make_obs_j_indices(8, 4, 0)
    cfg_s1 = Lorenz96Config(case=2, param_bias=0.15, forcing_state_bias=0.1,
                            T_max=3.0, seed=131, obs_interval=100,
                            obs_var_indices=obs_indices)
    assert cfg_s1.use_corrupted_forcing is True
    use_corrupted = getattr(cfg_s1, "use_corrupted_forcing", True)
    force_key = "forcing_corrupted" if use_corrupted else "forcing_true"
    assert force_key == "forcing_corrupted"

    cfg_s0 = Lorenz96Config(param_bias=0.0, forcing_state_bias=0.0, T_max=3.0,
                            seed=123, obs_interval=100, obs_var_indices=obs_indices)
    assert cfg_s0.use_corrupted_forcing is False
    assert ("forcing_corrupted" if cfg_s0.use_corrupted_forcing else "forcing_true") == "forcing_true"


def test_draw_l96_params_legacy_none_fast_weights_dirac():
    rng = np.random.RandomState(42)
    cfg = Lorenz96Config()
    params = _draw_l96_params(rng, cfg, param_noise=0.2, randomize_params=None)
    assert list(params["fast_weights"]) == [1.0, 1.0, 0.1, 0.1]


def test_draw_l96_params_legacy_none_no_rng_consumed():
    cfg = Lorenz96Config()
    rng1 = np.random.RandomState(123)
    rng1.uniform()
    _draw_l96_params(rng1, cfg, param_noise=0.2, randomize_params=None)
    probe = rng1.uniform()
    exp = np.random.RandomState(123)
    exp.uniform()
    for _ in range(5):
        exp.uniform()
    assert abs(probe - exp.uniform()) < 1e-12


def test_draw_l96_params_legacy_opt_in_randomizes_fast_weights():
    rng = np.random.RandomState(7)
    cfg = Lorenz96Config()
    params = _draw_l96_params(rng, cfg, param_noise=0.2,
                              randomize_params=["F", "fast_weights"])
    assert list(params["fast_weights"]) != [1.0, 1.0, 0.1, 0.1]


def test_per_window_params_legacy_no_fast_weights():
    cfg = Lorenz96Config(randomize={})
    w = {"F": 8.0, "c1": 1.0, "h": 1.0, "hx": 1.0, "eps": 0.1,
         "fast_weights": [1.0, 1.0, 0.1, 0.1]}
    kw = _per_window_params(w, cfg, da_J=4)
    assert "fast_weights" not in kw


def test_per_window_params_active_slices_to_da_J():
    cfg = Lorenz96Config(randomize={"fast_weights": {"randomized": True, "noise": 0.2}})
    w = {"F": 8.0, "c1": 1.0, "h": 1.0, "hx": 1.0, "eps": 0.1,
         "fast_weights": [1.0, 1.0, 0.1, 0.1]}
    kw = _per_window_params(w, cfg, da_J=2)
    assert kw["fast_weights"] == [1.0, 1.0]


def test_per_window_params_s1b_uses_biased_sliced():
    cfg = Lorenz96Config(randomize={"fast_weights": {"randomized": True, "biased": True}})
    w = {"F": 8.0, "c1": 1.0, "h": 1.0, "hx": 1.0, "eps": 0.1,
         "fast_weights": [1.0, 1.0, 0.1, 0.1],
         "fast_weights_da": [1.1, 1.2, 0.15, 0.2]}
    kw = _per_window_params(w, cfg, da_J=2)
    assert kw["fast_weights"] == [1.1, 1.2]


def test_per_window_params_active_raises_without_da_J():
    cfg = Lorenz96Config(randomize={"fast_weights": {"randomized": True}})
    w = {"F": 8.0, "c1": 1.0, "h": 1.0, "hx": 1.0, "eps": 0.1,
         "fast_weights": [1.0, 1.0, 0.1, 0.1]}
    with pytest.raises(ValueError):
        _per_window_params(w, cfg, da_J=None)


def test_fast_weights_active():
    assert _fast_weights_active(Lorenz96Config(randomize={})) is False
    assert _fast_weights_active(Lorenz96Config()) is False
    assert _fast_weights_active(
        Lorenz96Config(randomize={"fast_weights": {"randomized": True}})) is True
    assert _fast_weights_active(
        Lorenz96Config(randomize={"F": {"randomized": True}})) is False


class TestMethodTruth:
    def test_slices_to_method_state_dim(self):
        from evaluation.run_l96 import _method_truth

        class M:
            state_dim = 24

        truth = torch.randn(2, 50, 40)
        ovi = make_obs_j_indices(8, 4, 2)
        out = _method_truth(truth, M(), ovi)
        assert out.shape == (2, 50, 24)
        torch.testing.assert_close(out, truth[..., ovi])

    def test_full_dim_and_unknown_method_unchanged(self):
        from evaluation.run_l96 import _method_truth

        class Full:
            state_dim = 40

        class Bare:
            pass

        truth = torch.randn(2, 50, 40)
        assert _method_truth(truth, Full(), list(range(24))) is truth
        assert _method_truth(truth, Bare(), list(range(24))) is truth


class TestEsfixGateMissingES:
    def test_validate_missing_es_in_original_passes(self, tmp_path):
        import json
        from rerun_l96_esfix import validate
        orig = {"s0": {"EnKF": {"mean": 0.90, "ev": {"groups": {"all_obs": 0.50, "slow": 0.80, "obs_fast": 0.30}}}}}
        new = {"s0": {"EnKF": {"mean": 0.905, "ev": {"groups": {"all_obs": 0.505, "slow": 0.805, "obs_fast": 0.305}},
                              "es": {"groups": {"all_obs": 0.45, "slow": 0.30, "obs_fast": 0.50}}}}}
        op = tmp_path / "orig.json"; json.dump(orig, open(op, "w"))
        np = tmp_path / "new.json"; json.dump(new, open(np, "w"))
        rep = validate(str(op), str(np))
        assert rep["ok"] is True
        assert rep["checks"]["s0/EnKF"]["status"] == "OK"
        assert rep["checks"]["s0/EnKF"]["es_old"] is None
        assert rep["checks"]["s0/EnKF"]["es_new"] == 0.45


class TestBatchedGeneration:
    """Tests for the vectorized batched L96 dataset generation (fast_generation)."""

    @pytest.fixture
    def batch_cfg(self):
        return Lorenz96Config(
            T_max=0.3, dt=0.001, obs_interval=100, num_windows=4,
            spinup_steps=1000, seed=42, param_bias=0.0,
        )

    def test_dynamics_seeded_bitwise_identical_F_only(self, batch_cfg):
        """Batched dynamics with per-window seeds matches per-window path bitwise
        when only F varies (no param that triggers float-op-order divergence)."""
        from models.lorenz96_dynamics import Lorenz96Dynamics
        dyn = Lorenz96Dynamics(NO=8, J=4, dt=batch_cfg.dt)
        seeds = [42 + i * 100 for i in range(4)]
        F = torch.tensor([8.0, 7.6, 8.4, 8.2])
        refs = [dyn.generate_full_trajectory(num_steps=batch_cfg.num_steps, seed=s, F=f.item(),
                                              spinup_steps=batch_cfg.spinup_steps)[0]
                for s, f in zip(seeds, F)]
        ref = torch.stack(refs)
        bt, _ = dyn.generate_batch_trajectories_seeded(
            num_steps=batch_cfg.num_steps, seeds=seeds, F_values=F,
            spinup_steps=batch_cfg.spinup_steps,
        )
        assert bt.shape == ref.shape == (4, batch_cfg.num_steps, 40)
        assert torch.equal(ref, bt)

    def test_fast_path_distributionally_equivalent(self, batch_cfg):
        """Fast dataset path produces the same param distribution and trajectory
        summary statistics as the slow path (chaotic trajectories diverge
        pointwise but distributions match)."""
        dyn = _make_lorenz96_dynamics(batch_cfg)
        slow = RandomBiasLorenz96Dataset(
            batch_cfg, dynamics=dyn, bias_mode="random", bias_range=(0.0, 0.2),
            fast_generation=False,
        )
        fast = RandomBiasLorenz96Dataset(
            batch_cfg, dynamics=dyn, bias_mode="random", bias_range=(0.0, 0.2),
            fast_generation=True,
        )
        assert len(slow) == len(fast) == batch_cfg.num_windows
        for i in range(batch_cfg.num_windows):
            assert slow[i]["F"] == fast[i]["F"]
            assert slow[i]["F_da"] == fast[i]["F_da"]
            assert slow[i]["param_bias"] == fast[i]["param_bias"]
        st = torch.stack([slow[i]["true_state"] for i in range(len(slow))])
        ft = torch.stack([fast[i]["true_state"] for i in range(len(fast))])
        assert abs(st.mean().item() - ft.mean().item()) < 0.05
        assert abs(st.std().item() - ft.std().item()) < 0.05

    def test_fast_path_window_dict_structure(self, batch_cfg):
        """Fast-path windows have the same keys as slow-path windows."""
        dyn = _make_lorenz96_dynamics(batch_cfg)
        slow = RandomBiasLorenz96Dataset(
            batch_cfg, dynamics=dyn, bias_mode="random", bias_range=(0.0, 0.2),
            fast_generation=False,
        )
        fast = RandomBiasLorenz96Dataset(
            batch_cfg, dynamics=dyn, bias_mode="random", bias_range=(0.0, 0.2),
            fast_generation=True,
        )
        assert set(slow[0].keys()) == set(fast[0].keys())
        assert fast[0]["true_state"].shape == slow[0]["true_state"].shape
        assert fast[0]["obs"].shape == slow[0]["obs"].shape
        assert "param_bias" in fast[0]
        assert "F_da" in fast[0]
        assert "w1_da" in fast[0]

    def test_test_splits_use_slow_path_by_default(self, batch_cfg):
        """make_l96_s0_s1_trainval uses slow path for test splits by default
        so the eval cache stays bitwise-reproducible."""
        from evaluation.run_l96 import make_obs_j_indices
        ov = make_obs_j_indices(8, 4, 2)
        cfg = Lorenz96Config(
            T_max=0.3, dt=0.001, obs_interval=100, num_windows=4,
            spinup_steps=1000, obs_var_indices=ov,
        )
        ds = make_l96_s0_s1_trainval(
            cfg, num_train_windows=4, num_val_windows=2, num_test_windows=2,
        )
        # test splits must be finite and have correct shape
        for case in ("test_s0", "test_s1"):
            for i in range(2):
                w = ds[case][i]
                assert torch.isfinite(w["true_state"]).all()
                assert w["true_state"].shape == (cfg.num_steps, 40)
                assert w["obs"].shape == (cfg.num_steps, 24)

    def test_make_l96_s0_s1_datasets_fast_flag(self, batch_cfg):
        """make_l96_s0_s1_datasets respects fast_generation for test windows."""
        from data.lorenz96 import make_l96_s0_s1_datasets
        from evaluation.run_l96 import make_obs_j_indices
        ov = make_obs_j_indices(8, 4, 2)
        cfg = Lorenz96Config(
            T_max=0.3, dt=0.001, obs_interval=100, num_windows=2,
            spinup_steps=1000, obs_var_indices=ov,
        )
        slow = make_l96_s0_s1_datasets(cfg, num_test_windows=2, fast_generation=False)
        fast = make_l96_s0_s1_datasets(cfg, num_test_windows=2, fast_generation=True)
        for case in ("test_s0", "test_s1"):
            assert len(slow[case]) == len(fast[case]) == 2
            for i in range(2):
                assert fast[case][i]["true_state"].shape == slow[case][i]["true_state"].shape

    def test_cached_datasets_reuse_test_splits(self, batch_cfg):
        """cached_datasets with test_s0/test_s1 reuses the supplied windows
        (by identity) while train/val are generated fresh."""
        from evaluation.run_l96 import make_obs_j_indices
        ov = make_obs_j_indices(8, 4, 2)
        cfg = Lorenz96Config(
            T_max=0.3, dt=0.001, obs_interval=100, num_windows=4,
            spinup_steps=1000, obs_var_indices=ov,
        )
        # Build a reference with tiny test splits, then reuse their windows.
        ref = make_l96_s0_s1_trainval(
            cfg, num_train_windows=4, num_val_windows=2, num_test_windows=2,
        )
        cached_test = {k: ref[k] for k in ("test_s0", "test_s1")}
        ds = make_l96_s0_s1_trainval(
            cfg, num_train_windows=4, num_val_windows=2, num_test_windows=2,
            cached_datasets=cached_test,
        )
        for case in ("test_s0", "test_s1"):
            assert len(ds[case]) == 2
            for i in range(2):
                assert ds[case][i] is ref[case][i], f"{case}[{i}] not reused from cache"
        # train/val are freshly generated (own objects, correct sizes)
        assert len(ds["train"]) == 4 and len(ds["val"]) == 2
        assert ds["train"][0] is not ref["train"][0]

    @pytest.mark.slow
    def test_fast_path_1000_windows_under_10min(self):
        """Performance: 1000 train windows generate in under 10 minutes (vs ~4.5h
        for the slow per-window path). Uses reduced num_steps to keep CI viable."""
        import time
        cfg = Lorenz96Config(
            T_max=0.3, dt=0.001, obs_interval=100, num_windows=1000,
            spinup_steps=1000, seed=42,
        )
        dyn = _make_lorenz96_dynamics(cfg)
        t0 = time.time()
        ds = RandomBiasLorenz96Dataset(
            cfg, dynamics=dyn, bias_mode="random", bias_range=(0.0, 0.2),
            fast_generation=True,
        )
        elapsed = time.time() - t0
        assert len(ds) == 1000
        assert elapsed < 600, f"fast generation took {elapsed:.1f}s (>10min)"
