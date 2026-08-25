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


class TestBatchedTrajectoryGeneration:
    def _dyn(self):
        from models.lorenz96_dynamics import Lorenz96Dynamics
        return Lorenz96Dynamics(
            dt=0.001, coupling_exponent=1.6, c1=1.0, NO=8, J=4,
            h=1.0, hx=1.0, eps=0.1, fast_weights=[1.0, 1.0, 0.1, 0.1])

    def test_per_window_seeds_give_distinct_windows(self):
        dyn = self._dyn()
        seeds = np.arange(6) * 100 + 42
        F = torch.full((6,), 8.0)
        traj, forcing = dyn.generate_batch_trajectories(
            6, num_steps=100, spinup_steps=100, F_values=F, seeds=seeds,
            device=torch.device("cpu"))
        assert traj.shape == (6, 100, 40)
        assert forcing.shape == (6, 100)
        assert torch.isfinite(traj).all()
        assert not torch.allclose(traj[0], traj[1])
        assert not torch.allclose(forcing[0], forcing[1])

    def test_single_seed_legacy_path(self):
        dyn = self._dyn()
        traj, forcing = dyn.generate_batch_trajectories(
            6, num_steps=100, spinup_steps=100, device=torch.device("cpu"))
        assert traj.shape == (6, 100, 40)
        assert forcing.shape == (6, 100)
        assert torch.isfinite(traj).all()
        assert torch.allclose(traj[0], traj[1])

    def test_per_window_fast_weights_supported(self):
        dyn = self._dyn()
        seeds = np.arange(6) * 100 + 42
        F = torch.full((6,), 8.0)
        fw = torch.tensor(np.random.uniform(0.05, 1.0, size=(6, 4)).tolist())
        traj, _ = dyn.generate_batch_trajectories(
            6, num_steps=100, spinup_steps=100, F_values=F, seeds=seeds,
            fast_weights_values=fw, device=torch.device("cpu"))
        assert traj.shape == (6, 100, 40)
        assert torch.isfinite(traj).all()

    def test_batched_dataset_matches_structure(self, tiny_l96_cfg):
        rp = RandomParamLorenz96Dataset(
            tiny_l96_cfg, param_noise=0.2, max_window_retries=2)
        assert len(rp) == tiny_l96_cfg.num_windows
        w = rp[0]
        for key in ["true_state", "obs", "obs_mask", "forcing_true",
                    "forcing_corrupted", "F", "c1", "w1", "true_w1"]:
            assert key in w
        assert all(torch.isfinite(rp[i]["true_state"]).all().item()
                   for i in range(len(rp)))

    def test_batched_bias_dataset_has_da_params(self, tiny_l96_cfg):
        rb = RandomBiasLorenz96Dataset(
            tiny_l96_cfg, param_noise=0.2, bias_mode="fixed",
            max_window_retries=2)
        w = rb[0]
        for key in ["F_da", "w1_da", "param_bias"]:
            assert key in w
        assert all(torch.isfinite(rb[i]["true_state"]).all().item()
                   for i in range(len(rb)))

    def test_build_forcing_batch_vectorized(self):
        dyn = self._dyn()
        seeds = np.arange(4) * 100 + 42
        W = dyn._build_forcing_batch(4, 120, seeds, 1.0, 0.1, 0.05,
                                     0.0, 0.08, 0.2, 1.6)
        assert W.shape == (4, 120)
        assert np.isfinite(W).all()
        assert not np.allclose(W[0], W[1])

