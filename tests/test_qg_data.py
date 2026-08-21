import pytest
import torch

from data.qg import QGConfig, QGDataset, make_qg_datasets


def _tiny_cfg(**kw):
    base = {"nx": 32, "window_days": 0.5, "obs_interval": 3, "num_windows": 2,
            "window_spacing_days": 1.0, "spinup_years": 0.02, "seed": 7}
    base.update(kw)
    return QGConfig(**base)


def test_config_step_counts():
    cfg = QGConfig(dt=7200.0, window_days=60.0, window_spacing_days=90.0,
                   spinup_years=2.0)
    assert cfg.num_steps == 60 * 12
    assert cfg.window_spacing == 90 * 12
    assert cfg.spinup_steps == 2 * 365 * 12
    assert cfg.state_dim == 2 * cfg.nx * cfg.ny


def test_dataset_window_shapes():
    cfg = _tiny_cfg()
    ds = QGDataset(cfg)
    w = ds[0]
    T, D = cfg.num_steps, cfg.state_dim
    assert w["true_state"].shape == (T, D)
    assert w["obs"].shape == (T, D)
    assert w["obs_mask"].shape == (T,)
    assert w["forcing_true"].shape == (T,)
    assert w["wind_curl"].shape == (T, cfg.ny, cfg.nx)
    assert len(ds) == cfg.num_windows


def test_dataset_deterministic():
    ds_a = QGDataset(_tiny_cfg())
    ds_b = QGDataset(_tiny_cfg())
    for wa, wb in zip(ds_a, ds_b):
        assert torch.equal(wa["true_state"], wb["true_state"])
        assert torch.equal(wa["obs_mask"], wb["obs_mask"])
        m = wa["obs_mask"]
        assert torch.equal(wa["obs"][m], wb["obs"][m])


def test_obs_nan_pattern():
    cfg = _tiny_cfg()
    ds = QGDataset(cfg)
    w = ds[0]
    mask = w["obs_mask"]
    expected_times = torch.arange(0, cfg.num_steps, cfg.obs_interval)
    assert mask.nonzero().flatten().tolist() == expected_times.tolist()
    assert torch.isfinite(w["obs"][mask]).all()
    assert torch.isnan(w["obs"][~mask]).all()


def test_obs_noise_level():
    cfg = _tiny_cfg(R_var=1e-12)
    ds = QGDataset(cfg)
    w = ds[0]
    clean = w["true_state"][w["obs_mask"]]
    noisy = w["obs"][w["obs_mask"]]
    resid_std = (noisy - clean).std().item()
    assert 5e-7 < resid_std < 2e-6


def test_windows_disjoint():
    cfg = _tiny_cfg()
    assert cfg.window_spacing > cfg.num_steps
    ds = QGDataset(cfg)
    wa, wb = ds[0]["true_state"], ds[1]["true_state"]
    assert not torch.equal(wa[-1], wb[0])


def test_forcing_no_wind_is_zero():
    cfg = _tiny_cfg(wind_amp=0.0)
    ds = QGDataset(cfg)
    w = ds[0]
    assert torch.all(w["forcing_true"] == 0.0)
    assert torch.all(w["wind_curl"] == 0.0)
    assert torch.equal(w["forcing_corrupted"], w["forcing_true"])


def test_dataset_window_shapes_wind():
    cfg = _tiny_cfg(wind_amp=1e-11)
    ds = QGDataset(cfg)
    w = ds[0]
    T = cfg.num_steps
    assert w["wind_curl"].shape == (T, cfg.ny, cfg.nx)
    assert w["forcing_true"].shape == (T,)
    assert torch.isfinite(w["wind_curl"]).all()


def test_dataset_deterministic_wind():
    cfg = _tiny_cfg(wind_amp=1e-11)
    ds_a = QGDataset(cfg)
    ds_b = QGDataset(cfg)
    for wa, wb in zip(ds_a, ds_b):
        assert torch.equal(wa["wind_curl"], wb["wind_curl"])
        assert torch.equal(wa["forcing_true"], wb["forcing_true"])


def test_wind_curl_is_field_at_center():
    cfg = _tiny_cfg(wind_amp=1e-11)
    ds = QGDataset(cfg)
    w = ds[0]
    from data.qg import _make_qg_dynamics
    dynamics = _make_qg_dynamics(cfg)
    _traj, wind_state = dynamics.generate_full_trajectory(
        num_steps=cfg.num_steps, seed=cfg.seed, spinup_steps=cfg.spinup_steps)
    for t in (0, 1, cfg.num_steps - 1):
        expected = dynamics.wind_curl_field(wind_state[t:t + 1]).squeeze(0)
        assert torch.allclose(w["wind_curl"][t], expected,
                              rtol=1e-6, atol=1e-15)


def test_wind_amplitude_std_matches_config():
    cfg = _tiny_cfg(wind_amp=3e-12, wind_tau_days=5.0)
    from data.qg import _make_qg_dynamics
    dynamics = _make_qg_dynamics(cfg)
    state = dynamics.generate_wind_state(num_steps=2000, seed=cfg.seed)
    assert 0.5 * cfg.wind_amp < float(state[:, 0].std()) < 2.0 * cfg.wind_amp
    assert cfg.wind_tau_days == 5.0


def test_make_qg_datasets_structure():
    datasets = make_qg_datasets(_tiny_cfg(num_windows=1))
    assert set(datasets.keys()) == {"train", "val", "test_cs1", "test_cs2"}
    t_train = datasets["train"][0]["true_state"]
    t_cs1 = datasets["test_cs1"][0]["true_state"]
    t_cs2 = datasets["test_cs2"][0]["true_state"]
    assert not torch.equal(t_train, t_cs1)
    assert not torch.equal(t_cs1, t_cs2)


@pytest.mark.slow
def test_dataset_nominal_spinup_equilibrated():
    cfg = QGConfig(nx=32, window_days=1.0, num_windows=1,
                   window_spacing_days=2.0, spinup_years=1.0, seed=42)
    ds = QGDataset(cfg)
    q = ds[0]["true_state"]
    assert q.std().item() > 1e-6
