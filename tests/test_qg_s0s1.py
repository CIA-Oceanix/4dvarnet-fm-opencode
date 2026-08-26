import math

import pytest
import torch

from data.qg import (
    QGConfig,
    expand_obs_to_grid,
    make_qg_s0_s1_datasets,
)
from models.qg_dynamics import QGDynamics


def _tiny_cfg(**kw):
    base = {"nx": 32, "window_days": 0.5, "obs_interval": 3, "num_windows": 4,
            "window_spacing_days": 1.0, "spinup_years": 0.02, "seed": 7,
            "obs_geometry": "alongtrack", "track_repeat_days": 0.05,
            "track_advance_pts": 2}
    base.update(kw)
    return QGConfig(**base)


def _periodic_dist(a, b, L):
    d = torch.abs(a - b)
    d = torch.minimum(d, L - d)
    return d


def test_scenario_set_keys_and_shared_truth():
    cfg = _tiny_cfg()
    ds = make_qg_s0_s1_datasets(cfg)
    assert set(ds.keys()) == {"test_s0", "test_s1a", "test_s1b"}
    for i in range(len(ds["test_s0"])):
        assert torch.equal(ds["test_s0"][i]["true_state"],
                           ds["test_s1a"][i]["true_state"])
        assert torch.equal(ds["test_s0"][i]["true_state"],
                           ds["test_s1b"][i]["true_state"])


def test_window_shapes():
    cfg = _tiny_cfg()
    w = make_qg_s0_s1_datasets(cfg)["test_s0"][0]
    T, ny, nx = cfg.num_steps, cfg.ny, cfg.nx
    assert w["true_state"].shape == (T, 2 * ny * nx)
    assert w["target_state_psi"].shape == (T, ny * nx)
    assert w["target_state_q"].shape == (T, ny * nx)
    assert w["obs"].shape == (T, ny)
    assert w["obs_mask"].shape == (T,)
    assert w["track_x_index"].shape == (T,)
    assert w["wind_state_true"].shape == (T, 3)
    assert w["wind_state_corrupted"].shape == (T, 3)
    assert w["wind_curl"].shape == (T, ny, nx)
    assert set(w.keys()) == {
        "true_state", "target_state_psi", "target_state_q", "obs",
        "obs_mask", "track_x_index", "obs_field", "wind_curl",
        "wind_state_true", "wind_state_corrupted", "true_params",
        "da_model", "da_params", "wind_seed", "wind_amp", "forcing_true",
        "forcing_corrupted", "init_state", "init_dt_days", "init_lead_truth"}


def test_obs_nan_pattern_and_track_advance():
    cfg = _tiny_cfg()
    w = make_qg_s0_s1_datasets(cfg)["test_s0"][0]
    mask = w["obs_mask"]
    steps_per_day = round(86400.0 / cfg.dt)
    repeat = max(1, round(cfg.track_repeat_days * steps_per_day))
    expected_times = torch.arange(0, cfg.num_steps, repeat)
    assert mask.nonzero(as_tuple=False).flatten().tolist() == expected_times.tolist()
    assert torch.isfinite(w["obs"][mask]).all()
    assert torch.isnan(w["obs"][~mask]).all()
    cols = [int(w["track_x_index"][t]) for t in expected_times]
    for j in range(1, len(cols)):
        assert cols[j] == (cols[j - 1] + cfg.track_advance_pts) % cfg.nx
    assert all(0 <= c < cfg.nx for c in cols)


def test_deterministic():
    ds_a = make_qg_s0_s1_datasets(_tiny_cfg())
    ds_b = make_qg_s0_s1_datasets(_tiny_cfg())
    for k in ("test_s0", "test_s1a", "test_s1b"):
        for wa, wb in zip(ds_a[k], ds_b[k]):
            assert torch.equal(wa["true_state"], wb["true_state"])
            assert torch.equal(wa["target_state_psi"], wb["target_state_psi"])
            assert torch.equal(wa["obs_mask"], wb["obs_mask"])
            m = wa["obs_mask"]
            assert torch.equal(wa["obs"][m], wb["obs"][m])
            assert torch.equal(wa["wind_state_corrupted"],
                              wb["wind_state_corrupted"])


def test_s0_has_no_model_error():
    ds = make_qg_s0_s1_datasets(_tiny_cfg())
    for i in range(len(ds["test_s0"])):
        w = ds["test_s0"][i]
        assert w["da_model"] == "qg2l"
        assert torch.equal(w["wind_state_corrupted"], w["wind_state_true"])
        assert torch.equal(w["forcing_corrupted"], w["forcing_true"])
        assert w["da_params"]["U1"] == w["true_params"]["U1"]
        assert w["da_params"]["rd"] == w["true_params"]["rd"]
        assert w["da_params"]["rek"] == w["true_params"]["rek"]


def test_s1a_param_bias():
    ds = make_qg_s0_s1_datasets(_tiny_cfg())
    cfg = ds["test_s0"].cfg
    b = cfg.s1_param_bias
    for i in range(len(ds["test_s1a"])):
        w = ds["test_s1a"][i]
        assert w["da_model"] == "qg2l"
        assert math.isclose(w["da_params"]["rd"],
                            w["true_params"]["rd"] * (1 - b))
        assert math.isclose(w["da_params"]["rek"],
                            w["true_params"]["rek"] * (1 - b))
        assert w["da_params"]["U1"] == w["true_params"]["U1"]


def test_s1b_structural_no_param_bias():
    ds = make_qg_s0_s1_datasets(_tiny_cfg())
    for i in range(len(ds["test_s1b"])):
        w = ds["test_s1b"][i]
        assert w["da_model"] == "qg1l"
        assert w["da_params"]["rd"] == w["true_params"]["rd"]
        assert w["da_params"]["rek"] == w["true_params"]["rek"]
        assert w["da_params"]["U1"] == w["true_params"]["U1"]


def test_s1a_s1b_share_corrupted_wind():
    ds = make_qg_s0_s1_datasets(_tiny_cfg())
    for i in range(len(ds["test_s1a"])):
        assert torch.equal(ds["test_s1a"][i]["wind_state_corrupted"],
                           ds["test_s1b"][i]["wind_state_corrupted"])


@pytest.mark.slow
def test_s1a_corrupted_wind_stats():
    cfg = _tiny_cfg(num_windows=4)
    ds = make_qg_s0_s1_datasets(cfg)
    sigma_loc = cfg.s1_loc_sigma_frac * cfg.wind_sigma
    for i in range(len(ds["test_s1a"])):
        w = ds["test_s1a"][i]
        ws_true = w["wind_state_true"]
        ws_corrupt = w["wind_state_corrupted"]
        assert not torch.equal(ws_corrupt, ws_true)
        assert torch.isfinite(ws_corrupt).all()
        a = ws_true[:, 0].double()
        ac = ws_corrupt[:, 0].double()
        if float(a.std()) > 0:
            base = a * (1.0 + cfg.s1_amp_bias)
            eta_std = float((ac - base).std()) / float(a.std())
            assert 0.7 * cfg.s1_sigma_eta_frac < eta_std < 1.3 * cfg.s1_sigma_eta_frac
            ratio = float(ac.std()) / float(a.std())
            expected = math.sqrt((1.0 + cfg.s1_amp_bias) ** 2
                                 + cfg.s1_sigma_eta_frac ** 2)
            assert 0.7 * expected < ratio < 1.3 * expected
            d = _periodic_dist(ws_corrupt[:, 1:].double(),
                               ws_true[:, 1:].double(), float(cfg.L))
            mean_d = float(d.mean())
            assert 1e3 < mean_d < 4 * sigma_loc


def test_param_draws_in_range():
    cfg = _tiny_cfg()
    ds = make_qg_s0_s1_datasets(cfg)
    for k in ("test_s0", "test_s1a", "test_s1b"):
        for w in ds[k]:
            lo, hi = 1 - cfg.param_range, 1 + cfg.param_range
            assert lo * cfg.U1 <= w["true_params"]["U1"] <= hi * cfg.U1
            assert lo * cfg.rd <= w["true_params"]["rd"] <= hi * cfg.rd
            assert lo * cfg.rek <= w["true_params"]["rek"] <= hi * cfg.rek


def test_obs_noise_level():
    cfg = _tiny_cfg(obs_field="psi", num_windows=1)
    w = make_qg_s0_s1_datasets(cfg)["test_s0"][0]
    ny, nx = cfg.ny, cfg.nx
    clean_vals = []
    noisy_vals = []
    for t in w["obs_mask"].nonzero(as_tuple=False).flatten().tolist():
        xv = int(w["track_x_index"][t])
        idx = torch.arange(ny, dtype=torch.long) * nx + xv
        clean_vals.append(w["target_state_psi"][t, idx])
        noisy_vals.append(w["obs"][t])
    resid = (torch.cat(noisy_vals) - torch.cat(clean_vals)).double()
    sigma = cfg.obs_noise_std_frac * float(w["target_state_psi"].std())
    assert 0.5 * sigma < float(resid.std()) < 1.5 * sigma


def test_targets_match_field():
    cfg = _tiny_cfg()
    w = make_qg_s0_s1_datasets(cfg)["test_s0"][0]
    dyn = QGDynamics(nx=cfg.nx, L=cfg.L, dt=cfg.dt, beta=cfg.beta,
                     rd=w["true_params"]["rd"], delta=cfg.delta,
                     U1=w["true_params"]["U1"], U2=cfg.U2,
                     rek=w["true_params"]["rek"], filterfac=cfg.filterfac)
    psi = dyn.streamfunctions(w["true_state"])[..., 0, :, :]
    q = dyn._grid(w["true_state"])[..., 0, :, :]
    ny, nx = cfg.ny, cfg.nx
    assert torch.allclose(w["target_state_psi"], psi.reshape(-1, ny * nx),
                          rtol=1e-5, atol=1e-12)
    assert torch.allclose(w["target_state_q"], q.reshape(-1, ny * nx),
                          rtol=1e-5, atol=1e-12)


def test_expand_obs_to_grid_roundtrip():
    cfg = _tiny_cfg()
    w = make_qg_s0_s1_datasets(cfg)["test_s0"][0]
    g = expand_obs_to_grid(w, cfg)
    ny, nx = cfg.ny, cfg.nx
    for t in w["obs_mask"].nonzero(as_tuple=False).flatten().tolist():
        xv = int(w["track_x_index"][t])
        row = g[t].reshape(ny, nx)
        assert int(row.isfinite().sum()) == ny
        assert torch.allclose(row[:, xv], w["obs"][t])


def test_windows_disjoint():
    cfg = _tiny_cfg()
    ds = make_qg_s0_s1_datasets(cfg)["test_s0"]
    for i in range(1, len(ds)):
        assert not torch.equal(ds[i - 1]["true_state"][-1],
                               ds[i]["true_state"][0])
