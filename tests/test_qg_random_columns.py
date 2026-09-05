import pytest
import torch

from data.qg import QGConfig, expand_obs_to_grid, make_qg_s0_s1_datasets


def _cfg(**kw):
    base = {"nx": 16, "window_days": 4.0, "spinup_years": 0.05,
            "num_windows": 1, "obs_geometry": "random_columns",
            "cols_per_day": 3, "seed": 7}
    base.update(kw)
    return QGConfig(**base)


def _masked_steps(w):
    return w["obs_mask"].nonzero(as_tuple=False).flatten().tolist()


def test_random_columns_has_obs_columns_key():
    cfg = _cfg()
    ds = make_qg_s0_s1_datasets(cfg)
    w = ds["test_s0"][0]
    assert "obs_columns" in w
    assert w["obs_columns"].shape == (cfg.num_steps,)
    assert w["obs"].shape == (cfg.num_steps, cfg.ny)


def test_random_columns_independent_steps_per_day_and_mask():
    """Each of `cols_per_day` columns is observed once at its own intra-day
    step: every day carries exactly `cols_per_day` distinct masked steps inside
    its [day*spd, (day+1)*spd) window, each observing a single column."""
    cfg = _cfg()
    ds = make_qg_s0_s1_datasets(cfg)
    w = ds["test_s0"][0]
    spd = round(86400.0 / cfg.dt)
    C = cfg.cols_per_day
    ev = _masked_steps(w)
    assert len(ev) == (w["obs_mask"].shape[0] // spd) * C
    for day in range(w["obs_mask"].shape[0] // spd):
        day_steps = [t for t in ev if day * spd <= t < (day + 1) * spd]
        assert len(day_steps) == C
        assert len(set(day_steps)) == C
        for t in day_steps:
            assert 0 <= int(w["obs_columns"][t]) < cfg.nx


def test_random_columns_distinct_within_day():
    """No two columns of the same day share an x-location."""
    cfg = _cfg()
    ds = make_qg_s0_s1_datasets(cfg)
    w = ds["test_s0"][0]
    spd = round(86400.0 / cfg.dt)
    for day in range(w["obs_mask"].shape[0] // spd):
        day_steps = [t for t in _masked_steps(w) if day * spd <= t < (day + 1) * spd]
        cols = [int(w["obs_columns"][t]) for t in day_steps]
        assert len(set(cols)) == len(cols)


def test_random_columns_near_full_coverage():
    cfg = _cfg(window_days=30.0)
    ds = make_qg_s0_s1_datasets(cfg)
    w = ds["test_s0"][0]
    seen = {int(w["obs_columns"][t]) for t in _masked_steps(w)}
    assert len(seen) / cfg.nx > 0.9


def test_random_columns_obs_noise_scale():
    cfg = _cfg()
    ds = make_qg_s0_s1_datasets(cfg)
    w = ds["test_s0"][0]
    psi = w["target_state_psi"].reshape(cfg.num_steps, cfg.ny, cfg.nx)
    ev = _masked_steps(w)
    t = ev[0]
    xc = int(w["obs_columns"][t])
    clean = psi[t, :, xc]
    noisy = w["obs"][t]
    resid = noisy - clean
    sigma = cfg.obs_noise_std_frac * float(psi.std())
    assert float(resid.std()) == pytest.approx(sigma, rel=0.4)


def test_expand_obs_to_grid_random_columns_roundtrip():
    cfg = _cfg()
    ds = make_qg_s0_s1_datasets(cfg)
    w = ds["test_s0"][0]
    g = expand_obs_to_grid(w, cfg)
    assert g.shape == (cfg.num_steps, cfg.ny * cfg.nx)
    t = _masked_steps(w)[0]
    xc = int(w["obs_columns"][t])
    assert torch.allclose(
        g[t, torch.arange(cfg.ny) * cfg.nx + xc],
        w["obs"][t])


def test_random_columns_deterministic():
    cfg = _cfg()
    da = make_qg_s0_s1_datasets(cfg)
    db = make_qg_s0_s1_datasets(_cfg())
    for scen in ("test_s0", "test_s1"):
        wa = da[scen][0]
        wb = db[scen][0]
        assert torch.equal(wa["obs_columns"], wb["obs_columns"])
        m = wa["obs_mask"]
        assert torch.equal(wa["obs"][m], wb["obs"][m])
