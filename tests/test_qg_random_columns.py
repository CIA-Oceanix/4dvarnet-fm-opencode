import pytest
import torch

from data.qg import QGConfig, expand_obs_to_grid, make_qg_s0_s1_datasets


def _cfg(**kw):
    base = {"nx": 16, "window_days": 4.0, "spinup_years": 0.05,
            "num_windows": 1, "obs_geometry": "random_columns",
            "cols_per_day": 3, "seed": 7}
    base.update(kw)
    return QGConfig(**base)


def test_random_columns_has_obs_columns_key():
    cfg = _cfg()
    ds = make_qg_s0_s1_datasets(cfg)
    w = ds["test_s0"][0]
    assert "obs_columns" in w
    T, C = w["obs_columns"].shape
    assert C == cfg.cols_per_day
    assert T == cfg.num_steps
    assert w["obs"].shape == (T, C * cfg.ny)


def test_random_columns_daily_cadence_and_mask():
    cfg = _cfg()
    ds = make_qg_s0_s1_datasets(cfg)
    w = ds["test_s0"][0]
    spd = round(86400.0 / cfg.dt)
    ev = w["obs_mask"].nonzero(as_tuple=False).flatten().tolist()
    assert ev == list(range(0, cfg.num_steps, spd))
    for t in ev:
        assert int((w["obs_columns"][t] >= 0).all())


def test_random_columns_distinct_within_day():
    cfg = _cfg()
    ds = make_qg_s0_s1_datasets(cfg)
    w = ds["test_s0"][0]
    ev = w["obs_mask"].nonzero(as_tuple=False).flatten().tolist()
    for t in ev:
        cols = w["obs_columns"][t].tolist()
        assert len(set(cols)) == len(cols)
        assert all(0 <= c < cfg.nx for c in cols)


def test_random_columns_near_full_coverage():
    cfg = _cfg(window_days=30.0)
    ds = make_qg_s0_s1_datasets(cfg)
    w = ds["test_s0"][0]
    seen = set()
    ev = w["obs_mask"].nonzero(as_tuple=False).flatten().tolist()
    for t in ev:
        seen.update(w["obs_columns"][t].tolist())
    assert len(seen) / cfg.nx > 0.9


def test_random_columns_obs_noise_scale():
    cfg = _cfg()
    ds = make_qg_s0_s1_datasets(cfg)
    w = ds["test_s0"][0]
    psi = w["target_state_psi"].reshape(cfg.num_steps, cfg.ny, cfg.nx)
    ev = w["obs_mask"].nonzero(as_tuple=False).flatten().tolist()
    t = ev[0]
    cols = w["obs_columns"][t].tolist()
    clean = torch.cat([psi[t, :, c] for c in cols])
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
    ev = w["obs_mask"].nonzero(as_tuple=False).flatten().tolist()
    t = ev[0]
    cols = w["obs_columns"][t].tolist()
    for c, x_col in enumerate(cols):
        assert torch.allclose(
            g[t, torch.arange(cfg.ny) * cfg.nx + x_col],
            w["obs"][t, c * cfg.ny: (c + 1) * cfg.ny])


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
