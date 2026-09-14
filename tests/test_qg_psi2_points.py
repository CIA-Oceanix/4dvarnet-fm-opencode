import pytest
import torch

from data.qg import QGConfig, make_qg_s0_s1_datasets
from evaluation.baselines import ETKF
from evaluation.run_qg_baselines import (
    _build_dyn,
    _build_qg_col_point_loc_matrices,
    _combined_index_at,
    _combined_obs_mask,
    _combined_observations,
    _event_columns,
    _event_points,
    _make_obs_system,
    _psi_h,
    _psi_h_combined,
    run,
)

NX = 8


def _cfg(**kw):
    base = {"nx": NX, "window_days": 4.0, "spinup_years": 0.05,
            "num_windows": 1, "obs_geometry": "random_columns",
            "cols_per_day": 2, "seed": 3}
    base.update(kw)
    return QGConfig(**base)


def _window(**kw):
    cfg = _cfg(**kw)
    ds = make_qg_s0_s1_datasets(cfg)
    return cfg, ds["test_s0"][0]


def _masked_steps(mask):
    return mask.nonzero(as_tuple=False).flatten().tolist()


def test_psi2_disabled_by_default_no_obs2_keys():
    cfg, w = _window()
    assert cfg.psi2_points_per_day == 0
    assert "obs2" not in w and "obs2_mask" not in w and "obs2_points" not in w


def test_psi2_points_daily_count_and_no_within_stream_collision():
    cfg, w = _window(psi2_points_per_day=5)
    spd = round(86400.0 / cfg.dt)
    P = cfg.psi2_points_per_day
    ev = _masked_steps(w["obs2_mask"])
    assert len(ev) == (w["obs2_mask"].shape[0] // spd) * P
    for day in range(w["obs2_mask"].shape[0] // spd):
        day_steps = [t for t in ev if day * spd <= t < (day + 1) * spd]
        assert len(day_steps) == P
        assert len(set(day_steps)) == P
        for t in day_steps:
            x, y = int(w["obs2_points"][t, 0]), int(w["obs2_points"][t, 1])
            assert 0 <= x < cfg.nx
            assert 0 <= y < cfg.ny


def test_psi2_points_can_coincide_with_psi1_columns():
    """Unlike within-stream events, the psi1-column and psi2-point streams
    are drawn independently and may land on the same step."""
    cfg, w = _window(psi2_points_per_day=10, cols_per_day=8)
    both = int((w["obs_mask"] & w["obs2_mask"]).sum())
    assert both >= 0  # never errors; collisions (if any) are handled below
    # With 8+10=18 candidate events across 12 steps/day (dt=7200s), some
    # collisions across streams are all but guaranteed over a 4-day window.
    assert both > 0


def test_psi2_obs_noise_scale():
    cfg, w = _window(psi2_points_per_day=6)
    device = torch.device("cpu")
    dyn = _build_dyn(cfg, w, device)
    # `target_state_psi`/`target_state_q` are upper-layer only; the lower
    # layer psi2 field (what the psi2 obs stream reads) comes from the full
    # 2-layer `true_state` via the same spectral inversion the generator uses.
    psi2_field = dyn.inner.streamfunctions(w["true_state"])[:, 1]
    ev = _masked_steps(w["obs2_mask"])
    t = ev[0]
    x, y = int(w["obs2_points"][t, 0]), int(w["obs2_points"][t, 1])
    clean = psi2_field[t, y, x]
    noisy = w["obs2"][t]
    resid = float(noisy - clean)
    sigma = cfg.obs_noise_std_frac * float(psi2_field.std())
    assert abs(resid) < 8.0 * sigma  # loose bound, single-sample check


def test_psi2_deterministic():
    da = make_qg_s0_s1_datasets(_cfg(psi2_points_per_day=4))
    db = make_qg_s0_s1_datasets(_cfg(psi2_points_per_day=4))
    wa, wb = da["test_s0"][0], db["test_s0"][0]
    assert torch.equal(wa["obs2_points"], wb["obs2_points"])
    m = wa["obs2_mask"]
    assert torch.equal(wa["obs2"][m], wb["obs2"][m])


def test_event_points_extraction():
    cfg, w = _window(psi2_points_per_day=3)
    pts = _event_points(cfg, w)
    ev = _masked_steps(w["obs2_mask"])
    for t in ev:
        assert pts[t] == (int(w["obs2_points"][t, 0]), int(w["obs2_points"][t, 1]))
    for t in range(cfg.num_steps):
        if t not in ev:
            assert pts[t] is None


def test_event_points_all_none_without_psi2_stream():
    cfg, w = _window()
    pts = _event_points(cfg, w)
    assert pts == [None] * cfg.num_steps


def test_psi_h_combined_matches_psi_h_when_no_point_event():
    """At a step with only a column event (no psi2 point), the combined
    H-function must reproduce `_psi_h` exactly -- the backward-compatibility
    guarantee for the pure-column case."""
    cfg, w = _window(psi2_points_per_day=3)
    device = torch.device("cpu")
    dyn = _build_dyn(cfg, w, device)
    obs_cols = _event_columns(cfg, w)
    obs_points = _event_points(cfg, w)
    h_plain = _psi_h(dyn, obs_cols, cfg.ny, cfg.nx, device)
    h_combined = _psi_h_combined(dyn, obs_cols, obs_points, cfg.ny, cfg.nx, device)
    x = torch.randn(dyn.state_dim)
    col_only = [t for t in range(cfg.num_steps)
                if obs_cols[t] and obs_points[t] is None]
    assert col_only, "fixture should contain at least one column-only step"
    t = col_only[0]
    assert torch.allclose(h_plain(x, index=t), h_combined(x, index=t), atol=1e-6)


def test_psi_h_combined_variable_width():
    cfg, w = _window(psi2_points_per_day=6, cols_per_day=6)
    device = torch.device("cpu")
    dyn = _build_dyn(cfg, w, device)
    obs_cols = _event_columns(cfg, w)
    obs_points = _event_points(cfg, w)
    h = _psi_h_combined(dyn, obs_cols, obs_points, cfg.ny, cfg.nx, device)
    x = torch.randn(dyn.state_dim)
    for t in range(cfg.num_steps):
        cols, pt = obs_cols[t], obs_points[t]
        expected = (len(cols) * cfg.ny if cols else 0) + (1 if pt is not None else 0)
        out = h(x, index=t)
        assert out.numel() == expected
        if expected == 0:
            continue
        idx_at = _combined_index_at(obs_cols, obs_points, cfg.ny)(t)
        assert idx_at.numel() == expected


def test_etkf_per_time_od_t_matches_combined_width():
    cfg, w = _window(psi2_points_per_day=6, cols_per_day=6)
    obs_cols = _event_columns(cfg, w)
    obs_points = _event_points(cfg, w)
    dyn = _build_dyn(cfg, w, torch.device("cpu"))
    h = _psi_h_combined(dyn, obs_cols, obs_points, cfg.ny, cfg.nx, torch.device("cpu"))
    from evaluation.baselines import ObsOperator
    op = ObsOperator(dyn.state_dim, h=h,
                     h_index_at=_combined_index_at(obs_cols, obs_points, cfg.ny),
                     n_obs=cfg.ny)
    method = ETKF(N_ensemble=4, R_var=1.0, dynamics=dyn, obs_operator=op)
    # od_t is only consumed by assimilate() at observed steps (guarded by
    # `if obs_mask[t]`) -- at an unobserved step it falls back to the
    # operator's constant n_obs, which is fine since nothing reads it there.
    checked_any = False
    for t in range(cfg.num_steps):
        cols, pt = obs_cols[t], obs_points[t]
        if not cols and pt is None:
            continue
        expected = (len(cols) * cfg.ny if cols else 0) + (1 if pt is not None else 0)
        _, od_t, _, _ = method._per_time(t)
        assert od_t == expected
        checked_any = True
    assert checked_any


def test_combined_observations_and_mask_shapes():
    cfg, w = _window(psi2_points_per_day=5, cols_per_day=4)
    device = torch.device("cpu")
    obs = _combined_observations(w, cfg, device)
    mask = _combined_obs_mask(w, cfg)
    assert len(obs) == cfg.num_steps
    for t in range(cfg.num_steps):
        if bool(mask[t]):
            assert obs[t] is not None
        else:
            assert obs[t] is None


def test_col_point_loc_matrices_shapes():
    cfg, w = _window(psi2_points_per_day=5, cols_per_day=4)
    device = torch.device("cpu")
    dyn = _build_dyn(cfg, w, device)
    obs_cols = _event_columns(cfg, w)
    obs_points = _event_points(cfg, w)
    Lx_t, Ly_t = _build_qg_col_point_loc_matrices(
        dyn.state_dim, obs_cols, obs_points, 2, cfg.ny, cfg.nx, 6.0, device)
    for t in range(cfg.num_steps):
        cols, pt = obs_cols[t], obs_points[t]
        od = (len(cols) * cfg.ny if cols else 0) + (1 if pt is not None else 0)
        if od == 0:
            assert Lx_t[t] is None and Ly_t[t] is None
        else:
            assert Lx_t[t].shape == (dyn.state_dim, od)
            assert Ly_t[t].shape == (od, od)


def test_make_obs_system_uses_combined_path_when_psi2_enabled():
    cfg, w = _window(psi2_points_per_day=5, cols_per_day=4)
    device = torch.device("cpu")
    obs, r_var, obs_op, loc_fn = _make_obs_system(cfg, w, device, "psi", loc_radius=6.0)
    assert loc_fn is _build_qg_col_point_loc_matrices
    assert isinstance(obs, list)
    assert len(obs) == cfg.num_steps


def test_make_obs_system_unchanged_when_psi2_disabled():
    cfg, w = _window()
    device = torch.device("cpu")
    obs, r_var, obs_op, loc_fn = _make_obs_system(cfg, w, device, "psi", loc_radius=6.0)
    assert loc_fn.__name__ == "_build_qg_col_loc_matrices"
    assert torch.is_tensor(obs)


@pytest.mark.slow
def test_etkf_run_smoke_with_psi2_points():
    """End-to-end: a small ETKF run with the combined psi1-column +
    psi2-point stream must complete and produce finite metrics."""
    cfg = _cfg(nx=16, window_days=5.0, spinup_years=0.02, num_windows=1,
              cols_per_day=4, psi2_points_per_day=10, seed=123,
              init_lag_days=0.5)
    ds = make_qg_s0_s1_datasets(cfg)
    device = torch.device("cpu")
    summary = run("etkf", cfg, ds=ds, scenarios=("test_s0",), init="lagged",
                 geometry="random_columns", obs_var="psi", band_half=0.25,
                 device=device, N_ensemble=10, loc_radius=6.0, inflation=1.0,
                 etkf_ridge=1.0, init_lag_days=cfg.init_lag_days)
    s = summary["scenarios"]["test_s0"]
    assert all(torch.isfinite(torch.tensor(v)) for v in
              (s["rmse_mean"], s["metrics_per_field"]["psi"]["layer2"]["ev"]))
