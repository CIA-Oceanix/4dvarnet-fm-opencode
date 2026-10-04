import pytest
import torch

from data.qg import QGConfig, make_qg_s0_s1_datasets
from evaluation.run_qg_baselines import (
    _build_dyn,
    _event_column_groups,
    _make_obs_system,
    _psi_h_combined,
    run,
)

NX = 8


def _cfg(**kw):
    base = {"nx": NX, "window_days": 4.0, "spinup_years": 0.05,
            "num_windows": 1, "obs_geometry": "random_columns",
            "seed": 3}
    base.update(kw)
    return QGConfig(**base)


def _window(**kw):
    cfg = _cfg(**kw)
    ds = make_qg_s0_s1_datasets(cfg)
    return cfg, ds["test_s0"][0]


def test_cols_sampling_sequential_is_default():
    cfg, w = _window(cols_per_day=2)
    assert cfg.cols_sampling == "sequential"
    assert torch.is_tensor(w["obs_columns"])
    assert torch.is_tensor(w["obs"])


def test_cols_sampling_random_produces_list_storage():
    cfg, w = _window(cols_per_day=3, cols_sampling="random")
    assert isinstance(w["obs_columns"], list)
    assert isinstance(w["obs"], list)
    assert len(w["obs_columns"]) == cfg.num_steps


def test_cols_sampling_random_total_event_count():
    """Total column-events across the window == cols_per_day * n_days,
    regardless of how many land on the same step."""
    cfg, w = _window(cols_per_day=5, cols_sampling="random")
    spd = round(86400.0 / cfg.dt)
    n_days = cfg.num_steps // spd
    total = sum(len(c) for c in w["obs_columns"] if c is not None)
    assert total == 5 * n_days
    for c in w["obs_columns"]:
        if c is not None:
            for x in c:
                assert 0 <= x < cfg.nx


def test_cols_sampling_random_exceeds_steps_per_day_ceiling():
    """The whole point of this mode: a density that would hang the
    collision-avoiding sequential sampler must work here."""
    cfg = _cfg(cols_per_day=20, cols_sampling="random", dt=7200.0)  # steps_per_day = 12
    ds = make_qg_s0_s1_datasets(cfg)
    w = ds["test_s0"][0]
    spd = round(86400.0 / cfg.dt)
    assert 20 > spd
    n_days = cfg.num_steps // spd
    total = sum(len(c) for c in w["obs_columns"] if c is not None)
    assert total == 20 * n_days
    multi = [c for c in w["obs_columns"] if c is not None and len(c) > 1]
    assert len(multi) > 0, "expected at least one step with >1 column at this density"


def test_sequential_raises_instead_of_hanging_past_ceiling():
    cfg = _cfg(cols_per_day=20, dt=7200.0)  # steps_per_day = 12 < 20, sequential default
    with pytest.raises(ValueError, match="exceeds steps_per_day"):
        make_qg_s0_s1_datasets(cfg)


def test_cols_sampling_random_deterministic():
    da = make_qg_s0_s1_datasets(_cfg(cols_per_day=4, cols_sampling="random"))
    db = make_qg_s0_s1_datasets(_cfg(cols_per_day=4, cols_sampling="random"))
    wa, wb = da["test_s0"][0], db["test_s0"][0]
    assert wa["obs_columns"] == wb["obs_columns"]
    for a, b in zip(wa["obs"], wb["obs"]):
        if a is None:
            assert b is None
        else:
            assert torch.allclose(a, b)


def test_event_column_groups_matches_stored_columns():
    cfg, w = _window(cols_per_day=4, cols_sampling="random")
    groups = _event_column_groups(cfg, w)
    assert groups is w["obs_columns"]


def test_psi_h_combined_handles_multi_column_step():
    cfg, w = _window(cols_per_day=6, cols_sampling="random")
    device = torch.device("cpu")
    dyn = _build_dyn(cfg, w, device)
    obs_cols = _event_column_groups(cfg, w)
    obs_points = [None] * cfg.num_steps
    h = _psi_h_combined(dyn, obs_cols, obs_points, cfg.ny, cfg.nx, device)
    x = torch.randn(dyn.state_dim)
    multi_t = next(t for t in range(cfg.num_steps) if obs_cols[t] and len(obs_cols[t]) > 1)
    cols = obs_cols[multi_t]
    psi1 = dyn.inner.streamfunctions(x)
    manual = torch.cat([psi1[0, :, c] for c in cols])
    out = h(x, index=multi_t)
    assert out.shape == (len(cols) * cfg.ny,)
    assert torch.allclose(out, manual, atol=1e-6)


def test_make_obs_system_routes_random_sampling_through_combined_path():
    cfg, w = _window(cols_per_day=5, cols_sampling="random")
    device = torch.device("cpu")
    obs, r_var, obs_op, loc_fn = _make_obs_system(cfg, w, device, "psi", loc_radius=6.0)
    assert loc_fn.__name__ == "_build_qg_col_point_loc_matrices"
    assert isinstance(obs, list)


@pytest.mark.slow
def test_etkf_run_smoke_with_cols_sampling_random_above_ceiling():
    """The originally-requested cols_per_day=16 config: infeasible under the
    default 'sequential' sampler (would hang), must work via
    cols_sampling='random'."""
    cfg = _cfg(nx=16, window_days=5.0, spinup_years=0.02, num_windows=1,
              cols_per_day=16, cols_sampling="random", seed=123,
              init_lag_days=0.5, dt=7200.0)
    ds = make_qg_s0_s1_datasets(cfg)
    device = torch.device("cpu")
    summary = run("etkf", cfg, ds=ds, scenarios=("test_s0",), init="lagged",
                 geometry="random_columns", obs_var="psi", band_half=0.25,
                 device=device, N_ensemble=10, loc_radius=6.0, inflation=1.0,
                 etkf_ridge=1.0, init_lag_days=cfg.init_lag_days)
    s = summary["scenarios"]["test_s0"]
    assert all(torch.isfinite(torch.tensor(v)) for v in
              (s["rmse_mean"], s["metrics_per_field"]["psi"]["full"]["ev"]))


def _day_slots(cfg, w):
    """Per day, the list of observed (step-in-day, x) slots of a list-storage window."""
    spd = round(86400.0 / cfg.dt)
    days = [[] for _ in range(cfg.num_steps // spd)]
    for t, cols in enumerate(w["obs_columns"]):
        for x in cols or ():
            days[t // spd].append((t % spd, x))
    return days


def test_uniform_slots_draws_distinct_slots_per_day():
    cfg, w = _window(cols_per_day=5, cols_sampling="uniform_slots")
    assert isinstance(w["obs_columns"], list)
    for slots in _day_slots(cfg, w):
        assert len(slots) == 5
        assert len(set(slots)) == 5, "a (step, column) slot must never be drawn twice"
        assert all(0 <= x < cfg.nx for _, x in slots)


def test_uniform_slots_exceeds_steps_per_day_and_shares_steps():
    """30 > steps_per_day (12): must work, with several columns at some step."""
    cfg = _cfg(cols_per_day=30, cols_sampling="uniform_slots", dt=7200.0)
    w = make_qg_s0_s1_datasets(cfg)["test_s0"][0]
    for slots in _day_slots(cfg, w):
        assert len(slots) == 30 and len(set(slots)) == 30
    assert any(c is not None and len(c) > 1 for c in w["obs_columns"])


def test_uniform_slots_covers_the_slot_grid_uniformly():
    """Pooled over many draws, every step-in-day and every column is hit about equally."""
    from data.qg import _generate_uniform_slot_column_observations, _make_qg_dynamics
    cfg = _cfg(cols_per_day=6, window_days=1.0, dt=7200.0)
    spd = round(86400.0 / cfg.dt)
    state = torch.randn(cfg.num_steps, 2 * cfg.nx * cfg.nx, dtype=torch.float64) * 1e-5
    dyn = _make_qg_dynamics(cfg)
    steps, cols = torch.zeros(spd), torch.zeros(cfg.nx)
    for seed in range(400):
        _, _, groups = _generate_uniform_slot_column_observations(
            dyn, state, cfg.obs_field, cfg, cfg.cols_per_day, seed)
        for t, g in enumerate(groups):
            for x in g or ():
                steps[t] += 1
                cols[x] += 1
    for counts in (steps, cols):
        expected = counts.sum() / len(counts)
        assert torch.all((counts - expected).abs() < 0.2 * expected)


def test_uniform_slots_rejects_more_than_all_slots():
    cfg = _cfg(cols_per_day=12 * NX + 1, cols_sampling="uniform_slots", dt=7200.0)
    with pytest.raises(ValueError, match="slots"):
        make_qg_s0_s1_datasets(cfg)


def test_expand_obs_to_grid_handles_list_storage():
    from data.qg import expand_obs_to_grid
    cfg, w = _window(cols_per_day=20, cols_sampling="uniform_slots", dt=7200.0)
    grid = expand_obs_to_grid(w, cfg)
    for t, cols in enumerate(w["obs_columns"]):
        observed = set(cols or ())
        for x in range(cfg.nx):
            vals = grid[t, torch.arange(cfg.ny) * cfg.nx + x]
            if x in observed:
                j = cols.index(x)
                assert torch.equal(vals, w["obs"][t][j * cfg.ny:(j + 1) * cfg.ny])
            else:
                assert torch.isnan(vals).all()

