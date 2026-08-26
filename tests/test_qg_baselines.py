import numpy as np
import pytest
import torch

from data.qg import QGConfig, make_qg_s0_s1_datasets
from evaluation.baselines import (
    ETKF,
    EnKF,
    ObsOperator,
    _build_qg_col_loc_matrices,
    _build_qg_loc_matrices,
)
from evaluation.run_qg_baselines import (
    WindStateAdapter,
    _build_dyn,
    _event_columns,
    _make_obs_system,
    _per_pass_indices,
    _psi_h,
    _q_alongtrack_obs,
    _obs_spec_rc,
    _lagged_init_ensemble,
)

NX = 8


def _cfg():
    return QGConfig(nx=NX, window_days=6.0, spinup_years=0.05,
                    num_windows=1, obs_geometry="alongtrack", seed=3)


def test_obs_operator_fixed_fallback():
    op = ObsOperator(16, obs_indices=[0, 5, 10])
    x = torch.arange(16.0)
    assert torch.equal(op(x), x[[0, 5, 10]])
    assert op.index_at(0).equal(torch.tensor([0, 5, 10]))
    assert op.index_at(3).equal(torch.tensor([0, 5, 10]))


def test_obs_operator_per_time():
    op = ObsOperator(16, obs_indices=[0], obs_indices_t=[None, [1], [2], None])
    assert op.index_at(1).equal(torch.tensor([1]))
    assert op.index_at(2).equal(torch.tensor([2]))
    assert op.index_at(0) is None
    x = torch.arange(16.0)
    assert torch.equal(op(x, index=1), x[[1]])
    assert torch.equal(op(x, index=2), x[[2]])


def test_obs_operator_defaults_indices_to_first_pass():
    op = ObsOperator(16, obs_indices_t=[None, [2, 3], None, None])
    assert op.indices.equal(torch.tensor([2, 3]))
    assert op.obs_dim == 2


def test_qg_loc_matrices_shapes_and_cross_layer():
    ny = nx = NX
    state_dim = 2 * ny * nx
    obs_t = [list(range(ny)), None, [y * nx + 2 for y in range(ny)]]  # cols 0 and 2
    Lx_t, Ly_t = _build_qg_loc_matrices(state_dim, obs_t, 2, ny, nx,
                                        5.0, torch.device("cpu"))
    assert Lx_t[0].shape == (state_dim, ny)
    assert Ly_t[0].shape == (ny, ny)
    assert Lx_t[1] is None
    assert isinstance(Lx_t[0], torch.Tensor)
    assert bool(torch.isfinite(Lx_t[0]).all())
    # cross-layer rows (layer 1) should be strongly suppressed
    assert float(Lx_t[0][ny * nx:, :].max()) < 1e-3


def test_wind_adapter_forwards_wind_state():
    from models.qg1l_dynamics import QG1LDynamics
    inner = QG1LDynamics(nx=NX)
    adapter = WindStateAdapter(inner)
    state = torch.randn(inner.state_dim) * 1e-6
    ws = torch.tensor([1.0, 0.3 * inner.L, 0.5 * inner.W])
    direct = inner.step(state, wind_state_t=ws)
    via = adapter.step(state, ws)
    assert torch.allclose(direct, via, atol=1e-12)


def test_inversion_parity_between_psi_and_q_obs():
    cfg = _cfg()
    ds = make_qg_s0_s1_datasets(cfg)
    w = ds["test_s0"][0]
    obs, r_var, field_std = _q_alongtrack_obs(cfg, w, torch.device("cpu"))
    assert obs.shape == (cfg.num_steps, NX)
    assert r_var > 0 and field_std > 0
    assert torch.isnan(obs[~w["obs_mask"]]).all()
    assert torch.isfinite(obs[w["obs_mask"]]).all()


def test_per_pass_indices_layout():
    cfg = _cfg()
    ds = make_qg_s0_s1_datasets(cfg)
    w = ds["test_s0"][0]
    per_time, first = _per_pass_indices(cfg, w)
    assert len(per_time) == cfg.num_steps
    assert first is not None and len(first) == NX
    for t in w["obs_mask"].nonzero(as_tuple=False).flatten().tolist():
        assert len(per_time[t]) == NX


def test_enkf_smoke_finite_bounded():
    cfg = QGConfig(nx=NX, window_days=6.0, spinup_years=0.05,
                   num_windows=1, obs_geometry="alongtrack", seed=3)
    device = torch.device("cpu")
    ds = make_qg_s0_s1_datasets(cfg)
    w = ds["test_s0"][0]
    dyn = _build_dyn(cfg, w, device)
    obs, r_var, field_std = _q_alongtrack_obs(cfg, w, device)
    per_time, first = _per_pass_indices(cfg, w)
    op = ObsOperator(dyn.state_dim, obs_indices=first, obs_indices_t=per_time)
    filt = EnKF(N_ensemble=20, R_var=r_var, inflation=1.1, device=device,
                dynamics=dyn, obs_operator=op, noise_init_std=field_std)
    res = filt.assimilate(obs, w["obs_mask"].to(device),
                          w["wind_state_corrupted"].to(device),
                          true_state=w["true_state"])
    assert np.isfinite(res.trajectory).all()
    truth = w["true_state"].numpy()
    rmse = np.sqrt(np.mean((res.trajectory - truth) ** 2))
    assert rmse < 100.0 * field_std


def _rc_cfg(**kw):
    base = {"nx": NX, "window_days": 4.0, "spinup_years": 0.05,
            "num_windows": 1, "obs_geometry": "random_columns",
            "cols_per_day": 2, "seed": 3}
    base.update(kw)
    return QGConfig(**base)


def _rc_window(**kw):
    cfg = _rc_cfg(**kw)
    ds = make_qg_s0_s1_datasets(cfg)
    return cfg, ds["test_s0"][0]


def test_psi_h_matches_manual_inversion_slice():
    cfg, w = _rc_window()
    device = torch.device("cpu")
    dyn = _build_dyn(cfg, w, device)
    obs_cols = _event_columns(cfg, w)
    h = _psi_h(dyn, obs_cols, cfg.ny, device)
    x = torch.randn(dyn.state_dim)
    ev = w["obs_mask"].nonzero(as_tuple=False).flatten().tolist()
    t = ev[0]
    cols = obs_cols[t]
    psi1 = dyn.inner.streamfunctions(x)
    manual = torch.cat([psi1[0, :, c] for c in cols])
    auto = h(x, index=t)
    assert auto.shape == (cfg.cols_per_day * cfg.ny,)
    assert torch.allclose(auto, manual, atol=1e-6)
    # batched path
    xb = torch.randn(7, dyn.state_dim)
    ab = h(xb, index=t)
    assert ab.shape == (7, cfg.cols_per_day * cfg.ny)


def test_psi_h_per_time_columns():
    cfg, w = _rc_window()
    device = torch.device("cpu")
    dyn = _build_dyn(cfg, w, device)
    obs_cols = _event_columns(cfg, w)
    h = _psi_h(dyn, obs_cols, cfg.ny, device)
    x = torch.randn(dyn.state_dim)
    ev = w["obs_mask"].nonzero(as_tuple=False).flatten().tolist()
    t0, t1 = ev[0], ev[1]
    r0 = h(x, index=t0)
    r1 = h(x, index=t1)
    assert not torch.allclose(r0, r1, atol=1e-6)


def test_col_loc_matrices_shapes_and_cross_layer():
    ny = nx = NX
    state_dim = 2 * ny * nx
    cols_t = [[0, 3], None, [2]]
    Lx_t, Ly_t = _build_qg_col_loc_matrices(state_dim, cols_t, 2, ny, nx,
                                            5.0, torch.device("cpu"))
    assert Lx_t[0].shape == (state_dim, 2 * ny)
    assert Ly_t[0].shape == (2 * ny, 2 * ny)
    assert Lx_t[1] is None
    assert bool(torch.isfinite(Lx_t[0]).all())
    assert float(Lx_t[0][ny * nx:, :].max()) < 1e-3


def test_init_ensemble_respected_analysis0():
    cfg, w = _rc_window()
    device = torch.device("cpu")
    dyn = _build_dyn(cfg, w, device)
    obs, r_var, od = _obs_spec_rc(cfg, w, device)
    obs_op = ObsOperator(dyn.state_dim, h=lambda x, index=None: x[:, :od],
                         h_index_at=None, n_obs=od)
    init = torch.randn(12, dyn.state_dim)
    filt = ETKF(N_ensemble=12, R_var=r_var, inflation=1.1, device=device,
                dynamics=dyn, obs_operator=obs_op, init_ensemble=init.clone())
    assert np.allclose(filt.init_ensemble.mean(dim=0).numpy(), init.mean(dim=0).numpy(),
                       atol=1e-5)
    assert float(filt.init_ensemble.std()) > 0.1


def test_lagged_init_ensemble_diversity():
    cfg, w = _rc_window(window_days=6.0)
    device = torch.device("cpu")
    init_ensemble, mean_lag_days = _lagged_init_ensemble(cfg, w, N=20,
                                                         init_lag_days=1.5,
                                                         device=device)
    assert mean_lag_days == pytest.approx(1.5, rel=0.05)
    assert init_ensemble.shape == (20, cfg.state_dim)
    assert bool(torch.isfinite(init_ensemble).all())
    assert float(init_ensemble.std()) > 0.0
    q_std = float(w["target_state_q"].std()) + 1e-12
    assert float(init_ensemble.std()) > 0.1 * q_std


def test_etkf_q_cols_lagged_smoke_finite():
    cfg, w = _rc_window(window_days=6.0)
    device = torch.device("cpu")
    dynam = _build_dyn(cfg, w, device)
    obs, r_var, obs_op = _make_obs_system(cfg, w, device, "q", None)[:3]
    init_ensemble, _ = _lagged_init_ensemble(cfg, w, N=20,
                                              init_lag_days=2.0,
                                              device=device)
    filt = ETKF(N_ensemble=20, R_var=r_var, inflation=1.1, device=device,
                dynamics=dynam, obs_operator=obs_op)
    filt.init_ensemble = init_ensemble
    res = filt.assimilate(obs, w["obs_mask"].to(device),
                          w["wind_state_corrupted"].to(device),
                          true_state=w["true_state"])
    assert np.isfinite(res.trajectory).all()
    assert np.isfinite(res.ensemble_variance).all()




def test_psi_obs_run_smoke():
    """End-to-end smoke: run() with obs_var='psi' produces finite results."""
    cfg = QGConfig(nx=8, window_days=6.0, spinup_years=0.05,
                   num_windows=2, obs_geometry="random_columns",
                   cols_per_day=2, seed=3)
    ds = make_qg_s0_s1_datasets(cfg)
    from evaluation.run_qg_baselines import run
    p = run("etkf", cfg, device=torch.device("cpu"), N_ensemble=8,
            inflation=1.0, loc_radius=4.0, scenarios=("test_s0",),
            init="lagged", geometry="random_columns",
            obs_var="psi", init_lag_days=0.5, ds=ds)
    assert "test_s0" in p["scenarios"]
    s0 = p["scenarios"]["test_s0"]
    assert np.isfinite(s0["rmse_mean"])
    assert np.isfinite(s0["expvar_full"])
    assert s0["rmse_mean"] < 1.0  # bounded
