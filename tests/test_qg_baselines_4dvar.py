import numpy as np
import torch

from data.qg import QGConfig, make_qg_s0_s1_datasets
from evaluation.run_qg_baselines import (
    QG4DVar,
    _build_dyn,
    _evaluate_window,
    _make_obs_system,
    _sample_init_state,
)


def _cfg():
    return QGConfig(nx=8, window_days=6.0, spinup_years=0.05,
                    num_windows=1, obs_geometry="random_columns",
                    cols_per_day=2, seed=3)


def test_strong4dvar_psi_run_smoke():
    cfg = _cfg()
    ds = make_qg_s0_s1_datasets(cfg)
    from evaluation.run_qg_baselines import run
    p = run("strong4dvar", cfg, device=torch.device("cpu"),
            scenarios=("test_s0",), init="lagged", init_lag_days=0.5,
            geometry="random_columns", obs_var="psi", band_half=0.25,
            da_window_steps=12, optimizer="adam", fourdvar_opt_steps=40,
            fourdvar_lr=0.05, b_var_scale=1.0, ds=ds)
    s0 = p["scenarios"]["test_s0"]
    assert np.isfinite(s0["rmse_mean"])
    assert np.isfinite(s0["expvar_full"])
    # On the tiny error-free S0 window the solve must be skilful, not NaN.
    assert s0["expvar_full"] > 0.5
    assert s0["rmse_mean"] < 1.0


def test_weak4dvar_psi_run_smoke():
    cfg = _cfg()
    ds = make_qg_s0_s1_datasets(cfg)
    from evaluation.run_qg_baselines import run
    p = run("weak4dvar", cfg, device=torch.device("cpu"),
            scenarios=("test_s0",), init="lagged", init_lag_days=0.5,
            geometry="random_columns", obs_var="psi", band_half=0.25,
            da_window_steps=12, optimizer="adam", fourdvar_opt_steps=40,
            fourdvar_lr=0.05, b_var_scale=1.0, q_var_scale=1.0, ds=ds)
    s0 = p["scenarios"]["test_s0"]
    assert np.isfinite(s0["rmse_mean"])
    assert np.isfinite(s0["expvar_full"])
    assert s0["rmse_mean"] < 1.0


def test_qg4dvar_psi_h_uses_absolute_index():
    """The psi H-mode obs operator is called with the absolute time index."""
    cfg = _cfg()
    ds = make_qg_s0_s1_datasets(cfg)
    w = ds["test_s0"][0]
    device = torch.device("cpu")
    dyn = _build_dyn(cfg, w, device)
    obs, r_var, obs_op, _ = _make_obs_system(cfg, w, device, "psi", None, 1.0)
    x0, _lag = _sample_init_state(cfg, w, 0.5, 0.25, device)
    m = QG4DVar(cfg, dyn, obs_op, da_window_steps=12, b_var_scale=1.0,
                optimizer="adam", opt_steps=40, lr=0.05, mode="strong",
                device=device)
    m.x0_bg = x0
    m.r_var = float(r_var)
    res = _evaluate_window(cfg, w, m, device, obs=obs,
                           forcing=w["wind_state_corrupted"].to(device),
                           init_ensemble=None)
    traj = res.trajectory
    assert traj.shape == (cfg.num_steps, dyn.state_dim)
    assert np.isfinite(traj).all()
    # The absolute-index H call must reproduce the per-time psi columns.
    # Since PR #156 (per-column independent intra-day timing,
    # OBS_GEOMETRY_VERSION=2), only one column is observed per event, so the
    # mapped obs dimension is ny, not cols_per_day * ny (the old
    # constellation-style simultaneous-columns geometry this assertion
    # originally assumed).
    assert obs_op.obs_dim == cfg.ny


def test_qg4dvar_q_index_mode_weak():
    cfg = _cfg()
    ds = make_qg_s0_s1_datasets(cfg)
    w = ds["test_s0"][0]
    device = torch.device("cpu")
    dyn = _build_dyn(cfg, w, device)
    obs, r_var, obs_op, _ = _make_obs_system(cfg, w, device, "q", None, 1.0)
    x0, _lag = _sample_init_state(cfg, w, 0.5, 0.25, device)
    m = QG4DVar(cfg, dyn, obs_op, da_window_steps=12, b_var_scale=1.0,
                q_var_scale=1.0, optimizer="adam", opt_steps=40, lr=0.05,
                mode="weak", device=device)
    m.x0_bg = x0
    m.r_var = float(r_var)
    res = _evaluate_window(cfg, w, m, device, obs=obs,
                           forcing=w["wind_state_corrupted"].to(device),
                           init_ensemble=None)
    traj = res.trajectory
    assert traj.shape == (cfg.num_steps, dyn.state_dim)
    assert np.isfinite(traj).all()
