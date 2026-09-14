import numpy as np
import pytest
import torch

from data.qg import QGConfig, make_qg_s0_s1_datasets
from models.qg1l_dynamics import QG1LDynamics
from models.qg_dynamics import QGDynamics
from models.qg_psi_dynamics import QG1LPsiDynamics, QGPsiDynamics, wrap_psi


def _inner(inner_cls, nx=16, **kw):
    return inner_cls(nx=nx, wind_amp=0.0, **kw)


@pytest.mark.parametrize(
    "inner_cls,wrap_cls,nlayers",
    [(QGDynamics, QGPsiDynamics, 2), (QG1LDynamics, QG1LPsiDynamics, 1)],
)
def test_forward_pv_roundtrip(inner_cls, wrap_cls, nlayers):
    inner = _inner(inner_cls)
    psi = wrap_cls(inner)
    q0 = inner._initial_q(1, seed=0, device=torch.device("cpu"))
    qf = q0.reshape(-1)
    psif = psi.q_to_psi(qf)
    q_back = psi.psi_to_q(psif)
    rel = float((q_back - qf).abs().max() / qf.abs().max())
    assert rel < 1e-4, f"roundtrip rel err {rel:.2e}"


@pytest.mark.parametrize(
    "inner_cls,wrap_cls,nsteps",
    [(QGDynamics, QGPsiDynamics, 40), (QG1LDynamics, QG1LPsiDynamics, 40)],
)
def test_psi_state_free_forecast_matches_q_state(inner_cls, wrap_cls, nsteps):
    """Phase 1: a free forecast from the psi-state model reproduces the
    q-state model's forecast (in q space) up to the round-trip roundoff."""
    inner = _inner(inner_cls)
    psi = wrap_cls(inner)
    q0 = inner._initial_q(1, seed=0, device=torch.device("cpu"))
    qf = q0.reshape(-1)
    q_roll = inner.rollout_trajectory(qf, nsteps)          # (N+1, D) q
    psi_roll = psi.rollout_trajectory(psi.q_to_psi(qf), nsteps)  # (N+1, D) psi
    q_from_psi = psi.psi_to_q(psi_roll)                     # (N+1, D) q
    rel = float((q_roll - q_from_psi).abs().max() / qf.abs().max())
    assert rel < 1e-3, f"free-forecast rel err {rel:.2e}"


def test_wrap_psi_dispatches():
    assert isinstance(wrap_psi(_inner(QGDynamics), "qg2l"), QGPsiDynamics)
    assert isinstance(wrap_psi(_inner(QG1LDynamics), "qg1l"), QG1LPsiDynamics)


def _cfg(nx=8, num_windows=2, geometry="random_columns", da_nx=None, seed=3):
    return QGConfig(nx=nx, window_days=6.0, spinup_years=0.05,
                    num_windows=num_windows, obs_geometry=geometry,
                    cols_per_day=2, seed=seed, da_nx=da_nx)


def test_psi_state_etkf_s0_finite():
    """Phase 2 smoke: psi-state ETKF on S0 is finite and skilful."""
    from evaluation.run_qg_baselines import run
    cfg = _cfg()
    ds = make_qg_s0_s1_datasets(cfg)
    p = run("etkf", cfg, device=torch.device("cpu"), N_ensemble=8,
            inflation=1.0, loc_radius=4.0, scenarios=("test_s0",),
            init="lagged", geometry="random_columns",
            obs_var="psi_state", init_lag_days=0.5, ds=ds)
    s0 = p["scenarios"]["test_s0"]
    assert np.isfinite(s0["rmse_mean"])
    assert np.isfinite(s0["expvar_full"])
    assert s0["expvar_full"] > 0.3


def test_psi_state_etkf_matches_legacy_psi_obs():
    """Phase 2: psi-state ETKF and the legacy psi-obs (H-function) ETKF give
    comparable DA skill on the same S0 windows."""
    from evaluation.run_qg_baselines import run
    cfg = _cfg()
    ds = make_qg_s0_s1_datasets(cfg)
    evs = {}
    for ov in ("psi", "psi_state"):
        p = run("etkf", cfg, device=torch.device("cpu"), N_ensemble=8,
                inflation=1.0, loc_radius=4.0, scenarios=("test_s0",),
                init="lagged", geometry="random_columns",
                obs_var=ov, init_lag_days=0.5, ds=ds)
        evs[ov] = p["scenarios"]["test_s0"]["expvar_full"]
        assert np.isfinite(evs[ov])
    # Similar (not identical) skill; both beat a degenerate filter.
    assert abs(evs["psi"] - evs["psi_state"]) < 0.2
    assert evs["psi_state"] > 0.3


def test_s1_qg1l_psi_state_finite():
    """1-layer QG1L structural-error scenario with psi_state is finite."""
    from evaluation.run_qg_baselines import run
    cfg = _cfg()
    ds = make_qg_s0_s1_datasets(cfg)
    p = run("etkf", cfg, device=torch.device("cpu"), N_ensemble=8,
            inflation=1.0, loc_radius=4.0, scenarios=("test_s1_qg1l",),
            init="lagged", geometry="random_columns",
            obs_var="psi_state", init_lag_days=0.5, ds=ds)
    s = p["scenarios"]["test_s1_qg1l"]
    assert np.isfinite(s["rmse_mean"])
    assert np.isfinite(s["expvar_full"])


def test_s1_cross_res_psi_state_finite():
    """Cross-resolution psi_state (S1, da_nx < nx) is finite and skilful now
    that it uses the H-mode obs operator (`_psi_h` spectrally resamples the
    DA-model psi-state to the obs grid)."""
    from evaluation.run_qg_baselines import run
    cfg = _cfg(nx=8, da_nx=4)
    ds = make_qg_s0_s1_datasets(cfg)
    p = run("etkf", cfg, device=torch.device("cpu"), N_ensemble=8,
            inflation=1.0, loc_radius=4.0, scenarios=("test_s1",),
            init="lagged", geometry="random_columns",
            obs_var="psi_state", init_lag_days=0.5, ds=ds)
    s = p["scenarios"]["test_s1"]
    assert np.isfinite(s["rmse_mean"])
    assert np.isfinite(s["expvar_full"])
    # Analyses are produced (upsampled to the truth grid) and the free
    # forecast is computed via the psi->q cross-res path.
    assert np.isfinite(s["forecast_rmse_mean"])


def test_psi_state_cross_res_obs_op_matches_manual_h():
    """The H-mode psi_state obs operator is exactly the psi-state
    streamfunction (identity reshape) spectrally upsampled to the obs grid and
    column-selected -- reproducing a manual recomputation on the same model."""
    from evaluation.run_qg_baselines import _build_dyn, _event_columns, _make_obs_system
    from models.qg_interp import spectral_resize_2d
    cfg = _cfg(nx=8, da_nx=4)
    ds = make_qg_s0_s1_datasets(cfg)
    w = ds["test_s1"][0]
    device = torch.device("cpu")
    dyn = _build_dyn(cfg, w, device, psi_state=True)
    obs_op = _make_obs_system(cfg, w, device, "psi_state",
                              loc_radius=4.0)[2]
    obs_cols = _event_columns(cfg, w)
    assert obs_op.h_mode
    psi_state = torch.randn(dyn.state_dim, dtype=torch.float64)
    psi = dyn.inner.streamfunctions(psi_state)          # (2, da_nx, da_nx)
    psi_r = spectral_resize_2d(psi, cfg.ny, cfg.nx)     # (2, ny, nx)
    psi1 = psi_r[0]                                     # upper layer (ny, nx)
    for t in w["obs_mask"].nonzero(as_tuple=False).flatten().tolist():
        cols = obs_cols[t]
        if cols is None:
            continue
        out = obs_op(psi_state, index=t)
        want = torch.cat([psi1[:, c] for c in cols])
        assert torch.allclose(out, want, atol=1e-9), (
            f"t={t} psi_state H vs manual mismatch")
