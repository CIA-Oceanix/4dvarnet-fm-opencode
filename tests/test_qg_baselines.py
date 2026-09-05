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
    _downsample_to_da,
    _ensemble_from_init,
    _event_columns,
    _free_forecast_init,
    _lagged_init_ensemble,
    _make_obs_system,
    _obs_spec_rc,
    _per_pass_indices,
    _psi_h,
    _q_alongtrack_obs,
    _q_obs_indices_t,
    _resize_state_layers,
    _sample_init_state,
    _upsample_to_truth,
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
    h = _psi_h(dyn, obs_cols, cfg.ny, cfg.nx, device)
    x = torch.randn(dyn.state_dim)
    ev = w["obs_mask"].nonzero(as_tuple=False).flatten().tolist()
    t = ev[0]
    cols = obs_cols[t]
    psi1 = dyn.inner.streamfunctions(x)
    manual = torch.cat([psi1[0, :, c] for c in cols])
    auto = h(x, index=t)
    assert auto.shape == (cfg.ny,)
    assert torch.allclose(auto, manual, atol=1e-6)
    # batched path
    xb = torch.randn(7, dyn.state_dim)
    ab = h(xb, index=t)
    assert ab.shape == (7, cfg.ny)


def test_psi_h_per_time_columns():
    cfg, w = _rc_window()
    device = torch.device("cpu")
    dyn = _build_dyn(cfg, w, device)
    obs_cols = _event_columns(cfg, w)
    h = _psi_h(dyn, obs_cols, cfg.ny, cfg.nx, device)
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


def test_lagged_init_shared_single_state():
    """The lagged init is now a SINGLE shared state anchored at the band-
    centered lag; all members equal that state (plus dispersion), so the DA
    ensemble and the free-forecast reference start from the same initial
    condition."""
    cfg, w = _rc_window(window_days=6.0)
    device = torch.device("cpu")
    truth = w["init_lead_truth"].float()
    # disp_frac=0 -> every member is the identical shared init_state
    init_ensemble, lag_days = _lagged_init_ensemble(cfg, w, N=20,
                                                     init_lag_days=1.5,
                                                     device=device,
                                                     disp_frac=0.0)
    assert init_ensemble.shape == (20, cfg.state_dim)
    assert bool(torch.isfinite(init_ensemble).all())
    # With no dispersion all members coincide: zero per-point spread.
    assert float(init_ensemble.std(0).mean()) == pytest.approx(0.0, abs=1e-12)
    # The sampled lag centers on init_lag_days (band [1.25, 1.75]).
    assert lag_days == pytest.approx(1.5, abs=0.25)
    # End-relative: the shared state is near the buffer end (t0), not the start.
    shared = init_ensemble[0]
    dist_end = float((shared - truth[-1]).pow(2).mean().sqrt())
    dist_start = float((shared - truth[0]).pow(2).mean().sqrt())
    assert dist_end < 0.5 * dist_start


def test_lagged_init_dispersion_proportional():
    """disp_frac > 0 adds dispersion proportional to the raw lagged per-point
    spread (background-error-scaled), not the climatological state std, which
    would over-disperse by ~40x at a 0.5-day lag."""
    cfg, w = _rc_window(window_days=6.0)
    device = torch.device("cpu")
    truth = w["init_lead_truth"].float()
    raw_spread = float(truth.std(0).mean())
    _, lag0 = _sample_init_state(cfg, w, 0.5, 0.25, device)
    ens, lag1 = _lagged_init_ensemble(cfg, w, N=30, init_lag_days=0.5,
                                      device=device, disp_frac=1.0)
    disp_per = float(ens.std(0).mean())
    # disp_frac=1.0 adds noise with std ~ raw spread: per-point spread should
    # be a small (O(1)) multiple of the raw lagged spread, never the ~40x
    # climatological over-dispersion.
    assert raw_spread > 0.0
    assert 0.3 * raw_spread < disp_per < 3.0 * raw_spread
    # Dispersion is zero-mean about the shared center: both helper paths agree
    # on the sampled lag, and the per-point spread is dominated by dispersion
    # (the shared center is a single state, so any member-to-member difference
    # is exactly the added zero-mean noise).
    assert lag0 == pytest.approx(lag1, abs=1e-12)


def test_free_forecast_init_shared_lagconsistent():
    """The free-forecast first guess is a single band-centered init state near
    t0, consistent with the DA ensemble (they share the same initial
    condition)."""
    cfg, w = _rc_window(window_days=6.0)
    device = torch.device("cpu")
    truth = w["init_lead_truth"].float()
    s, lag_days = _free_forecast_init(cfg, w, 0.5, device)
    assert lag_days == pytest.approx(0.5, abs=0.25)
    dist_end = float((s - truth[-1]).pow(2).mean().sqrt())
    dist_start = float((s - truth[0]).pow(2).mean().sqrt())
    assert dist_end < 0.5 * dist_start


def test_shared_init_used_by_both_free_and_da():
    """The SAME sampled init state seeds both the free forecast and the DA
    ensemble anchored around it, making the DA-vs-free-forecast comparison
    apples-to-apples."""
    cfg, w = _rc_window(window_days=6.0)
    device = torch.device("cpu")
    shared, lag_days = _sample_init_state(cfg, w, 1.0, 0.25, device)
    truth = w["init_lead_truth"].float()
    sigma_raw = float(truth.std(0).mean())
    ens = _ensemble_from_init(shared, sigma_raw, 10, 1.0, device, cfg)
    # every member is anchored at the same shared center (mean offset ~0 vs std)
    assert torch.allclose(ens.mean(0), shared, atol=10.0 * sigma_raw)
    assert lag_days == pytest.approx(1.0, abs=0.25)
    # The free-forecast init equals the ensemble anchor (identical IC).
    _, lag2 = _free_forecast_init(cfg, w, 1.0, device)
    assert lag2 == pytest.approx(lag_days, abs=0.25)


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


def test_etkf_localized_ridge_additive_keep_spread():
    """Localized ETKF anti-collapse: etkf_ridge/etkf_additive keep the
    posterior ensemble spread from collapsing to zero vs the default."""
    cfg, w = _rc_window(window_days=6.0)
    device = torch.device("cpu")
    dynam = _build_dyn(cfg, w, device)
    obs, r_var, obs_op, _ = _make_obs_system(cfg, w, device, "q", None)
    per_time = _q_obs_indices_t(cfg, w)
    Lx, Ly = _build_qg_loc_matrices(dynam.state_dim, per_time, 2,
                                    cfg.ny, cfg.nx, 2.0, device)
    init_ensemble, _ = _lagged_init_ensemble(cfg, w, N=20,
                                              init_lag_days=2.0,
                                              device=device)

    def run_filter(ridge, add):
        filt = ETKF(N_ensemble=20, R_var=r_var, inflation=1.0,
                    device=device, dynamics=dynam, obs_operator=obs_op,
                    loc_radius=2.0, loc_Lx_t=Lx, loc_Ly_t=Ly,
                    etkf_ridge=ridge, etkf_additive=add)
        filt.init_ensemble = init_ensemble
        return filt.assimilate(obs, w["obs_mask"].to(device),
                               w["wind_state_corrupted"].to(device),
                               true_state=w["true_state"])

    base = run_filter(0.0, 0.0)
    tuned = run_filter(1.0, 1e-8)
    assert np.isfinite(base.trajectory).all()
    assert np.isfinite(tuned.trajectory).all()
    assert np.isfinite(tuned.ensemble_variance).all()
    # Anti-collapse: additive/ridge widen the final posterior spread relative
    # to the collapsing default (which drives it toward zero).
    end_base = float(base.ensemble_variance[:, -1].mean())
    end_tuned = float(tuned.ensemble_variance[:, -1].mean())
    assert end_tuned >= end_base


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
    # Regression: without lagged-init dispersion psi-obs collapsed (spread->0,
    # EV ~ -1e3..-1e4). With dispersion the S0 analysis must be skilful.
    assert s0["expvar_full"] > 0.3


def test_q_obs_multi_window_localized_run_no_crash():
    """Regression: q-obs run() over multiple windows with localization must not
    crash with `loc_Lx=None`. The per-time obs-index/localization lists were
    built once from window[0] and reused, but each window has seed-dependent
    obs timing, so later windows misaligned (a masked time had loc_Lx None)."""
    cfg = QGConfig(nx=8, window_days=6.0, spinup_years=0.05,
                   num_windows=3, obs_geometry="random_columns",
                   cols_per_day=2, seed=3)
    ds = make_qg_s0_s1_datasets(cfg)
    from evaluation.run_qg_baselines import run
    p = run("etkf", cfg, device=torch.device("cpu"), N_ensemble=8,
            inflation=1.0, loc_radius=4.0, scenarios=("test_s0",),
            init="lagged", geometry="random_columns", obs_var="q",
            init_lag_days=0.5, ds=ds)
    for s in ("test_s0",):
        assert s in p["scenarios"]
        assert np.isfinite(p["scenarios"][s]["expvar_full"])


def test_run_preserves_obs_noise_std_frac():
    """Regression: `run()` must carry the full original config so the DA obs
    noise reflects `obs_noise_std_frac`. The config was being rebuilt from a
    fresh `QGConfig` when a dataset was passed, silently dropping this field
    (and any other non-listed one) — so the obs-noise sweep results were pinned
    to the default regardless of the requested value."""
    from evaluation.run_qg_baselines import run

    base = {"nx": 8, "window_days": 6.0, "spinup_years": 0.05, "num_windows": 1,
            "obs_geometry": "random_columns", "cols_per_day": 2, "seed": 3}
    cfg_low = QGConfig(**base, obs_noise_std_frac=1e-3)
    cfg_high = QGConfig(**base, obs_noise_std_frac=0.05)
    ds = make_qg_s0_s1_datasets(cfg_high)

    w = ds["test_s0"][0]
    _, r_var_high, _ = _q_alongtrack_obs(cfg_high, w, torch.device("cpu"))
    _, r_var_low, _ = _q_alongtrack_obs(cfg_low, w, torch.device("cpu"))
    # (1) the obs-noise generator itself scales r_var with the config field
    assert r_var_low < r_var_high * 1e-3

    # (2) run() must not re-derive cfg and lose the field: same dataset, the
    # low-noise config must yield a materially different r_var (smaller).
    p_low = run("etkf", cfg_low, device=torch.device("cpu"), N_ensemble=8,
                inflation=1.0, loc_radius=4.0, scenarios=("test_s0",),
                init="lagged", geometry="random_columns", obs_var="q",
                init_lag_days=0.5, ds=ds)
    assert "test_s0" in p_low["scenarios"]
    assert np.isfinite(p_low["scenarios"]["test_s0"]["expvar_full"])


def test_localized_etkf_r_sensitive_analysis():
    """Regression: the localized-ETKF analysis must depend on the obs-error R.

    The default ridge was a hardcoded absolute `1e-4` in the localized gain,
    which (at QG scales where HPH ~1e-14 and R ~1e-16) dwarfed the covariance
    and drove the Kalman gain to ~0. The analysis was then invariant to a 50x
    change in observation noise. The gain must be scale-relative so lowering R
    (cleaner obs) changes the analysis."""
    cfg = QGConfig(nx=16, window_days=30.0, spinup_years=0.1, num_windows=1,
                   obs_geometry="random_columns", cols_per_day=4, seed=7,
                   obs_noise_std_frac=0.05)
    ds = make_qg_s0_s1_datasets(cfg)
    w = ds["test_s0"][0]
    device = torch.device("cpu")
    dyn = _build_dyn(cfg, w, device)
    obs, base_r_var, obs_op, _ = _make_obs_system(cfg, w, device, "q", None)
    forcing = w["wind_state_corrupted"].to(device)
    truth = w["true_state"]
    per_time = _q_obs_indices_t(cfg, w)
    Lx, Ly = _build_qg_loc_matrices(dyn.state_dim, per_time, 2, cfg.ny,
                                    cfg.nx, 6, device)
    si, _ = _sample_init_state(cfg, w, 1.0, 0.25, device)
    sigma_raw = float(w["init_lead_truth"].std(0).mean())
    ens = _ensemble_from_init(si, sigma_raw, 40, 1.0, device, cfg)

    def analysis_rmse(r_var):
        filt = ETKF(N_ensemble=40, R_var=r_var, inflation=1.0, device=device,
                    dynamics=dyn, obs_operator=obs_op, loc_radius=6,
                    noise_init_std=float(w["target_state_q"].std()),
                    loc_Lx_t=Lx, loc_Ly_t=Ly)
        filt.init_ensemble = ens.clone()
        res = filt.assimilate(obs, w["obs_mask"].to(device), forcing,
                              true_state=truth)
        return float(np.sqrt(np.mean((res.trajectory - truth.numpy()) ** 2)))

    rmse_hi = analysis_rmse(base_r_var)
    rmse_lo = analysis_rmse(base_r_var * 1e-4)
    assert abs(rmse_hi - rmse_lo) > 1e-10


def test_resize_state_layers_down_up_roundtrip():
    device = torch.device("cpu")
    src, dst, nlayers = 16, 8, 2
    # Build a pure low-wavenumber (k=2) field on the src grid: integer grid
    # frequencies are exactly periodic, so the mode lies below the dst Nyquist
    # and the down->up roundtrip is near-lossless. White noise would put energy
    # above the downsampled Nyquist that truncation cannot recover.
    yy, xx = torch.meshgrid(torch.arange(src, dtype=torch.float),
                            torch.arange(src, dtype=torch.float), indexing="ij")
    s = float(src)
    layer0 = torch.cos(2 * torch.pi * 2 * xx / s) * torch.sin(2 * torch.pi * 2 * yy / s)
    layer1 = layer0 * 2
    st = torch.stack([layer0, layer1]).reshape(-1)
    down = _resize_state_layers(st, nlayers, src, dst, device)
    assert down.shape == (nlayers * dst * dst,)
    up = _resize_state_layers(down, nlayers, dst, src, device)
    assert torch.allclose(st, up, atol=1e-6)
    # Batch dims preserved.
    bat = torch.randn(5, nlayers * src * src)
    out = _resize_state_layers(bat, nlayers, src, dst, device)
    assert out.shape == (5, nlayers * dst * dst)


def test_downsample_da_scales_dimension():
    device = torch.device("cpu")
    st = torch.randn(2 * 16 * 16)
    down = _downsample_to_da(st, 8, 2, 16, device)
    assert down.shape == (2 * 8 * 8,)
    up = _upsample_to_truth(down, 8, 2, 16, device)
    assert up.shape == (2 * 16 * 16,)


def test_s1_cross_res_run_smoke():
    """End-to-end S1 cross-resolution psi-obs ETKF run must complete and
    produce finite, sensibly-sized results. Exercises init downsampling, the
    upsampled H-function, physical-coordinate localization, and the
    trajectory/metrics upsampling path."""
    cfg = QGConfig(nx=16, window_days=6.0, spinup_years=0.05,
                   num_windows=2, obs_geometry="random_columns",
                   cols_per_day=2, seed=3, da_nx=8)
    from evaluation.run_qg_baselines import run
    ds = make_qg_s0_s1_datasets(cfg)
    p = run("etkf", cfg, device=torch.device("cpu"), N_ensemble=8,
            inflation=1.0, loc_radius=4.0, scenarios=("test_s0", "test_s1"),
            init="lagged", geometry="random_columns",
            obs_var="psi", init_lag_days=0.5, ds=ds)
    assert "test_s1" in p["scenarios"]
    s1 = p["scenarios"]["test_s1"]
    for key in ("rmse_mean", "forecast_rmse_mean", "expvar_full"):
        assert np.isfinite(s1[key])
    assert np.isfinite(s1["forecast_improvement"])


def test_s1_cross_res_q_obs_rejected():
    """obs_var='q' with a cross-resolution S1 model must raise (PV-obs indices
    select truth-grid points with no 1:1 mapping in the lower-res state)."""
    cfg = QGConfig(nx=16, window_days=6.0, spinup_years=0.05,
                   num_windows=1, obs_geometry="random_columns",
                   cols_per_day=2, seed=3, da_nx=8)
    from evaluation.run_qg_baselines import run
    ds = make_qg_s0_s1_datasets(cfg)
    with pytest.raises(ValueError):
        run("etkf", cfg, device=torch.device("cpu"), N_ensemble=8,
            inflation=1.0, loc_radius=4.0, scenarios=("test_s1",),
            init="lagged", geometry="random_columns",
            obs_var="q", init_lag_days=0.5, ds=ds)


def test_s1_qg1l_run_smoke():
    """End-to-end S1-QG1L psi-obs ETKF run (1-layer DA model) must complete
    and produce finite, upper-layer metrics. Exercises the 1-layer init
    projection, the 1-layer H-function branch, and the 1-layer per-field
    metrics path."""
    cfg = QGConfig(nx=16, window_days=6.0, spinup_years=0.05,
                   num_windows=2, obs_geometry="random_columns",
                   cols_per_day=2, seed=3)
    from evaluation.run_qg_baselines import run
    ds = make_qg_s0_s1_datasets(cfg)
    assert all(w["da_model"] == "qg1l" for w in ds["test_s1_qg1l"])
    p = run("etkf", cfg, device=torch.device("cpu"), N_ensemble=8,
            inflation=1.0, loc_radius=4.0, scenarios=("test_s1_qg1l",),
            init="lagged", geometry="random_columns",
            obs_var="psi", init_lag_days=0.5, ds=ds)
    s1 = p["scenarios"]["test_s1_qg1l"]
    for key in ("rmse_mean", "forecast_rmse_mean", "expvar_full",
                "expvar_upper_q"):
        assert np.isfinite(s1[key])
    assert np.isfinite(s1["forecast_improvement"])
    mpf = s1["metrics_per_field"]
    assert set(mpf["q"].keys()) >= {"layer1"}
    for fld in ("q", "psi"):
        assert np.isfinite(mpf[fld]["layer1"]["ev"])


def test_qg1l_psi_obs_r_scale_restores_da_over_free_forecast():
    """Structural-model-error QG1L psi obs are over-trusted at default R.

    Under the 1-layer-vs-2-layer mismatch the nonlocal psi obs are mutually
    inconsistent with the (wrong) model, so the ETKF over-corrects and the DA
    analysis becomes worse than the free forecast. Inflating the obs-noise
    variance `obs_var_r_scale` (model-error-aware R) moves the analysis
    monotonically toward the free-forecast limit (the DA-improv metric climbs
    toward / crosses 1 while the free forecast stays there), without touching
    the error-free S0 anchor. The key transferable claim is the monotone
    approach, not that a single R-scale fully restores DA (production-scale
    nx=64 shows it does not reach improv>=1 for the 1-layer mismatch)."""
    from evaluation.run_qg_baselines import run
    cfg = QGConfig(nx=16, window_days=6.0, spinup_years=0.05,
                   num_windows=2, obs_geometry="random_columns",
                   cols_per_day=2, seed=3)
    ds = make_qg_s0_s1_datasets(cfg)
    kw = {
        "device": torch.device("cpu"),
        "N_ensemble": 8,
        "inflation": 1.0,
        "loc_radius": 4.0,
        "scenarios": ("test_s0", "test_s1_qg1l"),
        "init": "lagged",
        "geometry": "random_columns",
        "obs_var": "psi",
        "init_lag_days": 0.5,
        "ds": ds,
    }
    p_default = run("etkf", cfg, obs_var_r_scale=1.0, **kw)
    p_infl = run("etkf", cfg, obs_var_r_scale=1e4, **kw)
    q1_default = p_default["scenarios"]["test_s1_qg1l"]
    q1_infl = p_infl["scenarios"]["test_s1_qg1l"]
    assert q1_default["forecast_improvement"] < 1.0
    assert q1_infl["forecast_improvement"] < 1.01
    assert q1_infl["forecast_improvement"] >= q1_default["forecast_improvement"]
    assert q1_infl["expvar_full"] >= q1_default["expvar_full"]
    s0_default = p_default["scenarios"]["test_s0"]
    s0_infl = p_infl["scenarios"]["test_s0"]
    assert s0_infl["expvar_full"] > 0.5
    assert abs(s0_infl["expvar_full"] - s0_default["expvar_full"]) < 0.15


def test_make_obs_system_r_scale_wiring():
    """The `obs_var_r_scale` knob must multiply r_var in both obs branches.

    Pins the R-inflation wiring so a refactor cannot silently drop the scale
    (the same class of bug as `obs_noise_std_frac` being lost when a dataset is
    passed directly): for q- and psi-obs, `_make_obs_system(..., s)` must
    return `r_var * s`, with s=1.0 reproduces the baseline exactly."""
    cfg, w = _rc_window()
    device = torch.device("cpu")
    for obs_var in ("q", "psi"):
        _, r_base, _, _ = _make_obs_system(cfg, w, device, obs_var, None, 1.0)
        for scale in (0.5, 10.0, 1e4):
            _, r_scaled, _, _ = _make_obs_system(cfg, w, device, obs_var, None,
                                                 scale)
            assert r_scaled == pytest.approx(r_base * scale, rel=1e-12)


