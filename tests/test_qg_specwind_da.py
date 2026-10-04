import os
import shutil

import pytest
import torch

from data.qg_datasets import QGDatasetSpec, assemble_split, split_dir, write_shard
from evaluation.baselines import _build_qg_col_loc_matrices
from evaluation.run_qg_baselines import (
    _bred_ensemble,
    _build_dyn,
    _build_qg_col_point_loc_matrices,
    _ensemble_from_init,
    _sample_init_state,
    run,
)
from evaluation.run_qg_specwind_da import build_cfg, s0_windows
from models.qg_batched import BatchedQGDynamics, SpectralWindForcing
from models.qg_dynamics import QGDynamics
from models.qg_wind_modes import FourierWindBasis

TINY = QGDatasetSpec(name="tiny_da", n_train=3, n_val=3, n_test=3, nx=16, spinup_days=1.0,
                     lead_days=0.5, window_days=1.0, burnin_units=2.0,
                     keep_every_trainval=6)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("da1"))
    for split in ("val", "test"):
        write_shard(TINY, split, root, 0, 1, batch_size=3)
        assemble_split(TINY, split, root)
    yield root
    d = split_dir(root, TINY, "test")
    for f in os.listdir(d):
        os.chmod(os.path.join(d, f), 0o644)
    shutil.rmtree(root, ignore_errors=True)


def test_spectral_qgdynamics_matches_batched_dynamics_with_eddy_drag():
    basis = FourierWindBasis(nx=16, kmax=2)
    a = torch.randn(10, basis.n_amp, dtype=torch.float64) * 2e-12
    ref = BatchedQGDynamics(1, nx=16, rd=14000.0, U1=0.05, rek=6e-7, r_cf=2.8e-8, dtype=torch.float64)
    q0 = ref.rollout(ref.initial_q([3]), 100)[1]
    batched, _ = ref.rollout(q0, 10, forcing=SpectralWindForcing(basis, a.unsqueeze(0)))
    dyn = QGDynamics(nx=16, rd=14000.0, U1=0.05, rek=6e-7, r_cf=2.8e-8, wind_driver="gyrostat",
                     dtype=torch.float64)
    single = dyn.rollout_trajectory(dyn._flatten(q0[0]), 10, wind_state=a)
    torch.testing.assert_close(single, ref._flatten(batched[0]), rtol=1e-10, atol=1e-20)
    assert dyn.wind_state_dim == 12


def test_default_qgdynamics_is_unchanged_by_the_hook():
    dyn = QGDynamics(nx=16, wind_amp=1e-11)
    assert dyn.wind_state_dim == 3 and dyn.r_cf == 0.0
    with pytest.raises(NotImplementedError):
        QGDynamics(nx=16, wind_driver="gyrostat").generate_wind_state(3)
    with pytest.raises(ValueError):
        QGDynamics(nx=16, wind_driver="storm")


def test_s0_windows_have_the_scenario_fields_and_index_seeded_obs(built):
    cfg = build_cfg(TINY, cols_per_day=2, obs_noise_std_frac=0.05, init_lag_days=0.25)
    many, _ = s0_windows(TINY, "test", built, [0, 1, 2], cfg)
    one, _ = s0_windows(TINY, "test", built, [2], cfg)
    torch.testing.assert_close(many[2]["obs"], one[0]["obs"], equal_nan=True)
    w = many[0]
    assert w["da_model"] == "qg2l" and w["da_params"] == w["true_params"]
    assert w["wind_state_corrupted"].shape == (TINY.n_window, 12)
    assert w["target_state_psi"].shape == (TINY.n_window, 16 * 16)
    dyn = _build_dyn(cfg, w, torch.device("cpu"))
    assert dyn.inner.wind_state_dim == 12 and dyn.inner.r_cf == TINY.r_cf


def test_val_windows_are_regenerated_and_checked(built):
    cfg = build_cfg(TINY, cols_per_day=2, obs_noise_std_frac=0.05, init_lag_days=0.25)
    ws, rep = s0_windows(TINY, "val", built, [1], cfg)
    assert rep["regenerated"] and rep["max_rel_diff"] <= 1e-4
    assert ws[0]["true_state"].shape[0] == TINY.n_window


def test_etkf_runs_end_to_end_on_s0_windows(built, tmp_path):
    # lag band [0.1, 0.42] d keeps the sampled lag above one model step: _sample_init_state
    # indexes out of range for a lag below one step (kk = 0), unreachable at the 5-day lag.
    cfg = build_cfg(TINY, cols_per_day=2, obs_noise_std_frac=0.05, init_lag_days=0.35)
    ws, _ = s0_windows(TINY, "test", built, [0, 1], cfg)
    payload = run("etkf", cfg, device=torch.device("cpu"), N_ensemble=6, inflation=1.0,
                  loc_radius=2.0, scenarios=("test_s0",), out_path=str(tmp_path / "s.json"),
                  init="lagged", geometry="random_columns", obs_var="psi", init_lag_days=0.35,
                  ds={"test_s0": ws}, etkf_ridge=0.1)
    s = payload["scenarios"]["test_s0"]
    assert s["expvar_full"] == s["expvar_full"]
    assert (tmp_path / "s.json").exists()


def test_cross_layer_localization_scales_the_horizontal_weight():
    cols_t = [None, [3], [5, 9]]
    base_x, base_y = _build_qg_col_loc_matrices(2 * 16 * 16, cols_t, 2, 16, 16, 2.0, "cpu")
    zero_x, _ = _build_qg_col_loc_matrices(2 * 16 * 16, cols_t, 2, 16, 16, 2.0, "cpu", cross_layer=0.0)
    lx, ly = _build_qg_col_loc_matrices(2 * 16 * 16, cols_t, 2, 16, 16, 2.0, "cpu", cross_layer=0.4)
    assert zero_x[0] is None and torch.equal(zero_x[2], base_x[2])
    assert torch.count_nonzero(base_x[2][256:]) == 0
    torch.testing.assert_close(lx[2][:256], base_x[2][:256])
    torch.testing.assert_close(lx[2][256:], 0.4 * base_x[2][:256])
    torch.testing.assert_close(ly[2], base_y[2])
    px, py = _build_qg_col_point_loc_matrices(2 * 16 * 16, [[4]], [(2, 7)], 2, 16, 16, 2.0, "cpu",
                                              cross_layer=0.4)
    assert px[0][256 + 7 * 16 + 2, 16] == pytest.approx(1.0)
    assert px[0][7 * 16 + 2, 16] == pytest.approx(0.4)
    assert float(py[0][7, 16]) == pytest.approx(0.4 * float(py[0][7, 5]))


def test_init_seed_key_gives_each_window_its_own_draw(built):
    cfg = build_cfg(TINY, cols_per_day=2, obs_noise_std_frac=0.05, init_lag_days=0.3)
    w = s0_windows(TINY, "test", built, [0], cfg)[0][0]
    legacy, lag_legacy = _sample_init_state(cfg, w, 0.3, 0.1, "cpu")
    again, _ = _sample_init_state(cfg, w, 0.3, 0.1, "cpu", seed_key=None)
    lags = {_sample_init_state(cfg, w, 0.3, 0.1, "cpu", seed_key=k)[1] for k in range(4)}
    assert torch.equal(legacy, again) and len(lags) == 4
    e0 = _ensemble_from_init(legacy, 1.0, 5, 1.0, "cpu", cfg, seed_key=0)
    e1 = _ensemble_from_init(legacy, 1.0, 5, 1.0, "cpu", cfg, seed_key=1)
    assert not torch.equal(e0, e1)
    torch.testing.assert_close(_ensemble_from_init(legacy, 1.0, 5, 1.0, "cpu", cfg),
                               _ensemble_from_init(legacy, 1.0, 5, 1.0, "cpu", cfg, seed_key=None))


def test_bred_ensemble_is_centred_on_the_init_state_with_white_noise_amplitude(built):
    cfg = build_cfg(TINY, cols_per_day=2, obs_noise_std_frac=0.05, init_lag_days=0.2)
    w = s0_windows(TINY, "test", built, [1], cfg)[0][0]
    assert w["wind_lead"].shape == (w["init_lead_truth"].shape[0], 12)
    dyn = _build_dyn(cfg, w, torch.device("cpu"))
    init, lag = _sample_init_state(cfg, w, 0.2, 0.05, "cpu", seed_key=1)
    sigma = float(w["init_lead_truth"].std(0).mean())
    ens = _bred_ensemble(cfg, dyn, w, init, lag, sigma, 8, 0.5, 2 / 12, "cpu", seed_key=1)
    assert ens.shape == (8, init.numel())
    torch.testing.assert_close(ens.mean(0), init, rtol=1e-4, atol=1e-6 * sigma)
    assert float((ens - init).std(0).mean()) == pytest.approx(0.5 * sigma, rel=1e-4)
    with pytest.raises(ValueError):
        _bred_ensemble(cfg, dyn, w, init, lag, sigma, 8, 0.5, 5.0, "cpu")


def test_etkf_runs_with_bred_init_and_vertical_localization(built, tmp_path):
    cfg = build_cfg(TINY, cols_per_day=2, obs_noise_std_frac=0.05, init_lag_days=0.2)
    ws, _ = s0_windows(TINY, "test", built, [0, 1], cfg)
    payload = run("etkf", cfg, device=torch.device("cpu"), N_ensemble=6, inflation=1.0,
                  loc_radius=2.0, scenarios=("test_s0",), out_path=str(tmp_path / "s.json"),
                  init="lagged", geometry="random_columns", obs_var="psi", init_lag_days=0.2,
                  band_half=0.05, ds={"test_s0": ws}, etkf_ridge=0.1, loc_cross_layer=0.5,
                  init_ensemble_kind="bred", breed_days=2 / 12)
    s = payload["scenarios"]["test_s0"]
    assert s["expvar_full"] == s["expvar_full"]
    with pytest.raises(NotImplementedError):
        run("etkf", cfg, device=torch.device("cpu"), N_ensemble=6, loc_radius=2.0,
            scenarios=("test_s0",), obs_var="q", ds={"test_s0": ws}, loc_cross_layer=0.5)


def test_localized_enks_runs_end_to_end_and_smooths_the_filter(built, tmp_path):
    cfg = build_cfg(TINY, cols_per_day=2, obs_noise_std_frac=0.05, init_lag_days=0.2)
    ws, _ = s0_windows(TINY, "test", built, [0, 1], cfg)
    kw = dict(device=torch.device("cpu"), N_ensemble=6, inflation=1.0, loc_radius=2.0,
              scenarios=("test_s0",), init="lagged", geometry="random_columns", obs_var="psi",
              init_lag_days=0.2, band_half=0.05, ds={"test_s0": ws}, etkf_ridge=0.1,
              loc_cross_layer=1.0, init_ensemble_kind="bred", breed_days=2 / 12,
              etkf_loc_mode="ensrf")
    filt = run("etkf", cfg, **kw)["scenarios"]["test_s0"]
    smooth = run("enks", cfg, enks_lag=2, **kw)["scenarios"]["test_s0"]
    assert smooth["expvar_full"] == smooth["expvar_full"]
    assert smooth["rmse_list"] != filt["rmse_list"]
    assert len(smooth["spread_ratio_list"]) == 2


@pytest.mark.parametrize("method,relax", [("etkf", "rtps"), ("etkf", "rtpp"), ("enkf", "rtps")])
def test_relaxation_inflation_runs_end_to_end_and_widens_the_spread(built, method, relax):
    cfg = build_cfg(TINY, cols_per_day=2, obs_noise_std_frac=0.05, init_lag_days=0.2)
    ws, _ = s0_windows(TINY, "test", built, [0, 1], cfg)
    kw = dict(device=torch.device("cpu"), N_ensemble=6, inflation=1.0, loc_radius=2.0,
              scenarios=("test_s0",), init="lagged", geometry="random_columns", obs_var="psi",
              init_lag_days=0.2, band_half=0.05, ds={"test_s0": ws}, etkf_ridge=0.1,
              loc_cross_layer=1.0, init_ensemble_kind="bred", breed_days=2 / 12,
              etkf_loc_mode="ensrf")
    torch.manual_seed(0)
    base = run(method, cfg, **kw)["scenarios"]["test_s0"]
    torch.manual_seed(0)
    relaxed = run(method, cfg, relax=relax, relax_alpha=0.9, **kw)["scenarios"]["test_s0"]
    assert relaxed["expvar_full"] == relaxed["expvar_full"]
    spread = [sum(w["q1"] for w in r["spread_ratio_list"]) for r in (base, relaxed)]
    assert spread[1] > spread[0]


@pytest.mark.parametrize("method", ["strong4dvar", "weak4dvar"])
def test_4dvar_runs_end_to_end_on_spectral_wind_windows(built, method):
    cfg = build_cfg(TINY, cols_per_day=2, obs_noise_std_frac=0.05, init_lag_days=0.2)
    ws, _ = s0_windows(TINY, "test", built, [0, 1], cfg)
    payload = run(method, cfg, device=torch.device("cpu"), N_ensemble=4, inflation=1.0,
                  scenarios=("test_s0",), init="lagged", geometry="random_columns", obs_var="psi",
                  init_lag_days=0.2, band_half=0.05, ds={"test_s0": ws}, da_window_steps=6,
                  optimizer="lbfgs", fourdvar_max_iter=5, fourdvar_lr=1.0, fourdvar_grad_clip=1000.0,
                  b_var_scale=1.0, q_var_scale=0.1)
    s = payload["scenarios"]["test_s0"]
    assert s["expvar_full"] == s["expvar_full"] and s["crps_is_deterministic"]


def test_spectral_b_sqrt_reproduces_the_climatological_covariance():
    from evaluation.run_qg_baselines import spectral_b_sqrt
    g = torch.Generator().manual_seed(0)
    ny = nx = 16
    base = torch.randn(400, ny, nx, generator=g, dtype=torch.float64)
    kk = torch.sqrt((torch.fft.fftfreq(ny) * ny)[:, None] ** 2 + (torch.fft.rfftfreq(nx) * nx)[None, :] ** 2)
    smooth = torch.fft.irfft2(torch.fft.rfft2(base) * torch.exp(-kk / 3.0), s=(ny, nx))
    lead = torch.stack([smooth, 0.6 * smooth + 0.3 * torch.randn(400, ny, nx, generator=g,
                                                                  dtype=torch.float64)], 1)
    lead = lead.reshape(400, -1)
    b = spectral_b_sqrt(lead, 2, ny, nx, "cpu")
    x = b(torch.randn(4000, 2 * ny * nx, generator=g, dtype=torch.float64)).reshape(4000, 2, ny, nx)
    clim = (lead - lead.mean(0)).reshape(400, 2, ny, nx)
    for li in range(2):
        assert float(x[:, li].var()) == pytest.approx(float(clim[:, li].var()), rel=0.1)
    corr = lambda f: float(torch.corrcoef(torch.stack([f[:, 0].flatten(), f[:, 1].flatten()]))[0, 1])  # noqa: E731
    assert corr(x) == pytest.approx(corr(clim), abs=0.05)
    assert abs(float(x.mean((-2, -1)).abs().max())) < 1e-6


def test_spectral_b_4dvar_runs_end_to_end(built):
    cfg = build_cfg(TINY, cols_per_day=2, obs_noise_std_frac=0.05, init_lag_days=0.2)
    ws, _ = s0_windows(TINY, "test", built, [0, 1], cfg)
    for method in ("strong4dvar", "weak4dvar"):
        payload = run(method, cfg, device=torch.device("cpu"), N_ensemble=4, inflation=1.0,
                      scenarios=("test_s0",), init="lagged", geometry="random_columns", obs_var="psi",
                      init_lag_days=0.2, band_half=0.05, ds={"test_s0": ws}, da_window_steps=6,
                      optimizer="lbfgs", fourdvar_max_iter=5, fourdvar_lr=1.0, fourdvar_grad_clip=1000.0,
                      b_var_scale=1.0, q_var_scale=0.1, fourdvar_b="spectral")
        s = payload["scenarios"]["test_s0"]
        assert s["expvar_full"] == s["expvar_full"]


def test_4dvar_rejects_an_optimization_that_increases_the_cost(built, monkeypatch):
    from evaluation.run_qg_baselines import QG4DVar

    def bad_optimize(self, loss_fn, params, bg):
        with torch.no_grad():
            for p in params:
                p.fill_(50.0)

    monkeypatch.setattr(QG4DVar, "_optimize", bad_optimize)
    cfg = build_cfg(TINY, cols_per_day=2, obs_noise_std_frac=0.05, init_lag_days=0.2)
    ws, _ = s0_windows(TINY, "test", built, [0], cfg)
    payload = run("strong4dvar", cfg, device=torch.device("cpu"), N_ensemble=4, inflation=1.0,
                  scenarios=("test_s0",), init="lagged", geometry="random_columns", obs_var="psi",
                  init_lag_days=0.2, band_half=0.05, ds={"test_s0": ws}, da_window_steps=6,
                  b_var_scale=1.0, fourdvar_b="spectral")
    s = payload["scenarios"]["test_s0"]
    assert s["fallback_list"][0] >= 1 and s["expvar_full"] == s["expvar_full"]
