import numpy as np
import pytest
import torch

from data.qg_datasets import QGDatasetSpec
from models.qg_batched import BatchedQGDynamics, SpectralWindForcing
from models.qg_coupled import CoupledSpectralQG, generate_coupled_windows
from models.qg_wind_batched import batched_spectral_wind
from models.qg_wind_modes import FourierWindBasis

NX = 16
L = 1e6
SIGMA = 250000.0
SEEDS = [11, 12, 13]
WIN = dict(amp=[1e-11, 2e-11, 0.0], cx=[0.5, 0.3, 0.6], cy=[0.0, 0.03, -0.02],
           x0=[0.0, 2e5, 4e5], y0=[1e5, 3e5, 5e5], time_unit_days=[40.0, 60.0, 80.0])


def _ocean(r_cf=2.8e-8):
    return BatchedQGDynamics(3, nx=NX, L=L, dt=7200.0, rd=[15000.0, 14000.0, 16000.0],
                             U1=[0.05, 0.045, 0.055], rek=[5.8e-7, 5.5e-7, 6.1e-7], r_cf=r_cf,
                             dtype=torch.float64)


def _coupled(kappa, ocean=None):
    basis = FourierWindBasis(nx=NX, L=L, kmax=2)
    return CoupledSpectralQG(ocean or _ocean(), basis, sigma=SIGMA, kappa_fb=kappa, **WIN)


def _spun_up(ocean, steps=200):
    _, q = ocean.rollout(ocean.initial_q([1, 2, 3]), steps)
    return q


def test_vorticity_read_from_fourier_coefficients_equals_grid_projection():
    model = _coupled(1.0)
    q = _spun_up(model.ocean, 50)
    ph1 = model.ocean._invert(torch.fft.rfft2(q, dim=(-2, -1)))[:, 0]
    zeta = torch.fft.irfft2(-model.ocean.K2 * ph1, s=(NX, NX), dim=(-2, -1))
    torch.testing.assert_close(model.vorticity_amplitudes(ph1), model.basis.project(zeta),
                               rtol=1e-9, atol=1e-18)


def test_zero_coupling_reproduces_the_one_way_path_bit_for_bit():
    n = 40
    model = _coupled(0.0)
    q0 = _spun_up(model.ocean)
    y0 = model.gyro.burn_in(model.gyro.initial_states(SEEDS), 2.0)
    res = model.rollout(q0, y0, n)
    basis = FourierWindBasis(nx=NX, L=L, kmax=2)
    amps = batched_spectral_wind(basis, "gyrostat", SEEDS, n, 7200.0, sigma=SIGMA,
                                 burnin_units=2.0, **WIN)
    one_way, _ = _ocean().rollout(q0, n, forcing=SpectralWindForcing(basis, amps))
    torch.testing.assert_close(res["amplitudes"], amps, rtol=0, atol=0)
    torch.testing.assert_close(res["frames"], one_way, rtol=0, atol=0)


def test_feedback_term_matches_the_bulk_formula():
    model = _coupled(10.0)
    q = _spun_up(model.ocean, 50)
    ph1 = model.ocean._invert(torch.fft.rfft2(q, dim=(-2, -1)))[:, 0]
    t = 3 * 7200.0
    fb = model.feedback_per_unit(ph1, t)
    zeta_lab = model.vorticity_amplitudes(ph1)
    zeta_co = model.basis.translate(zeta_lab, -(model.x0 + model.cx * t), -(model.y0 + model.cy * t))
    b, j = 0, 4
    m = model.order[j]
    expected = (10.0 * model.gamma_a * float(model.r_cf[b]) * float(zeta_co[b, j])
                / float(model.scale[b, j]) * float(model.gyro.std[m]) * float(model.unit_s[b]))
    assert np.isclose(float(fb[b, m]), expected, rtol=1e-10)


def test_calm_window_has_no_forcing_and_no_feedback():
    model = _coupled(100.0)
    q = _spun_up(model.ocean, 20)
    ph1 = model.ocean._invert(torch.fft.rfft2(q, dim=(-2, -1)))[:, 0]
    assert float(model.feedback_per_unit(ph1, 0.0)[2].abs().max()) == 0.0
    y = model.gyro.burn_in(model.gyro.initial_states(SEEDS), 1.0)
    assert float(model.amplitudes(y, 0.0)[2].abs().max()) == 0.0


def test_positive_coupling_changes_the_atmosphere_and_the_ocean():
    n = 60
    q0 = _spun_up(_ocean())
    runs = {}
    for kappa in (0.0, 100.0):
        model = _coupled(kappa)
        y0 = model.gyro.burn_in(model.gyro.initial_states(SEEDS), 2.0)
        runs[kappa] = model.rollout(q0, y0, n)
    assert not torch.equal(runs[0.0]["y_end"][0], runs[100.0]["y_end"][0])
    assert torch.equal(runs[0.0]["y_end"][2], runs[100.0]["y_end"][2])
    assert not torch.equal(runs[0.0]["frames"][0, -1], runs[100.0]["frames"][0, -1])


def test_feedback_grows_linearly_with_kappa():
    q = _spun_up(_ocean(), 50)
    ph1 = _ocean()._invert(torch.fft.rfft2(q, dim=(-2, -1)))[:, 0]
    f1 = _coupled(1.0).feedback_per_unit(ph1, 0.0)
    f10 = _coupled(10.0).feedback_per_unit(ph1, 0.0)
    torch.testing.assert_close(f10, 10.0 * f1, rtol=1e-12, atol=0)


def test_step_limit_is_enforced():
    basis = FourierWindBasis(nx=NX, L=L, kmax=2)
    with pytest.raises(ValueError):
        CoupledSpectralQG(_ocean(), basis, sigma=SIGMA, **{**WIN, "time_unit_days": [1.0, 60.0, 80.0]})


def test_generate_coupled_windows_in_g2_format():
    spec = QGDatasetSpec(name="tiny_c", n_train=3, n_val=2, n_test=2, nx=NX, spinup_days=1.0,
                         lead_days=0.5, window_days=1.0, burnin_units=2.0)
    res = generate_coupled_windows(spec, "train", [0, 1, 2], kappa_fb=10.0, batch_size=2,
                                   diagnose_every=3)
    n = spec.n_lead + spec.n_window
    assert res["true_state"].shape[:2] == (3, n // spec.keep_every_trainval + 1)
    assert res["wind_amplitudes"].shape == (3, n, 12)
    assert torch.isfinite(res["true_state"]).all()
    assert res["feedback_ratio"].shape == (3,)
    calm = res["factors"]["level"] == 0.0
    assert all(float(r) == 0.0 for r, c in zip(res["feedback_ratio"], calm) if c)
