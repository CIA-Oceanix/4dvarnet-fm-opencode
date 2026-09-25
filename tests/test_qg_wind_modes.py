import math

import numpy as np
import pytest
import torch

from models.qg_dynamics import QGDynamics
from models.qg_wind_modes import (
    FourierWindBasis,
    generate_spectral_wind,
)

L = 1e6
SIGMA = 250000.0


@pytest.fixture(scope="module")
def basis():
    return FourierWindBasis(nx=64, L=L, kmax=2)


def test_disk_kmax2_has_six_wavevectors(basis):
    assert basis.wavevectors == [(1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2)]
    assert basis.n_amp == 12


def test_square_kmax1_has_four_wavevectors():
    b = FourierWindBasis(nx=32, L=L, kmax=1, shape="square")
    assert sorted(b.wavevectors) == [(0, 1), (1, -1), (1, 0), (1, 1)]


def test_spectral_matches_rfft_of_field(basis):
    a = torch.randn(5, basis.n_amp, dtype=torch.float64)
    np.testing.assert_allclose(
        basis.curl_spectral(a).numpy(),
        torch.fft.rfft2(basis.curl_field(a), dim=(-2, -1)).numpy(),
        rtol=1e-10, atol=1e-8)


def test_project_inverts_field(basis):
    a = torch.randn(7, basis.n_amp, dtype=torch.float64)
    torch.testing.assert_close(basis.project(basis.curl_field(a)), a,
                               rtol=1e-10, atol=1e-10)


def test_translate_equals_grid_roll(basis):
    a = torch.randn(basis.n_amp, dtype=torch.float64)
    dx_pts, dy_pts = 5, -3
    dx, dy = dx_pts * L / 64, dy_pts * L / 64
    shifted = basis.curl_field(basis.translate(a, dx, dy))
    rolled = torch.roll(basis.curl_field(a), shifts=(dy_pts, dx_pts), dims=(-2, -1))
    torch.testing.assert_close(shifted, rolled, rtol=1e-10, atol=1e-10)


def test_current_storm_lives_in_kmax2_basis(basis):
    dyn = QGDynamics(nx=64, L=L, wind_amp=1e-11, wind_sigma=SIGMA, dtype=torch.float64)
    rng = np.random.default_rng(0)
    ws = torch.tensor([[1.0, rng.uniform(0, L), rng.uniform(0, L)] for _ in range(32)],
                      dtype=torch.float64)
    field = dyn.wind_curl_field(ws)
    recon = basis.curl_field(basis.project(field))
    kept = float((recon ** 2).sum() / (field ** 2).sum())
    assert kept > 0.995


def test_analytic_ricker_amplitudes_match_projection(basis):
    dyn = QGDynamics(nx=64, L=L, wind_amp=1e-11, wind_sigma=SIGMA, dtype=torch.float64)
    xc = torch.tensor([123456.0, 700000.0, 42000.0], dtype=torch.float64)
    yc = torch.tensor([876543.0, 310000.0, 999000.0], dtype=torch.float64)
    A = torch.tensor([1.0, -0.5, 2.0], dtype=torch.float64)
    ws = torch.stack([A, xc, yc], dim=-1)
    projected = basis.project(dyn.wind_curl_field(ws))
    analytic = basis.ricker_amplitudes(A, xc, yc, SIGMA)
    torch.testing.assert_close(projected, analytic, rtol=2e-3, atol=2e-3 * float(analytic.abs().max()))


def test_ricker_mode_std_is_position_average(basis):
    n = 4096
    rng = np.random.default_rng(1)
    xc = torch.from_numpy(rng.uniform(0, L, n))
    yc = torch.from_numpy(rng.uniform(0, L, n))
    a = basis.ricker_amplitudes(torch.ones(n, dtype=torch.float64), xc, yc, SIGMA)
    np.testing.assert_allclose(a.std(0).numpy(), basis.ricker_mode_std(SIGMA).numpy(), rtol=0.05)


@pytest.mark.parametrize("driver", ["spectral_ou", "gyrostat", "gyrostat_surrogate"])
def test_zero_amplitude_gives_exact_zeros(basis, driver):
    a = generate_spectral_wind(basis, driver, 48, 7200.0, amp=0.0, sigma=SIGMA,
                               cx=0.5, cy=0.03, x0=1e5, y0=2e5, seed=3, burnin_units=5.0)
    assert a.shape == (48, basis.n_amp)
    assert torch.count_nonzero(a) == 0


@pytest.mark.parametrize("driver", ["spectral_ou", "gyrostat", "gyrostat_surrogate"])
def test_drivers_are_deterministic_and_finite(basis, driver):
    kw = dict(n_steps=60, dt_seconds=7200.0, amp=1e-11, sigma=SIGMA, cx=0.5, cy=0.03,
              x0=1e5, y0=2e5, burnin_units=5.0)
    a = generate_spectral_wind(basis, driver, seed=9, **kw)
    b = generate_spectral_wind(basis, driver, seed=9, **kw)
    c = generate_spectral_wind(basis, driver, seed=10, **kw)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.isfinite(a).all()
    assert float((a - c).norm() / a.norm()) > 0.1


def test_drift_is_a_rigid_translation_for_frozen_amplitudes(basis):
    kw = dict(n_steps=13, dt_seconds=7200.0, amp=1e-11, sigma=SIGMA, x0=0.0, y0=0.0,
              seed=1, tau_days=1e9)
    still = generate_spectral_wind(basis, "spectral_ou", cx=0.0, cy=0.0, **kw)
    moving = generate_spectral_wind(basis, "spectral_ou", cx=0.5, cy=0.0, **kw)
    shift = 0.5 * 12 * 7200.0
    torch.testing.assert_close(moving[-1], basis.translate(still[-1], shift, 0.0),
                               rtol=1e-6, atol=1e-20)


@pytest.mark.slow
@pytest.mark.parametrize("driver", ["spectral_ou", "gyrostat"])
def test_long_series_reproduce_storm_mode_spectrum(basis, driver):
    a = generate_spectral_wind(basis, driver, 60000, 7200.0, amp=1.0, sigma=SIGMA,
                               cx=0.0, cy=0.0, x0=0.0, y0=0.0, seed=2)
    target = basis.ricker_mode_std(SIGMA).numpy()
    pair_rms = np.sqrt(0.5 * (a.numpy() ** 2).reshape(a.shape[0], -1, 2).sum(-1).mean(0))
    np.testing.assert_allclose(pair_rms, target[::2], rtol=0.1)


def test_spectral_forcing_drives_qg_through_tendency(basis):
    class _SpectralQG(QGDynamics):
        def _wind_curl_spectral(self, qh, wind_state_t):
            return basis.curl_spectral(wind_state_t).to(qh.dtype)

    dyn = _SpectralQG(nx=64, L=L, dtype=torch.float64)
    a = generate_spectral_wind(basis, "gyrostat", 2, 7200.0, amp=1e-11, sigma=SIGMA,
                               cx=0.5, cy=0.03, x0=0.0, y0=0.0, seed=0, burnin_units=5.0)
    qh = torch.zeros(1, 2, 64, 33, dtype=torch.complex128)
    tend = dyn._tendency(qh, 0.05, 0.0, 1.5e-11, 5.787e-7, a[0])
    q1 = torch.fft.irfft2(tend[0, 0], s=(64, 64))
    torch.testing.assert_close(q1, basis.curl_field(a[0]), rtol=1e-8, atol=1e-20)
    assert math.isclose(float(tend[0, 1].abs().max()), 0.0, abs_tol=1e-30)
