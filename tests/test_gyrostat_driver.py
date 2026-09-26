import numpy as np
import pytest

from models.gyrostat_driver import (
    GyrostatSystem,
    get_preset,
    lorenz63_ring_spec,
    lorenz63_spec,
    measure_attractor_stats,
    phase_randomized_surrogate,
)


def _textbook_l63(state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    return np.stack([sigma * (y - x), x * (rho - z) - y, x * y - beta * z], axis=-1)


def test_l63_preset_matches_textbook_after_z_shift():
    rng = np.random.default_rng(0)
    xyz = rng.normal(0.0, 10.0, (50, 3))
    shifted = xyz.copy()
    shifted[:, 2] -= 28.0 + 10.0
    got = GyrostatSystem(lorenz63_spec()).rhs(shifted)
    np.testing.assert_allclose(got, _textbook_l63(xyz), rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("preset", ["l63", "l63ring4"])
def test_core_conserves_energy(preset):
    system = GyrostatSystem(get_preset(preset))
    rng = np.random.default_rng(1)
    y = rng.normal(0.0, 3.0, system.n_modes)
    e0 = system.energy(y)
    for _ in range(2000):
        y = system._rk4(y, 1e-3, core_only=True)
    assert abs(system.energy(y) - e0) / e0 < 1e-8


@pytest.mark.parametrize("preset", ["l63", "l63ring4"])
def test_core_rhs_is_energy_neutral_pointwise(preset):
    system = GyrostatSystem(get_preset(preset))
    y = np.random.default_rng(2).normal(0.0, 5.0, (100, system.n_modes))
    power = np.sum(y * system.core_rhs(y), axis=-1)
    np.testing.assert_allclose(power, 0.0, atol=1e-9)


def test_integrate_is_deterministic_and_seed_dependent():
    system = GyrostatSystem(lorenz63_ring_spec())
    a = system.integrate(200, 0.01, seed=5, burnin_units=5.0)
    b = system.integrate(200, 0.01, seed=5, burnin_units=5.0)
    c = system.integrate(200, 0.01, seed=6, burnin_units=5.0)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)
    assert np.isfinite(a).all()


def test_surrogate_preserves_auto_and_cross_spectra():
    rng = np.random.default_rng(3)
    t = np.arange(4096)
    base = rng.normal(size=(4096, 1))
    y = np.concatenate([np.sin(0.05 * t)[:, None] + 0.3 * base,
                        np.cos(0.05 * t)[:, None] - 0.5 * base,
                        base ** 3], axis=1)
    s = phase_randomized_surrogate(y, seed=11)
    fy = np.fft.rfft(y, axis=0)
    fs = np.fft.rfft(s, axis=0)
    np.testing.assert_allclose(np.abs(fs), np.abs(fy), rtol=1e-9, atol=1e-8)
    cross_y = fy[:, 0] * np.conj(fy[:, 1])
    cross_s = fs[:, 0] * np.conj(fs[:, 1])
    np.testing.assert_allclose(cross_s, cross_y, rtol=1e-8, atol=1e-6)
    assert not np.allclose(s, y)


def test_surrogate_is_gaussian_for_bimodal_input():
    system = GyrostatSystem(lorenz63_spec())
    x = system.integrate(40000, 0.01, seed=0, burnin_units=20.0)[:, :1]
    x = (x - x.mean()) / x.std()
    s = phase_randomized_surrogate(x, seed=4)
    s = (s - s.mean()) / s.std()
    assert abs(float((x ** 4).mean()) - 3.0) > 0.4
    assert abs(float((s ** 4).mean()) - 3.0) < 0.3


@pytest.mark.slow
@pytest.mark.parametrize("preset", ["l63", "l63ring4"])
def test_stored_attractor_constants_match_fresh_run(preset):
    spec = get_preset(preset)
    stats = measure_attractor_stats(spec, total_units=1000.0, seed=123)
    np.testing.assert_allclose(stats["std"], spec.mode_std, rtol=0.05)
    assert np.all(np.abs(stats["mean"] - spec.mode_mean) < 0.1 * spec.mode_std)
    assert stats["largest_lyapunov"] > 0.5
