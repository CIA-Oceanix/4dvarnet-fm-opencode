import numpy as np
import pytest
import torch

from models.gyrostat_driver import GyrostatSystem, get_preset
from models.qg_batched import (
    BatchedQGDynamics,
    SpectralWindForcing,
    StormWindForcing,
)
from models.qg_dynamics import QGDynamics
from models.qg_wind_batched import (
    BatchedGyrostat,
    batched_phase_surrogate,
    batched_spectral_wind,
    batched_unit_ou,
)
from models.qg_wind_modes import FourierWindBasis

NX = 32
L = 1e6
SIGMA = 250000.0
PARAMS = {"rd": [15000.0, 13000.0, 17000.0], "U1": [0.05, 0.045, 0.057],
          "rek": [5.787e-7, 5.2e-7, 6.4e-7]}


def _reference(i, **extra):
    return QGDynamics(nx=NX, L=L, dt=7200.0, beta=1.5e-11, rd=PARAMS["rd"][i], delta=0.25,
                      U1=PARAMS["U1"][i], U2=0.0, rek=PARAMS["rek"][i], dtype=torch.float64, **extra)


def _batched(**extra):
    return BatchedQGDynamics(3, nx=NX, L=L, dt=7200.0, beta=1.5e-11, rd=PARAMS["rd"],
                             delta=0.25, U1=PARAMS["U1"], U2=0.0, rek=PARAMS["rek"],
                             dtype=torch.float64, **extra)


def _spun_up_q(model, seeds, steps=300):
    q0 = model.initial_q(seeds)
    _, q = model.rollout(q0, steps)
    return q


def test_unforced_matches_reference_per_window():
    bat = _batched()
    q0 = _spun_up_q(bat, [1, 2, 3])
    traj, _ = bat.rollout(q0, 20)
    for i in range(3):
        ref = _reference(i)
        ref_traj = ref.rollout_trajectory(ref._flatten(q0[i]), 20)
        torch.testing.assert_close(bat._flatten(traj[i]), ref_traj, rtol=1e-10, atol=1e-18)


def test_spectral_forcing_matches_reference_per_window():
    basis = FourierWindBasis(nx=NX, L=L, kmax=2)
    amps = torch.randn(3, 20, basis.n_amp, dtype=torch.float64) * 2e-12
    bat = _batched()
    q0 = _spun_up_q(bat, [4, 5, 6])
    traj, _ = bat.rollout(q0, 20, forcing=SpectralWindForcing(basis, amps))

    class _Spectral(QGDynamics):
        def _wind_curl_spectral(self, qh, w):
            return basis.curl_spectral(w).to(qh.dtype)

    for i in range(3):
        ref = _Spectral(nx=NX, L=L, dt=7200.0, beta=1.5e-11, rd=PARAMS["rd"][i], delta=0.25,
                        U1=PARAMS["U1"][i], U2=0.0, rek=PARAMS["rek"][i], dtype=torch.float64)
        ref_traj = ref.rollout_trajectory(ref._flatten(q0[i]), 20, wind_state=amps[i])
        torch.testing.assert_close(bat._flatten(traj[i]), ref_traj, rtol=1e-10, atol=1e-18)


def test_storm_forcing_matches_reference_per_window():
    bat = _batched()
    q0 = _spun_up_q(bat, [7, 8, 9])
    curl_model = _reference(0, wind_amp=1e-11, wind_sigma=SIGMA)
    ws = torch.stack([curl_model.generate_wind_state(20, seed=s, x0=3e5, y0=6e5) for s in (11, 12, 13)])
    traj, _ = bat.rollout(q0, 20, forcing=StormWindForcing(curl_model, ws))
    for i in range(3):
        ref = _reference(i, wind_amp=1e-11, wind_sigma=SIGMA)
        ref_traj = ref.rollout_trajectory(ref._flatten(q0[i]), 20, wind_state=ws[i])
        torch.testing.assert_close(bat._flatten(traj[i]), ref_traj, rtol=1e-10, atol=1e-18)


def test_eddy_drag_adds_exactly_rcf_k2_psi1():
    base = _batched()
    drag = _batched(r_cf=[0.0, 2.8e-8, 1e-7])
    q0 = _spun_up_q(base, [1, 2, 3], steps=50)
    qh = torch.fft.rfft2(q0, dim=(-2, -1))
    diff = drag._tendency(qh, None) - base._tendency(qh, None)
    expected = drag.rcf_b * drag.K2 * base._invert(qh)[:, 0]
    torch.testing.assert_close(diff[:, 0], expected.to(diff.dtype), rtol=1e-9,
                               atol=1e-9 * float(expected.abs().max()))
    assert float(diff[:, 1].abs().max()) == 0.0
    assert float(diff[0].abs().max()) == 0.0


def test_eddy_drag_removes_kinetic_energy():
    base = _batched()
    drag = _batched(r_cf=1e-6)
    q0 = _spun_up_q(base, [1, 2, 3])
    _, qa = base.rollout(q0, 100)
    _, qb = drag.rollout(q0, 100)
    ke = lambda m, q: (m.K2 * torch.fft.rfft2(m.streamfunctions(q), dim=(-2, -1)).abs() ** 2).sum(dim=(-2, -1))[:, 0]  # noqa: E731
    assert bool((ke(drag, qb) < ke(base, qa)).all())


def test_streamfunctions_accepts_time_axis():
    bat = _batched()
    q0 = _spun_up_q(bat, [1, 2, 3], steps=10)
    traj, _ = bat.rollout(q0, 4)
    psi = bat.streamfunctions(traj)
    assert psi.shape == traj.shape
    torch.testing.assert_close(psi[:, -1], bat.streamfunctions(traj[:, -1]))


def test_per_window_parameter_shape_is_checked():
    with pytest.raises(ValueError):
        BatchedQGDynamics(3, nx=NX, rd=[1.0, 2.0])


def test_batched_gyrostat_matches_numpy_system():
    spec = get_preset("l63ring4")
    gyro = BatchedGyrostat(spec)
    y0 = gyro.initial_states([1, 2])
    traj = gyro.integrate(y0, 200, 0.01)
    ref = GyrostatSystem(spec)
    for b in range(2):
        y = y0[b].numpy().copy()
        out = []
        for _ in range(200):
            out.append(y.copy())
            for _ in range(2):
                y = ref._rk4(y, 0.005)
        np.testing.assert_allclose(traj[b].numpy(), np.array(out), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("driver", ["spectral_ou", "gyrostat", "gyrostat_surrogate"])
def test_window_series_do_not_depend_on_batch_composition(driver):
    basis = FourierWindBasis(nx=NX, L=L, kmax=2)
    kw = dict(n_steps=40, dt_seconds=7200.0, sigma=SIGMA, burnin_units=5.0)
    full = batched_spectral_wind(basis, driver, [11, 12, 13], amp=[1e-11, 2e-11, 3e-12],
                                 cx=[0.5, 0.3, 0.7], cy=[0.0, 0.03, -0.02],
                                 x0=[0.0, 1e5, 2e5], y0=[3e5, 4e5, 5e5], **kw)
    alone = batched_spectral_wind(basis, driver, [12], amp=2e-11, cx=0.3, cy=0.03,
                                  x0=1e5, y0=4e5, **kw)
    torch.testing.assert_close(full[1:2], alone, rtol=0, atol=0)


def test_zero_amplitude_window_is_exactly_zero():
    basis = FourierWindBasis(nx=NX, L=L, kmax=2)
    a = batched_spectral_wind(basis, "gyrostat", [1, 2], 30, 7200.0, amp=[0.0, 1e-11],
                              cx=0.5, cy=0.03, x0=0.0, y0=0.0, sigma=SIGMA, burnin_units=5.0)
    assert torch.count_nonzero(a[0]) == 0
    assert torch.count_nonzero(a[1]) > 0


def test_per_window_time_unit_changes_memory():
    basis = FourierWindBasis(nx=NX, L=L, kmax=2)
    kw = dict(n_steps=1500, dt_seconds=12 * 3600.0, amp=1.0, cx=0.0, cy=0.0, x0=0.0, y0=0.0,
              sigma=SIGMA, burnin_units=5.0)
    a = batched_spectral_wind(basis, "gyrostat", [3, 3], time_unit_days=[30.0, 90.0], **kw)
    z = a[:, :, 0] - a[:, :, 0].mean(1, keepdim=True)
    lag = 10
    acf = (z[:, lag:] * z[:, :-lag]).mean(1) / (z * z).mean(1)
    assert float(acf[1]) > float(acf[0])


def test_ou_has_unit_variance_and_target_memory():
    z = batched_unit_ou([1, 2, 3, 4], 20000, 2, dt_days=0.25, tau_days=15.0)
    assert abs(float(z.var()) - 1.0) < 0.15
    lag = 60
    acf = float((z[:, lag:] * z[:, :-lag]).mean() / (z * z).mean())
    assert abs(acf - np.exp(-lag * 0.25 / 15.0)) < 0.1


def test_surrogate_preserves_spectra_and_uses_shared_phases():
    y = torch.randn(2, 512, 3, dtype=torch.float64)
    y[:, :, 1] = y[:, :, 0] ** 2
    s = batched_phase_surrogate(y, [5, 6])
    fy, fs = torch.fft.rfft(y, dim=1), torch.fft.rfft(s, dim=1)
    torch.testing.assert_close(fs.abs(), fy.abs(), rtol=1e-9, atol=1e-8)
    torch.testing.assert_close(fs[..., 0] * fs[..., 1].conj(), fy[..., 0] * fy[..., 1].conj(),
                               rtol=1e-8, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_batched_dynamics_runs_and_matches_cpu_float64():
    cpu = _batched()
    gpu = BatchedQGDynamics(3, nx=NX, L=L, dt=7200.0, beta=1.5e-11, rd=PARAMS["rd"], delta=0.25,
                            U1=PARAMS["U1"], U2=0.0, rek=PARAMS["rek"], dtype=torch.float64,
                            device="cuda")
    q0 = _spun_up_q(cpu, [1, 2, 3], steps=50)
    a, _ = cpu.rollout(q0, 10)
    b, _ = gpu.rollout(q0.cuda(), 10)
    torch.testing.assert_close(b.cpu(), a, rtol=1e-8, atol=1e-18)


def test_window_series_invariant_when_substeps_differ_across_the_batch():
    basis = FourierWindBasis(nx=NX, L=L, kmax=2)
    kw = dict(n_steps=30, dt_seconds=12 * 3600.0, amp=1e-11, cx=0.5, cy=0.0, x0=0.0, y0=0.0,
              sigma=SIGMA, burnin_units=2.0)
    full = batched_spectral_wind(basis, "gyrostat", [21, 22], time_unit_days=[30.0, 90.0], **kw)
    alone = batched_spectral_wind(basis, "gyrostat", [22], time_unit_days=90.0, **kw)
    torch.testing.assert_close(full[1:2], alone, rtol=0, atol=0)
