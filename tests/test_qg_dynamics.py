import numpy as np
import pytest
import torch

from models.qg_dynamics import QGDynamics

NX_SMALL = 32
NOMINAL = {"U1": 0.05, "U2": 0.0, "rd": 15000.0, "beta": 1.5e-11,
           "delta": 0.25, "rek": 5.787e-7}


@pytest.fixture
def dyn():
    return QGDynamics(nx=NX_SMALL, dt=7200.0, **NOMINAL)


def test_state_roundtrip(dyn):
    state = torch.randn(dyn.state_dim)
    grid = dyn._grid(state)
    assert grid.shape == (2, NX_SMALL, NX_SMALL)
    back = dyn._flatten(grid.unsqueeze(0)).squeeze(0)
    assert torch.equal(back, state)


def test_step_preserves_shape(dyn):
    single = torch.randn(dyn.state_dim)
    batch = torch.randn(4, dyn.state_dim)
    assert dyn.step(single).shape == single.shape
    assert dyn.step(batch).shape == batch.shape


def test_odd_grid_rejected():
    with pytest.raises(ValueError):
        QGDynamics(nx=33)


def test_deterministic_given_seed(dyn):
    t1, _ = dyn.generate_full_trajectory(num_steps=5, seed=123,
                                         spinup_steps=10)
    t2, _ = dyn.generate_full_trajectory(num_steps=5, seed=123,
                                         spinup_steps=10)
    assert torch.equal(t1, t2)


def test_batch_consistency(dyn):
    qb = dyn._flatten(dyn._initial_q(3, seed=7, device=dyn.device))
    rolled = dyn.rollout_steps(qb, steps=4)
    assert rolled.shape == qb.shape
    for i in range(3):
        single = dyn.rollout_steps(qb[i], steps=4)
        assert torch.allclose(rolled[i], single, atol=0.0)


def test_rollout_matches_step_loop(dyn):
    state = torch.randn(dyn.state_dim) * 1e-7
    rolled = dyn.rollout_steps(state, steps=3)
    stepped = state
    for _ in range(3):
        stepped = dyn.step(stepped)
    assert torch.allclose(rolled, stepped, atol=1e-12)


def test_inviscid_conservation():
    dyn = QGDynamics(nx=NX_SMALL, dt=7200.0, beta=0.0, U1=0.0, U2=0.0,
                     rek=0.0, filterfac=0.0, dtype=torch.float64)
    state = dyn._flatten(dyn._initial_q(1, seed=11, device=dyn.device)).squeeze(0)
    state = state * 10.0
    ke0 = dyn.kinetic_energy(state)
    ens0 = dyn.enstrophy(state)
    for _ in range(60):
        state = dyn.step(state)
    ke1 = dyn.kinetic_energy(state)
    ens1 = dyn.enstrophy(state)
    assert torch.isfinite(state).all()
    assert abs((ke1 - ke0) / ke0) < 1e-3
    assert abs((ens1 - ens0) / ens0) < 1e-3


def test_nominal_stability():
    dyn = QGDynamics(nx=NX_SMALL, dt=7200.0, **NOMINAL)
    traj, _ = dyn.generate_full_trajectory(num_steps=72, seed=42,
                                           spinup_steps=200)
    assert torch.isfinite(traj).all()
    kes = [dyn.kinetic_energy(traj[i]) for i in range(0, 72, 12)]
    assert all(k > 0 for k in kes)


def test_param_override_changes_trajectory(dyn):
    state = dyn._flatten(dyn._initial_q(1, seed=3, device=dyn.device)).squeeze(0)
    base = dyn.rollout_steps(state, steps=5)
    alt = dyn.rollout_steps(state, steps=5, U1=0.02)
    assert not torch.allclose(base, alt)


def test_forcing_argument_ignored(dyn):
    state = torch.randn(dyn.state_dim) * 1e-7
    forcing = torch.zeros(36, dyn.state_dim)
    assert torch.equal(dyn.step(state), dyn.step(state, forcing))


def test_streamfunction_inversion_residual():
    dyn = QGDynamics(nx=NX_SMALL, dtype=torch.float64, **NOMINAL)
    state = dyn._flatten(dyn._initial_q(1, seed=5, device=dyn.device)).squeeze(0)
    q = dyn._grid(state)
    qh = torch.fft.rfft2(q, dim=(-2, -1))
    ph = dyn._invert(qh)
    K2 = dyn.K2
    rec1h = (-K2 - dyn.F1) * ph[..., 0, :, :] + dyn.F1 * ph[..., 1, :, :]
    rec2h = (-K2 - dyn.F2) * ph[..., 1, :, :] + dyn.F2 * ph[..., 0, :, :]
    mask = K2 > 0
    assert torch.allclose(rec1h[..., mask], qh[..., 0, :, :][..., mask],
                          rtol=1e-8, atol=1e-15)
    assert torch.allclose(rec2h[..., mask], qh[..., 1, :, :][..., mask],
                          rtol=1e-8, atol=1e-15)


def test_pyqg_tendency_equivalence():
    pytest.importorskip("pyqg")
    import pyqg

    nx = 32
    params = {"beta": 1.5e-11, "rd": 15000.0, "delta": 0.25, "U1": 0.05,
              "U2": 0.0, "rek": 5.787e-7}
    rng = np.random.RandomState(1)
    q1 = 1e-5 * rng.randn(nx, nx)
    q2 = 1e-5 * rng.randn(nx, nx)

    m = pyqg.QGModel(nx=nx, dt=7200.0, tmax=7200.0, twrite=10 ** 9,
                     tavestart=1e18, log_level=0, **params)
    m.set_q1q2(q1, q2)
    m._invert()
    m._do_advection()
    m._do_friction()
    ref = np.asarray(m.dqhdt).copy()

    dyn = QGDynamics(nx=nx, dt=7200.0, dtype=torch.float64, **params)
    q_t = torch.tensor(np.stack([q1, q2]), dtype=torch.float64).unsqueeze(0)
    qh = torch.fft.rfft2(q_t, dim=(-2, -1))
    tend = dyn._tendency(qh, dyn.U1, dyn.U2, dyn.beta, dyn.rek)
    actual = tend[0].cpu().numpy()

    scale = np.abs(ref).max()
    np.testing.assert_allclose(actual, ref, rtol=1e-5,
                               atol=1e-9 * max(scale, 1e-30))
