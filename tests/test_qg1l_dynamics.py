import pytest
import torch

from models.qg1l_dynamics import QG1LDynamics

NX_SMALL = 32
NOMINAL = {"U1": 0.05, "rd": 15000.0, "beta": 1.5e-11, "rek": 5.787e-7}


@pytest.fixture
def dyn():
    return QG1LDynamics(nx=NX_SMALL, dt=7200.0, **NOMINAL)


def test_state_roundtrip(dyn):
    state = torch.randn(dyn.state_dim)
    grid = dyn._grid(state)
    assert grid.shape == (NX_SMALL, NX_SMALL)
    back = dyn._flatten(grid.unsqueeze(0)).squeeze(0)
    assert torch.equal(back, state)


def test_state_dim_single_layer(dyn):
    assert dyn.state_dim == NX_SMALL * NX_SMALL


def test_step_preserves_shape(dyn):
    single = torch.randn(dyn.state_dim)
    batch = torch.randn(4, dyn.state_dim)
    assert dyn.step(single).shape == single.shape
    assert dyn.step(batch).shape == batch.shape


def test_odd_grid_rejected():
    with pytest.raises(ValueError):
        QG1LDynamics(nx=33)


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
        assert torch.allclose(rolled[i], single, rtol=1e-5, atol=1e-10)


def test_rollout_matches_step_loop(dyn):
    state = torch.randn(dyn.state_dim) * 1e-7
    rolled = dyn.rollout_steps(state, steps=3)
    stepped = state
    for _ in range(3):
        stepped = dyn.step(stepped)
    assert torch.allclose(rolled, stepped, atol=1e-12)


def test_step_runtime_param_overrides(dyn):
    state = dyn._flatten(dyn._initial_q(1, seed=3, device=dyn.device)).squeeze(0)
    base = dyn.rollout_steps(state, steps=5)
    alt = dyn.rollout_steps(state, steps=5, U1=0.02)
    assert not torch.allclose(base, alt)


def test_forcing_argument_ignored(dyn):
    state = torch.randn(dyn.state_dim) * 1e-7
    forcing = torch.zeros(36, dyn.state_dim)
    assert torch.equal(dyn.step(state), dyn.step(state, forcing))


def test_streamfunction_inversion_residual(dyn):
    dyn = QG1LDynamics(nx=NX_SMALL, dtype=torch.float64, **NOMINAL)
    state = dyn._flatten(dyn._initial_q(1, seed=5, device=dyn.device)).squeeze(0)
    q = dyn._grid(state)
    qh = torch.fft.rfft2(q, dim=(-2, -1))
    ph = dyn._invert(qh)
    K2 = dyn.K2
    rec = (-K2 - dyn.rd ** -2) * ph
    mask = K2 > 0
    assert torch.allclose(rec[..., mask], qh[..., mask], rtol=1e-8, atol=1e-15)


def _total_energy(dyn, state):
    q = dyn._grid(state)
    qh = torch.fft.rfft2(q, dim=(-2, -1))
    ph = dyn._invert(qh)
    M2 = float(dyn.nx * dyn.ny) ** 2
    e = 0.5 * ((dyn.K2 + dyn.rd ** -2) * ph.abs() ** 2).sum(dim=(-2, -1)) / M2
    return e.squeeze(0)


def test_inviscid_conservation():
    dyn = QG1LDynamics(nx=NX_SMALL, dt=7200.0, beta=0.0, U1=0.0,
                       rek=0.0, filterfac=0.0, dtype=torch.float64)
    state = dyn._flatten(dyn._initial_q(1, seed=11, device=dyn.device)).squeeze(0)
    state = state * 10.0
    e0 = _total_energy(dyn, state)
    ens0 = dyn.enstrophy(state)
    for _ in range(60):
        state = dyn.step(state)
    e1 = _total_energy(dyn, state)
    ens1 = dyn.enstrophy(state)
    assert torch.isfinite(state).all()
    assert abs((e1 - e0) / e0) < 1e-3
    assert abs((ens1 - ens0) / ens0) < 1e-3


def test_nominal_stability():
    dyn = QG1LDynamics(nx=NX_SMALL, dt=7200.0, **NOMINAL)
    traj, _ = dyn.generate_full_trajectory(num_steps=72, seed=42,
                                           spinup_steps=200)
    assert torch.isfinite(traj).all()
    kes = [dyn.kinetic_energy(traj[i]) for i in range(0, 72, 12)]
    assert all(k > 0 for k in kes)


def test_wind_zero_returns_zero_state():
    dyn = QG1LDynamics(nx=NX_SMALL, dt=7200.0, wind_amp=0.0, **NOMINAL)
    state = dyn.generate_wind_state(num_steps=10)
    assert state.shape == (10, 3)
    assert torch.all(state == 0.0)


def test_wind_term_matches_hand_computation():
    dyn = QG1LDynamics(nx=NX_SMALL, dt=7200.0, dtype=torch.float64,
                       wind_amp=1e-8, **NOMINAL)
    state = dyn._flatten(dyn._initial_q(1, seed=5, device=dyn.device))
    qh = torch.fft.rfft2(dyn._grid(state), dim=(-2, -1))
    ws = torch.tensor([1.5, 0.4 * dyn.L, 0.6 * dyn.W], dtype=torch.float64)
    t0 = dyn._tendency(qh, dyn.U1, dyn.beta, dyn.rek)
    t1 = dyn._tendency(qh, dyn.U1, dyn.beta, dyn.rek, wind_state_t=ws)
    diff = t1 - t0
    expected = torch.fft.rfft2(dyn.wind_curl_field(ws), dim=(-2, -1)).to(t0.dtype)
    assert torch.allclose(diff, expected, rtol=1e-8, atol=1e-12)


@pytest.mark.slow
def test_wind_zero_matches_unforced_bitwise():
    dyn = QG1LDynamics(nx=NX_SMALL, dt=7200.0, wind_amp=0.0, **NOMINAL)
    traj1, _ = dyn.generate_full_trajectory(num_steps=20, spinup_steps=10,
                                            seed=3)
    traj2, _ = dyn.generate_full_trajectory(num_steps=20, spinup_steps=10,
                                            seed=3)
    assert torch.equal(traj1, traj2)


@pytest.mark.slow
def test_wind_positive_changes_trajectory():
    dyn0 = QG1LDynamics(nx=NX_SMALL, dt=7200.0, wind_amp=0.0, **NOMINAL)
    dyn1 = QG1LDynamics(nx=NX_SMALL, dt=7200.0, wind_amp=1e-8, **NOMINAL)
    traj0, _ = dyn0.generate_full_trajectory(num_steps=20, spinup_steps=10,
                                             seed=3)
    traj1, _ = dyn1.generate_full_trajectory(num_steps=20, spinup_steps=10,
                                             seed=3)
    assert not torch.equal(traj0, traj1)


def test_rollout_trajectory_reproduces_generate():
    dyn = QG1LDynamics(nx=NX_SMALL, dt=7200.0, wind_amp=1e-8, **NOMINAL)
    nsteps = 8
    traj, ws = dyn.generate_full_trajectory(num_steps=nsteps, spinup_steps=5,
                                            seed=3)
    roll = dyn.rollout_trajectory(traj[0], nsteps - 1, wind_state=ws)
    assert roll.shape == (nsteps, dyn.state_dim)
    assert torch.allclose(roll, traj, rtol=1e-5, atol=1e-5)


def test_rollout_trajectory_batched():
    dyn = QG1LDynamics(nx=NX_SMALL, dt=7200.0, **NOMINAL)
    batch = torch.randn(3, dyn.state_dim)
    ws = torch.zeros(5, 3)
    out = dyn.rollout_trajectory(batch, 5, wind_state=ws)
    assert out.shape == (6, 3, dyn.state_dim)
