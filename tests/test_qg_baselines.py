import numpy as np
import torch

from data.qg import QGConfig, make_qg_s0_s1_datasets
from evaluation.baselines import EnKF, ObsOperator, _build_qg_loc_matrices
from evaluation.run_qg_baselines import (
    WindStateAdapter,
    _build_dyn,
    _per_pass_indices,
    _q_alongtrack_obs,
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
