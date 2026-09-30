import numpy as np
import pytest
import torch

from evaluation.baselines import ETKF, DynamicsBase, EnKS, ObsOperator


class _Linear(DynamicsBase):
    def __init__(self, F: torch.Tensor):
        super().__init__()
        self.state_dim = F.shape[0]
        self.F = F

    def step(self, state, forcing=None, **kwargs):
        return state @ self.F.T


def _kalman_rts(m0, P0, F, H, R, obs, mask):
    T, D = len(mask), len(m0)
    mf, Pf = np.zeros((T, D)), np.zeros((T, D, D))
    ma, Pa = np.zeros((T, D)), np.zeros((T, D, D))
    ma[0], Pa[0] = m0, P0
    for t in range(1, T):
        mf[t], Pf[t] = F @ ma[t - 1], F @ Pa[t - 1] @ F.T
        ma[t], Pa[t] = mf[t], Pf[t]
        if mask[t]:
            K = Pf[t] @ H.T @ np.linalg.inv(H @ Pf[t] @ H.T + R)
            ma[t] = mf[t] + K @ (obs[t] - H @ mf[t])
            Pa[t] = (np.eye(D) - K @ H) @ Pf[t]
    ms, Ps = ma.copy(), Pa.copy()
    for t in range(T - 2, -1, -1):
        J = Pa[t] @ F.T @ np.linalg.inv(Pf[t + 1])
        ms[t] = ma[t] + J @ (ms[t + 1] - mf[t + 1])
        Ps[t] = Pa[t] + J @ (Ps[t + 1] - Pf[t + 1]) @ J.T
    return ma, ms, Ps


def _setup(seed: int = 3):
    D, N, T, R_var = 4, 40, 16, 0.3
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(D, D, generator=g, dtype=torch.float64))
    F = (1.02 * Q).float()
    idx = [0, 2]
    init = torch.randn(N, D, generator=g) * torch.tensor([1.0, 0.7, 1.3, 0.5]) + 0.3
    mask = torch.zeros(T, dtype=torch.bool)
    mask[3::3] = True
    obs = torch.randn(T, len(idx), generator=g)
    ones_x, ones_y = torch.ones(D, len(idx)), torch.ones(len(idx), len(idx))
    kw = dict(N_ensemble=N, R_var=R_var, inflation=1.0, dt=1.0, dynamics=_Linear(F),
              obs_operator=ObsOperator(D, obs_indices=idx), loc_mode="ensrf", etkf_ridge=0.0,
              loc_Lx_t=[ones_x] * T, loc_Ly_t=[ones_y] * T)
    x0 = init.double().numpy()
    ref = _kalman_rts(x0.mean(0), np.cov(x0.T), F.double().numpy(), np.eye(D)[idx],
                      R_var * np.eye(len(idx)), obs.double().numpy(), mask.numpy())
    return kw, init, obs, mask, T, D, ref


def test_unlocalized_enks_is_the_exact_rts_smoother():
    kw, init, obs, mask, T, D, (ma, ms, Ps) = _setup()
    f = ETKF(**kw)
    f.init_ensemble = init.clone()
    filt = f.assimilate(obs, mask, torch.zeros(T), true_state=torch.zeros(T, D))
    np.testing.assert_allclose(filt.trajectory, ma, atol=1e-3)
    s = EnKS(**kw)
    s.init_ensemble = init.clone()
    res = s.assimilate(obs, mask, torch.zeros(T), true_state=torch.zeros(T, D))
    np.testing.assert_allclose(res.trajectory, ms, atol=1e-3)
    ens = res.ensemble.astype(np.float64)
    cov = np.einsum("ntd,nte->tde", ens - ens.mean(0), ens - ens.mean(0)) / (ens.shape[0] - 1)
    np.testing.assert_allclose(cov, Ps, atol=2e-3)
    last = int(np.nonzero(mask.numpy())[0][-1])
    np.testing.assert_allclose(res.trajectory[last:], filt.trajectory[last:], atol=1e-5)


def test_lag_limits_how_far_back_an_analysis_reaches():
    kw, init, obs, mask, T, D, _ = _setup(seed=5)
    out = {}
    for lag in (1, None):
        s = EnKS(**kw, lag=lag)
        s.init_ensemble = init.clone()
        out[lag] = s.assimilate(obs, mask, torch.zeros(T), true_state=torch.zeros(T, D)).trajectory
    f = ETKF(**kw)
    f.init_ensemble = init.clone()
    filt = f.assimilate(obs, mask, torch.zeros(T), true_state=torch.zeros(T, D)).trajectory
    assert not np.allclose(out[1][:3], filt[:3], atol=1e-6)
    assert not np.allclose(out[1][:3], out[None][:3], atol=1e-6)
    np.testing.assert_allclose(out[1][12:15], out[None][12:15], atol=1e-5)
    assert not np.allclose(out[1][9:12], out[None][9:12], atol=1e-6)


def test_enks_refuses_unsupported_settings():
    kw, *_ = _setup()
    with pytest.raises(NotImplementedError):
        EnKS(**{**kw, "loc_mode": "square_root"})
    with pytest.raises(NotImplementedError):
        EnKS(**{**kw, "inflation": 1.1})
    with pytest.raises(ValueError):
        EnKS(**kw, lag=0)


def test_time_taper_recovers_the_untapered_smoother_when_wide_and_damps_when_narrow():
    kw, init, obs, mask, T, D, _ = _setup(seed=7)

    def smooth(**extra):
        s = EnKS(**kw, **extra)
        s.init_ensemble = init.clone()
        return s.assimilate(obs, mask, torch.zeros(T), true_state=torch.zeros(T, D)).trajectory

    full, wide, narrow = smooth(), smooth(taper_steps=1e6), smooth(taper_steps=1.0)
    np.testing.assert_allclose(wide, full, atol=1e-5)
    assert not np.allclose(narrow[:9], full[:9], atol=1e-5)
    f = ETKF(**kw)
    f.init_ensemble = init.clone()
    filt = f.assimilate(obs, mask, torch.zeros(T), true_state=torch.zeros(T, D)).trajectory
    np.testing.assert_allclose(narrow[:2], filt[:2], atol=1e-5)
    with pytest.raises(ValueError):
        EnKS(**kw, taper_steps=0.0)
