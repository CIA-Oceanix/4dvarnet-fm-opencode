"""ETKS (ensemble transform Kalman smoother) on the unlocalised ETKF.

Design: docs/plans/tech/l96_etks_smoother.md (tests 1-10 there).
Prototype measurements: docs/results/l96_etks_prototype.md.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from evaluation.baselines import (
    ETKF,
    ETKS,
    ObsOperator,
    _etkf_sqrt_transform,
    _etks_factor,
    _etks_smooth,
)
from models.dynamics import DynamicsBase
from models.lorenz96_dynamics import Lorenz96Dynamics

NO, J, DT = 2, 4, 0.01
STATE_DIM = NO + NO * J
L96_PARAMS = dict(F=8.0, c1=1.0, h=1.0, hx=1.0, eps=0.1)


def _l96_obs_operator() -> ObsOperator:
    return ObsOperator(STATE_DIM, list(range(NO)) + [NO + k * J for k in range(NO)])


def _l96_dynamics() -> Lorenz96Dynamics:
    return Lorenz96Dynamics(dt=DT, NO=NO, J=J, h=1.0, coupling_exponent=1.6, clip_range=50.0)


def _l96_window(T: int = 100, every: int = 5, seed: int = 0, nan_channels: bool = False):
    """A physical two-scale L96 trajectory, observed every ``every`` steps."""
    g = torch.Generator().manual_seed(seed)
    dyn, op = _l96_dynamics(), _l96_obs_operator()
    x = torch.randn(1, STATE_DIM, generator=g) + 2.0
    for _ in range(200):
        x = dyn.step(x, torch.zeros(1), **L96_PARAMS)
    truth = torch.zeros(T, STATE_DIM)
    truth[0] = x[0]
    for t in range(1, T):
        x = dyn.step(x, torch.zeros(1), **L96_PARAMS)
        truth[t] = x[0]
    idx = op.indices
    obs = torch.zeros(T, len(idx))
    mask = torch.zeros(T, dtype=torch.bool)
    for t in range(every, T, every):
        obs[t] = truth[t, idx] + 0.5 ** 0.5 * torch.randn(len(idx), generator=g)
        mask[t] = True
        if nan_channels and (t // every) % 2 == 0:
            obs[t, (t // every) % len(idx)] = float("nan")
    return truth, obs, mask, torch.zeros(T)


def _l96(cls, **kw):
    return cls(N_ensemble=10, NO=NO, J=J, dt=DT, dynamics=_l96_dynamics(),
               obs_operator=_l96_obs_operator(), **kw)


def _run(method, window, seed: int = 1234):
    truth, obs, mask, forcing = window
    torch.manual_seed(seed)
    return method.assimilate(obs, mask, forcing, true_state=truth, **L96_PARAMS)


def _run_batch(method, windows, seed: int = 1234):
    truth = torch.stack([w[0] for w in windows])
    obs = torch.stack([w[1] for w in windows])
    mask = torch.stack([w[2] for w in windows])
    forcing = torch.stack([w[3] for w in windows])
    torch.manual_seed(seed)
    return method.assimilate_batch(obs, mask, forcing, true_state=truth, **L96_PARAMS)


def _rmse(result, truth) -> float:
    return float(np.sqrt(np.mean((result.trajectory - truth.numpy()) ** 2)))


# 1 -------------------------------------------------------------------------
def test_recording_hook_leaves_filter_unchanged():
    window = _l96_window()
    plain = _run(_l96(ETKF, inflation=1.3), window)
    hooked_etkf = _l96(ETKF, inflation=1.3)
    hooked_etkf._record_transforms = True
    hooked = _run(hooked_etkf, window)
    assert np.array_equal(plain.trajectory, hooked.trajectory)
    assert np.array_equal(plain.ensemble, hooked.ensemble)
    assert len(hooked_etkf._transforms[0]) == int(window[2].sum())

    windows = [_l96_window(seed=s) for s in (0, 1)]
    plain_b = _run_batch(_l96(ETKF, inflation=1.3), windows)
    hooked_b_etkf = _l96(ETKF, inflation=1.3)
    hooked_b_etkf._record_transforms = True
    hooked_b = _run_batch(hooked_b_etkf, windows)
    for a, b in zip(plain_b, hooked_b):
        assert np.array_equal(a.trajectory, b.trajectory)


# 2 -------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["correct", "none"])
def test_lag_zero_is_the_filter(mode):
    window = _l96_window()
    filt = _run(_l96(ETKF, inflation=1.3), window)
    smooth = _run(_l96(ETKS, inflation=1.3, lag=0, retro_inflation=mode), window)
    np.testing.assert_allclose(smooth.trajectory, filt.trajectory, atol=1e-6)
    np.testing.assert_array_equal(smooth.ensemble, filt.ensemble)


# 3 -------------------------------------------------------------------------
def test_smoother_equals_filter_at_and_after_last_analysis():
    window = _l96_window()
    last = int(torch.nonzero(window[2]).max())
    filt = _run(_l96(ETKF, inflation=1.3), window)
    smooth = _run(_l96(ETKS, inflation=1.3), window)
    np.testing.assert_array_equal(smooth.ensemble[:, last:], filt.ensemble[:, last:])
    assert not np.allclose(smooth.trajectory[:last], filt.trajectory[:last])


# 4 -------------------------------------------------------------------------
class _LinearDynamics(DynamicsBase):
    def __init__(self, F: torch.Tensor):
        super().__init__()
        self.state_dim = F.shape[0]
        self.F = F

    def step(self, state, forcing, **kwargs):
        return state @ self.F.T


def _kalman_rts(m0, P0, F, H, R, obs, mask):
    T = len(mask)
    D = len(m0)
    mf, Pf = np.zeros((T, D)), np.zeros((T, D, D))
    ma, Pa = np.zeros((T, D)), np.zeros((T, D, D))
    ma[0], Pa[0] = m0, P0
    for t in range(1, T):
        mf[t], Pf[t] = F @ ma[t - 1], F @ Pa[t - 1] @ F.T
        ma[t], Pa[t] = mf[t], Pf[t]
        if mask[t]:
            S = H @ Pf[t] @ H.T + R
            K = Pf[t] @ H.T @ np.linalg.inv(S)
            ma[t] = mf[t] + K @ (obs[t] - H @ mf[t])
            Pa[t] = (np.eye(D) - K @ H) @ Pf[t]
    ms, Ps = ma.copy(), Pa.copy()
    for t in range(T - 2, -1, -1):
        Jt = Pa[t] @ F.T @ np.linalg.inv(Pf[t + 1])
        ms[t] = ma[t] + Jt @ (ms[t + 1] - mf[t + 1])
        Ps[t] = Pa[t] + Jt @ (Ps[t + 1] - Pf[t + 1]) @ Jt.T
    return ms, Ps


def test_linear_gaussian_matches_exact_rts_smoother():
    """lambda = 1 and N - 1 >= D: the ETKF is the Kalman filter of its initial
    sample moments and the ETKS is the RTS smoother, exactly up to float32.
    od = 2 < N exercises the null-space term of the #291 transform."""
    D, N, T, R_var = 4, 40, 16, 0.3
    g = torch.Generator().manual_seed(3)
    Q, _ = torch.linalg.qr(torch.randn(D, D, generator=g, dtype=torch.float64))
    F = (1.02 * Q).float()
    idx = [0, 2]
    H = np.eye(D)[idx]
    init = torch.randn(N, D, generator=g) * torch.tensor([1.0, 0.7, 1.3, 0.5]) + 0.3
    mask = torch.zeros(T, dtype=torch.bool)
    mask[3::3] = True
    obs = torch.randn(T, len(idx), generator=g)
    etks = ETKS(N_ensemble=N, R_var=R_var, inflation=1.0, dt=1.0, dynamics=_LinearDynamics(F),
                obs_operator=ObsOperator(D, idx), init_ensemble=init)
    res = etks.assimilate(obs, mask, torch.zeros(T), true_state=torch.zeros(T, D))
    x0 = init.double().numpy()
    ms, Ps = _kalman_rts(x0.mean(0), np.cov(x0.T), F.double().numpy(), H, R_var * np.eye(len(idx)),
                         obs.double().numpy(), mask.numpy())
    np.testing.assert_allclose(res.trajectory, ms, atol=2e-4)
    ens = res.ensemble.astype(np.float64)
    cov = np.einsum("ntd,nte->tde", ens - ens.mean(0), ens - ens.mean(0)) / (N - 1)
    np.testing.assert_allclose(cov, Ps, atol=2e-4)


# 5 -------------------------------------------------------------------------
def test_assimilate_and_assimilate_batch_agree():
    windows = [_l96_window(seed=s) for s in (0, 1)]
    init = torch.randn(10, STATE_DIM, generator=torch.Generator().manual_seed(9)) + windows[0][0][0]
    single = _run(_l96(ETKS, inflation=1.3, init_ensemble=init), windows[0])
    batch = _run_batch(_l96(ETKS, inflation=1.3, init_ensemble=init), windows)
    np.testing.assert_allclose(batch[0].trajectory, single.trajectory, atol=1e-4)
    np.testing.assert_allclose(batch[0].ensemble_variance, single.ensemble_variance, atol=1e-4)


# 6 -------------------------------------------------------------------------
def test_missing_obs_channels_run_and_smoother_beats_filter():
    windows = [_l96_window(seed=s, nan_channels=True) for s in (0, 1, 2)]
    filt = _run_batch(_l96(ETKF, inflation=1.3), windows)
    etks = _l96(ETKS, inflation=1.3)
    smooth = _run_batch(etks, windows)
    assert etks.unsmoothed_windows == []
    for f, s, w in zip(filt, smooth, windows):
        assert np.isfinite(s.ensemble).all()
        assert _rmse(s, w[0]) < _rmse(f, w[0])


# 7 -------------------------------------------------------------------------
def test_correct_reduces_to_none_without_inflation():
    window = _l96_window()
    a = _run(_l96(ETKS, inflation=1.0, lag=2, retro_inflation="correct"), window)
    b = _run(_l96(ETKS, inflation=1.0, lag=2, retro_inflation="none"), window)
    np.testing.assert_allclose(a.trajectory, b.trajectory, atol=1e-5)


# 8 -------------------------------------------------------------------------
def _analysis_transform(Xf: torch.Tensor, H: torch.Tensor, y: torch.Tensor, R: float):
    N1 = Xf.shape[0] - 1
    A = Xf - Xf.mean(0)
    HA = A @ H.T
    U, s, _ = torch.linalg.svd(HA / R ** 0.5, full_matrices=False)
    d = s ** 2 + N1
    Tmat = _etkf_sqrt_transform(U, d, N1, torch.tensor(float(N1), dtype=Xf.dtype))
    w = ((y - Xf.mean(0) @ H.T) / R) @ HA.T @ (U @ torch.diag(1 / d) @ U.T)
    return w, Tmat, HA


@pytest.mark.parametrize("m", [1, 3])
def test_correct_factor_is_kalman_with_scaled_cross_covariance(m):
    g = torch.Generator().manual_seed(m)
    N, Ds, Dj, od, R, lam = 12, 7, 9, 4, 0.3, 2.0
    c = lam ** -m
    Xs = torch.randn(N, Ds, generator=g, dtype=torch.float64)
    Xj = torch.randn(N, Dj, generator=g, dtype=torch.float64) + Xs[:, :1] @ torch.randn(1, Dj, generator=g, dtype=torch.float64)
    H = torch.randn(od, Dj, generator=g, dtype=torch.float64)
    y = torch.randn(od, generator=g, dtype=torch.float64)
    w, Tmat, HA = _analysis_transform(Xj, H, y, R)
    As = Xs - Xs.mean(0)
    Cross = c * As.T @ HA / (N - 1)
    S = HA.T @ HA / (N - 1) + R * torch.eye(od, dtype=torch.float64)
    K = Cross @ torch.linalg.inv(S)
    Xs_new = _etks_factor(w, Tmat, c) @ Xs
    torch.testing.assert_close(Xs_new.mean(0) - Xs.mean(0), K @ (y - Xj.mean(0) @ H.T))
    An = Xs_new - Xs_new.mean(0)
    torch.testing.assert_close(An.T @ An / (N - 1), As.T @ As / (N - 1) - K @ Cross.T)


def test_factor_limits():
    g = torch.Generator().manual_seed(0)
    Xf = torch.randn(12, 9, generator=g, dtype=torch.float64)
    H = torch.randn(4, 9, generator=g, dtype=torch.float64)
    y = torch.randn(4, generator=g, dtype=torch.float64)
    w, Tmat, _ = _analysis_transform(Xf, H, y, 0.3)
    torch.testing.assert_close(_etks_factor(w, Tmat, 0.0), torch.eye(12, dtype=torch.float64))
    analysis = Xf.mean(0) + w @ (Xf - Xf.mean(0)) + Tmat @ (Xf - Xf.mean(0))
    torch.testing.assert_close(_etks_factor(w, Tmat, 1.0) @ Xf, analysis)


# 9 -------------------------------------------------------------------------
def test_correct_is_flat_in_lag():
    window = _l96_window(T=150)
    full = _rmse(_run(_l96(ETKS, inflation=1.5), window), window[0])
    lag4 = _rmse(_run(_l96(ETKS, inflation=1.5, lag=4), window), window[0])
    assert abs(full - lag4) < 0.02 * full


def test_argument_validation():
    with pytest.raises(ValueError, match="lag 2"):
        _l96(ETKS, retro_inflation="none")
    with pytest.raises(ValueError, match="lag 2"):
        _l96(ETKS, retro_inflation="none", lag=4)
    with pytest.raises(ValueError, match="retro_inflation"):
        _l96(ETKS, retro_inflation="inflate")
    with pytest.raises(ValueError, match="lag"):
        _l96(ETKS, lag=-1)
    with pytest.raises(NotImplementedError):
        _l96(ETKS, loc_radius=2.0)
    with pytest.raises(NotImplementedError):
        _l96(ETKS, etkf_additive=0.1)
    _l96(ETKS, retro_inflation="none", lag=1)


def test_nan_guarded_window_is_left_unsmoothed():
    window = _l96_window()
    etks = _l96(ETKS, inflation=1.3)
    filt = _run(_l96(ETKF, inflation=1.3), window)
    torch.manual_seed(1234)
    result = etks.assimilate(window[1], window[2], window[3], true_state=window[0], **L96_PARAMS)
    etks._nan_windows = [True]
    unsmoothed = etks._smooth_result(filt, etks._transforms[0], True, window[0].numpy(), True)
    assert unsmoothed is filt
    assert not np.allclose(result.trajectory, filt.trajectory)


# 10 ------------------------------------------------------------------------
P1_CACHE = Path("/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt")


@pytest.mark.slow
@pytest.mark.skipif(not P1_CACHE.exists(), reason="needs the P1 L96 test cache")
def test_reproduces_prototype_on_p1_windows():
    """docs/results/l96_etks_prototype.md, S0 windows 0-19: per-window-RMS RMSE
    0.791 (ETKF) and 0.639 (ETKS correct, L = 2), seed 0 on CUDA. Loose tolerance:
    another device draws a different initial ensemble."""
    from data.lorenz96 import Lorenz96Config
    from evaluation.run_l96 import _build_eval_kwargs, _method_truth, _per_window_params, make_obs_j_indices

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_idx = make_obs_j_indices(8, 4, 2)
    ds = torch.load(P1_CACHE, weights_only=False)["test_s0"]
    cfg = Lorenz96Config(param_bias=0.0, forcing_state_bias=0.0, T_max=3.0, seed=123,
                         obs_interval=100, obs_var_indices=obs_idx)
    cfg.randomize = {"fast_weights": {"randomized": True}}
    batch = [ds[i] for i in range(20)]
    etkf = ETKF(dt=0.001, device=dev, coupling_exponent=1.6, dynamics=Lorenz96Dynamics(dt=0.001, coupling_exponent=1.6),
                obs_operator=ObsOperator(40, obs_idx), NO=8, J=4, inflation=1.5)
    etkf._record_transforms = True
    etkf.store_ensemble = True
    truth = torch.stack([w["true_state"] for w in batch])
    kw = _build_eval_kwargs([_per_window_params(w, cfg, da_J=4) for w in batch], dev)
    torch.manual_seed(0)
    res = etkf.assimilate_batch(torch.stack([w["obs"] for w in batch]).to(dev),
                                torch.stack([w["obs_mask"] for w in batch]).to(dev),
                                torch.stack([w["forcing_corrupted"] for w in batch]).to(dev),
                                _method_truth(truth, etkf, obs_idx), **kw)
    ref = truth[..., obs_idx].numpy()
    rf, rs = [], []
    for b, r in enumerate(res):
        smoothed = _etks_smooth(r.ensemble, etkf._transforms[b], 1.5, 2, "correct")
        rf.append(np.sqrt(np.mean((r.ensemble[..., obs_idx].mean(0) - ref[b]) ** 2)))
        rs.append(np.sqrt(np.mean((smoothed[..., obs_idx].mean(0) - ref[b]) ** 2)))
    assert np.mean(rf) == pytest.approx(0.791, rel=0.03)
    assert np.mean(rs) == pytest.approx(0.639, rel=0.03)
    assert all(s < f for s, f in zip(rs, rf))
