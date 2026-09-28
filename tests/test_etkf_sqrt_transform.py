"""The ETKF square-root transform must keep the anomaly directions the
observations do not constrain.

Regression for the thin-SVD bug: with fewer observations than members
(od < N, e.g. L96 24 obs vs N = 30, L63 <= 3 obs vs N = 30) the transform was
built from the r = od left singular vectors only, which zeroed N - 1 - od
anomaly directions at every analysis.
"""
import pytest
import torch

from evaluation.baselines import ETKF, ObsOperator, _etkf_sqrt_transform
from models.lorenz96_dynamics import Lorenz96Dynamics

N, D = 12, 10
N1 = N - 1


def _anomalies(od: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(N, D, generator=g, dtype=torch.float64)
    A = X - X.mean(0)
    HA = A @ torch.randn(D, od, generator=g, dtype=torch.float64)
    return A, HA - HA.mean(0)


def _exact_transform(Y: torch.Tensor, ridge_term: float = 0.0) -> torch.Tensor:
    ev, V = torch.linalg.eigh(N1 * torch.eye(N, dtype=Y.dtype) + Y @ Y.T
                              + ridge_term * torch.eye(N, dtype=Y.dtype))
    return V @ torch.diag(torch.sqrt(N1 / ev)) @ V.T


def _code_transform(Y: torch.Tensor, etkf_ridge: float = 0.0) -> torch.Tensor:
    U, s, _ = torch.linalg.svd(Y, full_matrices=False)
    s2 = s ** 2
    d = s2 + N1 + etkf_ridge * s2.max()
    return _etkf_sqrt_transform(U, d, N1, N1 + etkf_ridge * s2.max())


@pytest.mark.parametrize("od", [1, 4, N - 1, N, 2 * N])
@pytest.mark.parametrize("etkf_ridge", [0.0, 0.1])
def test_transform_matches_exact_inverse_sqrt(od, etkf_ridge):
    _, Y = _anomalies(od)
    ridge_term = etkf_ridge * float(torch.linalg.svdvals(Y).max() ** 2)
    torch.testing.assert_close(_code_transform(Y, etkf_ridge),
                               _exact_transform(Y, ridge_term))


@pytest.mark.parametrize("od", [1, 4, N - 1, 2 * N])
def test_analysis_covariance_is_kalman(od):
    """(T A)ᵀ(T A)/N1 equals the Kalman analysis covariance (I - K H) P^f."""
    A, Y = _anomalies(od)
    Aa = _code_transform(Y) @ A
    Pf = A.T @ A / N1
    PfHt = A.T @ Y / N1
    K = PfHt @ torch.linalg.inv(Y.T @ Y / N1 + torch.eye(od, dtype=Y.dtype))
    torch.testing.assert_close(Aa.T @ Aa / N1, Pf - K @ PfHt.T)


def test_analysis_keeps_full_ensemble_rank():
    """End to end through ETKF.assimilate: 4 observed channels, 12 members.
    The thin-SVD transform left rank 4 after the first analysis."""
    NO, J, T = 2, 4, 3
    state_dim = NO + NO * J
    obs_op = ObsOperator(state_dim, list(range(NO)) + [NO + k * J for k in range(NO)])
    dyn = Lorenz96Dynamics(dt=0.01, NO=NO, J=J, h=1.0, coupling_exponent=1.6, clip_range=50.0)
    torch.manual_seed(0)
    truth = torch.randn(T, state_dim) * 0.5
    obs = truth[:, obs_op.indices] + 0.1 * torch.randn(T, len(obs_op.indices))
    mask = torch.tensor([False, True, False])
    etkf = ETKF(N_ensemble=N, inflation=1.0, NO=NO, J=J, dt=0.01, dynamics=dyn,
                obs_operator=obs_op)
    ens = torch.as_tensor(etkf.assimilate(obs, mask, torch.zeros(T), true_state=truth,
                                          F=8.0, c1=1.0, h=1.0, hx=1.0, eps=0.1).ensemble)
    anomalies = ens[:, 1] - ens[:, 1].mean(0)
    assert torch.linalg.matrix_rank(anomalies, rtol=1e-4).item() == min(N1, state_dim)
