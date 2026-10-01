"""Flow-matching operator score of a forecast distribution against a truth.

On the path x_tau = tau * x1 + (1 - tau) * x0, x0 ~ N(0, I), the operator of a
candidate q is Psi_q(x_tau, tau) = E_q[x1 | x_tau]. For a truth x1* drawn from
the reference posterior p,

    E ||Psi_q(x_tau) - x1*||^2 = E ||Psi_q - Psi_p||^2 + mmse_p(tau),

so the per-tau score ranks candidates by their operator divergence from p up to
a candidate-independent constant. tau = 0 is the squared error of the mean.

All functions score elementwise (univariate blocks) unless a block axis is
given, and assume states already normalised to unit scale per channel.
"""
from __future__ import annotations

import numpy as np


def snr(tau: float) -> float:
    return (tau / (1.0 - tau)) ** 2


def gaussian_score(err: np.ndarray, var: np.ndarray, tau: float) -> np.ndarray:
    """Closed-form E_x0 score of q = N(m, var) at truth m + err (per element)."""
    s = snr(tau)
    return (err ** 2 + s * var ** 2) / (1.0 + s * var) ** 2


def mixture_operator(
    x_tau: np.ndarray, centres: np.ndarray, tau: float, h2: float | np.ndarray = 0.0,
    block_axes: tuple[int, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Psi of an equal-weight mixture of N(centre_i, h2 I); members on the last axis.

    ``x_tau`` has the shape of ``centres`` without the member axis. Weights are
    shared across ``block_axes`` (axes of ``x_tau``), i.e. the block is scored
    jointly. Returns (Psi, effective sample size of the weights).
    """
    c2 = tau ** 2 * h2 + (1.0 - tau) ** 2
    diff = x_tau[..., None] - tau * centres
    logw = -0.5 * diff ** 2 / c2
    if block_axes:
        logw = logw.sum(axis=block_axes, keepdims=True)
    logw = logw - logw.max(axis=-1, keepdims=True)
    w = np.exp(logw)
    w /= w.sum(axis=-1, keepdims=True)
    comp = centres + (tau * h2 / c2) * diff
    psi = (w * comp).sum(axis=-1)
    ess = 1.0 / (w ** 2).sum(axis=-1)
    return psi, ess


def ensemble_score(
    members: np.ndarray, truth: np.ndarray, tau: float, x0: np.ndarray,
    h2: float | np.ndarray = 0.0, block_axes: tuple[int, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Monte-Carlo score of an ensemble (members on the last axis) at one tau.

    ``x0`` has shape ``truth.shape + (K,)``: K noise draws, shared across
    candidates (common random numbers). Returns per-element score averaged over
    the K draws and the mean effective sample size.
    """
    sc = np.zeros(truth.shape, dtype=np.float64)
    ess = np.zeros(truth.shape, dtype=np.float64)
    K = x0.shape[-1]
    for k in range(K):
        x_tau = tau * truth + (1.0 - tau) * x0[..., k]
        psi, e = mixture_operator(x_tau, members, tau, h2, block_axes)
        sc += (psi - truth) ** 2
        ess += np.broadcast_to(e, truth.shape)
    return sc / K, ess / K


def silverman_h2(members: np.ndarray) -> np.ndarray:
    """Per-element Silverman KDE bandwidth squared (members on the last axis)."""
    n = members.shape[-1]
    sd = members.std(axis=-1, ddof=1)
    return (1.06 * sd * n ** (-0.2)) ** 2
