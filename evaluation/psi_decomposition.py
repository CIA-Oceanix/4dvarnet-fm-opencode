"""Psi_mean / Psi_G / Psi_NG read-out of a trained CFM posterior-mean operator.

With the linear interpolant ``x_tau = (1-tau)*x0 + tau*x1``, ``x0 ~ N(0, sigma0^2 I)``:

    Psi(x_tau,y,tau) = E[x1|x_tau,y] = Psi_mean + Psi_G + Psi_NG
      Psi_mean = m(y) = E[x1|y]
      Psi_G    = K_tau (x_tau - tau*m),  K_tau = tau*P[(1-tau)^2 sigma0^2 I + tau^2 P]^-1
      Psi_NG   = residual, orthogonal to span{m, x_tau} by construction

``Psi_NG == 0`` iff ``p(x1|y)`` is Gaussian with y-independent covariance. It also
vanishes at both endpoints for any posterior: at ``tau=0``, ``x0`` is independent of
``x1`` so ``Psi = m`` and ``K_0 = 0``; at ``tau=1``, ``Psi = x1`` and ``K_1 = I``.
Testing those two boundary conditions requires a **tau-independent** normalization
(``||Psi_NG|| / ||x1||``, or the raw per-element RMS) -- the velocity-relative ratio
``||Psi_NG|| / ||Psi - x_tau||`` carries the interpolant's ``1/(1-tau)`` and is a 0/0
at ``tau=1``, so it cannot decide the endpoint either way.

Shared by ``reports/l96/probe_cfm_affine_decomposition.py`` and
``reports/l96/probe_psi_decomposition_vs_density.py``.
"""
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from models.vanilla_cfm import JointCFM, TweedieCFM, VanillaCFM

COMPONENT_KEYS = ("mm", "mx", "xx", "mmu", "xmu", "mumu")

# Models whose forward(x_tau, batch, tau) returns mu = E[x1|x_tau,y] directly.
# Imported lazily inside psi_of's guard to avoid a circular import at module load.
def _mu_predicting_types():
    from models.fourdvarnet import FourDVarNetPredictStateCFM
    from models.vanilla_cfm import PredictStateCFM
    types = (PredictStateCFM, FourDVarNetPredictStateCFM)
    try:
        from models.decomposed_cfm import FourDVarNetDecomposedCFM
    except ImportError:
        return types
    return types + (FourDVarNetDecomposedCFM,)


def solve2(S_mm: float, S_mx: float, S_xx: float,
           S_mmu: float, S_xmu: float) -> Tuple[float, float]:
    """Least-squares ``(A, B)`` minimizing ``||mu - A*m - B*x_tau||^2``."""
    lhs = np.array([[S_mm, S_mx], [S_mx, S_xx]], dtype=np.float64)
    rhs = np.array([S_mmu, S_xmu], dtype=np.float64)
    a, b = np.linalg.solve(lhs, rhs)
    return float(a), float(b)


def psi_of(model, x_tau: torch.Tensor, batch, tau_scalar: float,
           tau_vec: torch.Tensor, mean_cache: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Assemble ``Psi(x_tau, y, tau) = E[x1|x_tau,y]`` for either interface.

    ``PredictStateCFM``/``FourDVarNetPredictStateCFM`` return ``mu`` directly.
    ``TweedieCFM`` is two-stage: the residual interpolant re-centers the standard
    one, ``r_tau = x_tau - tau*mean``, since
    ``r_tau = (1-tau)x0 + tau*(x1-mean)``. Stage 2 predicts
    ``v = E[(x1-mean) - x0 | r_tau, obs, mean]``, so the ``x1_hat`` identity in
    residual space gives ``Psi = mean + r_tau + (1-tau)*v``.
    """
    if isinstance(model, TweedieCFM):
        mean = mean_cache if mean_cache is not None else model.estimate_mean(batch.obs)
        r_tau = x_tau - tau_scalar * mean
        v = model.forward(r_tau, batch.obs, mean, tau_vec)
        return mean + r_tau + (1.0 - tau_scalar) * v
    if isinstance(model, VanillaCFM) and not isinstance(model, JointCFM):
        # VanillaCFM predicts the VELOCITY E[x1-x0|x_tau,y], not the mean, but its
        # forward signature is identical to PredictStateCFM's -- so without this
        # branch the fallback below would read a velocity as if it were mu and
        # return silently wrong numbers. Convert with the x1_hat identity:
        # x1 = x_tau + (1-tau) v, so Psi = E[x1|x_tau,y] = x_tau + (1-tau) v.
        return x_tau + (1.0 - tau_scalar) * model.forward(x_tau, batch, tau_vec)
    if not (isinstance(model, _mu_predicting_types())
            or getattr(model, "returns_mu", False)):
        raise TypeError(
            f"psi_of does not know what {type(model).__name__}'s forward returns. "
            "Either add an explicit branch, or set `returns_mu = True` on the "
            "model to opt into the mu-predicting path. The fallback is opt-in "
            "because VanillaCFM shares PredictStateCFM's exact forward signature "
            "while returning a velocity -- read as a mean, that fails silently.")
    return model.forward(x_tau, batch, tau_vec)


def psi_mean_estimate(model, batch, x1: torch.Tensor, sigma: float,
                      m_draws: int) -> Tuple[torch.Tensor, float]:
    """``m(y) = E[x1|y]``, plus the summed across-draw variance of the estimate.

    ``TweedieCFM`` exposes ``m`` natively (``estimate_mean``), so the estimate is
    exact and the reported dispersion is 0. Every other interface is sampled at
    ``tau=0``, where ``x_tau = x0`` is independent of ``x1`` and therefore
    ``E[x1|x_tau,y] = E[x1|y]`` exactly; averaging ``m_draws`` draws leaves a
    Monte-Carlo error whose size ``mc_floor`` converts into a residual floor.
    """
    if isinstance(model, TweedieCFM):
        return model.estimate_mean(batch.obs), 0.0
    tau0 = torch.zeros(x1.shape[0], device=x1.device)
    draws = torch.stack([psi_of(model, torch.randn_like(x1) * sigma, batch, 0.0, tau0)
                         for _ in range(m_draws)], 0)
    return draws.mean(0), float(draws.var(0, unbiased=False).sum())


def mc_floor(dispersion: float, m_draws: int) -> float:
    """Residual floor implied by estimating ``m(y)`` from ``m_draws`` draws.

    At ``tau=0`` the measured residual is ``mu(x0,0,y) - m_hat(y)`` with ``x0``
    drawn independently of the ``m_hat`` draws, so a model whose ``Psi_NG(0)``
    is exactly 0 still reports ``sigma_d * sqrt(1 + 1/K)``, where ``sigma_d`` is
    the per-draw dispersion. ``dispersion`` comes from a biased (``/K``)
    variance, hence the extra ``sqrt(K/(K-1))``; the two combine to
    ``sqrt((K+1)/(K-1))``. Returns 0 for an exact estimator (``dispersion=0``).
    """
    if dispersion <= 0.0 or m_draws < 2:
        return 0.0
    return float(dispersion * np.sqrt((m_draws + 1) / (m_draws - 1)))


def gamma_closed_form(tau: float, p_hat: float, s: float) -> float:
    """Isotropic gain ``gamma(tau) = tau*p / ((1-tau)^2 s + tau^2 p)``."""
    denom = (1.0 - tau) ** 2 * s + tau ** 2 * p_hat
    return float(tau * p_hat / denom) if denom > 0 else float("nan")


def new_accumulator() -> Dict[str, float]:
    return {k: 0.0 for k in COMPONENT_KEYS + ("vv", "n")}


def accumulate(acc: Dict[str, float], m: torch.Tensor, x_tau: torch.Tensor,
               mu: torch.Tensor) -> None:
    """Add one batch's second-moment sums to ``acc`` (in place)."""
    acc["mm"] += float((m * m).sum())
    acc["mx"] += float((m * x_tau).sum())
    acc["xx"] += float((x_tau * x_tau).sum())
    acc["mmu"] += float((m * mu).sum())
    acc["xmu"] += float((x_tau * mu).sum())
    acc["mumu"] += float((mu * mu).sum())
    acc["vv"] += float(((mu - x_tau) ** 2).sum())
    acc["n"] += m.numel()


def components(acc: Dict[str, float], tau: float, x1_sq: float,
               p_hat: float, s: float) -> Dict[str, float]:
    """Per-element RMS of each component, plus the four normalizations.

    ``Psi_G`` uses the constrained form ``A = 1 - tau*K`` (verified to hold to
    <0.015 -- see ``docs/results/cfm_affine_velocity_decomposition.md``), so
    ``Psi_G = K*(x_tau - tau*m)`` and ``||Psi_G||^2 = K^2 * ||x_tau - tau*m||^2``.

    ``ng_over_x1`` and ``psi_NG`` are the tau-independent normalizations -- the
    ones that can decide the endpoint conditions. ``ng_over_v`` is the
    velocity-relative ratio, kept for continuity with the earlier tables and
    **not** a valid endpoint test (see the module docstring).
    """
    n = acc["n"]
    A_fit, K = solve2(acc["mm"], acc["mx"], acc["xx"], acc["mmu"], acc["xmu"])
    rss = max(acc["mumu"] - A_fit * acc["mmu"] - K * acc["xmu"], 0.0)
    z_sq = max(acc["xx"] - 2 * tau * acc["mx"] + (tau ** 2) * acc["mm"], 0.0)

    psi_m = float(np.sqrt(acc["mm"] / n))
    psi_G = float(abs(K) * np.sqrt(z_sq / n))
    psi_NG = float(np.sqrt(rss / n))
    psi_tot = float(np.sqrt(acc["mumu"] / n))
    v_norm = float(np.sqrt(acc["vv"] / n))
    x1_rms = float(np.sqrt(x1_sq / n))
    gamma = gamma_closed_form(tau, p_hat, s)

    return dict(
        tau=tau, A=A_fit, K=K, gamma_theory=gamma,
        gain_rel_err=abs(K - gamma) / gamma if gamma > 0 else float("nan"),
        constrained_dev=abs(A_fit - (1.0 - tau * K)),
        psi_mean=psi_m, psi_G=psi_G, psi_NG=psi_NG, psi_total=psi_tot,
        v_norm=v_norm, x1_rms=x1_rms,
        ng_over_x1=psi_NG / x1_rms if x1_rms > 0 else float("nan"),
        ng_over_g=psi_NG / psi_G if psi_G > 0 else float("nan"),
        ng_over_psi=psi_NG / psi_tot if psi_tot > 0 else float("nan"),
        ng_over_v=psi_NG / v_norm if v_norm > 0 else float("nan"),
    )
