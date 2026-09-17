"""Configurable conditional sampler for flow-matching priors (SDA family).

One integrator, pluggable schemes. Every scheme is a way of editing the velocity
the network returns; the Euler loop, the observation guidance and the member loop
are written once here so that variants cannot drift apart.

Background. Writing the FM operator as ``Psi(x_tau, y) = E[x_1 | x_tau, y]``, the
Tweedie identity gives ``v = (Psi - x_tau) / (1 - tau)``. Under a Gaussian posterior
the operator decomposes as

    Psi = mu_p + K_tau (x_tau - beta_tau mu_p) + Psi_NG

with the isotropic gain ``K_tau = tau P / ((1-tau)^2 sigma_0^2 + tau^2 P)``
(``prior_gain``) and ``Psi_NG`` the non-Gaussian residual. The schemes below are the
edits to that operator that were measured; see
``reports/l96/outputs/l96_fm_sampler_benchmark.md``.

A deterministic estimator ``m_hat(y)`` is itself a flow: its operator is the constant
map ``Psi_det = m_hat``, hence ``v_det = (m_hat - x_tau)/(1-tau)``, which integrates
to ``x_tau = (1-tau) x_0 + tau m_hat`` -- the SDEdit warm-start state. So ``Warm`` and
``Blend`` are two members of one family rather than unrelated tricks.

Numerical determinism. ``sample`` draws exactly one ``torch.randn`` per member, in
member order, before that member's trajectory. Callers seed immediately before the
call. Reordering those draws changes every number while still looking plausible, so
the scheme classes deliberately have no access to the RNG.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

__all__ = [
    "SamplerConfig",
    "SchemeContext",
    "Scheme",
    "Cold",
    "Warm",
    "Blend",
    "Decoupled",
    "sample",
    "prior_gain",
    "prior_mean",
    "estimate_prior_gain_variance",
    "reliability_target",
    "EnsembleScores",
]


# --------------------------------------------------------------------------- #
# Gaussian pieces
# --------------------------------------------------------------------------- #
def prior_gain(tau: float, p_var: float | torch.Tensor,
               sigma0: float) -> float | torch.Tensor:
    """Isotropic gain ``tau p / ((1-tau)^2 sigma_0^2 + tau^2 p)``."""
    alpha2 = ((1.0 - tau) * sigma0) ** 2
    return tau * p_var / (alpha2 + tau ** 2 * p_var + 1e-12)


@torch.no_grad()
def prior_mean(model, batch, sigma0: float, n_draws: int = 4,
               seed: int = 0) -> torch.Tensor:
    """``mu_p = E[x_1|y]``, read off the operator at ``tau = 0``: ``x_0 + v(x_0, 0)``.

    Seeds before drawing. The pre-refactor code did not, so ``mu_p`` came from
    PyTorch's ambient (randomly initialised) generator and varied run to run --
    invisibly, because only ``Decoupled`` reads ``mu_p``, and every other quantity in
    the pipeline was seeded. That made the decoupled rows irreproducible at the
    ~0.4% level (0.6098 / 0.6102 / 0.6113 across three runs of identical code) while
    the blend rows reproduced bit-for-bit, which is exactly the pattern that hides
    such a defect.
    """
    torch.manual_seed(seed)
    B = batch.obs.shape[0]
    tau0 = torch.zeros(B, device=batch.obs.device)
    shape = (B, batch.obs.shape[1], model.state_dim)
    draws = [
        (lambda z: z + model.forward(z, batch, tau0))(
            torch.randn(*shape, device=batch.obs.device) * sigma0)
        for _ in range(n_draws)
    ]
    return torch.stack(draws).mean(0)


@torch.no_grad()
def estimate_prior_gain_variance(model, batch, n_outer: int, n_members: int,
                                 sigma0: float, seed: int = 0) -> float:
    """``P_prior``: terminal ensemble variance of the UNGUIDED flow.

    There is deliberately no ``guidance`` argument. ``K_p`` is the gain the network's
    operator already applies, so it must be measured without the likelihood term;
    measuring it on a guided ensemble understates it by roughly 5x (0.153 vs 0.804 on
    L96) and silently moves any calibration built on top of it. Inlining the unguided
    integration here removes the flag there was to forget.
    """
    B, T, _ = batch.obs.shape
    dt = 1.0 / n_outer
    torch.manual_seed(seed)
    members = []
    for _ in range(n_members):
        x = torch.randn(B, T, model.state_dim, device=batch.obs.device) * sigma0
        for step in range(n_outer):
            tau = torch.full((B,), step / n_outer, device=batch.obs.device)
            x = x + dt * model.forward(x, batch, tau)
        members.append(x)
    return float(torch.stack(members, dim=-1).var(-1, correction=0).mean())


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SamplerConfig:
    """Integration and guidance settings shared by every scheme."""

    n_outer: int = 50
    gamma: float = 1e-2
    members: int = 10
    seed: int = 0


@dataclass
class SchemeContext:
    """Per-batch quantities schemes may read. ``None`` when not needed."""

    sigma0: float
    m_hat: torch.Tensor | None = None
    mu_p: torch.Tensor | None = None
    p_prior: float | None = None
    p_target: float | None = None

    def require(self, scheme_name: str, *names: str) -> None:
        missing = [n for n in names if getattr(self, n) is None]
        if missing:
            raise ValueError(
                f"scheme {scheme_name!r} needs {', '.join(missing)} in SchemeContext")


# --------------------------------------------------------------------------- #
# Schemes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Scheme:
    """Base: start from pure noise, use the network velocity unchanged."""

    @property
    def name(self) -> str:
        return type(self).__name__.lower()

    def check(self, ctx: SchemeContext) -> None:
        return None

    def start_step(self, n_outer: int) -> int:
        return 0

    def initial_state(self, noise: torch.Tensor, n_outer: int,
                      ctx: SchemeContext) -> torch.Tensor:
        return noise

    def velocity(self, v: torch.Tensor, x: torch.Tensor, tau: float,
                 ctx: SchemeContext) -> torch.Tensor:
        return v


@dataclass(frozen=True)
class Cold(Scheme):
    """Pure FM prior with guidance. The reference sampler."""


@dataclass(frozen=True)
class Warm(Scheme):
    """SDEdit warm start: deterministic below ``tau0``, FM flow above.

    The discontinuous member ``lam(tau) = 1{tau < tau0}`` of ``Blend``'s family.
    """

    tau0: float

    @property
    def name(self) -> str:
        return f"warm(tau0={self.tau0:g})"

    def check(self, ctx: SchemeContext) -> None:
        ctx.require(self.name, "m_hat")

    def start_step(self, n_outer: int) -> int:
        return int(round(self.tau0 * n_outer))

    def initial_state(self, noise, n_outer, ctx):
        t0 = self.start_step(n_outer) / n_outer
        return (1.0 - t0) * noise + t0 * ctx.m_hat


@dataclass(frozen=True)
class Blend(Scheme):
    """Continuous blend ``v = (1-lam) v_FM + lam v_det``, ``lam = lam0 (1-tau)^p``.

    ``p >= 1`` is required, not merely advisable: ``v_det`` carries a ``1/(1-tau)``
    factor, so the blended velocity contains ``lam/(1-tau) = lam0 (1-tau)^(p-1)``,
    bounded as ``tau -> 1`` only for ``p >= 1``.
    """

    lam0: float
    p: float

    def __post_init__(self):
        if self.p < 1.0:
            raise ValueError(
                f"Blend needs p >= 1 (got {self.p}): lam/(1-tau) diverges otherwise")
        if not 0.0 <= self.lam0 <= 1.0:
            raise ValueError(f"Blend needs lam0 in [0, 1] (got {self.lam0})")

    @property
    def name(self) -> str:
        return f"blend(lam0={self.lam0:g},p={self.p:g})"

    def check(self, ctx: SchemeContext) -> None:
        if self.lam0 > 0.0:
            ctx.require(self.name, "m_hat")

    def velocity(self, v, x, tau, ctx):
        lam = self.lam0 * (1.0 - tau) ** self.p
        if lam <= 0.0:
            return v
        v_det = (ctx.m_hat - x) / max(1.0 - tau, 1e-6)
        return (1.0 - lam) * v + lam * v_det


@dataclass(frozen=True)
class Decoupled(Scheme):
    """General affine edit -- NEGATIVE RESULT, kept for reproducibility.

    Gives the mean, the gain and the non-Gaussian amplitude independent knobs::

        mu~ = (1-a) mu_p + a m_hat
        v = c v_FM + [mu~ - c mu_p + (K~ - c K_p)(x - beta mu_p) + (c-1) x] / (1-tau)

    ``gain="prior"`` uses ``K_p``; ``gain="target"`` uses ``K`` built from
    ``ctx.p_target``. ``scale_ng`` multiplies the gain and the NG term by ``(1-lam)``.

    ``Decoupled(lam0, p, gain="prior", scale_ng=True)`` is algebraically identical to
    ``Blend(lam0, p)`` -- asserted in ``tests/test_fm_sampler.py``.

    Measured outcome: decoupling is worse than ``Blend`` on BOTH RMSE and spread,
    because ``Psi_NG`` is defined relative to the pair ``(mu_p, K_p)`` and is not
    invariant under a gain change, so re-using the network's ``Psi_NG`` beneath a
    different ``K~`` is inconsistent.
    """

    lam0: float
    p: float
    gain: str = "prior"
    scale_ng: bool = False

    def __post_init__(self):
        if self.gain not in ("prior", "target"):
            raise ValueError(f"gain must be 'prior' or 'target' (got {self.gain!r})")

    @property
    def name(self) -> str:
        return (f"decoupled(lam0={self.lam0:g},p={self.p:g},"
                f"gain={self.gain},scale_ng={self.scale_ng})")

    def check(self, ctx: SchemeContext) -> None:
        ctx.require(self.name, "m_hat", "mu_p", "p_prior")
        if self.gain == "target":
            ctx.require(self.name, "p_target")

    def velocity(self, v, x, tau, ctx):
        lam = self.lam0 * (1.0 - tau) ** self.p
        k_p = prior_gain(tau, ctx.p_prior, ctx.sigma0)
        k_base = k_p if self.gain == "prior" else prior_gain(tau, ctx.p_target, ctx.sigma0)
        c = (1.0 - lam) if self.scale_ng else 1.0
        k_t = c * k_base if self.scale_ng else k_base
        mu_t = (1.0 - lam) * ctx.mu_p + lam * ctx.m_hat
        extra = (mu_t - c * ctx.mu_p + (k_t - c * k_p) * (x - tau * ctx.mu_p)
                 + (c - 1.0) * x)
        return c * v + extra / max(1.0 - tau, 1e-6)


# --------------------------------------------------------------------------- #
# Integrator
# --------------------------------------------------------------------------- #
def sample(model, batch, scheme: Scheme, config: SamplerConfig,
           ctx: SchemeContext, r_vec: torch.Tensor) -> torch.Tensor:
    """Draw ``config.members`` guided samples. Returns ``(B, T, D, N)``.

    Guidance is the SDA variance-weighted form (Rozet & Louppe): the Tweedie estimate
    ``x_hat_1 = x_tau + (1-tau) v`` is scored against the observations with weights
    ``1/(2(R + Gamma alpha^2/beta^2))`` and the state nudged along that gradient.

    ``r_vec`` is the per-channel observation variance in the model's (normalized)
    state units.
    """
    scheme.check(ctx)
    B, T, _ = batch.obs.shape
    device = batch.obs.device
    n_outer = config.n_outer
    dt = 1.0 / n_outer
    y = torch.nan_to_num(batch.obs, nan=0.0)
    tmask = batch.obs_mask.to(y.dtype).unsqueeze(-1)
    step0 = scheme.start_step(n_outer)

    def _one() -> torch.Tensor:
        noise = torch.randn(B, T, model.state_dim, device=device) * ctx.sigma0
        x = scheme.initial_state(noise, n_outer, ctx)
        for step in range(step0, n_outer):
            tau = step / n_outer
            tau_vec = torch.full((B,), tau, device=device)
            beta = max(tau, dt / 2.0)
            alpha2 = ((1.0 - tau) * ctx.sigma0) ** 2
            weight = 1.0 / (2.0 * (r_vec + config.gamma * alpha2 / beta ** 2))
            coef = dt * alpha2 / (beta * max(1.0 - tau, 1e-6))
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                v = scheme.velocity(model.forward(x, batch, tau_vec), x, tau, ctx)
                x_hat = x + (1.0 - tau) * v
                cost = ((((x_hat - y) ** 2) * tmask) * weight).sum()
                grad = torch.autograd.grad(cost, x)[0]
            x = (x + dt * v.detach() - coef * grad).detach()
        return x

    return torch.stack([_one() for _ in range(config.members)], dim=-1)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def reliability_target(n_members: int) -> float:
    """``spread/RMSE`` attained by a perfectly reliable N-member ensemble.

    Biased (``correction=0``) variance convention, matching ``EnsembleScores``.
    """
    return float(np.sqrt((n_members - 1) / (n_members + 1)))


@dataclass
class EnsembleScores:
    """Accumulates ensemble estimates in physical units and scores them.

    ``rmse_repo`` follows the repo's ``evaluate_estimates`` convention -- the mean
    over state dimensions of the per-dimension RMSE -- and is the number comparable
    to the consolidated benchmark. ``rmse_pooled`` pools over all entries instead;
    the two differ by ~8% on L96, so both are reported.
    """

    est: list = field(default_factory=list)
    tru: list = field(default_factory=list)
    spread_sum: float = 0.0
    count: int = 0

    def add(self, members: torch.Tensor, truth: torch.Tensor,
            std: torch.Tensor, mean: torch.Tensor) -> None:
        estimate = members.mean(-1) * std + mean
        var = (members.var(-1, correction=0) * (std ** 2) if members.shape[-1] > 1
               else torch.zeros_like(estimate))
        self.est.append(estimate.cpu())
        self.tru.append(truth.cpu())
        self.spread_sum += float(var.sum())
        self.count += estimate.numel()

    def summary(self) -> dict:
        e = torch.cat(self.est)
        t = torch.cat(self.tru)
        rmse_pooled = float(torch.sqrt(((e - t) ** 2).mean()))
        spread = float(np.sqrt(self.spread_sum / max(self.count, 1)))
        return dict(
            rmse_repo=float(torch.sqrt(((e - t) ** 2).mean(dim=(0, 1))).mean()),
            rmse_pooled=rmse_pooled,
            spread=spread,
            # undefined for a perfect estimate; NaN rather than a division error
            ratio=(spread / rmse_pooled) if rmse_pooled > 0.0 else float("nan"),
        )
