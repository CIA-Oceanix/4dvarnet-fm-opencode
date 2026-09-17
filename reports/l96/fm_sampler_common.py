"""Shared pieces for the FM-prior conditional samplers (L96).

Centralises the three things that were repeatedly got wrong when each probe
carried its own copy:

* ``prior_gain`` -- the isotropic Gaussian gain of the FM operator.
* ``estimate_prior_gain_variance`` -- the ``P_prior`` that gain is evaluated at.
  It MUST come from an *unguided* pass: ``K_p`` is the gain the network's own
  operator already applies, so measuring it on a guided ensemble (whose spread is
  collapsed by the likelihood term) understates it by ~5x and silently moves any
  calibration ratio built on top of it. The unguided integration is inlined here
  rather than taking a ``guidance`` flag, so there is no flag left to forget.
* ``EnsembleScores`` -- the RMSE/spread accumulator, using the repo's
  ``evaluate_estimates`` convention (mean over dimensions of per-dimension RMSE),
  alongside the pooled RMSE, since the two differ by ~8% on L96.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


def prior_gain(tau: float, p_var: float | torch.Tensor,
               sigma0: float) -> float | torch.Tensor:
    """Isotropic Gaussian gain gamma(tau) = tau p / ((1-tau)^2 sigma0^2 + tau^2 p)."""
    alpha2 = ((1.0 - tau) * sigma0) ** 2
    return tau * p_var / (alpha2 + tau ** 2 * p_var + 1e-12)


@torch.no_grad()
def prior_mean(model, batch, sigma0: float, n_draws: int = 4) -> torch.Tensor:
    """mu_p = E[x_1 | y] read off the operator at tau = 0: Psi(x_0) = x_0 + v(x_0, 0)."""
    B = batch.obs.shape[0]
    tau0 = torch.zeros(B, device=batch.obs.device)
    shape = (B, batch.obs.shape[1], model.state_dim)
    draws = []
    for _ in range(n_draws):
        z = torch.randn(*shape, device=batch.obs.device) * sigma0
        draws.append(z + model.forward(z, batch, tau0))
    return torch.stack(draws).mean(0)


@torch.no_grad()
def estimate_prior_gain_variance(model, batch, n_outer: int, n_members: int,
                                 sigma0: float, seed: int = 0) -> float:
    """P_prior: the variance of the UNGUIDED flow's terminal ensemble.

    No guidance option by design -- see the module docstring.
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


def reliability_target(n_members: int) -> float:
    """spread/RMSE a perfectly reliable N-member ensemble attains (biased variance)."""
    return float(np.sqrt((n_members - 1) / (n_members + 1)))


@dataclass
class EnsembleScores:
    """Accumulates ensemble estimates in physical units and scores them."""

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
        rmse_repo = float(torch.sqrt(((e - t) ** 2).mean(dim=(0, 1))).mean())
        rmse_pooled = float(torch.sqrt(((e - t) ** 2).mean()))
        spread = float(np.sqrt(self.spread_sum / max(self.count, 1)))
        return dict(rmse_repo=rmse_repo, rmse_pooled=rmse_pooled, spread=spread,
                    ratio=spread / rmse_pooled)
