"""Sampling-time blend of a trained VanillaCFM and a trained PredictStateCFM.

Both families integrate the same probability-flow ODE from x_0 ~ N(0, sigma^2) on the same
Euler grid; they differ only in how the velocity is parameterized -- VanillaCFM returns
v = E[x1 - x0 | x_tau, y] directly, PredictStateCFM returns D = E[x1 | x_tau, y], with
v = (D - x_tau) / (1 - tau). Their training losses are the same velocity regression with
weights 1 (VanillaCFM) and (1 - tau)^2 (PredictStateCFM). The blend integrates

    v(tau) = lam(tau) * v_PSC + (1 - lam(tau)) * v_VC

with a tau-varying weight lam (the weight on PredictStateCFM). lam = 1 and lam = 0 skip the
unused network, so they reproduce PredictStateCFM / VanillaCFM sampling exactly.

Any two flows can be blended (lam weights the second): two seeds of the same family give the
model-averaging control for a VanillaCFM + PredictStateCFM blend.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from models.vanilla_cfm import DEFAULT_STEP_POWER, PredictStateCFM, VanillaCFM, flow_tau_grid


def parse_blend_schedule(spec: str) -> Callable[[float], float]:
    """lam(tau), the weight on PredictStateCFM:
    const:a (a), switch:t (1 before t, 0 after), power:p ((1-tau)^p), rpower:p (tau^p)."""
    kind, _, arg = spec.partition(":")
    a = float(arg)
    if kind == "const" and 0.0 <= a <= 1.0:
        return lambda tau: a
    if kind == "switch" and 0.0 <= a <= 1.0:
        return lambda tau: 1.0 if tau < a else 0.0
    if kind == "power" and a > 0:
        return lambda tau: (1.0 - tau) ** a
    if kind == "rpower" and a > 0:
        return lambda tau: tau ** a
    raise ValueError(f"unknown blend schedule {spec!r} (const:a, switch:t, power:p, rpower:p)")


class TauBlendedFlow(VanillaCFM):
    """Subclasses VanillaCFM only so the evaluation dispatch treats it as a flow sampler."""

    def __init__(self, first: VanillaCFM, second: VanillaCFM, schedule: str):
        nn.Module.__init__(self)
        if not (isinstance(first, VanillaCFM) or isinstance(first, PredictStateCFM)) or \
                not (isinstance(second, VanillaCFM) or isinstance(second, PredictStateCFM)):
            raise TypeError("both flows must be VanillaCFM or PredictStateCFM")
        if first.sigma_prior != second.sigma_prior or first.state_dim != second.state_dim:
            raise ValueError("the two flows must share sigma_prior and state_dim")
        self.first = first
        self.second = second
        self.schedule = schedule
        self.lam = parse_blend_schedule(schedule)
        self.sigma_prior = first.sigma_prior
        self.state_dim = first.state_dim
        self.N_outer = first.N_outer
        self.train_tau_0_only = False

    @staticmethod
    def _velocity(flow: nn.Module, x: torch.Tensor, batch, tau: torch.Tensor) -> torch.Tensor:
        if isinstance(flow, PredictStateCFM):
            return (flow.forward(x, batch, tau) - x) / (1.0 - tau.clamp(max=0.999)).view(-1, 1, 1)
        return flow.forward(x, batch, tau)

    def velocity(self, x: torch.Tensor, batch, tau: torch.Tensor, lam: float) -> torch.Tensor:
        v = torch.zeros_like(x)
        if lam > 0:
            v = v + lam * self._velocity(self.second, x, batch, tau)
        if lam < 1:
            v = v + (1.0 - lam) * self._velocity(self.first, x, batch, tau)
        return v

    def sample(self, batch, N_outer: Optional[int] = None, step_power: Optional[float] = None):
        N_outer = self.N_outer if N_outer is None else N_outer
        step_power = DEFAULT_STEP_POWER if step_power is None else step_power
        B = batch.obs.shape[0]
        x = torch.randn_like(batch.obs) * self.sigma_prior
        taus = flow_tau_grid(N_outer, step_power)
        for t0, t1 in zip(taus[:-1], taus[1:]):
            tau = torch.full((B,), t0, device=x.device)
            x = x + (t1 - t0) * self.velocity(x, batch, tau, self.lam(t0))
        return x


def random_weight_schedule(n_flows: int, seed: int, n_knots: int = 5) -> tuple:
    """Random tau-varying simplex weights: Dirichlet(1) weights at n_knots equispaced tau,
    linearly interpolated (stays on the simplex). Returns (w(tau) -> weights, knot array)."""
    knots = np.random.RandomState(seed).dirichlet(np.ones(n_flows), size=n_knots)
    grid = np.linspace(0.0, 1.0, n_knots)
    return (lambda tau: [float(np.interp(tau, grid, knots[:, i])) for i in range(n_flows)]), knots


class MeanVelocityFlow(VanillaCFM):
    """Weighted average of the velocities of N flows (VanillaCFM and/or PredictStateCFM),
    integrated on one shared trajectory: an N-network flow ensemble at sampling time. Equal
    weights by default; ``weights(tau)`` gives tau-varying ones (see random_weight_schedule)."""

    def __init__(self, flows: list, weights: Optional[Callable[[float], Sequence[float]]] = None):
        nn.Module.__init__(self)
        if len(flows) < 2 or not all(isinstance(f, VanillaCFM) or isinstance(f, PredictStateCFM) for f in flows):
            raise TypeError("need at least two VanillaCFM / PredictStateCFM flows")
        if len({(f.sigma_prior, f.state_dim) for f in flows}) != 1:
            raise ValueError("the flows must share sigma_prior and state_dim")
        self.flows = nn.ModuleList(flows)
        self.weights = weights
        self.sigma_prior = flows[0].sigma_prior
        self.state_dim = flows[0].state_dim
        self.N_outer = flows[0].N_outer
        self.train_tau_0_only = False

    def velocity(self, x: torch.Tensor, batch, tau: torch.Tensor) -> torch.Tensor:
        if self.weights is None:
            return sum(TauBlendedFlow._velocity(f, x, batch, tau) for f in self.flows) / len(self.flows)
        w = self.weights(float(tau[0]))
        return sum(wi * TauBlendedFlow._velocity(f, x, batch, tau) for wi, f in zip(w, self.flows) if wi > 0)

    def sample(self, batch, N_outer: Optional[int] = None, step_power: Optional[float] = None):
        N_outer = self.N_outer if N_outer is None else N_outer
        step_power = DEFAULT_STEP_POWER if step_power is None else step_power
        B = batch.obs.shape[0]
        x = torch.randn_like(batch.obs) * self.sigma_prior
        taus = flow_tau_grid(N_outer, step_power)
        for t0, t1 in zip(taus[:-1], taus[1:]):
            tau = torch.full((B,), t0, device=x.device)
            x = x + (t1 - t0) * self.velocity(x, batch, tau)
        return x
