"""Sampling-time blend of a trained VanillaCFM and a trained PredictStateCFM.

Both families integrate the same probability-flow ODE from x_0 ~ N(0, sigma^2) on the same
Euler grid; they differ only in how the velocity is parameterized -- VanillaCFM returns
v = E[x1 - x0 | x_tau, y] directly, PredictStateCFM returns D = E[x1 | x_tau, y], with
v = (D - x_tau) / (1 - tau). Their training losses are the same velocity regression with
weights 1 (VanillaCFM) and (1 - tau)^2 (PredictStateCFM). The blend integrates

    v(tau) = lam(tau) * v_PSC + (1 - lam(tau)) * v_VC

with a tau-varying weight lam (the weight on PredictStateCFM). lam = 1 and lam = 0 skip the
unused network, so they reproduce PredictStateCFM / VanillaCFM sampling exactly.
"""
from __future__ import annotations

from typing import Callable, Optional

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

    def __init__(self, vanilla: VanillaCFM, predict_state: PredictStateCFM, schedule: str):
        nn.Module.__init__(self)
        if isinstance(vanilla, PredictStateCFM) or not isinstance(predict_state, PredictStateCFM):
            raise TypeError("need a VanillaCFM and a PredictStateCFM")
        if vanilla.sigma_prior != predict_state.sigma_prior or vanilla.state_dim != predict_state.state_dim:
            raise ValueError("the two flows must share sigma_prior and state_dim")
        self.vanilla = vanilla
        self.predict_state = predict_state
        self.schedule = schedule
        self.lam = parse_blend_schedule(schedule)
        self.sigma_prior = vanilla.sigma_prior
        self.state_dim = vanilla.state_dim
        self.N_outer = vanilla.N_outer
        self.train_tau_0_only = False

    def velocity(self, x: torch.Tensor, batch, tau: torch.Tensor, lam: float) -> torch.Tensor:
        v = torch.zeros_like(x)
        if lam > 0:
            one_minus = (1.0 - tau.clamp(max=0.999)).view(-1, 1, 1)
            v = v + lam * (self.predict_state.forward(x, batch, tau) - x) / one_minus
        if lam < 1:
            v = v + (1.0 - lam) * self.vanilla.forward(x, batch, tau)
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
