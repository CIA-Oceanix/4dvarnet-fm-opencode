"""SDA as published (Rozet & Louppe 2023), shared by the QG and L96 evaluations.

On our linear flow x_tau = (1 - tau) x0 + tau x1, x0 ~ N(0, sigma_p^2), the likelihood is
p(y | x_tau) ~ N(y | H x1_hat, R + gamma (sigma_tau / mu_tau)^2) with sigma_tau / mu_tau =
(1 - tau) sigma_p / tau, so the guidance fades where x1_hat is still mostly noise. Its score is
added to the prior's (the velocity gets + (1 - tau) sigma_p^2 / tau * grad log p(y | x_tau)),
and `n_corrector` Langevin steps (step lmc_tau / mean(score^2)) follow every Euler step.
Members start from pure noise.
"""
from __future__ import annotations

import torch


def published_sda_sample(model: torch.nn.Module, batch, obs: torch.Tensor, obs_mask: torch.Tensor,
                         obs_var: float | torch.Tensor, n_members: int, n_steps: int,
                         gamma: float = 0.01, n_corrector: int = 1, lmc_tau: float = 0.1,
                         tau_floor: float = 1e-3) -> torch.Tensor:
    """(n_members, *obs.shape) samples of the unconditional flow `model(x, batch, tau)` guided
    by `obs` (NaN = unobserved) where `obs_mask` (broadcastable to `obs`) is set. `obs_var` is
    the obs-error variance in the model's units: a scalar or a per-channel tensor."""
    y = torch.nan_to_num(obs, nan=0.0)
    mask = obs_mask.to(y.dtype) * torch.isfinite(obs).to(y.dtype)
    sigma_p = float(model.sigma_prior)
    dt = 1.0 / n_steps

    def velocity_and_loglik_grad(x: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            v = model(x, batch, torch.full((x.shape[0],), tau, device=x.device))
            x1_hat = x + (1.0 - tau) * v
            ratio = (1.0 - tau) * sigma_p / max(tau, tau_floor)
            var = obs_var + gamma * ratio ** 2
            sq = (x1_hat - y) ** 2 * mask
            loglik = -0.5 * ((sq / var).sum() if torch.is_tensor(var) else sq.sum() / var)
            grad = torch.autograd.grad(loglik, x)[0]
        return v.detach(), grad

    def one_member() -> torch.Tensor:
        x = torch.randn_like(y) * sigma_p
        for k in range(n_steps):
            tau = k * dt
            v, g = velocity_and_loglik_grad(x, tau)
            x = x + dt * (v + (1.0 - tau) * sigma_p ** 2 / max(tau, tau_floor) * g)
            tau_next = (k + 1) * dt
            if tau_next >= 1.0:
                continue
            for _ in range(n_corrector):
                v, g = velocity_and_loglik_grad(x, tau_next)
                score = -(x - tau_next * v) / ((1.0 - tau_next) * sigma_p ** 2) + g
                delta = lmc_tau / score.flatten(1).square().mean(1).clamp_min(1e-12)
                delta = delta.view(-1, *([1] * (x.dim() - 1)))
                x = x + delta * score + (2.0 * delta).sqrt() * torch.randn_like(x)
        return x.detach()

    return torch.stack([one_member() for _ in range(n_members)])


def nfe_per_member(n_steps: int, n_corrector: int) -> int:
    """Network passes per member (each forward also backpropagated once)."""
    return n_steps + max(n_steps - 1, 0) * n_corrector
