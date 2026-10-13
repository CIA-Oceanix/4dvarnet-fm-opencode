"""Stochastic flow-matching posterior sampler for an unconditional FM prior.

Plan: ``docs/plans/analysis/l96_fdv2_fm_posterior.md`` (sections 1 and 4, agent A1).

Linear interpolant ``x_tau = (1 - tau) x0 + tau x1`` with ``x0 ~ N(0, sigma_p^2 I)``.
``prior.forward(x, batch, tau)`` returns the velocity ``v``; with ``c_tau = (1-tau)^2 sigma_p^2 / tau``:

- state prediction (Tweedie): ``Psi = x + (1 - tau) v``;
- posterior state prediction: ``Psi_y = Psi + w * c_tau * grad_x log p(y | x_tau)``, ``w = guidance_weight``;
- posterior velocity ``v_y = (Psi_y - x) / (1 - tau)``;
- posterior score ``s_y = (tau Psi_y - x) / ((1 - tau)^2 sigma_p^2)``.

Likelihood ``p(y | x_tau)`` (``guidance``), with ``H`` the observed entries (temporal ``obs_mask``,
non-NaN ``y``, optional ``obs_indices`` / ``obs_channel_mask``), ``R = R_norm`` in normalised units
(``R_var / sigma_clim^2`` per channel):

- ``"dps"``: ``N(y; H Psi, R)``, gradient by a vector-Jacobian product through ``Psi``;
- ``"pigdm"``: ``N(y; H Psi, R + k_tau diag(H J_Psi H^T))`` with the covariance treated as a
  constant. ``pigdm_cov="tweedie"`` (default) uses ``k_tau = c_tau``, the exact Tweedie covariance
  ``Cov[x1 | x_tau] = c_tau J_Psi``, so ``Psi_y`` is the exact Kalman update for a Gaussian prior with a
  diagonal Jacobian; ``pigdm_cov="r2"`` uses the plan's ``k_tau = r_tau^2 = sigma^2 / (1 + sigma^2)``,
  ``sigma = sigma_p (1 - tau) / tau`` (unit data variance). ``diag(J_Psi)`` comes from a Hutchinson
  estimate ``E[z * J^T z]`` with Rademacher probes restricted to the observed entries
  (``pigdm_diag="hutchinson"``, ``hutchinson_probes`` probes), or from one vector-Jacobian product per
  entry (``pigdm_diag="exact"``, for tests and tiny problems);
- ``"normalized"``: the fixed-norm step of ``evaluation.sda_sampler.sda_guided_sample``,
  ``x <- x + dtau v - w g / ||g||`` with ``g = grad_x sum(H (Psi - y)^2 / R)`` (norm per member),
  i.e. ``Psi_y = Psi - ((1 - tau) / dtau) w g / ||g||``.

Euler-Maruyama on the grid ``tau_0 < ... < tau_N = 1``:
``x <- x + [v_y + 0.5 g(tau)^2 s_y] dtau + g(tau) sqrt(dtau) xi``, ``g(tau)^2 = 2 gamma sigma_p^2 (1 - tau) / tau``.
``gamma = 0`` is the deterministic probability-flow sampler (no random draws after the start). The last
step lands on ``Psi_y`` at ``tau_{N-1}`` (``x + dtau v - w g / ||g||`` for ``"normalized"``), which is the
Euler step with ``dtau = 1 - tau``, matching ``sda_guided_sample``. ``n_correctors`` Langevin steps
``x <- x + eps s_y + sqrt(2 eps) xi`` run at each grid point before the predictor step, with
``eps = 2 corrector_snr^2 n / ||s_y||^2`` per member, ``n`` the number of entries: Song et al. (2021)'s
predictor-corrector rule with ``||xi||^2`` replaced by its expectation ``n``, because reusing the drawn
``xi`` correlates the step with the noise and inflates the stationary variance in low dimension.

Grid: ``u_k = 1 - (1 - k / N)^p`` (``p = step_power``; ``p = 1`` uniform, ``p = 0.5`` the early-fine
``models.vanilla_cfm.DEFAULT_STEP_POWER``), ``tau_k = tau0 + (1 - tau0) u_k``. With ``tau0_per_window``
each window runs its own grid on ``[tau0_w, 1]`` with the same ``N`` steps (the network takes one ``tau``
per batch element), so the cost does not depend on ``tau0_w``.

Warm start: ``x_tau0 = tau0 * start + (1 - tau0) sigma_p eps`` with ``start = x_start`` (shared by
members) or ``member_starts[..., m]``; neither given requires ``tau0 = 0`` (pure noise). ``c_tau`` uses
``max(tau, TAU_EPS)``, and ``g = 0`` for ``tau < TAU_EPS``: ``p_0 = N(0, sigma_p^2 I)`` with or without ``y``,
so a pure-noise start needs no correction, while ``g^2 ~ 1 / tau`` would make that first step overshoot. Members run batched along the batch dimension (member-major);
start noises are drawn member by member with shape ``(B, T, D)``, in the order of ``sda_guided_sample``.
"""
import copy
from typing import Callable, Optional, Sequence, Union

import torch

GUIDANCE_MODES = ("normalized", "dps", "pigdm")
TAU_EPS = 1e-4
TAU0_MAX = 1.0 - 1e-6


def _repeat_batch(batch, n_members: int, batch_size: int):
    if n_members == 1:
        return batch
    rep = copy.copy(batch)
    for name, value in vars(batch).items():
        if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == batch_size:
            setattr(rep, name, value.repeat(n_members, *([1] * (value.dim() - 1))))
    if isinstance(getattr(batch, "batch_size", None), int):
        rep.batch_size = batch_size * n_members
    return rep


def observation_mask(obs: torch.Tensor, obs_mask: torch.Tensor, obs_indices: Optional[Sequence[int]] = None,
                     obs_channel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Float ``(B, T, D)`` mask of the entries entering the likelihood, as in ``guided_obs_cost``."""
    if obs_indices is not None and obs_channel_mask is not None:
        raise ValueError("obs_indices and obs_channel_mask are mutually exclusive")
    mask = obs_mask.to(obs.dtype).unsqueeze(-1) * (~torch.isnan(obs)).to(obs.dtype)
    if obs_indices is not None:
        keep = torch.zeros(obs.shape[-1], dtype=obs.dtype, device=obs.device)
        keep[list(obs_indices)] = 1.0
        mask = mask * keep
    if obs_channel_mask is not None:
        mask = mask * obs_channel_mask.to(obs.dtype)
    return mask


def _bcast(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return t.view(-1, *([1] * (ref.dim() - 1)))


def _jacobian_diag(psi: torch.Tensor, x: torch.Tensor, mask: torch.Tensor, mode: str, n_probes: int,
                   generator: Optional[torch.Generator]) -> torch.Tensor:
    diag = torch.zeros_like(psi)
    if mode == "exact":
        flat = psi.flatten(1)
        for i in range(flat.shape[1]):
            e = torch.zeros_like(flat)
            e[:, i] = 1.0
            row = torch.autograd.grad(psi, x, e.view_as(psi), retain_graph=True)[0]
            diag.flatten(1)[:, i] = row.flatten(1)[:, i]
        return diag * mask
    if mode != "hutchinson":
        raise ValueError(f"unknown pigdm_diag {mode!r}")
    for _ in range(n_probes):
        z = (torch.randint(0, 2, psi.shape, device=psi.device, generator=generator).to(psi.dtype) * 2 - 1) * mask
        diag = diag + z * torch.autograd.grad(psi, x, z, retain_graph=True)[0]
    return diag * mask / n_probes


def posterior_state_prediction(prior, batch, x: torch.Tensor, tau: torch.Tensor, *, y: torch.Tensor,
                               mask: torch.Tensor, R_norm: Union[float, torch.Tensor], guidance: str = "pigdm",
                               guidance_weight: Union[float, torch.Tensor] = 1.0, dtau: Optional[torch.Tensor] = None,
                               pigdm_cov: str = "tweedie", pigdm_diag: str = "hutchinson",
                               hutchinson_probes: int = 4, generator: Optional[torch.Generator] = None):
    """One guided prior evaluation at ``(x, tau)``.

    Returns ``(v, psi, psi_y, shift)``; ``shift`` is the ``"normalized"`` displacement ``w g / ||g||``
    (``None`` otherwise) and ``dtau`` is required for that mode only. ``y`` must be NaN-free
    (``mask`` carries the missing entries).
    """
    if guidance not in GUIDANCE_MODES:
        raise ValueError(f"guidance must be one of {GUIDANCE_MODES}, got {guidance!r}")
    sigma_p = prior.sigma_prior
    tau_b = _bcast(tau, x)
    w = guidance_weight if torch.is_tensor(guidance_weight) else float(guidance_weight)
    if not torch.is_tensor(w) and w == 0.0:
        with torch.no_grad():
            v = prior.forward(x, batch, tau)
        psi = x + (1.0 - tau_b) * v
        return v, psi, psi, None
    if torch.is_tensor(w):
        w = _bcast(w, x)
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        v = prior.forward(x, batch, tau)
        psi = x + (1.0 - tau_b) * v
        resid = (psi - y) * mask
        if guidance == "normalized":
            grad = torch.autograd.grad((resid * (psi - y) / R_norm).sum(), x)[0]
            norm = grad.flatten(1).norm(dim=1).clamp_min(1e-8)
            shift = (w / _bcast(norm, grad)) * grad
            v, psi = v.detach(), psi.detach()
            return v, psi, psi - (1.0 - tau_b) / _bcast(dtau, x) * shift, shift
        c_tau = (1.0 - tau_b) ** 2 * sigma_p ** 2 / tau_b.clamp_min(TAU_EPS)
        variance = R_norm
        if guidance == "pigdm":
            if pigdm_cov == "tweedie":
                k_tau = c_tau
            elif pigdm_cov == "r2":
                sig2 = (sigma_p * (1.0 - tau_b) / tau_b.clamp_min(TAU_EPS)) ** 2
                k_tau = sig2 / (1.0 + sig2)
            else:
                raise ValueError(f"unknown pigdm_cov {pigdm_cov!r}")
            jdiag = _jacobian_diag(psi, x, mask, pigdm_diag, hutchinson_probes, generator).clamp_min(0.0)
            variance = (R_norm + k_tau * jdiag).detach()
        grad = torch.autograd.grad(0.5 * (resid * (psi - y) / variance).sum(), x)[0]
    v, psi = v.detach(), psi.detach()
    return v, psi, psi - w * c_tau * grad, None


def tau_grid(n_steps: int, step_power: float = 1.0) -> list:
    if step_power == 1.0:
        return [k / n_steps for k in range(n_steps + 1)]
    return [1.0 - (1.0 - k / n_steps) ** step_power for k in range(n_steps + 1)]


def _integrate(prior, rep, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, tau0_rep: torch.Tensor,
               grid: list, *, gamma: float, guidance: str, guidance_weight, R_norm, n_correctors: int,
               corrector_snr: float, pigdm_cov: str, pigdm_diag: str, hutchinson_probes: int,
               generator: Optional[torch.Generator]) -> torch.Tensor:
    sigma_p = prior.sigma_prior
    n_steps = len(grid) - 1

    def randn(*shape):
        return torch.randn(*shape, device=x.device, dtype=x.dtype, generator=generator)

    def _guided(x, tau, dtau):
        w = guidance_weight(tau) if callable(guidance_weight) else guidance_weight
        return posterior_state_prediction(prior, rep, x, tau, y=y, mask=mask, R_norm=R_norm, guidance=guidance,
                                          guidance_weight=w, dtau=dtau, pigdm_cov=pigdm_cov,
                                          pigdm_diag=pigdm_diag, hutchinson_probes=hutchinson_probes,
                                          generator=generator)

    def _score(x, tau, psi_y):
        tau_b = _bcast(tau, x)
        return (tau_b * psi_y - x) / ((1.0 - tau_b) ** 2 * sigma_p ** 2)

    for k in range(n_steps):
        tau = tau0_rep + (1.0 - tau0_rep) * grid[k]
        dtau = tau0_rep + (1.0 - tau0_rep) * grid[k + 1] - tau
        for _ in range(n_correctors):
            _, _, psi_y, _ = _guided(x, tau, dtau)
            score = _score(x, tau, psi_y)
            eps = 2.0 * corrector_snr ** 2 * score[0].numel() / score.flatten(1).pow(2).sum(1).clamp_min(1e-24)
            eps = _bcast(eps, x)
            xi = randn(*x.shape)
            x = (x + eps * score + (2.0 * eps).sqrt() * xi).detach()
        v, psi, psi_y, shift = _guided(x, tau, dtau)
        dtau_b = _bcast(dtau, x)
        if k == n_steps - 1:
            x = x + dtau_b * v - shift if shift is not None else psi_y
            break
        drift_x = x + dtau_b * v - shift if shift is not None else x + dtau_b * (psi_y - x) / (1.0 - _bcast(tau, x))
        if gamma > 0.0:
            tau_b = _bcast(tau, x)
            tau_c = tau_b.clamp_min(TAU_EPS)
            g2 = torch.where(tau_b < TAU_EPS, 0.0, 2.0 * gamma * sigma_p ** 2 * (1.0 - tau_c) / tau_c)
            drift_x = drift_x + dtau_b * 0.5 * g2 * _score(x, tau, psi_y) + (g2 * dtau_b).sqrt() * randn(*x.shape)
        x = drift_x.detach()
    return x.detach()


def fm_posterior_sample(prior, batch, *, x_start: Optional[torch.Tensor] = None, tau0: float = 0.0,
                        gamma: float = 0.0, guidance: str = "pigdm",
                        guidance_weight: Union[float, Callable] = 1.0, R_norm: Union[float, torch.Tensor] = 1.0,
                        n_steps: int = 10, step_power: float = 1.0, n_correctors: int = 0,
                        corrector_snr: float = 0.1, n_members: int = 30,
                        member_starts: Optional[torch.Tensor] = None,
                        tau0_per_window: Optional[torch.Tensor] = None,
                        obs_indices: Optional[Sequence[int]] = None,
                        obs_channel_mask: Optional[torch.Tensor] = None, pigdm_cov: str = "tweedie",
                        pigdm_diag: str = "hutchinson", hutchinson_probes: int = 4,
                        member_chunk: Optional[int] = None, generator: Optional[torch.Generator] = None):
    """Posterior ensemble ``(B, T, D, n_members)`` in the prior's (normalised) space, and the NFE per member.

    ``guidance_weight`` is a float or a callable of the per-element ``tau`` tensor. ``tau0_per_window`` (B,)
    overrides ``tau0``; both are clipped to ``[0, TAU0_MAX]`` (``tau0 = 1`` returns the start up to
    ``1e-6`` noise). ``member_starts`` (B, T, D, M) overrides ``x_start`` and sets ``n_members = M``.
    ``R_norm`` is a float or a tensor broadcastable to the channel dimension. NFE = network evaluations per
    member, ``n_steps * (1 + n_correctors)``. ``member_chunk`` caps the members per network batch
    (memory); the start noises are drawn before chunking, the later draws chunk by chunk.
    """
    obs = batch.obs
    B, T, _ = obs.shape
    device, dtype = obs.device, obs.dtype
    D = prior.state_dim
    sigma_p = prior.sigma_prior
    if member_starts is not None:
        n_members = member_starts.shape[-1]
    M = n_members
    if tau0_per_window is not None:
        tau0_w = tau0_per_window.to(device=device, dtype=dtype).reshape(B)
    else:
        tau0_w = torch.full((B,), float(tau0), device=device, dtype=dtype)
    tau0_w = tau0_w.clamp(0.0, TAU0_MAX)
    if x_start is None and member_starts is None and bool((tau0_w > 0).any()):
        raise ValueError("tau0 > 0 needs x_start or member_starts")

    def _randn(*shape):
        return torch.randn(*shape, device=device, dtype=dtype, generator=generator)

    starts = []
    for m in range(M):
        noise = _randn(B, T, D) * sigma_p
        start = member_starts[..., m] if member_starts is not None else x_start
        starts.append(noise if start is None else
                      (1.0 - _bcast(tau0_w, noise)) * noise + _bcast(tau0_w, noise) * start)
    x = torch.cat(starts, dim=0)

    chunk = M if member_chunk is None else max(1, min(member_chunk, M))
    grid = tau_grid(n_steps, step_power)
    if torch.is_tensor(R_norm):
        R_norm = R_norm.to(device=device, dtype=dtype)
    obs_mask = observation_mask(obs, batch.obs_mask, obs_indices, None)
    if obs_channel_mask is not None:
        obs_mask = obs_mask * obs_channel_mask.to(dtype).expand(B, T, D)
    y_full = torch.nan_to_num(obs, nan=0.0)
    out = []
    for m0 in range(0, M, chunk):
        mc = min(chunk, M - m0)
        out.append(_integrate(prior, _repeat_batch(batch, mc, B), x[m0 * B:(m0 + mc) * B], y_full.repeat(mc, 1, 1),
                              obs_mask.repeat(mc, 1, 1), tau0_w.repeat(mc), grid, gamma=gamma, guidance=guidance,
                              guidance_weight=guidance_weight, R_norm=R_norm, n_correctors=n_correctors,
                              corrector_snr=corrector_snr, pigdm_cov=pigdm_cov, pigdm_diag=pigdm_diag,
                              hutchinson_probes=hutchinson_probes, generator=generator))
    x = torch.cat(out, dim=0)
    members = x.detach().view(M, B, T, D).permute(1, 2, 3, 0)
    return members, n_steps * (1 + n_correctors)
