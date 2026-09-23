"""Training-time random observation TIMES for the L96 study.

The canonical L96 protocol observes all channels on a regular grid (every
``obs_interval`` steps, i.e. 30 obs times over a 3000-step window). This
module redraws, for every window of every training batch, the SAME number of
observation times at random, stratified positions: the window is split into
``n_obs`` contiguous blocks and one step is drawn uniformly inside each block.
All observed channels are observed at each drawn time (same obs layout as the
regular grid, only the timing changes), and the observation noise is redrawn
too, since the observed values must come from the truth at the new times.

Val/test keep the regular grid; this is wired into
``data/dataloader.py::make_collate_fm`` for the train loader only.
"""
from __future__ import annotations

import math

import torch


def stratified_obs_times(
    batch_size: int,
    num_steps: int,
    n_obs: int,
    device=None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """``(B, n_obs)`` long tensor of strictly increasing time indices, one
    drawn uniformly inside each of ``n_obs`` contiguous blocks of
    ``[0, num_steps)`` (block edges ``floor(k * num_steps / n_obs)``).
    """
    if not 1 <= n_obs <= num_steps:
        raise ValueError(f"n_obs={n_obs} out of range [1, {num_steps}]")
    edges = torch.tensor([math.floor(k * num_steps / n_obs) for k in range(n_obs + 1)],
                         device=device)
    starts, widths = edges[:-1], edges[1:] - edges[:-1]
    u = torch.rand(batch_size, n_obs, device=device, generator=generator)
    offsets = torch.minimum((u * widths).long(), widths - 1)
    return starts + offsets


def resample_obs_times(
    states: torch.Tensor,
    obs_mask: torch.Tensor,
    R_var: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fresh ``(obs, obs_mask)`` observing ``states`` (B, T, D) at
    stratified random times, with as many obs times per window as the
    incoming regular ``obs_mask`` (B, T) had, plus N(0, R_var) noise.
    """
    B, T, D = states.shape
    counts = obs_mask.reshape(B, T).sum(dim=1)
    n_obs = int(counts[0])
    if not torch.all(counts == n_obs):
        raise ValueError(f"windows in a batch have different obs counts: {counts.tolist()}")
    t_idx = stratified_obs_times(B, T, n_obs, device=states.device, generator=generator)
    new_mask = torch.zeros(B, T, dtype=torch.bool, device=states.device)
    new_mask.scatter_(1, t_idx, True)
    noise = torch.randn(B, n_obs, D, device=states.device, dtype=states.dtype,
                        generator=generator) * math.sqrt(R_var)
    picked = torch.gather(states, 1, t_idx.unsqueeze(-1).expand(B, n_obs, D))
    obs = torch.full_like(states, float("nan"))
    obs.scatter_(1, t_idx.unsqueeze(-1).expand(B, n_obs, D), picked + noise)
    return obs, new_mask.reshape(obs_mask.shape).to(obs_mask.dtype)


def redraw_obs_noise(
    states: torch.Tensor,
    obs_mask: torch.Tensor,
    R_var: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Fresh ``obs`` observing ``states`` (B, T, D) at the SAME times as
    ``obs_mask`` (B, T), with new N(0, R_var) noise -- the fixed-grid control
    for :func:`resample_obs_times` (only the noise realization changes)."""
    B, T, D = states.shape
    mask = obs_mask.reshape(B, T).bool()
    noise = torch.randn(states.shape, device=states.device, dtype=states.dtype,
                        generator=generator) * math.sqrt(R_var)
    obs = torch.full_like(states, float("nan"))
    obs[mask] = states[mask] + noise[mask]
    return obs


def resample_obs_variable(
    states: torch.Tensor,
    obs_mask: torch.Tensor,
    R_var: float,
    n_obs_range: tuple[int, int],
    fast_range: tuple[int, int],
    num_slow: int = 8,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fresh ``(obs, obs_mask)`` with a random observing system per window:
    ``n_obs`` uniform in ``n_obs_range`` (inclusive) at stratified random times,
    and ``k`` uniform in ``fast_range`` (inclusive) drawn once per window: at
    every obs time of that window ``k`` of the fast channels (columns
    ``num_slow:``) are observed, the subset redrawn per obs time; slow
    channels are always observed. ``obs_mask`` (B, T) marks times with any obs, so
    dropped fast channels are NaN inside an observed row.
    """
    from data.obs_density import random_variable_keep_mask

    B, T, D = states.shape
    num_fast = D - num_slow
    lo, hi = n_obs_range
    if not 1 <= lo <= hi <= T:
        raise ValueError(f"n_obs_range={n_obs_range} out of range [1, {T}]")
    flo, fhi = fast_range
    if not 0 <= flo <= fhi <= num_fast:
        raise ValueError(f"fast_range={fast_range} out of range [0, {num_fast}]")
    device = states.device
    n_obs = torch.randint(lo, hi + 1, (B,), device=device, generator=generator)
    new_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    for b in range(B):
        t_idx = stratified_obs_times(1, T, int(n_obs[b]), device=device, generator=generator)[0]
        new_mask[b, t_idx] = True
    keep_k = torch.randint(flo, fhi + 1, (B, 1), device=device, generator=generator).expand(B, T)
    fast_keep = random_variable_keep_mask((B, T), num_fast, keep_k, device=device, generator=generator)
    keep = torch.cat([torch.ones(B, T, num_slow, dtype=torch.bool, device=device), fast_keep], dim=-1)
    keep &= new_mask.unsqueeze(-1)
    noise = torch.randn(states.shape, device=device, dtype=states.dtype,
                        generator=generator) * math.sqrt(R_var)
    obs = torch.full_like(states, float("nan"))
    obs[keep] = states[keep] + noise[keep]
    return obs, new_mask.reshape(obs_mask.shape).to(obs_mask.dtype)
