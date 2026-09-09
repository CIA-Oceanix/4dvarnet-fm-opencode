"""Inference-time fast-Y observation-density reduction for the L96 study.

No retraining: given the canonical obsj2 24D observed-subspace ordering
(``evaluation/run_l96.py::make_obs_j_indices`` -- 8 slow ``X`` first, then the
16 fast ``Y`` interleaved 2-per-node), this module randomly keeps only
``keep_k`` of the 16 fast-``Y`` channels, **redrawn independently at every
(window, observation time) pair** -- the harder, per-obs-time-varying OOD
generalization test (as opposed to a single fixed mask per window). The 8
slow-``X`` channels always stay fully observed.

Two independent consumption paths for the resulting mask:

- **DirectUNet / VanillaCFM / FourDVarNet / Joint\\* / plain SDA-guidance
  target**: these read ``obs`` only via ``torch.nan_to_num(obs, nan=0.0)``,
  with no separate mask channel -- so :func:`apply_density_mask_to_obs` NaNs
  out the dropped fast channels directly in the ``obs`` tensor. This is
  genuinely out-of-distribution for them (trained only on whole-timestep NaN
  blocks, never partial-channel NaN within an observed timestep): a masked
  (zeroed) channel is indistinguishable from a real near-zero observation.
- **SDA guided sampling** (``evaluation/sda_sampler.py``): the prior network
  never conditions on raw obs at all: only ``guided_obs_cost`` reads it, as a
  cost term. There the boolean keep-mask is passed through as
  ``obs_channel_mask`` and excludes dropped-channel terms from the cost
  directly -- no zero-imputation ambiguity, architecturally clean.
"""
from __future__ import annotations

import torch

# Canonical obsj2 ordering (evaluation/run_l96.py::make_obs_j_indices with
# NO=8, J_obs=2): the first NUM_SLOW columns are the fully-observed slow X;
# the remaining NUM_FAST are the fast Y (2 per slow node) subject to density
# reduction.
NUM_SLOW = 8
NUM_FAST = 16


def fast_column_indices(num_slow: int = NUM_SLOW, num_fast: int = NUM_FAST) -> tuple[int, ...]:
    """Column indices of the fast-Y channels within the canonical ordering."""
    return tuple(range(num_slow, num_slow + num_fast))


def random_keep_mask(
    leading_shape: tuple[int, ...],
    n_channels: int,
    keep_k: int,
    device=None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Boolean ``(*leading_shape, n_channels)`` mask with exactly ``keep_k``
    True entries per independent leading-dim slice (e.g. per (window,
    obs-time) pair when ``leading_shape == (B, T)``), drawn via per-slice
    random scores + ``topk`` so the count is exact rather than a Bernoulli
    approximation.
    """
    if not 0 <= keep_k <= n_channels:
        raise ValueError(f"keep_k={keep_k} out of range [0, {n_channels}]")
    if keep_k == n_channels:
        return torch.ones(*leading_shape, n_channels, dtype=torch.bool, device=device)
    if keep_k == 0:
        return torch.zeros(*leading_shape, n_channels, dtype=torch.bool, device=device)
    scores = torch.rand(*leading_shape, n_channels, device=device, generator=generator)
    keep_idx = scores.topk(keep_k, dim=-1).indices
    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep.scatter_(-1, keep_idx, True)
    return keep


def fast_channel_keep_mask(
    batch_size: int,
    num_steps: int,
    keep_k: int,
    num_slow: int = NUM_SLOW,
    num_fast: int = NUM_FAST,
    device=None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Full-width ``(B, T, num_slow + num_fast)`` boolean keep-mask.

    Slow columns are always ``True``; fast columns are ``True`` for a random
    ``keep_k``-subset redrawn independently at every ``(window, timestep)``
    pair (whether or not that timestep is actually an observation time --
    unobserved rows are already all-NaN in ``obs``, so masking them further
    is a no-op there).
    """
    slow = torch.ones(batch_size, num_steps, num_slow, dtype=torch.bool, device=device)
    fast = random_keep_mask((batch_size, num_steps), num_fast, keep_k, device=device, generator=generator)
    return torch.cat([slow, fast], dim=-1)


def apply_density_mask_to_obs(obs: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    """Return a copy of ``obs`` (..., D) with every channel not in
    ``keep_mask`` set to NaN. Rows that were already all-NaN (unobserved
    timesteps) stay all-NaN.
    """
    if keep_mask.shape != obs.shape:
        raise ValueError(f"keep_mask shape {tuple(keep_mask.shape)} != obs shape {tuple(obs.shape)}")
    out = obs.clone()
    out[~keep_mask] = float("nan")
    return out
