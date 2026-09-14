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

Also home to the TRAINING-time counterpart (:func:`sample_training_density_mask`,
wired into ``data/dataloader.py::make_collate_fm``): mixes full-density and
randomly-reduced-density observation events within the same batch, so a
direct-obs-consuming model (DirectUNet/CFM) sees partial-channel NaN patterns
during training instead of only ever at eval time -- closing the OOD gap
:func:`apply_density_mask_to_obs` otherwise exposes.
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


def random_variable_keep_mask(
    leading_shape: tuple[int, ...],
    n_channels: int,
    keep_k: torch.Tensor,
    device=None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Like :func:`random_keep_mask` but ``keep_k`` is a tensor broadcastable
    to ``leading_shape`` instead of a single scalar, giving a possibly
    different exact keep-count for every independent leading-dim slice (e.g.
    training-time augmentation mixing full-density and randomly-reduced-
    density observation events within the same batch -- see
    :func:`sample_training_density_mask`).

    Implemented via per-slice random-score ranks rather than ``topk`` (which
    only supports one ``k`` for a whole tensor): ``rank[..., c]`` is channel
    ``c``'s position in that slice's descending random-score order (0 =
    highest score), and a channel is kept iff its rank is below that slice's
    own ``keep_k`` -- equivalent in distribution to :func:`random_keep_mask`
    when ``keep_k`` happens to be constant.
    """
    keep_k_t = torch.as_tensor(keep_k, device=device)
    if keep_k_t.shape != tuple(leading_shape):
        keep_k_t = keep_k_t.expand(leading_shape)
    keep_k_t = keep_k_t.clamp(0, n_channels)
    scores = torch.rand(*leading_shape, n_channels, device=device, generator=generator)
    order = scores.argsort(dim=-1, descending=True)
    rank = torch.empty_like(order)
    idx = torch.arange(n_channels, device=device).expand(*leading_shape, n_channels)
    rank.scatter_(-1, order, idx)
    return rank < keep_k_t.unsqueeze(-1)


def sample_training_density_mask(
    batch_size: int,
    num_steps: int,
    full_prob: float,
    min_keep: int = 0,
    num_slow: int = NUM_SLOW,
    num_fast: int = NUM_FAST,
    device=None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Full-width ``(B, T, num_slow + num_fast)`` training-augmentation
    keep-mask.

    Independently at every ``(window, timestep)`` pair: with probability
    ``full_prob`` the event is full-density (all ``num_fast`` fast channels
    kept, so the model keeps seeing the canonical/most-common regime);
    otherwise ``keep_k`` is drawn uniformly from
    ``{min_keep, ..., num_fast - 1}`` and exactly that many fast channels are
    kept. Slow columns always stay ``True``. Matches
    :func:`fast_channel_keep_mask`'s eval-time convention (per-obs-time
    redraw, not per-window) so train- and eval-time masking are the same
    mechanics -- only the ``keep_k`` distribution differs (a fixed grid at
    eval, this mixture at train).
    """
    if not 0.0 <= full_prob <= 1.0:
        raise ValueError(f"full_prob={full_prob} out of range [0, 1]")
    if not 0 <= min_keep < num_fast:
        raise ValueError(f"min_keep={min_keep} out of range [0, {num_fast})")
    is_full = torch.rand(batch_size, num_steps, device=device, generator=generator) < full_prob
    rand_keep_k = torch.randint(min_keep, num_fast, (batch_size, num_steps),
                                device=device, generator=generator)
    keep_k = torch.where(is_full, torch.full_like(rand_keep_k, num_fast), rand_keep_k)
    slow = torch.ones(batch_size, num_steps, num_slow, dtype=torch.bool, device=device)
    fast = random_variable_keep_mask((batch_size, num_steps), num_fast, keep_k,
                                     device=device, generator=generator)
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
