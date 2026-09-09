"""Tests for L96 per-channel normalization (data/normalization.py) and its
opt-in wiring into collate_fm/collate_eval/collate_joint_eval.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.dataloader import collate_fm, make_collate_fm
from data.normalization import compute_channel_stats, denormalize, normalize
from evaluation.neural_inference import (
    collate_eval,
    collate_joint_eval,
    make_collate_eval,
    make_collate_joint_eval,
)


def _synthetic_states(n_slow=8, n_fast=16, n_windows=50, T=10, slow_std=2.0, fast_std=0.2, seed=0):
    """(n_windows, T, n_slow+n_fast) with distinct per-group scales, each
    channel offset by its index so per-channel means also differ.
    """
    g = torch.Generator().manual_seed(seed)
    slow = torch.randn(n_windows, T, n_slow, generator=g) * slow_std
    fast = torch.randn(n_windows, T, n_fast, generator=g) * fast_std
    offsets = torch.arange(n_slow + n_fast, dtype=torch.float32)
    return torch.cat([slow, fast], dim=-1) + offsets


def test_compute_channel_stats_recovers_distinct_group_scales():
    states = _synthetic_states(slow_std=2.0, fast_std=0.2, n_windows=200, T=20)
    stats = compute_channel_stats(states)
    assert stats["mean"].shape == (24,)
    assert stats["std"].shape == (24,)

    offsets = torch.arange(24, dtype=torch.float32)
    torch.testing.assert_close(stats["mean"], offsets, atol=0.1, rtol=0)
    assert torch.all(stats["std"][:8] > 1.5)  # slow channels, std~2.0
    assert torch.all(stats["std"][8:] < 0.5)  # fast channels, std~0.2


def test_compute_channel_stats_numpy_input():
    states = _synthetic_states(n_windows=200, T=20).numpy()
    stats = compute_channel_stats(states)
    assert isinstance(stats["mean"], np.ndarray)
    assert stats["mean"].shape == (24,)


def test_normalize_denormalize_roundtrip_torch():
    states = _synthetic_states(n_windows=50, T=10)
    stats = compute_channel_stats(states)
    x = torch.randn(4, 10, 24) * 3 + 1
    x_norm = normalize(x, stats)
    x_back = denormalize(x_norm, stats)
    torch.testing.assert_close(x_back, x, atol=1e-5, rtol=1e-5)


def test_normalize_denormalize_roundtrip_numpy():
    states = _synthetic_states(n_windows=50, T=10)
    stats = compute_channel_stats(states)
    x = np.random.randn(4, 10, 24).astype(np.float32) * 3 + 1
    x_norm = normalize(x, stats)
    x_back = denormalize(x_norm, stats)
    np.testing.assert_allclose(x_back, x, atol=1e-4, rtol=1e-4)


def test_normalize_actually_rescales():
    """A channel normalized by its own stats should end up ~unit variance,
    zero mean -- guards against a no-op normalize/denormalize pair.
    """
    states = _synthetic_states(n_windows=500, T=20, slow_std=2.0)
    stats = compute_channel_stats(states)
    x_norm = normalize(states, stats)
    flat = x_norm.reshape(-1, 24)
    torch.testing.assert_close(flat.mean(dim=0), torch.zeros(24), atol=0.05, rtol=0)
    torch.testing.assert_close(flat.std(dim=0), torch.ones(24), atol=0.05, rtol=0)


class _FMDataset(Dataset):
    """Minimal 4-tuple dataset matching data/dataloader.py's expected item shape."""
    def __init__(self, states, obs, mask, forcing):
        self.states, self.obs, self.mask, self.forcing = states, obs, mask, forcing

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        return self.states[idx], self.obs[idx], self.mask[idx], self.forcing[idx]


def _fm_batch(B=4, T=6, D=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    states = torch.randn(B, T, D, generator=g) * 3 + 1
    obs = torch.randn(B, T, D, generator=g) * 3 + 1
    mask = torch.ones(B, T, D, dtype=torch.bool)
    forcing = torch.randn(B, T, generator=g)
    return _FMDataset(states, obs, mask, forcing)


def test_make_collate_fm_none_is_collate_fm():
    """make_collate_fm(None) must be the exact same callable as collate_fm --
    the regression invariant that data.normalize=False changes nothing.
    """
    assert make_collate_fm(None) is collate_fm


def test_make_collate_fm_none_matches_bare_collate_fm_output():
    ds = _fm_batch()
    loader_a = DataLoader(ds, batch_size=4, collate_fn=collate_fm)
    loader_b = DataLoader(ds, batch_size=4, collate_fn=make_collate_fm(None))
    a, b = next(iter(loader_a)), next(iter(loader_b))
    torch.testing.assert_close(a.states, b.states)
    torch.testing.assert_close(a.obs, b.obs)


def test_make_collate_fm_normalizes_states_and_obs():
    ds = _fm_batch()
    stats = compute_channel_stats(ds.states)
    loader = DataLoader(ds, batch_size=4, collate_fn=make_collate_fm(stats))
    batch = next(iter(loader))
    expected_states = normalize(ds.states, stats)
    expected_obs = normalize(ds.obs, stats)
    torch.testing.assert_close(batch.states, expected_states)
    torch.testing.assert_close(batch.obs, expected_obs)
    # obs_mask/forcing are untouched by normalization
    torch.testing.assert_close(batch.forcing, ds.forcing)


def test_make_collate_fm_obs_density_cfg_none_is_noop():
    """obs_density_cfg=None (default) must reproduce plain collate_fm exactly
    -- the regression invariant that data.obs_density_augment=False (the
    DataConfig default) changes nothing for every pre-existing config.
    """
    assert make_collate_fm(None, obs_density_cfg=None) is collate_fm


def test_make_collate_fm_obs_density_nans_dropped_fast_channels():
    torch.manual_seed(0)
    ds = _fm_batch(D=24)
    loader = DataLoader(ds, batch_size=4, collate_fn=make_collate_fm(
        None, obs_density_cfg={"full_prob": 0.0}))
    batch = next(iter(loader))
    assert not torch.isnan(batch.obs[..., :8]).any()  # slow always kept
    assert torch.isnan(batch.obs[..., 8:]).any()  # some fast channels dropped
    assert not torch.isnan(batch.obs[..., 8:]).all()  # but not literally all of them, every batch


def test_make_collate_fm_obs_density_full_prob_one_is_noop_on_obs_values():
    """full_prob=1.0 always keeps full density -- obs values must be
    unchanged (still no NaN beyond whatever the raw obs already had)."""
    ds = _fm_batch(D=24)
    loader = DataLoader(ds, batch_size=4, collate_fn=make_collate_fm(
        None, obs_density_cfg={"full_prob": 1.0}))
    batch = next(iter(loader))
    torch.testing.assert_close(batch.obs, ds.obs)


def test_make_collate_fm_obs_density_wrong_dim_raises():
    import pytest
    ds = _fm_batch(D=10)  # not the canonical 8+16
    loader = DataLoader(ds, batch_size=4, collate_fn=make_collate_fm(
        None, obs_density_cfg={"full_prob": 0.4}))
    with pytest.raises(ValueError):
        next(iter(loader))


def test_make_collate_fm_obs_density_composes_with_normalization():
    """Density masking (NaN) must survive being fed through normalize (an
    elementwise affine op) -- order of the two must not silently drop the
    augmentation or crash.
    """
    torch.manual_seed(0)
    ds = _fm_batch(D=24)
    stats = compute_channel_stats(ds.states)
    loader = DataLoader(ds, batch_size=4, collate_fn=make_collate_fm(
        stats, obs_density_cfg={"full_prob": 0.0}))
    batch = next(iter(loader))
    assert torch.isnan(batch.obs[..., 8:]).any()
    assert not torch.isnan(batch.obs[..., :8]).any()


def _eval_batch(B=4, T=6, D=24, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [{
        "true_state": torch.randn(T, D, generator=g) * 3 + 1,
        "obs": torch.randn(T, D, generator=g) * 3 + 1,
        "obs_mask": torch.ones(T, D, dtype=torch.bool),
        "forcing_corrupted": torch.randn(T, generator=g),
    } for _ in range(B)]


def test_make_collate_eval_none_is_collate_eval():
    assert make_collate_eval(None) is collate_eval


def test_make_collate_joint_eval_none_is_collate_joint_eval():
    assert make_collate_joint_eval(None) is collate_joint_eval


def test_make_collate_eval_normalizes_obs_not_true_state():
    batch = _eval_batch()
    states = torch.stack([b["true_state"] for b in batch])
    stats = compute_channel_stats(states)

    plain = collate_eval(batch)
    normed = make_collate_eval(stats)(batch)

    # true_state stays raw always -- it's never fed to the model, only used
    # later for scoring.
    torch.testing.assert_close(normed["true_state"], plain["true_state"])
    # obs (the model's input) is normalized.
    torch.testing.assert_close(normed["obs"], normalize(plain["obs"], stats))
    assert not torch.allclose(normed["obs"], plain["obs"])
