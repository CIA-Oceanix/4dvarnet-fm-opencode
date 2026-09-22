"""Tests for training-time stratified random observation times
(data/obs_times.py) and their wiring into make_collate_fm.
"""
import math

import pytest
import torch
from torch.utils.data import DataLoader

from data.dataloader import collate_fm, make_collate_fm
from data.lorenz96 import _generate_observations
from data.normalization import compute_channel_stats
from data.obs_times import resample_obs_times, stratified_obs_times

T, OBS_INTERVAL, D, R_VAR = 3000, 100, 24, 0.5


def _regular_windows(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    items = []
    for i in range(n):
        states = torch.randn(T, D, generator=g) * 3 + 1
        obs, mask = _generate_observations(states, OBS_INTERVAL, R_VAR, seed=100 + i)
        items.append((states, obs, mask, torch.zeros(T)))
    return items


def test_stratified_times_one_per_block_and_increasing():
    t = stratified_obs_times(64, T, 30, generator=torch.Generator().manual_seed(0))
    assert t.shape == (64, 30)
    block = torch.arange(30) * OBS_INTERVAL
    assert torch.all(t >= block) and torch.all(t < block + OBS_INTERVAL)
    assert torch.all(t[:, 1:] > t[:, :-1])


def test_stratified_times_uneven_blocks_stay_in_range():
    t = stratified_obs_times(256, 101, 7, generator=torch.Generator().manual_seed(0))
    edges = torch.tensor([math.floor(k * 101 / 7) for k in range(8)])
    assert torch.all(t >= edges[:-1]) and torch.all(t < edges[1:])


def test_stratified_times_cover_whole_block():
    t = stratified_obs_times(4000, T, 30, generator=torch.Generator().manual_seed(0))
    offsets = (t - torch.arange(30) * OBS_INTERVAL).flatten()
    assert offsets.min() == 0 and offsets.max() == OBS_INTERVAL - 1
    assert len(offsets.unique()) == OBS_INTERVAL


def test_stratified_times_rejects_bad_count():
    with pytest.raises(ValueError):
        stratified_obs_times(2, 10, 11)


def test_resample_keeps_count_and_obs_values_follow_truth():
    items = _regular_windows()
    states = torch.stack([it[0] for it in items])
    mask = torch.stack([it[2] for it in items])
    obs, new_mask = resample_obs_times(states, mask, R_VAR, generator=torch.Generator().manual_seed(1))
    assert new_mask.shape == mask.shape and new_mask.dtype == mask.dtype
    assert torch.all(new_mask.sum(1) == 30)
    assert not torch.equal(new_mask, mask)
    assert torch.all(torch.isnan(obs[~new_mask]))
    assert not torch.isnan(obs[new_mask]).any()
    resid = obs[new_mask] - states[new_mask]
    assert abs(resid.var().item() - R_VAR) < 0.05
    assert abs(resid.mean().item()) < 0.05


def test_resample_rejects_mixed_counts():
    states = torch.randn(2, 20, 3)
    mask = torch.zeros(2, 20, dtype=torch.bool)
    mask[0, :4] = True
    mask[1, :5] = True
    with pytest.raises(ValueError):
        resample_obs_times(states, mask, R_VAR)


def test_collate_none_is_still_plain_collate_fm():
    assert make_collate_fm(None, obs_times_cfg=None) is collate_fm


def test_collate_redraws_times_every_batch():
    items = _regular_windows()
    collate = make_collate_fm(None, obs_times_cfg={"R_var": R_VAR})
    torch.manual_seed(0)
    a = collate(items)
    b = collate(items)
    assert torch.all(a.obs_mask.sum(1) == 30)
    assert not torch.equal(a.obs_mask, b.obs_mask)
    torch.testing.assert_close(a.states, torch.stack([it[0] for it in items]))


def test_collate_composes_with_normalization_and_density():
    items = _regular_windows()
    stats = compute_channel_stats(torch.stack([it[0] for it in items]))
    loader = DataLoader(items, batch_size=4, collate_fn=make_collate_fm(
        stats, obs_density_cfg={"full_prob": 0.0}, obs_times_cfg={"R_var": R_VAR}))
    batch = next(iter(loader))
    observed = batch.obs[batch.obs_mask]
    assert not torch.isnan(observed[:, :8]).any()
    assert torch.isnan(observed[:, 8:]).any()
    assert torch.all(torch.isnan(batch.obs[~batch.obs_mask]))


def test_collate_rejects_full_state_targets():
    items = [(torch.randn(T, 40), obs, mask, f) for _, obs, mask, f in _regular_windows()]
    with pytest.raises(ValueError):
        make_collate_fm(None, obs_times_cfg={"R_var": R_VAR})(items)
