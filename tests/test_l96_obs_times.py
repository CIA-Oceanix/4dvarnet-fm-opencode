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


def test_redraw_obs_noise_keeps_times_and_changes_noise():
    from data.obs_times import redraw_obs_noise
    items = _regular_windows()
    states = torch.stack([it[0] for it in items])
    mask = torch.stack([it[2] for it in items])
    old_obs = torch.stack([it[1] for it in items])
    obs = redraw_obs_noise(states, mask, R_VAR, generator=torch.Generator().manual_seed(3))
    assert torch.equal(~torch.isnan(obs).any(-1), mask)
    assert not torch.allclose(obs[mask], old_obs[mask])
    resid = obs[mask] - states[mask]
    assert abs(resid.var().item() - R_VAR) < 0.05


def test_collate_noise_only_mode_keeps_mask():
    items = _regular_windows()
    batch = make_collate_fm(None, obs_times_cfg={"R_var": R_VAR, "mode": "noise_only"})(items)
    torch.testing.assert_close(batch.obs_mask, torch.stack([it[2] for it in items]))
    with pytest.raises(ValueError):
        make_collate_fm(None, obs_times_cfg={"R_var": R_VAR, "mode": "bogus"})(items)


def test_resample_obs_variable_counts_and_fast_channels_in_range():
    from data.obs_times import resample_obs_variable
    states = torch.randn(64, T, D)
    mask = torch.zeros(64, T, dtype=torch.bool)
    obs, new_mask = resample_obs_variable(states, mask, R_VAR, (5, 50), (4, 16),
                                          generator=torch.Generator().manual_seed(0))
    counts = new_mask.sum(1)
    assert counts.min() >= 5 and counts.max() <= 50 and len(counts.unique()) > 10
    observed = obs[new_mask]
    assert not torch.isnan(observed[:, :8]).any()
    n_fast = (~torch.isnan(observed[:, 8:])).sum(1)
    assert n_fast.min() == 4 and n_fast.max() == 16
    assert torch.all(torch.isnan(obs[~new_mask]))
    per_time = (~torch.isnan(obs[..., 8:])).sum(-1)
    for b in range(64):
        k = per_time[b][new_mask[b]]
        assert torch.all(k == k[0])
    resid = observed[:, :8] - states[new_mask][:, :8]
    assert abs(resid.var().item() - R_VAR) < 0.05


def test_resample_obs_variable_times_are_stratified():
    from data.obs_times import resample_obs_variable
    states = torch.randn(8, T, D)
    _, new_mask = resample_obs_variable(states, torch.zeros(8, T, dtype=torch.bool), R_VAR,
                                        (20, 20), (16, 16), generator=torch.Generator().manual_seed(1))
    t = new_mask[0].nonzero().flatten()
    block = torch.arange(20) * (T // 20)
    assert torch.all(t >= block) and torch.all(t < block + T // 20)


def test_collate_variable_mode():
    items = _regular_windows()
    batch = make_collate_fm(None, obs_times_cfg={"R_var": R_VAR, "mode": "variable",
                                                 "n_obs_range": [5, 50], "fast_range": [4, 16]})(items)
    counts = batch.obs_mask.sum(1)
    assert torch.all((counts >= 5) & (counts <= 50))


def test_resample_obs_variable_first_step_keeps_count():
    from data.obs_times import resample_obs_variable
    states = torch.randn(32, T, D)
    mask = torch.zeros(32, T, dtype=torch.bool)
    g = torch.Generator().manual_seed(4)
    _, m = resample_obs_variable(states, mask, R_VAR, (10, 300), (4, 16), generator=g, first_step=True)
    assert torch.all(m[:, 0])
    g = torch.Generator().manual_seed(4)
    _, m_free = resample_obs_variable(states, mask, R_VAR, (10, 300), (4, 16), generator=g)
    torch.testing.assert_close(m.sum(1), m_free.sum(1))
    assert not torch.all(m_free[:, 0])


def test_reobserve_windows_fixed_is_deterministic_and_in_range():
    from data.obs_times import reobserve_windows_fixed
    idx = tuple(range(8)) + tuple(8 + k * 4 + j for k in range(8) for j in range(2))

    def windows():
        g = torch.Generator().manual_seed(0)
        return [{"true_state": torch.randn(T, 40, generator=g), "obs": None, "obs_mask": None}
                for _ in range(6)]

    a, b = windows(), windows()
    for ws in (a, b):
        reobserve_windows_fixed(ws, idx, R_VAR, (10, 300), (4, 16), seed=7, first_step=True)
    for x, y in zip(a, b):
        assert torch.equal(x["obs_mask"], y["obs_mask"])
        torch.testing.assert_close(x["obs"], y["obs"], equal_nan=True)
        n = int(x["obs_mask"].sum())
        assert 10 <= n <= 300 and x["obs_mask"][0] and x["obs"].shape == (T, 24)
        k = (~torch.isnan(x["obs"][x["obs_mask"], 8:])).sum(1)
        assert torch.all(k == k[0]) and 4 <= int(k[0]) <= 16
    assert len({int(w["obs_mask"].sum()) for w in a}) > 1


def test_new_obs_flags_default_off():
    from conf.schema import DataConfig
    d = DataConfig()
    assert not d.obs_first_step and not d.val_obs_random_layout and not d.obs_random_layout
