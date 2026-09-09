"""Tests for the fast-Y observation-density generalization utilities."""
import torch

from data.obs_density import (
    NUM_FAST,
    NUM_SLOW,
    apply_density_mask_to_obs,
    fast_channel_keep_mask,
    fast_column_indices,
    random_keep_mask,
    random_variable_keep_mask,
    sample_training_density_mask,
)


class TestFastColumnIndices:
    def test_default_matches_canonical_obsj2_layout(self):
        idx = fast_column_indices()
        assert idx == tuple(range(8, 24))
        assert len(idx) == NUM_FAST


class TestRandomKeepMask:
    def test_exact_count_per_slice(self):
        torch.manual_seed(0)
        mask = random_keep_mask((5, 7), 16, 4)
        assert mask.shape == (5, 7, 16)
        assert mask.dtype == torch.bool
        assert torch.all(mask.sum(dim=-1) == 4)

    def test_keep_k_zero_is_all_false(self):
        mask = random_keep_mask((3, 4), 16, 0)
        assert not mask.any()

    def test_keep_k_full_is_all_true(self):
        mask = random_keep_mask((3, 4), 16, 16)
        assert mask.all()

    def test_out_of_range_raises(self):
        import pytest
        with pytest.raises(ValueError):
            random_keep_mask((2, 2), 16, 17)
        with pytest.raises(ValueError):
            random_keep_mask((2, 2), 16, -1)

    def test_independent_draws_across_slices(self):
        """Different (leading-dim) slices should not all pick the same subset
        (statistically near-certain over many independent draws)."""
        torch.manual_seed(0)
        mask = random_keep_mask((200,), 16, 4)
        # Not every row identical to the first.
        assert not torch.all(mask == mask[0:1])


class TestFastChannelKeepMask:
    def test_shape_and_slow_always_true(self):
        torch.manual_seed(1)
        mask = fast_channel_keep_mask(batch_size=3, num_steps=6, keep_k=4)
        assert mask.shape == (3, 6, NUM_SLOW + NUM_FAST)
        assert mask[..., :NUM_SLOW].all()
        assert torch.all(mask[..., NUM_SLOW:].sum(dim=-1) == 4)

    def test_keep_k_16_is_full_density_noop(self):
        mask = fast_channel_keep_mask(batch_size=2, num_steps=3, keep_k=16)
        assert mask.all()

    def test_keep_k_0_drops_all_fast(self):
        mask = fast_channel_keep_mask(batch_size=2, num_steps=3, keep_k=0)
        assert mask[..., :NUM_SLOW].all()
        assert not mask[..., NUM_SLOW:].any()

    def test_redrawn_independently_per_timestep(self):
        """The same window's mask at different timesteps should differ
        (statistically near-certain for keep_k=4 of 16 over many steps)."""
        torch.manual_seed(2)
        mask = fast_channel_keep_mask(batch_size=1, num_steps=100, keep_k=4)
        fast = mask[0, :, NUM_SLOW:]
        assert not torch.all(fast == fast[0:1])


class TestRandomVariableKeepMask:
    def test_matches_constant_keep_k_distribution_properties(self):
        torch.manual_seed(0)
        keep_k = torch.full((5, 7), 4)
        mask = random_variable_keep_mask((5, 7), 16, keep_k)
        assert mask.shape == (5, 7, 16)
        assert mask.dtype == torch.bool
        assert torch.all(mask.sum(dim=-1) == 4)

    def test_per_slice_varying_keep_k_respected(self):
        torch.manual_seed(0)
        keep_k = torch.tensor([[0, 4, 8, 16]])
        mask = random_variable_keep_mask((1, 4), 16, keep_k)
        counts = mask.sum(dim=-1).squeeze(0)
        assert counts.tolist() == [0, 4, 8, 16]

    def test_scalar_keep_k_broadcasts(self):
        torch.manual_seed(0)
        mask = random_variable_keep_mask((3, 5), 16, torch.tensor(6))
        assert torch.all(mask.sum(dim=-1) == 6)


class TestSampleTrainingDensityMask:
    def test_shape_and_slow_always_true(self):
        torch.manual_seed(0)
        mask = sample_training_density_mask(batch_size=4, num_steps=10, full_prob=0.4)
        assert mask.shape == (4, 10, NUM_SLOW + NUM_FAST)
        assert mask[..., :NUM_SLOW].all()

    def test_full_prob_one_is_always_full_density(self):
        torch.manual_seed(0)
        mask = sample_training_density_mask(batch_size=4, num_steps=10, full_prob=1.0)
        assert mask.all()

    def test_full_prob_zero_never_keeps_all_16(self):
        """With full_prob=0, keep_k is drawn from {0,...,15} -- 16 (full
        density) should never occur."""
        torch.manual_seed(0)
        mask = sample_training_density_mask(batch_size=8, num_steps=50, full_prob=0.0)
        fast_counts = mask[..., NUM_SLOW:].sum(dim=-1)
        assert int(fast_counts.max()) < NUM_FAST

    def test_min_keep_respected(self):
        torch.manual_seed(0)
        mask = sample_training_density_mask(batch_size=8, num_steps=50, full_prob=0.0, min_keep=5)
        fast_counts = mask[..., NUM_SLOW:].sum(dim=-1)
        assert int(fast_counts.min()) >= 5

    def test_mixture_produces_both_full_and_reduced_density(self):
        """Over many draws at full_prob=0.4, both regimes should appear."""
        torch.manual_seed(0)
        mask = sample_training_density_mask(batch_size=1, num_steps=500, full_prob=0.4)
        fast_counts = mask[0, :, NUM_SLOW:].sum(dim=-1)
        assert (fast_counts == NUM_FAST).any()
        assert (fast_counts < NUM_FAST).any()

    def test_full_prob_out_of_range_raises(self):
        import pytest
        with pytest.raises(ValueError):
            sample_training_density_mask(batch_size=1, num_steps=1, full_prob=1.5)
        with pytest.raises(ValueError):
            sample_training_density_mask(batch_size=1, num_steps=1, full_prob=-0.1)

    def test_min_keep_out_of_range_raises(self):
        import pytest
        with pytest.raises(ValueError):
            sample_training_density_mask(batch_size=1, num_steps=1, full_prob=0.5, min_keep=16)
        with pytest.raises(ValueError):
            sample_training_density_mask(batch_size=1, num_steps=1, full_prob=0.5, min_keep=-1)

    def test_redrawn_independently_per_timestep(self):
        torch.manual_seed(3)
        mask = sample_training_density_mask(batch_size=1, num_steps=100, full_prob=0.4)
        fast = mask[0, :, NUM_SLOW:]
        assert not torch.all(fast == fast[0:1])


class TestApplyDensityMaskToObs:
    def test_dropped_channels_become_nan_kept_stay(self):
        obs = torch.arange(2 * 3 * 24, dtype=torch.float32).reshape(2, 3, 24)
        keep_mask = torch.ones(2, 3, 24, dtype=torch.bool)
        keep_mask[..., 10] = False
        out = apply_density_mask_to_obs(obs, keep_mask)
        assert torch.isnan(out[..., 10]).all()
        assert not torch.isnan(out[..., 0]).any()
        # Original tensor untouched (clone semantics).
        assert not torch.isnan(obs[..., 10]).any()

    def test_already_nan_rows_stay_nan(self):
        obs = torch.full((1, 4, 24), float("nan"))
        keep_mask = torch.ones(1, 4, 24, dtype=torch.bool)
        out = apply_density_mask_to_obs(obs, keep_mask)
        assert torch.isnan(out).all()

    def test_shape_mismatch_raises(self):
        import pytest
        obs = torch.zeros(2, 3, 24)
        keep_mask = torch.ones(2, 3, 16, dtype=torch.bool)
        with pytest.raises(ValueError):
            apply_density_mask_to_obs(obs, keep_mask)
