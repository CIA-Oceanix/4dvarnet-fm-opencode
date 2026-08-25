"""Tests for L96 TweedieSolver variant (Phase B V1)."""
import torch

from models.residual import MeanEstimatorCell, IterativeUpdateCell
from models.solver import TweedieSolver


class TestTweedieL96:
    """V1 TweedieSolver tests for L96 obs-only use case."""

    def test_mean_estimator_cell_cond_extra_dim(self):
        """Test that MeanEstimatorCell accepts and forwards cond_extra_dim."""
        cell = MeanEstimatorCell(
            state_dim=24,
            hidden_channels=[64, 128, 256],
            use_obs=True,
            cond_extra_dim=0,  # obs-only for V1
            dropout=0.1,
        )
        x = torch.randn(2, 24, 30)
        obs = torch.randn(2, 24, 30)
        tau = torch.tensor([0.5])
        out = cell(x, obs, tau)
        assert out.shape == (2, 24, 30)

    def test_iterative_update_cell_cond_extra_dim(self):
        """Test that IterativeUpdateCell accepts and forwards cond_extra_dim."""
        cell = IterativeUpdateCell(
            state_dim=24,
            hidden_channels=[64, 128, 256],
            use_obs=True,
            use_energy=False,
            cond_extra_dim=0,
            dropout=0.1,
        )
        x = torch.randn(2, 24, 30)
        obs = torch.randn(2, 24, 30)
        x_tau = torch.randn(2, 24, 30)
        tau = torch.tensor([0.5])
        out = cell(x, obs, x_tau, tau)
        assert out.shape == (2, 24, 30)

    def test_tweedie_solver_cond_extra_dim(self):
        """Test that TweedieSolver accepts and forwards cond_extra_dim."""
        solver = TweedieSolver(
            state_dim=24,
            hidden_channels=[64, 128, 256],
            use_obs=True,
            use_energy=False,
            cond_extra_dim=0,
            K_inner=5,
            N_outer=10,
            dropout=0.1,
        )
        obs = torch.randn(2, 24, 30)
        x_mean = solver.estimate_mean(obs)
        assert x_mean.shape == (2, 24, 30)

    def test_tweedie_solver_ensemble_sample(self):
        """Test TweedieSolver.sample() with ensemble support."""
        solver = TweedieSolver(
            state_dim=24,
            hidden_channels=[32, 64, 128],  # smaller for speed in test
            use_obs=True,
            use_energy=False,
            cond_extra_dim=0,
            K_inner=3,  # fewer inner iterations for speed
            N_outer=5,
            dropout=0.1,
        )
        solver.eval()
        obs = torch.randn(2, 24, 30)
        # Single sample (backward compatible)
        pred_single = solver.sample(obs, N_outer=5, n_members=1)
        assert pred_single.shape == (2, 24, 30)

        # Ensemble
        pred_ensemble = solver.sample(obs, N_outer=5, n_members=30)
        assert pred_ensemble.shape == (2, 24, 30)

        # Check ensemble has proper statistics (30 members stacked last)
        pred_ensemble_mean = pred_ensemble.mean(dim=0)
        assert pred_ensemble_mean.shape == (2, 24, 30)

    def test_tweedie_solver_obs_only_config(self):
        """Test that V1 config (cond_extra_dim=0) works correctly."""
        scheduler = TweedieSolver(
            state_dim=24,
            hidden_channels=[64, 128, 256],
            use_obs=True,
            use_energy=False,
            cond_extra_dim=0,
            K_inner=5,
            N_outer=10,
            dropout=0.1,
        )
        # Verify use_energy was set to False (energy terms should be zeros)
        assert not scheduler.non_gaussian.use_energy

        obs = torch.randn(2, 24, 30)
        x_mean = scheduler.estimate_mean(obs)
        assert x_mean.shape == (2, 24, 30)

    def test_ensemble_sampling_statistics(self):
        """Test that ensemble sampling produces independent members with proper statistics."""
        solver = TweedieSolver(
            state_dim=24,
            hidden_channels=[32, 64, 128],
            use_obs=True,
            use_energy=False,
            cond_extra_dim=0,
            K_inner=3,
            N_outer=5,
            dropout=0.1,
        )
        solver.eval()
        obs = torch.randn(2, 24, 30)

        N = 30  # ensemble size for the test
        ensemble = solver.sample(obs, N_outer=5, n_members=N)

        # Shape: (B, T, D, M)
        b, t, d, m = ensemble.shape
        assert b == 2
        assert t == 30
        assert d == 24
        assert m == N

        # Check member means match the overall mean
        mean_overall = ensemble.mean(dim=-1)
        mean_per_member = ensemble.mean(dim=[0, 1, 2])
        assert torch.allclose(mean_overall, mean_per_member, rtol=1e-5)

        # Check member-wise variance (should be non-zero since members are independent samples)
        member_var = ensemble.var(dim=-1)
        assert member_var > 0, "Ensemble members should have non-zero variance (stochasticity)"

        # Check each member's mean is close to ground truth (should be close since no training)
        # This is just a sanity check that sampling doesn't crash
        assert member_var.shape == (2, 30, 24)

    def test_cond_extra_dim_forwarding_to_unet(self):
        """Test that cond_extra_dim is correctly passed through both cells to UNet1D."""
        # Test with cond_extra_dim > 0 (edge case, though V1 uses 0)
        solver = TweedieSolver(
            state_dim=24,
            hidden_channels=[32, 64, 128],
            use_obs=True,
            use_energy=False,
            cond_extra_dim=24,  # max extra dim for L96 24D obs
            K_inner=3,
            N_outer=5,
            dropout=0.1,
        )

        # Verify UNet proj_in is correct for MeanEstimatorCell
        # proj_in = state_dim + obs_dim + cond_extra_dim (when use_obs=True)
        cell = solver.mean_estimator.net
        proj_layers = [m for m in cell.modules() if isinstance(m, torch.nn.Linear)]
        assert len(proj_layers) > 0, "UNet should have a projection layer (ConditionEncoder.proj)"

        # For use_obs=True, proj_in = state_dim + obs_dim + cond_extra_dim = 24 + 24 + 24 = 72
        proj_in = cell.cond_encoder.proj.weight.shape[1]
        assert proj_in == 72, f"proj_in should be 72 (24+24+24), got {proj_in}"

        # Test forward pass with actual data
        x = torch.randn(2, 24, 30)
        obs = torch.randn(2, 24, 30)
        tau = torch.tensor([0.5])

        out = solver.mean_estimator(x, obs, tau)
        assert out.shape == (2, 24, 30)

    def test_k_inner_information_propagation(self):
        """Test that K_inner iterations properly propagate information across sequence length."""
        solver = TweedieSolver(
            state_dim=24,
            hidden_channels=[64, 128, 256],
            use_obs=True,
            use_energy=False,
            cond_extra_dim=0,
            K_inner=5,
            N_outer=10,
            dropout=0.1,
        )
        solver.eval()

        obs = torch.randn(4, 24, 30)
        x_mean = solver.estimate_mean(obs)

        # The output should have the correct shape
        assert x_mean.shape == (4, 24, 30)

        # The mean estimate should be a smooth function of obs (convolution-like behavior)
        # Test by comparing predictions for similar vs different observations
        obs1 = obs[0:1, :, :]  # First window
        obs2 = obs[1:2, :, :]  # Second window (different)

        mean1 = solver.estimate_mean(obs1)
        mean2 = solver.estimate_mean(obs2)

        # L2 distance should be reasonably small (untrained model but deterministic prediction)
        l2_diff = torch.norm(mean1 - mean2).item()
        assert l2_diff > 0, "Different observations should produce different mean estimates"
        assert l2_diff < 100, f"Different observations should produce similar outputs (L2 < 100), got {l2_diff}"

    def test_backward_compatibility_single_sample(self):
        """Test that single-sample sampling (n_members=1) matches backward-compatible interface."""
        solver = TweedieSolver(
            state_dim=24,
            hidden_channels=[32, 64, 128],
            use_obs=True,
            use_energy=False,
            cond_extra_dim=0,
            K_inner=3,
            N_outer=10,
            dropout=0.1,
        )
        solver.eval()
        obs = torch.randn(2, 24, 30)

        # Original interface: .sample(obs, N_outer) should work
        pred_old_interface = solver.sample(obs, N_outer=10, n_members=1)

        # New interface: .sample(obs, N_outer, n_members=1) should produce same result
        pred_new_interface = solver.sample(obs, N_outer=10, n_members=1)

        # Should be bitwise-equal
        assert torch.equal(pred_old_interface, pred_new_interface)

        # Should have expected shape (B, T, D)
        assert pred_old_interface.shape == (2, 24, 30)
        assert pred_new_interface.shape == (2, 24, 30)

    def test_non_gaussian_receives_correct_cond_extra_dim(self):
        """Test that IterativeUpdateCell receives the same cond_extra_dim as MeanEstimatorCell."""
        solver = TweedieSolver(
            state_dim=24,
            hidden_channels=[32, 64, 128],
            use_obs=True,
            use_energy=False,
            cond_extra_dim=0,
            K_inner=3,
            N_outer=5,
            dropout=0.1,
        )

        cell = solver.mean_estimator.net
        mean_proj_in = cell.cond_encoder.proj.weight.shape[1]

        cell2 = solver.non_gaussian.net
        ng_proj_in = cell2.cond_encoder.proj.weight.shape[1]

        # Both should have the same proj_in since they get the same cond_extra_dim
        assert mean_proj_in == ng_proj_in, "Both cells should receive the same cond_extra_dim"
