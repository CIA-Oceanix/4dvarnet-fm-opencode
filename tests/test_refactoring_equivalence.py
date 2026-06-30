"""Refactoring equivalence tests: verify old and new pipelines produce same results."""
import pytest
import torch
import numpy as np
from conf.schema import DataConfig
from models.solver import TweedieSolver
from training.losses import StateMSELoss
from training.lightning_module import LitModel as Lit4DVarNetFM
from data.lorenz63 import Lorenz63Dataset, generate_long_trajectory


class TestForwardEquivalence:
    """TweedieSolver (unchanged nn.Module) should produce identical results."""

    def test_estimate_mean_identical(self):
        """estimate_mean produces same output with same weights and input."""
        torch.manual_seed(42)
        model1 = TweedieSolver(state_dim=3, hidden_channels=[4, 8], K_inner=2)
        state_dict = model1.state_dict()

        model2 = TweedieSolver(state_dim=3, hidden_channels=[4, 8], K_inner=2)
        model2.load_state_dict(state_dict)

        model1.eval()
        model2.eval()
        torch.manual_seed(99)
        obs = torch.randn(2, 50, 3)
        with torch.no_grad():
            out1 = model1.estimate_mean(obs)
            out2 = model2.estimate_mean(obs)
        torch.testing.assert_close(out1, out2)

    def test_full_forward_identical(self):
        """Forward produces same output with same seed, weights, and input."""
        torch.manual_seed(42)
        obs = torch.randn(2, 50, 3)
        torch.manual_seed(42)
        model1 = TweedieSolver(state_dim=3, hidden_channels=[4, 8], K_inner=2, N_outer=2)
        sd = model1.state_dict()
        out1 = model1(obs)

        torch.manual_seed(42)
        model2 = TweedieSolver(state_dim=3, hidden_channels=[4, 8], K_inner=2, N_outer=2)
        model2.load_state_dict(sd)
        out2 = model2(obs)

        torch.testing.assert_close(out1, out2)


class TestLossEquivalence:
    """StateMSELoss with gradient_weight should match old behavior."""

    def test_loss_identical_without_gradient(self):
        """Loss without gradient is pure MSE."""
        pred = torch.randn(4, 50, 3)
        target = torch.randn(4, 50, 3)
        loss_fn = StateMSELoss(use_gradient_loss=False)
        loss = loss_fn(pred, target)
        expected = torch.nn.MSELoss()(pred, target)
        torch.testing.assert_close(loss, expected)

    def test_loss_with_gradient(self):
        """Gradient term adds weighted MSE of temporal differences."""
        pred = torch.randn(4, 50, 3)
        target = torch.randn(4, 50, 3)
        loss_fn = StateMSELoss(use_gradient_loss=True, gradient_weight=0.1)
        loss = loss_fn(pred, target)
        mse = torch.nn.MSELoss()(pred, target)
        pred_grad = pred[:, 1:] - pred[:, :-1]
        target_grad = target[:, 1:] - target[:, :-1]
        grad_mse = torch.nn.MSELoss()(pred_grad, target_grad)
        expected = mse + 0.1 * grad_mse
        torch.testing.assert_close(loss, expected)

    def test_zero_loss_perfect_prediction(self):
        """Perfect prediction gives zero loss."""
        target = torch.randn(4, 50, 3)
        loss_fn = StateMSELoss(use_gradient_loss=True)
        loss = loss_fn(target, target)
        assert loss.item() == 0.0


class TestDataEquivalence:
    """Dataset generation should be deterministic and identical."""

    def test_dataset_reproducible(self):
        """Same DataConfig produces identical dataset."""
        cfg1 = DataConfig(case=1, seed=42, num_windows=2, T_max=0.5, dt=0.01, spinup_steps=500)
        cfg2 = DataConfig(case=1, seed=42, num_windows=2, T_max=0.5, dt=0.01, spinup_steps=500)
        ds1 = Lorenz63Dataset(cfg1)
        ds2 = Lorenz63Dataset(cfg2)
        for i in range(len(ds1)):
            for key in ["true_state", "obs", "forcing_true", "forcing_corrupted"]:
                torch.testing.assert_close(ds1[i][key], ds2[i][key])

    def test_trajectory_reproducible(self):
        """generate_long_trajectory is deterministic with same seed."""
        t1 = generate_long_trajectory(
            num_steps=100, dt=0.01, seed=42,
            sigma=10.0, rho=28.0, beta=8/3,
            gamma=0.05, W_L_bar=0.0, c1=1.0, c2=0.1,
            sigma_0=0.08, sigma_L=0.20,
        )
        t2 = generate_long_trajectory(
            num_steps=100, dt=0.01, seed=42,
            sigma=10.0, rho=28.0, beta=8/3,
            gamma=0.05, W_L_bar=0.0, c1=1.0, c2=0.1,
            sigma_0=0.08, sigma_L=0.20,
        )
        torch.testing.assert_close(t1, t2)

    def test_config_properties_match_old(self):
        """DataConfig properties match the old Lorenz63Config behavior."""
        cfg = DataConfig(case=1, param_bias=0.0, dt=0.01, T_max=5.0)
        assert cfg.num_steps == 500
        assert cfg.use_corrupted_forcing is False
        sig, rho, bet = cfg.da_params
        assert sig == 10.0 and rho == 28.0 and abs(bet - 8/3) < 1e-10

        cfg = DataConfig(case=2, param_bias=0.05, dt=0.01, T_max=5.0)
        assert cfg.use_corrupted_forcing is True
        sig, rho, bet = cfg.da_params
        assert abs(sig - 9.5) < 1e-10
        assert abs(rho - 26.6) < 1e-10
        assert abs(bet - 2.8) < 1e-10


class TestCheckpointLoading:
    """Legacy checkpoint loading should work."""

    def test_legacy_stage1_checkpoint(self, tmp_path):
        """Lit4DVarNetFM can load a raw mean_estimator state_dict."""
        model = TweedieSolver(state_dim=3, hidden_channels=[4, 8], K_inner=2)
        ckpt_path = tmp_path / "stage1.pt"
        torch.save(model.mean_estimator.state_dict(), ckpt_path)

        new_model = TweedieSolver(state_dim=3, hidden_channels=[4, 8], K_inner=2)
        lit_module = Lit4DVarNetFM(model=new_model, stage=1)
        lit_module.load_legacy_checkpoint(str(ckpt_path))

        for p1, p2 in zip(model.mean_estimator.parameters(), new_model.mean_estimator.parameters()):
            torch.testing.assert_close(p1, p2)

    def test_legacy_full_checkpoint(self, tmp_path):
        """Lit4DVarNetFM can load a raw full model state_dict."""
        model = TweedieSolver(state_dim=3, hidden_channels=[4, 8], K_inner=2, N_outer=2)
        ckpt_path = tmp_path / "full.pt"
        torch.save(model.state_dict(), ckpt_path)

        new_model = TweedieSolver(state_dim=3, hidden_channels=[4, 8], K_inner=2, N_outer=2)
        lit_module = Lit4DVarNetFM(model=new_model, stage=2)
        lit_module.load_legacy_checkpoint(str(ckpt_path))

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            torch.testing.assert_close(p1, p2)
