import torch
import torch.nn.functional as F

from models.vanilla_cfm import PredictStateCFM, TweedieCFM, VanillaCFM


class _MockBatch:
    def __init__(self, B=2, T=50, D=3):
        self.states = torch.randn(B, T, D)
        self.obs = torch.randn(B, T, D)
        self.obs_mask = torch.ones(B, T, dtype=torch.bool)
        self.forcing = torch.randn(B, T)
        self.params = torch.randn(B, 4)
        self.batch_size = B


class TestVanillaCFM:
    def test_forward_shape(self):
        model = VanillaCFM(state_dim=3, hidden_channels=[4, 8])
        B, T, D = 2, 50, 3
        x_t = torch.randn(B, T, D)
        batch = _MockBatch(B=B, T=T, D=D)
        tau = torch.rand(B)
        v = model(x_t, batch, tau)
        assert v.shape == (B, T, D), f"Expected (B,T,D), got {v.shape}"

    def test_loss_finite(self):
        model = VanillaCFM(state_dim=3, hidden_channels=[4, 8])
        batch = _MockBatch(B=2, T=50, D=3)
        loss = model.compute_cfm_loss(batch)
        assert torch.isfinite(loss), "Loss is not finite"
        assert loss.ndim == 0, "Loss should be scalar"

    def test_sample_shape(self):
        model = VanillaCFM(state_dim=3, hidden_channels=[4, 8], N_outer=3)
        B, T, D = 2, 50, 3
        batch = _MockBatch(B=B, T=T, D=D)
        samples = model.sample(batch)
        assert samples.shape == (B, T, D), f"Expected (B,T,D), got {samples.shape}"

    def test_sample_finite(self):
        model = VanillaCFM(state_dim=3, hidden_channels=[4, 8], N_outer=3)
        model.eval()
        batch = _MockBatch(B=1, T=50, D=3)
        with torch.no_grad():
            samples = model.sample(batch)
        assert torch.isfinite(samples).all(), "Samples contain NaN or Inf"

    def test_nan_obs_zeroed(self):
        model = VanillaCFM(state_dim=3, hidden_channels=[4, 8], N_outer=3)
        batch = _MockBatch(B=1, T=50, D=3)
        batch.obs[0, 0] = float('nan')
        with torch.no_grad():
            loss = model.compute_cfm_loss(batch)
        assert torch.isfinite(loss), "NaN in obs should be zeroed"

    def test_cond_extra_dim_0_proj_shape(self):
        model = VanillaCFM(state_dim=3, hidden_channels=[4, 8], N_outer=3,
                           cond_extra_dim=0)
        proj = model.unet.cond_encoder.proj
        assert proj.weight.shape == (4, 2 * 3 + 0), f"proj shape {proj.weight.shape}"

    def test_cond_extra_dim_gt0_proj_shape(self):
        model = VanillaCFM(state_dim=3, hidden_channels=[4, 8], N_outer=3,
                           param_dim=4, cond_extra_dim=5)
        proj = model.unet.cond_encoder.proj
        assert proj.weight.shape == (4, 2 * 3 + 5), f"proj shape {proj.weight.shape}"
        B, T, D = 2, 50, 3
        batch = _MockBatch(B=B, T=T, D=D)
        v = model(torch.randn(B, T, D), batch, torch.rand(B))
        assert v.shape == (B, T, D)


class TestTweedieCFM:
    def test_construct_and_nan_obs_safe(self):
        model = TweedieCFM(state_dim=3, hidden_channels=[4, 8], K_inner=3,
                           N_outer=3, cond_extra_dim=0)
        batch = _MockBatch(B=2, T=50, D=3)
        # Stage 1: mean estimator, NaN-masked obs (unobserved steps NaN)
        batch.obs[0, ::2] = float('nan')
        mean = model.estimate_mean(batch.obs)
        assert mean.shape == (2, 50, 3)
        assert torch.isfinite(mean).all(), "estimate_mean produced NaN from NaN obs"

    def test_loss_finite_with_nan_obs(self):
        model = TweedieCFM(state_dim=3, hidden_channels=[4, 8], K_inner=3, N_outer=3)
        model.set_stage(2)
        batch = _MockBatch(B=2, T=50, D=3)
        batch.obs[0, ::2] = float('nan')
        loss = model.compute_loss(batch)
        assert torch.isfinite(loss), "V2 stage-2 loss not finite with NaN obs"

    def test_sample_finite_with_nan_obs(self):
        model = TweedieCFM(state_dim=3, hidden_channels=[4, 8], K_inner=3, N_outer=3)
        model.eval()
        batch = _MockBatch(B=1, T=50, D=3)
        batch.obs[0, ::2] = float('nan')
        with torch.no_grad():
            samples = model.sample(batch, N_outer=3)
        assert samples.shape == (1, 50, 3)
        assert torch.isfinite(samples).all(), "V2 sample produced NaN/Inf"

    def test_estimate_mean_kinner1_no_div_by_zero(self):
        # K_inner=1 must not divide by (K_inner - 1) = 0 in estimate_mean.
        model = TweedieCFM(state_dim=3, hidden_channels=[4, 8], K_inner=1, N_outer=3)
        batch = _MockBatch(B=2, T=50, D=3)
        mean = model.estimate_mean(batch.obs)
        assert mean.shape == (2, 50, 3)
        assert torch.isfinite(mean).all()


class TestTweedieCFMStageDispatch:
    def _model(self, B=2, T=50, D=3):
        return TweedieCFM(state_dim=D, hidden_channels=[4, 8], K_inner=3, N_outer=3)

    def test_default_stage_is_1(self):
        model = self._model()
        assert model._stage == 1
        batch = _MockBatch(B=2, T=50, D=3)
        loss = model.compute_loss(batch)
        assert torch.isfinite(loss), "compute_loss should run without set_stage (regression: no AttributeError)"

    def test_stage1_loss_is_mean_mse(self):
        model = self._model()
        model.set_stage(1)
        model.eval()
        batch = _MockBatch(B=2, T=50, D=3)
        with torch.no_grad():
            loss = model.compute_loss(batch)
            mean = model.estimate_mean(batch.obs)
            expected = F.mse_loss(mean, batch.states)
        assert torch.allclose(loss, expected, atol=1e-6), "stage-1 loss should equal MSE(mean, states)"

    def test_stage2_loss_is_residual_cfm(self):
        model = self._model()
        batch = _MockBatch(B=2, T=50, D=3)
        model.set_stage(1)
        model.eval()
        loss1 = model.compute_loss(batch)
        model.set_stage(2)
        loss2 = model.compute_loss(batch)
        assert torch.isfinite(loss2)
        assert not torch.allclose(loss1, loss2, atol=1e-5), "stage-2 should compute the residual CFM loss, not the mean MSE"

    def test_stage2_val_loss_not_mean_mse(self):
        model = self._model()
        model.set_stage(2)
        model.eval()
        batch = _MockBatch(B=2, T=50, D=3)
        with torch.no_grad():
            loss_val = model.compute_loss(batch)
            mean = model.estimate_mean(batch.obs)
            mean_mse = F.mse_loss(mean, batch.states)
        assert torch.isfinite(loss_val)
        assert not torch.allclose(loss_val, mean_mse, atol=1e-5), \
            "stage-2 validation loss must NOT be the stage-1 mean MSE (the bug being fixed)"

    def test_mean_estimator_frozen_in_stage2(self):
        model = self._model()
        model.set_stage(2)
        model.train()
        # Mirror LitModel.on_train_start's stage-2 freeze of the mean estimator
        for p in model.mean_estimator.parameters():
            p.requires_grad = False
        batch = _MockBatch(B=2, T=50, D=3)
        loss = model.compute_loss(batch)
        loss.backward()
        for name, p in model.mean_estimator.named_parameters():
            assert p.grad is None, f"mean_estimator param '{name}' received gradient during stage 2 (should be frozen)"
        assert any(p.grad is not None and not torch.allclose(p.grad, torch.zeros_like(p.grad))
                   for _, p in model.velocity_unet.named_parameters()), \
            "velocity_unet received no gradient during stage 2"


class TestPredictStateCFM:
    def test_loss_finite_with_nan_obs(self):
        model = PredictStateCFM(state_dim=3, hidden_channels=[4, 8], N_outer=3,
                                param_dim=0, cond_extra_dim=0)
        batch = _MockBatch(B=2, T=50, D=3)
        batch.obs[0, ::2] = float('nan')
        loss = model.compute_loss(batch)
        assert torch.isfinite(loss), "V3 loss not finite with NaN obs"

    def test_sample_finite_with_nan_obs(self):
        model = PredictStateCFM(state_dim=3, hidden_channels=[4, 8], N_outer=3,
                                param_dim=0, cond_extra_dim=0)
        model.eval()
        batch = _MockBatch(B=1, T=50, D=3)
        batch.obs[0, ::2] = float('nan')
        with torch.no_grad():
            samples = model.sample(batch, N_outer=3)
        assert samples.shape == (1, 50, 3)
        assert torch.isfinite(samples).all(), "V3 sample produced NaN/Inf"