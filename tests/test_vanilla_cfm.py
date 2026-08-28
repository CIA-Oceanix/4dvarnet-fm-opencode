import torch

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