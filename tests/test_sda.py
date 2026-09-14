import torch

from models.sda import UnconditionalPriorCFM


class _MockBatch:
    def __init__(self, B=2, T=50, D=3):
        self.states = torch.randn(B, T, D)
        self.obs = torch.randn(B, T, D)
        self.obs_mask = torch.ones(B, T, dtype=torch.bool)
        self.forcing = torch.randn(B, T)
        self.params = torch.randn(B, 4)
        self.batch_size = B


class TestUnconditionalPriorCFM:
    def test_forward_shape(self):
        model = UnconditionalPriorCFM(state_dim=3, hidden_channels=[4, 8])
        B, T, D = 2, 50, 3
        x_t = torch.randn(B, T, D)
        batch = _MockBatch(B=B, T=T, D=D)
        tau = torch.rand(B)
        v = model(x_t, batch, tau)
        assert v.shape == (B, T, D), f"Expected (B,T,D), got {v.shape}"

    def test_loss_finite(self):
        model = UnconditionalPriorCFM(state_dim=3, hidden_channels=[4, 8])
        batch = _MockBatch(B=2, T=50, D=3)
        loss = model.compute_cfm_loss(batch)
        assert torch.isfinite(loss), "Loss is not finite"
        assert loss.ndim == 0, "Loss should be scalar"

    def test_sample_shape(self):
        model = UnconditionalPriorCFM(state_dim=3, hidden_channels=[4, 8], N_outer=3)
        B, T, D = 2, 50, 3
        batch = _MockBatch(B=B, T=T, D=D)
        with torch.no_grad():
            samples = model.sample(batch)
        assert samples.shape == (B, T, D), f"Expected (B,T,D), got {samples.shape}"

    def test_sample_finite(self):
        model = UnconditionalPriorCFM(state_dim=3, hidden_channels=[4, 8], N_outer=3)
        model.eval()
        batch = _MockBatch(B=1, T=50, D=3)
        with torch.no_grad():
            samples = model.sample(batch)
        assert torch.isfinite(samples).all(), "Samples contain NaN or Inf"

    def test_obs_ignored(self):
        """The prior must not read batch.obs/forcing/params at all."""
        model = UnconditionalPriorCFM(state_dim=3, hidden_channels=[4, 8])
        model.eval()
        B, T, D = 2, 50, 3
        x_t = torch.randn(B, T, D)
        tau = torch.rand(B)

        batch_real = _MockBatch(B=B, T=T, D=D)
        batch_garbage = _MockBatch(B=B, T=T, D=D)
        batch_garbage.states = batch_real.states.clone()
        batch_garbage.obs = torch.full((B, T, D), float("nan"))
        batch_garbage.forcing = torch.full((B, T), float("nan"))
        batch_garbage.params = torch.full((B, 4), float("nan"))

        with torch.no_grad():
            v_real = model(x_t, batch_real, tau)
            v_garbage = model(x_t, batch_garbage, tau)
        assert torch.allclose(v_real, v_garbage), \
            "forward() output changed when obs/forcing/params were replaced with NaN"

        with torch.no_grad():
            torch.manual_seed(0)
            loss_real = model.compute_cfm_loss(batch_real)
            torch.manual_seed(0)
            loss_garbage = model.compute_cfm_loss(batch_garbage)
        assert torch.allclose(loss_real, loss_garbage), \
            "compute_cfm_loss changed when obs/forcing/params were replaced with NaN"

    def test_cond_encoder_has_no_obs_proj(self):
        """use_obs=False must be wired all the way to the ConditionEncoder."""
        model = UnconditionalPriorCFM(state_dim=3, hidden_channels=[4, 8])
        assert model.unet.cond_encoder.use_obs is False
        proj = model.unet.cond_encoder.proj
        assert proj.weight.shape == (4, 3), f"proj shape {proj.weight.shape} (expected state_dim-only input)"
