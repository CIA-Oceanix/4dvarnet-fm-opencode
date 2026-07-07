import torch
import pytest
from models.vanilla_cfm import VanillaCFM


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