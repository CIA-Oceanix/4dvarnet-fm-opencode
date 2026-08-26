import torch
from torch.utils.data import TensorDataset
from data.dataloader import FlowMatchingBatch
from models.vanilla_cfm import PredictStateCFM


class TestPredictStateCFM:
    def test_forward_shape(self):
        model = PredictStateCFM(state_dim=3, hidden_channels=[4, 8], cond_extra_dim=0)
        B, T, D = 2, 50, 3
        x_t = torch.randn(B, T, D)
        batch = FlowMatchingBatch(
            torch.randn(B, T, D),
            torch.randn(B, T, D),
            torch.ones(B, T, dtype=torch.bool),
            torch.randn(B, T),
        )
        tau = torch.rand(B)
        mu = model(x_t, batch, tau)
        assert mu.shape == (B, T, D), f"Expected (B,T,D), got {mu.shape}"

    def test_forward_cond_extra_dim_gt0(self):
        model = PredictStateCFM(state_dim=3, hidden_channels=[4, 8], param_dim=4,
                                cond_extra_dim=5)
        B, T, D = 2, 50, 3
        x_t = torch.randn(B, T, D)
        batch = FlowMatchingBatch(
            torch.randn(B, T, D),
            torch.randn(B, T, D),
            torch.ones(B, T, dtype=torch.bool),
            torch.randn(B, T),
            params=torch.randn(B, 4),
        )
        tau = torch.rand(B)
        mu = model(x_t, batch, tau)
        assert mu.shape == (B, T, D), f"Expected (B,T,D), got {mu.shape}"

    def test_compute_loss_shape(self):
        model = PredictStateCFM(state_dim=3, hidden_channels=[4, 8])
        B, T, D = 2, 50, 3
        states = torch.randn(B, T, D)
        obs = torch.randn(B, T, D)
        obs_mask = torch.ones(B, T, dtype=torch.bool)
        forcing = torch.randn(B, T)
        batch = FlowMatchingBatch(states, obs, obs_mask, forcing)
        loss = model.compute_loss(batch)
        assert torch.isfinite(loss), "Loss is not finite"
        assert loss.ndim == 0, "Loss should be scalar"

    def test_sample_shape(self):
        model = PredictStateCFM(state_dim=3, hidden_channels=[4, 8], N_outer=3)
        B, T, D = 2, 50, 3
        batch = FlowMatchingBatch(
            torch.randn(B, T, D),
            torch.randn(B, T, D),
            torch.ones(B, T, dtype=torch.bool),
            torch.randn(B, T),
        )
        samples = model.sample(batch)
        assert samples.shape == (B, T, D), f"Expected (B,T,D), got {samples.shape}"

    def test_sample_finite(self):
        model = PredictStateCFM(state_dim=3, hidden_channels=[4, 8], N_outer=3)
        model.eval()
        batch = FlowMatchingBatch(
            torch.randn(1, 50, 3),
            torch.randn(1, 50, 3),
            torch.ones(1, 50, dtype=torch.bool),
            torch.randn(1, 50),
        )
        with torch.no_grad():
            samples = model.sample(batch)
        assert torch.isfinite(samples).all(), "Samples contain NaN or Inf"

    def test_sample_train_tau_0_only(self):
        model = PredictStateCFM(state_dim=3, hidden_channels=[4, 8], train_tau_0_only=True)
        B, T, D = 2, 50, 3
        batch = FlowMatchingBatch(
            torch.randn(B, T, D),
            torch.randn(B, T, D),
            torch.ones(B, T, dtype=torch.bool),
            torch.randn(B, T),
        )
        samples = model.sample(batch)
        assert samples.shape == (B, T, D), f"Expected (B,T,D), got {samples.shape}"

    def test_init_params(self):
        psc = PredictStateCFM(
            state_dim=3,
            hidden_channels=[4, 8],
            time_emb_dim=32,
            N_outer=5,
            sigma_prior=0.3,
            dropout=0.0,
            train_tau_0_only=True,
            param_dim=2,
            cond_extra_dim=1,
        )
        assert psc.param_dim == 2
        assert psc.cond_extra_dim == 1
        assert psc.N_outer == 5
        assert psc.time_emb_dim == 32

    def test_init_default_channels(self):
        model = PredictStateCFM(state_dim=3)
        assert model.hidden_channels == [64, 128, 256]
