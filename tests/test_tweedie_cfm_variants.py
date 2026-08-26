"""Tests for V2 TweedieCFM and V3 PredictStateCFM variants."""

import torch
import numpy as np
from data.dataloader import FlowMatchingBatch
from models.vanilla_cfm import TweedieCFM, PredictStateCFM


def test_tweedie_cfm_init():
    """Test TweedieCFM initialization with L96 config."""
    model = TweedieCFM(
        state_dim=24,
        hidden_channels=[64, 128, 256],
        time_emb_dim=64,
        K_inner=5,
        N_outer=10,
        sigma_prior=0.5,
        dropout=0.1,
        train_tau_0_only=False,
    )
    assert hasattr(model, "mean_estimator")
    assert hasattr(model, "velocity_unet")
    assert hasattr(model, "interpolant")
    assert model.K_inner == 5
    assert model.N_outer == 10


def test_tweedie_cfm_estimate_mean():
    """Test mean estimation returns correct shape."""
    obs = torch.randn(2, 3000, 24)
    model = TweedieCFM(state_dim=24, hidden_channels=[64, 128, 256], K_inner=5)
    model.eval()
    with torch.no_grad():
        mean = model.estimate_mean(obs)
    assert mean.shape == (2, 3000, 24), f"Expected (2, 3000, 24), got {mean.shape}"


def test_tweedie_cfm_forward():
    """Test forward pass for velocity prediction."""
    x_t = torch.randn(2, 3000, 24)  # Noised residual
    obs = torch.randn(2, 3000, 24)
    mean = torch.randn(2, 3000, 24)
    tau = torch.rand(2)
    model = TweedieCFM(state_dim=24, hidden_channels=[64, 128, 256])
    model.eval()
    with torch.no_grad():
        v = model.forward(x_t, obs, mean, tau)
    assert v.shape == (2, 3000, 24), f"Expected (2, 3000, 24), got {v.shape}"


def test_tweedie_cfm_compute_loss():
    """Test two-stage loss computation."""
    B = 32
    T = 3000
    D = 24
    batch = FlowMatchingBatch(
        torch.randn(B, T, D),
        torch.randn(B, T, D),
        torch.ones(B, T, dtype=torch.bool),
        torch.randn(B, T, 2),
    )
    model = TweedieCFM(state_dim=24, hidden_channels=[64, 128, 256])
    model.train()
    set_stage = torch.cuda.is_available()
    loss = model.compute_loss(batch)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.dim() == 0, "Loss should be scalar"
    assert torch.isfinite(loss).all(), "Loss should be finite"

    if set_stage:
        model.set_stage(2)
        model.eval()
        with torch.no_grad():
            v = model.compute_loss(batch)
        assert torch.isfinite(v).all(), "Stage 2 loss should be finite"


def test_tweedie_cfm_sample():
    """Sample should produce correct shape."""
    obs = torch.randn(2, 3000, 24)
    forcing = torch.randn(2, 3000)
    params = torch.ones(2, 3000, 1)
    true_params = torch.ones(2, 3000, 1)
    batch = FlowMatchingBatch(
        torch.randn(2, 3000, 24),
        obs,
        torch.ones(2, 3000, dtype=torch.bool),
        forcing,
        params=params,
        true_params=true_params,
    )

    model = TweedieCFM(state_dim=24, hidden_channels=[64, 128, 256])
    model.eval()
    with torch.no_grad():
        samples = model.sample(batch, N_outer=10)
    assert samples.shape == (2, 3000, 24), f"Expected (2, 3000, 24), got {samples.shape}"


def test_tweedie_cfm_set_stage():
    """Test stage setting."""
    model = TweedieCFM(state_dim=24, hidden_channels=[64, 128, 256])
    assert not hasattr(model, "_stage")
    model.set_stage(1)
    assert model._stage == 1
    model.set_stage(2)
    assert model._stage == 2


def test_predict_state_cfm_init():
    """Test PredictStateCFM initialization with L96 config."""
    model = PredictStateCFM(
        state_dim=24,
        hidden_channels=[64, 128, 256],
        time_emb_dim=64,
        N_outer=10,
        sigma_prior=0.5,
        dropout=0.1,
        train_tau_0_only=False,
        param_dim=0,
    )
    assert hasattr(model, "unet")
    assert hasattr(model, "interpolant")
    assert model.N_outer == 10
    assert model.state_dim == 24


def test_predict_state_cfm_forward():
    """Test forward pass produces correct shape."""
    x_t = torch.randn(2, 3000, 24)
    forcing = torch.randn(2, 3000)
    params = torch.ones(2, 3000, 4)
    true_params = torch.ones(2, 3000, 4)
    batch = FlowMatchingBatch(
        torch.randn(2, 3000, 24),
        x_t,
        torch.ones(2, 3000, dtype=torch.bool),
        forcing,
        params=params,
        true_params=true_params,
    )
    tau = torch.rand(2)
    model = PredictStateCFM(state_dim=24, hidden_channels=[64, 128, 256])
    model.eval()
    with torch.no_grad():
        mu = model.forward(x_t, batch, tau)
    assert mu.shape == (2, 3000, 24), f"Expected (2, 3000, 24), got {mu.shape}"


def test_predict_state_cfm_compute_loss():
    """T own V3 loss computation."""
    B = 32
    T = 3000
    D = 24
    batch = FlowMatchingBatch(
        torch.randn(B, T, D),
        torch.randn(B, T, D),
        torch.ones(B, T, dtype=torch.bool),
        torch.randn(B, T, 2),
    )
    model = PredictStateCFM(state_dim=24, hidden_channels=[64, 128, 256])
    model.train()
    loss = model.compute_loss(batch)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.dim() == 0, "Loss should be scalar"
    assert torch.isfinite(loss).all(), "Loss should be finite"


def test_predict_state_cfm_sample():
    """Test sampling with correct ODE integration."""
    obs = torch.randn(2, 3000, 24)
    forcing = torch.randn(2, 3000)
    params = torch.ones(2, 3000, 1)
    true_params = torch.ones(2, 3000, 1)
    batch = FlowMatchingBatch(
        torch.randn(2, 3000, 24),
        obs,
        torch.ones(2, 3000, dtype=torch.bool),
        forcing,
        params=params,
        true_params=true_params,
    )
    model = PredictStateCFM(state_dim=24, hidden_channels=[64, 128, 256])
    model.eval()
    with torch.no_grad():
        samples = model.sample(batch, N_outer=10)
    assert samples.shape == (2, 3000, 24), f"Expected (2, 3000, 24), got {samples.shape}"
    assert torch.isfinite(samples).all(), "Samples should be finite"


def test_predict_state_cfm_sample_tau_0_only():
    """Test tau=0 shortcut."""
    obs = torch.randn(2, 3000, 24)
    forcing = torch.randn(2, 3000)
    params = torch.ones(2, 3000, 1)
    true_params = torch.ones(2, 3000, 1)
    batch = FlowMatchingBatch(
        torch.randn(2, 3000, 24),
        obs,
        torch.ones(2, 3000, dtype=torch.bool),
        forcing,
        params=params,
        true_params=true_params,
    )
    model = PredictStateCFM(
        state_dim=24,
        hidden_channels=[64, 128, 256],
        train_tau_0_only=True,
    )
    model.eval()
    with torch.no_grad():
        x0 = torch.randn_like(obs) * model.sigma_prior
        mu = model.forward(x0, batch, tau=torch.zeros(2, device=obs.device))
        result = model.sample(batch, N_outer=1)
    assert result.shape == (2, 3000, 24)
    assert torch.isfinite(result).all()


def test_output_shape_alignment():
    """Both models output (B, T, D)."""
    B, T, D = 8, 1000, 24
    obs = torch.randn(B, T, D)
    forcing = torch.randn(B, T, 2)
    params = torch.ones(B, T, 1)
    true_params = torch.ones(B, T, 1)
    batch = FlowMatchingBatch(
        torch.randn(B, T, D),
        obs,
        torch.ones(B, T, dtype=torch.bool),
        forcing,
        params=params,
        true_params=true_params,
    )
    
    for model in [
        TweedieCFM(state_dim=D, hidden_channels=[32, 64, 128], K_inner=5),
        PredictStateCFM(state_dim=D, hidden_channels=[32, 64, 128]),
    ]:
        model.eval()
        with torch.no_grad():
            # Test forward (TweedieCFM stage 1 mean)
            if isinstance(model, TweedieCFM):
                mean = model.estimate_mean(obs)
                assert mean.shape == (B, T, D)
            else:
                forward_out = model.forward(obs, batch, tau=torch.zeros(B))
                assert forward_out.shape == (B, T, D)
            
            # Test sample
            samples = model.sample(batch, N_outer=5)
            assert samples.shape == (B, T, D)
    print("✓ output_shape_alignment passed")


if __name__ == "__main__":
    print("Running TweedieCFM + PredictStateCFM tests...")
    test_tweedie_cfm_init()
    test_tweedie_cfm_estimate_mean()
    test_tweedie_cfm_forward()
    test_tweedie_cfm_compute_loss()
    test_tweedie_cfm_sample()
    test_tweedie_cfm_set_stage()
    print("  ✓ tweedie_cfm tests passed")
    
    test_predict_state_cfm_init()
    test_predict_state_cfm_forward()
    test_predict_state_cfm_compute_loss()
    test_predict_state_cfm_sample()
    test_predict_state_cfm_sample_tau_0_only()
    print("  ✓ predict_state_cfm tests passed")
    
    test_output_shape_alignment()
    print("\n✅ All tests passed!")
