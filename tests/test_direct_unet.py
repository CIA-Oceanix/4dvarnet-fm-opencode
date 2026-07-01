import torch
import pytest
from models.direct_unet import DirectUNet


class TestDirectUNet:
    def test_forward_shape(self):
        model = DirectUNet(state_dim=3, hidden_channels=[4, 8])
        B, T, D = 2, 50, 3
        obs = torch.randn(B, T, D)
        out = model(obs)
        assert out.shape == (B, T, D), f"Expected (B,T,D), got {out.shape}"

    def test_forward_output(self):
        model = DirectUNet(state_dim=3, hidden_channels=[4, 8])
        model.eval()
        obs = torch.randn(1, 50, 3)
        with torch.no_grad():
            out = model(obs)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    def test_different_hidden_sizes(self):
        for hc in [[4, 8], [8, 16], [4, 8, 16]]:
            model = DirectUNet(state_dim=3, hidden_channels=hc)
            obs = torch.randn(2, 50, 3)
            out = model(obs)
            assert out.shape == (2, 50, 3), f"Failed for hidden_channels={hc}"

    def test_deterministic(self):
        model = DirectUNet(state_dim=3, hidden_channels=[4, 8])
        model.eval()
        obs = torch.randn(1, 50, 3)
        with torch.no_grad():
            out1 = model(obs)
            out2 = model(obs)
        assert torch.allclose(out1, out2), "DirectUNet should be deterministic"
