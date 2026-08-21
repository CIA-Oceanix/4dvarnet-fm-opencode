import torch
from models.direct_unet import DirectUNet


class _MockBatch:
    def __init__(self, B=2, T=50, D=3):
        self.states = torch.randn(B, T, D)
        self.obs = torch.randn(B, T, D)
        self.obs_mask = torch.ones(B, T, dtype=torch.bool)
        self.forcing = torch.randn(B, T)
        self.params = torch.randn(B, 4)
        self.batch_size = B


class TestDirectUNet:
    def test_forward_shape(self):
        model = DirectUNet(state_dim=3, hidden_channels=[4, 8])
        B, T, D = 2, 50, 3
        batch = _MockBatch(B=B, T=T, D=D)
        out = model(batch)
        assert out.shape == (B, T, D), f"Expected (B,T,D), got {out.shape}"

    def test_forward_output(self):
        model = DirectUNet(state_dim=3, hidden_channels=[4, 8])
        model.eval()
        batch = _MockBatch(B=1, T=50, D=3)
        with torch.no_grad():
            out = model(batch)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    def test_different_hidden_sizes(self):
        for hc in [[4, 8], [8, 16], [4, 8, 16]]:
            model = DirectUNet(state_dim=3, hidden_channels=hc)
            batch = _MockBatch(B=2, T=50, D=3)
            out = model(batch)
            assert out.shape == (2, 50, 3), f"Failed for hidden_channels={hc}"

    def test_deterministic(self):
        model = DirectUNet(state_dim=3, hidden_channels=[4, 8])
        model.eval()
        batch = _MockBatch(B=1, T=50, D=3)
        with torch.no_grad():
            out1 = model(batch)
            out2 = model(batch)
        assert torch.allclose(out1, out2), "DirectUNet should be deterministic"

    def test_nan_obs_zeroed(self):
        model = DirectUNet(state_dim=3, hidden_channels=[4, 8])
        batch = _MockBatch(B=1, T=50, D=3)
        batch.obs[0, 0] = float('nan')
        with torch.no_grad():
            out = model(batch)
        assert torch.isfinite(out).all(), "NaN in obs should be zeroed"

    def test_cond_extra_dim_0_proj_shape(self):
        model = DirectUNet(state_dim=3, hidden_channels=[4, 8], cond_extra_dim=0)
        proj = model.unet.cond_encoder.proj
        assert proj.weight.shape == (4, 2 * 3 + 0), f"proj shape {proj.weight.shape}"

    def test_cond_extra_dim_gt0_proj_shape(self):
        model = DirectUNet(state_dim=3, hidden_channels=[4, 8], param_dim=4,
                           cond_extra_dim=5)
        proj = model.unet.cond_encoder.proj
        assert proj.weight.shape == (4, 2 * 3 + 5), f"proj shape {proj.weight.shape}"
        B, T, D = 2, 50, 3
        batch = _MockBatch(B=B, T=T, D=D)
        out = model(batch)
        assert out.shape == (B, T, D)