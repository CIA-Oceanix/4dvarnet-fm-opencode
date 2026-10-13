import pytest
import torch

from data.qg import QGConfig
from data.qg_neural import QGBatch
from models.monai_unet_qg2d import MonaiDirectUNet3DQG


@pytest.fixture(autouse=True)
def _no_onednn():
    # As in test_qg_sda_prior_3d.py: oneDNN's first CPU Conv3d setup hangs a small CI runner.
    with torch.backends.mkldnn.flags(enabled=False):
        yield


def _batch(days: int, n: int = 16, b: int = 2) -> QGBatch:
    d = 2 * n * n
    mask = torch.zeros(b, days, d, dtype=torch.bool)
    mask[:, :, : n * n : 5] = True
    obs = torch.where(mask, torch.randn(b, days, d), torch.zeros(b, days, d))
    return QGBatch(torch.randn(b, days, d), obs, mask, torch.zeros(b, days, n, n),
                   torch.randn(b, days, d) * 1e-6, torch.full((b,), 1.5e4))


def _randomized(net: torch.nn.Module) -> torch.nn.Module:
    # MONAI zero-initializes the U-Net's output conv, so a fresh network outputs exactly 0 and
    # any equivariance or input-dependence check would pass vacuously.
    with torch.no_grad():
        for p in net.parameters():
            p.normal_(0.0, 0.1)
    return net


def _net(T: int = 5) -> MonaiDirectUNet3DQG:
    torch.manual_seed(0)
    return _randomized(MonaiDirectUNet3DQG(ny=16, nx=16, T=T, hidden_channels=[8, 16, 32, 32],
                                           dropout=0.0)).eval()


@pytest.mark.parametrize("days", [5, 15, 30])
def test_one_3d_direct_unet_runs_on_any_window_length(days):
    batch = _batch(days)
    out = _net()(batch)
    assert out.shape == batch.states.shape and torch.isfinite(out).all()


def test_3d_direct_unet_is_periodic_in_space():
    batch = _batch(8)
    net = _net()

    def roll(z):
        return z.reshape(*z.shape[:2], 2, 16, 16).roll(8, dims=-1).reshape(z.shape)

    shifted = QGBatch(batch.states, roll(batch.obs), roll(batch.obs_mask), batch.forcing,
                      batch.states_q, batch.rd)
    with torch.no_grad():
        out = net(batch)
        assert out.abs().mean() > 1e-3
        assert torch.allclose(net(shifted), roll(out), atol=1e-4)


def test_3d_direct_unet_reads_the_obs_mask():
    batch = _batch(6)
    net = _net()
    unmasked = QGBatch(batch.states, batch.obs, torch.zeros_like(batch.obs_mask), batch.forcing,
                       batch.states_q, batch.rd)
    with torch.no_grad():
        assert not torch.allclose(net(batch), net(unmasked))


def test_build_model_and_training_loss_of_the_3d_direct_unet():
    from train_qg_neural import DIRECT_TYPES, QGNeuralLightning, build_model
    cfg = QGConfig(nx=16, window_days=6.0)
    model = build_model("direct_unet_3d", cfg, hidden_channels=[8, 16, 32, 32])
    assert "direct_unet_3d" in DIRECT_TYPES and model.T == 6
    lit = QGNeuralLightning(model, "direct_unet_3d", {"mean": torch.zeros(2), "std": torch.ones(2)},
                            cfg, q_loss_weight=0.1, use_cosine_scheduler=False)
    loss, _, _ = lit._total_loss(_batch(6))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_size_matched_3d_direct_unet_has_about_g1_m_parameters():
    from train_qg_neural import build_model
    model = build_model("direct_unet_3d", QGConfig(nx=64, window_days=15.0),
                        hidden_channels=[32, 64, 128, 128])
    assert 16.7e6 < sum(p.numel() for p in model.parameters()) < 16.9e6


def test_activation_monitor_finds_the_decoder_of_every_monai_backbone():
    from models.monai_unet_qg2d import MonaiSDAPrior3DQG
    from train_qg_neural import build_model, decoder_output_module
    cfg = QGConfig(nx=16, window_days=4.0)
    for model in (_net(), MonaiSDAPrior3DQG(ny=16, nx=16, T=4, hidden_channels=[8, 16, 32]),
                  build_model("direct_unet_tchannels", cfg, hidden_channels=[8, 16, 32])):
        assert decoder_output_module(model) is not None
    assert decoder_output_module(torch.nn.Linear(2, 2)) is None
