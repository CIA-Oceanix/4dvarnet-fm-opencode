import pytest
import torch

from data.qg import QGConfig
from models.monai_unet_qg2d import MonaiSDAPrior3DQG, SpaceCircularConv3d


@pytest.fixture(autouse=True)
def _no_onednn():
    # oneDNN's first CPU Conv3d spends minutes building its kernels on a small runner (it hung
    # the CI pytest job); the native kernels run these tiny shapes in well under a second.
    with torch.backends.mkldnn.flags(enabled=False):
        yield


def _prior(T: int = 5) -> MonaiSDAPrior3DQG:
    torch.manual_seed(0)
    prior = MonaiSDAPrior3DQG(ny=16, nx=16, T=T, hidden_channels=[8, 16, 32], dropout=0.0).eval()
    # MONAI zero-initializes the output conv: without this the periodicity check is 0 == 0.
    with torch.no_grad():
        for p in prior.parameters():
            p.normal_(0.0, 0.1)
    return prior


@pytest.mark.parametrize("days", [5, 15, 30])
def test_one_3d_prior_runs_on_any_window_length(days):
    model = _prior()
    x = torch.randn(2, days, model.state_dim)
    v = model(x, None, torch.rand(2))
    assert v.shape == x.shape and torch.isfinite(v).all()


def test_3d_prior_is_periodic_in_space():
    model = _prior()
    x = torch.randn(1, 8, model.state_dim)
    tau = torch.full((1,), 0.3)

    def roll(z):
        return z.reshape(1, 8, 2, 16, 16).roll(4, dims=-1).reshape(1, 8, -1)

    with torch.no_grad():
        out = model(x, None, tau)
        assert out.abs().mean() > 1e-3
        assert torch.allclose(model(roll(x), None, tau), roll(out), atol=1e-4)


def test_3d_prior_does_not_wrap_in_time():
    conv = SpaceCircularConv3d(1, 1, kernel_size=3, padding=1, bias=False)
    with torch.no_grad():
        conv.weight.fill_(1.0)
    x = torch.zeros(1, 1, 4, 3, 3)
    x[0, 0, -1] = 1.0
    out = conv(x)
    assert out[0, 0, 0].abs().max() == 0.0
    assert torch.allclose(out[0, 0, -1], torch.full((3, 3), 9.0))


def test_build_model_3d_prior_and_its_training_loss():
    from data.qg_neural import QGBatch
    from train_qg_neural import QGNeuralLightning, build_model
    cfg = QGConfig(nx=16, window_days=6.0)
    model = build_model("sda_prior_3d", cfg, hidden_channels=[8, 16, 32])
    assert isinstance(model, MonaiSDAPrior3DQG) and model.T == 6
    b, d = 2, cfg.state_dim
    batch = QGBatch(torch.randn(b, 6, d), torch.zeros(b, 6, d), torch.zeros(b, 6, d, dtype=torch.bool),
                    torch.zeros(b, 6, 16, 16), torch.randn(b, 6, d) * 1e-6, torch.full((b,), 1.5e4))
    lit = QGNeuralLightning(model, "sda_prior_3d", {"mean": torch.zeros(2), "std": torch.ones(2)},
                            cfg, q_loss_weight=0.1, use_cosine_scheduler=False)
    loss, _, _ = lit._total_loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_narrow_3d_prior_has_ten_million_parameters():
    from train_qg_neural import build_model
    model = build_model("sda_prior_3d", QGConfig(nx=64, window_days=15.0), hidden_channels=[32, 64, 128])
    assert 9.9e6 < sum(p.numel() for p in model.parameters()) < 10.1e6


def test_3d_prior_runs_in_the_guided_sampler_on_the_full_window():
    from types import SimpleNamespace

    from evaluation.run_qg_specwind_neural import ComposedFlow, sda_members
    model = _prior(T=4)
    d = model.state_dim
    mask = torch.zeros(1, 10, d, dtype=torch.bool)
    mask[:, :, :256:7] = True
    batch = SimpleNamespace(obs=torch.where(mask, 1.0, 0.0), obs_mask=mask)
    full = sda_members(model, batch, n_members=2, guidance_weight=5.0, n_steps=3)
    composed = sda_members(ComposedFlow(model, 10, 2), batch, n_members=2, guidance_weight=5.0,
                           n_steps=3)
    assert full.shape == composed.shape == (2, 1, 10, d)
