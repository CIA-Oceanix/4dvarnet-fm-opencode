import pytest
import torch

from data.qg import QGConfig
from data.qg_neural import QGBatch
from models.monai_unet_qg2d import MonaiVanillaCFM3DQG


@pytest.fixture(autouse=True)
def _no_onednn():
    # As in test_qg_sda_prior_3d.py: oneDNN's first CPU Conv3d setup hangs a small CI runner.
    with torch.backends.mkldnn.flags(enabled=False):
        yield


def _batch(days: int, n: int = 16, b: int = 2) -> QGBatch:
    d = 2 * n * n
    mask = torch.zeros(b, days, d, dtype=torch.bool)
    mask[:, :, : n * n : 5] = True
    obs = torch.where(mask, torch.randn(b, days, d), torch.full((b, days, d), float("nan")))
    return QGBatch(torch.randn(b, days, d), obs, mask, torch.zeros(b, days, n, n),
                   torch.randn(b, days, d) * 1e-6, torch.full((b,), 1.5e4))


def _flow(T: int = 5) -> MonaiVanillaCFM3DQG:
    torch.manual_seed(0)
    flow = MonaiVanillaCFM3DQG(ny=16, nx=16, T=T, hidden_channels=[8, 16, 32, 32], dropout=0.0).eval()
    # MONAI zero-initializes the output conv: without this every check below is 0 == 0.
    with torch.no_grad():
        for p in flow.parameters():
            p.normal_(0.0, 0.1)
    return flow


@pytest.mark.parametrize("days", [5, 15, 30])
def test_one_3d_cfm_runs_on_any_window_length(days):
    batch = _batch(days)
    v = _flow()(torch.randn_like(batch.states), batch, torch.rand(2))
    assert v.shape == batch.states.shape and torch.isfinite(v).all()


def test_3d_cfm_is_periodic_in_space():
    batch = _batch(8)
    flow = _flow()
    x = torch.randn_like(batch.states)
    tau = torch.full((2,), 0.3)

    def roll(z):
        return z.reshape(*z.shape[:2], 2, 16, 16).roll(8, dims=-1).reshape(z.shape)

    shifted = QGBatch(batch.states, roll(batch.obs), roll(batch.obs_mask), batch.forcing,
                      batch.states_q, batch.rd)
    with torch.no_grad():
        out = flow(x, batch, tau)
        assert out.abs().mean() > 1e-3
        assert torch.allclose(flow(roll(x), shifted, tau), roll(out), atol=1e-4)


def test_3d_cfm_conditions_on_obs_and_mask():
    batch = _batch(6)
    flow = _flow()
    x = torch.randn_like(batch.states)
    tau = torch.full((2,), 0.4)
    no_mask = QGBatch(batch.states, batch.obs, torch.zeros_like(batch.obs_mask), batch.forcing,
                      batch.states_q, batch.rd)
    other_obs = QGBatch(batch.states, batch.obs + 1.0, batch.obs_mask, batch.forcing,
                        batch.states_q, batch.rd)
    with torch.no_grad():
        out = flow(x, batch, tau)
        assert not torch.allclose(out, flow(x, no_mask, tau))
        assert not torch.allclose(out, flow(x, other_obs, tau))


def test_build_model_and_training_loss_of_the_3d_cfm():
    from train_qg_neural import CFM_TYPES, FLOW_TYPES, QGNeuralLightning, build_model
    cfg = QGConfig(nx=16, window_days=6.0)
    model = build_model("vanilla_cfm_3d", cfg, hidden_channels=[8, 16, 32, 32])
    assert isinstance(model, MonaiVanillaCFM3DQG) and model.T == 6
    assert "vanilla_cfm_3d" in FLOW_TYPES and "vanilla_cfm_3d" in CFM_TYPES
    lit = QGNeuralLightning(model, "vanilla_cfm_3d", {"mean": torch.zeros(2), "std": torch.ones(2)},
                            cfg, q_loss_weight=0.1, use_cosine_scheduler=False)
    loss, _, _ = lit._total_loss(_batch(6))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_g2_3d_config_matches_g1_3d_capacity_and_protocol():
    from omegaconf import OmegaConf
    from train_qg_neural import build_model
    g2 = OmegaConf.load("config/experiment/G2_vanilla_cfm_3d_specwind_b16_w15_kpow.yaml")
    g1 = OmegaConf.load("config/experiment/G1_direct_unet_3d_specwind_b16_w15_kpow.yaml")
    assert g2.model_type == "vanilla_cfm_3d"
    assert g2.model == g1.model and g2.training == g1.training and g2.data == g1.data
    cfg = QGConfig(nx=64, window_days=15.0)
    n = {c.model_type: sum(p.numel() for p in build_model(
        c.model_type, cfg, hidden_channels=list(c.model.hidden_channels)).parameters())
        for c in (g1, g2)}
    assert 16.7e6 < n["vanilla_cfm_3d"] < 16.9e6
    assert abs(n["vanilla_cfm_3d"] - n["direct_unet_3d"]) < 1e4


def test_3d_cfm_samples_stitched_and_on_the_full_window():
    from evaluation.run_qg_specwind_neural import ComposedFlow, estimate_members
    flow = _flow(T=4)
    flow.N_outer = 3
    batch = _batch(10, b=1)
    full = estimate_members(flow, "vanilla_cfm_3d", batch, n_members=2)
    composed = estimate_members(ComposedFlow(flow, 10, 2), "vanilla_cfm_3d", batch, n_members=2)
    assert full.shape == composed.shape == (2, 1, 10, flow.state_dim)
    assert torch.isfinite(full).all() and torch.isfinite(composed).all()
