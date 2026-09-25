"""PredictStateCFM linear skip connection: D = tau * x_tau + (1 - tau) * net."""
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from models.monai_unet_adapter import MonaiPredictStateCFM


def _batch(B: int = 4, T: int = 32, D: int = 4) -> SimpleNamespace:
    return SimpleNamespace(obs=torch.randn(B, T, D), forcing=torch.randn(B, T),
                           params=None, states=torch.randn(B, T, D))


def _model(**opts) -> MonaiPredictStateCFM:
    torch.manual_seed(123)
    m = MonaiPredictStateCFM(state_dim=4, hidden_channels=[16, 32], N_outer=3, dropout=0.0,
                             param_dim=0, num_res_blocks=1, norm_num_groups=8, **opts)
    return m.eval()


def _net(model: MonaiPredictStateCFM, x: torch.Tensor, batch: SimpleNamespace, tau: torch.Tensor) -> torch.Tensor:
    return model.unet(x.transpose(1, 2), batch.obs.transpose(1, 2), tau=tau).transpose(1, 2)


def test_default_is_the_plain_network_output():
    model, batch = _model(), _batch()
    x, tau = torch.randn(4, 32, 4), torch.rand(4)
    assert model.skip_connection == "none"
    assert torch.equal(model.forward(x, batch, tau), _net(model, x, batch, tau))


def test_linear_skip_blends_input_and_network_output():
    model, batch = _model(skip_connection="linear"), _batch()
    x, tau = torch.randn(4, 32, 4), torch.tensor([0.0, 0.25, 0.6, 1.0])
    t = tau.view(-1, 1, 1)
    expected = t * x + (1 - t) * _net(model, x, batch, tau)
    assert torch.allclose(model.forward(x, batch, tau), expected, atol=1e-6)


def test_linear_skip_is_exact_at_tau_one_and_leaves_the_tau_zero_head_unchanged():
    skip, plain, batch = _model(skip_connection="linear"), _model(), _batch()
    x = torch.randn(4, 32, 4)
    assert torch.equal(skip.forward(x, batch, torch.ones(4)), x)
    assert torch.allclose(skip.forward(x, batch, torch.zeros(4)), plain.forward(x, batch, torch.zeros(4)))


def test_linear_skip_sampler_velocity_is_net_minus_x():
    model, batch = _model(skip_connection="linear"), _batch()
    x, tau = torch.randn(4, 32, 4), torch.full((4,), 0.4)
    v = (model.forward(x, batch, tau) - x) / (1 - 0.4)
    assert torch.allclose(v, _net(model, x, batch, tau) - x, atol=1e-5)


def test_linear_skip_trains_and_samples():
    model, batch = _model(skip_connection="linear"), _batch()
    model.train()
    loss = model.compute_loss(batch)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None for p in model.unet.parameters() if p.requires_grad)
    model.eval()
    with torch.no_grad():
        out = model.sample(batch, N_outer=5)
    assert out.shape == batch.obs.shape and torch.isfinite(out).all()


def test_invalid_skip_connection_raises():
    with pytest.raises(ValueError):
        _model(skip_connection="edm")


def test_train_py_wires_skip_connection():
    from train import model_factory
    section = {"hidden_channels": [16, 32], "N_outer": 3, "sigma_prior": 0.5, "dropout": 0.0,
               "cond_extra_dim": 0, "num_res_blocks": 1, "norm_num_groups": 8, "skip_connection": "linear",
               "skip_loss": "net"}
    cfg = OmegaConf.create({"model": {"model_type": "monai_predict_state_cfm", "state_dim": 4,
                                      "param_dim": 0, "monai_predict_state_cfm": section}})
    m = model_factory(cfg, torch.device("cpu"))
    assert m.skip_connection == "linear" and m.skip_loss == "net"


def test_net_loss_is_the_unweighted_skip_target():
    model, batch = _model(skip_connection="linear", skip_loss="net"), _batch()
    model.train()
    torch.manual_seed(0)
    loss = model.compute_loss(batch)
    torch.manual_seed(0)
    tau = torch.rand(4)
    x0 = torch.randn_like(batch.states) * model.sigma_prior
    x_tau = model.interpolant.mix(x0, batch.states, tau)
    t = tau.view(-1, 1, 1)
    net = _net(model, x_tau, batch, tau)
    assert torch.allclose(loss, ((net - ((1 + t) * batch.states - t * x0)) ** 2).mean(), atol=1e-6)
    d_residual = model.forward(x_tau, batch, tau) - batch.states
    assert torch.allclose(d_residual, (1 - t) * (net - ((1 + t) * batch.states - t * x0)), atol=1e-5)


def test_validation_loss_is_the_state_loss_for_both_skip_losses():
    batch = _batch()
    state, net = _model(skip_connection="linear"), _model(skip_connection="linear", skip_loss="net")
    torch.manual_seed(2)
    a = state.compute_loss(batch)
    torch.manual_seed(2)
    b = net.compute_loss(batch)
    assert torch.equal(a, b)


@pytest.mark.parametrize("bad", [{"skip_loss": "net"}, {"skip_connection": "linear", "skip_loss": "velocity"},
                                 {"skip_connection": "linear", "skip_loss": "net", "tau0_frac": 0.1}])
def test_invalid_skip_loss_raises(bad):
    with pytest.raises(ValueError):
        _model(**bad)
