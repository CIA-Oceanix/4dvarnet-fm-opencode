"""Early-fine Euler grid for VanillaCFM / PredictStateCFM sampling, and the T5 second-order
(variance) consistency loss. See docs/plans/analysis/l96_cfm_tau_consistency_next_steps.md v4."""
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from models.monai_unet_adapter import MonaiPredictStateCFM, MonaiVanillaCFM
from models.vanilla_cfm import DEFAULT_STEP_POWER, PredictStateCFM, flow_tau_grid


def _batch(B: int = 3, T: int = 32, D: int = 4) -> SimpleNamespace:
    return SimpleNamespace(obs=torch.randn(B, T, D), forcing=torch.randn(B, T),
                           params=None, states=torch.randn(B, T, D))


def _randomize(model: torch.nn.Module) -> torch.nn.Module:
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.2)
    return model


def _psc(**opts) -> MonaiPredictStateCFM:
    torch.manual_seed(7)
    return MonaiPredictStateCFM(state_dim=4, hidden_channels=[16, 32], N_outer=4, dropout=0.0,
                                param_dim=0, num_res_blocks=1, norm_num_groups=8, **opts)


def test_flow_tau_grid():
    assert flow_tau_grid(4, 1.0) == [0.0, 0.25, 0.5, 0.75, 1.0]
    g = flow_tau_grid(10)
    gaps = [b - a for a, b in zip(g[:-1], g[1:])]
    assert g[0] == 0.0 and g[-1] == 1.0 and gaps[0] < gaps[-1]
    assert DEFAULT_STEP_POWER == 0.5


def test_psc_step_power_one_reproduces_the_uniform_sampler():
    model, batch = _randomize(_psc()).eval(), _batch()
    torch.manual_seed(0)
    with torch.no_grad():
        new = model.sample(batch, N_outer=4, step_power=1.0)
    torch.manual_seed(0)
    with torch.no_grad():
        B, T, _ = batch.obs.shape
        x = torch.randn_like(batch.obs) * model.sigma_prior
        for step in range(4):
            tau = torch.full((B,), step / 4)
            mu = model.forward(x, batch, tau)
            x = x + 0.25 * (mu - x) / (1.0 - tau.clamp(max=0.999).view(B, 1, 1).expand(-1, T, -1))
    assert torch.allclose(new, x, atol=1e-6)


def test_vanilla_step_power_one_reproduces_the_uniform_sampler():
    torch.manual_seed(7)
    model = _randomize(MonaiVanillaCFM(state_dim=4, hidden_channels=[16, 32], N_outer=4, dropout=0.0,
                                       param_dim=0, cond_extra_dim=0, num_res_blocks=1,
                                       norm_num_groups=8)).eval()
    batch = _batch()
    torch.manual_seed(0)
    with torch.no_grad():
        new = model.sample(batch, N_outer=4, step_power=1.0)
    torch.manual_seed(0)
    with torch.no_grad():
        x = torch.randn_like(batch.obs) * model.sigma_prior
        for step in range(4):
            x = x + 0.25 * model.forward(x, batch, torch.full((batch.obs.shape[0],), step / 4))
    assert torch.allclose(new, x, atol=1e-6)


def test_default_sampler_uses_the_early_fine_grid_and_attribute_overrides_it():
    model, batch = _psc().eval(), _batch()
    seen = []
    orig = model.forward
    model.forward = lambda x, b, tau: (seen.append(float(tau[0])), orig(x, b, tau))[1]
    with torch.no_grad():
        model.sample(batch, N_outer=5)
    assert seen == pytest.approx(flow_tau_grid(5, 0.5)[:-1])
    seen.clear()
    model.step_power = 1.0
    with torch.no_grad():
        model.sample(batch, N_outer=5)
    assert seen == pytest.approx(flow_tau_grid(5, 1.0)[:-1])


def test_variance_terms_match_an_autograd_jvp():
    model, batch = _randomize(_psc(var_weight=1.0)), _batch()
    x1 = batch.states
    tau = torch.tensor([0.2, 0.5, 0.8])
    x_tau = model.interpolant.mix(torch.randn_like(x1) * model.sigma_prior, x1, tau)
    torch.manual_seed(11)
    jac, _ = model.variance_terms(x_tau, batch, tau, x1)
    torch.manual_seed(11)
    u = torch.randint(0, 2, x_tau.shape).float() * 2 - 1
    model.eval()
    xg = x_tau.clone().requires_grad_(True)
    v = torch.zeros_like(x_tau, requires_grad=True)
    g = torch.autograd.grad(model.forward(xg, batch, tau), xg, v, create_graph=True)[0]
    ju = torch.autograd.grad(g, v, u)[0]
    w = ((1 - tau) * model.sigma_prior) ** 2 / tau
    exact = (w.view(-1, 1, 1) * u * ju).mean(dim=1)
    assert ((jac - exact).norm() / exact.norm()).item() < 1e-2


class _GaussianExact(PredictStateCFM):
    """Exact D for x1 ~ N(m, p I), no conditioning: D = m + g (x - tau m), g = tau p / (b^2 + tau^2 p)."""
    m, p = 0.7, 0.3

    def forward(self, x_t, batch, tau):
        t = tau.view(-1, 1, 1)
        b2 = ((1 - t) * self.sigma_prior) ** 2
        return self.m + t * self.p / (b2 + t ** 2 * self.p) * (x_t - t * self.m)


def test_the_identity_holds_for_the_exact_gaussian_operator():
    model = _GaussianExact(state_dim=1, hidden_channels=[8], N_outer=2, dropout=0.0,
                           param_dim=0, var_weight=1.0)
    torch.manual_seed(5)
    n, T = 4000, 16
    x1 = model.m + model.p ** 0.5 * torch.randn(n, T, 1)
    batch = SimpleNamespace(obs=torch.zeros(n, T, 1), forcing=torch.zeros(n, T), params=None, states=x1)
    for t in (0.1, 0.5, 0.9):
        tau = torch.full((n,), t)
        x_tau = model.interpolant.mix(torch.randn_like(x1) * model.sigma_prior, x1, tau)
        jac, resid = model.variance_terms(x_tau, batch, tau, x1)
        assert jac.mean().item() == pytest.approx(resid.mean().item(), rel=3e-2)


def test_variance_terms_ignore_dropout_and_restore_the_mode():
    torch.manual_seed(7)
    model = _randomize(MonaiPredictStateCFM(state_dim=4, hidden_channels=[16, 32], N_outer=4,
                                            dropout=0.5, param_dim=0, num_res_blocks=1,
                                            norm_num_groups=8, var_weight=1.0))
    model.train()
    batch = _batch()
    tau = torch.full((3,), 0.4)
    x_tau = torch.randn_like(batch.states)
    torch.manual_seed(3)
    a, _ = model.variance_terms(x_tau, batch, tau, batch.states)
    torch.manual_seed(3)
    b, _ = model.variance_terms(x_tau, batch, tau, batch.states)
    assert torch.equal(a, b) and model.training


def test_var_loss_trains_and_eval_loss_is_unchanged():
    batch = _batch()
    plain, t5 = _randomize(_psc()), _randomize(_psc(var_weight=10.0))
    t5.load_state_dict(plain.state_dict())
    t5.train()
    loss = t5.compute_loss(batch)
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None for p in t5.parameters())
    plain.eval()
    t5.eval()
    torch.manual_seed(9)
    a = plain.compute_loss(batch)
    torch.manual_seed(9)
    b = t5.compute_loss(batch)
    assert torch.equal(a, b)
    assert torch.isfinite(t5.variance_ratio(batch))


def test_invalid_var_options_raise():
    with pytest.raises(ValueError):
        _psc(var_weight=-1.0)
    with pytest.raises(ValueError):
        _psc(var_tau_min=0.0)


def test_train_py_wires_var_options_and_step_power():
    from train import model_factory
    section = {"hidden_channels": [16, 32], "N_outer": 3, "sigma_prior": 0.5, "dropout": 0.0,
               "cond_extra_dim": 0, "num_res_blocks": 1, "norm_num_groups": 8,
               "var_weight": 10.0, "var_tau_min": 0.1, "step_power": 1.0}
    cfg = OmegaConf.create({"model": {"model_type": "monai_predict_state_cfm", "state_dim": 4,
                                      "param_dim": 0, "monai_predict_state_cfm": section}})
    m = model_factory(cfg, torch.device("cpu"))
    assert m.var_weight == 10.0 and m.var_tau_min == 0.1 and m.step_power == 1.0


def test_var_start_epoch_gates_the_loss():
    from training.lightning_module import LitModel
    model = _randomize(_psc(var_weight=10.0))
    lit = LitModel(model, model_type="monai_predict_state_cfm", use_cosine_scheduler=False,
                   var_start_epoch=5)
    lit.on_train_epoch_start()
    assert model.var_active is False
    model.train()
    batch = _batch()
    plain = _randomize(_psc())
    plain.load_state_dict(model.state_dict())
    plain.train()
    torch.manual_seed(1)
    gated = model.compute_loss(batch)
    torch.manual_seed(1)
    x1 = batch.states
    B = x1.shape[0]
    torch.rand(B)
    tau = torch.rand(B)
    x_tau = plain.interpolant.mix(torch.randn_like(x1) * plain.sigma_prior, x1, tau)
    assert torch.allclose(gated, torch.nn.functional.mse_loss(plain.forward(x_tau, batch, tau), x1))


def test_train_py_passes_var_start_epoch():
    from train import psc_teacher_options
    cfg = OmegaConf.create({"model": {"model_type": "monai_predict_state_cfm",
                                      "monai_predict_state_cfm": {"var_weight": 10.0, "var_start_epoch": 50}}})
    assert psc_teacher_options(cfg) == {"var_start_epoch": 50}
