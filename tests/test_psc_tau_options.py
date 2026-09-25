"""PredictStateCFM tau-consistency training options (T1', T1, T2a/T2b, T4a) and the
LitModel EMA teacher. See docs/plans/analysis/l96_cfm_tau_consistency_next_steps.md."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from models.monai_unet_adapter import MonaiPredictStateCFM
from training.lightning_module import LitModel


def _batch(B: int = 4, T: int = 32, D: int = 4) -> SimpleNamespace:
    return SimpleNamespace(obs=torch.randn(B, T, D), forcing=torch.randn(B, T),
                           params=None, states=torch.randn(B, T, D))


def _model(**opts) -> MonaiPredictStateCFM:
    torch.manual_seed(123)
    return MonaiPredictStateCFM(state_dim=4, hidden_channels=[16, 32], N_outer=3, dropout=0.0,
                                param_dim=0, num_res_blocks=1, norm_num_groups=8, **opts)


def _reference_loss(model: MonaiPredictStateCFM, batch: SimpleNamespace) -> torch.Tensor:
    B = batch.obs.shape[0]
    tau = torch.rand(B)
    x0 = torch.randn_like(batch.states) * model.sigma_prior
    x_tau = model.interpolant.mix(x0, batch.states, tau)
    return F.mse_loss(model.forward(x_tau, batch, tau), batch.states)


def test_defaults_reproduce_the_plain_loss_bit_for_bit():
    model, batch = _model(), _batch()
    model.train()
    torch.manual_seed(0)
    new = model.compute_loss(batch)
    torch.manual_seed(0)
    ref = _reference_loss(model, batch)
    assert torch.equal(new, ref)


def test_eval_mode_ignores_tau_options():
    batch = _batch()
    plain, opt = _model(), _model(tau0_frac=0.3, boot_frac=0.5, tau_sampling="low_mix")
    plain.eval()
    opt.eval()
    torch.manual_seed(1)
    a = plain.compute_loss(batch)
    torch.manual_seed(1)
    b = opt.compute_loss(batch, teacher=plain)
    assert torch.equal(a, b)


@pytest.mark.parametrize("tau,tau_p", [(0.0, 0.3), (0.2, 0.5), (0.35, 0.6)])
def test_pair_from_x1_has_the_path_joint_law(tau, tau_p):
    model = _model()
    s0 = model.sigma_prior
    n = 200_000
    x1 = torch.full((n, 1), 1.5)
    torch.manual_seed(2)
    x_t, x_tp = model.pair_from_x1(x1, torch.full((n,), tau), torch.full((n,), tau_p))
    b_t, b_tp = (1 - tau) * s0, (1 - tau_p) * s0
    assert x_t.mean().item() == pytest.approx(tau * 1.5, abs=4e-3)
    assert x_t.var().item() == pytest.approx(b_t ** 2, rel=2e-2)
    assert x_tp.mean().item() == pytest.approx(tau_p * 1.5, abs=4e-3)
    assert x_tp.var().item() == pytest.approx(b_tp ** 2, rel=2e-2)
    cov = ((x_t - x_t.mean()) * (x_tp - x_tp.mean())).mean().item()
    assert cov == pytest.approx((tau / tau_p) * b_tp ** 2, abs=4e-3)


class _EchoTeacher(torch.nn.Module):
    def forward(self, x, batch, tau):
        return batch.states


def test_bootstrap_with_exact_target_teacher_equals_the_teacherless_loss():
    batch = _batch()
    model = _model(boot_frac=1.0, boot_tau_min=0.0, boot_tau_max=0.0, boot_alpha=1.0)
    model.train()
    torch.manual_seed(3)
    with_teacher = model.compute_loss(batch, teacher=_EchoTeacher())
    torch.manual_seed(3)
    without = model.compute_loss(batch)
    assert torch.equal(with_teacher, without)


def test_bootstrap_target_is_the_teacher_and_carries_no_teacher_gradient():
    batch = _batch()
    model = _model(boot_frac=1.0, boot_tau_min=0.0, boot_tau_max=0.0, boot_alpha=1.0)
    teacher = _model()
    teacher.requires_grad_(False)
    model.train()
    torch.manual_seed(4)
    loss = model.compute_loss(batch, teacher=teacher)
    loss.backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert any(p.grad is not None for p in model.parameters())
    torch.manual_seed(4)
    assert not torch.equal(loss.detach(), model.compute_loss(batch).detach())


def test_tau0_fraction_sends_rows_to_tau_zero():
    batch = _batch(B=64)
    model = _model(tau0_frac=1.0)
    seen = []
    orig = model.forward
    model.forward = lambda x, b, tau: (seen.append(tau.clone()), orig(x, b, tau))[1]
    model.train()
    model.compute_loss(batch)
    assert torch.all(seen[0] == 0)


def test_low_mix_concentrates_tau_below_tau_low_max():
    batch = _batch(B=4000, T=4)
    model = _model(tau_sampling="low_mix", tau_low_frac=0.5, tau_low_max=0.4)
    seen = []
    model.forward = lambda x, b, tau: (seen.append(tau.clone()), torch.zeros_like(x))[1]
    model.train()
    model.compute_loss(batch)
    frac_low = (seen[0] < 0.4).float().mean().item()
    assert frac_low == pytest.approx(0.5 + 0.5 * 0.4, abs=0.03)


def test_tau0_zero_input_makes_tau0_output_independent_of_x():
    model = _model(tau0_zero_input=True)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.2)
    model.eval()
    batch = _batch()
    tau0 = torch.zeros(batch.obs.shape[0])
    with torch.no_grad():
        a = model.forward(torch.randn_like(batch.obs), batch, tau0)
        b = model.forward(torch.randn_like(batch.obs), batch, tau0)
        c = model.forward(torch.randn_like(batch.obs), batch, torch.full_like(tau0, 0.5))
        d = model.forward(torch.randn_like(batch.obs), batch, torch.full_like(tau0, 0.5))
    assert torch.allclose(a, b)
    assert not torch.allclose(c, d)


@pytest.mark.parametrize("bad", [dict(tau_sampling="beta"), dict(tau0_frac=0.6, boot_frac=0.6),
                                 dict(boot_tau_min=0.5, boot_tau_max=0.2), dict(boot_dtau_min=0.0)])
def test_invalid_options_raise(bad):
    with pytest.raises(ValueError):
        _model(**bad)


def test_ema_teacher_updates_by_lerp_and_stays_out_of_the_state_dict():
    model = _model()
    lit = LitModel(model, model_type="monai_predict_state_cfm", use_cosine_scheduler=False,
                   ema_decay=0.75)
    lit._init_teacher()
    teacher = lit.teacher
    before = [p.clone() for p in teacher.parameters()]
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    lit.on_train_batch_end(None, None, 0)
    for p_t, p0, p in zip(teacher.parameters(), before, model.parameters()):
        assert torch.allclose(p_t, 0.75 * p0 + 0.25 * p)
    assert not any(k.startswith("teacher") or "_teacher" in k for k in lit.state_dict())
    assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())


def test_teacher_is_inactive_before_start_epoch_and_without_ema():
    lit = LitModel(_model(), model_type="monai_predict_state_cfm", use_cosine_scheduler=False)
    lit._init_teacher()
    assert lit.teacher is None and lit._active_teacher() is None
    lit = LitModel(_model(), model_type="monai_predict_state_cfm", use_cosine_scheduler=False,
                   ema_decay=0.99, teacher_start_epoch=10)
    lit._init_teacher()
    lit.model.train()
    assert lit.current_epoch < 10 and lit._active_teacher() is None


def test_train_py_wires_options_and_teacher():
    from train import model_factory, psc_teacher_options
    section = {"hidden_channels": [16, 32], "N_outer": 3, "sigma_prior": 0.5, "dropout": 0.0,
               "cond_extra_dim": 0, "num_res_blocks": 1, "norm_num_groups": 8,
               "boot_frac": 0.25, "boot_dtau_min": 0.2, "boot_dtau_max": 0.5,
               "boot_ema_decay": 0.999, "boot_start_epoch": 50, "tau0_zero_input": True}
    cfg = OmegaConf.create({"model": {"model_type": "monai_predict_state_cfm", "state_dim": 4,
                                      "param_dim": 0, "monai_predict_state_cfm": section}})
    m = model_factory(cfg, torch.device("cpu"))
    assert m.boot_frac == 0.25 and m.tau0_zero_input and m.tau_sampling == "uniform"
    assert psc_teacher_options(cfg) == {"ema_decay": 0.999, "teacher_start_epoch": 50}
    del section["boot_ema_decay"]
    cfg = OmegaConf.create({"model": {"model_type": "monai_predict_state_cfm", "state_dim": 4,
                                      "param_dim": 0, "monai_predict_state_cfm": section}})
    assert psc_teacher_options(cfg) == {}
