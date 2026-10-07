"""``freeze_prior_in_solver``: Phi's parameters are detached where Phi feeds the solver
input (x - Phi(x)), so the solver's MSE loss trains the update UNet only and Phi is trained
by the aux prior loss alone; gradients still flow to x through Phi."""
import pytest
import torch

import models.fourdvarnet as F
from tests.test_fdv_zero_prior_input import D, _Batch


def _solver(backbone="unet1d", freeze=True, aux=0.0, mode="subgrad+mask+state"):
    torch.manual_seed(1)
    extra = {"monai_norm_num_groups": 4, "prior_output_init_std": 0.1} if backbone == "monai" else {}
    model = F.FourDVarNetSolver(state_dim=D, hidden_channels=[8, 16], time_emb_dim=8, N_outer=3,
                                update_input=mode, unet_backbone=backbone, dropout=0.0,
                                aux_var_cost_weight=aux, aux_detach_x_final=True,
                                freeze_prior_in_solver=freeze, **extra)
    if backbone == "monai":
        final = model.unet.backbone.out[-1]
        conv = final.conv if hasattr(final, "conv") else final
        torch.nn.init.normal_(conv.weight, std=0.1)
    return model


def _grad_norm(module):
    return sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)


def _grad_norms(model, batch):
    model.zero_grad()
    model.compute_loss(batch).backward()
    return _grad_norm(model.unet), _grad_norm(model.prior_unet)


@pytest.mark.parametrize("backbone", ["unet1d", "monai"])
@pytest.mark.parametrize("mode", ["subgrad+state", "subgrad+mask+state"])
def test_solver_loss_does_not_train_phi(backbone, mode):
    model = _solver(backbone, mode=mode).train()
    g_unet, g_phi = _grad_norms(model, _Batch())
    assert g_unet > 0 and g_phi == 0


@pytest.mark.parametrize("backbone", ["unet1d", "monai"])
def test_without_freeze_solver_loss_trains_phi(backbone):
    model = _solver(backbone, freeze=False).train()
    assert _grad_norms(model, _Batch())[1] > 0


def test_aux_loss_still_trains_phi():
    model = _solver(aux=0.01).train()
    assert _grad_norms(model, _Batch())[1] > 0


def test_forward_identical_and_gradient_reaches_x():
    a, b = _solver(freeze=True).eval(), _solver(freeze=False).eval()
    b.load_state_dict(a.state_dict())
    batch = _Batch()
    torch.testing.assert_close(a(batch), b(batch))
    x = torch.randn(2, 32, D, requires_grad=True)
    out = F._prior_ae_frozen(a.prior_unet, x)
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_rejects_other_modes():
    with pytest.raises(ValueError):
        _solver(mode="resid+mask+state")
