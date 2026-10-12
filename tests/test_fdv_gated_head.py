"""``update_head="gated"``: the update UNet outputs (a, c) and each iteration applies
x <- clamp(x - 2 * sigmoid(a) * c / N), a learned per-element gate on the free update."""
import torch

import models.fourdvarnet as F
from tests.test_fdv_zero_prior_input import D, _Batch


def _solver(head: str, backbone: str = "unet1d", n_outer: int = 3) -> F.FourDVarNetSolver:
    torch.manual_seed(1)
    extra = {"monai_norm_num_groups": 4} if backbone == "monai" else {}
    return F.FourDVarNetSolver(state_dim=D, hidden_channels=[8, 16], time_emb_dim=8, N_outer=n_outer,
                               update_input="obs+state", unet_backbone=backbone,
                               update_head=head, **extra).eval()


def test_gated_step_formula():
    gmod = torch.randn(2, 5, 2 * D)
    step = F.FourDVarNetSolver._gated_step(gmod, 4)
    torch.testing.assert_close(step, 2 * torch.sigmoid(gmod[..., :D]) * gmod[..., D:] / 4)


def test_gate_is_one_when_a_is_zero():
    c = torch.randn(2, 5, D)
    step = F.FourDVarNetSolver._gated_step(torch.cat([torch.zeros_like(c), c], dim=-1), 10)
    torch.testing.assert_close(step, c / 10)


def test_output_width_and_finite_unroll():
    m = _solver("gated")
    out = m(_Batch())
    assert out.shape[-1] == D and torch.isfinite(out).all()
    assert m.unet(torch.zeros(1, 2 * D, 6), tau=torch.zeros(1)).shape[1] == 2 * D


def test_monai_gated_starts_at_rest_like_direct():
    batch = _Batch()
    out = _solver("gated", "monai", n_outer=2)(batch).detach()
    ref = _solver("direct", "monai", n_outer=2)(batch).detach()
    torch.testing.assert_close(out, torch.zeros_like(out))
    torch.testing.assert_close(ref, torch.zeros_like(ref))
