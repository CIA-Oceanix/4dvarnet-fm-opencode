"""``update_head="gain"``: the update UNet outputs (a, c) and each iteration applies
x <- clamp(x - m * sigmoid(a + logit(gain_init)) * (x - y) - c / N)."""
import math

import pytest
import torch

import models.fourdvarnet as F
from tests.test_fdv_zero_prior_input import D, _Batch


def _solver(backbone="unet1d", n_outer=3, **kw) -> F.FourDVarNetSolver:
    torch.manual_seed(1)
    extra = {"monai_norm_num_groups": 4} if backbone == "monai" else {}
    return F.FourDVarNetSolver(state_dim=D, hidden_channels=[8, 16], time_emb_dim=8, N_outer=n_outer,
                               update_input="resid+mask+state", unet_backbone=backbone,
                               update_head="gain", **extra, **kw).eval()


def test_gain_step_formula():
    m = _solver(gain_init=0.2)
    B, T, N = 2, 5, 4
    x, y = torch.randn(B, T, D), torch.randn(B, T, D)
    mask = (torch.rand(B, T, D) > 0.5).float()
    gmod = torch.randn(B, T, 2 * D)
    step = m._gain_step(gmod, x, y, mask, N)
    gain = torch.sigmoid(gmod[..., :D] + math.log(0.2 / 0.8))
    torch.testing.assert_close(step, mask * gain * (x - y) + gmod[..., D:] / N)
    assert (step[mask == 0] == (gmod[..., D:] / N)[mask == 0]).all()


def test_output_width_and_finite_unroll():
    m = _solver()
    batch = _Batch()
    out = m(batch)
    assert out.shape[-1] == D and torch.isfinite(out).all()
    probe = torch.zeros(1, 3 * D, 6)
    assert m.unet(probe, tau=torch.zeros(1)).shape[1] == 2 * D


def test_monai_init_is_pure_nudging_at_gain_init():
    m = _solver("monai", n_outer=1, gain_init=0.3)
    batch = _Batch()
    x1 = m(batch).detach()
    observed = torch.isfinite(batch.obs) & batch.obs_mask.unsqueeze(-1)
    y = torch.nan_to_num(batch.obs)
    torch.testing.assert_close(x1[observed], 0.3 * y[observed])
    assert (x1[~observed] == 0).all()


def test_direct_head_unchanged_by_default():
    torch.manual_seed(1)
    a = F.FourDVarNetSolver(state_dim=D, hidden_channels=[8, 16], time_emb_dim=8, N_outer=3,
                            update_input="resid+mask+state", unet_backbone="unet1d")
    assert a.update_head == "direct"


@pytest.mark.parametrize("kw", [{"update_head": "kalman"}, {"gain_init": 0.0}, {"gain_init": 1.0}])
def test_invalid_args(kw):
    with pytest.raises(ValueError):
        F.FourDVarNetSolver(state_dim=D, hidden_channels=[8, 16], time_emb_dim=8, N_outer=3,
                            update_input="resid+mask+state", unet_backbone="unet1d", **kw)
