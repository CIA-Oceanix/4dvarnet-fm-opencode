"""``resid+mask+state`` / ``residnorm+mask+state``: ``resid+state`` plus an explicit
observation-mask channel block (and, for the second, the residual rescaled by its
per-sample RMS over observed entries with that RMS fed back as a log channel)."""
import pytest
import torch

import models.fourdvarnet as F
from tests.test_fdv_zero_prior_input import D, _Batch


def _solver(mode: str) -> F.FourDVarNetSolver:
    torch.manual_seed(1)
    return F.FourDVarNetSolver(state_dim=D, hidden_channels=[4, 8], time_emb_dim=8, N_outer=3,
                               update_input=mode, unet_backbone="unet1d").eval()


def _inputs(monkeypatch, mode):
    m = _solver(mode)
    seen = []
    orig = F._build_update_input

    def spy(update_input, x, obs_clean, obs_mask, *a, **k):
        out = orig(update_input, x, obs_clean, obs_mask, *a, **k)
        seen.append((out.detach(), x.detach(), obs_clean.detach(), obs_mask.detach()))
        return out

    monkeypatch.setattr(F, "_build_update_input", spy)
    batch = _Batch()
    out = m(batch)
    assert torch.isfinite(out).all()
    missing = ~torch.isfinite(batch.obs) | ~batch.obs_mask.unsqueeze(-1)
    return seen, missing


def test_resid_mask_state_channels(monkeypatch):
    seen, missing = _inputs(monkeypatch, "resid+mask+state")
    for inp, x, oc, mask in seen:
        mask = mask.expand_as(x).to(x.dtype)
        assert inp.shape[-1] == 3 * D
        torch.testing.assert_close(inp[..., :D], x)
        torch.testing.assert_close(inp[..., D:2 * D], (x - oc) * mask)
        torch.testing.assert_close(inp[..., 2 * D:], mask)
        assert (inp[..., 2 * D:][missing] == 0).all() and (inp[..., 2 * D:][~missing] == 1).all()


def test_residnorm_unit_rms_on_observed_entries(monkeypatch):
    seen, missing = _inputs(monkeypatch, "residnorm+mask+state")
    for inp, x, oc, mask in seen:
        assert inp.shape[-1] == 4 * D
        r = inp[..., D:2 * D]
        obs = ~missing
        for b in range(x.shape[0]):
            rms = r[b][obs[b]].pow(2).mean().sqrt()
            torch.testing.assert_close(rms, torch.tensor(1.0), atol=1e-4, rtol=1e-4)
            raw = ((x - oc)[b][obs[b]]).pow(2).mean().sqrt()
            torch.testing.assert_close(inp[b, ..., 3 * D:], torch.log(raw).expand(x.shape[1], D))
        assert (r[missing] == 0).all()


@pytest.mark.parametrize("mode,mult", [("resid+mask+state", 3), ("residnorm+mask+state", 4),
                                       ("subgrad+mask+state", 4)])
def test_input_width(mode, mult):
    assert F._UPDATE_INPUT_CHANNEL_MULTIPLIER[mode] == mult
    _solver(mode)


def test_subgrad_mask_state_is_subgrad_plus_mask(monkeypatch):
    seen, missing = _inputs(monkeypatch, "subgrad+mask+state")
    for inp, x, oc, mask in seen:
        mask = mask.expand_as(x).to(x.dtype)
        assert inp.shape[-1] == 4 * D
        torch.testing.assert_close(inp[..., :D], (oc - x) * mask)
        torch.testing.assert_close(inp[..., 2 * D:3 * D], x)
        torch.testing.assert_close(inp[..., 3 * D:], mask)
        assert (inp[..., 3 * D:][missing] == 0).all() and (inp[..., 3 * D:][~missing] == 1).all()


def test_subgrad_mask_state_prior_channel_matches_subgrad():
    a, b = _solver("subgrad+state"), _solver("subgrad+mask+state")
    assert a.prior_unet is not None and b.prior_unet is not None
    b.prior_unet.load_state_dict(a.prior_unet.state_dict())
    x = torch.randn(2, 32, D)
    oc = torch.randn(2, 32, D)
    m = (torch.rand(2, 32, D) > 0.5).float()
    ia = F._build_update_input("subgrad+state", x, oc, m, None, prior_unet=a.prior_unet)
    ib = F._build_update_input("subgrad+mask+state", x, oc, m, None, prior_unet=b.prior_unet)
    torch.testing.assert_close(ib[..., :3 * D], ia)
