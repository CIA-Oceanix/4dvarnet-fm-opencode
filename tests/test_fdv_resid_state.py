"""``resid+state``: FDV1 ("obs+state") with y replaced by the masked residual
(x - y) * mask -- same architecture and initialisation, so a run of it isolates
the observation representation against FDV1."""
import torch

import models.fourdvarnet as F
from tests.test_fdv_zero_prior_input import D, _Batch


def _solver(mode: str) -> F.FourDVarNetSolver:
    torch.manual_seed(1)
    return F.FourDVarNetSolver(state_dim=D, hidden_channels=[4, 8], time_emb_dim=8, N_outer=3,
                               update_input=mode, unet_backbone="unet1d").eval()


def test_same_parameters_and_initialisation_as_obs_state():
    a, b = _solver("obs+state"), _solver("resid+state")
    assert a.prior_unet is None and b.prior_unet is None
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_input_is_the_masked_residual(monkeypatch):
    m = _solver("resid+state")
    seen = []
    orig = F._build_update_input

    def spy(update_input, x, obs_clean, obs_mask, *a, **k):
        out = orig(update_input, x, obs_clean, obs_mask, *a, **k)
        seen.append((out.detach(), x.detach(), obs_clean.detach(), obs_mask.detach()))
        return out

    monkeypatch.setattr(F, "_build_update_input", spy)
    batch = _Batch()
    m(batch)
    missing = ~torch.isfinite(batch.obs) | ~batch.obs_mask.unsqueeze(-1)
    for inp, x, oc, mask in seen:
        assert inp.shape[-1] == 2 * D
        torch.testing.assert_close(inp[..., :D], x)
        torch.testing.assert_close(inp[..., D:], (x - oc) * mask)
        assert (inp[..., D:][missing] == 0).all()


def test_differs_from_obs_state_output():
    a, b = _solver("obs+state"), _solver("resid+state")
    batch = _Batch()
    torch.manual_seed(3)
    xa = a(batch).detach()
    torch.manual_seed(3)
    xb = b(batch).detach()
    assert not torch.allclose(xa, xb)
