"""``zero_prior_input``: the subgrad+state ablation that feeds zeros instead of
x - Phi(x) to the update UNet (same channel layout), leaving Phi trained by the
aux prior loss only."""
import pytest
import torch

import models.fourdvarnet as F

B, T, D = 2, 32, 6


class _Batch:
    def __init__(self, seed: int = 0):
        torch.manual_seed(seed)
        self.states = torch.randn(B, T, D)
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, ::4] = True
        obs = self.states + 0.1 * torch.randn(B, T, D)
        obs = torch.where(mask.unsqueeze(-1), obs, torch.full_like(obs, float("nan")))
        obs[:, 4, D - 2:] = float("nan")
        self.obs, self.obs_mask, self.batch_size = obs, mask, B


def _solver(zero_prior_input: bool, aux: float = 0.0) -> F.FourDVarNetSolver:
    torch.manual_seed(1)
    return F.FourDVarNetSolver(state_dim=D, hidden_channels=[4, 8], time_emb_dim=8, N_outer=3,
                               update_input="subgrad+state", unet_backbone="unet1d",
                               aux_var_cost_weight=aux, zero_prior_input=zero_prior_input)


def _perturb_prior(m: F.FourDVarNetSolver) -> None:
    with torch.no_grad():
        for p in m.prior_unet.parameters():
            p.add_(0.5 * torch.randn_like(p))


def test_prior_channel_is_zero_and_output_ignores_phi(monkeypatch):
    m = _solver(True).eval()
    seen = []
    orig = F._build_update_input

    def spy(*a, **k):
        out = orig(*a, **k)
        seen.append(out.detach())
        return out

    monkeypatch.setattr(F, "_build_update_input", spy)
    batch = _Batch()
    out = m(batch).detach()
    assert seen and all(inp.shape[-1] == 3 * D and torch.equal(inp[..., D:2 * D], torch.zeros_like(inp[..., D:2 * D]))
                        for inp in seen)
    _perturb_prior(m)
    torch.testing.assert_close(m(batch).detach(), out)


def test_without_flag_phi_changes_the_output():
    m = _solver(False).eval()
    batch = _Batch()
    out = m(batch).detach()
    _perturb_prior(m)
    assert not torch.allclose(m(batch).detach(), out)


def test_aux_loss_still_trains_phi():
    m = _solver(True, aux=0.01).train()
    m.compute_loss(_Batch()).backward()
    grads = [p.grad for p in m.prior_unet.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize("mode", ["obs+state", "grad+state", "gradsplit+state"])
def test_flag_rejected_outside_subgrad_state(mode):
    with pytest.raises(ValueError, match="zero_prior_input"):
        F.FourDVarNetSolver(state_dim=D, hidden_channels=[4, 8], time_emb_dim=8, N_outer=3,
                            update_input=mode, unet_backbone="unet1d", zero_prior_input=True)



def _grads(aux: float, detach: bool) -> tuple:
    torch.manual_seed(1)
    m = F.FourDVarNetSolver(state_dim=D, hidden_channels=[4, 8], time_emb_dim=8, N_outer=3, dropout=0.0,
                            update_input="subgrad+state", unet_backbone="unet1d",
                            aux_var_cost_weight=aux, aux_detach_x_final=detach).train()
    m.compute_loss(_Batch()).backward()
    solver = torch.cat([p.grad.flatten() for p in m.unet.parameters() if p.grad is not None])
    prior = torch.cat([p.grad.flatten() for p in m.prior_unet.parameters() if p.grad is not None])
    return solver, prior


def test_aux_detach_leaves_the_solver_gradient_of_the_supervised_loss():
    """With aux_detach_x_final the aux prior cost trains Phi only: the solver UNet's
    gradient equals that of the same model without the aux loss."""
    base_solver, _ = _grads(aux=0.0, detach=False)
    det_solver, det_prior = _grads(aux=0.01, detach=True)
    raw_solver, _ = _grads(aux=0.01, detach=False)
    torch.testing.assert_close(det_solver, base_solver)
    assert not torch.allclose(raw_solver, base_solver)
    assert det_prior.abs().sum() > 0
