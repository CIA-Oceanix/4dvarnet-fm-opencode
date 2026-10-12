"""tau_schedule="exp" (tau_k = 1 - exp(-k / tau_scale) on the absolute iteration index, step
1/N_outer at any number of iterations) and the training burn-in (M iterations without
gradient before the N_outer trained ones, which then start at iteration M)."""
import math

import pytest
import torch

import models.fourdvarnet as F
from tests.test_fdv_zero_prior_input import D, _Batch


def _solver(**kw) -> F.FourDVarNetSolver:
    torch.manual_seed(1)
    return F.FourDVarNetSolver(state_dim=D, hidden_channels=[8, 16], time_emb_dim=8, N_outer=3,
                               update_input="resid+mask+state", unet_backbone="unet1d", **kw)


def _record(monkeypatch, const=None):
    calls = []
    orig = F._solver_iteration

    def spy(unet, update_input, x, obs_clean, obs_mask, tau_k, *a, **k):
        calls.append((float(tau_k[0]), torch.is_grad_enabled()))
        out = orig(unet, update_input, x, obs_clean, obs_mask, tau_k, *a, **k)
        return torch.ones_like(out) if const else out

    monkeypatch.setattr(F, "_solver_iteration", spy)
    return calls


def test_default_linear_schedule_unchanged(monkeypatch):
    calls = _record(monkeypatch)
    _solver().eval()(_Batch())
    assert [t for t, _ in calls] == pytest.approx([0.0, 0.5, 1.0])


def test_exp_schedule_values(monkeypatch):
    calls = _record(monkeypatch)
    _solver(tau_schedule="exp", tau_scale=2.0).eval().sample(_Batch(), N_outer=5)
    assert [t for t, _ in calls] == pytest.approx([1 - math.exp(-k / 2.0) for k in range(5)])


def test_exp_schedule_fixed_step(monkeypatch):
    _record(monkeypatch, const=True)
    out = _solver(tau_schedule="exp").eval().sample(_Batch(), N_outer=4)
    torch.testing.assert_close(out, torch.full_like(out, -4 / 3))
    out_lin = _solver().eval().sample(_Batch(), N_outer=4)
    torch.testing.assert_close(out_lin, torch.full_like(out_lin, -1.0))


def test_burnin_runs_without_gradient_then_continues_the_schedule(monkeypatch):
    calls = _record(monkeypatch)
    model = _solver(tau_schedule="exp", tau_scale=2.0, burnin_prob=1.0, burnin_max=1).train()
    loss = model.compute_loss(_Batch())
    assert len(calls) == 1 + 3
    assert calls[0][1] is False and all(g for _, g in calls[1:])
    assert [t for t, _ in calls] == pytest.approx([1 - math.exp(-k / 2.0) for k in range(4)])
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.unet.parameters())


def test_no_burnin_in_eval_or_when_prob_zero(monkeypatch):
    calls = _record(monkeypatch)
    _solver(tau_schedule="exp", burnin_prob=1.0, burnin_max=5).eval()(_Batch())
    assert len(calls) == 3
    calls.clear()
    _solver(tau_schedule="exp", burnin_prob=0.0, burnin_max=5).train().compute_loss(_Batch())
    assert len(calls) == 3


@pytest.mark.parametrize("kw", [{"tau_schedule": "cosine"}, {"burnin_max": 3, "burnin_prob": 0.5},
                                {"tau_schedule": "exp", "burnin_prob": 1.5, "burnin_max": 3},
                                {"tau_schedule": "exp", "tau_scale": 0.0}])
def test_invalid(kw):
    with pytest.raises(ValueError):
        _solver(**kw)


def test_clamp_schedule_matches_linear_then_saturates(monkeypatch):
    calls = _record(monkeypatch)
    _solver(tau_schedule="clamp").eval().sample(_Batch(), N_outer=6)
    assert [t for t, _ in calls] == pytest.approx([0.0, 0.5, 1.0, 1.0, 1.0, 1.0])


def test_clamp_schedule_identical_to_linear_at_n_outer():
    a, b = _solver().eval(), _solver(tau_schedule="clamp").eval()
    batch = _Batch()
    torch.testing.assert_close(a(batch), b(batch))


def test_load_init_weights_both_formats(tmp_path):
    from training.resume import load_init_weights
    src, dst = _solver(), _solver(tau_schedule="clamp")
    for p in src.parameters():
        torch.nn.init.normal_(p)
    lightning = {"state_dict": {f"model.{k}": v for k, v in src.state_dict().items()}, "epoch": 3}
    for blob in (lightning, src.state_dict()):
        path = tmp_path / "w.ckpt"
        torch.save(blob, path)
        load_init_weights(dst, str(path))
        for k, v in src.state_dict().items():
            torch.testing.assert_close(dst.state_dict()[k], v)


def test_burnin_warmup_and_ramp_schedule():
    m = _solver(tau_schedule="clamp", burnin_prob=0.5, burnin_max=30, burnin_start_epoch=200,
                burnin_ramp_epochs=200)
    assert m.burnin_max_current == 0
    expected = {0: 0, 199: 0, 200: 1, 299: 15, 399: 30, 400: 30, 1199: 30}
    for epoch, value in expected.items():
        m.set_epoch(epoch)
        assert m.burnin_max_current == value, epoch


def test_burnin_without_warmup_is_active_from_the_start():
    m = _solver(tau_schedule="clamp", burnin_prob=0.5, burnin_max=30)
    assert m.burnin_max_current == 30
    m.set_epoch(0)
    assert m.burnin_max_current == 30


def test_no_burnin_during_warmup(monkeypatch):
    calls = _record(monkeypatch)
    model = _solver(tau_schedule="clamp", burnin_prob=1.0, burnin_max=5, burnin_start_epoch=10).train()
    model.set_epoch(3)
    model.compute_loss(_Batch())
    assert len(calls) == 3
    calls.clear()
    model.set_epoch(10)
    model.compute_loss(_Batch())
    assert len(calls) > 3


def test_lightning_hook_sets_the_epoch():
    from training.lightning_module import LitModel
    model = _solver(tau_schedule="clamp", burnin_prob=0.5, burnin_max=30, burnin_start_epoch=5)
    model.burnin_max_current = 99
    LitModel(model, model_type="fourdvarnet").on_train_epoch_start()
    assert model.burnin_max_current == 0


def test_train_n_blocks_unrolls_beyond_horizon_in_training_only(monkeypatch):
    calls = _record(monkeypatch)
    model = _solver(tau_schedule="clamp", train_n_blocks=2).train()
    blocks = model._unrolled_blocks(_Batch())
    assert len(calls) == 6 and len(blocks) == 2
    assert [t for t, _ in calls] == pytest.approx([0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    calls.clear()
    model.eval()
    assert len(model._unrolled_blocks(_Batch())) == 1 and len(calls) == 3


def test_train_n_blocks_loss_is_mean_over_block_ends():
    model = _solver(tau_schedule="clamp", train_n_blocks=2, dropout=0.0).train()
    batch = _Batch()
    torch.manual_seed(5)
    blocks = model._unrolled_blocks(batch)
    expected = sum(torch.nn.functional.mse_loss(b, batch.states) for b in blocks) / 2
    torch.manual_seed(5)
    torch.testing.assert_close(model.compute_loss(batch), expected)


def test_train_n_blocks_detaches_between_blocks():
    model = _solver(tau_schedule="clamp", train_n_blocks=2, dropout=0.0).train()
    first, second = model._unrolled_blocks(_Batch())
    seen = []

    def walk(fn):
        stack, visited = [fn], set()
        while stack:
            f = stack.pop()
            if f is None or f in visited:
                continue
            visited.add(f)
            seen.append(f)
            stack.extend(n for n, _ in f.next_functions)

    walk(second.grad_fn)
    assert first.grad_fn not in seen


@pytest.mark.parametrize("kw", [{"train_n_blocks": 0}, {"train_n_blocks": 2},
                                {"train_n_blocks": 2, "tau_schedule": "clamp", "tbptt_n_blocks": 3,
                                 "tbptt_block_size": 1}])
def test_train_n_blocks_invalid(kw):
    with pytest.raises(ValueError):
        _solver(**kw)
