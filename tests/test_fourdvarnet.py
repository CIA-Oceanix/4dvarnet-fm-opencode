from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from models.fourdvarnet import (
    FourDVarNetPredictStateCFM,
    FourDVarNetSolver,
    _build_update_input,
    _normalize_channels,
    _soft_clip,
)

_GRAD_MODES = ("grad-only", "grad+state", "subgrad+state")
_ALL_UPDATE_INPUT_MODES = ("obs+state", "obs-only", "grad-only", "grad+state", "subgrad+state")


def _bypass_checkpoint(fn, *args, **kwargs):
    """Drop-in stand-in for ``torch.utils.checkpoint.checkpoint`` that just
    calls the wrapped function directly -- no activation discarding, no
    recomputation. Patched into ``models.fourdvarnet.checkpoint`` to produce
    a "no checkpointing" reference run for TestGradientCheckpointing, so the
    same forward()/backward() code path is exercised either way and only the
    checkpoint mechanics differ."""
    return fn(*args)


def _forward_backward_grads(model, forward_fn, seed):
    """Runs ``forward_fn()`` (a closure calling ``model``'s forward with
    whatever args it needs) under a fixed seed -- so dropout draws land in
    the same order whether or not checkpointing recomputes any iteration --
    then backprops a simple sum-of-squares loss and returns the output and
    every parameter's gradient, for before/after comparison."""
    torch.manual_seed(seed)
    model.zero_grad()
    out = forward_fn()
    out.pow(2).sum().backward()
    grads = {name: (p.grad.clone() if p.grad is not None else None)
             for name, p in model.named_parameters()}
    return out.detach().clone(), grads


class _MockBatch:
    def __init__(self, B=2, T=50, D=3, obs_every=10, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.states = torch.randn(B, T, D)
        obs = torch.randn(B, T, D)
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, ::obs_every] = True
        self.obs = torch.where(mask.unsqueeze(-1), obs, torch.full_like(obs, float("nan")))
        self.obs_mask = mask
        self.batch_size = B


def _make_model(**kwargs):
    defaults = dict(state_dim=3, hidden_channels=[4, 8], N_outer=3)
    defaults.update(kwargs)
    return FourDVarNetSolver(**defaults)


class TestFourDVarNetSolver:
    def test_forward_shape(self):
        model = _make_model()
        batch = _MockBatch(B=2, T=50, D=3)
        out = model(batch)
        assert out.shape == (2, 50, 3)

    def test_forward_finite(self):
        model = _make_model()
        batch = _MockBatch(B=2, T=50, D=3)
        out = model(batch)
        assert torch.isfinite(out).all()

    def test_zero_iterations_returns_init_state(self):
        model = _make_model(N_outer=0)
        batch = _MockBatch(B=2, T=20, D=3)
        out = model(batch)
        assert torch.equal(out, torch.zeros_like(out))

    def test_weight_tied_param_count_independent_of_N_outer(self):
        model_a = _make_model(N_outer=3)
        model_b = _make_model(N_outer=10)
        n_a = sum(p.numel() for p in model_a.parameters())
        n_b = sum(p.numel() for p in model_b.parameters())
        assert n_a == n_b, "param count must not scale with N_outer (weight-tied)"

    def test_output_responds_to_obs(self):
        model = _make_model()
        model.eval()
        batch_a = _MockBatch(B=2, T=50, D=3, seed=0)
        batch_b = _MockBatch(B=2, T=50, D=3, seed=0)
        batch_b.states = batch_a.states.clone()
        batch_b.obs = torch.where(
            batch_b.obs_mask.unsqueeze(-1),
            torch.randn_like(batch_b.obs) + 100.0,
            batch_b.obs,
        )
        with torch.no_grad():
            out_a = model(batch_a)
            out_b = model(batch_b)
        assert not torch.allclose(out_a, out_b), \
            "output must depend on obs (unlike SDA1's unconditional prior)"

    def test_obs_state_differs_from_obs_only(self):
        torch.manual_seed(0)
        model_state = _make_model(update_input="obs+state")
        torch.manual_seed(0)
        model_obs_only = _make_model(update_input="obs-only")
        batch = _MockBatch(B=2, T=50, D=3)
        out_state = model_state(batch)
        out_obs_only = model_obs_only(batch)
        assert not torch.allclose(out_state, out_obs_only)

    def test_grad_modes_shape_finite_and_have_prior_unet(self):
        for mode in _GRAD_MODES:
            model = _make_model(update_input=mode)
            assert model.prior_unet is not None, f"{mode} needs a prior_unet"
            batch = _MockBatch(B=2, T=50, D=3)
            out = model(batch)
            assert out.shape == (2, 50, 3)
            assert torch.isfinite(out).all()

    def test_obs_state_modes_have_no_prior_unet(self):
        for mode in ("obs+state", "obs-only"):
            assert _make_model(update_input=mode).prior_unet is None

    def test_forward_output_bounded_by_clip_range(self):
        """Regression guard for a real training-time failure: a first
        "grad+state" FDV1 (deterministic) training run never trained at all
        (val_loss stuck flat from epoch 0) -- traced to FourDVarNetSolver
        having no clip_range clamp at all (unlike FourDVarNetPredictStateCFM,
        which already had one), leaving its N_outer-step unroll completely
        unbounded for the gradient-conditioned modes. Force the pathological
        regime with untrained (large-scale) weights and check forward()
        never exceeds clip_range."""
        torch.manual_seed(0)
        model = _make_model(update_input="grad+state", N_outer=3, clip_range=50.0)
        with torch.no_grad():
            for p in model.unet.parameters():
                p.mul_(20.0)
            for p in model.prior_unet.parameters():
                p.mul_(20.0)
        batch = _MockBatch(B=4, T=20, D=3, seed=1)
        out = model(batch)
        assert torch.isfinite(out).all()
        assert out.abs().max().item() <= 50.0 + 1e-4

    def test_clip_range_inactive_in_distribution(self):
        """The state-branch hard clamp must not perturb ordinary,
        in-distribution behavior -- exactly, since torch.clamp is the
        identity wherever it doesn't bind. grad_clip_range's tanh soft-clip
        is a *smooth* rescaling, not a hard cutoff -- it's never exactly the
        identity (tanh(u) ~= u only approximately for small u), so
        grad_clip_range is pinned equal and huge (1e9) on both sides here to
        isolate the state-branch clamp's own on/off invariance; the soft-
        clip's own near-identity behavior for a generous clip_range is
        checked separately (test_soft_clip_near_identity_for_large_clip_range)."""
        model_clipped = _make_model(update_input="grad+state", dropout=0.0,
                                     clip_range=50.0, grad_clip_range=1e9)
        model_unclipped = _make_model(update_input="grad+state", dropout=0.0,
                                       clip_range=1e6, grad_clip_range=1e9)
        model_unclipped.unet.load_state_dict(model_clipped.unet.state_dict())
        model_unclipped.prior_unet.load_state_dict(model_clipped.prior_unet.state_dict())
        model_clipped.eval()
        model_unclipped.eval()
        batch = _MockBatch(B=2, T=20, D=3)
        with torch.no_grad():
            out_clipped = model_clipped(batch)
            out_unclipped = model_unclipped(batch)
        assert torch.allclose(out_clipped, out_unclipped)

    def test_grad_modes_differ_from_each_other_and_from_obs_state(self):
        outs = {}
        for mode in ("obs+state",) + _GRAD_MODES:
            torch.manual_seed(0)
            model = _make_model(update_input=mode)
            batch = _MockBatch(B=2, T=20, D=3, seed=1)
            outs[mode] = model(batch)
        modes = list(outs.keys())
        for i in range(len(modes)):
            for j in range(i + 1, len(modes)):
                assert not torch.allclose(outs[modes[i]], outs[modes[j]]), \
                    f"{modes[i]} and {modes[j]} should not produce identical output"

    def test_gradients_flow_through_unroll_grad_modes(self):
        """The main correctness risk for grad-only/grad+state: torch.autograd.grad
        with create_graph=True nested inside the unroll must not block BPTT
        into either unet or prior_unet's parameters."""
        for mode in ("grad-only", "grad+state"):
            model = _make_model(update_input=mode, N_outer=3)
            batch = _MockBatch(B=2, T=20, D=3)
            loss = model.compute_loss(batch)
            loss.backward()
            for name, p in model.named_parameters():
                assert p.grad is not None, f"[{mode}] no gradient reached {name}"
                assert torch.isfinite(p.grad).all(), f"[{mode}] non-finite gradient at {name}"

    def test_gradients_flow_through_unroll_subgrad_state(self):
        """subgrad+state never calls torch.autograd.grad -- confirm ordinary
        autograd still reaches prior_unet's parameters via the plain
        subtraction path (x - prior_ae(x))."""
        model = _make_model(update_input="subgrad+state", N_outer=3)
        batch = _MockBatch(B=2, T=20, D=3)
        loss = model.compute_loss(batch)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"no gradient reached {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite gradient at {name}"

    def test_build_update_input_channel_counts(self):
        B, T, D = 2, 10, 3
        x = torch.randn(B, T, D).requires_grad_(True)
        obs_clean = torch.randn(B, T, D)
        obs_mask = torch.ones(B, T, 1)
        tau = torch.rand(B)
        prior_unet_model = _make_model(update_input="grad-only").prior_unet
        expected = {
            "obs-only": D, "obs+state": 2 * D,
            "grad-only": D, "grad+state": 2 * D, "subgrad+state": 3 * D,
        }
        for mode, expected_channels in expected.items():
            out = _build_update_input(mode, x, obs_clean, obs_mask, tau,
                                       prior_unet=prior_unet_model, R_var=0.5, obs_weight=1.0)
            assert out.shape == (B, T, expected_channels), f"{mode}: {out.shape}"

    def test_build_update_input_grad_channels_are_normalized(self):
        """Regression guard for a real training-time divergence: a first
        400-epoch grad+state run was stable for ~115 epochs then diverged in
        a single step and stayed permanently saturated -- traced to an
        unnormalized gradient/residual channel whose raw magnitude can grow
        arbitrarily large. Normalization is a single global (whole-tensor)
        RMS scalar (matching ocean4dvarnet's ConvLstmGradModel exactly:
        sqrt(mean(t**2)) ~= 1), not a per-sample norm. Force a large-
        magnitude adversarial state/obs and confirm this holds (approximately
        -- not exactly, since the post-normalization bound is now a smooth
        tanh soft-clip, not a hard clamp: it's never the exact identity, only
        very close to it when the RMS-1 tensor sits well inside
        grad_clip_range=50, hence the atol looser than a hard clamp would
        need) regardless of scale. "subgrad+state" is NOT normalized (see
        the test below) -- only the real-autograd grad-only/grad+state
        channel is."""
        B, T, D = 2, 10, 3
        prior_unet_model = _make_model(update_input="grad-only").prior_unet
        tau = torch.rand(B)
        obs_mask = torch.ones(B, T, 1)
        for scale in (1.0, 1000.0):
            x = (torch.randn(B, T, D) * scale).requires_grad_(True)
            obs_clean = torch.randn(B, T, D) * scale
            for mode, grad_blocks in (("grad-only", [(0, D)]), ("grad+state", [(0, D)])):
                out = _build_update_input(mode, x, obs_clean, obs_mask, tau,
                                           prior_unet=prior_unet_model, R_var=0.5, obs_weight=1.0,
                                           clip_range=50.0)
                for lo, hi in grad_blocks:
                    rms = (out[..., lo:hi] ** 2).mean().sqrt()
                    assert torch.isfinite(rms).all()
                    assert torch.allclose(rms, torch.ones_like(rms), atol=2e-3), \
                        f"{mode} @ scale={scale} block[{lo}:{hi}]: rms={rms}"

    def test_build_update_input_subgrad_channels_are_not_normalized(self):
        """subgrad+state's g_obs/g_prior are raw differences of two already-
        comparable-scale quantities (obs/state both live in the same
        normalized state space) -- deliberately NOT run through
        _normalize_channels, unlike grad-only/grad+state's real autograd
        gradient above (matches ronan_devs' own GradSolver_withStep, which
        never normalizes gobs/gprior either). Root-caused a severe fast-Y
        reconstruction collapse in every subgrad+state MonaiUNet1D run
        trained under the old normalized version: g_obs is masked to exactly
        zero outside observation times, so a *global* whole-tensor RMS norm
        over that mostly-zero tensor was diluted by the observation density
        and then inflated the sparse nonzero entries by that same factor.
        Confirmed here via the contrapositive of the test above: raw values
        scale linearly with the input, they don't stay pinned to RMS=1."""
        B, T, D = 2, 10, 3
        prior_unet_model = _make_model(update_input="subgrad+state").prior_unet
        tau = torch.rand(B)
        obs_mask = torch.ones(B, T, 1)
        x_small = torch.randn(B, T, D).requires_grad_(True)
        obs_small = torch.randn(B, T, D)
        out_small = _build_update_input("subgrad+state", x_small, obs_small, obs_mask, tau,
                                         prior_unet=prior_unet_model)
        x_large, obs_large = x_small.detach() * 1000.0, obs_small * 1000.0
        x_large.requires_grad_(True)
        out_large = _build_update_input("subgrad+state", x_large, obs_large, obs_mask, tau,
                                         prior_unet=prior_unet_model)
        g_obs_small, g_obs_large = out_small[..., :D], out_large[..., :D]
        assert torch.allclose(g_obs_large, g_obs_small * 1000.0, atol=1e-2), \
            "g_obs must scale linearly with the raw input, not stay pinned to RMS=1"
        assert torch.allclose(g_obs_small, (obs_small - x_small.detach()) * obs_mask, atol=1e-5)

    def test_normalize_channels_cache_reuses_first_norm(self):
        """Matches ocean4dvarnet's ConvLstmGradModel exactly: the norm is
        computed once (first call for a given key) and reused unchanged for
        later calls with the same cache/key, even if the input's own
        magnitude has since changed -- NOT recomputed fresh every time. The
        norm itself is a single global scalar (sqrt(mean(t**2)) over the
        whole tensor), not a per-sample vector. RMS-after-normalization is
        only approximately 1 (not exactly, per the looser atol below): the
        post-normalization bound is a smooth tanh soft-clip now, not a hard
        clamp, so it's never the exact identity even when comfortably inside
        clip_range."""
        cache = {}
        t1 = torch.randn(2, 10, 3) * 5.0
        out1 = _normalize_channels(t1, cache=cache, key="grad")
        assert cache["grad"].dim() == 0, "norm must be a global scalar, not per-sample"
        assert torch.allclose((out1 ** 2).mean().sqrt(), torch.tensor(1.0), atol=2e-3)

        t2 = torch.randn(2, 10, 3) * 5.0  # a different tensor, same cache/key
        out2 = _normalize_channels(t2, cache=cache, key="grad")
        # out2 must use the CACHED (t1's) norm, not recompute its own -- then
        # go through the same soft-clip _normalize_channels itself applies.
        expected = 50.0 * torch.tanh((t2 / cache["grad"]) / 50.0)
        assert torch.allclose(out2, expected, atol=1e-4)
        assert not torch.allclose((out2 ** 2).mean().sqrt(), torch.tensor(1.0), atol=1e-2)

    def test_grad_norm_cached_across_solve_iterations(self):
        """End-to-end: within one forward() call, the first iteration's grad
        norm is reused for later iterations (not recomputed), matching
        ocean4dvarnet's per-solve (not per-iteration) caching granularity."""
        model = _make_model(update_input="grad+state", N_outer=3, dropout=0.0)
        model.eval()
        batch = _MockBatch(B=2, T=20, D=3)
        with torch.no_grad():
            model(batch)  # just confirm no crash; cache is internal to forward()

        # Directly verify the caching contract via _build_update_input with a
        # shared cache dict, mirroring what forward() does internally.
        cache = {}
        x1 = torch.randn(2, 20, 3).requires_grad_(True)
        obs_clean = torch.randn(2, 20, 3)
        obs_mask = torch.ones(2, 20, 1)
        tau = torch.rand(2)
        _build_update_input("grad+state", x1, obs_clean, obs_mask, tau,
                             prior_unet=model.prior_unet, R_var=0.5, obs_weight=1.0,
                             grad_norm_cache=cache)
        assert "grad" in cache
        cached_norm = cache["grad"].clone()
        x2 = torch.randn(2, 20, 3).requires_grad_(True) * 100.0
        _build_update_input("grad+state", x2, obs_clean, obs_mask, tau,
                             prior_unet=model.prior_unet, R_var=0.5, obs_weight=1.0,
                             grad_norm_cache=cache)
        assert torch.allclose(cache["grad"], cached_norm), "cache must not be overwritten by a later call"

    def test_prior_weight_trainable_for_grad_modes(self):
        for mode in ("grad-only", "grad+state"):
            model = _make_model(update_input=mode)
            assert isinstance(model._prior_weight_raw, torch.nn.Parameter)
            assert model._prior_weight_raw in list(model.parameters())

    def test_prior_weight_fixed_for_other_modes(self):
        for mode in ("obs+state", "obs-only", "subgrad+state"):
            model = _make_model(update_input=mode, prior_weight=2.5)
            assert model._prior_weight_raw is None
            assert model.prior_weight == 2.5

    def test_prior_weight_matches_init_value_at_construction(self):
        for prior_weight in (1.0, 0.1, 5.0):
            model = _make_model(update_input="grad+state", prior_weight=prior_weight)
            assert abs(float(model.prior_weight) - prior_weight) < 1e-5

    def test_prior_weight_never_negative(self):
        model = _make_model(update_input="grad+state", prior_weight=1.0)
        with torch.no_grad():
            for extreme in (-1000.0, 0.0, 1000.0):
                model._prior_weight_raw.fill_(extreme)
                assert model.prior_weight >= 0.0
                assert torch.isfinite(model.prior_weight)

    def test_prior_weight_receives_gradient(self):
        model = _make_model(update_input="grad+state", N_outer=3)
        batch = _MockBatch(B=2, T=20, D=3)
        loss = model.compute_loss(batch)
        loss.backward()
        assert model._prior_weight_raw.grad is not None
        assert torch.isfinite(model._prior_weight_raw.grad).all()

    def test_trainable_prior_weight_false_forces_fixed_value(self):
        for mode in ("grad-only", "grad+state"):
            model = _make_model(update_input=mode, prior_weight=1.7, trainable_prior_weight=False)
            assert model._prior_weight_raw is None
            assert model.prior_weight == 1.7

    def test_prior_unet_has_no_time_conditioning(self):
        """FourDVarNetSolver's prior_unet is built with time_emb_dim=0 -- no
        tau-conditioning pathway at all (unlike the main solver self.unet,
        which keeps it)."""
        for mode in ("grad-only", "grad+state", "subgrad+state"):
            model = _make_model(update_input=mode)
            assert model.prior_unet.time_emb_dim == 0
            assert model.unet.time_emb_dim > 0

    def test_prior_tau_conditioning_true_restores_legacy_architecture(self):
        """prior_tau_conditioning=True is for reproducing checkpoints trained
        before this became configurable -- prior_unet gets the same
        time_emb_dim as the main unet, and forward() feeds it a real tau."""
        for mode in ("grad-only", "grad+state", "subgrad+state"):
            model = _make_model(update_input=mode, prior_tau_conditioning=True,
                                 time_emb_dim=16)
            assert model.prior_unet.time_emb_dim == 16
            assert model.unet.time_emb_dim == 16

    def test_prior_tau_conditioning_true_runs_without_error(self):
        model = _make_model(update_input="grad+state", N_outer=3, prior_tau_conditioning=True)
        batch = _MockBatch(B=2, T=20, D=3)
        out = model(batch)
        assert torch.isfinite(out).all()

    def test_aux_var_cost_weight_zero_is_noop(self):
        model = _make_model(update_input="grad+state", N_outer=3, aux_var_cost_weight=0.0)
        model.eval()
        batch = _MockBatch(B=2, T=20, D=3, seed=0)
        loss = model.compute_loss(batch)
        expected = torch.nn.functional.mse_loss(model.forward(batch), batch.states)
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_aux_var_cost_weight_nonzero_changes_loss(self):
        model = _make_model(update_input="grad+state", N_outer=3, aux_var_cost_weight=0.1)
        batch = _MockBatch(B=2, T=20, D=3, seed=0)
        loss = model.compute_loss(batch)
        plain_mse = torch.nn.functional.mse_loss(model.forward(batch), batch.states)
        assert torch.isfinite(loss)
        assert not torch.allclose(loss, plain_mse, atol=1e-5)

    def test_aux_var_cost_weight_skipped_without_prior_unet(self):
        model = _make_model(update_input="obs+state", N_outer=3, aux_var_cost_weight=0.1)
        model.eval()
        batch = _MockBatch(B=2, T=20, D=3, seed=0)
        loss = model.compute_loss(batch)
        expected = torch.nn.functional.mse_loss(model.forward(batch), batch.states)
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_aux_var_cost_weight_gradients_flow_to_prior_weight_and_prior(self):
        model = _make_model(update_input="grad+state", N_outer=3, aux_var_cost_weight=0.1)
        batch = _MockBatch(B=2, T=20, D=3)
        loss = model.compute_loss(batch)
        loss.backward()
        assert model._prior_weight_raw.grad is not None
        assert torch.isfinite(model._prior_weight_raw.grad).all()
        prior_grads = [p.grad for p in model.prior_unet.parameters() if p.grad is not None]
        assert len(prior_grads) > 0
        assert all(torch.isfinite(g).all() for g in prior_grads)

    def test_aux_var_cost_weight_is_pure_prior_cost_no_obs_or_prior_weight(self):
        """Regression test for a real bug: an earlier version multiplied the
        prior term by ``self.prior_weight`` and added ``_masked_obs_cost``
        (i.e. the full per-iteration ``var_cost`` formula) inside this outer
        auxiliary term. That let a *trainable* ``prior_weight`` cheat the
        loss down by driving itself to 0 -- the obs_cost half doesn't depend
        on prior_weight, so zeroing it is a free win that has nothing to do
        with actual prior quality. Confirmed empirically: grad+state's
        prior_weight collapsed from ~0.97 to exactly 0.0 by epoch ~120 under
        the old formula (job 53104). The aux term must equal exactly
        ``aux_var_cost_weight * (prior_cost(x_final) + prior_cost(states)) /
        numel``, with no ``obs_cost``/``prior_weight`` factor mixed in --
        reconstructed independently here and checked for exact equality
        (not just "changes the loss"), since the old buggy formula would
        also have changed the loss, just to the wrong value."""
        model = _make_model(update_input="grad+state", N_outer=3, aux_var_cost_weight=0.1)
        model.eval()
        batch = _MockBatch(B=2, T=20, D=3, seed=0)

        loss = model.compute_loss(batch)

        from models.fourdvarnet import _prior_cost
        x_final = model.forward(batch)
        mse = F.mse_loss(x_final, batch.states)
        numel = x_final.numel()
        expected_aux = 0.1 * (
            _prior_cost(model.prior_unet, x_final) / numel
            + _prior_cost(model.prior_unet, batch.states) / numel
        )
        expected = mse + expected_aux

        assert torch.allclose(loss, expected, atol=1e-4)

    def test_unknown_update_input_raises_value_error(self):
        try:
            _make_model(update_input="not-a-real-mode")
            raise AssertionError("expected ValueError for an unrecognized update_input")
        except ValueError:
            pass

    def test_compute_loss_matches_final_iteration_mse(self):
        import torch.nn.functional as F
        model = _make_model(dropout=0.0)
        model.eval()
        batch = _MockBatch(B=2, T=50, D=3)
        loss = model.compute_loss(batch)
        expected = F.mse_loss(model(batch), batch.states)
        assert torch.allclose(loss, expected)

    def test_gradients_flow_through_unroll(self):
        model = _make_model(N_outer=4)
        batch = _MockBatch(B=2, T=20, D=3)
        loss = model.compute_loss(batch)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"no gradient reached {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite gradient at {name}"

    def test_deterministic_eval(self):
        model = _make_model(dropout=0.0)
        model.eval()
        batch = _MockBatch(B=1, T=30, D=3)
        with torch.no_grad():
            out_a = model(batch)
            out_b = model(batch)
        assert torch.equal(out_a, out_b)

    def test_state_dim_24(self):
        model = _make_model(state_dim=24, hidden_channels=[8, 16])
        batch = _MockBatch(B=2, T=60, D=24, obs_every=20)
        out = model(batch)
        assert out.shape == (2, 60, 24)
        assert torch.isfinite(out).all()

    def test_more_unroll_iterations_do_not_hurt_fit(self):
        torch.manual_seed(0)
        batch = _MockBatch(B=4, T=20, D=3, obs_every=5)

        def _fit_loss(N_outer, steps=20):
            torch.manual_seed(1)
            model = _make_model(N_outer=N_outer, hidden_channels=[4, 8])
            opt = torch.optim.Adam(model.parameters(), lr=1e-2)
            for _ in range(steps):
                opt.zero_grad()
                loss = model.compute_loss(batch)
                loss.backward()
                opt.step()
            return model.compute_loss(batch).item()

        loss_1 = _fit_loss(N_outer=1)
        loss_5 = _fit_loss(N_outer=5)
        assert loss_5 <= loss_1 + 0.5, (
            f"N_outer=5 fit loss ({loss_5}) should not be much worse than "
            f"N_outer=1 ({loss_1}) after equal optimizer steps"
        )


class TestSoftClip:
    """``_soft_clip`` -- the tanh-based replacement for the grad term's
    post-normalization torch.clamp."""

    def test_identity_at_zero(self):
        t = torch.zeros(5)
        assert torch.allclose(_soft_clip(t, 50.0), t)

    def test_near_identity_for_small_input(self):
        """|t| << clip_range: tanh(u) ~= u, so _soft_clip should barely
        perturb a small input."""
        t = torch.tensor([0.1, -0.3, 1.0, -2.0])
        out = _soft_clip(t, 50.0)
        assert torch.allclose(out, t, atol=1e-2)

    def test_converges_to_identity_as_clip_range_grows(self):
        """clip_range*tanh(t/clip_range) -> t as clip_range -> inf. Confirms
        the deviation from identity shrinks monotonically, not just that it's
        small at one arbitrary clip_range."""
        t = torch.tensor([5.0, -8.0, 20.0])
        errs = [(t - _soft_clip(t, cr)).abs().max().item() for cr in (10.0, 100.0, 1e4, 1e8)]
        assert errs == sorted(errs, reverse=True), f"errors should shrink monotonically: {errs}"
        assert errs[-1] < 1e-4

    def test_asymptotes_to_clip_range_for_large_input(self):
        t = torch.tensor([1e6, -1e6])
        out = _soft_clip(t, 5.0)
        assert torch.allclose(out, torch.tensor([5.0, -5.0]), atol=1e-3)

    def test_never_exceeds_clip_range(self):
        # Deliberately not an absurd outlier: at extreme |t|/clip_range
        # (e.g. 1000/5=200), tanh saturates to exactly 1.0 in float32, so
        # out.abs().max() lands AT clip_range, not strictly under it --
        # still a correct bound (never exceeds), just not strict. Moderate
        # outliers (up to ~30/5=6) stay strictly under, which is the
        # meaningful case this guards.
        t = torch.randn(1000) * 30.0
        out = _soft_clip(t, 5.0)
        assert out.abs().max().item() <= 5.0

    def test_gradient_nonzero_beyond_clip_range(self):
        """The actual motivating property vs. a hard torch.clamp: an element
        moderately past clip_range must still carry SOME gradient back
        toward the bound, unlike torch.clamp's exactly-zero gradient there.
        Uses t=20 with clip_range=5 (4x past the bound, not an absurd
        outlier) -- at extreme enough |t|/clip_range, tanh's float32 gradient
        underflows to exactly 0 too (same degenerate limit as a hard clamp),
        so this only holds within tanh's numerically-representable range."""
        t = torch.tensor([20.0], requires_grad=True)
        out = _soft_clip(t, 5.0)
        out.backward()
        assert t.grad is not None
        assert t.grad.abs().item() > 0, "gradient must not be exactly zero past the clip bound"

        t_clamped = torch.tensor([20.0], requires_grad=True)
        torch.clamp(t_clamped, -5.0, 5.0).sum().backward()
        assert t_clamped.grad.abs().item() == 0.0, "sanity check: hard clamp's own gradient IS exactly zero here"


class TestGradClipRange:
    """``grad_clip_range`` (None default -> falls back to clip_range) bounds
    only the grad-only/grad+state autograd gradient, independent of
    clip_range's own (hard-clamped, untouched) state-branch bound."""

    def test_default_falls_back_to_clip_range(self):
        model = _make_model(update_input="grad+state", clip_range=7.0)
        assert model.grad_clip_range == 7.0

    def test_explicit_value_is_independent_of_clip_range(self):
        model = _make_model(update_input="grad+state", clip_range=50.0, grad_clip_range=5.0)
        assert model.clip_range == 50.0
        assert model.grad_clip_range == 5.0

    def test_only_grad_term_is_affected(self):
        """A tighter grad_clip_range must change grad-only's output (its
        whole return value IS the grad term) but not obs+state's (which
        never touches the grad path at all)."""
        torch.manual_seed(0)
        model_wide = _make_model(update_input="grad-only", dropout=0.0, grad_clip_range=1e6)
        model_narrow = _make_model(update_input="grad-only", dropout=0.0, grad_clip_range=0.01)
        model_narrow.unet.load_state_dict(model_wide.unet.state_dict())
        model_narrow.prior_unet.load_state_dict(model_wide.prior_unet.state_dict())
        model_wide.eval()
        model_narrow.eval()
        batch = _MockBatch(B=2, T=20, D=3, seed=1)
        with torch.no_grad():
            out_wide = model_wide(batch)
            out_narrow = model_narrow(batch)
        assert not torch.allclose(out_wide, out_narrow)


def _make_cfm_model(**kwargs):
    defaults = dict(state_dim=3, hidden_channels=[4, 8], N_outer=3, K_inner=2)
    defaults.update(kwargs)
    return FourDVarNetPredictStateCFM(**defaults)


class TestFourDVarNetPredictStateCFM:
    def test_forward_shape_and_finite(self):
        model = _make_cfm_model()
        batch = _MockBatch(B=2, T=50, D=3)
        x_tau = torch.randn(2, 50, 3)
        tau = torch.rand(2)
        mu = model(x_tau, batch, tau)
        assert mu.shape == (2, 50, 3)
        assert torch.isfinite(mu).all()

    def test_compute_loss_matches_manual_formula(self):
        model = _make_cfm_model(dropout=0.0)
        model.eval()
        batch = _MockBatch(B=2, T=50, D=3)

        torch.manual_seed(5)
        loss = model.compute_loss(batch)

        torch.manual_seed(5)
        B, device = batch.states.shape[0], batch.states.device
        tau = torch.rand(B, device=device)
        x0 = torch.randn_like(batch.states) * model.sigma_prior
        x_tau = model.interpolant.mix(x0, batch.states, tau)
        mu_pred = model(x_tau, batch, tau)
        expected = F.mse_loss(mu_pred, batch.states)

        assert torch.allclose(loss, expected)

    def test_sample_shape_finite_and_stochastic(self):
        model = _make_cfm_model()
        model.eval()
        batch = _MockBatch(B=2, T=30, D=3)
        with torch.no_grad():
            a = model.sample(batch)
            b = model.sample(batch)
        assert a.shape == (2, 30, 3)
        assert torch.isfinite(a).all()
        assert not torch.allclose(a, b), "sample() must be stochastic (fresh noise init each call)"

    def test_k_inner_one_matches_fourdvarnet_solver_n_outer_one(self):
        """K_inner=1 degenerates to a single UNet call, structurally identical
        to FourDVarNetSolver with N_outer=1 -- same formula, just starting
        from an arbitrary x_t instead of a hardcoded zero state."""
        cfm = _make_cfm_model(K_inner=1, dropout=0.0)
        cfm.eval()
        solver = FourDVarNetSolver(state_dim=3, hidden_channels=[4, 8], N_outer=1, dropout=0.0)
        solver.eval()
        solver.unet.load_state_dict(cfm.unet.state_dict())
        batch = _MockBatch(B=2, T=20, D=3)
        x_t = torch.zeros(2, 20, 3)  # matches FourDVarNetSolver's own x_0 = 0
        tau = torch.rand(2)
        with torch.no_grad():
            mu = cfm(x_t, batch, tau)
            solver_out = solver(batch)
        assert torch.allclose(mu, solver_out)

    def test_grad_modes_shape_finite_and_have_prior_unet(self):
        for mode in _GRAD_MODES:
            model = _make_cfm_model(update_input=mode)
            assert model.prior_unet is not None, f"{mode} needs a prior_unet"
            batch = _MockBatch(B=2, T=20, D=3)
            x_tau = torch.randn(2, 20, 3)
            tau = torch.rand(2)
            mu = model(x_tau, batch, tau)
            assert mu.shape == (2, 20, 3)
            assert torch.isfinite(mu).all()

    def test_obs_state_modes_have_no_prior_unet(self):
        for mode in ("obs+state", "obs-only"):
            assert _make_cfm_model(update_input=mode).prior_unet is None

    def test_gradients_flow_through_inner_unroll_grad_modes(self):
        for mode in ("grad-only", "grad+state"):
            model = _make_cfm_model(update_input=mode, K_inner=4)
            batch = _MockBatch(B=2, T=20, D=3)
            loss = model.compute_loss(batch)
            loss.backward()
            for name, p in model.named_parameters():
                assert p.grad is not None, f"[{mode}] no gradient reached {name}"
                assert torch.isfinite(p.grad).all(), f"[{mode}] non-finite gradient at {name}"

    def test_gradients_flow_through_inner_unroll_subgrad_state(self):
        model = _make_cfm_model(update_input="subgrad+state", K_inner=4)
        batch = _MockBatch(B=2, T=20, D=3)
        loss = model.compute_loss(batch)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"no gradient reached {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite gradient at {name}"

    def test_gradients_flow_through_inner_unroll(self):
        model = _make_cfm_model(K_inner=4)
        batch = _MockBatch(B=2, T=20, D=3)
        loss = model.compute_loss(batch)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"no gradient reached {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite gradient at {name}"

    def test_forward_output_bounded_by_clip_range(self):
        """Regression guard for a rare (~1-in-several-thousand ens30 samples)
        divergence found during FDV1-CFM's first full evaluation: an
        out-of-distribution x_t can make the K_inner unroll blow up within a
        handful of steps, corrupting the ensemble-mean point estimate. Force
        the pathological regime with untrained (large-scale) weights and an
        adversarial x_t, and check forward() never exceeds clip_range."""
        torch.manual_seed(0)
        model = FourDVarNetPredictStateCFM(
            state_dim=3, hidden_channels=[4, 8], N_outer=3, K_inner=5, clip_range=50.0,
        )
        with torch.no_grad():
            for p in model.unet.parameters():
                p.mul_(20.0)
        batch = _MockBatch(B=4, T=20, D=3, seed=1)
        x_t = torch.randn(4, 20, 3) * 1000.0
        tau = torch.rand(4)
        mu = model(x_t, batch, tau)
        assert torch.isfinite(mu).all()
        assert mu.abs().max().item() <= 50.0 + 1e-4

    def test_clip_range_inactive_in_distribution(self):
        """The clamp must not perturb ordinary, in-distribution behavior --
        rerunning test_k_inner_one_matches_fourdvarnet_solver_n_outer_one-style
        inputs should give identical output whether or not clip_range binds."""
        model_clipped = _make_cfm_model(dropout=0.0, clip_range=50.0)
        model_unclipped = _make_cfm_model(dropout=0.0, clip_range=1e6)
        model_unclipped.load_state_dict(model_clipped.state_dict())
        model_clipped.eval()
        model_unclipped.eval()
        batch = _MockBatch(B=2, T=20, D=3)
        x_t = torch.randn(2, 20, 3)
        tau = torch.rand(2)
        with torch.no_grad():
            out_clipped = model_clipped(x_t, batch, tau)
            out_unclipped = model_unclipped(x_t, batch, tau)
        assert torch.allclose(out_clipped, out_unclipped)

    def test_sample_mean_estimate_none_reproduces_baseline(self):
        """mean_estimate=None must reproduce the pre-existing sample() exactly
        -- the regression invariant for the FDV1+FDV1-CFM warm-start hybrid."""
        model = _make_cfm_model(dropout=0.0)
        model.eval()
        batch = _MockBatch(B=2, T=20, D=3)
        torch.manual_seed(7)
        out_explicit_none = model.sample(batch, mean_estimate=None)
        torch.manual_seed(7)
        out_default = model.sample(batch)
        assert torch.allclose(out_explicit_none, out_default)

    def test_sample_tau0_zero_matches_baseline_up_to_the_snapped_start(self):
        """tau0=0.0 with a mean_estimate must still take step0=0 (pure noise
        start, no mixing) -- same code path as mean_estimate=None."""
        model = _make_cfm_model(dropout=0.0)
        model.eval()
        batch = _MockBatch(B=2, T=20, D=3)
        mean_estimate = torch.randn(2, 20, 3) * 10.0
        torch.manual_seed(3)
        out_tau0_zero = model.sample(batch, mean_estimate=mean_estimate, tau0=0.0)
        torch.manual_seed(3)
        out_baseline = model.sample(batch, mean_estimate=None)
        assert torch.allclose(out_tau0_zero, out_baseline)

    def test_sample_warm_start_matches_manual_replay(self):
        """tau0>0 must start from interpolant.mix(noise, mean_estimate, tau0)
        (snapped to the nearest step/N_outer) and run only the remaining
        steps -- replay the same noise draw manually and compare."""
        model = _make_cfm_model(N_outer=4, dropout=0.0)
        model.eval()
        batch = _MockBatch(B=2, T=20, D=3)
        mean_estimate = torch.randn(2, 20, 3) * 5.0
        tau0 = 0.5  # step0 = round(0.5*4) = 2
        torch.manual_seed(11)
        out = model.sample(batch, mean_estimate=mean_estimate, tau0=tau0)

        torch.manual_seed(11)
        noise = torch.randn(2, 20, 3) * model.sigma_prior
        x = model.interpolant.mix(noise, mean_estimate, torch.full((2,), 0.5))
        dt = 0.25
        with torch.no_grad():
            for step in range(2, 4):
                tau_val = step / 4
                tau = torch.full((2,), tau_val)
                mu = model(x, batch, tau)
                x = x + dt * (mu - x) / (1 - tau_val)
        assert torch.allclose(out, x)

    def test_sample_tau0_near_one_runs_a_single_step(self):
        model = _make_cfm_model(N_outer=10, dropout=0.0)
        model.eval()
        batch = _MockBatch(B=2, T=20, D=3)
        mean_estimate = torch.randn(2, 20, 3)
        with torch.no_grad():
            out = model.sample(batch, mean_estimate=mean_estimate, tau0=0.95)
        assert torch.isfinite(out).all()

    def test_sample_warm_start_still_stochastic(self):
        """Two calls with the same mean_estimate/tau0 must still differ
        (fresh noise each call) -- an ensemble around the anchor, not a
        single point."""
        model = _make_cfm_model(dropout=0.0)
        model.eval()
        batch = _MockBatch(B=2, T=20, D=3)
        mean_estimate = torch.randn(2, 20, 3)
        with torch.no_grad():
            a = model.sample(batch, mean_estimate=mean_estimate, tau0=0.5)
            b = model.sample(batch, mean_estimate=mean_estimate, tau0=0.5)
        assert not torch.allclose(a, b)


class TestGradientCheckpointing:
    """``FourDVarNetSolver.forward``/``FourDVarNetPredictStateCFM.forward``
    wrap each unrolled iteration in ``torch.utils.checkpoint.checkpoint(...,
    use_reentrant=False)`` (see ``_solver_iteration``) to keep activation
    memory from scaling with the unroll length. These tests confirm that's
    exactly transparent -- same output, same gradients, for every
    ``update_input`` mode including "grad-only"/"grad+state" (whose
    ``torch.autograd.grad(..., create_graph=True)`` inside the checkpointed
    region is the main risk `use_reentrant=False` is meant to cover) --
    by comparing against `_bypass_checkpoint`, which runs the identical
    forward() code path with checkpointing mechanically disabled.
    """

    def _assert_checkpoint_matches_reference(self, model, forward_fn):
        out_ckpt, grads_ckpt = _forward_backward_grads(model, forward_fn, seed=42)
        with patch("models.fourdvarnet.checkpoint", _bypass_checkpoint):
            out_plain, grads_plain = _forward_backward_grads(model, forward_fn, seed=42)
        assert torch.allclose(out_ckpt, out_plain, atol=1e-5), "checkpointed output diverged"
        assert grads_ckpt.keys() == grads_plain.keys()
        for name in grads_ckpt:
            g_ckpt, g_plain = grads_ckpt[name], grads_plain[name]
            assert (g_ckpt is None) == (g_plain is None), f"{name}: grad presence differs"
            if g_ckpt is not None:
                assert torch.allclose(g_ckpt, g_plain, atol=1e-4), f"{name}: gradient diverged"

    def test_solver_checkpoint_matches_uncheckpointed(self):
        for mode in _ALL_UPDATE_INPUT_MODES:
            model = _make_model(update_input=mode, N_outer=4, dropout=0.1)
            model.train()
            batch = _MockBatch(B=2, T=20, D=3, seed=0)
            self._assert_checkpoint_matches_reference(
                model, lambda model=model, batch=batch: model(batch))

    def test_predict_state_cfm_checkpoint_matches_uncheckpointed(self):
        for mode in _ALL_UPDATE_INPUT_MODES:
            model = _make_cfm_model(update_input=mode, K_inner=4, dropout=0.1)
            model.train()
            batch = _MockBatch(B=2, T=20, D=3, seed=0)
            x_tau = torch.randn(2, 20, 3)
            tau = torch.rand(2)
            self._assert_checkpoint_matches_reference(
                model,
                lambda model=model, x_tau=x_tau, batch=batch, tau=tau: model(x_tau, batch, tau))

    def test_double_backward_create_graph_survives_checkpoint(self):
        """The specific higher-order-autograd risk `use_reentrant=False` is
        meant to cover: "grad-only"/"grad+state" call
        ``torch.autograd.grad(..., create_graph=True)`` *inside* the
        checkpointed region, so the outer ``loss.backward()`` is itself a
        double-backward through that inner autograd.grad call, now combined
        with checkpoint recomputation. Must not raise, and must reach every
        parameter (including prior_unet's) with a finite gradient -- checked
        via the same ``compute_loss`` path actual training uses, not just a
        toy scalar."""
        for mode in ("grad-only", "grad+state"):
            model = _make_model(update_input=mode, N_outer=4)
            batch = _MockBatch(B=2, T=20, D=3)
            loss = model.compute_loss(batch)
            loss.backward()
            for name, p in model.named_parameters():
                assert p.grad is not None, f"[{mode}] no gradient reached {name}"
                assert torch.isfinite(p.grad).all(), f"[{mode}] non-finite gradient at {name}"


class TestPriorHiddenChannels:
    """``prior_hidden_channels`` (None by default -- prior_unet shares
    hidden_channels with the main solver unet)."""

    def test_default_shares_hidden_channels(self):
        model = _make_model(update_input="subgrad+state", hidden_channels=[8, 16])
        n_unet = sum(p.numel() for p in model.unet.parameters())
        n_prior = sum(p.numel() for p in model.prior_unet.parameters())
        # Not required to be exactly equal (different in_channel counts from
        # _UPDATE_INPUT_CHANNEL_MULTIPLIER), but same hidden_channels tier
        # means the same order of magnitude -- a narrower prior (tested
        # below) is what actually differs materially.
        assert n_prior > 0 and n_unet > 0

    def test_narrower_prior_has_fewer_params(self):
        model_equal = _make_model(update_input="subgrad+state", hidden_channels=[16, 32])
        model_narrow = _make_model(update_input="subgrad+state", hidden_channels=[16, 32],
                                    prior_hidden_channels=[8, 16])
        n_prior_equal = sum(p.numel() for p in model_equal.prior_unet.parameters())
        n_prior_narrow = sum(p.numel() for p in model_narrow.prior_unet.parameters())
        assert n_prior_narrow < n_prior_equal
        # The main solver unet is untouched by prior_hidden_channels.
        n_unet_equal = sum(p.numel() for p in model_equal.unet.parameters())
        n_unet_narrow = sum(p.numel() for p in model_narrow.unet.parameters())
        assert n_unet_equal == n_unet_narrow

    def test_only_affects_prior_modes(self):
        # obs+state has no prior_unet at all -- prior_hidden_channels is
        # simply unused, not an error.
        model = _make_model(update_input="obs+state", prior_hidden_channels=[4])
        assert model.prior_unet is None


class TestTruncatedBPTT:
    """``tbptt_n_blocks``/``tbptt_block_size`` -- splitting the N_outer
    unroll into detached blocks with a deep-supervision training loss,
    final-answer-only validation loss."""

    def test_default_is_single_block(self):
        model = _make_model(N_outer=4)
        assert model.tbptt_n_blocks == 1
        assert model.tbptt_block_size == 4

    def test_block_size_required_when_n_blocks_not_one(self):
        with pytest.raises(ValueError):
            _make_model(N_outer=4, tbptt_n_blocks=2)

    def test_mismatched_product_raises(self):
        with pytest.raises(ValueError):
            _make_model(N_outer=4, tbptt_n_blocks=2, tbptt_block_size=3)

    def test_matching_product_accepted(self):
        model = _make_model(N_outer=4, tbptt_n_blocks=2, tbptt_block_size=2)
        assert model.tbptt_n_blocks == 2
        assert model.tbptt_block_size == 2

    def test_unrolled_blocks_returns_one_per_block(self):
        model = _make_model(N_outer=4, tbptt_n_blocks=2, tbptt_block_size=2,
                             update_input="subgrad+state")
        batch = _MockBatch(B=2, T=20, D=3, seed=0)
        block_states = model._unrolled_blocks(batch)
        assert len(block_states) == 2
        for s in block_states:
            assert s.shape == (2, 20, 3)

    def test_single_block_equivalent_to_no_tbptt(self):
        model = _make_model(N_outer=4, update_input="subgrad+state")
        model.eval()  # dropout off -- this checks the block-count/plumbing,
        # not stochastic determinism (already covered by test_deterministic_eval)
        batch = _MockBatch(B=2, T=20, D=3, seed=0)
        block_states = model._unrolled_blocks(batch)
        assert len(block_states) == 1
        assert torch.equal(block_states[0], model(batch))

    def test_forward_value_unaffected_by_block_split(self):
        """Detach changes only the backward graph, not the forward values --
        a blocked and an unblocked model with identical weights must produce
        the exact same final state."""
        torch.manual_seed(0)
        model_full = _make_model(N_outer=4, update_input="subgrad+state", dropout=0.0)
        model_blocked = _make_model(N_outer=4, update_input="subgrad+state", dropout=0.0,
                                     tbptt_n_blocks=2, tbptt_block_size=2)
        model_blocked.load_state_dict(model_full.state_dict())
        model_full.eval()
        model_blocked.eval()
        batch = _MockBatch(B=2, T=20, D=3, seed=1)
        out_full = model_full(batch)
        out_blocked = model_blocked(batch)
        assert torch.allclose(out_full, out_blocked, atol=1e-6)

    def test_training_loss_averages_across_blocks(self):
        model = _make_model(N_outer=4, update_input="subgrad+state", dropout=0.0,
                             tbptt_n_blocks=2, tbptt_block_size=2)
        model.train()
        batch = _MockBatch(B=2, T=20, D=3, seed=2)
        torch.manual_seed(3)
        loss = model.compute_loss(batch)
        torch.manual_seed(3)
        block_states = model._unrolled_blocks(batch)
        expected = sum(F.mse_loss(s, batch.states) for s in block_states) / len(block_states)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_eval_loss_uses_final_block_only(self):
        """The specific behavior requested: val_loss must reflect only the
        final answer, never a block average, regardless of tbptt_n_blocks."""
        model = _make_model(N_outer=4, update_input="subgrad+state", dropout=0.0,
                             tbptt_n_blocks=2, tbptt_block_size=2)
        model.eval()
        batch = _MockBatch(B=2, T=20, D=3, seed=4)
        loss_eval = model.compute_loss(batch)
        block_states = model._unrolled_blocks(batch)
        x_final = block_states[-1]
        expected_final_only = F.mse_loss(x_final, batch.states)
        block_average = sum(F.mse_loss(s, batch.states) for s in block_states) / len(block_states)
        assert torch.allclose(loss_eval, expected_final_only, atol=1e-6)
        # Sanity: the block average genuinely differs from the final-only
        # loss for this batch/model (otherwise the two branches would be
        # indistinguishable and this test wouldn't prove anything).
        assert not torch.allclose(loss_eval, block_average, atol=1e-6)

    def test_gradient_reaches_shared_weights_from_both_blocks(self):
        """Weight-tying means block 1's own forward pass still needs a
        gradient signal -- detach severs the *activation* path from block 2
        into block 1, not the shared parameters' training signal."""
        model = _make_model(N_outer=4, update_input="subgrad+state",
                             tbptt_n_blocks=2, tbptt_block_size=2)
        model.train()
        batch = _MockBatch(B=2, T=20, D=3, seed=5)
        loss = model.compute_loss(batch)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"no gradient reached {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite gradient at {name}"
