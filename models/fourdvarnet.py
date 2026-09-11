import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.interpolant import LinearInterpolant
from models.unet import UNet1D
try:
    from models.monai_unet_adapter import MonaiUNet1D
except ImportError:
    # monai is an optional, deliberately-isolated dependency (see
    # models/monai_unet_adapter.py) -- not installed by default. The
    # sentinel class below is never constructed; _build_backbone_unet raises
    # a clear error before reaching it if unet_backbone="monai" is requested
    # without monai installed.
    class MonaiUNet1D:
        pass

_VALID_UNET_BACKBONES = ("unet1d", "monai")


def _validate_unet_backbone(unet_backbone):
    if unet_backbone not in _VALID_UNET_BACKBONES:
        raise ValueError(
            f"Unknown unet_backbone={unet_backbone!r}; expected one of {_VALID_UNET_BACKBONES}"
        )


def _build_backbone_unet(unet_backbone, *, state_dim, hidden_channels, time_emb_dim,
                          dropout, output_dim, monai_norm_num_groups=32):
    """Dispatches ``self.unet``/``self.prior_unet`` construction between
    ``UNet1D`` (default) and ``models.monai_unet_adapter.MonaiUNet1D``
    (``unet_backbone="monai"``) -- both built with ``use_obs=False``
    (FDV's own channel-concat convention: whatever conditioning the caller
    wants is already concatenated into ``state_dim`` before this call, see
    ``_UPDATE_INPUT_CHANNEL_MULTIPLIER``/``_build_update_input``), so the two
    backbones are true drop-in replacements for each other at every call site
    in this module -- ``forward(x, tau=...)`` has the same signature and
    return shape either way.

    ``MonaiUNet1D`` has no ``time_emb_dim``-style architectural switch to
    fully omit tau-conditioning (unlike ``UNet1D(time_emb_dim=0)``, a literal
    omission of the conditioning pathway): passing ``tau=None`` at call time
    (this module's convention for an "unconditioned" ``prior_unet``, see
    ``_prior_ae``) makes ``MonaiUNet1D`` feed a constant zero timestep through
    its real, trainable time-embedding/FiLM layers instead -- functionally
    close (the embedding never varies) but not architecturally identical
    (those parameters still exist and run). ``time_emb_dim`` is accordingly
    ignored for ``unet_backbone="monai"``: MONAI's ``DiffusionModelUNet``
    always carries its own internal time embedding, sized by its own
    ``channels``, not by ``time_emb_dim``.
    """
    _validate_unet_backbone(unet_backbone)
    if unet_backbone == "unet1d":
        return UNet1D(
            state_dim=state_dim,
            hidden_channels=hidden_channels,
            time_emb_dim=time_emb_dim,
            use_obs=False,
            use_energy=False,
            dropout=dropout,
            output_dim=output_dim,
        )
    if MonaiUNet1D.__module__ == __name__:
        raise ImportError(
            "unet_backbone='monai' requires the optional 'monai' package "
            "(not installed in this environment) -- see models/monai_unet_adapter.py"
        )
    return MonaiUNet1D(
        state_dim=state_dim,
        hidden_channels=hidden_channels,
        use_obs=False,
        output_dim=output_dim,
        dropout=dropout,
        norm_num_groups=monai_norm_num_groups,
    )

# Recognized update_input tokens (mirrors the config-string taxonomy explored on
# CIA-Oceanix/4dvarnet-global-mapping's ronan_devs branch, contrib/4dvarnet_latent/
# models.py::GradSolver_withStep). "obs+state"/"obs-only" are gradient-free
# (the per-iteration UNet is fed the raw state/obs, no cost function at all).
# "grad-only"/"grad+state"/"subgrad+state" ("FDV2") are gradient-conditioned:
# a variational cost prior_cost(state) + obs_weight*obs_cost(state, obs) is
# built each iteration, and either its real autograd gradient ("grad-only"/
# "grad+state", ported from ocean4dvarnet's GradSolver -- see _build_update_input)
# or a cheap two-residual proxy that never calls autograd ("subgrad+state",
# ported from ronan_devs' GradSolver_withStep) is fed to the update UNet.
_IMPLEMENTED_UPDATE_INPUTS = ("obs+state", "obs-only", "grad-only", "grad+state", "subgrad+state")

# Modes needing a real torch.autograd.grad call each iteration.
_AUTOGRAD_MODES = ("grad-only", "grad+state")
# Modes needing the trainable prior operator (prior_unet).
_PRIOR_MODES = ("grad-only", "grad+state", "subgrad+state")
# Number of state_dim-sized channel blocks the main update UNet's input has,
# per mode -- drives in_state_dim at construction time.
_UPDATE_INPUT_CHANNEL_MULTIPLIER = {
    "obs-only": 1,
    "obs+state": 2,
    "grad-only": 1,
    "grad+state": 2,
    "subgrad+state": 3,
}


def _validate_update_input(update_input):
    if update_input not in _IMPLEMENTED_UPDATE_INPUTS:
        raise ValueError(
            f"Unknown update_input={update_input!r}; expected one of {_IMPLEMENTED_UPDATE_INPUTS}"
        )


def _init_positive_weight_raw(init_value, min_value):
    """sqrt(init_value - min_value): the raw (unconstrained) parameter value
    such that ``min_value + raw**2 == init_value`` at construction time --
    lets a trainable weight START at the same value the (formerly fixed)
    ``obs_weight`` config default used, while still being free to move via
    ordinary backprop from there."""
    y = init_value - min_value
    if y < 0:
        raise ValueError(f"obs_weight ({init_value}) must be >= min_obs_weight ({min_value})")
    return y ** 0.5


def _positive_weight_value(raw, min_value):
    """min_value + raw**2 -- always >= min_value, smoothly trainable via
    ordinary backprop, never negative regardless of raw's value."""
    return min_value + raw ** 2


def _masked_obs_cost(state, obs_clean, obs_mask, R_var):
    """sum(||obs - state||^2 * mask) / R_var -- a masked *sum* divided by the
    fixed scalar R_var (not a per-observation mean): the classical weak-
    constraint 4D-Var convention, matching evaluation/sda_sampler.py's
    guided_obs_cost and evaluation/baselines.py's Strong4DVar exactly.
    Deliberately NOT normalized by the observed-element count, so the cost
    doesn't depend on how many positions happen to be observed -- each
    observation contributes independently to the total penalty, the same way
    classical 4D-Var's J_o = sum_t (H(x_t)-y_t)^T R^-1 (H(x_t)-y_t) does. (A
    masked-mean variant, matching ocean4dvarnet's own ML-style
    F.mse_loss-based BaseObsCost, was tried and reverted -- this codebase's
    own classical-DA convention is what's actually trained under, including
    by the currently-running FDV2 job, so this must stay consistent with it.)
    """
    diff = (state - obs_clean) * obs_mask
    return diff.pow(2).sum() / R_var


def _prior_ae(prior_unet, state, tau=None):
    """The trainable prior operator Phi(state), applied channel-last -> UNet1D's
    channel-first convention and back. ``tau=None`` (the default) applies no
    time-conditioning at all (``UNet1D.forward`` skips its time-embedding
    branch entirely when ``tau is None``) -- used by ``FourDVarNetSolver``,
    which deliberately keeps iteration-conditioning in the main solver UNet
    only: the prior is meant as a fixed background/regularization operator,
    not one that behaves differently per unrolled iteration. Passing a real
    ``tau`` (``FourDVarNetPredictStateCFM``'s own usage, unchanged) applies
    the same per-iteration tau embedding as the main update UNet."""
    return prior_unet(state.transpose(1, 2), tau=tau).transpose(1, 2)


def _prior_cost(prior_unet, state, tau=None):
    """sum((state - Phi(state))^2) -- MSE(state, Phi(state)) with
    reduction="sum", matching _masked_obs_cost's sum-based (not
    count-normalized) convention, so the two terms combine consistently in
    var_cost. Phi = prior_unet (ocean4dvarnet's BilinAEPriorCost/ronan_devs'
    GenericAEPriorCost formula). See ``_prior_ae`` re: ``tau=None``."""
    return F.mse_loss(state, _prior_ae(prior_unet, state, tau), reduction="sum")


def _soft_clip(t, clip_range):
    """``clip_range * tanh(t / clip_range)`` -- a smooth, everywhere-
    differentiable alternative to ``torch.clamp(t, -clip_range, clip_range)``.
    Identity-ish for ``|t| << clip_range`` (``tanh(u) ~= u`` near 0), asymptotes
    smoothly to ``+/-clip_range`` for ``|t| >> clip_range``, but -- unlike a
    hard clamp, whose gradient is exactly zero the instant a value saturates
    -- keeps a small but nonzero gradient everywhere, so an element that
    strays past the bound still gets some training signal pulling it back
    in, instead of going permanently dead. Used only for the ``grad`` term's
    post-normalization bound (``_normalize_channels``); the per-iteration
    state-branch clamps in ``FourDVarNetSolver``/``FourDVarNetPredictStateCFM``
    still use a hard ``torch.clamp`` -- not in scope here."""
    return clip_range * torch.tanh(t / clip_range)


def _normalize_channels(t, cache=None, key=None, clip_range=50.0):
    """RMS normalization by a single global (whole-tensor) scalar -- matches
    ocean4dvarnet's ``ConvLstmGradModel.forward`` exactly: ``self._grad_norm
    = (x**2).mean().sqrt(); x = x / self._grad_norm``. A raw, unnormalized
    gradient/residual channel can grow arbitrarily large in magnitude (the
    underlying var_cost is an unbounded sum-of-squares over all elements),
    and feeding that directly into a plain UNet1D is numerically fragile over
    long training runs.

    ``norm`` is floored at ``1e-8`` but that alone doesn't bound the output:
    late in training the raw gradient's RMS can shrink toward that floor
    (both prior/obs residuals shrinking as the model converges), and dividing
    by a near-zero norm inflates ``t / norm`` unboundedly -- this fed a
    ``grad+state`` MonaiUNet1D run's ``create_graph=True`` double-backward
    into a NaN (job 52672, 2026-09-10). Bounded via ``_soft_clip`` (a smooth
    ``tanh``-based soft-clip, not a hard ``torch.clamp``) so an element that
    strays past ``clip_range`` still carries a small gradient back toward the
    bound, instead of the exactly-zero gradient a hard clamp gives there --
    this normalization is deliberately kept (not removed, unlike
    "subgrad+state"'s own residual channels) per prior published results
    (Fablet et al., JAMES) on its importance for this class of gradient-
    conditioned solver.

    ``cache``/``key`` (both optional) reproduce the *caching* granularity
    ocean4dvarnet also uses: the norm is computed once, on the first call for
    a given ``key`` within one unrolled solve, then reused unchanged for
    every subsequent iteration of that same solve (``ConvLstmGradModel``'s
    own ``self._grad_norm``, reset via ``reset_state`` once per batch). The
    caller (``FourDVarNetSolver.forward``/``FourDVarNetPredictStateCFM.forward``)
    creates a fresh ``{}`` once per ``forward()`` call (one call = one
    complete unrolled solve). Without a cache (``cache=None``, e.g. the unit
    tests that call this directly), the norm is just computed fresh every
    call.

    The norm is detached (a stop-gradient scale factor) so the normalized
    tensor stays fully differentiable w.r.t. upstream parameters (via the
    un-normalized numerator) without backpropagating through the norm
    computation itself.

    Deliberately computes ``norm`` unconditionally (even on a cache hit,
    where the fresh value is immediately discarded via ``setdefault``)
    rather than branching on ``key in cache`` to skip the computation: each
    call site is wrapped in ``torch.utils.checkpoint.checkpoint`` (see
    ``_solver_iteration``), which requires a checkpointed segment's
    recomputation (during backward) to perform the *exact* same ops as its
    original forward -- a cache-hit branch that skips computing ``norm``
    would make the very first iteration's checkpoint recompute (which always
    lands after the whole unrolled solve has already populated the cache)
    diverge from that same iteration's original forward (a cache miss at the
    time it first ran), raising ``torch.utils.checkpoint.CheckpointError:
    ... different number of tensors was saved``. Unconditional computation
    keeps the op sequence identical regardless of cache state; the discarded
    norm is detached and unused, so this changes no gradient, only adds one
    redundant (and cheap) elementwise reduction per already-cached call.
    """
    norm = (t ** 2).mean().sqrt().clamp_min(1e-8).detach()
    if cache is not None:
        norm = cache.setdefault(key, norm)
    return _soft_clip(t / norm, clip_range)


def _build_update_input(update_input, x, obs_clean, obs_mask, tau,
                         prior_unet=None, R_var=0.5, obs_weight=1.0, prior_weight=1.0,
                         grad_norm_cache=None, clip_range=50.0):
    """Returns the tensor fed to the main per-iteration update UNet.

    "grad-only"/"grad+state" compute a real autograd gradient of
    ``prior_weight*prior_cost(x) + obs_weight*obs_cost(x, obs)`` w.r.t. ``x``
    (``prior_weight``/``obs_weight`` both default to the neutral 1.0; only one
    of the two is ever trainable per model -- ``FourDVarNetSolver`` trains
    ``prior_weight`` (obs_weight fixed at 1.0: the observation noise model,
    R_var, is already known, unlike the prior), ``FourDVarNetPredictStateCFM``
    trains ``obs_weight`` (prior_weight fixed at 1.0, its original/unchanged
    behavior) --
    ``torch.autograd.grad(var_cost, x, create_graph=True)[0]`` (ported from
    ocean4dvarnet's ``GradSolver.solver_step``) -- wrapped in
    ``torch.enable_grad()`` so this works even when called from inside an
    outer ``torch.no_grad()`` eval loop (same pattern as
    ``evaluation/sda_sampler.py::sda_guided_sample``). ``create_graph=True``
    (unlike SDA's inference-only guidance) because here the OUTER training
    loop backprops through this gradient computation itself to train
    ``prior_unet``/the main UNet -- the caller (``forward()``) is responsible
    for ensuring ``x`` is a ``requires_grad`` tensor before this is called
    (leaf-ified once at the start of the unroll, re-leafed after each step
    only at eval time -- see ``FourDVarNetSolver.forward``/
    ``FourDVarNetPredictStateCFM.forward``).

    "subgrad+state" (ported from ronan_devs' ``GradSolver_withStep``) never
    calls autograd -- it's a cheap two-residual proxy: the raw observation
    residual ``obs - x`` and the prior-autoencoder reconstruction residual
    ``x - Phi(x)`` (the standard denoising-residual approximation of
    ``prior_cost``'s true gradient, valid when treating ``Phi(x)`` as locally
    constant), concatenated with ``x``. Cheaper per iteration (plain
    forward-mode tensor ops only) while remaining fully backpropagable
    through ``prior_unet``'s parameters via ordinary autograd.

    Deliberately NOT run through ``_normalize_channels`` (unlike
    "grad-only"/"grad+state"'s real autograd gradient below, which does need
    it -- an unbounded var_cost gradient can grow arbitrarily large).
    ``obs - x`` and ``x - Phi(x)`` are differences of two already-comparable-
    scale quantities (``obs``/``x`` both live in the same normalized state
    space), so they need no extra rescaling -- and this specific channel is
    architecturally masked to exactly zero outside observation times
    (``obs_mask``), unlike the dense ``grad`` tensor: a *global* whole-tensor
    RMS norm computed over that mostly-zero tensor is diluted by the
    observation density (``sqrt(1/obs_density)`` too small, e.g. 10x for
    ``obs_interval=100``), which then inflates the sparse nonzero entries by
    that same factor at every observed timestep -- confirmed empirically as
    the root cause of a severe fast-Y reconstruction collapse (variance
    ratio ~0.6 vs ~0.9+ for every other scheme) across every "subgrad+state"
    MonaiUNet1D run trained under the old normalized version, independent of
    which capacity/tbptt config was used. Matches ronan_devs' own
    ``GradSolver_withStep``, which never normalizes ``gobs``/``gprior``
    either -- only the real-autograd ``grad`` branch's combined tensor goes
    through its ``ConvLstmGradModel``'s internal norm.

    Simplification vs. the ronan_devs port: that implementation adds an extra
    ``lr_grad``-weighted raw-gradient term to the *outer* state update
    whenever the Python substring check ``'grad' in input_grad_update`` is
    true -- which (read literally) also matches ``"subgrad+state"`` (since
    ``"subgrad"`` contains ``"grad"``), a likely-unintentional legacy quirk.
    This port keeps the three gradient-conditioned modes fully separate (no
    such cross-talk) and does not add any extra outer-loop term at all --
    only what feeds the main UNet changes per mode; the existing
    ``x - (1/N)*gmod`` update rule (shared with "obs+state"/"obs-only") is
    unchanged.
    """
    if update_input == "obs-only":
        return obs_clean
    if update_input == "obs+state":
        return torch.cat([x, obs_clean], dim=-1)
    if update_input == "subgrad+state":
        g_obs = (obs_clean - x) * obs_mask
        g_prior = x - _prior_ae(prior_unet, x, tau)
        return torch.cat([g_obs, g_prior, x], dim=-1)
    # grad-only / grad+state
    with torch.enable_grad():
        var_cost = prior_weight * _prior_cost(prior_unet, x, tau) \
            + obs_weight * _masked_obs_cost(x, obs_clean, obs_mask, R_var)
        grad = _normalize_channels(torch.autograd.grad(var_cost, x, create_graph=True)[0],
                                    cache=grad_norm_cache, key="grad", clip_range=clip_range)
    if update_input == "grad-only":
        return grad
    return torch.cat([grad, x], dim=-1)


def _solver_iteration(unet, update_input, x, obs_clean, obs_mask, tau_k, prior_tau_k,
                       prior_unet, R_var, prior_weight, obs_weight, grad_norm_cache,
                       clip_range=50.0):
    """One unrolled solver step -- build the per-iteration update-UNet input
    (``_build_update_input``) then run the main solver UNet -- factored out
    of ``FourDVarNetSolver.forward``/``FourDVarNetPredictStateCFM.forward``
    so both can wrap a single call to it in
    ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`` instead
    of keeping every iteration's UNet forward (and, for the gradient-
    conditioned modes, the prior-operator forward inside
    ``_build_update_input``) alive for backprop: activation memory would
    otherwise scale linearly with the unroll length (``N_outer``/``K_inner``)
    with no bound. ``use_reentrant=False`` specifically -- not the older
    reentrant checkpoint implementation -- because "grad-only"/"grad+state"
    call ``torch.autograd.grad(..., create_graph=True)`` inside
    ``_build_update_input``, and only the non-reentrant checkpoint supports
    nested/higher-order autograd correctly.

    Safe to checkpoint unconditionally (no config flag): under
    ``torch.no_grad()`` (eval/sampling), ``checkpoint`` just runs the
    function directly with no recomputation, so this adds no eval-time cost.
    ``grad_norm_cache`` (a plain dict, not a tensor -- passed through
    unchanged across the checkpoint boundary) is always fully populated by
    the very first iteration's real forward pass, before ``backward()`` is
    ever called, so a checkpoint-triggered recomputation of the ``k=0``
    iteration during backward always hits the already-cached branch in
    ``_normalize_channels`` and reproduces the exact same norm -- no stale-
    cache risk despite the forward code re-running.
    """
    inp = _build_update_input(update_input, x, obs_clean, obs_mask, prior_tau_k,
                               prior_unet=prior_unet, R_var=R_var,
                               obs_weight=obs_weight, prior_weight=prior_weight,
                               grad_norm_cache=grad_norm_cache, clip_range=clip_range).transpose(1, 2)
    return unet(inp, tau=tau_k).transpose(1, 2)


class FourDVarNetSolver(nn.Module):
    """Unrolled 4DVarNet-style solver: the per-iteration update is the output
    of a UNet fed a mode-dependent input built from the current state,
    observations, and/or a variational-cost (sub)gradient, run for a fixed
    number of iterations.

    ``update_input`` selects what the update block sees each iteration,
    matching the config-string taxonomy explored on
    CIA-Oceanix/4dvarnet-global-mapping's ``ronan_devs`` branch
    (``GradSolver_withStep``'s ``input_grad_update``):

    - ``"obs+state"`` (default): ``concat(state, obs)``, no cost function at
      all -- the original FDV1 variant.
    - ``"obs-only"``: just the observations, no state feedback.
    - ``"grad-only"``/``"grad+state"``: the real autograd gradient of
      ``prior_cost(state) + obs_weight*obs_cost(state, obs)`` w.r.t. state
      (ported from ``ocean4dvarnet``'s ``GradSolver``), alone or concatenated
      with state. See ``_build_update_input``.
    - ``"subgrad+state"``: a cheap two-residual proxy gradient (obs residual +
      prior-autoencoder residual), concatenated with state, no autograd call.

    ``unet_backbone`` ("unet1d" default, or "monai") selects the nn.Module
    class backing ``self.unet``/``self.prior_unet`` -- ``models.unet.UNet1D``
    or ``models.monai_unet_adapter.MonaiUNet1D`` (MONAI's DiffusionModelUNet,
    already validated as a drop-in backbone for DirectUNet, see
    reports/l96/outputs/l96_normalization_ablation.md). Both are built with
    ``use_obs=False`` and have the identical ``forward(x, tau=...)`` call
    signature, so this is a pure backbone swap -- no other FDV logic changes.
    See ``_build_backbone_unet`` for the one known semantic gap (no true
    tau-conditioning omission for the Monai-backed ``prior_unet``).

    The gradient-conditioned modes need a trainable prior operator
    (``self.prior_unet``, a second ``UNet1D`` sharing the main UNet's
    ``hidden_channels``/``time_emb_dim``/``dropout`` -- mirrors
    ``TweedieCFM``'s ``mean_estimator``/``velocity_unet`` hyperparameter
    sharing), constructed only when ``update_input`` needs it so
    ``"obs+state"``/``"obs-only"`` checkpoints stay exactly as before (no
    dead weights).

    Unlike ``"obs+state"``/``"obs-only"`` (a bounded-by-construction update,
    empirically stable throughout FDV1's own training -- no clamp was ever
    needed there), the gradient-conditioned modes couple ``x`` into a
    variational cost whose gradient feeds back into the next iteration's
    input, repeated ``N_outer`` times with no bound -- exactly the kind of
    loop that made ``FourDVarNetPredictStateCFM`` diverge during its own
    training (see that class's docstring) before it gained a ``clip_range``
    clamp. ``FourDVarNetSolver`` lacked the same clamp entirely (only
    ``FourDVarNetPredictStateCFM`` had one) until a "grad+state" FDV1
    (deterministic) training run never trained at all -- stuck flat from
    epoch 0, unlike the CFM variant, which trained normally for hundreds of
    epochs before eventually diverging. Clamping ``x`` to
    ``[-clip_range, clip_range]`` after each iteration (same convention as
    ``models/lorenz96_dynamics.py``, ``evaluation/baselines.py``, and
    ``FourDVarNetPredictStateCFM`` itself) closes this gap; inactive for
    ``"obs+state"``/``"obs-only"`` in practice (in-distribution state range
    ``|x|<10``, well inside the default ``clip_range=50.0``). This is a hard
    ``torch.clamp``, unrelated to (and unaffected by) ``grad_clip_range``
    below, which bounds only the "grad-only"/"grad+state" gradient term via
    a smooth ``tanh`` soft-clip -- see ``_soft_clip``/``_normalize_channels``.
    """

    def __init__(self, state_dim=24, hidden_channels=None, time_emb_dim=64,
                 N_outer=10, dropout=0.1, update_input="obs+state",
                 R_var=0.5, prior_weight=1.0, clip_range=50.0,
                 trainable_prior_weight=True,
                 aux_var_cost_weight=0.0,
                 prior_tau_conditioning=False,
                 unet_backbone="unet1d",
                 monai_norm_num_groups=32,
                 prior_hidden_channels=None,
                 tbptt_n_blocks=1,
                 tbptt_block_size=None,
                 grad_clip_range=None):
        super().__init__()
        _validate_update_input(update_input)
        _validate_unet_backbone(unet_backbone)
        if unet_backbone == "monai" and prior_tau_conditioning:
            raise ValueError(
                "prior_tau_conditioning=True exists only to reproduce legacy "
                "UNet1D checkpoints trained with a tau-conditioned prior_unet -- "
                "no such MonaiUNet1D checkpoint exists, so this combination is "
                "not supported (see _build_backbone_unet)."
            )
        if tbptt_block_size is None:
            if tbptt_n_blocks != 1:
                raise ValueError(
                    "tbptt_block_size must be set explicitly whenever "
                    "tbptt_n_blocks != 1 -- both must be given together in "
                    "the config, no derivation from N_outer alone."
                )
            tbptt_block_size = N_outer
        if tbptt_n_blocks * tbptt_block_size != N_outer:
            raise ValueError(
                f"tbptt_n_blocks ({tbptt_n_blocks}) * tbptt_block_size "
                f"({tbptt_block_size}) must equal N_outer ({N_outer})."
            )
        self.update_input = update_input
        self.state_dim = state_dim
        self.N_outer = N_outer
        self.R_var = R_var
        self.clip_range = clip_range
        # grad_clip_range (None default -> falls back to clip_range, today's
        # behavior): bounds ONLY the grad-only/grad+state autograd gradient
        # after _normalize_channels' RMS division -- deliberately independent
        # of clip_range, which bounds the raw state branch every iteration
        # (a hard clamp, unrelated mechanism, unaffected by this). Kept
        # separate so tightening the grad-term bound (e.g. to better engage
        # _soft_clip's nonlinearity near its actual operating range) doesn't
        # also start clipping legitimate state excursions.
        self.grad_clip_range = grad_clip_range if grad_clip_range is not None else clip_range
        self.aux_var_cost_weight = aux_var_cost_weight
        self.prior_tau_conditioning = prior_tau_conditioning
        self.unet_backbone = unet_backbone
        self.tbptt_n_blocks = tbptt_n_blocks
        self.tbptt_block_size = tbptt_block_size
        self._prior_weight_raw = None
        self._prior_weight_fixed = prior_weight
        if update_input in _AUTOGRAD_MODES and trainable_prior_weight:
            self._prior_weight_raw = nn.Parameter(torch.tensor(
                prior_weight ** 0.5, dtype=torch.float32))
        in_state_dim = _UPDATE_INPUT_CHANNEL_MULTIPLIER[update_input] * state_dim
        self.unet = _build_backbone_unet(
            unet_backbone,
            state_dim=in_state_dim,
            hidden_channels=hidden_channels,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
            output_dim=state_dim,
            monai_norm_num_groups=monai_norm_num_groups,
        )
        self.prior_unet = None
        if update_input in _PRIOR_MODES:
            # prior_tau_conditioning=False (the default): time_emb_dim=0, no
            # iteration/tau conditioning at all for the prior operator
            # (architecturally absent for unet1d, not just unfed -- for
            # unet_backbone="monai" this is instead enforced by always
            # calling with tau=None, see _build_backbone_unet's docstring)
            # -- the prior is a fixed background/regularization operator,
            # unlike the main solver ``self.unet`` above, which keeps its
            # per-iteration tau conditioning (time_emb_dim=time_emb_dim)
            # unchanged. See ``forward()`` and ``_prior_ae``.
            #
            # prior_tau_conditioning=True exists ONLY for reproducing
            # checkpoints trained before this became configurable (e.g.
            # FDV2_grad_state_l96_fixedw's job 52205), whose prior_unet *was*
            # tau-conditioned at train time -- reconstructing such a
            # checkpoint with prior_tau_conditioning=False silently drops its
            # real time_proj weights (shape-mismatch skip in load_model),
            # evaluating a model that behaves differently from how it was
            # actually trained. New configs should leave this False.
            #
            # prior_hidden_channels (None by default): the prior_unet shares
            # the main solver's hidden_channels unless a narrower tier is
            # given explicitly here -- see ronan_devs' own convention
            # (glo12-sla-4th-unrolling-ossev1.yaml gives the prior UNet half
            # the solver's model_channels), which this codebase did not
            # previously reproduce (both networks were always equal capacity).
            self.prior_unet = _build_backbone_unet(
                unet_backbone,
                state_dim=state_dim,
                hidden_channels=(prior_hidden_channels if prior_hidden_channels is not None
                                  else hidden_channels),
                time_emb_dim=(time_emb_dim if prior_tau_conditioning else 0),
                dropout=dropout,
                output_dim=state_dim,
                monai_norm_num_groups=monai_norm_num_groups,
            )

    @property
    def prior_weight(self):
        """The prior_cost weight in var_cost = prior_weight*prior_cost +
        obs_cost (obs_cost's own weight is fixed at 1.0 always: its scaling
        is already fully determined by the known observation noise model,
        R_var, unlike the prior, which is itself a learned operator with no
        a priori known scale). Trainable via a plain ``raw**2``
        reparametrization -- squaring alone already guarantees
        non-negativity, no floor needed: unlike a trainable obs_weight
        (which must stay bounded away from zero, since zeroing it would mean
        ignoring all observations -- empirically unstable, see git history),
        prior_weight -> 0 is a valid, non-catastrophic limit (pure
        strong-constraint "trust the observations fully", no prior
        regularization) -- for the modes that actually use it
        ("grad-only"/"grad+state"); a plain fixed float otherwise (unused by
        "obs+state"/"obs-only"/"subgrad+state")."""
        if self._prior_weight_raw is None:
            return self._prior_weight_fixed
        return self._prior_weight_raw ** 2

    def _unrolled_blocks(self, batch, N_outer=None):
        """Runs the ``N``-iteration unroll and returns the state at the end
        of every truncated-BPTT block (``self.tbptt_n_blocks`` elements, the
        last being the usual final estimate). ``forward()`` returns just the
        last one (unchanged external contract); ``compute_loss()`` averages
        an MSE term over all of them.

        Block-truncated BPTT only applies when running at the configured
        ``self.N_outer`` (``N_outer=None``) with ``self.tbptt_n_blocks>1`` --
        an explicit ``N_outer`` override (e.g. eval-time ``--n-outer``) always
        runs as one continuous block, since ``self.tbptt_block_size`` need
        not divide an arbitrary override. Between blocks, ``x`` is
        ``.detach()``-ed (severing the backward graph there -- standard
        truncated-BPTT for this weight-tied unroll, ported from the
        ``detach()``-at-a-stage-boundary + averaged multi-stage loss pattern
        in ``4dvarnet-global-mapping``'s ``ronan_devs`` branch,
        ``Lit4dVarNetTwoSolvers.base_step`` -- adapted here to one weight-tied
        solver called repeatedly rather than two distinct solver instances).
        This changes nothing about the forward *values* (detach is a no-op on
        values, only on the graph), so ``tbptt_n_blocks=1`` (the default)
        reproduces the pre-existing single-block behavior exactly.
        """
        N = self.N_outer if N_outer is None else N_outer
        obs_clean = torch.nan_to_num(batch.obs, nan=0.0)  # (B, T, D)
        obs_mask = batch.obs_mask.to(obs_clean.dtype).unsqueeze(-1)
        B, T, D = obs_clean.shape
        x = torch.zeros(B, T, D, device=obs_clean.device)  # x_0 = 0
        if self.update_input in _AUTOGRAD_MODES:
            x = x.detach().requires_grad_(True)
        grad_norm_cache = {}  # fresh per forward() call -- one unrolled solve
        denom = max(N - 1, 1)
        block_size = self.tbptt_block_size if (N_outer is None and self.tbptt_n_blocks > 1) else N
        block_states = []
        for k in range(N):
            tau_k = torch.full((B,), k / denom, device=x.device)
            # prior_tau_k=None (default): the prior operator gets no
            # iteration-conditioning (self.prior_unet built with
            # time_emb_dim=0) -- only the main solver UNet below is
            # conditioned on tau_k. prior_tau_conditioning=True (legacy
            # checkpoints only) instead feeds it the same tau_k.
            prior_tau_k = tau_k if self.prior_tau_conditioning else None
            gmod = checkpoint(
                _solver_iteration, self.unet, self.update_input, x, obs_clean, obs_mask,
                tau_k, prior_tau_k, self.prior_unet, self.R_var, self.prior_weight, 1.0,
                grad_norm_cache, self.grad_clip_range, use_reentrant=False,
            )
            x = torch.clamp(x - (1.0 / N) * gmod, -self.clip_range, self.clip_range)
            if self.update_input in _AUTOGRAD_MODES and not self.training:
                x = x.detach().requires_grad_(True)
            if (k + 1) % block_size == 0:
                block_states.append(x)
                if k + 1 < N:
                    x = x.detach()
                    if self.update_input in _AUTOGRAD_MODES:
                        x = x.requires_grad_(True)
        if not block_states:
            # N=0 (no iterations at all, e.g. a degenerate-N_outer test):
            # the loop never runs and never hits a block boundary -- the
            # sole "final" state is just the untouched x_0.
            block_states.append(x)
        return block_states

    def forward(self, batch, N_outer=None):
        return self._unrolled_blocks(batch, N_outer=N_outer)[-1]

    def compute_loss(self, batch):
        """**Training** (``self.training``, i.e. ``LitModel``'s
        ``training_step``): mean of ``F.mse_loss(block_state, states)`` over
        every truncated-BPTT block's end-of-block state (weight 1.0 total,
        evenly split) -- the deep-supervision signal that makes the mid-unroll
        blocks' weights get a gradient even though their own forward path is
        detached from later blocks. With the default ``tbptt_n_blocks=1``
        this is exactly ``F.mse_loss(x_final, states)``, unchanged.

        **Validation/eval** (``not self.training``, i.e. ``validation_step``
        and any other eval-mode call): ``F.mse_loss(x_final, states)`` only
        -- the actual final-iteration answer's quality, deliberately NOT
        averaged with the mid-unroll blocks' (necessarily worse,
        still-refining) intermediate estimates. This keeps ``val_loss`` (and
        therefore ``stage1_best.ckpt`` checkpoint selection) measuring what
        the model is actually deployed to produce, regardless of
        ``tbptt_n_blocks`` -- the training-time deep-supervision objective
        and the eval-time model-selection metric are deliberately different
        functions of the same unroll.

        Both cases add -- only when ``prior_unet`` exists (``_PRIOR_MODES``)
        and ``aux_var_cost_weight>0`` -- two auxiliary terms at
        ``aux_var_cost_weight`` each, evaluated at the *final* block only:
        ``var_cost(x_final, obs)`` and ``var_cost(states, obs)``, both using
        the same ``prior_weight*prior_cost + obs_cost`` formula as the per-
        iteration ``_build_update_input`` gradient (``tau=None``, matching
        the prior operator's own no-conditioning convention -- see
        ``_prior_ae``). Gives ``prior_weight`` (and ``prior_unet``) a direct,
        single-hop gradient path to the loss, instead of relying solely on
        the 10-deep chained double-backward through
        ``torch.autograd.grad(..., create_graph=True)`` at every unrolled
        iteration -- added because a trainable weight in ``var_cost`` was
        empirically unstable (stalls, and eventually diverges) under
        "grad+state" with only that indirect signal (see git history /
        session notes), while a fixed weight trains fine.

        ``_masked_obs_cost``/``_prior_cost`` are unnormalized *sums* (not
        means) over all ``B*T*D`` elements -- the right convention for the
        *inner* per-iteration variational cost (deliberately observation-
        count-independent, see ``_masked_obs_cost``'s docstring), but at
        realistic batch/window sizes that sum is ~4-5 orders of magnitude
        larger than the MSE term above (empirically: MSE~1.0 vs. raw
        var_cost~1e4-1e5 at B=32,T=300,D=24). Divide by ``B*T*D`` here --
        for this *outer* auxiliary term only -- so ``aux_var_cost_weight``
        actually controls the intended balance against the MSE term instead
        of being swamped by a convention mismatch.
        """
        block_states = self._unrolled_blocks(batch)
        x_final = block_states[-1]
        if self.training:
            loss = sum(F.mse_loss(s, batch.states) for s in block_states) / len(block_states)
        else:
            loss = F.mse_loss(x_final, batch.states)
        if self.prior_unet is not None and self.aux_var_cost_weight > 0:
            obs_clean = torch.nan_to_num(batch.obs, nan=0.0)
            obs_mask = batch.obs_mask.to(obs_clean.dtype).unsqueeze(-1)
            numel = obs_clean.numel()
            var_cost_pred = (self.prior_weight * _prior_cost(self.prior_unet, x_final)
                              + _masked_obs_cost(x_final, obs_clean, obs_mask, self.R_var)) / numel
            var_cost_true = (self.prior_weight * _prior_cost(self.prior_unet, batch.states)
                              + _masked_obs_cost(batch.states, obs_clean, obs_mask, self.R_var)) / numel
            loss = loss + self.aux_var_cost_weight * (var_cost_pred + var_cost_true)
        return loss

    def sample(self, batch, N_outer=None):
        return self.forward(batch, N_outer=N_outer)


class FourDVarNetPredictStateCFM(nn.Module):
    """V3 (``PredictStateCFM``) CFM parameterization -- predicts
    ``mu = E[x1|x_tau,y]`` at each outer flow-time ``tau``, trained via
    ``MSE(mu, x1)`` and sampled by forward ODE integration
    ``x += dt*(mu-x)/(1-tau)`` -- but ``mu`` is computed by ``FourDVarNetSolver``'s
    own weight-tied unrolled refinement (``K_inner`` steps, mode selected by
    ``update_input`` -- see ``FourDVarNetSolver``/``_build_update_input`` for
    the full taxonomy), started from the current ``x_tau``, instead of a
    single ``UNet1D`` forward pass as plain ``PredictStateCFM`` uses.

    Deliberately does NOT compose via a nested ``FourDVarNetSolver`` instance:
    that would produce checkpoint keys like ``model.solver.unet....``, breaking
    ``evaluation/neural_inference.py``'s checkpoint-introspection (hardcoded to
    the flat ``model.unet....``/``model.velocity_unet....`` names every other
    model in this codebase uses). Instead this class owns flat ``self.unet``/
    ``self.prior_unet`` submodules and re-implements ``FourDVarNetSolver``'s
    loop body and ``_build_update_input`` dispatch inline -- if that update
    rule changes, mirror the change here too.

    The inner ``K_inner`` refinement uses its own ``k/(K_inner-1)`` iteration-
    index embedding, independent of the outer CFM ``tau`` (a documented
    simplification, not an oversight) -- ``tau`` is accepted by ``forward``
    only for interface parity with ``VanillaCFM``/``PredictStateCFM``.

    Unlike ``FourDVarNetSolver`` (zero-initialized, deterministic), ``forward``
    here is called on an arbitrary ``x_t`` -- during sampling this can start
    far from the data manifold (fresh Gaussian noise at early outer ``tau``),
    and occasionally (~1 in a few thousand full ``ens30`` samples, empirically)
    the ``K_inner``-step unrolled refinement diverges within a handful of
    steps: an out-of-distribution ``x`` produces a large UNet update, which
    produces an even-more-out-of-distribution ``x`` on the next inner
    iteration. A single such outlier member is enough to blow up the
    ensemble-mean point estimate (though not the ensemble scoring rule (ES),
    which is comparatively robust to one bad member). Guarded the same way
    every other L96 state-space integrator in this codebase guards against
    unbounded divergence (``models/lorenz96_dynamics.py``,
    ``evaluation/baselines.py``): clamp ``x`` to ``[-clip_range, clip_range]``
    after each inner update. ``clip_range=50.0`` matches those call sites'
    default and is >5x the observed in-distribution state range (|x|<10),
    so this is inactive for every normal trajectory and only bounds the rare
    divergent one.
    """

    def __init__(self, state_dim=24, hidden_channels=None, time_emb_dim=64,
                 N_outer=10, K_inner=5, sigma_prior=0.5, dropout=0.1,
                 train_tau_0_only=False, update_input="obs+state",
                 clip_range=50.0, R_var=0.5, obs_weight=1.0,
                 min_obs_weight=1e-3, trainable_obs_weight=True,
                 grad_clip_range=None):
        super().__init__()
        _validate_update_input(update_input)
        self.update_input = update_input
        self.state_dim = state_dim
        self.N_outer = N_outer
        self.K_inner = K_inner
        self.sigma_prior = sigma_prior
        self.train_tau_0_only = train_tau_0_only
        self.clip_range = clip_range
        # See FourDVarNetSolver's identical field for the rationale: bounds
        # only the grad-only/grad+state autograd gradient, independent of
        # clip_range's state-branch hard clamp below.
        self.grad_clip_range = grad_clip_range if grad_clip_range is not None else clip_range
        self.R_var = R_var
        self.min_obs_weight = min_obs_weight
        self._obs_weight_raw = None
        self._obs_weight_fixed = obs_weight
        if update_input in _AUTOGRAD_MODES and trainable_obs_weight:
            self._obs_weight_raw = nn.Parameter(torch.tensor(
                _init_positive_weight_raw(obs_weight, min_obs_weight), dtype=torch.float32))
        in_state_dim = _UPDATE_INPUT_CHANNEL_MULTIPLIER[update_input] * state_dim
        self.unet = UNet1D(
            state_dim=in_state_dim,
            hidden_channels=hidden_channels,
            time_emb_dim=time_emb_dim,
            use_obs=False,
            use_energy=False,
            dropout=dropout,
            output_dim=state_dim,
        )
        self.prior_unet = None
        if update_input in _PRIOR_MODES:
            self.prior_unet = UNet1D(
                state_dim=state_dim,
                hidden_channels=hidden_channels,
                time_emb_dim=time_emb_dim,
                use_obs=False,
                use_energy=False,
                dropout=dropout,
                output_dim=state_dim,
            )
        self.interpolant = LinearInterpolant(nu=1.0)

    @property
    def obs_weight(self):
        """The obs_cost weight in var_cost = prior_cost + obs_weight*obs_cost.
        Trainable (via a raw**2 + min_obs_weight reparametrization guaranteeing
        obs_weight >= min_obs_weight always) for the modes that actually use
        it ("grad-only"/"grad+state"); a plain fixed float otherwise."""
        if self._obs_weight_raw is None:
            return self._obs_weight_fixed
        return _positive_weight_value(self._obs_weight_raw, self.min_obs_weight)

    def forward(self, x_t, batch, tau):
        obs_clean = torch.nan_to_num(batch.obs, nan=0.0)
        obs_mask = batch.obs_mask.to(obs_clean.dtype).unsqueeze(-1)
        x = x_t
        if self.update_input in _AUTOGRAD_MODES and not x.requires_grad:
            x = x.detach().requires_grad_(True)
        grad_norm_cache = {}  # fresh per forward() call -- one unrolled (inner) solve
        denom = max(self.K_inner - 1, 1)
        for k in range(self.K_inner):
            tau_k = torch.full((x.shape[0],), k / denom, device=x.device)
            gmod = checkpoint(
                _solver_iteration, self.unet, self.update_input, x, obs_clean, obs_mask,
                tau_k, tau_k, self.prior_unet, self.R_var, 1.0, self.obs_weight,
                grad_norm_cache, self.grad_clip_range, use_reentrant=False,
            )
            x = torch.clamp(x - (1.0 / self.K_inner) * gmod, -self.clip_range, self.clip_range)
            if self.update_input in _AUTOGRAD_MODES and not self.training:
                x = x.detach().requires_grad_(True)
        return x

    def compute_loss(self, batch):
        B = batch.states.shape[0]
        device = batch.states.device
        tau = torch.zeros(B, device=device) if self.train_tau_0_only else torch.rand(B, device=device)
        x0 = torch.randn_like(batch.states) * self.sigma_prior
        x_tau = self.interpolant.mix(x0, batch.states, tau)
        mu_pred = self.forward(x_tau, batch, tau)
        return F.mse_loss(mu_pred, batch.states)

    def sample(self, batch, N_outer=None, mean_estimate=None, tau0: float = 0.0):
        """Sample via forward ODE integration, ``N_outer`` steps.

        ``mean_estimate``/``tau0`` implement a "SDEdit"-style warm start
        (same mechanism and naming as ``evaluation/sda_sampler.py``'s
        ``sda_guided_sample``): instead of starting the trajectory from pure
        noise at tau=0, start from ``interpolant.mix(noise, mean_estimate,
        tau0)`` at ``tau0`` and only run the Euler loop from there to tau=1
        (fewer steps -- cheaper NFE too). ``tau0`` is snapped to the existing
        ``step/N_outer`` discretization (``step0 = round(tau0*N_outer)``) so
        the warm-started point lands on a training-time-valid interpolant
        point.

        A naive version of this warm start that instead set ``x_0 =
        mean_estimate + noise`` *at* tau=0 (running the full ``N_outer``-step
        trajectory, no steps skipped) was tried and empirically made things
        *worse* than no warm start at all (measured RMSE 0.897 vs. 0.555 for
        a single unwarm-started sample, on the canonical S0 test set): CFM
        training pairs tau=0 with near-zero-magnitude noise only (``x0 =
        randn*sigma_prior``), so injecting a real-state-scale ``mean_estimate``
        there is an out-of-training-distribution (tau, |x_tau|) combination
        that confuses rather than helps the first refinement step -- and that
        confusion compounds since no steps are skipped. Warm-starting at an
        intermediate ``tau0`` (this implementation) avoids this exactly the
        way the FDV1+SDA hybrid already does, since the interpolant's blend at
        ``tau0>0`` is consistent with what the network saw in training at
        that ``tau0``.

        ``mean_estimate=None`` (the default) reproduces the pre-existing
        behavior exactly (``step0=0``, ``x`` starts at pure noise) -- the key
        regression invariant, checked in ``tests/test_fourdvarnet.py``.
        """
        N = self.N_outer if N_outer is None else N_outer
        B, T, D = batch.obs.shape
        device = batch.obs.device
        noise = torch.randn(B, T, self.state_dim, device=device) * self.sigma_prior
        step0 = int(round(tau0 * N)) if mean_estimate is not None else 0
        if step0 > 0:
            x = self.interpolant.mix(noise, mean_estimate, torch.full((B,), step0 / N, device=device))
        else:
            x = noise
        dt = 1.0 / N
        for step in range(step0, N):
            tau_val = step / N
            tau = torch.full((B,), tau_val, device=device)
            mu = self.forward(x, batch, tau)
            x = x + dt * (mu - x) / (1 - tau_val)
        return x
