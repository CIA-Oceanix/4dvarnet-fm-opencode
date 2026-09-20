import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.interpolant import LinearInterpolant
from models.lorenz96_dynamics import Lorenz96Dynamics
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
try:
    from models.monai_unet_qg2d import MonaiUNet2DQGSolver
except ImportError:
    # Same optional-dependency guard as MonaiUNet1D above.
    class MonaiUNet2DQGSolver:
        pass

_VALID_UNET_BACKBONES = ("unet1d", "monai", "monai2d")


def _validate_unet_backbone(unet_backbone):
    if unet_backbone not in _VALID_UNET_BACKBONES:
        raise ValueError(
            f"Unknown unet_backbone={unet_backbone!r}; expected one of {_VALID_UNET_BACKBONES}"
        )


def _build_backbone_unet(unet_backbone, *, state_dim, hidden_channels, time_emb_dim,
                          dropout, output_dim, monai_norm_num_groups=32,
                          monai_num_res_blocks=2, qg_T=None, qg_ny=None, qg_nx=None,
                          monai_output_init_std=0.0):
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

    ``monai_num_res_blocks`` (default 2, ``MonaiUNet1D``'s own default,
    unchanged): together with ``hidden_channels`` this selects a capacity
    tier from the ladder in ``project_l96_monai_unet_complexity_tiers``
    memory -- ``hidden_channels=[64,128,256], monai_num_res_blocks=2`` is
    "M" (5,889,048 params, today's default for every FDV1/FDV2 config);
    ``hidden_channels=[32,64,128], monai_num_res_blocks=1`` is "S"
    (1,055,544 params); the same width with ``monai_num_res_blocks=2`` is
    "S+" (1,482,264 params) instead. Ignored for ``unet_backbone="unet1d"``
    (that backbone has no such knob).

    ``monai_output_init_std`` (default 0.0, no-op -- MonaiUNet1D's own
    default, MONAI's standard ``zero_module()`` init unchanged): overrides
    the final output conv's zero-init with ``N(0, monai_output_init_std)``
    instead. Only meaningful (and only ever passed nonzero by
    ``FourDVarNetSolver``) for a ``prior_unet`` built with
    ``prior_residual=True`` -- see ``MonaiUNet1D.__init__``'s docstring for
    why zero-init is a permanent dead end there specifically (not for the
    main solver ``self.unet``, which never sets this). Ignored for
    ``unet_backbone="unet1d"``.
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
    if unet_backbone == "monai2d":
        if MonaiUNet2DQGSolver.__module__ == __name__:
            raise ImportError(
                "unet_backbone='monai2d' requires the optional 'monai' package "
                "(not installed in this environment) -- see models/monai_unet_qg2d.py"
            )
        if qg_T is None or qg_ny is None or qg_nx is None:
            raise ValueError(
                "unet_backbone='monai2d' requires qg_T/qg_ny/qg_nx (QG's window "
                "day-count and grid shape) -- see FourDVarNetSolver's own "
                "qg_T/qg_ny/qg_nx constructor args")
        return MonaiUNet2DQGSolver(
            state_dim=state_dim, T=qg_T, ny=qg_ny, nx=qg_nx,
            hidden_channels=hidden_channels, output_dim=output_dim,
            dropout=dropout, norm_num_groups=monai_norm_num_groups,
            num_res_blocks=monai_num_res_blocks,
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
        num_res_blocks=monai_num_res_blocks,
        output_init_std=monai_output_init_std,
    )

# Recognized update_input tokens (mirrors the config-string taxonomy explored on
# CIA-Oceanix/4dvarnet-global-mapping's ronan_devs branch, contrib/4dvarnet_latent/
# models.py::GradSolver_withStep). "obs+state"/"obs-only" are gradient-free
# (the per-iteration UNet is fed the raw state/obs, no cost function at all).
# "grad-only"/"grad+state"/"subgrad+state"/"gradsplit+state" ("FDV2") are
# gradient-conditioned: a variational cost prior_cost(state) +
# obs_weight*obs_cost(state, obs) is built each iteration, and one of three
# things is fed to the update UNet: its real autograd gradient as ONE
# combined tensor ("grad-only"/"grad+state", ported from ocean4dvarnet's
# GradSolver -- see _build_update_input); a cheap two-residual proxy that
# never calls autograd ("subgrad+state", ported from ronan_devs'
# GradSolver_withStep); or the two terms' real autograd gradients kept
# SEPARATE ("gradsplit+state" -- same two-channel input shape as
# "subgrad+state", but each channel is a true gradient via its own
# torch.autograd.grad call instead of a cheap proxy).
_IMPLEMENTED_UPDATE_INPUTS = ("obs+state", "obs-only", "grad-only", "grad+state",
                              "subgrad+state", "gradsplit+state", "subgrad+state+xtau",
                              "subgrad+state+trueprior", "subgrad+trueprior")

# Modes needing a real torch.autograd.grad call each iteration.
_AUTOGRAD_MODES = ("grad-only", "grad+state", "gradsplit+state")
# Modes needing the trainable prior operator (prior_unet). "subgrad+state+trueprior"
# deliberately excluded -- its "prior" is the KNOWN true ODE (zero trainable
# parameters), not a learned network; see _true_ode_prior_residual.
_PRIOR_MODES = ("grad-only", "grad+state", "subgrad+state", "gradsplit+state",
                "subgrad+state+xtau")
# Number of state_dim-sized channel blocks the main update UNet's input has,
# per mode -- drives in_state_dim at construction time.
_UPDATE_INPUT_CHANNEL_MULTIPLIER = {
    "obs-only": 1,
    "obs+state": 2,
    "grad-only": 1,
    "grad+state": 2,
    "subgrad+state": 3,
    "gradsplit+state": 3,
    "subgrad+state+xtau": 4,
    "subgrad+state+trueprior": 3,
    "subgrad+trueprior": 2,
}
# "subgrad+state+xtau" needs an outer flow-time conditioning value (x_tau,
# beta_tau) that only exists for FourDVarNetPredictStateCFM's CFM
# formulation (see its own docstring) -- structurally undefined for
# FourDVarNetSolver, which has no outer flow-time at all.
_CFM_ONLY_UPDATE_INPUTS = ("subgrad+state+xtau",)
# "subgrad+state+trueprior"/"subgrad+trueprior" need the FULL physical state
# (see _true_ode_prior_residual's docstring -- the true dynamics is
# undefined on a partially-observed subspace), i.e. state_dim == the
# model's own obs_var_indices-implied full dimension, not the usual 24D
# observed subspace every other mode operates in. Enforced in
# FourDVarNetSolver's own __init__ (obs_var_indices/true_dynamics_dt
# required together with either mode). "subgrad+trueprior" is
# "subgrad+state+trueprior" minus the raw-state channel x -- an ablation
# testing whether the two residuals (g_obs, g_prior) alone are enough,
# mirroring the existing "grad-only" (no state) vs "grad+state" (with
# state) pair for the real-autograd-gradient family.
_FULL_STATE_UPDATE_INPUTS = ("subgrad+state+trueprior", "subgrad+trueprior")


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


def _prior_ae(prior_unet, state, tau=None, residual=False):
    """The trainable prior operator Phi(state), applied channel-last -> UNet1D's
    channel-first convention and back. ``tau=None`` (the default) applies no
    time-conditioning at all (``UNet1D.forward`` skips its time-embedding
    branch entirely when ``tau is None``) -- used by ``FourDVarNetSolver``,
    which deliberately keeps iteration-conditioning in the main solver UNet
    only: the prior is meant as a fixed background/regularization operator,
    not one that behaves differently per unrolled iteration. Passing a real
    ``tau`` (``FourDVarNetPredictStateCFM``'s own usage, unchanged) applies
    the same per-iteration tau embedding as the main update UNet.

    ``residual`` (default False, backward-compatible): when True, returns
    ``state + prior_unet(state, tau)`` -- an explicit architectural identity
    anchor around the whole backbone -- instead of the bare network output.
    Diagnostic knob added after a Jacobian decomposition of "gradsplit+state"
    ``prior_cost``'s true gradient (2026-09-14) found the Jacobian term
    ``-2*J^T@r`` dominates the residual term ``2*r`` by 11-53x and is nearly
    orthogonal to it (cos~0.04-0.13) on a plateaued MonaiUNet1D checkpoint --
    i.e. Phi's Jacobian, once training moves the backbone's zero-initialized
    output layers away from zero, has nothing architecturally anchoring it
    near identity. ``residual=True`` bakes that anchor in explicitly
    (``d(state + f(state))/d(state) = I + df/d(state)``, permanently, not
    just at init), matching what a residual/denoising-style prior wrapper
    (``Phi(x) = x - CNN(x)`` or similar) would give structurally for free."""
    raw = prior_unet(state.transpose(1, 2), tau=tau).transpose(1, 2)
    return state + raw if residual else raw


def _prior_cost(prior_unet, state, tau=None, residual=False):
    """sum((state - Phi(state))^2) -- MSE(state, Phi(state)) with
    reduction="sum", matching _masked_obs_cost's sum-based (not
    count-normalized) convention, so the two terms combine consistently in
    var_cost. Phi = prior_unet (ocean4dvarnet's BilinAEPriorCost/ronan_devs'
    GenericAEPriorCost formula). See ``_prior_ae`` re: ``tau=None``/``residual``."""
    return F.mse_loss(state, _prior_ae(prior_unet, state, tau, residual=residual), reduction="sum")


def _true_ode_prior_residual(x, forcing, params, dynamics):
    """"subgrad+state+trueprior"'s g_prior: a time-shifted dynamical-
    consistency residual against the KNOWN, true L96 ODE (not a learned
    Phi) -- ``g_prior[:, t, :] = x[:, t, :] - dynamics.step(x[:, t-1, :],
    forcing[:, t-1])`` for ``t=1..T-1`` (does the trajectory's actual step
    from ``t-1`` to ``t`` match what the true dynamics predicts from
    ``x[t-1]``), zero-padded at ``t=0`` (no valid predecessor within the
    window). This is the real weak-constraint-4DVar-style model-error term,
    deliberately NOT the pointwise ``x - Phi(x)`` used elsewhere in this
    module: unlike a LEARNED, autoencoder-like ``prior_unet`` (trained so
    ``Phi(x) ~= x`` for real trajectories), the true ODE is a genuine
    evolution operator -- ``dynamics.step(x) != x`` almost everywhere even
    for a perfectly correct trajectory, so a pointwise ``x - dynamics.step
    (x)`` would mostly just feed the local tendency/velocity at x, not a
    "is x correct" signal the way subgrad+state's own residual is.

    ``x``: (B, T, D) -- the FULL physical state (D = NO + NO*J, e.g. 40 for
    NO=8,J=4 -- NOT the partially-observed subspace every other update_input
    mode operates in; the true dynamics needs every fast-Y component to be
    well-defined at all, see FourDVarNetSolver's own docstring). ``forcing``:
    (B, T) -- one scalar per (batch, timestep), matching x's own T
    resolution (data/lorenz96.py stores forcing at the same per-dt
    resolution as the state trajectory, confirmed against
    Lorenz96Dynamics.generate_full_trajectory: no downsampling, x[:,t,:] ->
    x[:,t+1,:] is exactly one dynamics.step() call at the same dt).
    ``params``: (B, 8) = [F, c1, hx, eps, w1, w2, w3, w4] (L96_JOINT_PARAM_NAMES
    order, matching evaluation/neural_inference.py's own convention).
    Never detached -- Lorenz96Dynamics.step is plain differentiable tensor
    ops (RK4 of elementwise arithmetic, no deep network), cheap to
    backprop through unlike a learned prior_unet, so there's no
    computational reason to give up the real local sensitivity information
    (see the 2026-09-15 detach_var_cost_grad discussion for the analogous
    but much more expensive learned-network case)."""
    x_prev = x[:, :-1, :]
    forcing_prev = forcing[:, :-1]
    phi_pred = dynamics.step(
        x_prev, forcing_prev,
        F=params[:, 0], c1=params[:, 1], hx=params[:, 2], eps=params[:, 3],
        fast_weights=params[:, 4:8],
    )
    resid = x[:, 1:, :] - phi_pred
    pad = torch.zeros_like(x[:, :1, :])
    return torch.cat([pad, resid], dim=1)


def _var_cost_training_loss(block_state, obs_clean, obs_mask, forcing, params, dynamics, R_var, Q_var):
    """Weak-constraint-4DVar-style self-supervised training loss:
    ``obs_cost/R_var + prior_cost/Q_var``, evaluated on the SOLVER'S OWN
    output -- no direct ground-truth supervision at all (only indirectly,
    via ``obs``, as always). ``loss_type="var_cost"``'s alternative to the
    default MSE-against-truth objective (see ``FourDVarNetSolver.compute_loss``).

    Mirrors ``evaluation/baselines.py``'s ``Weak4DVar.assimilate_batch``
    exactly: its ``J_o = sum(((H(x)-obs)*mask)^2) / R_var`` is this
    function's ``obs_cost`` (via ``_masked_obs_cost``, same sum/R_var
    convention); its ``J_q = sum(q^2) / Q_var`` -- where ``q[t] =
    x[t] - dynamics.step(x[t-1])``, the per-step weak-constraint model-error
    control variable -- is EXACTLY this function's ``prior_cost`` on
    ``_true_ode_prior_residual``'s own ``g_prior`` (the identical residual,
    just computed on the whole window at once rather than as an optimized
    control variable). ``R_var``/``Q_var`` default to Weak4DVar's own class
    defaults (0.5/0.05) -- reusing an already-established, physically-
    motivated weighting (observation-noise variance vs. model-error
    variance) rather than introducing a fresh unjustified ratio: Q_var
    being 10x tighter than R_var means the classical weak-4DVar formulation
    trusts the known dynamics an order of magnitude more than the raw
    observations, per unit squared error.

    Normalized by ``numel`` (matching ``compute_loss``'s existing
    ``aux_var_cost_weight`` term's own convention) so the loss magnitude
    stays comparable to the default MSE objective's scale (~1.0), instead
    of the raw unnormalized sum's ~1e4-1e5 -- avoids needing a fresh
    lr/gradient_clip_val retune just from switching ``loss_type``.
    """
    numel = block_state.numel()
    obs_cost = _masked_obs_cost(block_state, obs_clean, obs_mask, R_var) / numel
    g_prior = _true_ode_prior_residual(block_state, forcing, params, dynamics)
    prior_cost = g_prior.pow(2).sum() / Q_var / numel
    return obs_cost + prior_cost


def _embed_obs_to_full_state(obs, obs_mask, obs_var_indices, full_dim):
    """Scatters a (B,T,len(obs_var_indices))-shaped observed-subspace obs/
    mask into a (B,T,full_dim)-shaped tensor at the given channel indices
    (NaN/0 elsewhere) -- needed when the solver's own state_dim is the FULL
    physical state (see _true_ode_prior_residual) while obs/obs_mask, as
    always, only ever cover the actually-observed subspace.

    Unlike every other update_input mode's obs_mask (purely TEMPORAL,
    shape (B,T,1), broadcasting uniformly across every state_dim channel --
    correct there because every observed channel is sampled together at an
    observation time or not at all), the returned mask here is genuinely
    per-CHANNEL as well: the ``full_dim - len(obs_var_indices)`` unobserved
    physical channels are masked out at EVERY timestep, permanently, AND-ed
    with the existing temporal pattern for the channels that ARE observed.

    ``obs``: (B,T,D_obs) raw (NaN-at-unobserved-TIMES) obs tensor.
    ``obs_mask``: (B,T,1) boolean/float temporal mask (pre-unsqueeze, same
    convention as every other call site in this module).
    ``obs_var_indices``: sequence of ``D_obs`` distinct ints in
    ``[0, full_dim)`` -- the physical channels observation-covers.
    Returns ``(obs_full, obs_mask_full)``, both (B,T,full_dim)."""
    B, T, _ = obs.shape
    idx = torch.as_tensor(obs_var_indices, device=obs.device, dtype=torch.long)
    obs_full = obs.new_full((B, T, full_dim), float("nan"))
    obs_full[..., idx] = obs
    channel_mask = torch.zeros(full_dim, device=obs.device, dtype=obs_mask.dtype)
    channel_mask[idx] = 1
    obs_mask_full = obs_mask * channel_mask.view(1, 1, full_dim)
    return obs_full, obs_mask_full


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
                         grad_norm_cache=None, clip_range=50.0, gradsplit_prior_scale=1.0,
                         prior_residual=False, detach_var_cost_grad=False,
                         x_tau=None, beta_tau=None,
                         prior_ode_forcing=None, prior_ode_params=None, true_dynamics=None):
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
    This port keeps the four gradient-conditioned modes fully separate (no
    such cross-talk) and does not add any extra outer-loop term at all --
    only what feeds the main UNet changes per mode; the existing
    ``x - (1/N)*gmod`` update rule (shared with "obs+state"/"obs-only") is
    unchanged.

    "subgrad+state+xtau" (``FourDVarNetPredictStateCFM`` only -- see
    ``_CFM_ONLY_UPDATE_INPUTS``) extends "subgrad+state" with a fourth
    channel for the CFM outer flow-time's own conditioning value ``x_tau``,
    per ``reports/notes_4dvarnet_fm.pdf``'s "energy-informed parameterization"
    (Sec. 3.3): the paper's third residual term is
    ``(alpha_tau^2*sigma_0^2)^-1 * (beta_tau*x^(k) - beta_tau^2*x_tau)``,
    which is ill-conditioned as ``alpha_tau -> 0`` (``tau -> 1``, dividing by
    a vanishing squared term). This mode instead feeds the un-prefactored
    core residual ``g_flow = x - beta_tau*x_tau`` (dropping the
    ``(alpha_tau^2*sigma_0^2)^-1`` scaling entirely) -- bounded, well-
    conditioned everywhere on ``tau in [0,1]`` since ``x``/``x_tau`` are both
    state-scale and ``beta_tau in [0,1]``. ``x_tau`` is the CFM's outer
    conditioning value (``FourDVarNetPredictStateCFM.forward``'s own ``x_t``
    argument, held fixed across the whole ``K_inner`` unroll -- distinct
    from ``x``, which is this same quantity's evolving *estimate* at inner
    iteration ``k``); ``beta_tau = self.interpolant.beta(tau)`` (the OUTER
    CFM flow-time's own beta, not the inner-iteration ``tau``/``prior_tau_k``
    this function's own ``tau`` parameter already carries for
    ``prior_unet``'s conditioning -- two different time variables).

    "gradsplit+state" is the real-autograd counterpart to "subgrad+state":
    same two-residual-plus-state input shape, but ``g_obs``/``g_prior`` are
    each a true ``torch.autograd.grad`` of their own cost term
    (``obs_weight*obs_cost(x, obs)`` / ``prior_weight*prior_cost(x)``) taken
    SEPARATELY, instead of "grad-only"/"grad+state"'s single combined
    gradient of the summed ``var_cost``, and instead of "subgrad+state"'s
    cheap proxy residuals. ``g_prior`` is normalized/soft-clipped exactly
    like "grad-only"/"grad+state"'s combined ``grad`` tensor -- it is a real
    gradient of an unbounded cost term (``prior_cost``, dense: nonzero
    everywhere), the same reason that tensor needs it. ``g_obs`` is
    deliberately NOT normalized, even though it IS a real gradient here (not
    a proxy) -- ``obs_cost``'s gradient w.r.t. ``x`` is
    ``2*mask*(x-obs)/R_var``, architecturally masked to exactly zero outside
    observation times exactly like "subgrad+state"'s own ``g_obs`` proxy, so
    it inherits that mode's normalization-dilution vulnerability (see the
    2026-09-11 fix) identically -- being a true gradient rather than a proxy
    doesn't change its sparsity, and the same global whole-tensor RMS norm
    would dilute/inflate it exactly the same way. Whether a channel needs
    normalization is decided by its sparsity/boundedness, not by whether it
    came from ``torch.autograd.grad`` or a hand-written proxy formula.

    ``gradsplit_prior_scale`` (default 1.0, no-op): multiplies ``g_prior``
    AFTER normalization/soft-clipping, for "gradsplit+state" only. Added as
    a diagnostic knob -- forcing it near 0 (e.g. 1e-4) makes the fed tensor
    ``cat([g_obs, ~0, x])``, functionally close to "obs+state"'s own
    ``cat([x, obs_clean])`` (``g_obs`` is proportional to ``x-obs`` at
    observed times, the same information "obs+state" gets directly), to
    check whether a persistent training plateau traces back to the
    ``g_prior`` channel's contribution specifically, by comparing against
    a config where it's been silenced almost entirely.

    ``prior_residual`` (default False, backward-compatible): forwarded
    unchanged to every ``_prior_ae``/``_prior_cost`` call below -- see
    ``_prior_ae`` for what it does and why. Affects "subgrad+state"'s
    ``g_prior`` proxy and "gradsplit+state"/"grad-only"/"grad+state"'s real
    ``torch.autograd.grad``-computed ``g_prior``/``grad`` alike, all of which
    differentiate through ``Phi`` w.r.t. ``x`` (directly or via
    ``_prior_cost``) and are therefore all sensitive to whether ``Phi``'s
    Jacobian carries an explicit identity anchor.

    ``detach_var_cost_grad`` (default False, backward-compatible;
    "grad-only"/"grad+state" only): passed straight through as
    ``create_graph=not detach_var_cost_grad`` to the combined
    ``torch.autograd.grad(var_cost, x, ...)`` call below. Does NOT change
    what gets fed to the main solver UNet at all -- ``grad`` is still the
    real ``2*(x-Phi(x)) - 2*J^T@(x-Phi(x))`` (Jacobian term included, full
    backward through ``prior_unet``, same cost as always for THIS forward
    pass). What changes is purely downstream: with ``create_graph=False``,
    this tensor carries no graph, so the OUTER supervised loss's own
    ``.backward()`` can no longer differentiate through it at all -- neither
    into ``prior_unet``'s weights (and ``prior_weight``, used inside the
    same ``var_cost`` expression) NOR into ``x``'s own upstream dependency
    on earlier unrolled iterations via this channel (the plain ``x``
    channel, concatenated alongside ``grad``, still carries a normal graph
    and is unaffected). ``prior_unet``/``prior_weight`` then train ONLY via
    the aux ``prior_cost`` loss (``aux_var_cost_weight>0``), not via the
    per-iteration double-backward. Added specifically to let a genuinely
    differentiated prior's TRUE gradient still reach the solver every
    iteration while decoupling "does the solver benefit from consuming that
    true gradient" from "does jointly training the prior through the
    per-iteration double-backward help" -- two questions this mode
    otherwise conflates. Distinct from (and strictly less destructive than)
    detaching ``Phi(x)`` inside ``_prior_cost`` itself, which would instead
    discard the Jacobian term entirely and collapse the fed signal to
    "subgrad+state"'s own proxy.

    CAVEAT: ``prior_weight`` (when ``trainable_prior_weight=True``) has NO
    OTHER gradient source than this same per-iteration pathway --
    ``compute_loss``'s aux ``prior_cost`` term deliberately never
    references ``self.prior_weight`` (see its docstring: the 2026-09-11
    ``prior_weight``-collapse-to-zero bugfix). So
    ``detach_var_cost_grad=True`` with ``trainable_prior_weight=True``
    makes ``prior_weight`` permanently untrainable -- stuck at its init
    value, receiving literally no gradient from any loss term. Set
    ``trainable_prior_weight=False`` alongside ``detach_var_cost_grad=True``
    to avoid carrying a dead parameter.
    """
    if update_input == "obs-only":
        return obs_clean
    if update_input == "obs+state":
        return torch.cat([x, obs_clean], dim=-1)
    if update_input == "subgrad+state":
        g_obs = (obs_clean - x) * obs_mask
        g_prior = x - _prior_ae(prior_unet, x, tau, residual=prior_residual)
        return torch.cat([g_obs, g_prior, x], dim=-1)
    if update_input == "subgrad+state+xtau":
        assert x_tau is not None and beta_tau is not None, (
            "subgrad+state+xtau requires x_tau/beta_tau (FourDVarNetPredictStateCFM "
            "only -- see _CFM_ONLY_UPDATE_INPUTS)"
        )
        g_obs = (obs_clean - x) * obs_mask
        g_prior = x - _prior_ae(prior_unet, x, tau, residual=prior_residual)
        g_flow = x - beta_tau * x_tau
        return torch.cat([g_obs, g_prior, g_flow, x], dim=-1)
    if update_input in ("subgrad+state+trueprior", "subgrad+trueprior"):
        assert prior_ode_forcing is not None and prior_ode_params is not None and true_dynamics is not None, (
            f"{update_input} requires prior_ode_forcing/prior_ode_params/true_dynamics "
            "(FourDVarNetSolver constructed with obs_var_indices+true_dynamics_dt -- see "
            "_FULL_STATE_UPDATE_INPUTS)"
        )
        g_obs = (obs_clean - x) * obs_mask
        g_prior = _true_ode_prior_residual(x, prior_ode_forcing, prior_ode_params, true_dynamics)
        if update_input == "subgrad+trueprior":
            return torch.cat([g_obs, g_prior], dim=-1)
        return torch.cat([g_obs, g_prior, x], dim=-1)
    if update_input == "gradsplit+state":
        with torch.enable_grad():
            prior_cost_val = prior_weight * _prior_cost(prior_unet, x, tau, residual=prior_residual)
            g_prior = torch.autograd.grad(prior_cost_val, x, create_graph=True)[0]
            obs_cost_val = obs_weight * _masked_obs_cost(x, obs_clean, obs_mask, R_var)
            g_obs = torch.autograd.grad(obs_cost_val, x, create_graph=True)[0]
        g_prior = gradsplit_prior_scale * _normalize_channels(
            g_prior, cache=grad_norm_cache, key="g_prior_gradsplit", clip_range=clip_range)
        return torch.cat([g_obs, g_prior, x], dim=-1)
    # grad-only / grad+state
    with torch.enable_grad():
        var_cost = prior_weight * _prior_cost(prior_unet, x, tau, residual=prior_residual) \
            + obs_weight * _masked_obs_cost(x, obs_clean, obs_mask, R_var)
        raw_grad = torch.autograd.grad(var_cost, x, create_graph=not detach_var_cost_grad)[0]
        grad = _normalize_channels(raw_grad, cache=grad_norm_cache, key="grad", clip_range=clip_range)
    if update_input == "grad-only":
        return grad
    return torch.cat([grad, x], dim=-1)


def _solver_iteration(unet, update_input, x, obs_clean, obs_mask, tau_k, prior_tau_k,
                       prior_unet, R_var, prior_weight, obs_weight, grad_norm_cache,
                       clip_range=50.0, gradsplit_prior_scale=1.0, prior_residual=False,
                       detach_var_cost_grad=False, x_tau=None, beta_tau=None,
                       prior_ode_forcing=None, prior_ode_params=None, true_dynamics=None):
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
    nested/higher-order autograd correctly. (``detach_var_cost_grad=True``
    makes that specific call ``create_graph=False`` instead -- still safe
    under ``use_reentrant=False`` either way, just no longer exercising the
    nested-autograd path for this term.)

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

    ``x_tau``/``beta_tau`` (both default None): forwarded unchanged to
    ``_build_update_input``, only actually used by "subgrad+state+xtau"
    (``FourDVarNetPredictStateCFM`` only -- see ``_CFM_ONLY_UPDATE_INPUTS``).
    ``FourDVarNetSolver`` never passes these (defaults apply).

    ``prior_ode_forcing``/``prior_ode_params``/``true_dynamics`` (all
    default None): forwarded unchanged to ``_build_update_input``, only
    actually used by the ``_FULL_STATE_UPDATE_INPUTS`` modes
    ("subgrad+state+trueprior"/"subgrad+trueprior") -- constant across the whole unroll
    (unlike ``x``), so the caller computes them once per ``forward()`` call.
    """
    inp = _build_update_input(update_input, x, obs_clean, obs_mask, prior_tau_k,
                               prior_unet=prior_unet, R_var=R_var,
                               obs_weight=obs_weight, prior_weight=prior_weight,
                               grad_norm_cache=grad_norm_cache, clip_range=clip_range,
                               gradsplit_prior_scale=gradsplit_prior_scale,
                               prior_residual=prior_residual,
                               detach_var_cost_grad=detach_var_cost_grad,
                               x_tau=x_tau, beta_tau=beta_tau,
                               prior_ode_forcing=prior_ode_forcing,
                               prior_ode_params=prior_ode_params,
                               true_dynamics=true_dynamics).transpose(1, 2)
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
    - ``"gradsplit+state"``: like "subgrad+state"'s two-residual-plus-state
      shape, but each residual is a real, SEPARATE autograd gradient
      (``torch.autograd.grad`` of ``obs_cost``/``prior_cost`` individually,
      not the combined ``var_cost`` "grad-only"/"grad+state" differentiate).

    ``unet_backbone`` ("unet1d" default, "monai", or "monai2d") selects the
    nn.Module class backing ``self.unet``/``self.prior_unet`` --
    ``models.unet.UNet1D``, ``models.monai_unet_adapter.MonaiUNet1D``
    (MONAI's DiffusionModelUNet treating the T axis as a downsampled 1D
    sequence, already validated as a drop-in backbone for DirectUNet, see
    reports/l96/outputs/l96_normalization_ablation.md), or
    ``models.monai_unet_qg2d.MonaiUNet2DQGSolver`` (QG-only: true 2D
    circular convs over ``(ny, nx)``, with the T axis merged into the
    channel dimension instead -- requires ``qg_T``/``qg_ny``/``qg_nx``).
    All three are built with ``use_obs=False`` and have the identical
    ``forward(x, tau=...)`` call signature, so this is a pure backbone swap
    -- no other FDV logic changes. See ``_build_backbone_unet`` for the one
    known semantic gap (no true tau-conditioning omission for the
    Monai-backed ``prior_unet``, either 1D or 2D flavor).

    The gradient-conditioned modes need a trainable prior operator
    (``self.prior_unet``, a second ``UNet1D`` sharing the main UNet's
    ``hidden_channels``/``time_emb_dim``/``dropout`` -- mirrors
    ``TweedieCFM``'s ``mean_estimator``/``velocity_unet`` hyperparameter
    sharing), constructed only when ``update_input`` needs it so
    ``"obs+state"``/``"obs-only"`` checkpoints stay exactly as before (no
    dead weights). ``dropout`` sharing is the default, not forced: pass
    ``prior_dropout`` to decouple it (e.g. dropout for the main solver unet
    only, ``prior_dropout=0.0`` -- see ``self.prior_dropout``'s docstring).

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
                 monai_num_res_blocks=2,
                 prior_hidden_channels=None,
                 tbptt_n_blocks=1,
                 tbptt_block_size=None,
                 grad_clip_range=None,
                 init_state_var=0.0,
                 qg_T=None, qg_ny=None, qg_nx=None,
                 gradsplit_prior_scale=1.0,
                 prior_residual=False,
                 prior_dropout=None,
                 prior_output_init_std=0.0,
                 detach_var_cost_grad=False,
                 obs_var_indices=None,
                 true_dynamics_dt=None,
                 true_dynamics_NO=8,
                 true_dynamics_J=4,
                 true_dynamics_h=1.0,
                 true_dynamics_coupling_exponent=1.6,
                 loss_type="mse",
                 var_cost_Q_var=0.05):
        super().__init__()
        _validate_update_input(update_input)
        if loss_type not in ("mse", "var_cost"):
            raise ValueError(f"Unknown loss_type={loss_type!r}; expected 'mse' or 'var_cost'")
        if loss_type == "var_cost" and update_input not in _FULL_STATE_UPDATE_INPUTS:
            raise ValueError(
                f"loss_type='var_cost' needs the true ODE prior "
                f"(_true_ode_prior_residual/self.true_dynamics), only available "
                f"for the _FULL_STATE_UPDATE_INPUTS modes -- not update_input={update_input!r}."
            )
        if update_input in _CFM_ONLY_UPDATE_INPUTS:
            raise ValueError(
                f"update_input={update_input!r} needs an outer flow-time "
                "conditioning value (x_tau/beta_tau) that only exists for "
                "FourDVarNetPredictStateCFM's CFM formulation -- structurally "
                "undefined for FourDVarNetSolver (see _CFM_ONLY_UPDATE_INPUTS)."
            )
        if update_input in _FULL_STATE_UPDATE_INPUTS:
            if obs_var_indices is None or true_dynamics_dt is None:
                raise ValueError(
                    f"update_input={update_input!r} needs the FULL physical "
                    "L96 state (the true dynamics is undefined on a partially-"
                    "observed subspace) -- obs_var_indices (which of "
                    "state_dim's channels observation actually covers) and "
                    "true_dynamics_dt (matching data.dt) must both be given "
                    "(see _FULL_STATE_UPDATE_INPUTS)."
                )
        elif obs_var_indices is not None:
            raise ValueError(
                f"obs_var_indices is only meaningful for "
                f"_FULL_STATE_UPDATE_INPUTS modes, not update_input={update_input!r} "
                "-- every other mode's obs already lives in the same state_dim "
                "space as x (no embedding needed)."
            )
        _validate_unet_backbone(unet_backbone)
        if unet_backbone in ("monai", "monai2d") and prior_tau_conditioning:
            raise ValueError(
                "prior_tau_conditioning=True exists only to reproduce legacy "
                "UNet1D checkpoints trained with a tau-conditioned prior_unet -- "
                "no such MonaiUNet1D/MonaiUNet2DQGSolver checkpoint exists, so "
                "this combination is not supported (see _build_backbone_unet)."
            )
        if unet_backbone == "monai2d" and (qg_ny is None or qg_nx is None or qg_T is None):
            raise ValueError(
                "unet_backbone='monai2d' requires qg_T/qg_ny/qg_nx (QG's window "
                "day-count and grid shape) to be given.")
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
        # loss_type="var_cost" (default "mse", backward-compatible): compute_loss
        # trains on a weak-constraint-4DVar-style variational cost
        # (obs_cost/R_var + prior_cost/var_cost_Q_var, evaluated on the solver's
        # OWN output) instead of MSE against the ground truth -- see
        # compute_loss's docstring. var_cost_Q_var (0.05 default, matching
        # evaluation/baselines.py's Weak4DVar class default, itself matching
        # this project's data.R_var=0.5/data.B_var=2.0 convention) is the
        # model-error-term variance normalizer -- self.R_var (already 0.5 by
        # every _FULL_STATE_UPDATE_INPUTS config in this codebase) is reused
        # for the obs term, the SAME quantity Weak4DVar's own J_o/R_var uses.
        self.loss_type = loss_type
        self.var_cost_Q_var = var_cost_Q_var
        # Set only inside compute_loss, during training, under
        # loss_type="var_cost" -- see that method's docstring. Initialized
        # to None here so LitModel.training_step's getattr check is always
        # well-defined, even before the first training step.
        self._last_train_mse_proxy = None
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
        # x_0 = randn * sqrt(init_state_var) instead of the default all-zeros
        # start (init_state_var=0.0, backward-compatible). VARIANCE, not std
        # -- e.g. init_state_var=0.1 means x_0 ~ N(0, 0.1), std ~ 0.316 in
        # this normalized state space. Sampled fresh every forward() call
        # (train and eval alike), independent of update_input -- this is a
        # property of the unroll's starting point, ported from nowhere in
        # particular, added specifically to test whether a nonzero-variance
        # random init changes FDV1's ("obs+state") own healthy fast-Y
        # reconstruction at all, as a sanity/robustness check.
        self.init_state_var = init_state_var
        # Diagnostic knob, "gradsplit+state" only (see _build_update_input):
        # multiplies g_prior AFTER normalization/soft-clipping. Default 1.0
        # is a no-op; forcing it near 0 (e.g. 1e-4) makes the fed tensor
        # cat([g_obs, ~0, x]), functionally close to "obs+state"'s own
        # cat([x, obs_clean]) -- used to check whether a persistent training
        # plateau traces back to g_prior's contribution specifically.
        self.gradsplit_prior_scale = gradsplit_prior_scale
        # Diagnostic knob (default False, backward-compatible): forwarded to
        # every _prior_ae/_prior_cost call this solver makes (per-iteration
        # update-input construction AND the aux prior-consistency loss term
        # in compute_loss below) -- see _prior_ae's docstring for the
        # Jacobian-decomposition finding that motivated it.
        self.prior_residual = prior_residual
        # None (default): prior_unet uses the same `dropout` as the main
        # solver unet -- today's behavior, backward-compatible. Set to
        # decouple the two, e.g. to add dropout as a regularizer for the
        # main solver unet only while keeping prior_unet's dropout unchanged
        # (or 0.0) -- relevant for grad-only/grad+state/gradsplit+state,
        # whose prior_unet forward is the one repeatedly re-run inside
        # torch.autograd.grad(..., create_graph=True): any dropout there
        # injects a fresh random mask into that higher-order (double-
        # backward) computation at every unrolled iteration, on top of
        # whatever noise the double-backward itself already contributes --
        # unlike the main solver unet, whose own forward is always a single,
        # ordinary (first-order) backward.
        self.prior_dropout = dropout if prior_dropout is None else prior_dropout
        # 0.0 (default, no-op): overrides prior_unet's (monai backbone only)
        # final output conv's zero_module init with N(0, prior_output_init_std)
        # instead. See MonaiUNet1D.__init__'s docstring -- only meaningful
        # (and only intended to be set) alongside prior_residual=True, where
        # exact zero-init is a provable permanent dead end for this layer.
        self.prior_output_init_std = prior_output_init_std
        # False (default, backward-compatible; "grad-only"/"grad+state"
        # only): create_graph=not detach_var_cost_grad for the per-iteration
        # combined torch.autograd.grad(var_cost, x, ...) call. Does NOT
        # change what's fed to the solver UNet (still the real, Jacobian-
        # including gradient) -- only whether the OUTER supervised loss can
        # later differentiate through that computation into prior_unet's
        # weights (and prior_weight) / x's upstream dependency on earlier
        # iterations via this channel. True decouples "does the solver
        # benefit from consuming the true differentiated-prior gradient"
        # from "does jointly training the prior through the per-iteration
        # double-backward help" -- prior_unet then trains only via the aux
        # prior_cost loss. See _build_update_input's docstring for the full
        # derivation and how this differs from detaching Phi(x) itself
        # (which would instead collapse the fed signal to subgrad+state's
        # own proxy).
        self.detach_var_cost_grad = detach_var_cost_grad
        # obs_var_indices/true_dynamics_*: _FULL_STATE_UPDATE_INPUTS modes only
        # (see that validation above -- both None for every other mode).
        # obs_var_indices: which of state_dim's channels
        # observation actually covers (state_dim itself is the FULL
        # physical state for this mode, e.g. 40 for NO=8,J=4 -- see
        # _true_ode_prior_residual). true_dynamics: a real, non-trainable
        # Lorenz96Dynamics instance (zero nn.Parameters -- not registered as
        # a submodule, just a plain attribute) used to compute the true
        # one-step-ahead prediction every iteration.
        #
        # true_dynamics_coupling_exponent defaults to 1.6, NOT
        # Lorenz96Dynamics' own class default of 1.0 -- 1.6 is
        # data/lorenz96.py's Lorenz96Config.coupling_exponent_truth, the
        # value that actually generates every training/test window's true
        # trajectory (data/lorenz96.py:110,113,122,139,368,392,448,564, all
        # via coupling_exponent_truth). Every _FULL_STATE_UPDATE_INPUTS
        # training run before this fix (jobs 53564/53595/53681) silently
        # used coupling_exponent=1.0 here -- a subtly WRONG "true" ODE, not
        # literally the dynamics that generated the data, since this
        # parameter was never threaded through at all. Matches the same
        # "default already equals the project's one canonical L96 config"
        # convention as true_dynamics_NO=8/J=4/h=1.0 above.
        self.obs_var_indices = obs_var_indices
        self.true_dynamics = None
        if update_input in _FULL_STATE_UPDATE_INPUTS:
            self.true_dynamics = Lorenz96Dynamics(
                dt=true_dynamics_dt, NO=true_dynamics_NO, J=true_dynamics_J,
                h=true_dynamics_h, coupling_exponent=true_dynamics_coupling_exponent,
                clip_range=clip_range,
            )
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
            monai_num_res_blocks=monai_num_res_blocks,
            qg_T=qg_T, qg_ny=qg_ny, qg_nx=qg_nx,
        )
        self.prior_unet = None
        if update_input in _PRIOR_MODES or aux_var_cost_weight > 0:
            # Also built for update_input NOT in _PRIOR_MODES (e.g.
            # "obs+state") whenever aux_var_cost_weight>0: this lets a plain
            # FDV1 ("obs+state") config train with the same
            # prior_cost(x_final)+prior_cost(states) auxiliary loss term
            # FDV2 uses (see compute_loss), with _build_update_input's
            # tensor construction ("obs+state" still gets cat([x, obs_clean])
            # only -- prior_unet is never referenced there for this mode)
            # completely untouched. Added specifically to ablate the
            # auxiliary loss term's effect on fast-Y reconstruction quality
            # independently of the update-input construction, after
            # measuring that the aux term's actual weight in the total loss
            # (0.2-9.5% across the three FDV2 configs) didn't correlate with
            # collapse severity -- i.e. to test whether the loss term alone,
            # applied to FDV1's own healthy architecture, degrades it.
            #
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
                dropout=self.prior_dropout,
                output_dim=state_dim,
                monai_norm_num_groups=monai_norm_num_groups,
                monai_num_res_blocks=monai_num_res_blocks,
                qg_T=qg_T, qg_ny=qg_ny, qg_nx=qg_nx,
                monai_output_init_std=self.prior_output_init_std,
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
        # `batch.obs_mask` is either (B, T) -- one mask value per timestep,
        # broadcast across the whole D-dim state (L96's convention: which
        # channels are observable is fixed over time, so only the *time*
        # axis needs masking) -- or already (B, T, D) -- a genuine per-cell
        # mask (QG's convention: which grid cells are observed varies both
        # per day *and* per cell within an observed day, e.g.
        # `cols_per_day` sparse columns; collapsing to a per-timestep-only
        # mask would silently treat every unobserved cell's zero-fill as a
        # real obs=0 measurement). Only unsqueeze the 2D case -- the 3D case
        # is used as-is.
        raw_mask = batch.obs_mask.to(batch.obs.dtype)
        if raw_mask.dim() == batch.obs.dim() - 1:
            raw_mask = raw_mask.unsqueeze(-1)
        if self.obs_var_indices is not None:
            # _FULL_STATE_UPDATE_INPUTS modes only: state_dim is the FULL physical
            # state (see _true_ode_prior_residual), but obs/obs_mask only
            # ever cover the actually-observed subspace -- embed both into
            # state_dim-shaped tensors (NaN/0 at the state_dim-obs_var_indices
            # never-observed channels, at every timestep, unlike every other
            # mode's purely-temporal mask) before anything else touches them.
            obs_full, obs_mask = _embed_obs_to_full_state(
                batch.obs, raw_mask, self.obs_var_indices, self.state_dim)
            obs_clean = torch.nan_to_num(obs_full, nan=0.0)
        else:
            obs_clean = torch.nan_to_num(batch.obs, nan=0.0)  # (B, T, D)
            obs_mask = raw_mask
        B, T, D = obs_clean.shape
        prior_ode_forcing = prior_ode_params = None
        if self.update_input in _FULL_STATE_UPDATE_INPUTS:
            # Constant across the whole unroll (unlike x) -- computed once
            # per forward() call, same convention as obs_clean/obs_mask
            # above. See _true_ode_prior_residual's docstring for the
            # exact (B,T)/(B,8) shapes expected.
            prior_ode_forcing = batch.true_forcing
            prior_ode_params = batch.true_params
        if self.init_state_var > 0:
            x = torch.randn(B, T, D, device=obs_clean.device) * (self.init_state_var ** 0.5)
        else:
            x = torch.zeros(B, T, D, device=obs_clean.device)  # x_0 = 0 (default)
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
                grad_norm_cache, self.grad_clip_range, self.gradsplit_prior_scale,
                self.prior_residual, self.detach_var_cost_grad,
                prior_ode_forcing=prior_ode_forcing, prior_ode_params=prior_ode_params,
                true_dynamics=self.true_dynamics,
                use_reentrant=False,
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

        ``self.loss_type=="var_cost"`` (default ``"mse"``, only allowed for
        the ``_FULL_STATE_UPDATE_INPUTS`` modes): replaces ONLY the
        **training**-time deep-supervision sum above with
        ``_var_cost_training_loss(block_state, ...)`` -- a genuinely
        self-supervised, weak-constraint-4DVar-style objective with NO
        ground-truth supervision at all (see that function's docstring).
        **Validation/eval stays ``F.mse_loss(x_final, states)`` regardless
        of ``loss_type``** -- deliberately NOT var_cost, even under
        ``loss_type="var_cost"`` -- so ``val_loss``/``stage1_best.ckpt``
        selection is always on the exact same scale/meaning as every other
        config (MSE-trained or var_cost-trained alike), which is also what
        lets us directly measure how well the self-supervised var_cost
        proxy tracks the true supervised MSE objective it's meant to
        approximate. When ``self.training`` and ``loss_type=="var_cost"``,
        this method also stashes ``self._last_train_mse_proxy =
        F.mse_loss(x_final, states).detach()`` -- a monitoring-only value
        (never entering the backward graph) that ``LitModel.training_step``
        additionally logs as ``train_mse_proxy`` alongside the real
        ``train_loss`` (the var_cost value actually optimized), purely to
        compare the two objectives' trajectories epoch-by-epoch.

        Both cases add -- only when ``prior_unet`` exists (built whenever
        ``update_input in _PRIOR_MODES`` OR ``aux_var_cost_weight>0``, so
        even ``"obs+state"`` gets one if the latter is set -- see
        ``__init__``) and ``aux_var_cost_weight>0`` -- a pure prior-consistency term at
        ``aux_var_cost_weight``, evaluated at the *final* block only:
        ``prior_cost(x_final) + prior_cost(states)``, i.e.
        ``||x_final - prior_unet(x_final)||^2 + ||states - prior_unet(states)||^2``
        (``tau=None``, matching the prior operator's own no-conditioning
        convention -- see ``_prior_ae``). Gives ``prior_unet`` a direct,
        single-hop gradient path to the loss, instead of relying solely on
        the 10-deep chained double-backward through
        ``torch.autograd.grad(..., create_graph=True)`` at every unrolled
        iteration.

        Deliberately does NOT multiply by ``self.prior_weight``: an earlier
        version did (``prior_weight * prior_cost(...) + obs_cost(...)``, the
        full per-iteration ``var_cost`` formula), which let a *trainable*
        ``prior_weight`` shrink this auxiliary loss simply by driving itself
        to 0 -- the ``obs_cost`` half is computed directly on
        ``x_final``/``states`` and isn't gated by ``prior_weight``, so
        zeroing ``prior_weight`` costs nothing on that half while erasing the
        entire prior-consistency term, a free win for the optimizer that has
        nothing to do with actual prior quality. Confirmed empirically: under
        this old formula, ``grad+state``'s ``prior_weight`` collapsed from
        ~0.97 to exactly 0.0 by epoch ~120 (job 53104), degrading train_loss
        after that point. This term must never reference ``self.prior_weight``
        or ``self._prior_weight_raw`` -- that parameter's only legitimate
        role is inside the per-iteration solver update (``_build_update_input``
        / ``_solver_iteration``), never in this outer supervised objective.

        ``_prior_cost`` is an unnormalized *sum* (not mean) over all
        ``B*T*D`` elements -- the right convention for the *inner* per-
        iteration variational cost (deliberately observation-count-
        independent, see ``_masked_obs_cost``'s docstring), but at realistic
        batch/window sizes that sum is ~4-5 orders of magnitude larger than
        the MSE term above (empirically: MSE~1.0 vs. raw var_cost~1e4-1e5 at
        B=32,T=300,D=24). Divide by ``B*T*D`` here -- for this *outer*
        auxiliary term only -- so ``aux_var_cost_weight`` actually controls
        the intended balance against the MSE term instead of being swamped
        by a convention mismatch.
        """
        block_states = self._unrolled_blocks(batch)
        x_final = block_states[-1]
        if self.loss_type == "var_cost" and self.training:
            raw_mask = batch.obs_mask.to(batch.obs.dtype).unsqueeze(-1)
            if self.obs_var_indices is not None:
                obs_full, obs_mask = _embed_obs_to_full_state(
                    batch.obs, raw_mask, self.obs_var_indices, self.state_dim)
                obs_clean = torch.nan_to_num(obs_full, nan=0.0)
            else:
                obs_clean = torch.nan_to_num(batch.obs, nan=0.0)
                obs_mask = raw_mask

            def _vc(s):
                return _var_cost_training_loss(
                    s, obs_clean, obs_mask, batch.true_forcing, batch.true_params,
                    self.true_dynamics, self.R_var, self.var_cost_Q_var)
            loss = sum(_vc(s) for s in block_states) / len(block_states)
            self._last_train_mse_proxy = F.mse_loss(x_final, batch.states).detach()
        elif self.training:
            loss = sum(F.mse_loss(s, batch.states) for s in block_states) / len(block_states)
        else:
            # Validation/eval: always the supervised MSE criterion,
            # regardless of loss_type -- see this method's docstring.
            loss = F.mse_loss(x_final, batch.states)
        if self.prior_unet is not None and self.aux_var_cost_weight > 0:
            numel = x_final.numel()
            prior_cost_pred = _prior_cost(self.prior_unet, x_final, residual=self.prior_residual) / numel
            prior_cost_true = _prior_cost(self.prior_unet, batch.states, residual=self.prior_residual) / numel
            loss = loss + self.aux_var_cost_weight * (prior_cost_pred + prior_cost_true)
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
        # x_tau/beta_tau: "subgrad+state+xtau"'s own outer-flow-time inputs
        # (see _build_update_input's docstring) -- x_tau_const is the outer
        # conditioning value held FIXED across the whole K_inner unroll,
        # deliberately never reassigned to x (which evolves every inner
        # iteration as the current estimate of E[x1|x_tau,y]). beta_tau uses
        # the OUTER tau (this forward() call's own tau argument), not the
        # inner-iteration tau_k below -- two different time variables.
        # Computed unconditionally (cheap) regardless of update_input, same
        # convention as every other always-computed-but-only-sometimes-used
        # quantity in this module.
        x_tau_const = x_t
        beta_tau = self.interpolant.beta(tau)
        while beta_tau.dim() < x_t.dim():
            beta_tau = beta_tau.unsqueeze(-1)
        if self.update_input in _AUTOGRAD_MODES and not x.requires_grad:
            x = x.detach().requires_grad_(True)
        grad_norm_cache = {}  # fresh per forward() call -- one unrolled (inner) solve
        denom = max(self.K_inner - 1, 1)
        for k in range(self.K_inner):
            tau_k = torch.full((x.shape[0],), k / denom, device=x.device)
            gmod = checkpoint(
                _solver_iteration, self.unet, self.update_input, x, obs_clean, obs_mask,
                tau_k, tau_k, self.prior_unet, self.R_var, 1.0, self.obs_weight,
                grad_norm_cache, self.grad_clip_range,
                x_tau=x_tau_const, beta_tau=beta_tau, use_reentrant=False,
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
