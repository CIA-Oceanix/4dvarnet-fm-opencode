import torch

from models.interpolant import LinearInterpolant

_INTERPOLANT = LinearInterpolant(nu=1.0)


def guided_obs_cost(x_hat_1: torch.Tensor, y: torch.Tensor,
                     obs_mask: torch.Tensor, R_var: float,
                     obs_indices=None) -> torch.Tensor:
    """Observation cost sum(||y - x_hat_1||^2) / R_var over observed steps only.

    ``x_hat_1``/``y`` already live in the CFM models' observed-subspace state
    (``obs_var_indices``-restricted at the dataset level, see
    ``data/dataloader.py::FlowMatchingDataset``), so unlike the DA baselines'
    ``Strong4DVar`` closure (``evaluation/baselines.py``) no spatial
    observation operator is needed here -- only the temporal ``obs_mask``
    (True at observed steps; ``y`` is NaN-filled elsewhere, matching the
    ``_make_cond``/``nan_to_num`` convention used throughout
    ``models/vanilla_cfm.py``).

    ``obs_indices`` (optional) restricts the cost to a subset of the last
    (channel) dimension of both ``x_hat_1`` and ``y`` -- e.g. ``range(8)`` to
    simulate a slow-only observation density (see ``evaluation/run_l96.py``'s
    ``make_obs_j_indices``, whose 24D canonical ordering always places the 8
    slow ``X`` variables first) without regenerating a narrower dataset: the
    excluded channels of ``y`` simply never enter the sum, which for SDA is
    exactly equivalent to not having observed them, since obs is never a
    network input here -- only this cost term ever reads it.
    """
    if obs_indices is not None:
        x_hat_1 = x_hat_1[..., obs_indices]
        y = y[..., obs_indices]
    y_clean = torch.nan_to_num(y, nan=0.0)
    mask = obs_mask.to(x_hat_1.dtype).unsqueeze(-1)
    sq_diff = (x_hat_1 - y_clean) ** 2 * mask
    return sq_diff.sum() / R_var


def sda_guided_sample(model, batch, R_var: float, N_outer: int = 10,
                       guidance_weight=1.0, n_members: int = 1,
                       interpolant: LinearInterpolant = None, obs_indices=None,
                       mean_estimate=None, tau0: float = 0.0):
    """DPS/Pi-GDM-style guided sampling from an unconditional prior.

    At each Euler step, nudges the prior's ODE update by the *normalized*
    gradient (w.r.t. the current ``x_tau``) of the observation cost evaluated
    at the interpolant's Tweedie posterior-mean estimate
    ``x_hat_1 = x_tau + (1-tau) * v`` (``LinearInterpolant.x1_hat``):

        g = grad(J_o, x_tau)
        x_tau_next = x_tau + dt * v - guidance_weight(tau) * g / ||g||

    matching the adaptive step size of DPS (Chung et al. 2022): raw
    ``grad(J_o, x_tau)`` scales with the residual ``||x_hat_1 - y||`` through
    the *squared*-error cost, so a fixed (un-normalized) weight compounds
    across Euler steps -- through an untrained/early-training network in
    particular, whose Jacobian is unconstrained, this diverges exponentially
    within a handful of steps (verified: ``|x|`` growing 5 -> 8.5e8 over 5
    steps prior to this fix). Normalizing by ``||g||`` bounds each step to
    exactly ``guidance_weight(tau)`` regardless of the raw gradient's scale.

    ``guidance_weight`` may be a float (constant schedule) or a callable
    ``f(tau: float) -> float`` -- this knob is deliberately exposed, not
    hardcoded, per the discussion's framing of the guidance weight as another
    instance of the soft/weighted-prior design axis (see
    ``docs/research_notes_cfm_da_originality_and_benchmarking.md`` sec 4/5).

    ``obs_indices`` is forwarded to ``guided_obs_cost`` unchanged (see there)
    -- passing e.g. ``range(8)`` simulates slow-only observation density on
    top of an existing 24D-cached test set/checkpoint, with no retraining and
    no new dataset.

    ``guidance_weight == 0`` everywhere must reduce EXACTLY to the model's
    unconditional ``sample(batch, N_outer=N_outer)`` (same RNG draw, same
    Euler loop, no autograd/detach overhead) -- this is the key correctness
    invariant, checked in ``tests/test_sda_sampler.py``.

    ``mean_estimate``/``tau0`` implement a "SDEdit"-style warm start: instead
    of starting the trajectory from pure noise at tau=0, start from
    ``interpolant.mix(noise, mean_estimate, tau0)`` at ``tau0`` and only run
    the Euler loop from there to tau=1 (fewer steps -- cheaper NFE too).
    ``tau0`` is snapped to the existing ``step/N_outer`` discretization
    (``step0 = round(tau0*N_outer)``) so the warm-started point lands exactly
    on a training-time-valid interpolant point rather than an arbitrary
    off-grid tau, keeping it in-distribution for the network. This is how a
    much better initial guess (e.g. from a deterministic 4DVarNet-style
    solver -- see ``models/fourdvarnet.py``) than pure noise can anchor the
    guidance term throughout the trajectory without changing
    ``guided_obs_cost`` at all -- only the trajectory's starting point
    changes, everything downstream is untouched. ``mean_estimate=None`` (the
    default) reproduces the pre-existing behavior exactly (``step0=0``,
    ``x`` starts at pure noise) -- this is the key regression invariant for
    this extension, checked in ``tests/test_sda_sampler.py`` alongside the
    ``guidance_weight==0`` one above.

    Runs under ``torch.enable_grad()`` internally so it also works when the
    caller wraps the surrounding eval loop in ``torch.no_grad()`` (as
    ``evaluation/neural_inference.py``'s inference helpers do).

    Returns ``(x_1, n_forward)`` where ``x_1`` has shape ``(B, T, D)`` for
    ``n_members == 1`` or ``(B, T, D, n_members)`` otherwise, and
    ``n_forward`` is the number of network evaluations per sample --
    ``N_outer`` normally, or ``N_outer - step0`` when warm-starting (fewer
    steps run, so a warm-started sample is cheaper, not just better) -- the
    NFE/cost accounting knob for the report's cost table.
    """
    if interpolant is None:
        interpolant = _INTERPOLANT
    weight_fn = guidance_weight if callable(guidance_weight) else (lambda _tau: guidance_weight)

    obs = batch.obs
    B, T, _ = obs.shape
    device = obs.device
    dt = 1.0 / N_outer
    step0 = int(round(tau0 * N_outer)) if mean_estimate is not None else 0

    def _run_one():
        noise = torch.randn(B, T, model.state_dim, device=device) * model.sigma_prior
        if step0 > 0:
            x = interpolant.mix(noise, mean_estimate, torch.full((B,), step0 / N_outer, device=device))
        else:
            x = noise
        for step in range(step0, N_outer):
            tau_val = step / N_outer
            tau = torch.full((B,), tau_val, device=device)
            w = weight_fn(tau_val)
            if w == 0.0:
                with torch.no_grad():
                    v = model.forward(x, batch, tau)
                x = x + dt * v
            else:
                with torch.enable_grad():
                    x = x.detach().requires_grad_(True)
                    v = model.forward(x, batch, tau)
                    x_hat_1 = interpolant.x1_hat(x, v, tau)
                    cost = guided_obs_cost(x_hat_1, batch.obs, batch.obs_mask, R_var,
                                            obs_indices=obs_indices)
                    grad = torch.autograd.grad(cost, x)[0]
                grad_norm = grad.flatten(1).norm(dim=1).clamp_min(1e-8)
                grad_norm = grad_norm.view(B, *([1] * (grad.dim() - 1)))
                x = (x + dt * v - (w / grad_norm) * grad).detach()
        return x

    n_forward = N_outer - step0
    if n_members == 1:
        return _run_one(), n_forward
    samples = [_run_one() for _ in range(n_members)]
    return torch.stack(samples, dim=-1), n_forward
