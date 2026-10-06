"""Iterative Corrupted Trajectory Matching (ICTM).

Zhang et al., "Flow Priors for Linear Inverse Problems via Iterative
Corrupted Trajectory Matching" (NeurIPS 2024) -- see ``reports/FlowPrior.pdf``.
A MAP-style point-estimate reconstruction from a pretrained flow-matching
prior, applied here to this repo's DA observation setup (the same
obs_mask-restricted identity operator ``evaluation/sda_sampler.py``'s
``guided_obs_cost`` uses -- the paper's linear ``A`` is spatial subsampling/
blur/etc. for images; here it's "observed timesteps of the window", so ``A``
reduces to that same mask) instead of the paper's image inverse problems.

Distinct from ``evaluation/sda_sampler.py::sda_guided_sample`` (Rozet &
Louppe 2023, "Score-based Data Assimilation"): SDA takes one DPS/Pi-GDM-
style *normalized*-gradient nudge per Euler step and is built for
*ensemble* sampling (many independent noise draws approximate p(x|y)).
ICTM instead refines each Euler step's state in place via ``K`` inner
gradient (Adam) steps on a genuine **local MAP objective** (the paper's
Theorem 1 / Eq. 10) before advancing -- a deterministic point estimate by
design, not an ensemble method (though ``n_members`` independent MAP solves
can still be stacked, e.g. for a spread sanity check against SDA).

Local objective at Euler step t -> t+dt (Algorithm 1, adapted to this
module's identity-style ``A``):

    L(x_t) = lam * ||x_t + dt*v_theta(x_t,t) - y_tau_next||_obs^2   # local data likelihood
             - log p(x_t)                                           # local prior (Eq. 11 score)
             + tr(dv_theta(x_t,t)/dx) * dt                          # log-density Riemannian-sum term

``y_tau_next`` is the paper's "corrupted trajectory" auxiliary path: an
interpolation between the true observation ``y`` (weight -> 1 as tau -> 1)
and the initial noise draw ``x_0`` (weight -> 1 as tau -> 0), computed via
``interpolant.mix(x0, y_clean, tau_next)`` -- literally the same ``mix``
used elsewhere to blend noise and data, just fed ``(x0, y)`` instead of
``(x0, x1)``. ``||.||_obs`` is ``evaluation.sda_sampler.guided_obs_cost``,
reused unchanged (same obs_mask/obs_indices/obs_channel_mask semantics),
with ``R_var=1.0`` so ``lam`` alone controls the data-term weight, matching
the paper's own choice to replace the exact ``1/(2*sigma_y^2)`` coefficient
with a single tuned hyperparameter (their Sec. 3.2 discussion after
Algorithm 1: "we choose lambda as a new hyper-parameter to tune").

``-log p(x_t)``'s *gradient* is available in closed form from the flow's
own velocity (Proposition 1, ``LinearInterpolant.score``) -- no extra
backward pass through the model needed for that term, and no separate
value/gradient needed for its own sake. This also means Algorithm 1's
explicit t=0 special case (lines 5-8: use the exact
``||x_0||^2/(2*sigma^2)`` prior since x_0 ~ N(0, sigma^2 I) is known in
closed form) is *algebraically identical* to the general Proposition-1
score formula evaluated at tau=0 (``LinearInterpolant.score``'s docstring
and ``tests/test_interpolant.py::test_score_at_tau0_ignores_v``) -- so this
implementation uses one unified formula for every step instead of
Algorithm 1's branching.

``tr(dv/dx)`` is estimated via the Hutchinson/Skilling trace estimator
(``hutchinson_trace``, standard in continuous normalizing flows -- FFJORD,
Grathwohl et al. 2019), a single vector-Jacobian product rather than the
full D-dimensional Jacobian, and is itself differentiable w.r.t. ``x``
(``create_graph=True``) so its contribution to the K-step gradient descent
is exact, not an approximation stacked on an approximation. Requires the
model to support double backward (no in-place activations/BatchNorm in
``models/unet.py`` as of this writing -- verified by
``test_ictm_map_sample_runs_end_to_end`` below using the real
``UnconditionalPriorCFM``/``UNet1D`` stack, not a toy linear model).

Matches the paper's stated practice of using Adam for the K inner steps
(their Sec. 4.3 "We use the Adam optimizer... due to its effectiveness in
neural network computations"); a fresh ``Adam([x])`` is created per outer
Euler step (no momentum carried across steps -- consistent with Algorithm
1's per-step ``x_t <- x_t.detach()`` framing).
"""
import torch

from evaluation.sda_sampler import guided_obs_cost
from models.interpolant import LinearInterpolant
from models.vanilla_cfm import PredictStateCFM


def _velocity(model, x: torch.Tensor, batch, tau: torch.Tensor) -> torch.Tensor:
    """Returns the velocity field ``v`` at ``(x, tau)`` regardless of what
    ``model.forward`` itself predicts.

    Every formula in ``ictm_map_sample`` (the Euler step, the Tweedie
    posterior estimate, ``LinearInterpolant.score``) assumes ``v = x1 - x0``
    (VanillaCFM/TweedieCFM's convention, and what ``model.forward`` returns
    for them directly). ``PredictStateCFM`` (and its Monai subclass) instead
    predicts ``mu = E[x1|x_tau,y]`` -- a direct state estimate, not a
    velocity -- and only derives its own velocity via
    ``v = (mu - x) / (1 - tau)`` inside ``PredictStateCFM.sample()``. Feeding
    ``mu`` straight into the velocity-shaped formulas above silently computes
    something else entirely (bounded, so it doesn't diverge, but wrong) --
    this reproduces that same ``PredictStateCFM.sample()`` conversion so
    ICTM sees a real velocity for that model family too."""
    out = model.forward(x, batch, tau)
    if isinstance(model, PredictStateCFM):
        B, T, _ = x.shape
        denom = (1.0 - tau.clamp(max=0.999)).view(B, 1, 1).expand(-1, T, -1)
        return (out - x) / denom
    return out

_INTERPOLANT = LinearInterpolant(nu=1.0)


def hutchinson_trace(v: torch.Tensor, x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Differentiable stochastic estimate of tr(dv/dx), one sample per batch
    element, via a single vector-Jacobian product:
    E_eps[eps^T (dv/dx) eps] = tr(dv/dx) for eps with E[eps eps^T] = I (e.g.
    Rademacher or standard Gaussian noise -- see ``rademacher_like``).
    ``create_graph=True`` on the vjp keeps the result differentiable w.r.t.
    ``x`` for a second backward pass (needed since the local objective's
    gradient w.r.t. ``x_t`` includes this term).

    Returns a ``(B,)`` tensor (summed over all non-batch dims of v/x).
    """
    vjp = torch.autograd.grad(v, x, grad_outputs=eps, create_graph=True)[0]
    return (vjp * eps).flatten(1).sum(dim=1)


def rademacher_like(x: torch.Tensor) -> torch.Tensor:
    """+-1 noise with the same shape/device/dtype as x (E[eps eps^T] = I,
    lower-variance Hutchinson estimator than standard Gaussian noise)."""
    return torch.randint(0, 2, x.shape, device=x.device, dtype=x.dtype) * 2.0 - 1.0


def ictm_map_sample(model, batch, N_outer: int = 100, K: int = 1,
                     lam: float = 1.0, step_size: float = 1e-2,
                     n_members: int = 1, interpolant: LinearInterpolant = None,
                     obs_indices=None, obs_channel_mask=None,
                     trace_samples: int = 1):
    """ICTM MAP reconstruction from an unconditional/conditional flow prior.

    Returns ``(x_1, n_forward)``, matching ``sda_guided_sample``'s return
    convention: ``x_1`` has shape ``(B,T,D)`` for ``n_members==1`` or
    ``(B,T,D,n_members)`` otherwise, and ``n_forward`` is network
    evaluations per sample -- ``N_outer * (K + 1)`` (``K`` inner-refinement
    forwards per outer step, plus one final forward to commit the Euler
    advance with the refined ``x_t``, Algorithm 1 line 15).

    ``lam=0`` still runs the prior-only (no observation) refinement -- there
    is no analogue of ``sda_guided_sample``'s ``guidance_weight==0`` exact-
    unconditional-sample invariant here, since the trace + prior-score
    terms are always active by construction (this *is* the local-MAP
    objective, not an add-on to an otherwise-unconditional Euler loop).

    Call with ``model.eval()`` (matches ``sda_guided_sample``'s convention)
    -- also required here so dropout doesn't break the double-backward
    trace-gradient computation.
    """
    interpolant = interpolant or _INTERPOLANT
    obs = batch.obs
    B, T, _ = obs.shape
    device = obs.device
    dt = 1.0 / N_outer
    y_clean = torch.nan_to_num(obs, nan=0.0)

    def _run_one():
        x0 = torch.randn(B, T, model.state_dim, device=device) * model.sigma_prior
        x = x0
        for step in range(N_outer):
            tau = torch.full((B,), step / N_outer, device=device)
            tau_next = torch.full((B,), (step + 1) / N_outer, device=device)
            y_tau_next = interpolant.mix(x0, y_clean, tau_next)

            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                opt = torch.optim.Adam([x], lr=step_size)
                v = None
                for _ in range(K):
                    opt.zero_grad()
                    v = _velocity(model, x, batch, tau)
                    x_next_pred = x + dt * v
                    data_cost = lam * guided_obs_cost(
                        x_next_pred, y_tau_next, batch.obs_mask, R_var=1.0,
                        obs_indices=obs_indices, obs_channel_mask=obs_channel_mask)
                    trace_cost = x.new_zeros(())
                    for _ in range(trace_samples):
                        eps = rademacher_like(x)
                        trace_cost = trace_cost + hutchinson_trace(v, x, eps).sum() * dt
                    trace_cost = trace_cost / trace_samples
                    loss = data_cost + trace_cost
                    loss.backward()
                    prior_grad = -interpolant.score(x.detach(), v.detach(), tau, sigma=model.sigma_prior)
                    x.grad = x.grad + prior_grad
                    opt.step()
                x = x.detach()

            with torch.no_grad():
                v_final = _velocity(model, x, batch, tau)
                x = x + dt * v_final
        return x

    n_forward = N_outer * (K + 1)
    if n_members == 1:
        return _run_one(), n_forward
    samples = [_run_one() for _ in range(n_members)]
    return torch.stack(samples, dim=-1), n_forward
