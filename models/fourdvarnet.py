import torch
import torch.nn as nn
import torch.nn.functional as F

from models.interpolant import LinearInterpolant
from models.unet import UNet1D

# Recognized update_input tokens (mirrors the config-string taxonomy explored on
# CIA-Oceanix/4dvarnet-global-mapping's ronan_devs branch, contrib/4dvarnet_latent/
# models.py::GradSolver_withStep). Only the gradient-free modes are implemented
# here; the gradient/pseudo-gradient modes need obs_cost/prior_cost (a real or
# proxy variational cost) ported from ocean4dvarnet's GradSolver, deferred to a
# follow-up ("FDV2").
_IMPLEMENTED_UPDATE_INPUTS = ("obs+state", "obs-only")
_DEFERRED_UPDATE_INPUTS = ("grad-only", "grad+state", "subgrad+state")


def _validate_update_input(update_input):
    if update_input in _DEFERRED_UPDATE_INPUTS:
        raise NotImplementedError(
            f"update_input={update_input!r} needs a ported obs_cost/prior_cost "
            "(gradient-based modes) -- not yet implemented, see FDV2."
        )
    if update_input not in _IMPLEMENTED_UPDATE_INPUTS:
        raise ValueError(
            f"Unknown update_input={update_input!r}; expected one of "
            f"{_IMPLEMENTED_UPDATE_INPUTS + _DEFERRED_UPDATE_INPUTS}"
        )


class FourDVarNetSolver(nn.Module):
    """Unrolled 4DVarNet-style solver: the per-iteration update is the output
    of a UNet fed the current state and/or the observations, run for a fixed
    number of iterations -- no explicit variational cost or gradient term.

    This is deliberately simpler than ``ocean4dvarnet``'s ``GradSolver``
    (which computes ``torch.autograd.grad`` of a ``prior_cost + obs_cost``
    variational cost each step, and feeds only that gradient to a ConvLSTM
    update block). ``update_input`` selects what the update block sees each
    iteration, matching the config-string taxonomy explored on
    CIA-Oceanix/4dvarnet-global-mapping's ``ronan_devs`` branch
    (``GradSolver_withStep``'s ``input_grad_update``):

    - ``"obs+state"`` (default): ``concat(state, obs)`` -- exactly the variant
      requested first: a UNet block conditioned on both the observations and
      the running state estimate, no cost function at all.
    - ``"obs-only"``: just the observations, no state feedback.

    ``"grad-only"``/``"grad+state"``/``"subgrad+state"`` (real or pseudo
    gradient-conditioned modes) are recognized names but not yet implemented
    -- they need a ported ``obs_cost``/``prior_cost`` -- and raise
    ``NotImplementedError`` at construction rather than silently falling back
    to a different mode.
    """

    def __init__(self, state_dim=24, hidden_channels=None, time_emb_dim=64,
                 N_outer=10, dropout=0.1, update_input="obs+state"):
        super().__init__()
        _validate_update_input(update_input)
        self.update_input = update_input
        self.state_dim = state_dim
        self.N_outer = N_outer
        in_state_dim = state_dim if update_input == "obs-only" else 2 * state_dim
        self.unet = UNet1D(
            state_dim=in_state_dim,
            hidden_channels=hidden_channels,
            time_emb_dim=time_emb_dim,
            use_obs=False,
            use_energy=False,
            dropout=dropout,
            output_dim=state_dim,
        )

    def _update_input_tensor(self, x, obs_clean):
        if self.update_input == "obs-only":
            return obs_clean
        return torch.cat([x, obs_clean], dim=-1)

    def forward(self, batch, N_outer=None):
        N = self.N_outer if N_outer is None else N_outer
        obs_clean = torch.nan_to_num(batch.obs, nan=0.0)  # (B, T, D)
        B, T, D = obs_clean.shape
        x = torch.zeros(B, T, D, device=obs_clean.device)  # x_0 = 0
        denom = max(N - 1, 1)
        for k in range(N):
            tau_k = torch.full((B,), k / denom, device=x.device)
            inp = self._update_input_tensor(x, obs_clean).transpose(1, 2)
            gmod = self.unet(inp, tau=tau_k).transpose(1, 2)
            x = x - (1.0 / N) * gmod
        return x

    def compute_loss(self, batch):
        return F.mse_loss(self.forward(batch), batch.states)

    def sample(self, batch, N_outer=None):
        return self.forward(batch, N_outer=N_outer)


class FourDVarNetPredictStateCFM(nn.Module):
    """V3 (``PredictStateCFM``) CFM parameterization -- predicts
    ``mu = E[x1|x_tau,y]`` at each outer flow-time ``tau``, trained via
    ``MSE(mu, x1)`` and sampled by forward ODE integration
    ``x += dt*(mu-x)/(1-tau)`` -- but ``mu`` is computed by ``FourDVarNetSolver``'s
    own ``K_inner``-step weight-tied unrolled ``obs+state`` refinement, started
    from the current ``x_tau``, instead of a single ``UNet1D`` forward pass as
    plain ``PredictStateCFM`` uses (``models/vanilla_cfm.py``).

    Deliberately does NOT compose via a nested ``FourDVarNetSolver`` instance:
    that would produce checkpoint keys like ``model.solver.unet....``, breaking
    ``evaluation/neural_inference.py``'s checkpoint-introspection (hardcoded to
    the flat ``model.unet....``/``model.velocity_unet....`` names every other
    model in this codebase uses). Instead this class owns a flat ``self.unet``
    and re-implements ``FourDVarNetSolver.forward``'s ~8-line loop body inline
    -- if that update rule changes, mirror the change here too.

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
                 clip_range=50.0):
        super().__init__()
        _validate_update_input(update_input)
        self.update_input = update_input
        self.state_dim = state_dim
        self.N_outer = N_outer
        self.K_inner = K_inner
        self.sigma_prior = sigma_prior
        self.train_tau_0_only = train_tau_0_only
        self.clip_range = clip_range
        in_state_dim = state_dim if update_input == "obs-only" else 2 * state_dim
        self.unet = UNet1D(
            state_dim=in_state_dim,
            hidden_channels=hidden_channels,
            time_emb_dim=time_emb_dim,
            use_obs=False,
            use_energy=False,
            dropout=dropout,
            output_dim=state_dim,
        )
        self.interpolant = LinearInterpolant(nu=1.0)

    def _update_input_tensor(self, x, obs_clean):
        if self.update_input == "obs-only":
            return obs_clean
        return torch.cat([x, obs_clean], dim=-1)

    def forward(self, x_t, batch, tau):
        obs_clean = torch.nan_to_num(batch.obs, nan=0.0)
        x = x_t
        denom = max(self.K_inner - 1, 1)
        for k in range(self.K_inner):
            tau_k = torch.full((x.shape[0],), k / denom, device=x.device)
            inp = self._update_input_tensor(x, obs_clean).transpose(1, 2)
            gmod = self.unet(inp, tau=tau_k).transpose(1, 2)
            x = torch.clamp(x - (1.0 / self.K_inner) * gmod, -self.clip_range, self.clip_range)
        return x

    def compute_loss(self, batch):
        B = batch.states.shape[0]
        device = batch.states.device
        tau = torch.zeros(B, device=device) if self.train_tau_0_only else torch.rand(B, device=device)
        x0 = torch.randn_like(batch.states) * self.sigma_prior
        x_tau = self.interpolant.mix(x0, batch.states, tau)
        mu_pred = self.forward(x_tau, batch, tau)
        return F.mse_loss(mu_pred, batch.states)

    def sample(self, batch, N_outer=None):
        N = self.N_outer if N_outer is None else N_outer
        B, T, D = batch.obs.shape
        device = batch.obs.device
        x = torch.randn(B, T, self.state_dim, device=device) * self.sigma_prior
        dt = 1.0 / N
        for step in range(N):
            tau_val = step / N
            tau = torch.full((B,), tau_val, device=device)
            mu = self.forward(x, batch, tau)
            x = x + dt * (mu - x) / (1 - tau_val)
        return x
