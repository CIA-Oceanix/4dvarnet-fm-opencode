import torch
import torch.nn as nn
import torch.nn.functional as F

from models.interpolant import LinearInterpolant
from models.unet import UNet1D


class UnconditionalPriorCFM(nn.Module):
    """Unconditional flow-matching prior p(x_1) for score-based DA (SDA-style).

    Trains a pure x -> v(x, tau) velocity field with NO observation
    conditioning: unlike every other CFM variant in ``models/vanilla_cfm.py``,
    ``batch.obs``/``forcing``/``params`` are never read (only ``batch.states``,
    for the CFM training target, and ``batch.obs`` for its shape/device at
    sample time). Subclassing ``VanillaCFM`` would work too but wastes one
    ``use_obs=True`` ``UNet1D`` construction before being overwritten; this
    class instead mirrors ``VanillaCFM``'s attribute layout directly so
    ``isinstance(model, VanillaCFM)`` dispatch elsewhere doesn't apply here --
    the SDA eval path (``eval_sda_l96.py``) does not rely on it.

    Meant to be combined with the observation-guided sampler in
    ``evaluation/sda_sampler.py`` at inference time. See
    ``docs/phase_D_l96_sda.md`` for the design rationale (this is the score-
    based DA axis from the 2026-09-02 publication-positioning discussion).
    """

    def __init__(self, state_dim=3, hidden_channels=None, time_emb_dim=64,
                 N_outer=10, sigma_prior=0.5, dropout=0.1):
        super().__init__()
        self.unet = UNet1D(
            state_dim=state_dim,
            hidden_channels=hidden_channels,
            use_obs=False,
            use_energy=False,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
        )
        self.interpolant = LinearInterpolant(nu=1.0)
        self.N_outer = N_outer
        self.sigma_prior = sigma_prior
        self.state_dim = state_dim
        self.train_tau_0_only = False

    def forward(self, x_t, batch, tau):
        v = self.unet(x_t.transpose(1, 2), obs=None, tau=tau)
        return v.transpose(1, 2)

    def compute_cfm_loss(self, batch):
        B = batch.states.shape[0]
        device = batch.states.device
        tau = torch.rand(B, device=device)
        x0 = torch.randn_like(batch.states) * self.sigma_prior
        x_tau = self.interpolant.mix(x0, batch.states, tau)
        v_target = batch.states - x0
        v_pred = self.forward(x_tau, batch, tau)
        return F.mse_loss(v_pred, v_target)

    def sample(self, batch, N_outer=None):
        if N_outer is None:
            N_outer = self.N_outer
        obs = batch.obs
        B, T, D = obs.shape
        device = obs.device
        dt = 1.0 / N_outer
        x = torch.randn_like(obs) * self.sigma_prior
        for step in range(N_outer):
            tau = torch.full((B,), step / N_outer, device=device)
            v = self.forward(x, batch, tau)
            x = x + dt * v
        return x


class ConditionalPriorCFM(nn.Module):
    """Params+forcing-conditioned flow-matching prior p(x_1 | params, forcing).

    Same "requires guidance for state estimation" contract as
    ``UnconditionalPriorCFM`` (obs is still never a network input, only
    ``evaluation/sda_sampler.sda_guided_sample``'s test-time gradient term
    conditions on it) -- but unlike that class, the per-window physical
    parameters and the corrupted forcing signal (``batch.params``/
    ``batch.forcing``, both assumed known quantities in this benchmark, same
    convention as ``models/vanilla_cfm.py``'s ``_make_cond``) ARE fed to the
    network. This narrows the learned distribution to
    p(x_1 | params, forcing) instead of ``UnconditionalPriorCFM``'s single
    prior blended across the whole training mixture -- the point of the
    comparison is whether that narrower conditioning changes how much the
    guided sampler needs to do at inference time (and how it degrades S0 vs
    S1) relative to the fully unconditional prior.

    The conditioning tensor mirrors ``_make_cond`` (forcing + params
    broadcast over time) minus the ``obs`` term, passed to ``UNet1D`` as its
    ``obs`` argument with ``obs_dim=1+param_dim`` -- ``ConditionEncoder``
    only ever concatenates whatever tensor it's given, so this reuses the
    existing conditioning machinery unmodified.
    """

    def __init__(self, state_dim=3, param_dim=8, hidden_channels=None, time_emb_dim=64,
                 N_outer=10, sigma_prior=0.5, dropout=0.1):
        super().__init__()
        self.param_dim = param_dim
        self.unet = UNet1D(
            state_dim=state_dim,
            hidden_channels=hidden_channels,
            use_obs=True,
            use_energy=False,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
            obs_dim=1 + param_dim,
            cond_extra_dim=0,
        )
        self.interpolant = LinearInterpolant(nu=1.0)
        self.N_outer = N_outer
        self.sigma_prior = sigma_prior
        self.state_dim = state_dim
        self.train_tau_0_only = False

    def _cond(self, batch):
        B, T = batch.forcing.shape
        cond = batch.forcing.unsqueeze(-1)
        if self.param_dim > 0:
            params_t = batch.params.unsqueeze(1).expand(B, T, self.param_dim)
            cond = torch.cat([cond, params_t], dim=-1)
        return cond

    def forward(self, x_t, batch, tau):
        cond = self._cond(batch)
        v = self.unet(x_t.transpose(1, 2), cond.transpose(1, 2), tau=tau)
        return v.transpose(1, 2)

    def compute_cfm_loss(self, batch):
        B = batch.states.shape[0]
        device = batch.states.device
        tau = torch.rand(B, device=device)
        x0 = torch.randn_like(batch.states) * self.sigma_prior
        x_tau = self.interpolant.mix(x0, batch.states, tau)
        v_target = batch.states - x0
        v_pred = self.forward(x_tau, batch, tau)
        return F.mse_loss(v_pred, v_target)

    def sample(self, batch, N_outer=None):
        if N_outer is None:
            N_outer = self.N_outer
        obs = batch.obs
        B, T, D = obs.shape
        device = obs.device
        dt = 1.0 / N_outer
        x = torch.randn_like(obs) * self.sigma_prior
        for step in range(N_outer):
            tau = torch.full((B,), step / N_outer, device=device)
            v = self.forward(x, batch, tau)
            x = x + dt * v
        return x
