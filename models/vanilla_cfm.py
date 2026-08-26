import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet import UNet1D
from models.interpolant import LinearInterpolant


def _make_cond(obs, forcing, params, param_dim=0, cond_extra_dim=0):
    obs_clean = torch.nan_to_num(obs, nan=0.0)
    if cond_extra_dim > 0:
        B, T, D = obs.shape
        cond = torch.cat([obs_clean, forcing.unsqueeze(-1)], dim=-1)
        if param_dim > 0:
            params_t = params.unsqueeze(1).expand(B, T, -1)
            cond = torch.cat([cond, params_t], dim=-1)
    else:
        cond = obs_clean
    return cond


class VanillaCFM(nn.Module):
    def __init__(self, state_dim=3, hidden_channels=None, time_emb_dim=64, N_outer=10, sigma_prior=0.5, dropout=0.1, train_tau_0_only=False, param_dim=4, cond_extra_dim=0):
        super().__init__()
        self.cond_extra_dim = cond_extra_dim
        self.param_dim = param_dim
        self.unet = UNet1D(
            state_dim=state_dim,
            obs_dim=state_dim,
            cond_extra_dim=cond_extra_dim,
            hidden_channels=hidden_channels,
            use_obs=True,
            use_energy=False,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
        )
        self.interpolant = LinearInterpolant(nu=1.0)
        self.N_outer = N_outer
        self.sigma_prior = sigma_prior
        self.state_dim = state_dim
        self.train_tau_0_only = train_tau_0_only

    def forward(self, x_t, batch, tau):
        cond = _make_cond(batch.obs, batch.forcing, batch.params, self.param_dim, self.cond_extra_dim)
        v = self.unet(x_t.transpose(1, 2), cond.transpose(1, 2), tau=tau)
        return v.transpose(1, 2)

    def compute_cfm_loss(self, batch):
        B = batch.obs.shape[0]
        device = batch.obs.device
        tau = torch.zeros(B, device=device) if self.train_tau_0_only else torch.rand(B, device=device)
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
        x = torch.randn_like(obs) * self.sigma_prior
        if self.train_tau_0_only:
            v = self.forward(x, batch, tau=torch.zeros(B, device=device))
            return x + v
        dt = 1.0 / N_outer
        for step in range(N_outer):
            tau = torch.full((B,), step / N_outer, device=device)
            v = self.forward(x, batch, tau)
            x = x + dt * v
        return x


class JointCFM(VanillaCFM):
    def __init__(self, state_dim=3, param_dim=4, hidden_channels=None, time_emb_dim=64,
                 N_outer=10, sigma_prior=0.5, dropout=0.1, param_loss_weight=0.1,
                 train_tau_0_only=False):
        super().__init__(state_dim=state_dim, param_dim=param_dim,
                         hidden_channels=hidden_channels,
                         time_emb_dim=time_emb_dim, N_outer=N_outer,
                         sigma_prior=sigma_prior, dropout=dropout,
                         cond_extra_dim=1 + param_dim)
        self.unet = UNet1D(
            state_dim=state_dim,
            obs_dim=state_dim,
            cond_extra_dim=1 + param_dim,
            hidden_channels=hidden_channels,
            use_obs=True,
            use_energy=False,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
            output_dim=state_dim + param_dim,
        )
        self.param_dim = param_dim
        self.param_loss_weight = param_loss_weight
        self.train_tau_0_only = train_tau_0_only

    def forward(self, x_t, batch, tau):
        cond = _make_cond(batch.obs, batch.forcing, batch.params, self.param_dim, 1 + self.param_dim)
        v = self.unet(x_t.transpose(1, 2), cond.transpose(1, 2), tau=tau)
        v = v.transpose(1, 2)
        v_state = v[..., :self.state_dim]
        param_feats = v[..., self.state_dim:]
        return v_state, param_feats

    def estimate_params(self, batch):
        obs = batch.obs
        B, T, D = obs.shape
        device = obs.device
        x_t = torch.randn_like(obs) * self.sigma_prior
        _, param_feats = self.forward(x_t, batch, tau=torch.ones(B, device=device))
        pooled = param_feats.mean(dim=1)
        return F.softplus(pooled)

    def compute_cfm_loss(self, batch):
        B = batch.obs.shape[0]
        device = batch.obs.device
        tau = torch.zeros(B, device=device) if self.train_tau_0_only else torch.rand(B, device=device)
        x0 = torch.randn_like(batch.states) * self.sigma_prior
        x_tau = self.interpolant.mix(x0, batch.states, tau)
        v_target = batch.states - x0
        v_pred_state, param_feats = self.forward(x_tau, batch, tau)
        loss_cfm = F.mse_loss(v_pred_state, v_target)
        if batch.true_params is not None and self.param_loss_weight > 0:
            pooled = param_feats.mean(dim=1)
            param_pred = F.softplus(pooled)
            loss_param = F.mse_loss(param_pred, batch.true_params.to(device))
            return loss_cfm + self.param_loss_weight * loss_param
        return loss_cfm

    def sample(self, batch, N_outer=None, return_params=False):
        if N_outer is None:
            N_outer = self.N_outer
        obs = batch.obs
        B, T, D = obs.shape
        device = obs.device
        dt = 1.0 / N_outer
        x = torch.randn_like(obs) * self.sigma_prior
        if self.train_tau_0_only:
            v_state, param_feats = self.forward(x, batch, tau=torch.zeros(B, device=device))
            x = x + v_state
        else:
            for step in range(N_outer):
                tau = torch.full((B,), step / N_outer, device=device)
                v_state, _ = self.forward(x, batch, tau)
                x = x + dt * v_state
        if return_params:
            _, param_feats = self.forward(x, batch, tau=torch.ones(B, device=device))
            pooled = param_feats.mean(dim=1)
            params = F.softplus(pooled)
            return x, params
        return x


class PredictStateCFM(nn.Module):
    """V3 CFM variant where the network predicts E[x1|xt,y] instead of E[x1-x0|xt,y].

    ODE formulation:
        v = (μ - x) / (1 - τ)  where μ = E[x1 | x_τ, y]
    This represents a backward-drift mechanism that pulls the state towards
    the predicted final state.
    """
    def __init__(self, state_dim=3, hidden_channels=None, time_emb_dim=64,
                 N_outer=10, sigma_prior=0.5, dropout=0.1,
                 train_tau_0_only=False, param_dim=4, cond_extra_dim=0):
        super().__init__()
        self.param_dim = param_dim
        self.cond_extra_dim = cond_extra_dim
        self.hidden_channels = hidden_channels if hidden_channels is not None else [64, 128, 256]
        self.time_emb_dim = time_emb_dim
        self.unet = UNet1D(
            state_dim=state_dim,
            obs_dim=state_dim,
            cond_extra_dim=cond_extra_dim,
            hidden_channels=hidden_channels,
            use_obs=True,
            use_energy=False,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
            output_dim=state_dim,
        )
        self.interpolant = LinearInterpolant(nu=1.0)
        self.N_outer = N_outer
        self.sigma_prior = sigma_prior
        self.state_dim = state_dim
        self.train_tau_0_only = train_tau_0_only

    def forward(self, x_t, batch, tau):
        """Forward pass: predict final state mean μ = E[x1|xt,y]."""
        cond = _make_cond(batch.obs, batch.forcing, batch.params,
                          self.param_dim, self.cond_extra_dim)
        μ = self.unet(x_t.transpose(1, 2), cond.transpose(1, 2), tau=tau)
        return μ.transpose(1, 2)

    def compute_loss(self, batch):
        """Compute CFM loss: MSE(μ, x1) where μ = network prediction."""
        B = batch.obs.shape[0]
        device = batch.obs.device
        tau = torch.zeros(B, device=device) if self.train_tau_0_only else torch.rand(B, device=device)
        x0 = torch.randn_like(batch.states) * self.sigma_prior
        x_tau = self.interpolant.mix(x0, batch.states, tau)
        μ_pred = self.forward(x_tau, batch, tau)
        return F.mse_loss(μ_pred, batch.states)

    def sample(self, batch, N_outer=None):
        """Sample trajectories via forward ODE integration.

        The network predicts μ = E[x_τ=1 | x_τ, y]. We sample by integrating forward:
            x_0 ~ N(0, σ²)
            For τ from 0 to 1: x_τ ← x_τ + dt * (μ_τ - x_τ) / (1 - τ)
        """
        if N_outer is None:
            N_outer = self.N_outer
        obs = batch.obs
        B, T, D = obs.shape
        device = obs.device

        if self.train_tau_0_only:
            x0 = torch.randn_like(obs) * self.sigma_prior
            μ = self.forward(x0, batch, tau=torch.zeros(B, device=device))
            return μ  # single-step: x0 + (μ - x0)/1 = μ

        # Start from random x_0
        x = torch.randn_like(obs) * self.sigma_prior
        dt = 1.0 / N_outer

        # Forward integration with tau as tensor (avoid tau=1 division by zero)
        for step in range(N_outer):
            tau_step = torch.full((B,), step / N_outer, device=device)
            mu = self.forward(x, batch, tau_step)
            v = (mu - x) / (1.0 - tau_step.clamp(max=0.999).view(B, 1, 1).expand(-1, T, -1))
            x = x + dt * v

        return x
