import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet import UNet1D
from models.interpolant import LinearInterpolant


class VanillaCFM(nn.Module):
    def __init__(self, state_dim=3, hidden_channels=None, time_emb_dim=64, N_outer=10, sigma_prior=0.5, dropout=0.1):
        super().__init__()
        self.unet = UNet1D(
            state_dim=state_dim,
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

    def forward(self, x_t, obs, tau):
        B, T, D = x_t.shape
        v = self.unet(x_t.transpose(1, 2), obs.transpose(1, 2), tau=tau)
        return v.transpose(1, 2)

    def compute_cfm_loss(self, batch):
        B = batch.obs.shape[0]
        device = batch.obs.device
        tau = torch.rand(B, device=device)
        x0 = torch.randn_like(batch.states) * self.sigma_prior
        x_tau = self.interpolant.mix(x0, batch.states, tau)
        v_target = batch.states - x0
        v_pred = self.forward(x_tau, batch.obs, tau)
        return F.mse_loss(v_pred, v_target)

    def sample(self, obs, N_outer=None):
        if N_outer is None:
            N_outer = self.N_outer
        B, T, D = obs.shape
        device = obs.device
        dt = 1.0 / N_outer
        x = torch.randn_like(obs) * self.sigma_prior
        for step in range(N_outer):
            tau = torch.full((B,), step / N_outer, device=device)
            v = self.forward(x, obs, tau)
            x = x + dt * v
        return x
