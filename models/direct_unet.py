import torch
import torch.nn as nn

from models.unet import UNet1D


class DirectUNet(nn.Module):
    def __init__(self, state_dim=3, hidden_channels=None, dropout=0.1, param_dim=4,
                 cond_extra_dim=0):
        super().__init__()
        self.state_dim = state_dim
        self.param_dim = param_dim
        self.cond_extra_dim = cond_extra_dim
        if hidden_channels is None:
            hidden_channels = [64, 128, 256]
        self.unet = UNet1D(
            state_dim=state_dim,
            obs_dim=state_dim,
            cond_extra_dim=cond_extra_dim,
            hidden_channels=hidden_channels,
            use_obs=True,
            use_energy=False,
            time_emb_dim=0,
            dropout=dropout,
        )

    def forward(self, batch):
        obs = batch.obs
        forcing = batch.forcing
        params = batch.params
        B, T, D = obs.shape
        obs_clean = torch.nan_to_num(obs, nan=0.0)
        if self.cond_extra_dim > 0:
            cond = torch.cat([obs_clean, forcing.unsqueeze(-1)], dim=-1)
            if self.param_dim > 0:
                params_t = params.unsqueeze(1).expand(B, T, -1)
                cond = torch.cat([cond, params_t], dim=-1)
        else:
            cond = obs_clean
        x = torch.zeros(B, D, T, device=obs.device)
        tau = torch.zeros(B, device=obs.device)
        out = self.unet(x, cond.transpose(1, 2), tau=tau)
        return out.transpose(1, 2)
