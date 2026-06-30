import torch
import torch.nn as nn

from models.unet import UNet1D


class DirectUNet(nn.Module):
    def __init__(self, state_dim=3, hidden_channels=None, dropout=0.1):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [64, 128, 256]
        self.unet = UNet1D(
            state_dim=state_dim,
            hidden_channels=hidden_channels,
            use_obs=True,
            use_energy=False,
            time_emb_dim=0,
            dropout=dropout,
        )

    def forward(self, obs):
        B, T, D = obs.shape
        x = torch.zeros(B, D, T, device=obs.device)
        tau = torch.zeros(B, device=obs.device)
        out = self.unet(x, obs.transpose(1, 2), tau=tau)
        return out.transpose(1, 2)
