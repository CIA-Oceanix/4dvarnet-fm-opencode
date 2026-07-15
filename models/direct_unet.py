import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import UNet1D


class DirectUNet(nn.Module):
    def __init__(self, state_dim=3, hidden_channels=None, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = state_dim + 1 + 4
        if hidden_channels is None:
            hidden_channels = [64, 128, 256]
        self.unet = UNet1D(
            state_dim=state_dim,
            obs_dim=self.obs_dim,
            hidden_channels=hidden_channels,
            use_obs=True,
            use_energy=False,
            time_emb_dim=0,
            dropout=dropout,
        )

    def _make_cond(self, batch):
        obs = batch.obs
        forcing = batch.forcing
        params = batch.params
        B, T, D = obs.shape
        obs_clean = torch.nan_to_num(obs, nan=0.0)
        params_t = params.unsqueeze(1).expand(B, T, -1)
        cond = torch.cat([obs_clean, forcing.unsqueeze(-1), params_t], dim=-1)
        return cond, B, T, D

    def forward(self, batch):
        cond, B, T, D = self._make_cond(batch)
        x = torch.zeros(B, D, T, device=batch.obs.device)
        tau = torch.zeros(B, device=batch.obs.device)
        out = self.unet(x, cond.transpose(1, 2), tau=tau)
        return out.transpose(1, 2)


class JointDirectUNet(DirectUNet):
    def __init__(self, state_dim=3, param_dim=4, hidden_channels=None, dropout=0.1,
                 param_loss_weight=0.1):
        super().__init__(state_dim=state_dim, hidden_channels=hidden_channels, dropout=dropout)
        self.unet = UNet1D(
            state_dim=state_dim,
            obs_dim=self.obs_dim,
            hidden_channels=hidden_channels,
            use_obs=True,
            use_energy=False,
            time_emb_dim=0,
            dropout=dropout,
            output_dim=state_dim + param_dim,
        )
        self.param_dim = param_dim
        self.param_loss_weight = param_loss_weight

    def forward(self, batch):
        cond, B, T, D = self._make_cond(batch)
        x = torch.zeros(B, D, T, device=batch.obs.device)
        tau = torch.zeros(B, device=batch.obs.device)
        out = self.unet(x, cond.transpose(1, 2), tau=tau)
        out = out.transpose(1, 2)
        pred_state = out[..., :self.state_dim]
        param_feats = out[..., self.state_dim:]
        return pred_state, param_feats

    def predict(self, batch, return_params=False):
        pred_state, param_feats = self.forward(batch)
        if return_params:
            pooled = param_feats.mean(dim=1)
            params = F.softplus(pooled)
            return pred_state, params
        return pred_state

    def compute_loss(self, batch, loss_fn):
        pred_state, param_feats = self.forward(batch)
        loss_state = loss_fn(pred_state, batch.states)
        if batch.true_params is not None and self.param_loss_weight > 0:
            pooled = param_feats.mean(dim=1)
            param_pred = F.softplus(pooled)
            loss_param = F.mse_loss(param_pred, batch.true_params.to(pred_state.device))
            return loss_state + self.param_loss_weight * loss_param
        return loss_state
