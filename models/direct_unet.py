import torch
import torch.nn as nn
import torch.nn.functional as F

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


class JointDirectUNet(nn.Module):
    def __init__(self, state_dim=3, hidden_channels=None, dropout=0.1, param_dim=4,
                 param_loss_weight=0.1, cond_extra_dim=None):
        super().__init__()
        self.state_dim = state_dim
        self.param_dim = param_dim
        self.param_loss_weight = param_loss_weight
        self.cond_extra_dim = cond_extra_dim if cond_extra_dim is not None else 1 + param_dim
        if hidden_channels is None:
            hidden_channels = [64, 128, 256]
        self.unet = UNet1D(
            state_dim=state_dim,
            obs_dim=state_dim,
            cond_extra_dim=self.cond_extra_dim,
            hidden_channels=hidden_channels,
            use_obs=True,
            use_energy=False,
            time_emb_dim=0,
            dropout=dropout,
            output_dim=state_dim + param_dim,
        )

    def _cond(self, batch):
        obs = batch.obs
        forcing = batch.forcing
        params = batch.params
        B, T, _ = obs.shape
        obs_clean = torch.nan_to_num(obs, nan=0.0)
        if self.cond_extra_dim > 0:
            cond = torch.cat([obs_clean, forcing.unsqueeze(-1)], dim=-1)
            if self.param_dim > 0:
                params_t = params.unsqueeze(1).expand(B, T, -1)
                cond = torch.cat([cond, params_t], dim=-1)
        else:
            cond = obs_clean
        return cond

    def forward(self, batch):
        B, T, D = batch.obs.shape
        cond = self._cond(batch)
        x = torch.zeros(B, D, T, device=batch.obs.device)
        tau = torch.zeros(B, device=batch.obs.device)
        v = self.unet(x, cond.transpose(1, 2), tau=tau).transpose(1, 2)
        v_state = v[..., :self.state_dim]
        param_feats = v[..., self.state_dim:]
        return v_state, param_feats

    def estimate_params(self, batch):
        _, param_feats = self.forward(batch)
        return F.softplus(param_feats.mean(dim=1))

    def compute_loss(self, batch):
        v_state, param_feats = self.forward(batch)
        loss_state = F.mse_loss(v_state, batch.states)
        if batch.true_params is not None and self.param_loss_weight > 0:
            param_pred = F.softplus(param_feats.mean(dim=1))
            loss_param = F.mse_loss(param_pred, batch.true_params.to(batch.states.device))
            return loss_state + self.param_loss_weight * loss_param
        return loss_state

    def sample(self, batch, return_params=False):
        v_state, param_feats = self.forward(batch)
        if return_params:
            return v_state, F.softplus(param_feats.mean(dim=1))
        return v_state
