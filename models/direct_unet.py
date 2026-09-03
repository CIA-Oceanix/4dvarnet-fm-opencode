import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import AttentionPool1D, ConvBlock, Down, Up, UNet1D


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


class ParamHeadCNN(nn.Module):
    """Deterministic parameter regression head for JointDirectUNet (L8).

    Reads the model's own (oracle-free) state estimate x_hat together with
    obs/forcing and regresses a single (B, param_dim) parameter vector via a
    1D CNN over the time axis + global average pooling, exactly as the
    JointCFM param flow does but WITHOUT any tau interpolation (L8 is the
    deterministic direct-regression joint model). The output is raw (signed),
    matching the L9 param-flow convention.
    """

    def __init__(self, param_dim=4, state_dim=24, hidden_channels=None, dropout=0.1,
                 pool="mean"):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
        self.param_dim = param_dim
        self.pool = pool
        in_c = state_dim + 1 + state_dim
        self.blocks = nn.ModuleList()
        cin = in_c
        for hc in hidden_channels:
            self.blocks.append(ConvBlock(cin, hc, time_emb_dim=0, dropout=dropout))
            cin = hc
        self.head = nn.Conv1d(cin, param_dim, 1)
        if pool == "attn":
            self.attn_pool = AttentionPool1D(param_dim)

    def forward(self, obs, forcing, x_hat):
        obs_clean = torch.nan_to_num(obs, nan=0.0)
        B, T, _ = obs_clean.shape
        forcing_b = forcing.unsqueeze(-1).expand(B, T, 1)
        x_hat_clean = torch.nan_to_num(x_hat, nan=0.0)
        x = torch.cat([obs_clean, forcing_b, x_hat_clean], dim=-1)
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        if self.pool == "attn":
            return self.attn_pool(x)
        return x.mean(dim=-1)


class ParamHeadUNet(nn.Module):
    """UNet backbone deterministic parameter regression head.

    Replaces the shallow ``ParamHeadCNN`` backbone with a full encoder-decoder
    (down / bottleneck / up with skip connections) so the head can extract
    multi-scale temporal features -- including gradient/longer-timescale signal
    -- from the raw state estimate directly, instead of a shallow ~7-step
    receptive field. Same deterministic contract as ``ParamHeadCNN``:
    ``forward(obs, forcing, x_hat) -> (B, param_dim)`` raw signed, no tau.
    """

    def __init__(self, param_dim=4, state_dim=24, hidden_channels=None, dropout=0.1,
                 pool="mean"):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
        self.param_dim = param_dim
        self.pool = pool
        in_c = state_dim + 1 + state_dim
        self.enc_in = nn.Sequential(
            nn.Conv1d(in_c, hidden_channels[0], 3, padding=1),
            nn.SiLU(),
        )
        self.downs = nn.ModuleList()
        in_cc = hidden_channels[0]
        for out_c in hidden_channels:
            self.downs.append(Down(in_cc, out_c, 0))
            in_cc = out_c
        self.bottleneck = ConvBlock(hidden_channels[-1], hidden_channels[-1], 0, dropout)
        self.ups = nn.ModuleList()
        for out_c in reversed(hidden_channels):
            self.ups.append(Up(in_cc, out_c, 0))
            in_cc = out_c
        self.head = nn.Sequential(
            nn.Conv1d(in_cc, in_cc, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(in_cc, param_dim, 1),
        )
        if pool == "attn":
            self.attn_pool = AttentionPool1D(param_dim)

    def forward(self, obs, forcing, x_hat):
        obs_clean = torch.nan_to_num(obs, nan=0.0)
        B, T, _ = obs_clean.shape
        forcing_b = forcing.unsqueeze(-1).expand(B, T, 1)
        x_hat_clean = torch.nan_to_num(x_hat, nan=0.0)
        x = torch.cat([obs_clean, forcing_b, x_hat_clean], dim=-1)
        x = x.transpose(1, 2)
        h = self.enc_in(x)
        skips = []
        for down in self.downs:
            skip, h = down(h, None)
            skips.append(skip)
        h = self.bottleneck(h, None)
        for up in self.ups:
            h = up(h, skips.pop(), None)
        x = self.head(h)
        if self.pool == "attn":
            return self.attn_pool(x)
        return x.mean(dim=-1)


class JointDirectUNet(nn.Module):
    """Deterministic joint state+parameter estimator (L8).

    State flow mirrors DirectUNet (observations + corrupted forcing only, no
    parameters) so the true per-window parameters never enter conditioning --
    the joint parameters are instead recovered by a dedicated ParamHeadCNN that
    reads the model's own oracle-free state estimate. The true parameters
    appear only as the regression target in training.
    """

    def __init__(self, state_dim=3, hidden_channels=None, dropout=0.1, param_dim=4,
                 param_loss_weight=0.1, param_head_channels=None, param_ref=None,
                 param_head_pool="mean", param_head_backbone="cnn"):
        super().__init__()
        self.state_dim = state_dim
        self.param_dim = param_dim
        self.param_loss_weight = param_loss_weight
        self.cond_extra_dim = 1
        self._stage = 1
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
            output_dim=state_dim,
        )
        head_cls = ParamHeadUNet if param_head_backbone == "unet" else ParamHeadCNN
        self.param_head = head_cls(
            param_dim=param_dim,
            state_dim=state_dim,
            hidden_channels=param_head_channels,
            dropout=dropout,
            pool=param_head_pool,
        )
        if param_ref is None:
            param_ref = [1.0] * param_dim
        ref = torch.tensor(param_ref, dtype=torch.float32)
        if ref.numel() != param_dim:
            raise ValueError(f"param_ref length {ref.numel()} != param_dim {param_dim}")
        self.register_buffer("param_ref", ref)
        self.register_buffer("param_scale", 0.2 * ref)

    def _norm(self, param):
        return (param - self.param_ref) / self.param_scale

    def _denorm(self, param_norm):
        return param_norm * self.param_scale + self.param_ref

    def set_stage(self, stage):
        self._stage = stage

    def _cond(self, batch):
        obs = batch.obs
        forcing = batch.forcing
        obs_clean = torch.nan_to_num(obs, nan=0.0)
        cond = torch.cat([obs_clean, forcing.unsqueeze(-1)], dim=-1)
        return cond

    def _forward_state(self, batch):
        B, T, D = batch.obs.shape
        cond = self._cond(batch)
        x = torch.zeros(B, D, T, device=batch.obs.device)
        tau = torch.zeros(B, device=batch.obs.device)
        v_state = self.unet(x, cond.transpose(1, 2), tau=tau).transpose(1, 2)
        return v_state

    def _forward_norm(self, batch):
        v_state = self._forward_state(batch)
        params_norm = self.param_head(batch.obs, batch.forcing, v_state.detach())
        return v_state, params_norm

    def forward(self, batch):
        v_state, params_norm = self._forward_norm(batch)
        return v_state, self._denorm(params_norm)

    def estimate_params(self, batch):
        _, params = self.forward(batch)
        return params

    def compute_loss(self, batch):
        v_state, params_norm = self._forward_norm(batch)
        loss_state = F.mse_loss(v_state, batch.states)
        if batch.true_params is not None and self.param_loss_weight > 0:
            target = self._norm(batch.true_params.to(batch.states.device))
            loss_param = F.mse_loss(params_norm, target)
            return loss_state + self.param_loss_weight * loss_param
        return loss_state

    def compute_param_loss(self, batch):
        _, params_norm = self._forward_norm(batch)
        if batch.true_params is None:
            return torch.tensor(0.0, device=batch.states.device)
        target = self._norm(batch.true_params.to(batch.states.device))
        return F.mse_loss(params_norm, target)

    def sample(self, batch, return_params=False):
        v_state, params = self.forward(batch)
        if return_params:
            return v_state, params
        return v_state
