import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import AttentionPool1D, ConvBlock


class StateParamHead(nn.Module):
    """Decoupled deterministic parameter regression (cascade test).

    Estimates the per-window parameters from FOUR inputs -- raw observations,
    the biased (``*_da``) parameters, the corrupted forcing and an external
    frozen state estimate ``x_hat`` (e.g. from a state-only DirectUNet). This
    decouples param estimation from state estimation: unlike the joint models
    (L7/L8/L9), the state input is NOT the model's own collapsed state.

    Inputs are stacked over the time axis and reduced to a single ``(B, P)``
    parameter vector by a 1D CNN + global pooling, mirroring ``ParamHeadCNN``
    (JointDirectUNet, L8) but with the raw obs and biased params added.
    The output is raw (signed), matching the L9/L8 param convention. True params
    appear only as the regression target (never as an input), so no oracle leaks.
    """

    def __init__(self, state_dim=24, param_dim=8, hidden_channels=None, dropout=0.1,
                 param_ref=None, param_head_pool="mean", augment_derivatives=False):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
        self.state_dim = state_dim
        self.param_dim = param_dim
        self.pool = param_head_pool
        self.augment_derivatives = augment_derivatives
        # obs (D) + biased params (P) + forcing (1) + state estimate (D) [+ derivative (D)]
        in_c = state_dim + param_dim + 1 + state_dim
        if augment_derivatives:
            in_c += state_dim
        self.blocks = nn.ModuleList()
        cin = in_c
        for hc in hidden_channels:
            self.blocks.append(ConvBlock(cin, hc, time_emb_dim=0, dropout=dropout))
            cin = hc
        self.head = nn.Conv1d(cin, param_dim, 1)
        if param_head_pool == "attn":
            self.attn_pool = AttentionPool1D(param_dim)

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

    def _final_inputs(self, batch, x_hat):
        obs = torch.nan_to_num(batch.obs, nan=0.0)
        B, T, _ = obs.shape
        forcing_b = batch.forcing.unsqueeze(-1).expand(B, T, 1)
        params_b = batch.params.unsqueeze(1).expand(B, T, self.param_dim)
        x_hat_clean = torch.nan_to_num(x_hat, nan=0.0)
        parts = [obs, params_b, forcing_b, x_hat_clean]
        if self.augment_derivatives:
            parts.append(torch.diff(x_hat_clean, dim=1, prepend=x_hat_clean[:, :1]))
        x = torch.cat(parts, dim=-1)
        return x

    def forward(self, batch, x_hat):
        x = self._final_inputs(batch, x_hat)
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        if self.pool == "attn":
            out = self.attn_pool(x)
        else:
            out = x.mean(dim=-1)
        return out

    def compute_loss(self, batch, x_hat):
        params_norm = self.forward(batch, x_hat)
        if batch.true_params is None:
            return torch.tensor(0.0, device=params_norm.device)
        target = self._norm(batch.true_params.to(params_norm.device))
        return F.mse_loss(params_norm, target)

    def estimate_params(self, batch, x_hat):
        return self._denorm(self.forward(batch, x_hat))


def _strip_model_prefix(sd, prefix="model."):
    if all(k.startswith(prefix) for k in sd):
        return {k[len(prefix):]: v for k, v in sd.items()}
    return sd


def _build_and_load_direct_unet(state_checkpoint, state_dim, hidden_channels,
                                cond_extra_dim, device):
    """Build a state-only DirectUNet and load weights from a Lightning ckpt.

    The checkpoint's ``state_dict`` keys are the encapsulated ``LitModel``
    flattened layout (``model.unet.*``); stripping the leading ``model.`` gives
    the ``DirectUNet`` layout (``unet.*``). The resulting encoder is frozen.
    """
    from models.direct_unet import DirectUNet
    enc = DirectUNet(state_dim=state_dim, hidden_channels=hidden_channels,
                     dropout=0.1, param_dim=0, cond_extra_dim=cond_extra_dim)
    ckpt = torch.load(state_checkpoint, map_location="cpu", weights_only=False)
    sd = _strip_model_prefix(ckpt["state_dict"])
    enc.load_state_dict(sd, strict=True)
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


class StateParamModel(nn.Module):
    """Composite cascade: frozen state-only DirectUNet + trainable param head.

    ``self.state_encoder`` is a pre-trained state-only estimator (e.g. L1b)
    frozen at eval, producing ``x_hat``; ``self.param_head`` is the trainable
    ``StateParamHead`` that regresses the parameters from (obs, biased params,
    forcing, x_hat). Decouples param estimation from state estimation so the
    param head never has to reconstruct a state of its own (the pathology that
    broke the joint L7/L8/L9 models on S1).

    ``state_source`` selects the state fed to the head:
      - ``"l1b"``: the frozen pre-trained state-only encoder (default)
      - ``"true"``: the true observed-subspace state (``batch.states``), used
        to ablate whether the encoder's estimate quality (vs the information
        bottleneck) is what limits parameter recovery.
    """

    def __init__(self, state_dim=24, param_dim=8, state_checkpoint=None,
                 state_model_type="direct_unet", state_hidden_channels=None,
                 state_cond_extra_dim=0, param_head_channels=None,
                 param_ref=None, param_head_pool="mean", state_source="l1b",
                 augment_derivatives=False, device=None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if state_hidden_channels is None:
            state_hidden_channels = [64, 128, 256]
        self.state_dim = state_dim
        self.param_dim = param_dim
        self.state_checkpoint = state_checkpoint
        self.state_source = state_source
        if state_source == "l1b":
            if state_checkpoint is None:
                raise ValueError("StateParamModel(state_source='l1b') requires a state_checkpoint")
            self.state_encoder = _build_and_load_direct_unet(
                state_checkpoint, state_dim, state_hidden_channels,
                state_cond_extra_dim, device)
        else:
            self.state_encoder = None
        self.param_head = StateParamHead(
            state_dim=state_dim, param_dim=param_dim,
            hidden_channels=param_head_channels, dropout=0.1,
            param_ref=param_ref, param_head_pool=param_head_pool,
            augment_derivatives=augment_derivatives,
        )
        self._stage = 1
        self._device = device

    def _xhat(self, batch):
        if self.state_source in ("true", True):
            return batch.states
        return self.state_encoder(batch)

    def compute_loss(self, batch):
        x_hat = self._xhat(batch)
        return self.param_head.compute_loss(batch, x_hat)

    def estimate_params(self, batch):
        x_hat = self._xhat(batch)
        return self.param_head.estimate_params(batch, x_hat)

    def forward(self, batch, **kwargs):
        x_hat = self._xhat(batch)
        params = self.param_head.estimate_params(batch, x_hat)
        return x_hat, params

    def set_stage(self, stage):
        self._stage = stage

