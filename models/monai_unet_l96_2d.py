"""MONAI-backed 2D (time x sector) DirectUNet for the two-scale L96 case study.

`MonaiDirectUNet` treats the 24 observed L96 variables as channels of a 1D
sequence over time, so every sector learns its own weights. This backbone
instead lays the observed state out on a (T, NO) grid -- one column per slow
sector k, with channels [X_k, Y_{k,0}, ..., Y_{k,J_obs-1}] -- and convolves
over time and sector jointly, matching L96's cyclic symmetry in k:

- sector axis: circular padding, never downsampled (NO = 8 is already the
  coarsest useful resolution, and stride-2 pooling would break the
  equivariance to odd sector shifts);
- time axis: zero padding, downsampled/upsampled by 2 per level as in the 1D
  backbone.

With no positional input along the sector axis the whole network (convs,
GroupNorm, the mid-block self-attention) is exactly equivariant to cyclic
sector shifts. The fast index j stays a channel: `fast_weights` and the
observed subset (j < J_obs) differ per j, so it carries no symmetry.

Unlike `models.monai_unet_qg2d`, which re-registers MONAI's global 2D conv
factory with circular padding on both axes (wrong here: time is not
periodic), the padding is changed per module after construction, so both
backbones can coexist in one process.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import DiffusionModelUNet
from monai.networks.nets import diffusion_model_unet as _dmu

from models.monai_unet_adapter import MonaiDirectUNet, _patch_resblock_for_1d


class SectorCircularConv2d(nn.Conv2d):
    """Conv over (time, sector): zero padding in time, circular in sector."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t, pad_k = self.padding
        if pad_k:
            x = F.pad(x, (pad_k, pad_k, 0, 0), mode="circular")
        if pad_t:
            x = F.pad(x, (0, 0, pad_t, pad_t))
        return F.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


def _make_time_only_resampling(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, _dmu.DiffusionUnetDownsample) and m.use_conv:
            conv = m.op.conv if hasattr(m.op, "conv") else m.op
            conv.stride = (2, 1)
        if isinstance(m, nn.Upsample):
            m.scale_factor = (2.0, 1.0)


def _make_sector_circular(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            m.__class__ = SectorCircularConv2d


class TimeAxialAttention(nn.Module):
    """Runs a MONAI spatial attention block along time only, separately for
    each sector with shared weights: (B, C, T, K) -> (B*K, C, T, 1)."""

    def __init__(self, attention: nn.Module):
        super().__init__()
        self.attention = attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, K = x.shape
        h = x.permute(0, 3, 1, 2).reshape(B * K, C, T, 1)
        h = self.attention(h)
        return h.reshape(B, K, C, T).permute(0, 2, 3, 1)


MID_ATTENTION_MODES = ("time", "full", "none")


def l96_sector_layout(NO: int, D: int) -> tuple[int, int]:
    if (D - NO) % NO != 0:
        raise ValueError(f"state_dim={D} is not NO + NO*J_obs for NO={NO}")
    return NO, D // NO


def to_sector_grid(v: torch.Tensor, NO: int) -> torch.Tensor:
    """(B, T, NO + NO*J) -> (B, 1 + J, T, NO), channel 0 = X_k."""
    B, T, D = v.shape
    J = (D - NO) // NO
    X = v[..., :NO].unsqueeze(-1)
    Y = v[..., NO:].reshape(B, T, NO, J)
    return torch.cat([X, Y], dim=-1).permute(0, 3, 1, 2)


def from_sector_grid(g: torch.Tensor) -> torch.Tensor:
    """Inverse of `to_sector_grid`: (B, 1 + J, T, NO) -> (B, T, NO + NO*J)."""
    B, C, T, NO = g.shape
    g = g.permute(0, 2, 3, 1)
    return torch.cat([g[..., 0], g[..., 1:].reshape(B, T, NO * (C - 1))], dim=-1)


class MonaiUNet2DSector(nn.Module):
    """MONAI `DiffusionModelUNet` (spatial_dims=2) on a (time, sector) grid,
    circular in sector, downsampled in time only.

    `mid_attention` sets the bottleneck self-attention: "time" (per sector,
    along time, as in the 1D backbone; default), "full" (over all time x
    sector positions, ~2.4x the training step time) or "none"."""

    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: list[int] | None = None,
                 num_res_blocks: int = 2, norm_num_groups: int = 32,
                 dropout: float = 0.1, use_flash_attention: bool = True,
                 mid_attention: str = "time"):
        super().__init__()
        if mid_attention not in MID_ATTENTION_MODES:
            raise ValueError(f"mid_attention={mid_attention!r}; expected one of {MID_ATTENTION_MODES}")
        _patch_resblock_for_1d()
        if hidden_channels is None:
            hidden_channels = [64, 128, 256]
        self.backbone = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=tuple(hidden_channels),
            attention_levels=tuple(False for _ in hidden_channels),
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            use_flash_attention=use_flash_attention,
        )
        if dropout > 0:
            for module in self.backbone.modules():
                if isinstance(module, _dmu.DiffusionUNetResnetBlock):
                    module.dropout = nn.Dropout(dropout)
        _make_time_only_resampling(self.backbone)
        _make_sector_circular(self.backbone)
        mid = self.backbone.middle_block
        if mid_attention == "time":
            mid.attention = TimeAxialAttention(mid.attention)
        elif mid_attention == "none":
            mid.attention = nn.Identity()

    def forward(self, x: torch.Tensor, tau: torch.Tensor | None = None) -> torch.Tensor:
        timesteps = tau if tau is not None else torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return self.backbone(x, timesteps)


class MonaiDirectUNet2D(MonaiDirectUNet):
    """2D (time x sector) drop-in for `MonaiDirectUNet`: same
    `forward(batch) -> (B, T, D)` contract on the 24-D observed L96 subspace
    (`evaluation.run_l96.make_obs_j_indices` order: X_0..X_{NO-1}, then
    Y_{k,j} for k, j < J_obs). Subclassing keeps the `isinstance(model,
    MonaiDirectUNet)` dispatch in `evaluation.neural_inference` working.

    Only observation-only conditioning is supported (`cond_extra_dim=0`,
    `param_dim=0`, the benchmark default): the scalar forcing and parameters
    carry no sector structure. As in the 1D model, unobserved entries enter
    as zeros, with no mask channel.
    """

    def __init__(self, state_dim: int = 24, NO: int = 8, hidden_channels=None,
                 dropout: float = 0.1, param_dim: int = 0, cond_extra_dim: int = 0,
                 num_res_blocks: int = 2, norm_num_groups: int = 32,
                 use_flash_attention: bool = True, mid_attention: str = "time"):
        nn.Module.__init__(self)
        if cond_extra_dim or param_dim:
            raise ValueError("MonaiDirectUNet2D supports obs-only conditioning "
                             f"(got cond_extra_dim={cond_extra_dim}, param_dim={param_dim})")
        self.state_dim = state_dim
        self.param_dim = param_dim
        self.cond_extra_dim = cond_extra_dim
        self.NO, self.channels_per_sector = l96_sector_layout(NO, state_dim)
        self.unet2d = MonaiUNet2DSector(
            in_channels=self.channels_per_sector,
            out_channels=self.channels_per_sector,
            hidden_channels=hidden_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            dropout=dropout,
            use_flash_attention=use_flash_attention,
            mid_attention=mid_attention,
        )

    def forward(self, batch) -> torch.Tensor:
        obs = torch.nan_to_num(batch.obs, nan=0.0)
        tau = torch.zeros(obs.shape[0], device=obs.device)
        out = self.unet2d(to_sector_grid(obs, self.NO), tau=tau)
        return from_sector_grid(out)
