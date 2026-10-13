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

from models.interpolant import LinearInterpolant
from models.monai_unet_adapter import MonaiDirectUNet, _patch_resblock_for_1d
from models.vanilla_cfm import VanillaCFM


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


class MonaiUNet2DSectorSolver(nn.Module):
    """2D (time x sector) backbone for `models.fourdvarnet.FourDVarNetSolver`
    (`unet_backbone="monai_sector2d"`), drop-in for `MonaiUNet1D` at every
    FDV call site: `forward(x, tau=...)` maps `(B, C, T)` to `(B, C_out, T)`.

    FDV concatenates `C // block_dim` state-sized blocks along channels (e.g.
    `[(y - x) * m, x - Phi(x), x, m]` for "subgrad+mask+state"); each
    `block_dim`-sized block is laid out on the sector grid by
    `to_sector_grid`, so sector k of the backbone sees `C // block_dim`
    blocks of `[X_k, Y_k0, ..., Y_k(J_obs-1)]` and the network stays exactly
    equivariant to cyclic sector shifts. `output_init_std` mirrors
    `MonaiUNet1D` (only used for a residual prior)."""

    def __init__(self, state_dim: int, block_dim: int, NO: int = 8,
                 hidden_channels: list[int] | None = None,
                 num_res_blocks: int = 2, norm_num_groups: int = 32,
                 output_dim: int | None = None, dropout: float = 0.1,
                 use_flash_attention: bool = True, mid_attention: str = "time",
                 output_init_std: float = 0.0):
        super().__init__()
        output_dim = output_dim if output_dim is not None else state_dim
        for name, dim in (("state_dim", state_dim), ("output_dim", output_dim)):
            if dim % block_dim != 0:
                raise ValueError(f"{name}={dim} is not a multiple of block_dim={block_dim}")
        self.block_dim = block_dim
        self.NO, self.channels_per_sector = l96_sector_layout(NO, block_dim)
        self.in_blocks = state_dim // block_dim
        self.out_blocks = output_dim // block_dim
        self.unet2d = MonaiUNet2DSector(
            in_channels=self.in_blocks * self.channels_per_sector,
            out_channels=self.out_blocks * self.channels_per_sector,
            hidden_channels=hidden_channels, num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups, dropout=dropout,
            use_flash_attention=use_flash_attention, mid_attention=mid_attention)
        if output_init_std > 0:
            final_conv = self.unet2d.backbone.out[-1]
            conv = final_conv.conv if hasattr(final_conv, "conv") else final_conv
            nn.init.normal_(conv.weight, mean=0.0, std=output_init_std)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def _to_grid(self, x: torch.Tensor, n_blocks: int) -> torch.Tensor:
        B, _, T = x.shape
        v = x.reshape(B, n_blocks, self.block_dim, T).permute(0, 1, 3, 2)
        g = to_sector_grid(v.reshape(B * n_blocks, T, self.block_dim), self.NO)
        return g.reshape(B, n_blocks * self.channels_per_sector, T, self.NO)

    def _from_grid(self, g: torch.Tensor, n_blocks: int) -> torch.Tensor:
        B, _, T, NO = g.shape
        v = from_sector_grid(g.reshape(B * n_blocks, self.channels_per_sector, T, NO))
        return v.reshape(B, n_blocks, T, self.block_dim).permute(0, 1, 3, 2).reshape(
            B, n_blocks * self.block_dim, T)

    def forward(self, x: torch.Tensor, obs: torch.Tensor = None,
                tau: torch.Tensor = None) -> torch.Tensor:
        out = self.unet2d(self._to_grid(x, self.in_blocks), tau=tau)
        return self._from_grid(out, self.out_blocks)


class SectorObsConditionedUNet(nn.Module):
    """`(x, obs, tau)` -> velocity on the sector grid: `[x, obs]` enter as two
    state-sized blocks of `MonaiUNet2DSectorSolver`, matching the
    `UNet1D`/`MonaiUNet1D` `use_obs=True` call signature used by the CFMs."""

    def __init__(self, state_dim: int = 24, NO: int = 8, **backbone_kwargs):
        super().__init__()
        self.backbone = MonaiUNet2DSectorSolver(state_dim=2 * state_dim, block_dim=state_dim,
                                                NO=NO, output_dim=state_dim, **backbone_kwargs)

    def forward(self, x: torch.Tensor, obs: torch.Tensor, tau: torch.Tensor = None) -> torch.Tensor:
        return self.backbone(torch.cat([x, obs], dim=1), tau=tau)


class MonaiVanillaCFM2D(VanillaCFM):
    """`MonaiVanillaCFM` with the 2D (time x sector) circular backbone: the
    same CFM loss and sampler (inherited), the velocity network replaced by
    `SectorObsConditionedUNet`. Subclassing keeps the `isinstance(model,
    VanillaCFM)` eval dispatch. Observation-only conditioning, as for
    `MonaiDirectUNet2D`."""

    def __init__(self, state_dim: int = 24, NO: int = 8, hidden_channels=None,
                 N_outer: int = 10, sigma_prior: float = 0.5, dropout: float = 0.1,
                 train_tau_0_only: bool = False, param_dim: int = 0, cond_extra_dim: int = 0,
                 num_res_blocks: int = 2, norm_num_groups: int = 32,
                 use_flash_attention: bool = True, mid_attention: str = "time"):
        nn.Module.__init__(self)
        if cond_extra_dim:
            raise ValueError(f"MonaiVanillaCFM2D supports obs-only conditioning (got cond_extra_dim={cond_extra_dim})")
        self.cond_extra_dim = 0
        self.param_dim = param_dim
        self.unet = SectorObsConditionedUNet(
            state_dim=state_dim, NO=NO, hidden_channels=hidden_channels,
            num_res_blocks=num_res_blocks, norm_num_groups=norm_num_groups, dropout=dropout,
            use_flash_attention=use_flash_attention, mid_attention=mid_attention)
        self.interpolant = LinearInterpolant(nu=1.0)
        self.N_outer = N_outer
        self.sigma_prior = sigma_prior
        self.state_dim = state_dim
        self.train_tau_0_only = train_tau_0_only
