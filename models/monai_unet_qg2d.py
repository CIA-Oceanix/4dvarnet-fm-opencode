"""MONAI-backed, circular-padded 2D DirectUNet for the QG case study.

Same rationale as `models.monai_unet_adapter` (L96's `MonaiUNet1D`), but for
QG's genuinely 2D, doubly-periodic (ny, nx) grid (see `models.qg_dynamics.
QGDynamics`'s docstring: "double-periodic beta-plane channel") instead of
L96's 1D ring. `models.direct_unet.DirectUNet` never applies a spatial
convolution at all for QG -- it flattens the (ny, nx) field into a plain
channel axis and only convolves along the time/day axis.

MUST run in the `fdv-monai-proto` env (monai==1.6.0 needs torch==2.8.0+cu126;
see `models.monai_unet_adapter`'s docstring), not this project's default
`fdv` env.

Upstream gap patched here (not a MONAI config issue): `monai.networks.blocks.
Convolution` builds its conv layer via the `Conv[Conv.CONV, spatial_dims]`
factory with no `padding_mode` parameter anywhere in its own or
`DiffusionModelUNet`'s public API -- every conv is hardcoded to zero-padding,
which would leak/discard information across the domain's actual periodic
boundary. We register a replacement "conv" factory function (MONAI's own
public per-name-override mechanism, `LayerFactory.add_factory_callable` --
not a private monkeypatch) that swaps in a `padding_mode="circular"`
`nn.Conv2d` subclass for `spatial_dims=2` only; dims 1 and 3 fall back to the
stock `nn.Conv1d`/`nn.Conv3d` unchanged, so this cannot affect L96's
`MonaiUNet1D` (spatial_dims=1) sharing the same process.
"""
import torch
import torch.nn as nn

_PATCHED_CIRCULAR_CONV2D = False


class CircularConv2d(nn.Conv2d):
    def __init__(self, *args, padding_mode: str = "circular", **kwargs):
        super().__init__(*args, padding_mode=padding_mode, **kwargs)


def _patch_conv2d_circular() -> None:
    global _PATCHED_CIRCULAR_CONV2D
    if _PATCHED_CIRCULAR_CONV2D:
        return
    from monai.networks.layers.factories import Conv

    def _circular_conv_factory(dim: int):
        types = (nn.Conv1d, CircularConv2d, nn.Conv3d)
        return types[dim - 1]

    Conv.add_factory_callable("conv", _circular_conv_factory)
    _PATCHED_CIRCULAR_CONV2D = True


class MonaiUNet2DCircular(nn.Module):
    """Doubly-periodic 2D backbone: MONAI's `DiffusionModelUNet` with every
    conv circularly padded (see module docstring), attention disabled
    (unneeded for a deterministic direct-regression head; also sidesteps the
    same spatial_dims==1-only ResBlock temb-broadcast gap `MonaiUNet1D`
    patches -- not applicable here, but attention reshape logic in this MONAI
    version is likewise only exercised/tested for 2D/3D image use, so we
    leave it off to keep this a minimal, well-tested conv-only backbone).

    `forward(x, obs, tau)` mirrors `MonaiUNet1D`'s call shape, with `x`/`obs`
    as (B, C, ny, nx) grids instead of (B, C, T).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: list[int] | None = None,
                 num_res_blocks: int = 2, norm_num_groups: int = 8,
                 use_obs: bool = True, obs_channels: int | None = None):
        super().__init__()
        _patch_conv2d_circular()
        from monai.networks.nets import DiffusionModelUNet

        if hidden_channels is None:
            hidden_channels = [64, 128, 256]
        self.use_obs = use_obs
        self.obs_channels = obs_channels if obs_channels is not None else in_channels
        total_in = in_channels + (self.obs_channels if use_obs else 0)
        self.backbone = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=total_in,
            out_channels=out_channels,
            channels=tuple(hidden_channels),
            attention_levels=tuple(False for _ in hidden_channels),
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
        )

    def forward(self, x: torch.Tensor, obs: torch.Tensor = None,
                tau: torch.Tensor = None) -> torch.Tensor:
        inp = x
        if self.use_obs:
            if obs is None:
                raise ValueError("use_obs=True but obs is None")
            inp = torch.cat([x, obs], dim=1)
        B = x.shape[0]
        timesteps = tau if tau is not None else torch.zeros(
            B, device=x.device, dtype=x.dtype)
        return self.backbone(inp, timesteps)


class MonaiDirectUNetQG(nn.Module):
    """MONAI-backed circular DirectUNet for QG, drop-in for
    `models.direct_unet.DirectUNet`.

    Same `forward(batch) -> (B, T, D)` contract (`batch.obs`/`batch.forcing`/
    `batch.params`, `D = nlayers*ny*nx` layer-major); each of the `T` days is
    folded into the batch dim for the 2D backbone (matches the existing
    per-day QG training loss -- no explicit temporal coupling term).
    """

    def __init__(self, ny: int, nx: int, nlayers: int = 2,
                 hidden_channels: list[int] | None = None,
                 param_dim: int = 0, cond_extra_dim: int = 0,
                 num_res_blocks: int = 2, norm_num_groups: int = 8):
        super().__init__()
        self.ny = ny
        self.nx = nx
        self.nlayers = nlayers
        self.state_dim = nlayers * ny * nx
        self.param_dim = param_dim
        self.cond_extra_dim = cond_extra_dim
        # Zeroed "state" input placeholder + obs conditioning, matching
        # MonaiDirectUNet/DirectUNet's convention (a CFM-architecture
        # artifact carried over for direct deterministic regression -- the
        # backbone's own use_obs branch does the actual channel concat).
        self.unet = MonaiUNet2DCircular(
            in_channels=nlayers, out_channels=nlayers,
            hidden_channels=hidden_channels, num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups, use_obs=True,
            obs_channels=nlayers + cond_extra_dim + param_dim)

    def forward(self, batch) -> torch.Tensor:
        obs = batch.obs
        B, T, D = obs.shape
        obs_clean = torch.nan_to_num(obs, nan=0.0)
        obs_grid = obs_clean.reshape(B * T, self.nlayers, self.ny, self.nx)
        cond = [obs_grid]
        if self.cond_extra_dim > 0:
            forcing = batch.forcing.reshape(B * T, 1, 1, 1).expand(
                B * T, self.cond_extra_dim, self.ny, self.nx)
            cond.append(forcing)
        if self.param_dim > 0:
            params_t = batch.params.unsqueeze(1).expand(B, T, -1).reshape(
                B * T, self.param_dim, 1, 1).expand(-1, -1, self.ny, self.nx)
            cond.append(params_t)
        cond = torch.cat(cond, dim=1) if len(cond) > 1 else cond[0]
        x = torch.zeros(B * T, self.nlayers, self.ny, self.nx, device=obs.device)
        tau = torch.zeros(B * T, device=obs.device)
        out = self.unet(x, obs=cond, tau=tau)  # (B*T, nlayers, ny, nx)
        return out.reshape(B, T, D)
