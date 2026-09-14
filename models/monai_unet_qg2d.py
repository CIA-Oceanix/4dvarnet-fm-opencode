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


class MonaiUNet2DQGSolver(nn.Module):
    """QG-native backbone for `models.fourdvarnet.FourDVarNetSolver`
    (`unet_backbone="monai2d"`): merges the T (days) axis into the *channel*
    dimension (`T * channels_per_day` total input channels) instead of
    `MonaiUNet1D`'s convention of treating T as a downsampled 1D-sequence
    axis (which requires T divisible by the backbone's downsampling depth --
    QG's 30-day windows often aren't, see `train_qg_neural.py`'s
    `_pad_batch_for_monai1d`) or `MonaiDirectUNetQG`'s convention of folding
    T into the *batch* dimension (every day processed fully independently,
    no cross-day coupling at all). Here every day keeps its true `(ny, nx)`
    circular-conv grid, and the 2D backbone's ordinary channel-mixing lets
    it learn correlations across days too.

    Drop-in for `MonaiUNet1D`/`UNet1D` at every `FourDVarNetSolver` call
    site (`_build_backbone_unet`'s `use_obs=False` convention: FDV's own
    channel-concat, e.g. `cat([x, obs], dim=-1)` for `update_input=
    "obs+state"`, is already baked into the incoming channel axis before
    this class ever sees it): `forward(x, tau=...)` takes `x` shaped
    `(B, C, T)` (channel-first, sequence-last -- `MonaiUNet1D`'s own
    convention) and returns the same `(B, C_out, T)` shape. `C` (`state_dim`/
    `output_dim` at construction) must be a whole multiple of `ny*nx` --
    `C // (ny*nx)` is the per-day channel count (e.g. `2*nlayers` for
    `update_input="obs+state"`'s `state`+`obs` concatenation, matching QG's
    `nlayers=2`-layer-major `D` layout exactly, the same one
    `MonaiDirectUNetQG.forward` reshapes directly into `(nlayers, ny, nx)`).

    `time_emb_dim` is accepted (for call-site parity with
    `_build_backbone_unet`'s uniform kwargs) but ignored, same as
    `MonaiUNet1D` -- MONAI's `DiffusionModelUNet` always carries its own
    internal time embedding.
    """

    def __init__(self, state_dim: int, T: int, ny: int, nx: int,
                 hidden_channels: list[int] | None = None,
                 num_res_blocks: int = 2, norm_num_groups: int = 8,
                 output_dim: int | None = None, dropout: float = 0.1,
                 time_emb_dim: int = 0):
        super().__init__()
        if state_dim % (ny * nx) != 0:
            raise ValueError(
                f"state_dim ({state_dim}) must be a whole multiple of "
                f"ny*nx ({ny}*{nx}={ny * nx}) for unet_backbone='monai2d'")
        output_dim = output_dim if output_dim is not None else state_dim
        if output_dim % (ny * nx) != 0:
            raise ValueError(
                f"output_dim ({output_dim}) must be a whole multiple of "
                f"ny*nx ({ny}*{nx}={ny * nx}) for unet_backbone='monai2d'")
        self.T = T
        self.ny = ny
        self.nx = nx
        self.in_ch_per_day = state_dim // (ny * nx)
        self.out_ch_per_day = output_dim // (ny * nx)
        self.backbone2d = MonaiUNet2DCircular(
            in_channels=T * self.in_ch_per_day, out_channels=T * self.out_ch_per_day,
            hidden_channels=hidden_channels, num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups, use_obs=False)
        if dropout > 0:
            from monai.networks.nets import diffusion_model_unet as _dmu
            for module in self.backbone2d.backbone.modules():
                if isinstance(module, _dmu.DiffusionUNetResnetBlock):
                    module.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, obs: torch.Tensor = None,
                tau: torch.Tensor = None) -> torch.Tensor:
        B, C, T = x.shape
        if T != self.T:
            raise ValueError(f"expected T={self.T} (days), got {T}")
        ch_per_day = C // (self.ny * self.nx)
        # (B, C, T) -> (B, T, C) -> (B, T*ch_per_day, ny, nx): the day axis
        # becomes additional channels, each day's C channels keep their
        # true (ny, nx) grid layout.
        x_grid = x.permute(0, 2, 1).reshape(B, T * ch_per_day, self.ny, self.nx)
        out = self.backbone2d(x_grid, tau=tau)  # (B, T*out_ch_per_day, ny, nx)
        out = out.reshape(B, T, self.out_ch_per_day * self.ny * self.nx)
        return out.permute(0, 2, 1)  # (B, C_out, T)


class MonaiDirectUNetQG(nn.Module):
    """MONAI-backed circular DirectUNet for QG, drop-in for
    `models.direct_unet.DirectUNet`.

    Same `forward(batch) -> (B, T, D)` contract (`batch.obs`/`batch.forcing`/
    `batch.params`/`batch.ic`, `D = nlayers*ny*nx` layer-major); each of the
    `T` days is folded into the batch dim for the 2D backbone (matches the
    existing per-day QG training loss -- no explicit temporal coupling
    term). `ic_dim` (Q5) is broadcast identically across all `T` days
    (one static field per window), unlike `forcing` (varies per day).
    """

    def __init__(self, ny: int, nx: int, nlayers: int = 2,
                 hidden_channels: list[int] | None = None,
                 param_dim: int = 0, cond_extra_dim: int = 0, ic_dim: int = 0,
                 num_res_blocks: int = 2, norm_num_groups: int = 8):
        super().__init__()
        self.ny = ny
        self.nx = nx
        self.nlayers = nlayers
        self.state_dim = nlayers * ny * nx
        self.param_dim = param_dim
        self.cond_extra_dim = cond_extra_dim
        self.ic_dim = ic_dim
        # Zeroed "state" input placeholder + obs conditioning, matching
        # MonaiDirectUNet/DirectUNet's convention (a CFM-architecture
        # artifact carried over for direct deterministic regression -- the
        # backbone's own use_obs branch does the actual channel concat).
        self.unet = MonaiUNet2DCircular(
            in_channels=nlayers, out_channels=nlayers,
            hidden_channels=hidden_channels, num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups, use_obs=True,
            obs_channels=nlayers + cond_extra_dim + param_dim + ic_dim)

    def forward(self, batch) -> torch.Tensor:
        obs = batch.obs
        B, T, D = obs.shape
        obs_clean = torch.nan_to_num(obs, nan=0.0)
        obs_grid = obs_clean.reshape(B * T, self.nlayers, self.ny, self.nx)
        cond = [obs_grid]
        if self.cond_extra_dim > 0:
            # `batch.forcing` is a genuine (B, T, ny, nx) spatial field (the
            # wind-curl forcing map, see data.qg_neural's Q3/Q4 conditioning
            # docstring) -- reshape into cond_extra_dim (=1) channel(s), no
            # spatial broadcast: unlike a scalar param, the forcing map's
            # spatial structure (storm location) is exactly the physically
            # meaningful signal.
            forcing = batch.forcing.reshape(B * T, self.cond_extra_dim, self.ny, self.nx)
            cond.append(forcing)
        if self.param_dim > 0:
            params_t = batch.params.unsqueeze(1).expand(B, T, -1).reshape(
                B * T, self.param_dim, 1, 1).expand(-1, -1, self.ny, self.nx)
            cond.append(params_t)
        if self.ic_dim > 0:
            # `batch.ic` (Q5) is one static (B, ic_dim*ny*nx) spatial field
            # per WINDOW (the initial-condition snapshot, see
            # data.qg_neural._ic_field) -- unlike `forcing` (varies per
            # day), it's broadcast identically across all T days, not
            # reshaped per-day.
            ic_grid = batch.ic.reshape(B, self.ic_dim, self.ny, self.nx)
            ic_t = ic_grid.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(
                B * T, self.ic_dim, self.ny, self.nx)
            cond.append(ic_t)
        cond = torch.cat(cond, dim=1) if len(cond) > 1 else cond[0]
        x = torch.zeros(B * T, self.nlayers, self.ny, self.nx, device=obs.device)
        tau = torch.zeros(B * T, device=obs.device)
        out = self.unet(x, obs=cond, tau=tau)  # (B*T, nlayers, ny, nx)
        return out.reshape(B, T, D)


class MonaiDirectUNetQGChannelTime(nn.Module):
    """DirectUNet-style single-pass QG estimator (like `MonaiDirectUNetQG`),
    but merges the T (days) axis into the *channel* dimension via
    `MonaiUNet2DQGSolver` instead of folding it into the *batch* dimension.
    `MonaiDirectUNetQG` processes every day fully independently (no
    cross-day coupling at all); this variant lets the single-pass
    regression see cross-day context through the 2D backbone's ordinary
    channel mixing, at the cost of a *fixed* `T` (the window length must
    match what the model was constructed with -- unlike `MonaiDirectUNetQG`,
    which tolerates any `T` since it never appears in a conv axis at all).

    Obs-only: no forcing/param/IC conditioning support (matching Q1's own
    `cond_mode="none"`) -- this variant isolates "does merging T into
    channels help the DirectUNet scheme too", not conditioning composition
    with cross-day coupling. `forward(batch) -> (B, T, D)`, same external
    contract as `MonaiDirectUNetQG` (`batch.obs` only; `batch.forcing`/
    `batch.params`/`batch.ic`, if non-trivial, are ignored -- same as
    passing `cond_extra_dim=param_dim=ic_dim=0` to `MonaiDirectUNetQG`).
    """

    def __init__(self, ny: int, nx: int, T: int, nlayers: int = 2,
                 hidden_channels: list[int] | None = None,
                 num_res_blocks: int = 2, norm_num_groups: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.ny = ny
        self.nx = nx
        self.T = T
        self.nlayers = nlayers
        self.state_dim = nlayers * ny * nx
        # Zeroed "state" input placeholder + obs conditioning, matching
        # MonaiDirectUNetQG's own convention -- concatenated along the
        # per-day channel block (2*nlayers total) before T is merged in.
        self.unet = MonaiUNet2DQGSolver(
            state_dim=2 * self.state_dim, T=T, ny=ny, nx=nx,
            hidden_channels=hidden_channels, output_dim=self.state_dim,
            dropout=dropout, norm_num_groups=norm_num_groups,
            num_res_blocks=num_res_blocks)

    def forward(self, batch) -> torch.Tensor:
        obs = batch.obs
        B, T, D = obs.shape
        if T != self.T:
            raise ValueError(f"expected T={self.T} (days), got {T}")
        obs_clean = torch.nan_to_num(obs, nan=0.0)
        x = torch.zeros_like(obs_clean)
        inp = torch.cat([x, obs_clean], dim=-1).transpose(1, 2)  # (B, 2D, T)
        out = self.unet(inp)  # (B, D, T)
        return out.transpose(1, 2)  # (B, T, D)
