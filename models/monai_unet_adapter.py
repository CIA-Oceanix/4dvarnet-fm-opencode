"""Adapter exposing MONAI's DiffusionModelUNet with the same call signature as
models.unet.UNet1D, for prototyping (see /homes/rfablet/.claude/plans/monai-diffunet-prototype.md).

monai is intentionally NOT in requirements.txt: monai==1.6.0 requires
torch==2.8.0+cu126, newer than this project's standard torch==2.4.1+cu121 (a
prior attempt to add monai to the shared env broke CUDA for everything else).
This module -- and anything importing it -- must run in a separate env built
from requirements-monai.txt (e.g. `fdv-monai-proto`), not the project's
default `fdv` env. Every other importer of this module already guards the
import (see evaluation/neural_inference.py's try/except) so the rest of the
codebase keeps working with the default env.

Verified directly against the installed monai==1.6.0 wheel:
- DiffusionModelUNet.forward(x, timesteps, context=None, class_labels=None, ...)
  takes a raw (N,) float tensor for `timesteps` and embeds it with a plain
  sinusoidal encoding (args = timesteps * freqs, max_period=10000) -- the same
  formula as this repo's own SinusoidalEmbedding. A continuous tau in [0, 1]
  is passed through directly; no num_train_timesteps / integer-step config
  needed.
- Constructor kwarg is `channels`, not `num_channels`.

Upstream bug patched here (not a MONAI config issue): monai.networks.nets.
diffusion_model_unet.DiffusionUNetResnetBlock.forward hardcodes the temb
broadcast as `[:, :, None, None]` for spatial_dims == 2 and unconditionally
`[:, :, None, None, None]` otherwise -- there is no spatial_dims == 1 branch,
so with spatial_dims=1 the addition broadcasts incorrectly and corrupts the
output shape (confirmed: forward crashes/produces garbage without this
patch). We monkeypatch only this one method, only when spatial_dims == 1 is
requested, rather than editing the installed package.
"""

import torch
import torch.nn as nn

from monai.networks.nets import diffusion_model_unet as _dmu
from monai.networks.nets import DiffusionModelUNet

from models.interpolant import LinearInterpolant
from models.sda import ConditionalPriorCFM, UnconditionalPriorCFM
from models.vanilla_cfm import VanillaCFM

_PATCHED_1D_RESBLOCK = False


def _patch_resblock_for_1d() -> None:
    global _PATCHED_1D_RESBLOCK
    if _PATCHED_1D_RESBLOCK:
        return

    def patched_forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)
        if self.upsample is not None:
            x = self.upsample(x)
            h = self.upsample(h)
        elif self.downsample is not None:
            x = self.downsample(x)
            h = self.downsample(h)
        h = self.conv1(h)
        if self.spatial_dims == 1:
            temb = self.time_emb_proj(self.nonlinearity(emb))[:, :, None]
        elif self.spatial_dims == 2:
            temb = self.time_emb_proj(self.nonlinearity(emb))[:, :, None, None]
        else:
            temb = self.time_emb_proj(self.nonlinearity(emb))[:, :, None, None, None]
        h = h + temb
        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.conv2(h)
        # MONAI's own DiffusionUNetResnetBlock has no dropout mechanism at all
        # (no constructor arg, no layer) -- unlike UNet1D.ConvBlock, which
        # applies real nn.Dropout before its residual add. `dropout` is
        # attached as a submodule post-construction (see MonaiUNet1D.__init__)
        # so this comparison isn't silently missing regularization.
        if hasattr(self, "dropout"):
            h = self.dropout(h)
        output: torch.Tensor = self.skip_connection(x) + h
        return output

    _dmu.DiffusionUNetResnetBlock.forward = patched_forward
    _PATCHED_1D_RESBLOCK = True


class MonaiUNet1D(nn.Module):
    """Drop-in-shaped replacement for models.unet.UNet1D backed by MONAI's
    DiffusionModelUNet, restricted to spatial_dims=1 with attention disabled
    (see plan: MONAI's attention reshape has no spatial_dims==1 branch
    either -- a second, separate gap from the ResBlock bug patched above).

    Conditioning: obs is concatenated onto the state along the channel axis
    before the backbone call (matching FDV's channel-concat convention),
    rather than using UNet1D's separate ConditionEncoder-style branch.
    """

    def __init__(
        self,
        state_dim: int = 3,
        obs_dim: int = None,
        hidden_channels: list = None,
        num_res_blocks: int = 2,
        norm_num_groups: int = 8,
        use_obs: bool = True,
        output_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        _patch_resblock_for_1d()

        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
        self.state_dim = state_dim
        self.obs_dim = obs_dim if obs_dim is not None else state_dim
        self.use_obs = use_obs
        self.output_dim = output_dim if output_dim is not None else state_dim

        in_channels = state_dim + (self.obs_dim if use_obs else 0)

        self.backbone = DiffusionModelUNet(
            spatial_dims=1,
            in_channels=in_channels,
            out_channels=self.output_dim,
            channels=tuple(hidden_channels),
            attention_levels=tuple(False for _ in hidden_channels),
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
        )
        if dropout > 0:
            for module in self.backbone.modules():
                if isinstance(module, _dmu.DiffusionUNetResnetBlock):
                    module.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        obs: torch.Tensor = None,
        x_tau: torch.Tensor = None,
        tau: torch.Tensor = None,
        energy_terms: list = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        inp = x
        if self.use_obs:
            if obs is None:
                raise ValueError("use_obs=True but obs is None")
            inp = torch.cat([x, obs], dim=1)
        timesteps = tau if tau is not None else torch.zeros(B, device=x.device, dtype=x.dtype)
        return self.backbone(inp, timesteps)


class MonaiDirectUNet(nn.Module):
    """MonaiUNet1D-backed drop-in for models.direct_unet.DirectUNet: same
    FlowMatchingBatch-taking forward() (obs/forcing/params -> single-pass
    state regression via a zeroed state input + channel-concat conditioning),
    so it plugs into train.py's model_type dispatch (model_factory /
    evaluate_model / save_trajectories) exactly where "direct_unet" does.
    """

    def __init__(self, state_dim=24, hidden_channels=None, dropout=0.1, param_dim=0,
                 cond_extra_dim=0, num_res_blocks=2, norm_num_groups=32):
        super().__init__()
        self.state_dim = state_dim
        self.param_dim = param_dim
        self.cond_extra_dim = cond_extra_dim
        if hidden_channels is None:
            hidden_channels = [64, 128, 256]
        self.monai_unet = MonaiUNet1D(
            state_dim=state_dim,
            obs_dim=state_dim + cond_extra_dim,
            hidden_channels=hidden_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            use_obs=True,
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
        out = self.monai_unet(x, obs=cond.transpose(1, 2), tau=tau)
        return out.transpose(1, 2)


class MonaiVanillaCFM(VanillaCFM):
    """MonaiUNet1D-backed drop-in for VanillaCFM: identical CFM training/
    sampling logic (inherited unchanged -- forward/compute_cfm_loss/sample
    only ever call ``self.unet(...)``, and MonaiUNet1D's forward has the same
    (x, obs=, tau=) shape as UNet1D's), only the backbone construction
    differs. Subclassing VanillaCFM (rather than composing) means
    ``isinstance(model, VanillaCFM)`` dispatch in evaluation/neural_inference.py
    picks this up automatically -- no eval-side changes needed.
    """

    def __init__(self, state_dim=3, hidden_channels=None, time_emb_dim=64,
                 N_outer=10, sigma_prior=0.5, dropout=0.1, train_tau_0_only=False,
                 param_dim=4, cond_extra_dim=0, num_res_blocks=2, norm_num_groups=32):
        nn.Module.__init__(self)
        self.cond_extra_dim = cond_extra_dim
        self.param_dim = param_dim
        self.unet = MonaiUNet1D(
            state_dim=state_dim,
            obs_dim=state_dim + cond_extra_dim,
            hidden_channels=hidden_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            use_obs=True,
            dropout=dropout,
        )
        self.interpolant = LinearInterpolant(nu=1.0)
        self.N_outer = N_outer
        self.sigma_prior = sigma_prior
        self.state_dim = state_dim
        self.train_tau_0_only = train_tau_0_only


class MonaiUnconditionalPriorCFM(UnconditionalPriorCFM):
    """MonaiUNet1D-backed drop-in for UnconditionalPriorCFM (SDA1): same
    unconditional p(x_1) prior contract, only the backbone construction
    differs. Subclassing keeps the ``isinstance(model, (UnconditionalPriorCFM,
    ConditionalPriorCFM))`` dispatch in evaluation/neural_inference.py and
    ``eval_sda_l96.py``'s guided-sampler path working unchanged.
    """

    def __init__(self, state_dim=3, hidden_channels=None, time_emb_dim=64,
                 N_outer=10, sigma_prior=0.5, dropout=0.1,
                 num_res_blocks=2, norm_num_groups=32):
        nn.Module.__init__(self)
        self.unet = MonaiUNet1D(
            state_dim=state_dim,
            hidden_channels=hidden_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            use_obs=False,
            dropout=dropout,
        )
        self.interpolant = LinearInterpolant(nu=1.0)
        self.N_outer = N_outer
        self.sigma_prior = sigma_prior
        self.state_dim = state_dim
        self.train_tau_0_only = False


class MonaiConditionalPriorCFM(ConditionalPriorCFM):
    """MonaiUNet1D-backed drop-in for ConditionalPriorCFM (SDA2/SDA3): same
    params+forcing-conditioned prior contract (obs never a network input,
    only ``_cond``'s forcing+params tensor), only the backbone construction
    differs.
    """

    def __init__(self, state_dim=3, param_dim=8, hidden_channels=None, time_emb_dim=64,
                 N_outer=10, sigma_prior=0.5, dropout=0.1,
                 num_res_blocks=2, norm_num_groups=32):
        nn.Module.__init__(self)
        self.param_dim = param_dim
        self.unet = MonaiUNet1D(
            state_dim=state_dim,
            obs_dim=1 + param_dim,
            hidden_channels=hidden_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            use_obs=True,
            dropout=dropout,
        )
        self.interpolant = LinearInterpolant(nu=1.0)
        self.N_outer = N_outer
        self.sigma_prior = sigma_prior
        self.state_dim = state_dim
        self.train_tau_0_only = False
