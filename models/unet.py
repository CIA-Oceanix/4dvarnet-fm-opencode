import torch
import torch.nn as nn
import math


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_c)
        self.conv1 = nn.Conv1d(in_c, out_c, 3, padding=1)
        self.norm2 = nn.LayerNorm(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.time_proj = nn.Linear(time_emb_dim, out_c) if time_emb_dim > 0 else None
        self.skip = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None) -> torch.Tensor:
        B, C, L = x.shape
        x_in = x
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x = self.conv1(x)
        x = nn.SiLU()(x)
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = self.conv2(x)
        if self.time_proj is not None and t_emb is not None:
            t = self.time_proj(t_emb)
            x = x + t.unsqueeze(-1)
        x = nn.SiLU()(x)
        x = self.drop(x)
        return x + self.skip(x_in)


class Down(nn.Module):
    def __init__(self, in_c: int, out_c: int, time_emb_dim: int):
        super().__init__()
        self.block = ConvBlock(in_c, out_c, time_emb_dim)
        self.pool = nn.AvgPool1d(2)

    def forward(self, x, t_emb=None):
        x = self.block(x, t_emb)
        return x, self.pool(x)

    def forward_pool(self, x, t_emb=None):
        return self.pool(self.block(x, t_emb))


class Up(nn.Module):
    def __init__(self, in_c: int, out_c: int, time_emb_dim: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.block = ConvBlock(in_c + out_c, out_c, time_emb_dim)

    def forward(self, x, skip, t_emb=None):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            diff = skip.shape[-1] - x.shape[-1]
            x = nn.functional.pad(x, (0, diff))
        x = torch.cat([x, skip], dim=1)
        return self.block(x, t_emb)


class ConditionEncoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, use_obs: bool, use_energy: bool,
                 obs_dim: int = None):
        super().__init__()
        self.use_obs = use_obs
        self.use_energy = use_energy
        self.obs_dim = obs_dim if obs_dim is not None else state_dim
        proj_in = state_dim
        if use_obs:
            proj_in += self.obs_dim
        if use_energy:
            proj_in += state_dim * 3
        self.proj = nn.Linear(proj_in, hidden_dim) if proj_in != hidden_dim else nn.Identity()

    def forward(self, x: torch.Tensor, obs: torch.Tensor, energy_terms: list = None) -> torch.Tensor:
        B, C, L = x.shape
        cond = x
        if self.use_obs:
            cond = torch.cat([cond, torch.nan_to_num(obs, nan=0.0)], dim=1)
        if self.use_energy and energy_terms is not None:
            for term in energy_terms:
                if isinstance(term, torch.Tensor):
                    if term.shape[1] != 0:
                        cond = torch.cat([cond, term], dim=1)
        cond = self.proj(cond.transpose(1, 2)).transpose(1, 2)
        return cond


class UNet1D(nn.Module):
    def __init__(
        self,
        state_dim: int = 3,
        hidden_channels: list = None,
        time_emb_dim: int = 64,
        use_obs: bool = True,
        use_energy: bool = False,
        dropout: float = 0.1,
        output_dim: int = None,
        obs_dim: int = None,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [64, 128, 256]

        self.state_dim = state_dim
        self.output_dim = output_dim if output_dim is not None else state_dim
        self.use_obs = use_obs
        self.use_energy = use_energy
        self.obs_dim = obs_dim if obs_dim is not None else state_dim
        self.time_emb_dim = time_emb_dim
        self.time_embed = SinusoidalEmbedding(time_emb_dim)

        self.cond_encoder = ConditionEncoder(state_dim, hidden_channels[0], use_obs, use_energy,
                                             obs_dim=self.obs_dim)

        self.enc_in = nn.Conv1d(hidden_channels[0], hidden_channels[0], 3, padding=1)

        self.downs = nn.ModuleList()
        in_c = hidden_channels[0]
        for out_c in hidden_channels:
            self.downs.append(Down(in_c, out_c, time_emb_dim))
            in_c = out_c

        self.bottleneck = ConvBlock(hidden_channels[-1], hidden_channels[-1], time_emb_dim, dropout)

        self.ups = nn.ModuleList()
        for out_c in reversed(hidden_channels):
            self.ups.append(Up(in_c, out_c, time_emb_dim))
            in_c = out_c

        self.enc_out = nn.Sequential(
            nn.Conv1d(in_c, in_c, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(in_c, self.output_dim, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        obs: torch.Tensor = None,
        x_tau: torch.Tensor = None,
        tau: torch.Tensor = None,
        energy_terms: list = None,
    ) -> torch.Tensor:
        B, C, L = x.shape

        t_emb = None
        if tau is not None:
            t_emb = self.time_embed(tau)

        cond = self.cond_encoder(x, obs, energy_terms)
        h = self.enc_in(cond)

        skips = []
        for down in self.downs:
            skip, h = down(h, t_emb)
            skips.append(skip)

        h = self.bottleneck(h, t_emb)

        for up in self.ups:
            s = skips.pop()
            h = up(h, s, t_emb)

        return self.enc_out(h)
