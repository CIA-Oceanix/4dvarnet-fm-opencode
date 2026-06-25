import torch
import torch.nn as nn
from .unet import UNet1D


class IterativeUpdateCell(nn.Module):
    def __init__(
        self,
        state_dim: int = 3,
        hidden_channels: list = None,
        time_emb_dim: int = 64,
        use_obs: bool = True,
        use_energy: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_energy = use_energy
        input_dim = state_dim
        if use_obs:
            input_dim += state_dim
        if use_energy:
            input_dim += state_dim * 3

        self.net = UNet1D(
            state_dim=state_dim,
            hidden_channels=hidden_channels,
            time_emb_dim=time_emb_dim,
            use_obs=use_obs,
            use_energy=use_energy,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        obs: torch.Tensor,
        x_tau: torch.Tensor,
        tau: torch.Tensor,
        y_diff: torch.Tensor = None,
        phi_diff: torch.Tensor = None,
        bg_diff: torch.Tensor = None,
    ) -> torch.Tensor:
        energy_terms = None
        if self.use_energy:
            energy_terms = []
            if y_diff is not None:
                energy_terms.append(y_diff)
            else:
                energy_terms.append(torch.zeros_like(x))
            if phi_diff is not None:
                energy_terms.append(phi_diff)
            else:
                energy_terms.append(torch.zeros_like(x))
            if bg_diff is not None:
                energy_terms.append(bg_diff)
            else:
                energy_terms.append(torch.zeros_like(x))

        residual = self.net(x=x, obs=obs, x_tau=x_tau, tau=tau, energy_terms=energy_terms)
        return residual


class MeanEstimatorCell(nn.Module):
    def __init__(
        self,
        state_dim: int = 3,
        hidden_channels: list = None,
        time_emb_dim: int = 64,
        use_obs: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = UNet1D(
            state_dim=state_dim,
            hidden_channels=hidden_channels,
            time_emb_dim=time_emb_dim,
            use_obs=use_obs,
            use_energy=False,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, obs: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return self.net(x=x, obs=obs, tau=tau)
