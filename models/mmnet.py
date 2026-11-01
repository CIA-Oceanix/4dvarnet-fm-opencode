import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvPrior(nn.Module):
    def __init__(self, state_dim: int = 3, hidden_channels: int = 32, dilations=(1, 2, 4)):
        super().__init__()
        layers = []
        cin = state_dim
        for d in dilations:
            layers += [nn.Conv1d(cin, hidden_channels, 3, padding=d, dilation=d), nn.SiLU()]
            cin = hidden_channels
        for d in reversed(dilations[1:]):
            layers += [nn.Conv1d(cin, hidden_channels, 3, padding=d, dilation=d), nn.SiLU()]
        layers.append(nn.Conv1d(cin, state_dim, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class MMNet(nn.Module):
    def __init__(self, state_dim: int = 3, hidden_channels: int = 32, n_iter: int = 10,
                 step: float = 0.5, prior_weight: float = 1.0, init_std: float = 1e-3,
                 dilations=(1, 2, 4)):
        super().__init__()
        self.state_dim = state_dim
        self.n_iter = n_iter
        self.step = step
        self.prior_weight = prior_weight
        self.init_std = init_std
        self.prior = ConvPrior(state_dim, hidden_channels, tuple(dilations))

    def _cost(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        data = 0.5 * (mask * (y - x) ** 2).sum()
        prior = 0.5 * self.prior_weight * ((x - self.prior(x)) ** 2).sum()
        return data + prior

    def forward(self, batch) -> torch.Tensor:
        obs = batch.obs
        mask = (~torch.isnan(obs)).to(obs.dtype)
        y = torch.nan_to_num(obs, nan=0.0)
        x = self.init_std * torch.randn_like(y)
        create_graph = self.training and torch.is_grad_enabled()
        with torch.enable_grad():
            for _ in range(self.n_iter):
                if not create_graph:
                    x = x.detach()
                x = x.requires_grad_(True) if not x.requires_grad else x
                (g,) = torch.autograd.grad(self._cost(x, y, mask), x, create_graph=create_graph)
                x = x - self.step * g
        return x if create_graph else x.detach()

    def compute_loss(self, batch) -> torch.Tensor:
        x_hat = self(batch)
        rep = self.prior(batch.states)
        return F.mse_loss(x_hat, batch.states) + F.mse_loss(rep, batch.states)
