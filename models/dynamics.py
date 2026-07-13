import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class DynamicsBase(ABC, nn.Module):
    state_dim: int
    param_names: list[str]
    param_dim: int
    forcing_dim: int = 1

    def __init__(self):
        super().__init__()

    @abstractmethod
    def step(self, state: torch.Tensor, forcing: torch.Tensor,
             *args, **kwargs) -> torch.Tensor:
        pass

    def generate_forcing(self, num_steps: int, seed: int = 42,
                         device=None, **params) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement generate_forcing"
        )

    def rollout(self, x0: torch.Tensor, forcing: torch.Tensor,
                steps: int, *args, **kwargs) -> tuple:
        traj = [x0]
        for t in range(1, steps):
            next_s = self.step(traj[-1], forcing[..., t - 1], *args, **kwargs)
            traj.append(next_s)
        return torch.stack(traj, dim=-2), forcing


def get_dynamics(cfg) -> DynamicsBase:
    system = cfg.data.get("system", "lorenz63")
    dc = cfg.data
    if system == "lorenz63":
        from models.lorenz63_dynamics import Lorenz63Dynamics
        exponent = dc.get("coupling_exponent_truth",
                         dc.get("coupling_exponent_da", 1.6))
        return Lorenz63Dynamics(
            dt=dc.dt, coupling_exponent=exponent,
            c1=dc.c1, sigma_0=dc.sigma_0, gamma=dc.gamma,
            W_L_bar=dc.W_L_bar, c2=dc.c2, sigma_L=dc.sigma_L,
        )
    if system == "lorenz96":
        from models.lorenz96_dynamics import Lorenz96Dynamics
        exponent = dc.get("coupling_exponent_truth",
                         dc.get("coupling_exponent_da", 1.6))
        return Lorenz96Dynamics(
            dt=dc.dt, coupling_exponent=exponent,
            c1=dc.c1, NO=dc.get("NO", 8), J=dc.get("J", 4),
            h=dc.get("h", 1.0), hx=dc.get("hx", 1.0), eps=dc.get("eps", 0.1),
            sigma_0=dc.sigma_0, gamma=dc.gamma,
            W_L_bar=dc.W_L_bar, c2=dc.c2, sigma_L=dc.sigma_L,
        )
    if system == "shallow_water":
        from models.shallow_water_dynamics import ShallowWaterDynamics
        return ShallowWaterDynamics(
            Nx=dc.get("Nx", 64), Ny=dc.get("Ny", 64),
            dt=dc.dt, K=dc.get("K", 5),
            tau0=dc.get("tau0", 0.08), f_cor=dc.get("f_cor", 0.1),
            g1=dc.get("g1", 0.02), g2=dc.get("g2", 0.01),
            coupling=dc.get("coupling", 0.05),
            friction=dc.get("friction", 0.1),
            viscosity=dc.get("viscosity", 0.001),
            land_mask_type=dc.get("land_mask_type", "none"),
        )
    raise ValueError(f"Unknown dynamical system: {system}")