import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class DynamicsBase(ABC, nn.Module):
    state_dim: int
    param_names: list[str]
    param_dim: int

    def __init__(self):
        super().__init__()

    @abstractmethod
    def step(self, state: torch.Tensor, forcing: torch.Tensor,
             *args, **kwargs) -> torch.Tensor:
        pass

    def rollout(self, x0: torch.Tensor, forcing: torch.Tensor,
                steps: int, *args, **kwargs) -> torch.Tensor:
        traj = [x0]
        for t in range(1, steps):
            next_s = self.step(traj[-1], forcing[..., t - 1], *args, **kwargs)
            traj.append(next_s)
        return torch.stack(traj, dim=-2)


def get_dynamics(cfg) -> DynamicsBase:
    system = cfg.data.get("system", "lorenz63")
    if system == "lorenz63":
        from models.lorenz63_dynamics import Lorenz63Dynamics
        dc = cfg.data
        return Lorenz63Dynamics(
            dt=dc.dt, coupling_type=dc.get("forcing_coupling", "linear"),
            c1=dc.c1,
        )
    raise ValueError(f"Unknown dynamical system: {system}")