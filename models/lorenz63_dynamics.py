import torch
from models.dynamics import DynamicsBase


def _apply_coupling(W: torch.Tensor, c1: float, coupling_type: str) -> torch.Tensor:
    if coupling_type == "quartic":
        return c1 * torch.sign(W) * W ** 2
    return c1 * W


class Lorenz63Dynamics(DynamicsBase):
    state_dim = 3
    param_names = ["sigma", "rho", "beta"]
    param_dim = 3

    def __init__(self, dt: float = 0.01, coupling_type: str = "linear",
                 c1: float = 1.0, clip_range: float = 50.0):
        super().__init__()
        self.dt = dt
        self.c1 = c1
        self.clip_range = clip_range
        self.coupling_type = coupling_type

    def step(self, state: torch.Tensor, forcing: torch.Tensor,
             sigma, rho, beta) -> torch.Tensor:
        X, Y, Z = state[..., 0], state[..., 1], state[..., 2]
        W = forcing
        coupling = _apply_coupling(W, self.c1, self.coupling_type)
        dX = sigma * (Y - X) + coupling
        dY = X * (rho - Z) - Y
        dZ = X * Y - beta * Z
        next_s = torch.stack([
            X + dX * self.dt,
            Y + dY * self.dt,
            Z + dZ * self.dt,
        ], dim=-1)
        if self.clip_range is not None:
            next_s = torch.clamp(next_s, -self.clip_range, self.clip_range)
        return next_s

    def rollout_with_q(self, x0: torch.Tensor, q: torch.Tensor,
                        forcing: torch.Tensor, steps: int,
                        sigma, rho, beta) -> torch.Tensor:
        traj = [x0]
        for t in range(1, steps):
            next_s = self.step(traj[-1], forcing[..., t - 1], sigma, rho, beta)
            next_s = next_s + q[..., t, :]
            traj.append(next_s)
        return torch.stack(traj, dim=-2)