import torch
from models.dynamics import DynamicsBase


def _apply_coupling(W: torch.Tensor, c1, exponent: float = 1.0) -> torch.Tensor:
    if isinstance(c1, torch.Tensor) and c1.dim() == 1:
        c1 = c1.view(-1, *([1] * (W.dim() - 1)))
    if exponent == 1.0:
        return c1 * W
    return c1 * torch.sign(W) * torch.abs(W) ** exponent


class Lorenz63Dynamics(DynamicsBase):
    state_dim = 3
    param_names = ["sigma", "rho", "beta"]
    param_dim = 3

    def __init__(self, dt: float = 0.01, coupling_exponent: float = 1.0,
                 c1: float = 1.0, clip_range: float = 50.0,
                 sigma_0: float = 0.08, gamma: float = 0.05,
                 W_L_bar: float = 0.0, c2: float = 0.1,
                 sigma_L: float = 0.20):
        super().__init__()
        self.dt = dt
        self.c1 = c1
        self.clip_range = clip_range
        self.coupling_exponent = coupling_exponent
        self.sigma_0 = sigma_0
        self.gamma = gamma
        self.W_L_bar = W_L_bar
        self.c2 = c2
        self.sigma_L = sigma_L

    def generate_full_trajectory(self, num_steps: int, seed: int = 42,
                                  device=None, sigma=10.0, rho=28.0,
                                  beta=8/3, c1=None, c2=None,
                                  W_L_bar=None, gamma=None,
                                  sigma_0=None, sigma_L=None,
                                  coupling_exponent: float = 1.6,
                                  spinup_steps: int = 10000) -> tuple:
        from data.lorenz63 import generate_long_trajectory
        c1 = c1 if c1 is not None else self.c1
        c2 = c2 if c2 is not None else self.c2
        gamma = gamma if gamma is not None else self.gamma
        W_L_bar = W_L_bar if W_L_bar is not None else self.W_L_bar
        sigma_0 = sigma_0 if sigma_0 is not None else self.sigma_0
        sigma_L = sigma_L if sigma_L is not None else self.sigma_L
        traj = generate_long_trajectory(
            num_steps=num_steps + spinup_steps, dt=self.dt, seed=seed,
            sigma=sigma, rho=rho, beta=beta,
            gamma=gamma, W_L_bar=W_L_bar, c1=c1, c2=c2,
            sigma_0=sigma_0, sigma_L=sigma_L,
            device=device, coupling_exponent=coupling_exponent,
        )
        seg = traj[-num_steps:].clone()
        return seg[:, :3], seg[:, 3]

    def step(self, state: torch.Tensor, forcing: torch.Tensor,
             sigma, rho, beta) -> torch.Tensor:
        X, Y, Z = state[..., 0], state[..., 1], state[..., 2]
        W = forcing
        coupling = _apply_coupling(W, self.c1, self.coupling_exponent)
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