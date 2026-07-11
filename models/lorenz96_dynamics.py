import torch
import numpy as np
from models.dynamics import DynamicsBase


def _apply_coupling(W: torch.Tensor, c1, exponent: float = 1.0) -> torch.Tensor:
    if isinstance(c1, torch.Tensor) and c1.dim() == 1:
        c1 = c1.view(-1, *([1] * (W.dim() - 1)))
    if exponent == 1.0:
        return c1 * W
    return c1 * torch.sign(W) * torch.abs(W) ** exponent


def _periodic_shift(X, shift):
    if isinstance(X, torch.Tensor):
        return torch.roll(X, shifts=shift, dims=-1)
    return np.roll(X, shift, axis=-1)


class Lorenz96Dynamics(DynamicsBase):
    state_dim: int
    param_names = ["F"]
    param_dim = 1
    forcing_dim = 1

    def __init__(self, dt: float = 0.001, coupling_exponent: float = 1.0,
                 c1: float = 1.0, clip_range: float = 50.0,
                 NO: int = 8, J: int = 4, h: float = 1.0, hx: float = 1.0,
                 eps: float = 0.1, sigma_0: float = 0.08, gamma: float = 0.05,
                 W_L_bar: float = 0.0, c2: float = 0.1, sigma_L: float = 0.20):
        super().__init__()
        self.dt = dt
        self.c1 = c1
        self.clip_range = clip_range
        self.coupling_exponent = coupling_exponent
        self.NO = NO
        self.J = J
        self.h = h
        self.hx = hx
        self.eps = eps
        self.state_dim = NO + NO * J
        self.sigma_0 = sigma_0
        self.gamma = gamma
        self.W_L_bar = W_L_bar
        self.c2 = c2
        self.sigma_L = sigma_L

    def _derivative(self, state, forcing, F):
        NO, J, h, hx, eps = self.NO, self.J, self.h, self.hx, self.eps
        X = state[..., :NO]
        Y = state[..., NO:].reshape(*state.shape[:-1], NO, J)
        Y_sum = Y.sum(dim=-1)
        Xm1 = _periodic_shift(X, 1)
        Xp1 = _periodic_shift(X, -1)
        Xm2 = _periodic_shift(X, 2)
        adv_slow = Xm1 * (Xp1 - Xm2)
        coupling = _apply_coupling(forcing, self.c1, self.coupling_exponent)
        dX = adv_slow - X + F - h * Y_sum + coupling
        Yp1 = _periodic_shift(Y, -1)
        Ym1 = _periodic_shift(Y, 1)
        Ym2 = _periodic_shift(Y, 2)
        adv_fast = Yp1 * (Ym1 - Ym2)
        dY = (adv_fast - Y + hx * X.unsqueeze(-1)) / eps
        return torch.cat([dX, dY.reshape(*state.shape[:-1], NO * J)], dim=-1)

    def _rk4_step(self, state, forcing, F, dt):
        k1 = self._derivative(state, forcing, F)
        k2 = self._derivative(state + 0.5 * dt * k1, forcing, F)
        k3 = self._derivative(state + 0.5 * dt * k2, forcing, F)
        k4 = self._derivative(state + dt * k3, forcing, F)
        next_s = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if self.clip_range is not None:
            next_s = torch.clamp(next_s, -self.clip_range, self.clip_range)
        return next_s

    def step(self, state: torch.Tensor, forcing: torch.Tensor,
             **kwargs) -> torch.Tensor:
        F = kwargs.get("F", 8.0)
        return self._rk4_step(state, forcing, F, self.dt)

    def _forecast_loop(self, s0, forcing_arr, steps, F):
        s = s0
        for i in range(steps):
            s = self._rk4_step(s, forcing_arr[i], F, self.dt)
        return s

    def _build_forcing(self, length, seed, c1, c2, gamma, W_L_bar, sigma_0, sigma_L, coupling_exponent):
        rng = np.random.RandomState(seed)
        W_raw = rng.randn(length) * sigma_0
        W_AR = np.zeros(length)
        for i in range(1, length):
            W_AR[i] = gamma * W_AR[i - 1] + np.sqrt(1 - gamma ** 2) * W_raw[i]
        W_AR += W_L_bar
        W_AR += c2 * np.sin(np.arange(length) * 2 * np.pi / 80.0)
        W_arr = c1 * np.sign(W_AR) * np.abs(W_AR) ** coupling_exponent
        return W_arr

    def generate_full_trajectory(self, num_steps: int, seed: int = 42,
                                  device=None, F=8.0, c1=None, c2=None,
                                  W_L_bar=None, gamma=None,
                                  sigma_0=None, sigma_L=None,
                                  coupling_exponent: float = 1.6,
                                  spinup_steps: int = 10000) -> tuple:
        c1 = c1 if c1 is not None else self.c1
        c2 = c2 if c2 is not None else self.c2
        gamma = gamma if gamma is not None else self.gamma
        W_L_bar = W_L_bar if W_L_bar is not None else self.W_L_bar
        sigma_0 = sigma_0 if sigma_0 is not None else self.sigma_0
        sigma_L = sigma_L if sigma_L is not None else self.sigma_L

        total = num_steps + spinup_steps
        W_arr = self._build_forcing(total, seed, c1, c2, gamma, W_L_bar, sigma_0, sigma_L, coupling_exponent)

        rng = np.random.RandomState(seed + 1)
        s0 = torch.tensor(np.concatenate([
            rng.randn(self.NO) * 0.01,
            rng.randn(self.NO * self.J) * 0.01,
        ]), dtype=torch.float32)

        W_t = torch.tensor(W_arr, dtype=torch.float32)
        s = s0
        for i in range(spinup_steps):
            s = self._rk4_step(s, W_t[i], F, self.dt)

        traj_list = [s.clone()]
        for i in range(spinup_steps, total - 1):
            s = self._rk4_step(s, W_t[i], F, self.dt)
            traj_list.append(s.clone())
        traj = torch.stack(traj_list)
        forcing_t = W_t[-num_steps:]
        return traj, forcing_t

    def rollout_with_q(self, x0: torch.Tensor, q: torch.Tensor,
                        forcing: torch.Tensor, steps: int,
                        **kwargs) -> torch.Tensor:
        traj = [x0]
        for t in range(1, steps):
            next_s = self.step(traj[-1], forcing[..., t - 1], **kwargs)
            next_s = next_s + q[..., t, :]
            traj.append(next_s)
        return torch.stack(traj, dim=-2)