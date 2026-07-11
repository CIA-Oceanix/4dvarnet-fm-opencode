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

    def _slow_derivative(self, X, Y_sum, forcing, F):
        Xm1 = _periodic_shift(X, 1)
        Xp1 = _periodic_shift(X, -1)
        Xm2 = _periodic_shift(X, 2)
        adv = Xm1 * (Xp1 - Xm2)
        W = forcing
        coupling = _apply_coupling(W, self.c1, self.coupling_exponent)
        return adv - X + F - self.h * Y_sum + coupling

    def _fast_derivative(self, Y, X_k):
        Yp1 = _periodic_shift(Y, -1)
        Ym1 = _periodic_shift(Y, 1)
        Ym2 = _periodic_shift(Y, 2)
        adv = Yp1 * (Ym1 - Ym2)
        return (adv - Y + self.hx * X_k) / self.eps

    def _slow_derivative_np(self, X, Y_sum, forcing, F):
        Xm1 = _periodic_shift(X, 1)
        Xp1 = _periodic_shift(X, -1)
        Xm2 = _periodic_shift(X, 2)
        adv = Xm1 * (Xp1 - Xm2)
        W = forcing
        coupling = _apply_coupling(torch.from_numpy(W), self.c1, self.coupling_exponent)
        return adv + coupling.numpy() - X + F - self.h * Y_sum

    def _fast_derivative_np(self, Y, X_k):
        Yp1 = _periodic_shift(Y, -1)
        Ym1 = _periodic_shift(Y, 1)
        Ym2 = _periodic_shift(Y, 2)
        adv = Yp1 * (Ym1 - Ym2)
        return (adv - Y + self.hx * X_k) / self.eps

    def _rk4_step(self, state, forcing, F, dt):
        NO, J = self.NO, self.J

        def derivs(s):
            x = s[:NO]
            y = s[NO:].reshape(NO, J)
            ys = y.sum(axis=-1)
            dx = self._slow_derivative(x, ys, forcing, F)
            dy = torch.zeros(NO, J, device=state.device, dtype=state.dtype)
            for k in range(NO):
                dy[k] = self._fast_derivative(y[k], x[k])
            return torch.cat([dx, dy.reshape(-1)])

        k1 = derivs(state)
        k2 = derivs(state + 0.5 * dt * k1)
        k3 = derivs(state + 0.5 * dt * k2)
        k4 = derivs(state + dt * k3)
        next_s = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if self.clip_range is not None:
            next_s = torch.clamp(next_s, -self.clip_range, self.clip_range)
        return next_s

    def step(self, state: torch.Tensor, forcing: torch.Tensor,
             **kwargs) -> torch.Tensor:
        F = kwargs.get("F", 8.0)
        return self._rk4_step(state, forcing, F, self.dt)

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
        rng = np.random.RandomState(seed)

        X0 = rng.randn(self.NO) * 0.01
        Y0 = rng.randn(self.NO * self.J) * 0.01
        s0_np = np.concatenate([X0, Y0])

        s0_t = torch.from_numpy(s0_np).float()
        spin_trajs = []
        s = s0_t
        total = num_steps + spinup_steps

        W_raw = rng.randn(total) * sigma_0
        W_AR = np.zeros(total)
        for i in range(1, total):
            W_AR[i] = gamma * W_AR[i - 1] + np.sqrt(1 - gamma ** 2) * W_raw[i]
        W_AR += W_L_bar
        W_AR += c2 * np.sin(np.arange(total) * 2 * np.pi / 80.0)
        W_arr = c1 * np.sign(W_AR) * np.abs(W_AR) ** coupling_exponent

        for i in range(total):
            force = torch.tensor(W_arr[i], dtype=s.dtype, device=device)
            s = self._rk4_step(s, force, F, self.dt)
            if i >= spinup_steps - 1:
                spin_trajs.append(s.clone())

        traj = torch.stack(spin_trajs)
        forcing_t = torch.tensor(W_arr[-num_steps:], dtype=s.dtype, device=device)
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