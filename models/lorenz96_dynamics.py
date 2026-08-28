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
                 W_L_bar: float = 0.0, c2: float = 0.1, sigma_L: float = 0.20,
                 fast_weights=None):
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
        if fast_weights is not None:
            self.fast_weights = torch.tensor(fast_weights, dtype=torch.float32)
        else:
            self.fast_weights = None

    def _resolve(self, key, value):
        if value is None:
            return getattr(self, key)
        return value

    def _derivative(self, state, forcing, F, c1=None, h=None, hx=None, eps=None,
                    fast_weights=None):
        NO, J = self.NO, self.J
        h = self._resolve("h", h)
        hx = self._resolve("hx", hx)
        eps = self._resolve("eps", eps)
        c1 = self._resolve("c1", c1)
        X = state[..., :NO]
        Y = state[..., NO:].reshape(*state.shape[:-1], NO, J)
        if isinstance(F, torch.Tensor) and F.dim() == 1:
            F = F.view(-1, *([1] * (X.dim() - 1)))
        h_slow = h.view(-1, *([1] * (X.dim() - 1))) if isinstance(h, torch.Tensor) and h.dim() == 1 else h
        hx_fast = hx.view(-1, *([1] * (Y.dim() - 1))) if isinstance(hx, torch.Tensor) and hx.dim() == 1 else hx
        eps_fast = eps.view(-1, *([1] * (Y.dim() - 1))) if isinstance(eps, torch.Tensor) and eps.dim() == 1 else eps
        if isinstance(forcing, torch.Tensor) and forcing.dim() < X.dim():
            forcing = forcing.view(*forcing.shape, *([1] * (X.dim() - forcing.dim())))
        w = self._resolve("fast_weights", fast_weights)
        if w is not None:
            if isinstance(w, (list, tuple)):
                w = torch.tensor(w, dtype=torch.float32)
            w = w.to(Y.device)
            while w.dim() < Y.dim():
                w = w.unsqueeze(-2)
            Y_sum = (Y * w).sum(dim=-1)
        else:
            Y_sum = Y.sum(dim=-1)
        Xm1 = _periodic_shift(X, 1)
        Xp1 = _periodic_shift(X, -1)
        Xm2 = _periodic_shift(X, 2)
        adv_slow = Xm1 * (Xp1 - Xm2)
        coupling = _apply_coupling(forcing, c1, self.coupling_exponent)
        while coupling.dim() < X.dim():
            coupling = coupling.unsqueeze(-1)
        dX = adv_slow - X + F - h_slow * Y_sum + coupling
        Yp1 = _periodic_shift(Y, -1)
        Ym1 = _periodic_shift(Y, 1)
        Ym2 = _periodic_shift(Y, 2)
        adv_fast = Yp1 * (Ym1 - Ym2)
        dY = (adv_fast - Y + hx_fast * X.unsqueeze(-1)) / eps_fast
        return torch.cat([dX, dY.reshape(*state.shape[:-1], NO * J)], dim=-1)

    def _rk4_step(self, state, forcing, F, dt, **kw):
        k1 = self._derivative(state, forcing, F, **kw)
        k2 = self._derivative(state + 0.5 * dt * k1, forcing, F, **kw)
        k3 = self._derivative(state + 0.5 * dt * k2, forcing, F, **kw)
        k4 = self._derivative(state + dt * k3, forcing, F, **kw)
        next_s = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if self.clip_range is not None:
            next_s = torch.clamp(next_s, -self.clip_range, self.clip_range)
        return next_s

    def step(self, state: torch.Tensor, forcing: torch.Tensor,
             **kwargs) -> torch.Tensor:
        F = kwargs.get("F", 8.0)
        if isinstance(F, torch.Tensor):
            while F.dim() < state.dim():
                F = F.unsqueeze(-1)
        return self._rk4_step(state, forcing, F, self.dt,
                              c1=kwargs.get("c1"), h=kwargs.get("h"),
                              hx=kwargs.get("hx"), eps=kwargs.get("eps"),
                              fast_weights=kwargs.get("fast_weights"))

    def _forecast_loop(self, s0, forcing_arr, steps, F, **kw):
        s = s0
        for i in range(steps):
            s = self._rk4_step(s, forcing_arr[i], F, self.dt, **kw)
        return s

    def _forecast_loop_batch(self, s0_batch, forcing_arr, steps, F_batch, **kw):
        s = s0_batch
        for i in range(steps):
            s = self._rk4_step(s, forcing_arr[i], F_batch, self.dt, **kw)
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
                                  h=None, hx=None, eps=None,
                                  fast_weights=None,
                                  coupling_exponent: float = 1.6,
                                  spinup_steps: int = 10000) -> tuple:
        c1 = c1 if c1 is not None else self.c1
        c2 = c2 if c2 is not None else self.c2
        gamma = gamma if gamma is not None else self.gamma
        W_L_bar = W_L_bar if W_L_bar is not None else self.W_L_bar
        sigma_0 = sigma_0 if sigma_0 is not None else self.sigma_0
        sigma_L = sigma_L if sigma_L is not None else self.sigma_L
        h = self._resolve("h", h)
        hx = self._resolve("hx", hx)
        eps = self._resolve("eps", eps)

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
            s = self._rk4_step(s, W_t[i], F, self.dt, c1=c1, h=h, hx=hx, eps=eps,
                               fast_weights=fast_weights)

        traj_list = [s.clone()]
        for i in range(spinup_steps, total - 1):
            s = self._rk4_step(s, W_t[i], F, self.dt, c1=c1, h=h, hx=hx, eps=eps,
                               fast_weights=fast_weights)
            traj_list.append(s.clone())
        traj = torch.stack(traj_list)
        forcing_t = W_t[-num_steps:]
        return traj, forcing_t

    def generate_batch_trajectories(self, num_windows: int, num_steps: int,
                                  spinup_steps: int = 10000,
                                  F_values: torch.Tensor = None,
                                  c1_values: torch.Tensor = None,
                                  h_values: torch.Tensor = None,
                                  hx_values: torch.Tensor = None,
                                  eps_values: torch.Tensor = None,
                                  fast_weights_values: torch.Tensor = None,
                                  seed: int = 42,
                                  device=None) -> tuple:
        if device is None:
            device = torch.device("cpu")
        NO, J = self.NO, self.J
        sd = self.state_dim

        rng = np.random.RandomState(seed)
        W_arr = self._build_forcing(num_steps + spinup_steps, seed,
                                     self.c1, self.c2, self.gamma,
                                     self.W_L_bar, self.sigma_0, self.sigma_L,
                                     self.coupling_exponent)
        W_t = torch.tensor(W_arr, dtype=torch.float32, device=device)

        rng_np = np.random.RandomState(seed + 1)
        s0 = torch.tensor(np.concatenate([
            rng_np.randn(NO) * 0.01,
            rng_np.randn(NO * J) * 0.01,
        ]), dtype=torch.float32, device=device)

        if F_values is None:
            F_values = torch.full((num_windows,), 8.0, device=device)
        B = num_windows
        c1 = torch.full((B,), self.c1, device=device) if c1_values is None else c1_values
        h = torch.full((B,), self.h, device=device) if h_values is None else h_values
        hx = torch.full((B,), self.hx, device=device) if hx_values is None else hx_values
        eps = torch.full((B,), self.eps, device=device) if eps_values is None else eps_values
        if fast_weights_values is None:
            fw = self.fast_weights.to(device) if self.fast_weights is not None else None
        else:
            fw = torch.tensor(fast_weights_values, dtype=torch.float32).to(device)

        s = s0.unsqueeze(0).expand(B, -1).clone()
        F = F_values

        for i in range(spinup_steps):
            s = self._rk4_step(s, W_t[i].expand(B), F, self.dt, c1=c1, h=h, hx=hx, eps=eps,
                               fast_weights=fw)

        traj_list = [s.clone()]
        for i in range(spinup_steps, num_steps + spinup_steps):
            s = self._rk4_step(s, W_t[i].expand(B), F, self.dt, c1=c1, h=h, hx=hx, eps=eps,
                               fast_weights=fw)
            traj_list.append(s.clone())

        traj = torch.stack(traj_list, dim=1)
        forcing_t = W_t[-num_steps:].expand(B, -1)
        return traj, forcing_t

    def generate_batch_trajectories_seeded(
        self,
        num_steps: int,
        seeds: list,
        F_values: torch.Tensor,
        c1_values: torch.Tensor = None,
        h_values: torch.Tensor = None,
        hx_values: torch.Tensor = None,
        eps_values: torch.Tensor = None,
        fast_weights_values: torch.Tensor = None,
        spinup_steps: int = 10000,
        coupling_exponent: float = 1.6,
        device=None,
    ) -> tuple:
        """Vectorized batched trajectory generation with per-window seeding.

        Mirrors `generate_full_trajectory` per window but advances all windows
        in parallel. Each window i uses `seeds[i]` to build its own forcing
        series (via `_build_forcing`) and its own initial condition
        (`RandomState(seed+1)`), exactly matching the per-window path. Per-window
        params (F, c1, h, hx, eps, fast_weights) are passed as `(B,)` /
        `(B, J)` tensors.

        Returns (traj (B, num_steps, state_dim), forcing (B, num_steps)).
        """
        if device is None:
            device = torch.device("cpu")
        NO, J = self.NO, self.J
        B = len(seeds)
        total = num_steps + spinup_steps
        c1 = torch.full((B,), self.c1, device=device) if c1_values is None else c1_values
        h = torch.full((B,), self.h, device=device) if h_values is None else h_values
        hx = torch.full((B,), self.hx, device=device) if hx_values is None else hx_values
        eps = torch.full((B,), self.eps, device=device) if eps_values is None else eps_values
        if fast_weights_values is None:
            fw = self.fast_weights.to(device) if self.fast_weights is not None else None
        elif isinstance(fast_weights_values, torch.Tensor):
            fw = fast_weights_values.clone().detach().to(device)
        else:
            fw = torch.tensor(fast_weights_values, dtype=torch.float32).to(device)

        W_all = np.zeros((B, total), dtype=np.float32)
        s0_all = torch.zeros((B, NO + NO * J), dtype=torch.float32, device=device)
        for j, seed in enumerate(seeds):
            W_all[j] = self._build_forcing(
                total, seed, self.c1, self.c2, self.gamma,
                self.W_L_bar, self.sigma_0, self.sigma_L, coupling_exponent,
            )
            rng_ic = np.random.RandomState(seed + 1)
            s0_all[j] = torch.tensor(
                np.concatenate([rng_ic.randn(NO) * 0.01, rng_ic.randn(NO * J) * 0.01]),
                dtype=torch.float32, device=device,
            )
        W_t = torch.tensor(W_all, dtype=torch.float32, device=device)

        s = s0_all
        for i in range(spinup_steps):
            s = self._rk4_step(s, W_t[:, i], F_values, self.dt,
                               c1=c1, h=h, hx=hx, eps=eps, fast_weights=fw)

        traj_list = [s.clone()]
        for i in range(spinup_steps, total - 1):
            s = self._rk4_step(s, W_t[:, i], F_values, self.dt,
                               c1=c1, h=h, hx=hx, eps=eps, fast_weights=fw)
            traj_list.append(s.clone())

        traj = torch.stack(traj_list, dim=1)
        forcing_t = W_t[:, -num_steps:]
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