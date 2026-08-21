import math

import torch

from models.dynamics import DynamicsBase


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if dtype == torch.float64 else torch.complex64


class QGDynamics(DynamicsBase):
    """Two-layer Phillips QG model on a double-periodic beta-plane channel.

    Torch port of pyqg.QGModel (https://github.com/pyqg/pyqg, v0.4.0 formulation):
        q1 = lap psi1 + F1 (psi2 - psi1),  q2 = lap psi2 + F2 (psi1 - psi2)
        d q1/dt = -J(psi1, q1) - Qy1 psi1_x
        d q2/dt = -J(psi2, q2) - Qy2 psi2_x + rek lap psi2
    with Qy1 = beta + F1 (U1 - U2), Qy2 = beta - F2 (U1 - U2),
    F1 = rd^-2 / (1 + delta), F2 = delta F1.
    Advection uses the flux form -(ik fft(u q) + il fft(v q)) with total
    zonal velocity u + U_k, as in pyqg. Time stepping is RK4 (pyqg uses AB3)
    with the pyqg exponential filter applied once per step.

    State layout: flattened [..., 2 * ny * nx], layer-major row-major grids,
    layer 0 first. The system is autonomous; the `forcing` argument of
    step/rollout is accepted for DynamicsBase compatibility and ignored.
    """

    param_dim = 5
    forcing_dim = 1

    def __init__(self, nx: int = 64, ny: int | None = None,
                 L: float = 1e6, W: float | None = None, dt: float = 7200.0,
                 beta: float = 1.5e-11, rd: float = 15000.0, delta: float = 0.25,
                 U1: float = 0.025, U2: float = 0.0, rek: float = 5.787e-7,
                 filterfac: float = 23.6, clip_range: float | None = None,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.param_names: list[str] = ["beta", "rd", "rek", "U1", "U2"]
        if ny is None:
            ny = nx
        if nx % 2 or ny % 2:
            raise ValueError("QGDynamics requires even nx and ny")
        self.nx = int(nx)
        self.ny = int(ny)
        self.L = float(L)
        self.W = float(W) if W is not None else float(L)
        self.dt = float(dt)
        self.beta = float(beta)
        self.rd = float(rd)
        self.delta = float(delta)
        self.U1 = float(U1)
        self.U2 = float(U2)
        self.rek = float(rek)
        self.filterfac = float(filterfac)
        self.clip_range = clip_range
        self.state_dim = 2 * self.ny * self.nx
        self.dtype = dtype

        nk = self.nx // 2 + 1
        dk = 2.0 * math.pi / self.L
        dl = 2.0 * math.pi / self.W
        kk = dk * torch.arange(nk, dtype=torch.float64)
        ll = dl * torch.cat([torch.arange(0.0, self.ny // 2, dtype=torch.float64),
                             torch.arange(-self.ny // 2, 0.0, dtype=torch.float64)])
        l2d, k2d = torch.meshgrid(ll, kk, indexing="ij")
        K2 = k2d ** 2 + l2d ** 2

        F1 = self.rd ** -2 / (1.0 + self.delta)
        F2 = self.delta * F1
        self.F1 = F1
        self.F2 = F2
        det_inv = torch.where(K2 > 0, 1.0 / (K2 * (K2 + F1 + F2)),
                              torch.zeros_like(K2))
        a11 = -(K2 + F2) * det_inv
        a12 = -F1 * det_inv
        a21 = -F2 * det_inv
        a22 = -(K2 + F1) * det_inv

        cphi = 0.65 * math.pi
        dx = self.L / self.nx
        dy = self.W / self.ny
        wvx = torch.sqrt((k2d * dx) ** 2 + (l2d * dy) ** 2)
        filtr = torch.exp(-self.filterfac * (wvx - cphi) ** 4)
        filtr = torch.where(wvx <= cphi, torch.ones_like(filtr), filtr)

        cdtype = _complex_dtype(dtype)
        self.register_buffer("K2", K2.to(dtype))
        self.register_buffer("a11", a11.to(dtype))
        self.register_buffer("a12", a12.to(dtype))
        self.register_buffer("a21", a21.to(dtype))
        self.register_buffer("a22", a22.to(dtype))
        self.register_buffer("filtr", filtr.to(dtype))
        self.register_buffer("ik", (1j * k2d).to(cdtype))
        self.register_buffer("il", (1j * l2d).to(cdtype))

    @property
    def device(self) -> torch.device:
        return self.K2.device

    def to(self, device=None, dtype=None):
        if dtype is not None and dtype != self.dtype:
            raise ValueError("QGDynamics dtype is fixed at construction")
        return super().to(device=device)

    def _grid(self, state: torch.Tensor) -> torch.Tensor:
        shape = state.shape[:-1] + (2, self.ny, self.nx)
        return state.reshape(shape)

    def _flatten(self, q: torch.Tensor) -> torch.Tensor:
        return q.reshape(*q.shape[:-3], self.state_dim)

    def _initial_q(self, batch_size: int, seed: int,
                   device: torch.device) -> torch.Tensor:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        q1 = (1e-7 * torch.rand((batch_size, self.ny, self.nx), generator=gen)
              + 1e-6 * torch.rand((batch_size, 1, self.nx), generator=gen))
        q2 = torch.zeros((batch_size, self.ny, self.nx))
        q = torch.stack([q1, q2], dim=-3)
        q = q - q.mean(dim=(-2, -1), keepdim=True)
        return q.to(device=device, dtype=self.dtype)

    def _invert(self, qh: torch.Tensor) -> torch.Tensor:
        ph = self.a11 * qh[..., 0, :, :] + self.a12 * qh[..., 1, :, :]
        ph2 = self.a21 * qh[..., 0, :, :] + self.a22 * qh[..., 1, :, :]
        return torch.stack([ph, ph2], dim=-3)

    def _tendency(self, qh: torch.Tensor, U1: float, U2: float,
                  beta: float, rek: float) -> torch.Tensor:
        q = torch.fft.irfft2(qh, s=(self.ny, self.nx), dim=(-2, -1))
        ph = self._invert(qh)
        u = torch.fft.irfft2(-self.il * ph, s=(self.ny, self.nx), dim=(-2, -1))
        v = torch.fft.irfft2(self.ik * ph, s=(self.ny, self.nx), dim=(-2, -1))
        Ubg = torch.tensor([U1, U2], device=q.device, dtype=q.dtype)
        uq = (u + Ubg.view(1, 2, 1, 1)) * q
        vq = v * q
        ikQy = self.ik * torch.tensor(
            [beta + self.F1 * (U1 - U2), beta - self.F2 * (U1 - U2)],
            device=q.device, dtype=q.dtype).view(1, 2, 1, 1).to(self.ik.dtype)
        tend = -(self.ik * torch.fft.rfft2(uq, dim=(-2, -1))
                 + self.il * torch.fft.rfft2(vq, dim=(-2, -1))
                 + ikQy * ph)
        tend = torch.cat([
            tend[..., :1, :, :],
            tend[..., 1:, :, :] + rek * self.K2.to(self.ik.dtype) * ph[..., 1:, :, :],
        ], dim=-3)
        return tend

    def _rk4_step(self, qh: torch.Tensor, dt: float, U1: float, U2: float,
                  beta: float, rek: float) -> torch.Tensor:
        k1 = self._tendency(qh, U1, U2, beta, rek)
        k2 = self._tendency(qh + 0.5 * dt * k1, U1, U2, beta, rek)
        k3 = self._tendency(qh + 0.5 * dt * k2, U1, U2, beta, rek)
        k4 = self._tendency(qh + dt * k3, U1, U2, beta, rek)
        qh_new = self.filtr.to(qh.real.dtype).to(qh.dtype) * (
            qh + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
        if self.clip_range is not None:
            q_real = torch.fft.irfft2(qh_new, s=(self.ny, self.nx), dim=(-2, -1))
            q_real = torch.clamp(q_real, -self.clip_range, self.clip_range)
            qh_new = torch.fft.rfft2(q_real, dim=(-2, -1))
        return qh_new

    def step(self, state: torch.Tensor, forcing: torch.Tensor | None = None,
             **kwargs) -> torch.Tensor:
        U1 = kwargs.get("U1", self.U1)
        U2 = kwargs.get("U2", self.U2)
        beta = kwargs.get("beta", self.beta)
        rek = kwargs.get("rek", self.rek)
        single = state.dim() == 1
        state_b = state.unsqueeze(0) if single else state
        q = self._grid(state_b)
        qh = torch.fft.rfft2(q, dim=(-2, -1))
        qh_next = self._rk4_step(qh, self.dt, U1, U2, beta, rek)
        q_next = torch.fft.irfft2(qh_next, s=(self.ny, self.nx), dim=(-2, -1))
        out = self._flatten(q_next)
        if single:
            out = out.squeeze(0)
        return out

    def rollout_steps(self, state: torch.Tensor, steps: int,
                      **kwargs) -> torch.Tensor:
        single = state.dim() == 1
        state_b = state.unsqueeze(0) if single else state
        q = self._grid(state_b)
        qh = torch.fft.rfft2(q, dim=(-2, -1))
        U1 = kwargs.get("U1", self.U1)
        U2 = kwargs.get("U2", self.U2)
        beta = kwargs.get("beta", self.beta)
        rek = kwargs.get("rek", self.rek)
        for _ in range(steps):
            qh = self._rk4_step(qh, self.dt, U1, U2, beta, rek)
        q_final = torch.fft.irfft2(qh, s=(self.ny, self.nx), dim=(-2, -1))
        out = self._flatten(q_final)
        return out.squeeze(0) if single else out

    def generate_full_trajectory(self, num_steps: int, seed: int = 42,
                                 device: torch.device | None = None,
                                 spinup_steps: int = 4380,
                                 **kwargs) -> tuple:
        device = device if device is not None else self.device
        q0 = self._initial_q(1, seed, device)
        qh = torch.fft.rfft2(q0, dim=(-2, -1))
        U1 = kwargs.get("U1", self.U1)
        U2 = kwargs.get("U2", self.U2)
        beta = kwargs.get("beta", self.beta)
        rek = kwargs.get("rek", self.rek)
        for _ in range(spinup_steps):
            qh = self._rk4_step(qh, self.dt, U1, U2, beta, rek)
        traj = [torch.fft.irfft2(qh, s=(self.ny, self.nx), dim=(-2, -1))]
        for _ in range(num_steps - 1):
            qh = self._rk4_step(qh, self.dt, U1, U2, beta, rek)
            traj.append(torch.fft.irfft2(qh, s=(self.ny, self.nx), dim=(-2, -1)))
        traj_t = self._flatten(torch.cat(traj, dim=0).unsqueeze(1)).squeeze(1)
        forcing_t = torch.zeros(num_steps, device=traj_t.device,
                                dtype=traj_t.dtype)
        return traj_t, forcing_t

    def generate_batch_trajectories(self, num_windows: int, num_steps: int,
                                    spinup_steps: int = 4380,
                                    seed: int = 42,
                                    device: torch.device | None = None,
                                    **kwargs) -> tuple:
        device = device if device is not None else self.device
        q0 = self._initial_q(num_windows, seed, device)
        qh = torch.fft.rfft2(q0, dim=(-2, -1))
        U1 = kwargs.get("U1", self.U1)
        U2 = kwargs.get("U2", self.U2)
        beta = kwargs.get("beta", self.beta)
        rek = kwargs.get("rek", self.rek)
        for _ in range(spinup_steps):
            qh = self._rk4_step(qh, self.dt, U1, U2, beta, rek)
        traj = [torch.fft.irfft2(qh, s=(self.ny, self.nx), dim=(-2, -1))]
        for _ in range(num_steps - 1):
            qh = self._rk4_step(qh, self.dt, U1, U2, beta, rek)
            traj.append(torch.fft.irfft2(qh, s=(self.ny, self.nx), dim=(-2, -1)))
        traj_t = self._flatten(torch.stack(traj, dim=1))
        forcing_t = torch.zeros(num_windows, num_steps, device=traj_t.device,
                                dtype=traj_t.dtype)
        return traj_t, forcing_t

    def streamfunctions(self, state: torch.Tensor) -> torch.Tensor:
        single = state.dim() == 1
        state_b = state.unsqueeze(0) if single else state
        q = self._grid(state_b)
        qh = torch.fft.rfft2(q, dim=(-2, -1))
        ph = self._invert(qh)
        psi = torch.fft.irfft2(ph, s=(self.ny, self.nx), dim=(-2, -1))
        return psi.squeeze(0) if single else psi

    def kinetic_energy(self, state: torch.Tensor) -> torch.Tensor:
        single = state.dim() == 1
        state_b = state.unsqueeze(0) if single else state
        q = self._grid(state_b)
        qh = torch.fft.rfft2(q, dim=(-2, -1))
        ph = self._invert(qh)
        M2 = float(self.nx * self.ny) ** 2
        H1 = 500.0
        H2 = H1 / self.delta
        ke1 = 0.5 * H1 * (self.K2 * ph[..., 0, :, :].abs() ** 2).sum(dim=(-2, -1)) / M2
        ke2 = 0.5 * H2 * (self.K2 * ph[..., 1, :, :].abs() ** 2).sum(dim=(-2, -1)) / M2
        ke = (ke1 + ke2) / (H1 + H2)
        return ke.squeeze(0) if single else ke

    def enstrophy(self, state: torch.Tensor) -> torch.Tensor:
        single = state.dim() == 1
        state_b = state.unsqueeze(0) if single else state
        q = self._grid(state_b)
        del1 = self.delta / (self.delta + 1.0)
        del2 = 1.0 / (self.delta + 1.0)
        ens = 0.5 * (del1 * (q[..., 0, :, :] ** 2).mean(dim=(-2, -1))
                     + del2 * (q[..., 1, :, :] ** 2).mean(dim=(-2, -1)))
        return ens.squeeze(0) if single else ens
