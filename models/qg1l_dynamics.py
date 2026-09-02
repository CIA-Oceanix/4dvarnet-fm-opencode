import math

import torch

from models.dynamics import DynamicsBase


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if dtype == torch.float64 else torch.complex64


class QG1LDynamics(DynamicsBase):
    """Reduced-gravity single-layer QG model (structural-error DA reference).

    One active upper layer over a motionless deep layer, on a double-periodic
    beta-plane. PV anomaly relative to the beta-plane background:

        q = lap psi - psi / rd^2

    with masked zero-mode inversion (K2 == 0 -> psi_hat = 0):

        psi_hat = -q_hat / (K2 + rd^-2)

    Evolution (flux-form advection, RK4, pyqg exponential filter):

        dq/dt = -J(psi, q + beta*y) - rek * lap psi + curl_tau

    where `-rek*lap psi` is linear bottom drag and `curl_tau` is the same
    moving-storm wind-stress-curl forcing as `QGDynamics`, applied directly to
    the single active layer.

    State layout: flattened [..., ny * nx], row-major grid. The system is
    autonomous; the `forcing` argument of step/rollout is accepted for
    DynamicsBase compatibility and ignored (wind enters via `wind_state_t`).
    `param_names=["beta", "rd", "rek", "U1"]`; runtime overrides via step
    kwargs (rd is fixed at construction). `wind_amp=0` reproduces the unforced
    trajectory bitwise.
    """

    param_dim = 4
    forcing_dim = 1

    def __init__(self, nx: int = 64, ny: int | None = None,
                 L: float = 1e6, W: float | None = None, dt: float = 7200.0,
                 beta: float = 1.5e-11, rd: float = 15000.0,
                 U1: float = 0.05, rek: float = 5.787e-7,
                 filterfac: float = 23.6, clip_range: float | None = None,
                 H: float = 500.0,
                 wind_amp: float = 0.0, wind_tau_days: float = 15.0,
                 wind_sigma: float = 250000.0, wind_cx: float = 0.5,
                 wind_cy: float = 0.03,
                 wind_drift_tau_days: float = 10.0,
                 wind_drift_sigma: float = 50000.0,
                 wind_seed: int = 7,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.param_names: list[str] = ["beta", "rd", "rek", "U1"]
        if ny is None:
            ny = nx
        if nx % 2 or ny % 2:
            raise ValueError("QG1LDynamics requires even nx and ny")
        self.nx = int(nx)
        self.ny = int(ny)
        self.L = float(L)
        self.W = float(W) if W is not None else float(L)
        self.dt = float(dt)
        self.beta = float(beta)
        self.rd = float(rd)
        self.U1 = float(U1)
        self.rek = float(rek)
        self.filterfac = float(filterfac)
        self.clip_range = clip_range
        self.H = float(H)
        self.wind_amp = float(wind_amp)
        self.wind_tau_days = float(wind_tau_days)
        self.wind_sigma = float(wind_sigma)
        self.wind_cx = float(wind_cx)
        self.wind_cy = float(wind_cy)
        self.wind_drift_tau_days = float(wind_drift_tau_days)
        self.wind_drift_sigma = float(wind_drift_sigma)
        self.wind_seed = int(wind_seed)
        self.state_dim = self.ny * self.nx
        self.dtype = dtype

        nk = self.nx // 2 + 1
        dk = 2.0 * math.pi / self.L
        dl = 2.0 * math.pi / self.W
        kk = dk * torch.arange(nk, dtype=torch.float64)
        ll = dl * torch.cat([torch.arange(0.0, self.ny // 2, dtype=torch.float64),
                             torch.arange(-self.ny // 2, 0.0, dtype=torch.float64)])
        l2d, k2d = torch.meshgrid(ll, kk, indexing="ij")
        K2 = k2d ** 2 + l2d ** 2

        inv = torch.where(K2 > 0, 1.0 / (K2 + self.rd ** -2),
                          torch.zeros_like(K2))
        a = -inv

        cphi = 0.65 * math.pi
        dx = self.L / self.nx
        dy = self.W / self.ny
        wvx = torch.sqrt((k2d * dx) ** 2 + (l2d * dy) ** 2)
        filtr = torch.exp(-self.filterfac * (wvx - cphi) ** 4)
        filtr = torch.where(wvx <= cphi, torch.ones_like(filtr), filtr)

        cdtype = _complex_dtype(dtype)
        self.register_buffer("K2", K2.to(dtype))
        self.register_buffer("a", a.to(dtype))
        self.register_buffer("filtr", filtr.to(dtype))
        self.register_buffer("ik", (1j * k2d).to(cdtype))
        self.register_buffer("il", (1j * l2d).to(cdtype))

        x = torch.arange(self.nx, dtype=torch.float64) * (self.L / self.nx)
        y = torch.arange(self.ny, dtype=torch.float64) * (self.W / self.ny)
        self.register_buffer("x_grid", x.to(dtype))
        self.register_buffer("y_grid", y.to(dtype))

    @property
    def device(self) -> torch.device:
        return self.K2.device

    def to(self, device=None, dtype=None):
        if dtype is not None and dtype != self.dtype:
            raise ValueError("QG1LDynamics dtype is fixed at construction")
        return super().to(device=device)

    def _grid(self, state: torch.Tensor) -> torch.Tensor:
        shape = state.shape[:-1] + (self.ny, self.nx)
        return state.reshape(shape)

    def _flatten(self, q: torch.Tensor) -> torch.Tensor:
        return q.reshape(*q.shape[:-2], self.state_dim)

    def _initial_q(self, batch_size: int, seed: int,
                   device: torch.device) -> torch.Tensor:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        q = (1e-7 * torch.rand((batch_size, self.ny, self.nx), generator=gen)
             + 1e-6 * torch.rand((batch_size, 1, self.nx), generator=gen))
        q = q - q.mean(dim=(-2, -1), keepdim=True)
        return q.to(device=device, dtype=self.dtype)

    def generate_wind_state(self, num_steps: int,
                            seed: int | None = None,
                            x0: float | None = None,
                            y0: float | None = None) -> torch.Tensor:
        out = torch.zeros(num_steps, 3, dtype=torch.float64)
        if self.wind_amp == 0.0:
            return out.to(self.dtype).to(self.device)
        gen = torch.Generator(device="cpu").manual_seed(seed or self.wind_seed)
        dt = self.dt
        tau_a = self.wind_tau_days * 86400.0
        coeff_a = self.wind_amp * math.sqrt(2.0 / tau_a * dt)
        tau_d = self.wind_drift_tau_days * 86400.0
        coeff_d = self.wind_drift_sigma * math.sqrt(2.0 / tau_d * dt)
        a = torch.zeros((), dtype=torch.float64)
        wx = torch.zeros((), dtype=torch.float64)
        wy = torch.zeros((), dtype=torch.float64)
        x0 = self.L / 2.0 if x0 is None else float(x0)
        y0 = self.W / 2.0 if y0 is None else float(y0)
        for k in range(num_steps):
            a = a - (1.0 / tau_a) * a * dt + coeff_a * torch.randn((), generator=gen)
            wx = wx - (1.0 / tau_d) * wx * dt + coeff_d * torch.randn((), generator=gen)
            wy = wy - (1.0 / tau_d) * wy * dt + coeff_d * torch.randn((), generator=gen)
            xc = (x0 + self.wind_cx * dt * k + wx) % self.L
            yc = (y0 + self.wind_cy * dt * k + wy) % self.W
            out[k, 0] = a
            out[k, 1] = xc
            out[k, 2] = yc
        return out.to(self.dtype).to(self.device)

    def wind_curl_field(self, wind_state: torch.Tensor) -> torch.Tensor:
        wind_state = wind_state.double().to(self.device)
        a = wind_state[..., 0]
        xc = wind_state[..., 1]
        yc = wind_state[..., 2]
        x = self.x_grid.double()
        y = self.y_grid.double()
        n_im = (-1, 0, 1)
        field = torch.zeros(*wind_state.shape[:-1], self.ny, self.nx,
                            dtype=torch.float64, device=self.device)
        sig2 = self.wind_sigma ** 2
        for ix in n_im:
            for iy in n_im:
                dx = x[None, None, :] - (xc - ix * self.L)[..., None, None]
                dy = y[None, :, None] - (yc - iy * self.W)[..., None, None]
                r2 = dx ** 2 + dy ** 2
                field = field + ((1.0 - r2 / (2.0 * sig2))
                                 * torch.exp(-r2 / (2.0 * sig2)))
        return (a[..., None, None] * field).to(self.dtype)

    def _wind_curl_spectral(self, qh: torch.Tensor,
                            wind_state_t) -> torch.Tensor:
        if self.wind_amp == 0.0 or wind_state_t is None \
                or float(wind_state_t[0]) == 0.0:
            return torch.zeros(self.ny, qh.shape[-1],
                               device=qh.device, dtype=qh.dtype)
        curl = self.wind_curl_field(wind_state_t.unsqueeze(0)).squeeze(0)
        curlh = torch.fft.rfft2(curl, dim=(-2, -1))
        cdtype = torch.complex64 if curl.real.dtype == torch.float32 \
            else torch.complex128
        return curlh.to(cdtype)

    def _invert(self, qh: torch.Tensor) -> torch.Tensor:
        return self.a[None, :, :] * qh

    def state_from_streamfunction(self, psi: torch.Tensor) -> torch.Tensor:
        single = psi.dim() == 1
        psi_g = psi.reshape(*psi.shape[:-1], self.ny, self.nx)
        psh = torch.fft.rfft2(psi_g, dim=(-2, -1))
        qh = (self.K2[None, :, :] + self.rd ** -2).to(psh.dtype) * psh
        q = torch.fft.irfft2(qh, s=(self.ny, self.nx), dim=(-2, -1))
        out = self._flatten(q)
        return out.squeeze(0) if single else out

    def _tendency(self, qh: torch.Tensor, U1: float, beta: float,
                  rek: float, wind_state_t=None) -> torch.Tensor:
        q = torch.fft.irfft2(qh, s=(self.ny, self.nx), dim=(-2, -1))
        ph = self._invert(qh)
        u = torch.fft.irfft2(-self.il * ph, s=(self.ny, self.nx), dim=(-2, -1))
        v = torch.fft.irfft2(self.ik * ph, s=(self.ny, self.nx), dim=(-2, -1))
        uq = (u + U1) * q
        vq = v * q
        ikB = (self.ik * beta).to(self.ik.dtype)
        tend = -(self.ik * torch.fft.rfft2(uq, dim=(-2, -1))
                 + self.il * torch.fft.rfft2(vq, dim=(-2, -1))
                 + ikB * ph)
        tend = (tend + self._wind_curl_spectral(qh, wind_state_t)
                + rek * self.K2.to(self.ik.dtype) * ph)
        return tend

    def _rk4_step(self, qh: torch.Tensor, dt: float, U1: float, beta: float,
                  rek: float, wind_state_t=None) -> torch.Tensor:
        k1 = self._tendency(qh, U1, beta, rek, wind_state_t)
        k2 = self._tendency(qh + 0.5 * dt * k1, U1, beta, rek, wind_state_t)
        k3 = self._tendency(qh + 0.5 * dt * k2, U1, beta, rek, wind_state_t)
        k4 = self._tendency(qh + dt * k3, U1, beta, rek, wind_state_t)
        qh_new = self.filtr.to(qh.real.dtype).to(qh.dtype) * (
            qh + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
        if self.clip_range is not None:
            q_real = torch.fft.irfft2(qh_new, s=(self.ny, self.nx), dim=(-2, -1))
            q_real = torch.clamp(q_real, -self.clip_range, self.clip_range)
            qh_new = torch.fft.rfft2(q_real, dim=(-2, -1))
        return qh_new

    def step(self, state: torch.Tensor, forcing: torch.Tensor | None = None,
             wind_state_t=None, **kwargs) -> torch.Tensor:
        U1 = kwargs.get("U1", self.U1)
        beta = kwargs.get("beta", self.beta)
        rek = kwargs.get("rek", self.rek)
        single = state.dim() == 1
        state_b = state.unsqueeze(0) if single else state
        q = self._grid(state_b)
        qh = torch.fft.rfft2(q, dim=(-2, -1))
        qh_next = self._rk4_step(qh, self.dt, U1, beta, rek, wind_state_t)
        q_next = torch.fft.irfft2(qh_next, s=(self.ny, self.nx), dim=(-2, -1))
        out = self._flatten(q_next)
        if single:
            out = out.squeeze(0)
        return out

    def rollout_steps(self, state: torch.Tensor, steps: int,
                      wind_state: torch.Tensor | None = None,
                      **kwargs) -> torch.Tensor:
        single = state.dim() == 1
        state_b = state.unsqueeze(0) if single else state
        q = self._grid(state_b)
        qh = torch.fft.rfft2(q, dim=(-2, -1))
        U1 = kwargs.get("U1", self.U1)
        beta = kwargs.get("beta", self.beta)
        rek = kwargs.get("rek", self.rek)
        wstate = wind_state if wind_state is not None \
            else torch.zeros(steps, 3, device=qh.device, dtype=self.dtype)
        for k in range(steps):
            qh = self._rk4_step(qh, self.dt, U1, beta, rek, wstate[k])
        q_final = torch.fft.irfft2(qh, s=(self.ny, self.nx), dim=(-2, -1))
        out = self._flatten(q_final)
        return out.squeeze(0) if single else out

    def rollout_trajectory(self, state: torch.Tensor, steps: int,
                           wind_state: torch.Tensor | None = None,
                           **kwargs) -> torch.Tensor:
        single = state.dim() == 1
        state_b = state.unsqueeze(0) if single else state
        q = self._grid(state_b)
        qh = torch.fft.rfft2(q, dim=(-2, -1))
        U1 = kwargs.get("U1", self.U1)
        beta = kwargs.get("beta", self.beta)
        rek = kwargs.get("rek", self.rek)
        wstate = wind_state if wind_state is not None \
            else torch.zeros(steps, 3, device=qh.device, dtype=self.dtype)
        traj = [torch.fft.irfft2(qh, s=(self.ny, self.nx), dim=(-2, -1))]
        for k in range(steps):
            qh = self._rk4_step(qh, self.dt, U1, beta, rek, wstate[k])
            traj.append(torch.fft.irfft2(qh, s=(self.ny, self.nx), dim=(-2, -1)))
        out = self._flatten(torch.stack(traj, dim=0))
        if single:
            out = out.squeeze(1)
        return out

    def generate_full_trajectory(self, num_steps: int, seed: int = 42,
                                 device: torch.device | None = None,
                                 spinup_steps: int = 4380,
                                 wind_state: torch.Tensor | None = None,
                                 **kwargs) -> tuple:
        device = device if device is not None else self.device
        q0 = self._initial_q(1, seed, device)
        qh = torch.fft.rfft2(q0, dim=(-2, -1))
        U1 = kwargs.get("U1", self.U1)
        beta = kwargs.get("beta", self.beta)
        rek = kwargs.get("rek", self.rek)
        ws = wind_state if wind_state is not None else self.generate_wind_state(
            num_steps, seed)
        for _ in range(spinup_steps):
            qh = self._rk4_step(qh, self.dt, U1, beta, rek)
        traj = [torch.fft.irfft2(qh, s=(self.ny, self.nx), dim=(-2, -1))]
        for k in range(num_steps - 1):
            qh = self._rk4_step(qh, self.dt, U1, beta, rek, ws[k])
            traj.append(torch.fft.irfft2(qh, s=(self.ny, self.nx), dim=(-2, -1)))
        traj_t = self._flatten(torch.cat(traj, dim=0).unsqueeze(1)).squeeze(1)
        return traj_t, ws

    def generate_batch_trajectories(self, num_windows: int, num_steps: int,
                                    spinup_steps: int = 4380,
                                    seed: int = 42,
                                    device: torch.device | None = None,
                                    wind_state: torch.Tensor | None = None,
                                    **kwargs) -> tuple:
        device = device if device is not None else self.device
        q0 = self._initial_q(num_windows, seed, device)
        qh = torch.fft.rfft2(q0, dim=(-2, -1))
        U1 = kwargs.get("U1", self.U1)
        beta = kwargs.get("beta", self.beta)
        rek = kwargs.get("rek", self.rek)
        if wind_state is None:
            wind_state = torch.stack([
                self.generate_wind_state(num_steps, seed + i)
                for i in range(num_windows)
            ])
        step_ws = wind_state[0]
        for _ in range(spinup_steps):
            qh = self._rk4_step(qh, self.dt, U1, beta, rek)
        traj = [torch.fft.irfft2(qh, s=(self.ny, self.nx), dim=(-2, -1))]
        for k in range(num_steps - 1):
            qh = self._rk4_step(qh, self.dt, U1, beta, rek, step_ws[k])
            traj.append(torch.fft.irfft2(qh, s=(self.ny, self.nx), dim=(-2, -1)))
        traj_t = self._flatten(torch.stack(traj, dim=1))
        return traj_t, wind_state

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
        ke = 0.5 * (self.K2 * ph.abs() ** 2).sum(dim=(-2, -1)) / M2
        return ke.squeeze(0) if single else ke

    def enstrophy(self, state: torch.Tensor) -> torch.Tensor:
        single = state.dim() == 1
        state_b = state.unsqueeze(0) if single else state
        q = self._grid(state_b)
        ens = 0.5 * (q ** 2).mean(dim=(-2, -1))
        return ens.squeeze(0) if single else ens
