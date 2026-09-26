"""Batched two-layer QG dynamics with per-window parameters (GPU generation).

Same numerics as `models.qg_dynamics.QGDynamics` (pyqg-compatible inversion,
flux-form advection, RK4 with the wind held over the step, exponential filter,
optional clip), with the ocean parameters held as ``(B,)`` tensors so that B
windows with different ``rd, U1, U2, beta, rek`` advance in one batch. It adds
the relative-wind eddy drag ``-r_cf * zeta_1`` on the upper layer.

`QGDynamics` is left untouched: the legacy benchmark path stays bit-identical.
The wind enters as a precomputed spectral PV source through a `WindForcing`.
"""
from __future__ import annotations

import math

import torch

from models.qg_dynamics import QGDynamics
from models.qg_wind_modes import FourierWindBasis


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if dtype == torch.float64 else torch.complex64


def _as_batch(value, batch: int, dtype: torch.dtype, device) -> torch.Tensor:
    t = torch.as_tensor(value, dtype=dtype, device=device)
    if t.dim() == 0:
        return t.expand(batch).clone()
    if t.shape != (batch,):
        raise ValueError(f"per-window parameter has shape {tuple(t.shape)}, expected ({batch},)")
    return t


class WindForcing:
    def spectral(self, step: int, like: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SpectralWindForcing(WindForcing):
    def __init__(self, basis: FourierWindBasis, amplitudes: torch.Tensor):
        self.basis = basis
        self.amplitudes = amplitudes

    def spectral(self, step: int, like: torch.Tensor) -> torch.Tensor:
        return self.basis.curl_spectral(self.amplitudes[:, step]).to(like.dtype)


class StormWindForcing(WindForcing):
    def __init__(self, curl_model: QGDynamics, wind_state: torch.Tensor):
        self.curl_model = curl_model
        self.wind_state = wind_state

    def spectral(self, step: int, like: torch.Tensor) -> torch.Tensor:
        curl = self.curl_model.wind_curl_field(self.wind_state[:, step])
        return torch.fft.rfft2(curl, dim=(-2, -1)).to(like.dtype)


class BatchedQGDynamics(torch.nn.Module):
    def __init__(self, batch: int, nx: int = 64, ny: int | None = None,
                 L: float = 1e6, W: float | None = None, dt: float = 7200.0,
                 beta=1.5e-11, rd=15000.0, delta: float = 0.25,
                 U1=0.025, U2=0.0, rek=5.787e-7, r_cf=0.0,
                 filterfac: float = 23.6, clip_range: float | None = None,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device | str | None = None):
        super().__init__()
        ny = ny or nx
        if nx % 2 or ny % 2:
            raise ValueError("BatchedQGDynamics requires even nx and ny")
        self.batch = int(batch)
        self.nx, self.ny = int(nx), int(ny)
        self.L = float(L)
        self.W = float(W) if W is not None else float(L)
        self.dt = float(dt)
        self.delta = float(delta)
        self.filterfac = float(filterfac)
        self.clip_range = clip_range
        self.dtype = dtype
        self.state_dim = 2 * self.ny * self.nx
        device = torch.device(device) if device is not None else torch.device("cpu")
        f64 = torch.float64

        self.beta = _as_batch(beta, batch, f64, "cpu")
        self.rd = _as_batch(rd, batch, f64, "cpu")
        self.U1 = _as_batch(U1, batch, f64, "cpu")
        self.U2 = _as_batch(U2, batch, f64, "cpu")
        self.rek = _as_batch(rek, batch, f64, "cpu")
        self.r_cf = _as_batch(r_cf, batch, f64, "cpu")

        nk = self.nx // 2 + 1
        dk = 2.0 * math.pi / self.L
        dl = 2.0 * math.pi / self.W
        kk = dk * torch.arange(nk, dtype=f64)
        ll = dl * torch.cat([torch.arange(0.0, self.ny // 2, dtype=f64),
                             torch.arange(-self.ny // 2, 0.0, dtype=f64)])
        l2d, k2d = torch.meshgrid(ll, kk, indexing="ij")
        K2 = k2d ** 2 + l2d ** 2

        F1 = self.rd ** -2 / (1.0 + self.delta)
        F2 = self.delta * F1
        self.F1, self.F2 = F1, F2
        F1b, F2b = F1.view(-1, 1, 1), F2.view(-1, 1, 1)
        det_inv = torch.where(K2 > 0, 1.0 / (K2 * (K2 + F1b + F2b)), torch.zeros_like(K2 + F1b))
        a11 = -(K2 + F2b) * det_inv
        a12 = -F1b * det_inv
        a21 = -F2b * det_inv
        a22 = -(K2 + F1b) * det_inv

        cphi = 0.65 * math.pi
        wvx = torch.sqrt((k2d * self.L / self.nx) ** 2 + (l2d * self.W / self.ny) ** 2)
        filtr = torch.exp(-self.filterfac * (wvx - cphi) ** 4)
        filtr = torch.where(wvx <= cphi, torch.ones_like(filtr), filtr)

        cdtype = _complex_dtype(dtype)
        qy1 = self.beta + F1 * (self.U1 - self.U2)
        qy2 = self.beta - F2 * (self.U1 - self.U2)
        self.register_buffer("K2", K2.to(dtype))
        self.register_buffer("a11", a11.to(dtype))
        self.register_buffer("a12", a12.to(dtype))
        self.register_buffer("a21", a21.to(dtype))
        self.register_buffer("a22", a22.to(dtype))
        self.register_buffer("filtr", filtr.to(dtype))
        self.register_buffer("ik", (1j * k2d).to(cdtype))
        self.register_buffer("il", (1j * l2d).to(cdtype))
        self.register_buffer("Ubg", torch.stack([self.U1, self.U2], -1).view(-1, 2, 1, 1).to(dtype))
        self.register_buffer("Qy", torch.stack([qy1, qy2], -1).view(-1, 2, 1, 1).to(dtype))
        self.register_buffer("rek_b", self.rek.view(-1, 1, 1).to(dtype))
        self.register_buffer("rcf_b", self.r_cf.view(-1, 1, 1).to(dtype))
        self.to(device)

    @property
    def device(self) -> torch.device:
        return self.K2.device

    def _grid(self, state: torch.Tensor) -> torch.Tensor:
        return state.reshape(state.shape[:-1] + (2, self.ny, self.nx))

    def _flatten(self, q: torch.Tensor) -> torch.Tensor:
        return q.reshape(*q.shape[:-3], self.state_dim)

    def _invert(self, qh: torch.Tensor) -> torch.Tensor:
        ph1 = self.a11 * qh[:, 0] + self.a12 * qh[:, 1]
        ph2 = self.a21 * qh[:, 0] + self.a22 * qh[:, 1]
        return torch.stack([ph1, ph2], dim=1)

    def initial_q(self, seeds: list[int]) -> torch.Tensor:
        qs = []
        for s in seeds:
            gen = torch.Generator(device="cpu").manual_seed(int(s))
            q1 = (1e-7 * torch.rand((self.ny, self.nx), generator=gen)
                  + 1e-6 * torch.rand((1, self.nx), generator=gen))
            q = torch.stack([q1, torch.zeros((self.ny, self.nx))], dim=0)
            qs.append(q - q.mean(dim=(-2, -1), keepdim=True))
        return torch.stack(qs).to(device=self.device, dtype=self.dtype)

    def _tendency(self, qh: torch.Tensor, wind_h: torch.Tensor | None) -> torch.Tensor:
        q = torch.fft.irfft2(qh, s=(self.ny, self.nx), dim=(-2, -1))
        ph = self._invert(qh)
        u = torch.fft.irfft2(-self.il * ph, s=(self.ny, self.nx), dim=(-2, -1))
        v = torch.fft.irfft2(self.ik * ph, s=(self.ny, self.nx), dim=(-2, -1))
        uq = (u + self.Ubg) * q
        vq = v * q
        ikQy = self.ik * self.Qy.to(self.ik.dtype)
        tend = -(self.ik * torch.fft.rfft2(uq, dim=(-2, -1))
                 + self.il * torch.fft.rfft2(vq, dim=(-2, -1))
                 + ikQy * ph)
        K2c = self.K2.to(self.ik.dtype)
        upper = tend[:, 0] + (self.rcf_b * K2c) * ph[:, 0]
        if wind_h is not None:
            upper = upper + wind_h
        lower = tend[:, 1] + (self.rek_b * K2c) * ph[:, 1]
        return torch.stack([upper, lower], dim=1)

    def rk4_step(self, qh: torch.Tensor, wind_h: torch.Tensor | None = None) -> torch.Tensor:
        dt = self.dt
        k1 = self._tendency(qh, wind_h)
        k2 = self._tendency(qh + 0.5 * dt * k1, wind_h)
        k3 = self._tendency(qh + 0.5 * dt * k2, wind_h)
        k4 = self._tendency(qh + dt * k3, wind_h)
        qh_new = self.filtr.to(qh.real.dtype).to(qh.dtype) * (
            qh + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
        if self.clip_range is not None:
            q_real = torch.fft.irfft2(qh_new, s=(self.ny, self.nx), dim=(-2, -1))
            q_real = torch.clamp(q_real, -self.clip_range, self.clip_range)
            qh_new = torch.fft.rfft2(q_real, dim=(-2, -1))
        return qh_new

    def rollout(self, q0: torch.Tensor, n_steps: int,
                forcing: WindForcing | None = None, forcing_offset: int = 0,
                keep_every: int = 1, keep_from: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        qh = torch.fft.rfft2(q0, dim=(-2, -1))
        kept = []
        for k in range(n_steps + 1):
            if k >= keep_from and (k - keep_from) % keep_every == 0:
                kept.append(torch.fft.irfft2(qh, s=(self.ny, self.nx), dim=(-2, -1)))
            if k == n_steps:
                break
            wind_h = None if forcing is None else forcing.spectral(forcing_offset + k, qh[:, 0])
            qh = self.rk4_step(qh, wind_h)
        q_end = torch.fft.irfft2(qh, s=(self.ny, self.nx), dim=(-2, -1))
        return torch.stack(kept, dim=1), q_end

    def streamfunctions(self, q: torch.Tensor) -> torch.Tensor:
        qh = torch.fft.rfft2(q, dim=(-2, -1))
        shape = (self.batch,) + (1,) * (q.dim() - 4) + self.a11.shape[1:]
        a11, a12 = self.a11.view(shape), self.a12.view(shape)
        a21, a22 = self.a21.view(shape), self.a22.view(shape)
        ph = torch.stack([a11 * qh[..., 0, :, :] + a12 * qh[..., 1, :, :],
                          a21 * qh[..., 0, :, :] + a22 * qh[..., 1, :, :]], dim=-3)
        return torch.fft.irfft2(ph, s=(self.ny, self.nx), dim=(-2, -1))
