"""Batched GPU generation of spectral wind amplitudes (Option B drivers).

Every window has its own seed, amplitude level, drift, start phase and
(optionally) gyrostat time unit. A window's amplitudes do not depend on the
batch it is generated in: noise comes from per-window generators, and the
gyrostat right-hand side uses sequential column updates (no atomics, no
reductions), which are exact elementwise operations on a given device. This
matters because the gyrostat is chaotic: its burn-in amplifies roundoff by
~e^60, so any batch-dependent rounding would change the wind entirely.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from models.gyrostat_driver import GyrostatSpec, get_preset
from models.qg_wind_modes import (
    MODE_MAPPINGS,
    SPECTRAL_DRIVERS,
    FourierWindBasis,
)

STREAM_GYROSTAT_IC = 0
STREAM_OU_NOISE = 1
STREAM_SURROGATE_PHASES = 2


def stream_seed(window_seed: int, stream: int) -> int:
    state = np.random.SeedSequence(int(window_seed), spawn_key=(int(stream),)).generate_state(2)
    return int(state[0]) << 32 | int(state[1])


def _per_window(value, batch: int) -> torch.Tensor:
    t = torch.as_tensor(value, dtype=torch.float64)
    if t.dim() == 0:
        return t.expand(batch).clone()
    if t.shape != (batch,):
        raise ValueError(f"per-window value has shape {tuple(t.shape)}, expected ({batch},)")
    return t


class BatchedGyrostat:
    def __init__(self, spec: GyrostatSpec, device: torch.device | str = "cpu"):
        self.spec = spec
        self.device = torch.device(device)
        f64 = torch.float64
        self.damping = torch.as_tensor(spec.damping, dtype=f64, device=self.device)
        self.forcing = torch.as_tensor(spec.forcing, dtype=f64, device=self.device)
        triads = np.asarray(spec.triads, dtype=np.int64)
        coeffs = np.asarray(spec.coeffs, dtype=np.float64)
        n_tri = triads.shape[0]
        self._i = torch.as_tensor(triads[:, 0], device=self.device)
        self._j = torch.as_tensor(triads[:, 1], device=self.device)
        self._k = torch.as_tensor(triads[:, 2], device=self.device)
        self._p, self._q, self._r, self._a, self._b, self._c = (
            torch.as_tensor(coeffs[:, n], dtype=f64, device=self.device) for n in range(6))
        roles: list[list[int]] = [[] for _ in range(spec.n_modes)]
        for role in range(3):
            for g in range(n_tri):
                roles[int(triads[g, role])].append(role * n_tri + g)
        depth = max(len(r) for r in roles)
        pad = 3 * n_tri
        table = np.full((depth, spec.n_modes), pad, dtype=np.int64)
        for m, r in enumerate(roles):
            table[:len(r), m] = r
        self._role_src = [torch.as_tensor(table[d], device=self.device) for d in range(depth)]
        self.mean = torch.as_tensor(spec.mode_mean, dtype=f64, device=self.device) \
            if spec.mode_mean is not None else None
        self.std = torch.as_tensor(spec.mode_std, dtype=f64, device=self.device) \
            if spec.mode_std is not None else None

    def rhs(self, y: torch.Tensor) -> torch.Tensor:
        y1, y2, y3 = y[:, self._i], y[:, self._j], y[:, self._k]
        d1 = self._p * y2 * y3 + self._b * y3 - self._c * y2
        d2 = self._q * y3 * y1 + self._c * y1 - self._a * y3
        d3 = self._r * y1 * y2 + self._a * y2 - self._b * y1
        terms = torch.cat([d1, d2, d3, torch.zeros_like(d1[:, :1])], dim=1)
        out = self.forcing - self.damping * y
        for src in self._role_src:
            out = out + terms[:, src]
        return out

    def _rk4(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        k1 = self.rhs(y)
        k2 = self.rhs(y + 0.5 * h * k1)
        k3 = self.rhs(y + 0.5 * h * k2)
        k4 = self.rhs(y + h * k3)
        return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def initial_states(self, seeds: list[int]) -> torch.Tensor:
        ys = []
        for s in seeds:
            gen = torch.Generator(device="cpu").manual_seed(stream_seed(s, STREAM_GYROSTAT_IC))
            ys.append(5.0 * torch.randn(self.spec.n_modes, generator=gen, dtype=torch.float64))
        return torch.stack(ys).to(self.device)

    def burn_in(self, y: torch.Tensor, units: float, max_substep: float = 0.005) -> torch.Tensor:
        n = int(math.ceil(units / max_substep))
        h = torch.full((y.shape[0], 1), units / max(n, 1), dtype=torch.float64, device=self.device)
        for _ in range(n):
            y = self._rk4(y, h)
        return y

    def integrate(self, y: torch.Tensor, n_steps: int, dt_units,
                  max_substep: float = 0.005) -> torch.Tensor:
        dt = _per_window(dt_units, y.shape[0])
        n_sub = torch.ceil(dt / max_substep).clamp_min(1).to(torch.int64)
        out = torch.empty((y.shape[0], n_steps, y.shape[1]), dtype=torch.float64, device=self.device)
        for n in torch.unique(n_sub).tolist():
            idx = torch.nonzero(n_sub == n).flatten().to(self.device)
            h = (dt[n_sub == n] / n).to(self.device).view(-1, 1)
            ys = y[idx]
            for t in range(n_steps):
                out[idx, t] = ys
                for _ in range(n):
                    ys = self._rk4(ys, h)
        return out

    def standardize(self, y: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise ValueError(f"preset {self.spec.name!r} has no attractor constants")
        return (y - self.mean) / self.std


def batched_unit_ou(seeds: list[int], n_steps: int, n_channels: int, dt_days: float,
                    tau_days: float, device: torch.device | str = "cpu") -> torch.Tensor:
    rho = math.exp(-dt_days / tau_days)
    kick = math.sqrt(1.0 - rho * rho)
    noise = []
    for s in seeds:
        gen = torch.Generator(device="cpu").manual_seed(stream_seed(s, STREAM_OU_NOISE))
        noise.append(torch.randn((n_steps, n_channels), generator=gen, dtype=torch.float64))
    xi = torch.stack(noise).to(device)
    z = torch.empty_like(xi)
    z[:, 0] = xi[:, 0]
    for t in range(1, n_steps):
        z[:, t] = rho * z[:, t - 1] + kick * xi[:, t]
    return z


def batched_phase_surrogate(y: torch.Tensor, seeds: list[int]) -> torch.Tensor:
    n = y.shape[1]
    out = torch.empty_like(y)
    for b, s in enumerate(seeds):
        spec = torch.fft.rfft(y[b], dim=0)
        gen = torch.Generator(device="cpu").manual_seed(stream_seed(s, STREAM_SURROGATE_PHASES))
        ph = 2.0 * math.pi * torch.rand(spec.shape[0], generator=gen, dtype=torch.float64)
        ph[0] = 0.0
        if n % 2 == 0:
            ph[-1] = 0.0
        rot = torch.exp(1j * ph.to(y.device))
        out[b] = torch.fft.irfft(spec * rot[:, None], n=n, dim=0)
    return out


def batched_spectral_wind(basis: FourierWindBasis, driver: str, seeds: list[int],
                          n_steps: int, dt_seconds: float, amp, cx, cy, x0, y0,
                          sigma: float, tau_days: float = 15.0, preset: str = "l63ring4",
                          mapping: str | None = None, time_unit_days=None,
                          burnin_units: float = 50.0, surrogate_factor: int = 4,
                          device: torch.device | str = "cpu") -> torch.Tensor:
    if driver not in SPECTRAL_DRIVERS:
        raise ValueError(f"unknown spectral driver {driver!r}; known: {SPECTRAL_DRIVERS}")
    batch = len(seeds)
    dt_days = dt_seconds / 86400.0
    if driver == "spectral_ou":
        z = batched_unit_ou(seeds, n_steps, basis.n_amp, dt_days, tau_days, device)
    else:
        spec = get_preset(preset)
        mapping = mapping or next(iter(MODE_MAPPINGS.get(preset, {})), None)
        order = MODE_MAPPINGS.get(preset, {}).get(mapping)
        if order is None:
            raise ValueError(f"no mapping {mapping!r} for preset {preset!r}")
        if len(order) != basis.n_amp:
            raise ValueError(f"preset {preset!r} with mapping {mapping!r} gives {len(order)} "
                             f"amplitudes, basis needs {basis.n_amp}")
        unit = time_unit_days if time_unit_days is not None else tau_days / spec.efold_units
        dt_units = dt_days / _per_window(unit, batch)
        gyro = BatchedGyrostat(spec, device)
        y = gyro.burn_in(gyro.initial_states(seeds), burnin_units)
        n_gen = surrogate_factor * n_steps if driver == "gyrostat_surrogate" else n_steps
        traj = gyro.standardize(gyro.integrate(y, n_gen, dt_units))
        if driver == "gyrostat_surrogate":
            traj = batched_phase_surrogate(traj, seeds)
            start = (n_gen - n_steps) // 2
            traj = traj[:, start:start + n_steps]
        z = traj[:, :, list(order)]
    scale = _per_window(amp, batch).to(z.device).view(-1, 1, 1) \
        * basis.ricker_mode_std(sigma).to(z.device).view(1, 1, -1)
    a = z * scale
    t = torch.arange(n_steps, dtype=torch.float64, device=z.device) * dt_seconds
    dx = _per_window(x0, batch).to(z.device).view(-1, 1) + _per_window(cx, batch).to(z.device).view(-1, 1) * t
    dy = _per_window(y0, batch).to(z.device).view(-1, 1) + _per_window(cy, batch).to(z.device).view(-1, 1) * t
    return basis.translate(a, dx, dy)
