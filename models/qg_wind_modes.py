"""Spectral wind-stress-curl forcing for the doubly periodic QG ocean.

The upper-layer PV source is ``curl_tau(x, y, t) = sum_k a_k(t) phi_k(x, y)``
with ``phi_k`` the cos/sin parts of low-order Fourier modes. Amplitudes are
laid out as ``[c_0, s_0, c_1, s_1, ...]``, one (cos, sin) pair per wavevector
of the half-plane, ordered by ``(|k|^2, -kx, -ky)``.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from models.gyrostat_driver import (
    GyrostatSystem,
    get_preset,
    phase_randomized_surrogate,
)

SPECTRAL_DRIVERS = ("spectral_ou", "gyrostat", "gyrostat_surrogate")

MODE_MAPPINGS = {
    "l63ring4": {"xz_pairs": (0, 2, 3, 5, 6, 8, 9, 11, 1, 4, 7, 10)},
    "l63": {"identity": (0, 1, 2)},
}


def _half_plane_wavevectors(kmax: int, shape: str) -> list[tuple[int, int]]:
    out = []
    for kx in range(0, kmax + 1):
        for ky in range(-kmax, kmax + 1):
            if kx == 0 and ky <= 0:
                continue
            if shape == "disk" and kx * kx + ky * ky > kmax * kmax:
                continue
            if shape not in ("disk", "square"):
                raise ValueError(f"unknown basis shape {shape!r}")
            out.append((kx, ky))
    return sorted(out, key=lambda k: (k[0] ** 2 + k[1] ** 2, -k[0], -k[1]))


class FourierWindBasis:
    def __init__(self, nx: int, ny: int | None = None, L: float = 1e6,
                 W: float | None = None, kmax: int = 2, shape: str = "disk",
                 dtype: torch.dtype = torch.float64,
                 device: torch.device | str | None = None):
        self.nx = nx
        self.ny = ny or nx
        self.L = float(L)
        self.W = float(W or L)
        self.kmax = kmax
        self.shape = shape
        self.wavevectors = _half_plane_wavevectors(kmax, shape)
        self.n_amp = 2 * len(self.wavevectors)
        kv = torch.tensor(self.wavevectors, dtype=torch.float64)
        self._kx = kv[:, 0]
        self._ky = kv[:, 1]
        x = torch.arange(self.nx, dtype=torch.float64) * (self.L / self.nx)
        y = torch.arange(self.ny, dtype=torch.float64) * (self.W / self.ny)
        theta = 2.0 * math.pi * (
            self._kx[:, None, None] * x[None, None, :] / self.L
            + self._ky[:, None, None] * y[None, :, None] / self.W)
        patterns = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        patterns = patterns.reshape(self.n_amp, self.ny, self.nx)
        flat = patterns.reshape(self.n_amp, -1)
        self._proj = torch.linalg.pinv(flat.T)
        self._patterns64 = patterns
        self.to(dtype=dtype, device=device)

    def to(self, dtype: torch.dtype | None = None,
           device: torch.device | str | None = None) -> "FourierWindBasis":
        dtype = dtype or getattr(self, "patterns", self._patterns64).dtype
        device = device if device is not None else getattr(self, "patterns", self._patterns64).device
        self.patterns = self._patterns64.to(dtype=dtype, device=device)
        cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        self.spectral_patterns = torch.fft.rfft2(
            self._patterns64, dim=(-2, -1)).to(dtype=cdtype, device=device)
        self.proj = self._proj.to(dtype=dtype, device=device)
        return self

    @property
    def wavenumbers(self) -> torch.Tensor:
        return 2.0 * math.pi * torch.sqrt(
            (self._kx / self.L) ** 2 + (self._ky / self.W) ** 2)

    def curl_field(self, a: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...k,kyx->...yx", a.to(self.patterns), self.patterns)

    def curl_spectral(self, a: torch.Tensor) -> torch.Tensor:
        sp = self.spectral_patterns
        return torch.einsum("...k,kyx->...yx", a.to(sp.real.dtype).to(sp.dtype), sp)

    def project(self, field: torch.Tensor) -> torch.Tensor:
        flat = field.reshape(*field.shape[:-2], -1).to(self.proj)
        return flat @ self.proj.T

    def translate(self, a: torch.Tensor, dx, dy) -> torch.Tensor:
        dev = a.device
        dx = torch.as_tensor(dx, dtype=torch.float64, device=dev)
        dy = torch.as_tensor(dy, dtype=torch.float64, device=dev)
        phi = 2.0 * math.pi * (dx[..., None] * self._kx.to(dev) / self.L
                               + dy[..., None] * self._ky.to(dev) / self.W)
        pairs = a.to(torch.float64).reshape(*a.shape[:-1], -1, 2)
        c, s = pairs[..., 0], pairs[..., 1]
        cp, sp = torch.cos(phi), torch.sin(phi)
        out = torch.stack([c * cp - s * sp, c * sp + s * cp], dim=-1)
        return out.reshape(a.shape).to(a.dtype)

    def ricker_mode_std(self, sigma: float) -> torch.Tensor:
        k = self.wavenumbers
        ks2 = (k * sigma) ** 2
        radius = 2.0 * math.pi * sigma ** 2 * ks2 * torch.exp(-0.5 * ks2) / (self.L * self.W)
        per_wv = radius / math.sqrt(2.0)
        return per_wv.repeat_interleave(2)

    def ricker_amplitudes(self, A: torch.Tensor, xc: torch.Tensor, yc: torch.Tensor,
                          sigma: float) -> torch.Tensor:
        A = torch.as_tensor(A, dtype=torch.float64)
        dev = A.device
        k = self.wavenumbers.to(dev)
        ks2 = (k * sigma) ** 2
        radius = 2.0 * math.pi * sigma ** 2 * ks2 * torch.exp(-0.5 * ks2) / (self.L * self.W)
        xc = torch.as_tensor(xc, dtype=torch.float64, device=dev)
        yc = torch.as_tensor(yc, dtype=torch.float64, device=dev)
        phase = 2.0 * math.pi * (xc[..., None] * self._kx.to(dev) / self.L
                                 + yc[..., None] * self._ky.to(dev) / self.W)
        c = A[..., None] * radius * torch.cos(phase)
        s = A[..., None] * radius * torch.sin(phase)
        return torch.stack([c, s], dim=-1).reshape(*A.shape, self.n_amp)


def _unit_ou(n_steps: int, n_channels: int, dt: float, tau: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rho = math.exp(-dt / tau)
    kick = math.sqrt(1.0 - rho * rho)
    z = np.empty((n_steps, n_channels))
    z[0] = rng.normal(size=n_channels)
    for t in range(1, n_steps):
        z[t] = rho * z[t - 1] + kick * rng.normal(size=n_channels)
    return z


def _gyrostat_series(preset: str, mapping: str | None, n_steps: int, dt_days: float,
                     time_unit_days: float, seed: int, burnin_units: float,
                     surrogate: bool) -> np.ndarray:
    spec = get_preset(preset)
    system = GyrostatSystem(spec)
    mapping = mapping or next(iter(MODE_MAPPINGS.get(preset, {})), None)
    order = MODE_MAPPINGS.get(preset, {}).get(mapping)
    if order is None:
        raise ValueError(f"no mapping {mapping!r} for preset {preset!r}")
    n_gen = 4 * n_steps if surrogate else n_steps
    y = system.standardize(system.integrate(n_gen, dt_days / time_unit_days, seed=seed,
                                            burnin_units=burnin_units))
    if surrogate:
        y = phase_randomized_surrogate(y, seed=seed + 7919)
        y = y[(n_gen - n_steps) // 2:(n_gen - n_steps) // 2 + n_steps]
    return y[:, list(order)]


def default_time_unit_days(preset: str, tau_days: float) -> float:
    spec = get_preset(preset)
    if spec.efold_units is None:
        raise ValueError(f"preset {preset!r} has no measured e-folding time")
    return tau_days / spec.efold_units


def generate_spectral_wind(basis: FourierWindBasis, driver: str, n_steps: int,
                           dt_seconds: float, amp: float, sigma: float,
                           cx: float, cy: float, x0: float, y0: float, seed: int,
                           tau_days: float = 15.0, preset: str = "l63ring4",
                           mapping: str | None = None,
                           time_unit_days: float | None = None,
                           burnin_units: float = 50.0) -> torch.Tensor:
    if driver not in SPECTRAL_DRIVERS:
        raise ValueError(f"unknown spectral driver {driver!r}; known: {SPECTRAL_DRIVERS}")
    dt_days = dt_seconds / 86400.0
    if driver == "spectral_ou":
        z = _unit_ou(n_steps, basis.n_amp, dt_days, tau_days, seed)
    else:
        unit = time_unit_days or default_time_unit_days(preset, tau_days)
        z = _gyrostat_series(preset, mapping, n_steps, dt_days, unit, seed,
                             burnin_units, surrogate=(driver == "gyrostat_surrogate"))
        if z.shape[1] != basis.n_amp:
            raise ValueError(f"preset {preset!r} with mapping {mapping!r} gives "
                             f"{z.shape[1]} amplitudes, basis needs {basis.n_amp}")
    a = torch.from_numpy(z) * (amp * basis.ricker_mode_std(sigma))
    t = torch.arange(n_steps, dtype=torch.float64) * dt_seconds
    return basis.translate(a, x0 + cx * t, y0 + cy * t)
