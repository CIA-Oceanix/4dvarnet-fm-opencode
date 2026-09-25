"""Low-order chaotic drivers built from coupled Volterra gyrostats.

A gyrostat on modes ``(y1, y2, y3)`` has the energy-conserving core

    y1' = p y2 y3 + b y3 - c y2
    y2' = q y3 y1 + c y1 - a y3
    y3' = r y1 y2 + a y2 - b y1,      p + q + r = 0,

to which linear damping ``-kappa * y`` and a constant forcing ``F`` are added.
Lorenz-63 is one gyrostat after the shift ``Z = z - (rho + sigma)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class GyrostatSpec:
    name: str
    n_modes: int
    triads: np.ndarray
    coeffs: np.ndarray
    damping: np.ndarray
    forcing: np.ndarray
    mode_mean: np.ndarray | None = None
    mode_std: np.ndarray | None = None
    efold_units: float | None = None
    notes: dict = field(default_factory=dict)


def lorenz63_spec(sigma: float = 10.0, rho: float = 28.0,
                  beta: float = 8.0 / 3.0) -> GyrostatSpec:
    return GyrostatSpec(
        name="l63",
        n_modes=3,
        triads=np.array([[0, 1, 2]]),
        coeffs=np.array([[0.0, -1.0, 1.0, 0.0, 0.0, -sigma]]),
        damping=np.array([sigma, 1.0, beta]),
        forcing=np.array([0.0, 0.0, -beta * (rho + sigma)]),
        mode_mean=_L63_MEAN,
        mode_std=_L63_STD,
        efold_units=_L63_EFOLD,
    )


def lorenz63_ring_spec(n_copies: int = 4, coupling: float = 0.5,
                       sigma: float = 10.0, rho: float = 28.0,
                       beta: float = 8.0 / 3.0) -> GyrostatSpec:
    triads, coeffs = [], []
    damping = np.zeros(3 * n_copies)
    forcing = np.zeros(3 * n_copies)
    for k in range(n_copies):
        x, y, z = 3 * k, 3 * k + 1, 3 * k + 2
        triads.append([x, y, z])
        coeffs.append([0.0, -1.0, 1.0, 0.0, 0.0, -sigma])
        damping[[x, y, z]] = [sigma, 1.0, beta]
        forcing[z] = -beta * (rho + sigma)
    for k in range(n_copies):
        nxt = (k + 1) % n_copies
        triads.append([3 * k, 3 * nxt + 1, 3 * nxt + 2])
        coeffs.append([0.0, -coupling, coupling, 0.0, 0.0, 0.0])
    default = (n_copies == 4 and coupling == 0.5 and sigma == 10.0
               and rho == 28.0 and beta == 8.0 / 3.0)
    return GyrostatSpec(
        name="l63ring4" if default else f"l63ring{n_copies}",
        n_modes=3 * n_copies,
        triads=np.array(triads),
        coeffs=np.array(coeffs),
        damping=damping,
        forcing=forcing,
        mode_mean=_RING4_MEAN if default else None,
        mode_std=_RING4_STD if default else None,
        efold_units=_RING4_EFOLD if default else None,
    )


PRESETS = {"l63": lorenz63_spec, "l63ring4": lorenz63_ring_spec}


def get_preset(name: str) -> GyrostatSpec:
    if name not in PRESETS:
        raise ValueError(f"unknown gyrostat preset {name!r}; known: {sorted(PRESETS)}")
    return PRESETS[name]()


class GyrostatSystem:
    def __init__(self, spec: GyrostatSpec):
        self.spec = spec
        self._i = spec.triads[:, 0]
        self._j = spec.triads[:, 1]
        self._k = spec.triads[:, 2]
        self._p, self._q, self._r, self._a, self._b, self._c = spec.coeffs.T

    @property
    def n_modes(self) -> int:
        return self.spec.n_modes

    def core_rhs(self, y: np.ndarray) -> np.ndarray:
        y1, y2, y3 = y[..., self._i], y[..., self._j], y[..., self._k]
        d1 = self._p * y2 * y3 + self._b * y3 - self._c * y2
        d2 = self._q * y3 * y1 + self._c * y1 - self._a * y3
        d3 = self._r * y1 * y2 + self._a * y2 - self._b * y1
        out = np.zeros_like(y)
        np.add.at(out, (..., self._i), d1)
        np.add.at(out, (..., self._j), d2)
        np.add.at(out, (..., self._k), d3)
        return out

    def rhs(self, y: np.ndarray) -> np.ndarray:
        return self.core_rhs(y) - self.spec.damping * y + self.spec.forcing

    @staticmethod
    def energy(y: np.ndarray) -> np.ndarray:
        return 0.5 * np.sum(y * y, axis=-1)

    def _rk4(self, y: np.ndarray, h: float, core_only: bool = False) -> np.ndarray:
        f = self.core_rhs if core_only else self.rhs
        k1 = f(y)
        k2 = f(y + 0.5 * h * k1)
        k3 = f(y + 0.5 * h * k2)
        k4 = f(y + h * k3)
        return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def integrate(self, n_steps: int, dt_units: float, seed: int,
                  burnin_units: float = 50.0,
                  max_substep_units: float = 0.005) -> np.ndarray:
        rng = np.random.default_rng(seed)
        y = rng.normal(0.0, 5.0, self.n_modes)
        n_burn = int(math.ceil(burnin_units / max_substep_units))
        h_burn = burnin_units / max(n_burn, 1)
        for _ in range(n_burn):
            y = self._rk4(y, h_burn)
        n_sub = max(1, int(math.ceil(dt_units / max_substep_units)))
        h = dt_units / n_sub
        out = np.empty((n_steps, self.n_modes))
        for t in range(n_steps):
            out[t] = y
            for _ in range(n_sub):
                y = self._rk4(y, h)
        return out

    def standardize(self, y: np.ndarray) -> np.ndarray:
        if self.spec.mode_mean is None or self.spec.mode_std is None:
            raise ValueError(f"preset {self.spec.name!r} has no attractor constants; "
                             "measure them with measure_attractor_stats")
        return (y - self.spec.mode_mean) / self.spec.mode_std


def measure_attractor_stats(spec: GyrostatSpec, total_units: float = 2000.0,
                            dt_units: float = 0.005, seed: int = 0) -> dict:
    system = GyrostatSystem(spec)
    n = int(round(total_units / dt_units))
    y = system.integrate(n, dt_units, seed=seed)
    mean = y.mean(axis=0)
    std = y.std(axis=0)
    efold = np.empty(spec.n_modes)
    for m in range(spec.n_modes):
        efold[m] = _efold_time(y[:, m] - mean[m], dt_units)
    kurt = (((y - mean) / std) ** 4).mean(axis=0)
    return {"mean": mean, "std": std, "efold_units": efold, "kurtosis": kurt,
            "largest_lyapunov": largest_lyapunov(spec, seed=seed)}


def largest_lyapunov(spec: GyrostatSpec, total_units: float = 500.0,
                     dt_units: float = 0.005, renorm_every: int = 20,
                     seed: int = 0) -> float:
    system = GyrostatSystem(spec)
    rng = np.random.default_rng(seed + 1)
    y = system.integrate(1, dt_units, seed=seed)[0]
    d0 = 1e-8
    yp = y + d0 * rng.normal(0.0, 1.0, spec.n_modes) / math.sqrt(spec.n_modes)
    n = int(round(total_units / dt_units))
    log_sum = 0.0
    n_renorm = 0
    for i in range(n):
        y = system._rk4(y, dt_units)
        yp = system._rk4(yp, dt_units)
        if (i + 1) % renorm_every == 0:
            d = float(np.linalg.norm(yp - y))
            log_sum += math.log(d / d0)
            n_renorm += 1
            yp = y + (yp - y) * (d0 / d)
    return log_sum / (n_renorm * renorm_every * dt_units)


def _efold_time(v: np.ndarray, dt: float) -> float:
    n = v.shape[0]
    spec = np.fft.rfft(v, 2 * n)
    acf = np.fft.irfft(spec * np.conj(spec))[:n]
    acf = acf / acf[0]
    below = np.nonzero(acf < 1.0 / math.e)[0]
    return float(below[0] * dt) if below.size else float("nan")


def phase_randomized_surrogate(y: np.ndarray, seed: int) -> np.ndarray:
    n = y.shape[0]
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(y, axis=0)
    phases = rng.uniform(0.0, 2.0 * math.pi, spec.shape[0])
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0
    rotated = spec * np.exp(1j * phases)[:, None]
    return np.fft.irfft(rotated, n=n, axis=0)


_L63_MEAN = np.array([0.0, 0.0, -14.452])
_L63_STD = np.array([7.924, 9.011, 8.624])
_L63_EFOLD = 0.31
_RING4_MEAN = np.tile([0.0, 0.0, -10.195], 4)
_RING4_STD = np.tile([7.320, 8.758, 7.327], 4)
_RING4_EFOLD = 0.241
