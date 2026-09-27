"""S1 (model error) for the spectral-wind QG DA: corrupted wind and biased ocean parameters.

Four independent error components, the spectral analogues of the legacy QG
S1 (`data/qg.py`: +15% storm amplitude, OU amplitude noise, OU storm
location error, -15% rd and bottom drag):

* `amp`   - wind amplitude bias: every mode amplitude scaled by 1 + `amp_bias`;
* `noise` - random wind error: per-mode OU noise (correlation time `tau_days`)
  with std `noise_frac` x that mode's climatological RMS (`MODE_RMS`), so the
  error keeps the wind's spectrum and calm windows also get a wrong wind;
* `shift` - wind position error: the whole pattern translated by an OU
  displacement per axis with std `shift_frac` x domain size, i.e. a phase
  rotation of each (cos, sin) pair by 2 pi k . d;
* `param` - ocean parameter bias: rd and the bottom drag rek scaled by
  1 - `param_bias` (the eddy drag r_cf and U1 stay true).

The wind is corrupted as one continuous series over the lead buffer and the
window, so the bred initial ensemble is also integrated with the DA model's
(wrong) wind. Each component draws from its own generator seeded by the
window's dataset index, so switching one component on or off never changes
another's draw (common random numbers for the attribution).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch

S1_SEED = 20_260_927
COMPONENTS = ("amp", "noise", "shift", "param")
MODE_RMS = (3.39e-12,) * 4 + (1.96e-12,) * 4 + (3.36e-13,) * 4


@dataclass(frozen=True)
class S1Levels:
    amp_bias: float = 0.0
    noise_frac: float = 0.0
    shift_frac: float = 0.0
    param_bias: float = 0.0
    tau_days: float = 10.0

    def scaled(self, kappa: float) -> "S1Levels":
        return replace(self, amp_bias=kappa * self.amp_bias, noise_frac=kappa * self.noise_frac,
                       shift_frac=kappa * self.shift_frac, param_bias=kappa * self.param_bias)

    def only(self, components) -> "S1Levels":
        keep = set(components)
        unknown = keep - set(COMPONENTS)
        if unknown:
            raise ValueError(f"unknown S1 components {sorted(unknown)}")
        return replace(self, amp_bias=self.amp_bias if "amp" in keep else 0.0,
                       noise_frac=self.noise_frac if "noise" in keep else 0.0,
                       shift_frac=self.shift_frac if "shift" in keep else 0.0,
                       param_bias=self.param_bias if "param" in keep else 0.0)

    def as_dict(self) -> dict:
        return asdict(self)


REFERENCE = S1Levels(amp_bias=0.15, noise_frac=0.3, shift_frac=0.05, param_bias=0.15)


def _ou(n: int, tau_steps: float, rng: np.random.Generator, dims: int) -> np.ndarray:
    phi = math.exp(-1.0 / max(tau_steps, 1e-9))
    out = np.empty((n, dims))
    out[0] = rng.standard_normal(dims)
    innov = math.sqrt(1.0 - phi * phi)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + innov * rng.standard_normal(dims)
    return out


def _generators(key: int) -> dict:
    return {c: np.random.default_rng([S1_SEED, int(key), i]) for i, c in enumerate(COMPONENTS)}


def corrupt_amplitudes(amps: torch.Tensor, wavevectors: list[tuple[int, int]], levels: S1Levels,
                       dt: float, key: int) -> torch.Tensor:
    """Corrupted copy of a (T, 2 * n_k) amplitude series ([c_0, s_0, c_1, s_1, ...])."""
    n, m = amps.shape
    if m != 2 * len(wavevectors) or m != len(MODE_RMS):
        raise ValueError(f"expected {len(MODE_RMS)} amplitudes, got {m}")
    gens = _generators(key)
    tau_steps = levels.tau_days * 86400.0 / dt
    a = amps.double().clone()
    if levels.shift_frac:
        d = levels.shift_frac * _ou(n, tau_steps, gens["shift"], 2)
        k = np.asarray(wavevectors, dtype=np.float64)
        phi = torch.from_numpy(2.0 * math.pi * (d @ k.T)).to(a.device)
        c, s = a[:, 0::2].clone(), a[:, 1::2].clone()
        a[:, 0::2] = c * torch.cos(phi) - s * torch.sin(phi)
        a[:, 1::2] = c * torch.sin(phi) + s * torch.cos(phi)
    if levels.amp_bias:
        a = a * (1.0 + levels.amp_bias)
    if levels.noise_frac:
        eta = _ou(n, tau_steps, gens["noise"], m) * np.asarray(MODE_RMS)
        a = a + levels.noise_frac * torch.from_numpy(eta).to(a.device)
    return a.to(amps.dtype)


def s1_windows(windows: list[dict], levels: S1Levels, dt: float,
               wavevectors: list[tuple[int, int]]) -> list[dict]:
    """S0 windows (`evaluation.run_qg_specwind_da.s0_windows`) -> S1 windows.

    The truth, observations and metrics are unchanged; only the DA model's
    parameters (`da_params`) and wind (`wind_state_corrupted`, `wind_lead_da`)
    are corrupted.
    """
    out = []
    for w in windows:
        key = int(w["init_seed_key"])
        lead, win = w["wind_lead"], w["wind_state_true"]
        full = corrupt_amplitudes(torch.cat([lead, win]), wavevectors, levels, dt, key)
        da_params = dict(w["true_params"])
        da_params["rd"] = da_params["rd"] * (1.0 - levels.param_bias)
        da_params["rek"] = da_params["rek"] * (1.0 - levels.param_bias)
        out.append({**w, "da_params": da_params, "wind_lead_da": full[:len(lead)].to(lead.dtype),
                    "wind_state_corrupted": full[len(lead):].to(win.dtype),
                    "s1_levels": levels.as_dict()})
    return out
