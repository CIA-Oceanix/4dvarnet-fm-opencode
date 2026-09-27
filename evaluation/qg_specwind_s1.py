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

The realism-anchored scenario (`REALISTIC`) adds components the four above
cannot represent and splits the parameter error:

* `rd`    - rd scaled by 1 + `rd_bias` (stratification known to ~10%);
* `drag`  - bottom drag rek scaled by 1 + `drag_bias` (uncertain by ~2x);
* `obs`   - altimetry error beyond the S0 5% white noise: extra white noise up
  to a total of `obs_white_frac`, plus a correlated per-pass error (offset
  and tilt along the observed column, RMS `obs_corr_frac`), both as
  fractions of the upper-layer psi std. The filter's R is the total
  variance, diagonal (the correlation is not represented, as in practice);
* `res`   - structural error: the DA model runs on a `da_nx` x `da_nx` grid
  (coarser than the 64 x 64 truth; the observation operator upsamples).

`forcing` groups `amp`, `noise` and `shift` for the attribution.

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
REALISTIC_GROUPS = ("forcing", "rd", "drag", "obs", "res")
SEED_ORDER = ("amp", "noise", "shift", "param", "rd", "drag", "obs", "res")
FIELDS_OF = {"amp": ("amp_bias",), "noise": ("noise_frac",), "shift": ("shift_frac",),
             "param": ("param_bias",), "rd": ("rd_bias",), "drag": ("drag_bias",),
             "obs": ("obs_white_frac", "obs_corr_frac"), "res": ("da_nx",),
             "forcing": ("amp_bias", "noise_frac", "shift_frac")}
S0_OBS_WHITE = 0.05
MODE_RMS = (3.39e-12,) * 4 + (1.96e-12,) * 4 + (3.36e-13,) * 4


@dataclass(frozen=True)
class S1Levels:
    amp_bias: float = 0.0
    noise_frac: float = 0.0
    shift_frac: float = 0.0
    param_bias: float = 0.0
    tau_days: float = 10.0
    rd_bias: float = 0.0
    drag_bias: float = 0.0
    obs_white_frac: float = 0.0
    obs_corr_frac: float = 0.0
    da_nx: int = 0

    def scaled(self, kappa: float) -> "S1Levels":
        return replace(self, amp_bias=kappa * self.amp_bias, noise_frac=kappa * self.noise_frac,
                       shift_frac=kappa * self.shift_frac, param_bias=kappa * self.param_bias)

    def only(self, components) -> "S1Levels":
        keep = set(components)
        unknown = keep - set(FIELDS_OF)
        if unknown:
            raise ValueError(f"unknown S1 components {sorted(unknown)}")
        kept = {f for c in keep for f in FIELDS_OF[c]}
        zero = {f: (0 if f == "da_nx" else 0.0) for fs in FIELDS_OF.values() for f in fs
                if f not in kept}
        return replace(self, **zero)

    def as_dict(self) -> dict:
        return asdict(self)


REFERENCE = S1Levels(amp_bias=0.15, noise_frac=0.3, shift_frac=0.05, param_bias=0.15)
REALISTIC = S1Levels(amp_bias=0.15, noise_frac=0.25, shift_frac=0.05, rd_bias=-0.10,
                     drag_bias=-0.50, obs_white_frac=0.15, obs_corr_frac=0.15, da_nx=32)
REALISTIC_VARIANTS = {
    "base": REALISTIC,
    "res48": replace(REALISTIC, da_nx=48),
    "res64": replace(REALISTIC, da_nx=0),
    "low": S1Levels(amp_bias=0.10, noise_frac=0.15, shift_frac=0.03, rd_bias=-0.05,
                    drag_bias=-0.30, obs_white_frac=0.10, obs_corr_frac=0.10, da_nx=48),
    "high": S1Levels(amp_bias=0.20, noise_frac=0.35, shift_frac=0.07, rd_bias=-0.15,
                     drag_bias=-0.60, obs_white_frac=0.20, obs_corr_frac=0.20, da_nx=32),
}
REALISTIC_SELECTED = "high"


def _ou(n: int, tau_steps: float, rng: np.random.Generator, dims: int) -> np.ndarray:
    phi = math.exp(-1.0 / max(tau_steps, 1e-9))
    out = np.empty((n, dims))
    out[0] = rng.standard_normal(dims)
    innov = math.sqrt(1.0 - phi * phi)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + innov * rng.standard_normal(dims)
    return out


def _generators(key: int) -> dict:
    return {c: np.random.default_rng([S1_SEED, int(key), i]) for i, c in enumerate(SEED_ORDER)}


def corrupt_obs(obs: torch.Tensor, signal_std: float, levels: S1Levels, key: int) -> torch.Tensor:
    """Add the `obs` component's extra altimetry error to a (T, ny) NaN-padded obs array.

    Extra white noise brings the total white std to `obs_white_frac` (the S0
    obs already carry `S0_OBS_WHITE`); each observed column (pass) also gets
    an offset plus a linear tilt with total RMS `obs_corr_frac`.
    """
    if not (levels.obs_white_frac or levels.obs_corr_frac):
        return obs
    rng = _generators(key)["obs"]
    t_obs = (~torch.isnan(obs)).any(dim=1).nonzero().flatten().tolist()
    ny = obs.shape[1]
    out = obs.double().clone()
    extra = math.sqrt(max(levels.obs_white_frac ** 2 - S0_OBS_WHITE ** 2, 0.0)) * signal_std
    tilt = torch.linspace(-1.0, 1.0, ny, dtype=torch.float64)
    c = levels.obs_corr_frac * signal_std
    for t in t_obs:
        e = extra * torch.from_numpy(rng.standard_normal(ny))
        a, b = rng.standard_normal(2)
        e = e + c * (math.sqrt(0.5) * a + math.sqrt(1.5) * b * tilt)
        out[t] = out[t] + e
    return out.to(obs.dtype)


def obs_error_frac(levels: S1Levels) -> float:
    """Total observation-error std (fraction of the signal std) the filter's R should assume."""
    white = max(levels.obs_white_frac, S0_OBS_WHITE)
    return math.sqrt(white ** 2 + levels.obs_corr_frac ** 2)


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

    The truth and metrics are unchanged. The DA model's parameters
    (`da_params`), wind (`wind_state_corrupted`, `wind_lead_da`) and grid
    (`da_nx`) are corrupted, and the `obs` component adds altimetry error to
    the observations. Run the filter with `obs_error_frac(levels)` as its
    observation-error fraction.
    """
    out = []
    for w in windows:
        key = int(w["init_seed_key"])
        lead, win = w["wind_lead"], w["wind_state_true"]
        full = corrupt_amplitudes(torch.cat([lead, win]), wavevectors, levels, dt, key)
        da_params = dict(w["true_params"])
        da_params["rd"] = da_params["rd"] * (1.0 - levels.param_bias) * (1.0 + levels.rd_bias)
        da_params["rek"] = da_params["rek"] * (1.0 - levels.param_bias) * (1.0 + levels.drag_bias)
        extra = {"da_nx": int(levels.da_nx)} if levels.da_nx else {}
        out.append({**w, **extra, "da_params": da_params,
                    "wind_lead_da": full[:len(lead)].to(lead.dtype),
                    "wind_state_corrupted": full[len(lead):].to(win.dtype),
                    "obs": corrupt_obs(w["obs"], float(w["target_state_psi"].std()), levels, key),
                    "s1_levels": levels.as_dict()})
    return out
