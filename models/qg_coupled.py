"""Two-way coupled QG ocean + gyrostat atmosphere (Option C, G3).

The ocean (`BatchedQGDynamics`, with the relative-wind eddy drag
``-r_cf * zeta_1``) and the gyrostat atmosphere advance in one RK4 step. The
coupling follows decision 4 of `docs/plans/tech/qg_batched_generation_datasets.md`
(§8.1), from one bulk relative-wind stress:

* atmosphere -> ocean: the spectral PV source built from the gyrostat modes
  (the Option B mapping), held over the step like the one-way path;
* ocean -> atmosphere: ``da_k/dt += kappa_fb * gamma_a * r_cf * zeta_{o,k}`` on
  each wind amplitude, where ``zeta_{o,k}`` is the ocean's upper-layer
  vorticity projected on mode k (read from the Fourier coefficients of psi_1,
  rotated into the gyrostat's co-moving frame). In standardized gyrostat units
  the term is divided by the mode's wind scale ``s_k``; calm windows
  (``s_k = 0``) get no feedback.

``kappa_fb = 1`` is the physical coupling; ``kappa_fb > 1`` is a labelled,
non-physical sensitivity setting. With ``kappa_fb = 0`` the gyrostat update uses
the same arithmetic as `BatchedGyrostat`, so the run reproduces the one-way
Option B path (spectral forcing + eddy drag) bit for bit.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from models.gyrostat_driver import get_preset
from models.qg_batched import BatchedQGDynamics
from models.qg_wind_batched import BatchedGyrostat, _per_window, stream_seed
from models.qg_wind_modes import MODE_MAPPINGS, FourierWindBasis

GAMMA_A = 1.2e-6


class CoupledSpectralQG:
    def __init__(self, ocean: BatchedQGDynamics, basis: FourierWindBasis, amp, sigma: float,
                 cx, cy, x0, y0, time_unit_days, kappa_fb: float = 1.0,
                 gamma_a: float = GAMMA_A, preset: str = "l63ring4",
                 mapping: str | None = None, max_substep: float = 0.005):
        self.ocean = ocean
        self.basis = basis
        self.device = ocean.device
        B = ocean.batch
        self.batch = B
        self.kappa_fb = float(kappa_fb)
        self.gamma_a = float(gamma_a)
        spec = get_preset(preset)
        mapping = mapping or next(iter(MODE_MAPPINGS[preset]))
        self.order = list(MODE_MAPPINGS[preset][mapping])
        if len(self.order) != basis.n_amp:
            raise ValueError(f"preset {preset!r} gives {len(self.order)} amplitudes, basis needs {basis.n_amp}")
        self.gyro = BatchedGyrostat(spec, self.device)
        f64 = torch.float64
        dev = self.device
        self.scale = (_per_window(amp, B).to(dev).view(-1, 1)
                      * basis.ricker_mode_std(sigma).to(device=dev, dtype=f64).view(1, -1))
        self.x0 = _per_window(x0, B).to(dev)
        self.y0 = _per_window(y0, B).to(dev)
        self.cx = _per_window(cx, B).to(dev)
        self.cy = _per_window(cy, B).to(dev)
        self.dt_units = (ocean.dt / 86400.0) / _per_window(time_unit_days, B).to(dev)
        if float(self.dt_units.max()) > max_substep:
            raise ValueError("the ocean step exceeds the gyrostat substep limit; use a longer time unit")
        self.h = self.dt_units.view(-1, 1)
        self.unit_s = (_per_window(time_unit_days, B).to(dev) * 86400.0).view(-1, 1)
        self.r_cf = ocean.r_cf.to(device=dev, dtype=f64).view(-1, 1)
        inv = np.empty(len(self.order), dtype=np.int64)
        inv[np.asarray(self.order)] = np.arange(len(self.order))
        self.slot_of_mode = torch.as_tensor(inv, device=dev)
        rows = [ky % ocean.ny for kx, ky in basis.wavevectors]
        cols = [kx for kx, ky in basis.wavevectors]
        self._rows = torch.as_tensor(rows, device=dev)
        self._cols = torch.as_tensor(cols, device=dev)
        self._norm = 2.0 / float(ocean.nx * ocean.ny)
        safe = torch.where(self.scale > 0, self.scale, torch.ones_like(self.scale))
        self.inv_scale = torch.where(self.scale > 0, 1.0 / safe, torch.zeros_like(self.scale))

    def amplitudes(self, y: torch.Tensor, t: float) -> torch.Tensor:
        z = self.gyro.standardize(y)[:, self.order]
        a = z * self.scale
        tt = torch.as_tensor(t, dtype=torch.float64, device=self.device)
        return self.basis.translate(a, self.x0 + self.cx * tt, self.y0 + self.cy * tt)

    def vorticity_amplitudes(self, psih1: torch.Tensor) -> torch.Tensor:
        zeta = -(self.ocean.K2.to(psih1.real.dtype) * psih1)
        coef = zeta[:, self._rows, self._cols].to(torch.complex128)
        a_c = self._norm * coef.real
        a_s = -self._norm * coef.imag
        return torch.stack([a_c, a_s], dim=-1).reshape(self.batch, -1)

    def feedback_per_unit(self, psih1: torch.Tensor, t: float) -> torch.Tensor:
        if self.kappa_fb == 0.0:
            return torch.zeros((self.batch, len(self.order)), dtype=torch.float64, device=self.device)
        tt = torch.as_tensor(t, dtype=torch.float64, device=self.device)
        zeta_lab = self.vorticity_amplitudes(psih1)
        zeta_co = self.basis.translate(zeta_lab, -(self.x0 + self.cx * tt), -(self.y0 + self.cy * tt))
        per_slot = self.kappa_fb * self.gamma_a * self.r_cf * zeta_co * self.inv_scale
        per_mode = per_slot[:, self.slot_of_mode]
        return per_mode * self.gyro.std * self.unit_s

    def step(self, qh: torch.Tensor, y: torch.Tensor, t: float) -> tuple[torch.Tensor, torch.Tensor]:
        oc = self.ocean
        dt, h = oc.dt, self.h
        wind_h = self.basis.curl_spectral(self.amplitudes(y, t)).to(qh.dtype)
        c = (0.0, 0.5 * dt, 0.5 * dt, dt)

        def gyro_rate(q, yy, ts):
            return self.gyro.rhs(yy) + self.feedback_per_unit(oc._invert(q)[:, 0], ts)

        k1 = oc._tendency(qh, wind_h)
        g1 = gyro_rate(qh, y, t + c[0])
        q2, y2 = qh + 0.5 * dt * k1, y + 0.5 * h * g1
        k2 = oc._tendency(q2, wind_h)
        g2 = gyro_rate(q2, y2, t + c[1])
        q3, y3 = qh + 0.5 * dt * k2, y + 0.5 * h * g2
        k3 = oc._tendency(q3, wind_h)
        g3 = gyro_rate(q3, y3, t + c[2])
        q4, y4 = qh + dt * k3, y + h * g3
        k4 = oc._tendency(q4, wind_h)
        g4 = gyro_rate(q4, y4, t + c[3])
        qh_new = oc.filtr.to(qh.real.dtype).to(qh.dtype) * (qh + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
        if oc.clip_range is not None:
            q_real = torch.fft.irfft2(qh_new, s=(oc.ny, oc.nx), dim=(-2, -1))
            qh_new = torch.fft.rfft2(torch.clamp(q_real, -oc.clip_range, oc.clip_range), dim=(-2, -1))
        y_new = y + (h / 6.0) * (g1 + 2.0 * g2 + 2.0 * g3 + g4)
        return qh_new, y_new

    def rollout(self, q0: torch.Tensor, y0: torch.Tensor, n_steps: int, keep_from: int = 0,
                keep_every: int = 1, amps_from: int | None = None,
                diagnose_every: int = 0) -> dict:
        oc = self.ocean
        qh, y = torch.fft.rfft2(q0, dim=(-2, -1)), y0
        kept, amps, diag = [], [], []
        amps_from = keep_from if amps_from is None else amps_from
        for k in range(n_steps + 1):
            if k >= keep_from and (k - keep_from) % keep_every == 0:
                kept.append(torch.fft.irfft2(qh, s=(oc.ny, oc.nx), dim=(-2, -1)))
            if k == n_steps:
                break
            t = k * oc.dt
            if k >= amps_from:
                amps.append(self.amplitudes(y, t))
            if diagnose_every and k % diagnose_every == 0 and self.kappa_fb != 0.0:
                fb = self.feedback_per_unit(oc._invert(qh)[:, 0], t)
                own = self.gyro.rhs(y)
                diag.append((fb.pow(2).mean(1).sqrt() / own.pow(2).mean(1).sqrt()).cpu())
            qh, y = self.step(qh, y, t)
        out = {"frames": torch.stack(kept, dim=1),
               "amplitudes": torch.stack(amps, dim=1) if amps else None,
               "y_end": y}
        if diag:
            out["feedback_over_own_rms"] = torch.stack(diag, dim=1)
        return out


def generate_coupled_windows(spec, split: str, indices: list[int], kappa_fb: float = 1.0,
                             device: torch.device | str = "cpu", batch_size: int = 256,
                             dtype: torch.dtype = torch.float32, keep_every: int | None = None,
                             diagnose_every: int = 0) -> dict:
    """G2-format windows from the coupled system (forced-and-coupled spin-up)."""
    from data.qg_datasets import STREAM_OCEAN_IC, design_factors, window_seed

    device = torch.device(device)
    fac = design_factors(spec, split)
    keep = spec.keep_every(split) if keep_every is None else keep_every
    basis = FourierWindBasis(nx=spec.nx, L=spec.L, kmax=spec.kmax, dtype=dtype, device=device)
    t0_frame = spec.n_lead // keep
    out = {"true_state": [], "wind_amplitudes": [], "window_seed": [], "feedback_ratio": []}
    t_start = time.time()
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        seeds = [window_seed(spec, split, i) for i in idx]
        pick = lambda name: fac[name][idx]  # noqa: E731
        ocean = BatchedQGDynamics(len(idx), nx=spec.nx, L=spec.L, dt=spec.dt, beta=spec.beta,
                                  rd=pick("rd"), delta=spec.delta, U1=pick("U1"), U2=spec.U2,
                                  rek=pick("rek"), r_cf=spec.r_cf, dtype=dtype, device=device)
        model = CoupledSpectralQG(ocean, basis, amp=pick("level"), sigma=spec.sigma, cx=pick("cx"),
                                  cy=pick("cy"), x0=pick("x0"), y0=pick("y0"),
                                  time_unit_days=pick("time_unit_days"), kappa_fb=kappa_fb,
                                  preset=spec.preset)
        y0 = model.gyro.burn_in(model.gyro.initial_states(seeds), spec.burnin_units)
        q0 = ocean.initial_q([stream_seed(s, STREAM_OCEAN_IC) for s in seeds])
        res = model.rollout(q0, y0, spec.n_total, keep_from=spec.n_spinup, keep_every=keep,
                            diagnose_every=diagnose_every)
        out["true_state"].append(res["frames"].to(torch.float32).cpu())
        out["wind_amplitudes"].append(res["amplitudes"].to(torch.float32).cpu())
        out["window_seed"].extend(seeds)
        if "feedback_over_own_rms" in res:
            out["feedback_ratio"].append(res["feedback_over_own_rms"].mean(1))
    return {
        "indices": list(indices),
        "true_state": torch.cat(out["true_state"]),
        "wind_amplitudes": torch.cat(out["wind_amplitudes"]),
        "window_seed": out["window_seed"],
        "factors": {k: np.asarray(v[indices]) for k, v in fac.items() if k != "unit"},
        "keep_every": keep, "window_start_frame": t0_frame, "kappa_fb": float(kappa_fb),
        "feedback_ratio": torch.cat(out["feedback_ratio"]) if out["feedback_ratio"] else None,
        "seconds": time.time() - t_start,
    }
