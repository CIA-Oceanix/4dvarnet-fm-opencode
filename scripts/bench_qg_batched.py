"""Benchmark batched QG window generation (ocean + spectral wind) on one device.

Measures, for a batch of windows with per-window ocean parameters:
  * wind generation (gyrostat burn-in + integration, or spectral OU),
  * the forced ocean rollout (spectral forcing + eddy drag),
  * the end-to-end cost per window for the forced 2-year spin-up + window,
and checks whether a window's ocean trajectory depends on its batch.
"""
import argparse
import json
import time

import numpy as np
import torch

from models.qg_batched import BatchedQGDynamics, SpectralWindForcing
from models.qg_wind_batched import batched_spectral_wind
from models.qg_wind_modes import FourierWindBasis


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _model(batch, rng, args, device, rd=None, u1=None, rek=None):
    rd = rng.uniform(0.85, 1.15, batch) * 15000.0 if rd is None else rd
    u1 = rng.uniform(0.85, 1.15, batch) * 0.05 if u1 is None else u1
    rek = rng.uniform(0.85, 1.15, batch) * 5.787e-7 if rek is None else rek
    return BatchedQGDynamics(batch, nx=args.nx, dt=args.dt, rd=rd, U1=u1, rek=rek,
                             r_cf=args.r_cf, dtype=torch.float32, device=device), (rd, u1, rek)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--dt", type=float, default=7200.0)
    p.add_argument("--driver", default="gyrostat")
    p.add_argument("--spinup-years", type=float, default=2.0)
    p.add_argument("--window-steps", type=int, default=481)
    p.add_argument("--bench-steps", type=int, default=200)
    p.add_argument("--r-cf", type=float, default=2.8e-8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--full", action="store_true", help="also run one full spin-up + window batch")
    args = p.parse_args()
    device = torch.device(args.device)
    rng = np.random.default_rng(0)
    B = args.batch
    spd = round(86400.0 / args.dt)
    n_total = int(args.spinup_years * 365 * spd) + args.window_steps
    basis = FourierWindBasis(nx=args.nx, kmax=2, dtype=torch.float32, device=device)
    basis64 = FourierWindBasis(nx=args.nx, kmax=2, device=device)
    seeds = list(range(1000, 1000 + B))
    out = {"device": str(device), "batch": B, "driver": args.driver}

    t0 = time.perf_counter()
    amps = batched_spectral_wind(basis64, args.driver, seeds, args.bench_steps, args.dt,
                                 amp=rng.uniform(0, 3e-11, B), cx=rng.uniform(0.25, 0.75, B),
                                 cy=rng.uniform(-0.06, 0.06, B), x0=rng.uniform(0, 1e6, B),
                                 y0=rng.uniform(0, 1e6, B), sigma=2.5e5,
                                 time_unit_days=rng.uniform(30, 90, B), device=device)
    _sync(device)
    out["wind_s_burnin_plus_bench_steps"] = time.perf_counter() - t0

    model, _ = _model(B, rng, args, device)
    q0 = model.initial_q(seeds)
    forcing = SpectralWindForcing(basis, amps.float())
    model.rollout(q0, 5, forcing=forcing)
    _sync(device)
    t0 = time.perf_counter()
    model.rollout(q0, args.bench_steps - 1, forcing=forcing, keep_every=args.bench_steps)
    _sync(device)
    step = (time.perf_counter() - t0) / (args.bench_steps - 1)
    out["ocean_ms_per_batch_step"] = 1e3 * step
    out["ocean_ms_per_window_step"] = 1e3 * step / B
    out["est_ocean_s_per_window_full"] = n_total * step / B

    one, _ = _model(1, np.random.default_rng(0), args, device,
                    rd=np.array([model.rd[3].item()]), u1=np.array([model.U1[3].item()]),
                    rek=np.array([model.rek[3].item()]))
    _, qb = model.rollout(q0, 50, forcing=forcing)
    _, q1 = one.rollout(q0[3:4], 50, forcing=SpectralWindForcing(basis, amps[3:4].float()))
    out["batch_vs_alone_max_rel_diff_after_50_steps"] = float(
        (qb[3] - q1[0]).abs().max() / q1[0].abs().max())

    if args.full:
        t0 = time.perf_counter()
        amps_full = batched_spectral_wind(basis64, args.driver, seeds, n_total, args.dt,
                                          amp=rng.uniform(0, 3e-11, B),
                                          cx=rng.uniform(0.25, 0.75, B),
                                          cy=rng.uniform(-0.06, 0.06, B),
                                          x0=rng.uniform(0, 1e6, B), y0=rng.uniform(0, 1e6, B),
                                          sigma=2.5e5, time_unit_days=rng.uniform(30, 90, B),
                                          device=device)
        _sync(device)
        t_wind = time.perf_counter() - t0
        t0 = time.perf_counter()
        traj, _ = model.rollout(q0, n_total, forcing=SpectralWindForcing(basis, amps_full.float()),
                                keep_from=n_total - args.window_steps + 1, keep_every=6)
        _sync(device)
        t_ocean = time.perf_counter() - t0
        out.update({"full_wind_s": t_wind, "full_ocean_s": t_ocean,
                    "full_s_per_window": (t_wind + t_ocean) / B,
                    "full_kept_frames": int(traj.shape[1]),
                    "full_finite": bool(torch.isfinite(traj).all())})
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
