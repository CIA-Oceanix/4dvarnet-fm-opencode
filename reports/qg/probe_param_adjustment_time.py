"""Empirical diagnostic: how many days does the flow need, after switching
from nominal to perturbed (rd, rek, U1) parameters, to reach a representative
statistical equilibrium for the NEW parameters?

For each of a few representative (u, r, k) draws at the edges/middle of the
+-15% range:
  1. Build a "ground truth" reference: a full independent spin-up (2 years)
     AT the perturbed parameters, then record kinetic energy (KE) and
     enstrophy over the last `--ref-days` days -- this is what a fully
     equilibrated perturbed-parameter flow actually looks like.
  2. Build an "adjustment" trajectory: starting from a NOMINAL-parameter
     equilibrium state (its own 2-year spin-up, done once, shared across all
     test draws), switch to the perturbed parameters and roll forward for
     `--test-days` days, recording KE/enstrophy at every step.
  3. Report when the adjustment trajectory's KE/enstrophy enters and stays
     within +-2 reference-std of the reference mean -- a data-driven N.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data.qg import QGConfig
from models.qg_dynamics import QGDynamics

TEST_DRAWS = {
    "high": (1.15, 1.15, 1.15),
    "low": (0.85, 0.85, 0.85),
    "mixed": (1.15, 0.85, 1.15),
    "mid": (1.0, 1.0, 1.0),  # control: no perturbation, should need ~0 days
}


def build_dyn(cfg, u, r, k, device):
    return QGDynamics(
        nx=cfg.nx, L=cfg.L, dt=cfg.dt, beta=cfg.beta, rd=cfg.rd * r,
        delta=cfg.delta, U1=cfg.U1 * u, U2=cfg.U2, rek=cfg.rek * k,
        filterfac=cfg.filterfac, wind_amp=1e-11, wind_tau_days=cfg.wind_tau_days,
        wind_sigma=cfg.wind_sigma, wind_cx=0.5, wind_cy=0.03,
        wind_drift_tau_days=cfg.wind_drift_tau_days,
        wind_drift_sigma=cfg.wind_drift_sigma, wind_seed=1234,
    ).to(device)


def bulk_stats(dyn, state):
    ke = float(dyn.kinetic_energy(state))
    ens = float(dyn.enstrophy(state))
    return ke, ens


def rollout_from(dyn, x0, steps, seed):
    wind = dyn.generate_wind_state(steps, seed=seed)
    traj_ke, traj_ens = [], []
    state = x0
    for t in range(steps):
        ke, ens = bulk_stats(dyn, state)
        traj_ke.append(ke)
        traj_ens.append(ens)
        state = dyn.step(state, wind_state_t=wind[t])
    return np.array(traj_ke), np.array(traj_ens)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = QGConfig(nx=64, dt=7200.0)
    steps_per_day = round(86400.0 / cfg.dt)
    spinup_days = 730  # 2 years
    ref_days = 90
    test_days = 200
    spinup_steps = spinup_days * steps_per_day
    ref_steps = ref_days * steps_per_day
    test_steps = test_days * steps_per_day

    print(f"device={device} steps_per_day={steps_per_day}")

    # Shared nominal-equilibrium state (spin-up once).
    t0 = time.time()
    nominal_dyn = build_dyn(cfg, 1.0, 1.0, 1.0, device)
    traj_nom, _ = nominal_dyn.generate_full_trajectory(
        num_steps=1, seed=42, spinup_steps=spinup_steps)
    x0_nominal = traj_nom[0].to(device)
    print(f"nominal 2yr spin-up: {time.time() - t0:.1f}s")

    for name, (u, r, k) in TEST_DRAWS.items():
        t0 = time.time()
        dyn = build_dyn(cfg, u, r, k, device)

        # Ground-truth equilibrium reference at these perturbed params.
        traj_ref, _ = dyn.generate_full_trajectory(
            num_steps=ref_steps, seed=42, spinup_steps=spinup_steps)
        ref_ke = np.array([float(dyn.kinetic_energy(traj_ref[t])) for t in range(ref_steps)])
        ref_ens = np.array([float(dyn.enstrophy(traj_ref[t])) for t in range(ref_steps)])
        ref_ke_mean, ref_ke_std = ref_ke.mean(), ref_ke.std()
        ref_ens_mean, ref_ens_std = ref_ens.mean(), ref_ens.std()

        # Adjustment trajectory from the shared nominal-equilibrium state.
        adj_ke, adj_ens = rollout_from(dyn, x0_nominal, test_steps, seed=99)

        # First day the adjustment trajectory enters AND STAYS within +-2 std
        # of the reference mean for both KE and enstrophy.
        ke_ok = np.abs(adj_ke - ref_ke_mean) < 2 * ref_ke_std
        ens_ok = np.abs(adj_ens - ref_ens_mean) < 2 * ref_ens_std
        both_ok = ke_ok & ens_ok
        n_day = None
        for day in range(test_days):
            start = day * steps_per_day
            if both_ok[start:].all():
                n_day = day
                break

        print(f"[{name}] u={u} r={r} k={k}  wall={time.time()-t0:.1f}s  "
              f"ref_KE={ref_ke_mean:.4g}+-{ref_ke_std:.2g}  "
              f"ref_ens={ref_ens_mean:.4g}+-{ref_ens_std:.2g}  "
              f"adj_KE_day0={adj_ke[0]:.4g} adj_KE_final={adj_ke[-1]:.4g}  "
              f"N_days={n_day}")


if __name__ == "__main__":
    main()
