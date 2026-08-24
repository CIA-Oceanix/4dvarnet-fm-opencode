import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.qg import QGConfig, make_qg_s0_s1_datasets
from evaluation.baselines import ETKF, EnKF, ObsOperator, _build_qg_loc_matrices
from models.dynamics import DynamicsBase
from models.qg1l_dynamics import QG1LDynamics
from models.qg_dynamics import QGDynamics

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class WindStateAdapter(DynamicsBase):
    """Adapter exposing QG wind as the generic baseline `forcing` channel.

    The generic filters call `dynamics.step(state, W, **params)` with a
    per-step forcing `W`; QG needs `wind_state_t=(A, xc, yc)` instead. This
    forwards `W` as the wind-state row and drops the L63 default params.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.state_dim = inner.state_dim
        self.param_dim = inner.param_dim
        self.param_names = inner.param_names
        self.forcing_dim = inner.forcing_dim

    def step(self, state, forcing=None, *args, **kwargs):
        return self.inner.step(state, wind_state_t=forcing)

    def rollout_trajectory(self, state, steps, wind_state=None, **kwargs):
        return self.inner.rollout_trajectory(
            state, steps, wind_state=wind_state, **kwargs)

    def rollout(self, x0, forcing, steps, *args, **kwargs):
        traj = [x0]
        for k in range(1, steps):
            traj.append(self.inner.step(traj[-1], wind_state_t=forcing[..., k - 1]))
        return torch.stack(traj, dim=-2), forcing


def _build_dyn(cfg, window, device):
    da_params = window["da_params"]
    model = window["da_model"]
    common = {
        "nx": cfg.nx, "L": cfg.L, "dt": cfg.dt, "beta": da_params["beta"],
        "rd": da_params["rd"], "U1": da_params["U1"], "rek": da_params["rek"],
        "filterfac": cfg.filterfac,
        "wind_amp": window["wind_amp"], "wind_sigma": cfg.wind_sigma,
        "clip_range": 1e-3,
    }
    if model == "qg1l":
        inner = QG1LDynamics(**common)
    else:
        inner = QGDynamics(**common, delta=cfg.delta,
                           U2=da_params.get("U2", cfg.U2))
    return WindStateAdapter(inner.to(device))


def _upper_indices(cfg):
    nx = cfg.nx
    return [y * nx + x for y in range(cfg.ny) for x in range(nx)]


def _q_alongtrack_obs(cfg, window, device):
    """Upper-layer PV-anomaly along-track obs (state-consistent index obs).

    Returns (obs (T, ny), R_var, field_std).
    """
    T, ny, nx = cfg.num_steps, cfg.ny, cfg.nx
    q1 = window["target_state_q"].reshape(T, ny, nx)
    field_std = float(q1.std())
    sigma = cfg.obs_noise_std_frac * field_std
    rng = torch.Generator().manual_seed(cfg.seed + 8000)
    obs = torch.full((T, ny), float("nan"))
    track = window["track_x_index"]
    for t in window["obs_mask"].nonzero(as_tuple=False).flatten().tolist():
        x_col = int(track[t])
        obs[t] = q1[t, :, x_col] + sigma * torch.randn(ny, generator=rng)
    return obs.to(device), sigma ** 2, field_std


def _per_pass_indices(cfg, window):
    """T-length list of upper-layer flat indices (None off-pass) + first pass."""
    ny, nx = cfg.ny, cfg.nx
    first_pass = None
    per_time = [None] * cfg.num_steps
    for t in window["obs_mask"].nonzero(as_tuple=False).flatten().tolist():
        x_col = int(window["track_x_index"][t])
        idx = [y * nx + x_col for y in range(ny)]
        per_time[t] = idx
        if first_pass is None:
            first_pass = idx
    return per_time, first_pass


def _evaluate_window(cfg, window, method, device, obs=None, forcing=None):
    if obs is None:
        obs, _, _ = _q_alongtrack_obs(cfg, window, device)
    mask = window["obs_mask"].to(device)
    truth = window["true_state"].to(device)
    if forcing is None:
        forcing = window["wind_state_corrupted"].to(device)
    result = method.assimilate(obs, mask, forcing, true_state=truth)
    return result


def _pooled_expvar(analyses, refs):
    sq = np.concatenate([(a - r) ** 2 for a, r in zip(analyses, refs)], axis=0)
    ref_all = np.concatenate(refs, axis=0)
    var = np.maximum(np.var(ref_all, axis=0), 1e-12)
    return 1.0 - np.mean(sq, axis=0) / var


def _free_forecast_rmse(cfg, dyn, window, device, forcing):
    """RMSE of the no-obs model forecast (roll from true t=0 state)."""
    truth = window["true_state"].float()
    ic = truth[0].to(device)
    roll = dyn.rollout_trajectory(ic, cfg.num_steps - 1, wind_state=forcing)
    return float(np.sqrt(np.mean((roll.detach().cpu().numpy()
                                  - truth.numpy()) ** 2)))


def run(method_name, cfg, device=None, N_ensemble=60, inflation=1.05,
        loc_radius=None, scenarios=("test_s0", "test_s1a", "test_s1b"),
        out_path=None):
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    cfg = QGConfig(nx=cfg.nx, window_days=cfg.window_days,
                   spinup_years=cfg.spinup_years, num_windows=cfg.num_windows,
                   obs_geometry="alongtrack", seed=7)
    ds = make_qg_s0_s1_datasets(cfg)

    per_layer = cfg.ny * cfg.nx
    nlayers = 2
    summary = {}
    for scen in scenarios:
        if scen not in ds:
            continue
        rmse_list = []
        fcast_rmse = []
        analyses = []
        refs = []
        d = ds[scen]
        for i in range(len(d)):
            w = d[i]
            dyn = _build_dyn(cfg, w, device)
            obs, r_var, field_std = _q_alongtrack_obs(cfg, w, device)
            forcing = w["wind_state_corrupted"].to(device)
            per_time, first_pass = _per_pass_indices(cfg, w)
            obs_op = ObsOperator(dyn.state_dim, obs_indices=first_pass,
                                 obs_indices_t=per_time)
            Lx_t = Ly_t = None
            if loc_radius is not None:
                Lx_t, Ly_t = _build_qg_loc_matrices(
                    dyn.state_dim, per_time, nlayers, cfg.ny, cfg.nx,
                    loc_radius, device)
            if method_name == "enkf":
                method = EnKF(N_ensemble=N_ensemble, R_var=r_var,
                              inflation=inflation, device=device, dynamics=dyn,
                              obs_operator=obs_op, loc_radius=loc_radius,
                              noise_init_std=field_std,
                              loc_Lx_t=Lx_t, loc_Ly_t=Ly_t)
            else:
                method = ETKF(N_ensemble=N_ensemble, R_var=r_var,
                              inflation=inflation, device=device, dynamics=dyn,
                              obs_operator=obs_op, loc_radius=loc_radius,
                              noise_init_std=field_std,
                              loc_Lx_t=Lx_t, loc_Ly_t=Ly_t)
            res = _evaluate_window(cfg, w, method, device, obs=obs,
                                   forcing=forcing)
            ref = w["true_state"].numpy()
            analyses.append(res.trajectory)
            refs.append(ref)
            rmse_list.append(float(np.sqrt(np.mean(
                (res.trajectory - ref) ** 2))))
            fcast_rmse.append(_free_forecast_rmse(
                cfg, dyn, w, device, forcing))
        ev = _pooled_expvar(analyses, refs)
        ev_upper = _pooled_expvar(
            [a[:, :per_layer] for a in analyses],
            [r[:, :per_layer] for r in refs])
        da_r = float(np.mean(rmse_list))
        fc_r = float(np.mean(fcast_rmse))
        summary[scen] = {
            "rmse_mean": da_r,
            "rmse_list": rmse_list,
            "forecast_rmse_mean": fc_r,
            "forecast_improvement": fc_r / max(da_r, 1e-30),
            "expvar_full": float(np.mean(ev)),
            "expvar_upper_q": float(np.mean(ev_upper)),
        }
        print(f"{scen}: rmse={da_r:.3e} forecast_rmse={fc_r:.3e} "
              f"improv={summary[scen]['forecast_improvement']:.2f}x "
              f"ev_full={summary[scen]['expvar_full']:.3f}")

    payload = {"method": method_name, "nx": cfg.nx,
               "N_ensemble": N_ensemble, "inflation": inflation,
               "loc_radius": loc_radius, "scenarios": summary}
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved {out_path}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["enkf", "etkf"], default="etkf")
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--num-windows", type=int, default=3)
    ap.add_argument("--window-days", type=float, default=20.0)
    ap.add_argument("--spinup-years", type=float, default=0.3)
    ap.add_argument("--ensemble", type=int, default=60)
    ap.add_argument("--inflation", type=float, default=1.05)
    ap.add_argument("--loc-radius", type=float, default=None)
    ap.add_argument("--scenarios", default="test_s0,test_s1a,test_s1b")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    cfg = QGConfig(nx=args.nx, window_days=args.window_days,
                   spinup_years=args.spinup_years, num_windows=args.num_windows,
                   obs_geometry="alongtrack", seed=7)
    print(f"device={device}")
    run(args.method, cfg, device=device, N_ensemble=args.ensemble,
        inflation=args.inflation, loc_radius=args.loc_radius,
        scenarios=tuple(args.scenarios.split(",")), out_path=args.out)


if __name__ == "__main__":
    main()
