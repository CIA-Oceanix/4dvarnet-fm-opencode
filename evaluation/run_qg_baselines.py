import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.qg import QGConfig, make_qg_s0_s1_datasets
from evaluation.baselines import (
    ETKF,
    EnKF,
    ObsOperator,
    _build_qg_col_loc_matrices,
    _build_qg_loc_matrices,
)
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
    """Upper-layer PV-anomaly obs for either geometry.
    
    Returns (obs, R_var).
    """
    T, ny, nx = cfg.num_steps, cfg.ny, cfg.nx
    q1 = window["target_state_q"].reshape(T, ny, nx)
    field_std = float(q1.std())
    sigma = cfg.obs_noise_std_frac * field_std
    
    if "track_x_index" in window:
        obs = torch.full((T, ny), float("nan"))
        track = window["track_x_index"]
        obs_mask = window["obs_mask"]
        obs_steps = obs_mask.nonzero(as_tuple=False).flatten().tolist()
        for t in obs_steps:
            x_col = int(track[t])
            obs[t] = q1[t, :, x_col] + sigma * torch.randn(ny, generator=torch.Generator().manual_seed(cfg.seed + 8000))
    elif "obs_columns" in window:
        cols_t = window["obs_columns"]
        C = cols_t.shape[1]
        obs = torch.full((T, C * ny), float("nan"))
        obs_mask = window["obs_mask"]
        obs_steps = obs_mask.nonzero(as_tuple=False).flatten().tolist()
        rng = torch.Generator().manual_seed(cfg.seed + 8000)
        for t in obs_steps:
            for ci, x_col in enumerate(cols_t[t].tolist()):
                if 0 <= x_col < nx:
                    obs[t, ci * ny:(ci + 1) * ny] = q1[t, :, x_col] + sigma * torch.randn(ny, generator=rng)
    else:
        obs = torch.full((T, ny), float("nan"))
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


def _event_columns(cfg, window):
    """Extract column lists per time for random_columns geometry."""
    return window["obs_columns"]


def _psi_h(dyn, obs_cols, ny, device):
    """H-function for obs of upper-layer streamfunction columns.

    Args:
        dyn: QG dynamics (access via dyn.inner.streamfunctions)
        obs_cols: list of column lists per time (e.g., [[0,3], None, [2]])
        ny: ny of grid
        device: torch device

    Returns:
        Callable that takes (state (state_dim,), index=int) -> (C*ny,)
    """
    def h(state, index=None):
        batch = state.ndim > 1
        cols = obs_cols[index]
        if not batch:
            psi1 = dyn.inner.streamfunctions(state)
            return torch.cat([psi1[0, :, c] for c in cols])
        else:
            psi1 = dyn.inner.streamfunctions(state)
            C = len(cols)
            o = C * dyn.inner.ny
            stacked = torch.stack([psi1[:, 0, :, c] for c in cols], dim=1)
            return stacked.reshape(state.shape[0], o)
    return h


def _q_obs_indices_t(cfg, window):
    """Per-time upper-layer q flat-indices for q-obs."""
    ny, nx = cfg.ny, cfg.nx
    T = cfg.num_steps
    if "track_x_index" in window:
        per_time = [None] * T
        for t in window["obs_mask"].nonzero(as_tuple=False).flatten().tolist():
            x_col = int(window["track_x_index"][t])
            per_time[t] = [y * nx + x_col for y in range(ny)]
        return per_time
    elif "obs_columns" in window:
        cols_t = window["obs_columns"]
        C = cols_t.shape[1]
        per_time = [None] * T
        for t in window["obs_mask"].nonzero(as_tuple=False).flatten().tolist():
            idx = []
            for ci in range(C):
                x_col = int(cols_t[t, ci])
                if 0 <= x_col < nx:
                    idx.extend([y * nx + x_col for y in range(ny)])
            per_time[t] = idx if idx else None
        return per_time
    return [None] * T


def _make_obs_system(cfg, window, device, obs_var, loc_radius):
    """Build ObsOperator with index or H-mode based on obs_var."""
    if obs_var == "q":
        obs, r_var, _ = _q_alongtrack_obs(cfg, window, device)
        per_time = _q_obs_indices_t(cfg, window)
        obs_operator = ObsOperator(cfg.state_dim, obs_indices_t=per_time)
        return obs, r_var, obs_operator, _build_qg_loc_matrices
    else:
        obs_cols = _event_columns(cfg, window)
        h = _psi_h(_build_dyn(cfg, window, device), obs_cols, cfg.ny, device)
        obs, r_var, od = _obs_spec_rc(cfg, window, device)
        obs_op = ObsOperator(cfg.state_dim, h=h, h_index_at=None, n_obs=od)
        return obs, r_var, obs_op, _build_qg_col_loc_matrices


def _obs_spec_rc(cfg, window, device):
    obs = window["obs"].to(device)
    r_var = (cfg.obs_noise_std_frac * float(window["target_state_psi"].std())) ** 2
    od = cfg.cols_per_day * cfg.ny
    return obs, r_var, od


def _lagged_init_ensemble(cfg, window, N, init_lag_days, device):
    """Lagged-truth ensemble: members = x(t0 - dt_k), dt_k ~ U(0,DT].

    Args:
        cfg: QGConfig
        window: dict per-window
        N: ensemble size
        init_lag_days: float (mean lag)
        device: torch device

    Returns:
        init_ensemble: (N, state_dim) tensor
        mean_lag_days: float
    """
    truth = window["init_lead_truth"].float()
    dt_steps = int(init_lag_days / cfg.dt)
    dt_steps = max(dt_steps, 1)
    lead = dt_steps + 1
    if lead >= len(truth):
        lead = len(truth) - 1
    dt = float(init_lag_days / dt_steps)
    mean_lag_days = 0.0
    gens = [torch.Generator(device=device).manual_seed(cfg.seed + 7 + i) for i in range(N)]
    init_ensemble = torch.zeros(N, truth.shape[-1], device=device)
    r = torch.rand(N, generator=gens[0], device=device)
    for i in range(N):
        i_gen = gens[i]
        k_tplus1 = int(r[i] * lead) + 1
        x_tminus1 = truth[k_tplus1 - 1, :]
        x_t = truth[k_tplus1, :]
        alpha = torch.rand(1, generator=i_gen, device=device).item()
        init_ensemble[i] = (1 - alpha) * x_tminus1 + alpha * x_t
        mean_lag_days += float(k_tplus1 * dt)
    mean_lag_days /= N
    return init_ensemble, mean_lag_days


def _evaluate_window(cfg, window, method, device, obs=None, forcing=None,
                     init_ensemble=None, init_lag_days=None):
    if obs is None:
        obs, _ = _q_alongtrack_obs(cfg, window, device)
    mask = window["obs_mask"].to(device)
    truth = window["true_state"].to(device)
    if forcing is None:
        forcing = window["wind_state_corrupted"].to(device)
    if init_ensemble is not None:
        method.init_ensemble = init_ensemble
    result = method.assimilate(obs, mask, forcing, true_state=truth)
    return result


def _pooled_expvar(analyses, refs):
    sq = np.concatenate([(a - r) ** 2 for a, r in zip(analyses, refs)], axis=0)
    ref_all = np.concatenate(refs, axis=0)
    var = np.maximum(np.var(ref_all, axis=0), 1e-12)
    return 1.0 - np.mean(sq, axis=0) / var


def _free_forecast_rmse(cfg, dyn, window, device, forcing):
    """RMSE of the no-obs model forecast (roll from init_state)."""
    truth = window["true_state"].float()
    init_state = window["init_lead_truth"][0].to(device)
    roll = dyn.rollout_trajectory(init_state, cfg.num_steps - 1, wind_state=forcing)
    return float(np.sqrt(np.mean((roll.detach().cpu().numpy()
                                  - truth.numpy()) ** 2)))


def run(method_name, cfg, device=None, N_ensemble=60, inflation=1.05,
        loc_radius=None, scenarios=("test_s0", "test_s1a", "test_s1b"),
        out_path=None, init="lagged", geometry="random_columns",
        obs_var="q", init_lag_days=None, ds=None):
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    if ds is None:
        cfg = QGConfig(nx=cfg.nx, window_days=cfg.window_days,
                       spinup_years=cfg.spinup_years, num_windows=cfg.num_windows,
                       obs_geometry=geometry, cols_per_day=cfg.cols_per_day,
                       obs_field=cfg.obs_field, seed=7)
        ds = make_qg_s0_s1_datasets(cfg)
    else:
        cfg = QGConfig(nx=cfg.nx, window_days=cfg.window_days,
                       spinup_years=cfg.spinup_years, num_windows=cfg.num_windows,
                       obs_geometry=geometry, cols_per_day=cfg.cols_per_day,
                       obs_field=cfg.obs_field, seed=7)

    per_layer = cfg.ny * cfg.nx
    summary = {}
    for scen in scenarios:
        if scen not in ds:
            continue
        rmse_list = []
        fcast_rmse = []
        analyses = []
        refs = []
        d = ds[scen]
        spread_t0_list = []
        spread_final_list = []
        mean_init_lag_list = []
        method = None
        per_time = None
        for i in range(len(d)):
            w = d[i]
            dyn = _build_dyn(cfg, w, device)
            obs, r_var, _, obs_op = _make_obs_system(cfg, w,
                                                                      device,
                                                                      obs_var,
                                                                      loc_radius)
            forcing = w["wind_state_corrupted"].to(device)
            init_ensemble = None
            init_lag_val = 0.0
            if init == "lagged":
                init_ensemble, init_lag_val = _lagged_init_ensemble(
                    cfg, w, N_ensemble, init_lag_days, device)
                spread_t0_list.append(float(init_ensemble.std()))
            if obs_var == "q":
                if per_time is None:
                    per_time = _q_obs_indices_t(cfg, w)
                field_std = float(w["target_state_q"].std())
                Lx_t = Ly_t = None
                if loc_radius is not None:
                    Lx_t, Ly_t = _build_qg_loc_matrices(
                        dyn.state_dim, per_time, 2, cfg.ny, cfg.nx,
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
            else:  # obs_var == "psi"
                field_std = float(w["target_state_psi"].std())
                Lx_t = Ly_t = None
                if loc_radius is not None:
                    cols_t = _event_columns(cfg, w)
                    Lx_t, Ly_t = _build_qg_col_loc_matrices(
                        dyn.state_dim, cols_t, 2, cfg.ny, cfg.nx,
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
                                   forcing=forcing, init_ensemble=init_ensemble)
            spread_final_list.append(0.0)
            mean_init_lag_list.append(init_lag_val)
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
            "mean_init_lag_days": float(np.mean(mean_init_lag_list)) if mean_init_lag_list else None,
            "spread_t0_mean": float(np.mean(spread_t0_list)) if spread_t0_list else None,
            "spread_final_mean": 0.0,
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
    ap.add_argument("--method-list", default="etkf")
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--num-windows", type=int, default=3)
    ap.add_argument("--window-days", type=float, default=30.0)
    ap.add_argument("--spinup-years", type=float, default=0.3)
    ap.add_argument("--ensemble", type=int, default=60)
    ap.add_argument("--inflation", type=float, default=1.05)
    ap.add_argument("--loc-radius", type=float, default=None)
    ap.add_argument("--scenarios", default="test_s0,test_s1a,test_s1b")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--init", choices=["lagged", "white"], default="lagged")
    ap.add_argument("--geometry", choices=["alongtrack", "random_columns"], default="alongtrack")
    ap.add_argument("--obs-var", choices=["q", "psi"], default="q")
    ap.add_argument("--init-lag-days", type=float, default=2.0)
    ap.add_argument("--cols-per-day", type=int, default=3)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    cfg = QGConfig(nx=args.nx, window_days=args.window_days,
                   spinup_years=args.spinup_years, num_windows=args.num_windows,
                   obs_geometry=args.geometry, cols_per_day=args.cols_per_day,
                   seed=7)
    print(f"device={device}")
    for method in args.method_list.split(","):
        run(method, cfg, device=device, N_ensemble=args.ensemble,
            inflation=args.inflation, loc_radius=args.loc_radius,
            scenarios=tuple(args.scenarios.split(",")), out_path=args.out,
            init=args.init, geometry=args.geometry, obs_var=args.obs_var,
            init_lag_days=args.init_lag_days)


if __name__ == "__main__":
    main()
