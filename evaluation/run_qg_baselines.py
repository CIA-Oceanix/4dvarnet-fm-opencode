import argparse
import json
import math
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
from models.qg_interp import spectral_resize_2d

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


def _da_nx_for_window(cfg, window):
    """DA model grid size: window['da_nx'] (set by the scenario) else truth nx."""
    return int(window.get("da_nx") or cfg.nx)


def _resize_state_layers(traj, nlayers, src_n, dst_n, device):
    """Spectral down/upsample a layer-major flattened state to a new grid.

    `traj` is a (..., nlayers*src_n*src_n) real tensor or numpy array (single
    state, ensemble member, or full trajectory); each layer is reshaped to
    (src_n, src_n), spectrally resized to (dst_n, dst_n), and re-flattened
    layer-major. Both grids are square (the QG grid). Returns the same type
    (torch/numpy) as the input; torch output is on `device`, numpy output is
    moved back to CPU (numpy cannot hold a CUDA tensor).
    """
    is_np = isinstance(traj, np.ndarray)
    t = torch.from_numpy(traj).float() if is_np else traj
    lead = t.shape[:-1]
    x = t.reshape(*lead, nlayers, src_n, src_n)
    y = spectral_resize_2d(x, dst_n, dst_n, device)
    out = y.reshape(*lead, nlayers * dst_n * dst_n)
    return out.cpu().numpy() if is_np else out


def _downsample_to_da(state, da_nx, nlayers, truth_n, device):
    """Downsample a truth-resolution state to the DA-model grid."""
    return _resize_state_layers(state, nlayers, truth_n, da_nx, device)


def _upsample_to_truth(state, da_nx, nlayers, truth_n, device):
    """Upsample a DA-model-resolution state to the truth/obs grid (truth_n)."""
    return _resize_state_layers(state, nlayers, da_nx, truth_n, device)


def _build_dyn(cfg, window, device):
    da_params = window["da_params"]
    model = window["da_model"]
    nx_da = _da_nx_for_window(cfg, window)
    common = {
        "nx": nx_da, "L": cfg.L, "dt": cfg.dt, "beta": da_params["beta"],
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


def _psi_h(dyn, obs_cols, ny, nx, device):
    """H-function for obs of upper-layer streamfunction columns.

    Computes the DA model's streamfunction and (for a cross-resolution DA model,
    e.g. S1 with a lower-res grid) spectrally upsamples it to the obs grid
    (ny, nx) before selecting the observed meridional columns.

    Args:
        dyn: QG dynamics (access via dyn.inner.streamfunctions)
        obs_cols: list of column lists per time (e.g., [[0,3], None, [2]])
        ny, nx: obs-grid dimensions
        device: torch device

    Returns:
        Callable that takes (state (state_dim,), index=int) -> (C*ny,)
    """
    def h(state, index=None):
        batch = state.ndim > 1
        psi = dyn.inner.streamfunctions(state)
        if dyn.inner.ny != ny or dyn.inner.nx != nx:
            psi = spectral_resize_2d(psi, ny, nx)
        cols = obs_cols[index]
        if not batch:
            # 2-layer: psi is (2, ny, nx) -> upper layer; 1-layer: (ny, nx).
            psi1 = psi[0] if psi.ndim == 3 else psi
            return torch.cat([psi1[:, c] for c in cols])
        else:
            # 2-layer: psi is (B, 2, ny, nx) -> upper layer; 1-layer: (B, ny, nx).
            psi1 = psi[:, 0] if psi.ndim == 4 else psi
            C = len(cols)
            o = C * ny
            stacked = torch.stack([psi1[:, :, c] for c in cols], dim=1)
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
        dyn = _build_dyn(cfg, window, device)
        obs_cols = _event_columns(cfg, window)
        h = _psi_h(dyn, obs_cols, cfg.ny, cfg.nx, device)
        obs, r_var, od = _obs_spec_rc(cfg, window, device)
        obs_op = ObsOperator(dyn.state_dim, h=h, h_index_at=None, n_obs=od)
        return obs, r_var, obs_op, _build_qg_col_loc_matrices


def _obs_spec_rc(cfg, window, device):
    obs = window["obs"].to(device)
    r_var = (cfg.obs_noise_std_frac * float(window["target_state_psi"].std())) ** 2
    od = cfg.cols_per_day * cfg.ny
    return obs, r_var, od


def _sample_init_state(cfg, window, lag_param, band_half, device):
    """Sample ONE shared initial state at a band-centered lag.

    With the band-centered scheme, `lag_param` is the physical lag in days at
    the center of the sampling band; a single lag `dt* ~ U[lag-0.25, lag+0.25]`
    is drawn and the initial state `x(t0 - dt*)` is obtained by linear
    interpolation of the `init_lead_truth` buffer (which spans
    [t0 - init_lead_days, t0]). The SAME state is used both as the free-
    forecast first guess and as the anchor of the DA ensemble, so the DA-vs-
    free-forecast comparison is apples-to-apples (identical initial condition;
    only assimilated observations differ).

    Args:
        cfg: QGConfig
        window: per-window dict
        lag_param: nominal (center) lag in days
        band_half: half-width of the lag sampling band in days
        device: torch device

    Returns:
        init_state: (state_dim,) tensor on `device`
        init_lag_days: the sampled lag in days (for reporting)
    """
    truth = window["init_lead_truth"].float()
    steps_per_day = round(86400.0 / cfg.dt)
    max_lag_days = max(0.0, (len(truth) - 2) / steps_per_day)
    lo = max(0.0, lag_param - band_half)
    hi = min(max_lag_days, lag_param + band_half)
    gen = torch.Generator(device=device).manual_seed(
        cfg.seed + 7000 + int(lag_param * 10))
    lag_days = float(lo + (hi - lo) * torch.rand(1, generator=gen, device=device).item())
    lag_steps = lag_days * steps_per_day
    kk = math.floor(lag_steps)
    kk = min(kk, len(truth) - 2)
    alpha = lag_steps - kk
    a = truth[len(truth) - kk - 1, :]
    b = truth[len(truth) - kk, :]
    return ((1.0 - alpha) * a + alpha * b).to(device), lag_days


def _ensemble_from_init(init_state, sigma_raw, N, disp_frac, device, cfg):
    """DA ensemble anchored at the shared init state (all members + dispersion).

    All ensemble members start from the SAME init_state (the one shared with
    the free-forecast reference); spread comes from independent Gaussian
    dispersion scaled to the raw per-point spread, consistent with the
    PR #105/#110 background-error re-proportioning.

    Returns:
        init_ensemble: (N, state_dim) tensor
    """
    init_ensemble = init_state.unsqueeze(0).expand(N, -1).clone()
    if disp_frac > 0.0:
        disp_std = disp_frac * sigma_raw
        disp = torch.Generator(device=device).manual_seed(cfg.seed + 9000 + N)
        init_ensemble = init_ensemble + disp_std * torch.randn(
            init_ensemble.shape, generator=disp, device=device)
    return init_ensemble


def _lagged_init_ensemble(cfg, window, N, init_lag_days, device,
                          disp_frac: float = 1.0):
    """Single-init ensemble helper (kept for tests/back-compat).

    Samples a single shared init state at a band-centered lag around
    `init_lag_days` (band half-width 0.25 d) and builds the ensemble by adding
    dispersion to that shared state. Returns (ensemble, sampled_lag).
    """
    init_state, lag_days = _sample_init_state(
        cfg, window, init_lag_days, 0.25, device)
    truth = window["init_lead_truth"].float()
    sigma_raw = float(truth.std(0).mean())
    return _ensemble_from_init(init_state, sigma_raw, N, disp_frac, device, cfg), lag_days



def _evaluate_window(cfg, window, method, device, obs=None, forcing=None,
                     init_ensemble=None, init_lag_days=None):
    if obs is None:
        obs, _ = _q_alongtrack_obs(cfg, window, device)
    mask = window["obs_mask"].to(device)
    truth = window["true_state"].to(device)
    if window["da_model"] == "qg1l":
        # 1-layer DA compares against the truth's upper layer only.
        truth = truth[:, :cfg.ny * cfg.nx]
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


def _field_layer_metrics(analyses, refs, free_ra, inner, per_layer, device):
    """Per-field (q/psi) per-layer (upper/lower) RMSE + pooled EV for DA and free forecast.

    `analyses`/`free_ra` are lists of DA/free-forecast trajectories (each (T, 2*per_layer),
    layer-major q-state), `refs` the matching truth. Computes:
      - q: per-layer via state slicing
      - psi: upper/lower/full via the model spectral inversion `inner.streamfunctions`.
    Returns a nested dict of scalars: rmse / rmse_free / improv / ev / ev_free per (field, layer).
    """
    def q_traj(traj):
        nlayer = inner.state_dim // per_layer
        slices = {"full": traj}
        for li in range(nlayer):
            slices[f"layer{li + 1}"] = traj[:, li * per_layer:(li + 1) * per_layer]
        return slices

    def psi_traj(traj):
        psi = inner.streamfunctions(
            torch.from_numpy(traj).float().to(device)).detach().cpu().numpy()
        if psi.ndim == 3:
            # 1-layer model: streamfunctions returns (T, ny, nx) without a
            # leading layer dim; normalize to (T, 1, ny, nx).
            psi = psi[:, None]
        nlayer = psi.shape[-3]
        slices = {"full": psi.reshape(psi.shape[0], -1)}
        for li in range(nlayer):
            pflat = psi[..., li, :, :].reshape(psi.shape[0], per_layer)
            slices[f"layer{li + 1}"] = pflat
        return slices

    def layer_sets(fields_anal, fields_ref, fields_free):
        out = {}
        for name, getter in (("q", q_traj), ("psi", psi_traj)):
            sample = getter(fields_ref[0])
            layers = [k for k in sample if k != "full"] + ["full"]
            out[name] = {}
            for k in layers:
                a = [getter(x)[k] for x in fields_anal]
                ref = [getter(x)[k] for x in fields_ref]
                fr = [getter(x)[k] for x in fields_free]
                rmse = float(np.sqrt(np.mean(
                    np.concatenate([(x - r) ** 2 for x, r in zip(a, ref)], axis=0))))
                rmse_free = float(np.sqrt(np.mean(
                    np.concatenate([(x - r) ** 2 for x, r in zip(fr, ref)], axis=0))))
                out[name][k] = {
                    "rmse": rmse,
                    "rmse_free": rmse_free,
                    "improv": rmse_free / max(rmse, 1e-30),
                    "ev": float(np.mean(_pooled_expvar(a, ref))),
                    "ev_free": float(np.mean(_pooled_expvar(fr, ref))),
                }
        return out

    return layer_sets(analyses, refs, free_ra)


def _free_forecast_init(cfg, window, init_lag_days, device):
    """Sample a single band-centered init state (test/back-compat helper).

    Delegates to `_sample_init_state` with a 0.25-day band half-width so the
    reference sits at the same physical lag as the ensemble being DA'd. In
    `run()` the free forecast instead reuses the exact shared `init_state`; the
    helper is kept for the standalone test path.
    """
    return _sample_init_state(cfg, window, init_lag_days, 0.25, device)


def _free_forecast_rmse(cfg, dyn, window, device, forcing, init_state,
                        upper_only=False):
    """RMSE of the no-obs model forecast rolled from a shared init state.

    `init_state` is the SAME initial condition used to seed the DA ensemble, so
    the free-forecast and DA results are compared apples-to-apples (identical
    initial state; only assimilated observations differ). For a cross-resolution
    DA model (S1), `init_state` is in DA-model space and the rolled forecast is
    spectrally upsampled to the truth grid before the RMSE is taken. For a
    1-layer DA model (qg1l), `upper_only=True` compares the roll against the
    truth's upper layer only.
    """
    truth = window["true_state"].float()
    roll = dyn.rollout_trajectory(init_state, cfg.num_steps - 1, wind_state=forcing)
    roll = roll.detach().cpu().numpy()
    nlayers = dyn.state_dim // (dyn.inner.ny * dyn.inner.nx)
    if upper_only:
        truth = truth[:, : per_layer_for(cfg)]
    if dyn.inner.ny != cfg.ny:
        roll = _upsample_to_truth(
            roll, dyn.inner.ny, nlayers, cfg.ny, torch.device("cpu"))
    return float(np.sqrt(np.mean((roll - truth.numpy()) ** 2)))


def per_layer_for(cfg):
    return cfg.ny * cfg.nx


def run(method_name, cfg, device=None, N_ensemble=60, inflation=1.05,
        loc_radius=None, scenarios=("test_s0", "test_s1"),
        out_path=None, init="lagged", geometry="random_columns",
        obs_var="q", init_lag_days=None, ds=None, disp_frac=1.0,
        etkf_ridge=0.0, etkf_additive=0.0, band_half=0.25,
        save_traj=None):
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    if ds is None:
        ds = make_qg_s0_s1_datasets(cfg)

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
        mean_init_lag_list = []
        free_ra = []
        method = None
        truth_inner = None
        for i in range(len(d)):
            w = d[i]
            dyn = _build_dyn(cfg, w, device)
            da_nx = _da_nx_for_window(cfg, w)
            nlayers = dyn.state_dim // (dyn.inner.ny * dyn.inner.nx)
            is_qg1l = w["da_model"] == "qg1l"
            cross_res = da_nx != cfg.nx
            if obs_var == "q" and cross_res:
                raise ValueError(
                    "obs_var='q' is not supported for a cross-resolution DA model "
                    "(S1): the PV-obs indices select truth-grid points that have no "
                    "one-to-one mapping in the lower-res state. Use obs_var='psi'.")
            obs, r_var, obs_op, _ = _make_obs_system(cfg, w,
                                                               device,
                                                               obs_var,
                                                               loc_radius)
            forcing = w["wind_state_corrupted"].to(device)
            init_ensemble = None
            init_lag_val = 0.0
            shared_init = None
            if init == "lagged":
                # ONE shared initial state per window, reused by BOTH the
                # DA ensemble and the free-forecast reference so the comparison
                # is apples-to-apples (identical initial condition).
                shared_init, init_lag_val = _sample_init_state(
                    cfg, w, init_lag_days, band_half, device)
                if is_qg1l:
                    # 1-layer DA state = truth's upper-layer PV q1 (the 1-layer
                    # model represents the upper layer). All DA members and the
                    # free forecast start from this same projective init.
                    shared_init = shared_init[:per_layer]
                    lead = w["init_lead_truth"].float()[:, :per_layer]
                    sigma_raw = float(lead.std(0).mean())
                elif cross_res:
                    shared_init = _downsample_to_da(
                        shared_init, da_nx, nlayers, cfg.nx, device)
                    lead = _downsample_to_da(
                        w["init_lead_truth"].float(), da_nx, nlayers, cfg.nx, device)
                    sigma_raw = float(lead.std(0).mean())
                else:
                    sigma_raw = float(w["init_lead_truth"].std(0).mean())
                init_ensemble = _ensemble_from_init(
                    shared_init, sigma_raw, N_ensemble, disp_frac, device, cfg)
                spread_t0_list.append(float(init_ensemble.std(0).mean()))
            if obs_var == "q":
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
                                  loc_Lx_t=Lx_t, loc_Ly_t=Ly_t,
                                  etkf_ridge=etkf_ridge, etkf_additive=etkf_additive)
            else:  # obs_var == "psi"
                field_std = float(w["target_state_psi"].std())
                Lx_t = Ly_t = None
                if loc_radius is not None:
                    cols_t = _event_columns(cfg, w)
                    Lx_t, Ly_t = _build_qg_col_loc_matrices(
                        dyn.state_dim, cols_t, 2, cfg.ny, cfg.nx,
                        loc_radius, device, state_ny=da_nx, state_nx=da_nx)
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
                                  loc_Lx_t=Lx_t, loc_Ly_t=Ly_t,
                                  etkf_ridge=etkf_ridge, etkf_additive=etkf_additive)
            res = _evaluate_window(cfg, w, method, device, obs=obs,
                                   forcing=forcing, init_ensemble=init_ensemble)
            mean_init_lag_list.append(init_lag_val)
            ref = w["true_state"].numpy()
            traj_da = res.trajectory
            if cross_res:
                traj_da = _upsample_to_truth(
                    traj_da, da_nx, nlayers, cfg.nx, device)
            if is_qg1l:
                # 1-layer DA output vs truth's upper layer only.
                ref = ref[:, :per_layer]
            analyses.append(traj_da)
            refs.append(ref)
            rmse_list.append(float(np.sqrt(np.mean(
                (traj_da - ref) ** 2))))
            fcast_rmse.append(_free_forecast_rmse(
                cfg, dyn, w, device, forcing, shared_init, upper_only=is_qg1l))
            if shared_init is not None:
                free_roll = dyn.rollout_trajectory(
                    shared_init, cfg.num_steps - 1, wind_state=forcing)
                free_roll = free_roll.detach().cpu().numpy()
                if is_qg1l:
                    free_roll = free_roll[:, :per_layer]
                if cross_res:
                    free_roll = _upsample_to_truth(
                        free_roll, da_nx, nlayers, cfg.nx,
                        torch.device("cpu"))
                free_ra.append(free_roll)
        ev = _pooled_expvar(analyses, refs)
        ev_upper = _pooled_expvar(
            [a[:, :per_layer] for a in analyses],
            [r[:, :per_layer] for r in refs])
        da_r = float(np.mean(rmse_list))
        fc_r = float(np.mean(fcast_rmse))
        ev_free = None
        if free_ra:
            ev_free = float(np.mean(_pooled_expvar(free_ra, refs)))
        metrics_per_field = None
        if free_ra:
            if truth_inner is None:
                # psi metric inversion uses a truth-res model. Its params come
                # from window 0's true_params (per-window draws vary, so the
                # per-field PSI diagnostic for later windows uses window-0's
                # params; the DA RMSE/EV themselves are computed directly from
                # the q-state and are unaffected).
                tp = d[0]["true_params"]
                if d[0]["da_model"] == "qg1l":
                    # 1-layer structural-error scenario: invert the 1-layer DA
                    # state to a 1-layer streamfunction; metrics are upper-layer
                    # only (the refs/analyses are already 1-layer here).
                    from models.qg1l_dynamics import QG1LDynamics
                    truth_inner = QG1LDynamics(
                        nx=cfg.nx, L=cfg.L, dt=cfg.dt, beta=tp["beta"], rd=tp["rd"],
                        U1=tp["U1"], rek=tp["rek"], filterfac=cfg.filterfac,
                        wind_amp=d[0]["wind_amp"], wind_sigma=cfg.wind_sigma,
                        clip_range=1e-3).to(device)
                else:
                    truth_inner = QGDynamics(
                        nx=cfg.nx, L=cfg.L, dt=cfg.dt, beta=tp["beta"], rd=tp["rd"],
                        delta=cfg.delta, U1=tp["U1"], U2=tp["U2"], rek=tp["rek"],
                        filterfac=cfg.filterfac,
                        wind_amp=d[0]["wind_amp"], wind_sigma=cfg.wind_sigma,
                        clip_range=1e-3).to(device)
            metrics_per_field = _field_layer_metrics(
                analyses, refs, free_ra, truth_inner, per_layer, device)
        traj_path = None
        if save_traj and free_ra:
            os.makedirs(save_traj, exist_ok=True)
            name = f"{scen}_m{method_name}_ov{obs_var}_lag{init_lag_days}"
            traj_path = os.path.join(save_traj, f"traj_{name}.npz")
            np.savez_compressed(
                traj_path,
                analyses=np.stack(analyses).astype(np.float32),
                free_forecast=np.stack(free_ra).astype(np.float32),
                refs=np.stack(refs).astype(np.float32),
                per_layer=per_layer,
            )
        summary[scen] = {
            "rmse_mean": da_r,
            "rmse_list": rmse_list,
            "forecast_rmse_mean": fc_r,
            "forecast_improvement": fc_r / max(da_r, 1e-30),
            "expvar_full": float(np.mean(ev)),
            "expvar_upper_q": float(np.mean(ev_upper)),
            "expvar_free": ev_free,
            "metrics_per_field": metrics_per_field,
            "traj_path": traj_path,
            "mean_init_lag_days": float(np.mean(mean_init_lag_list)) if mean_init_lag_list else None,
            "spread_t0_mean": float(np.mean(spread_t0_list)) if spread_t0_list else None,
        }
        print(f"{scen}: rmse={da_r:.3e} forecast_rmse={fc_r:.3e} "
              f"improv={summary[scen]['forecast_improvement']:.2f}x "
              f"ev_full={summary[scen]['expvar_full']:.3f} "
              f"ev_free={summary[scen]['expvar_free']:.3f}")

    payload = {"method": method_name, "nx": cfg.nx,
               "N_ensemble": N_ensemble, "inflation": inflation,
               "loc_radius": loc_radius, "scenarios": summary}

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
    ap.add_argument("--scenarios", default="test_s0,test_s1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--init", choices=["lagged", "white"], default="lagged")
    ap.add_argument("--geometry", choices=["alongtrack", "random_columns"], default="alongtrack")
    ap.add_argument("--obs-var", choices=["q", "psi"], default="q")
    ap.add_argument("--init-lag-days", type=float, default=2.0)
    ap.add_argument("--band", dest="band_half", type=float, default=0.25)
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
            init_lag_days=args.init_lag_days, band_half=args.band_half)


if __name__ == "__main__":
    main()
