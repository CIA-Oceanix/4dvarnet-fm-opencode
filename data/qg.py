import math
from dataclasses import dataclass

import numpy as np
import torch

from data.lorenz96 import _generate_observations


@dataclass
class QGConfig:
    nx: int = 64
    L: float = 1e6
    dt: float = 7200.0
    beta: float = 1.5e-11
    rd: float = 15000.0
    delta: float = 0.25
    U1: float = 0.05
    U2: float = 0.0
    rek: float = 5.787e-7
    filterfac: float = 23.6

    window_days: float = 60.0
    obs_interval: int = 6
    R_var: float = 1e-12
    num_windows: int = 200
    window_spacing_days: float = 90.0
    spinup_years: float = 2.0
    seed: int = 42
    obs_var_indices: tuple[int, ...] | None = None

    wind_amp: float = 1e-11
    wind_tau_days: float = 15.0
    wind_sigma: float = 250000.0
    wind_cx: float = 0.5
    wind_cy: float = 0.03
    wind_drift_tau_days: float = 10.0
    wind_drift_sigma: float = 50000.0
    wind_seed: int = 7

    obs_geometry: str = "grid"
    obs_field: str = "psi"
    track_repeat_days: float = 5.0
    track_advance_pts: int = 4
    track_phase_seed: int = 0
    obs_noise_std_frac: float = 0.05
    store_targets: bool = True
    param_range: float = 0.15
    s1_param_bias: float = 0.15
    s1_amp_bias: float = 0.15
    s1_loc_sigma_frac: float = 0.25
    s1_tau_days: float = 10.0
    s1_sigma_eta_frac: float = 0.3

    @property
    def ny(self) -> int:
        return self.nx

    @property
    def state_dim(self) -> int:
        return 2 * self.ny * self.nx

    @property
    def num_steps(self) -> int:
        steps_per_day = round(86400.0 / self.dt)
        return int(self.window_days * steps_per_day)

    @property
    def window_spacing(self) -> int:
        steps_per_day = round(86400.0 / self.dt)
        return int(self.window_spacing_days * steps_per_day)

    @property
    def spinup_steps(self) -> int:
        steps_per_day = round(86400.0 / self.dt)
        return int(self.spinup_years * 365.0 * steps_per_day)


def _make_qg_dynamics(cfg: QGConfig):
    from models.qg_dynamics import QGDynamics
    return QGDynamics(
        nx=cfg.nx, L=cfg.L, dt=cfg.dt, beta=cfg.beta, rd=cfg.rd,
        delta=cfg.delta, U1=cfg.U1, U2=cfg.U2, rek=cfg.rek,
        filterfac=cfg.filterfac, wind_amp=cfg.wind_amp,
        wind_tau_days=cfg.wind_tau_days, wind_sigma=cfg.wind_sigma,
        wind_cx=cfg.wind_cx, wind_cy=cfg.wind_cy,
        wind_drift_tau_days=cfg.wind_drift_tau_days,
        wind_drift_sigma=cfg.wind_drift_sigma, wind_seed=cfg.wind_seed,
    )


class QGDataset:
    def __init__(self, cfg: QGConfig):
        self.cfg = cfg
        self.device = torch.device("cpu")
        dynamics = _make_qg_dynamics(cfg)

        full_len = (cfg.num_windows - 1) * cfg.window_spacing + cfg.num_steps
        traj, wind_state = dynamics.generate_full_trajectory(
            num_steps=full_len, seed=cfg.seed, spinup_steps=cfg.spinup_steps,
        )

        self.windows = []
        start_indices = (
            np.arange(cfg.num_windows) * cfg.window_spacing
        ).astype(int)

        for idx in start_indices:
            true_state = traj[idx: idx + cfg.num_steps].clone()
            noisy_obs, obs_mask = _generate_observations(
                true_state, cfg.obs_interval, cfg.R_var, cfg.seed + 1,
                self.device,
                obs_var_indices=(np.asarray(cfg.obs_var_indices, dtype=np.int64)
                                 if cfg.obs_var_indices is not None else None),
            )
            ws_slice = wind_state[idx: idx + cfg.num_steps]
            forcing_true = ws_slice[:, 0].clone()
            wind_curl = dynamics.wind_curl_field(ws_slice)
            self.windows.append({
                "true_state": true_state,
                "obs": noisy_obs,
                "obs_mask": obs_mask,
                "forcing_true": forcing_true,
                "forcing_corrupted": forcing_true.clone(),
                "wind_curl": wind_curl,
            })

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.windows[idx]


def make_qg_datasets(cfg: QGConfig) -> dict[str, QGDataset]:
    train_cfg = QGConfig(**{**cfg.__dict__, "seed": 42})
    val_cfg = QGConfig(**{**cfg.__dict__, "seed": 99})
    test_cfg_cs1 = QGConfig(**{**cfg.__dict__, "seed": 123})
    test_cfg_cs2 = QGConfig(**{**cfg.__dict__, "seed": 131})
    return {
        "train": QGDataset(train_cfg),
        "val": QGDataset(val_cfg),
        "test_cs1": QGDataset(test_cfg_cs1),
        "test_cs2": QGDataset(test_cfg_cs2),
    }


_S1_WIND_LEVELS = (0.0, 3e-12, 1e-11, 2e-11, 3e-11)


def _upper_field(dynamics, state: torch.Tensor, field: str) -> torch.Tensor:
    """Return the upper-layer field (T, ny, nx) of `state` (T, 2*ny*nx)."""
    if field == "q":
        grid = dynamics._grid(state)
        return grid[..., 0, :, :]
    psi = dynamics.streamfunctions(state)
    return psi[..., 0, :, :]


def _generate_alongtrack_observations(
    dynamics, state: torch.Tensor, field: str, cfg: QGConfig, seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nadir-along-track obs: one meridional column per repeat cycle.

    Returns (obs (T, ny), obs_mask (T,), track_x_index (T,)) with NaN-padded
    values off the pass times.
    """
    T, ny, nx = cfg.num_steps, cfg.ny, cfg.nx
    f = _upper_field(dynamics, state, field)
    sigma = cfg.obs_noise_std_frac * float(f.std())
    repeat = max(1, round((cfg.track_repeat_days * 86400.0) / cfg.dt))
    rng = torch.Generator().manual_seed(seed)
    obs = torch.full((T, ny), float("nan"))
    obs_mask = torch.zeros(T, dtype=torch.bool)
    track_idx = torch.full((T,), -1, dtype=torch.long)
    col = cfg.track_phase_seed
    for t in range(0, T, repeat):
        x_col = col % nx
        noise = torch.randn(ny, generator=rng) * sigma
        obs[t] = f[t, :, x_col] + noise
        obs_mask[t] = True
        track_idx[t] = x_col
        col += cfg.track_advance_pts
    return obs, obs_mask, track_idx


def expand_obs_to_grid(window: dict, cfg: QGConfig) -> torch.Tensor:
    """Expand compact (T, ny) track obs to a (T, ny*nx) NaN grid."""
    T, ny, nx = cfg.num_steps, cfg.ny, cfg.nx
    grid = torch.full((T, ny * nx), float("nan"))
    idx_t = window["track_x_index"]
    obs = window["obs"]
    for t in window["obs_mask"].nonzero(as_tuple=False).flatten().tolist():
        x_col = int(idx_t[t])
        if x_col >= 0:
            grid[t, torch.arange(ny, dtype=torch.long) * nx + x_col] = obs[t]
    return grid


def _ou_series(T: int, rng: np.random.RandomState, tau_days: float,
               dt: float) -> np.ndarray:
    """Zero-mean OU path normalized to unit std (N(0,1) increments)."""
    tau = tau_days * 86400.0
    coeff = math.sqrt(2.0 / tau * dt)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = y[t - 1] - (1.0 / tau) * y[t - 1] * dt + coeff * rng.normal(0.0, 1.0)
    s = y.std()
    if s > 0:
        y = y / s
    return y


def _make_corrupted_wind_state(cfg: QGConfig, wind_true: torch.Tensor,
                               idx: int) -> torch.Tensor:
    """Corrupt storm location (OU jitter) + amplitude (bias + OU eta)."""
    A = wind_true[:, 0].double()
    xc = wind_true[:, 1].double()
    yc = wind_true[:, 2].double()
    T = A.shape[0]
    rng = np.random.RandomState(cfg.seed + 5000 + idx * 17)
    sig_loc = cfg.s1_loc_sigma_frac * cfg.wind_sigma
    ex = sig_loc * torch.tensor(_ou_series(T, rng, cfg.s1_tau_days, cfg.dt),
                                dtype=xc.dtype)
    ey = sig_loc * torch.tensor(_ou_series(T, rng, cfg.s1_tau_days, cfg.dt),
                                dtype=yc.dtype)
    a_std = float(A.std())
    eta = torch.zeros_like(A)
    if a_std > 0:
        eta = (cfg.s1_sigma_eta_frac * a_std
               * torch.tensor(_ou_series(T, rng, cfg.s1_tau_days, cfg.dt),
                              dtype=A.dtype))
    xc_c = (xc + ex) % cfg.L
    yc_c = (yc + ey) % cfg.L
    A_c = A * (1.0 + cfg.s1_amp_bias) + eta
    return torch.stack([A_c, xc_c, yc_c], dim=-1).float()


class QGS01Dataset:
    """S0/S1 evaluation dataset with along-track altimetry obs of the upper layer.

    Windows store truth (full 2-layer PV), upper-layer targets (psi/q), compact
    along-track obs, and scenario metadata (da_model, da_params, corrupted wind).
    """

    def __init__(self, cfg: QGConfig, scenario: str,
                 base_windows: list[dict] | None = None,
                 num_windows: int | None = None):
        self.cfg = cfg
        self.scenario = scenario
        self.device = torch.device("cpu")
        n = num_windows or cfg.num_windows
        if base_windows is None:
            base_windows = self._generate_truth(cfg, n)
        self.windows = [
            self._scenario_window(cfg, scenario, w, i)
            for i, w in enumerate(base_windows)
        ]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        return self.windows[idx]

    @staticmethod
    def _generate_truth(cfg: QGConfig, n: int) -> list[dict]:
        from models.qg_dynamics import QGDynamics
        rng = np.random.RandomState(cfg.seed)
        levels = list(_S1_WIND_LEVELS)
        out = []
        for i in range(n):
            u = rng.uniform(1 - cfg.param_range, 1 + cfg.param_range)
            r = rng.uniform(1 - cfg.param_range, 1 + cfg.param_range)
            k = rng.uniform(1 - cfg.param_range, 1 + cfg.param_range)
            x0 = rng.uniform(0.0, cfg.L)
            y0 = rng.uniform(0.0, cfg.L)
            cx = rng.uniform(0.25, 0.75)
            cy = rng.uniform(-0.06, 0.06)
            amp = levels[i % len(levels)]
            win_seed = cfg.seed + 2000 + i * 101
            dyn = QGDynamics(
                nx=cfg.nx, L=cfg.L, dt=cfg.dt, beta=cfg.beta, rd=cfg.rd * r,
                delta=cfg.delta, U1=cfg.U1 * u, U2=cfg.U2, rek=cfg.rek * k,
                filterfac=cfg.filterfac, wind_amp=amp,
                wind_tau_days=cfg.wind_tau_days, wind_sigma=cfg.wind_sigma,
                wind_cx=cx, wind_cy=cy,
                wind_drift_tau_days=cfg.wind_drift_tau_days,
                wind_drift_sigma=cfg.wind_drift_sigma, wind_seed=win_seed,
            )
            wind_true = dyn.generate_wind_state(cfg.num_steps, seed=win_seed,
                                                x0=x0, y0=y0)
            traj, _ = dyn.generate_full_trajectory(
                num_steps=cfg.num_steps, seed=cfg.seed + 3000 + i * 101,
                spinup_steps=cfg.spinup_steps, wind_state=wind_true,
            )
            obs, obs_mask, track_idx = _generate_alongtrack_observations(
                dyn, traj, cfg.obs_field, cfg, cfg.seed + 4000 + i * 101,
            )
            psi1 = _upper_field(dyn, traj, "psi").reshape(cfg.num_steps, cfg.ny * cfg.nx)
            q1 = _upper_field(dyn, traj, "q").reshape(cfg.num_steps, cfg.ny * cfg.nx)
            wind_curl = dyn.wind_curl_field(wind_true)
            true_params = {"U1": dyn.U1, "rd": dyn.rd, "rek": dyn.rek,
                           "beta": dyn.beta, "U2": dyn.U2}
            out.append({
                "true_state": traj,
                "obs": obs,
                "obs_mask": obs_mask,
                "track_x_index": track_idx,
                "obs_field": cfg.obs_field,
                "target_state_psi": psi1,
                "target_state_q": q1,
                "wind_curl": wind_curl,
                "wind_state_true": wind_true,
                "true_params": true_params,
                "wind_seed": win_seed,
                "wind_amp": amp,
            })
        return out

    @staticmethod
    def _scenario_window(cfg: QGConfig, scenario: str, w: dict, i: int) -> dict:
        w = dict(w)
        ws_true = w["wind_state_true"]
        if scenario == "test_s0":
            ws_corrupt = ws_true
            da_params = dict(w["true_params"])
            da_model = "qg2l"
        else:
            ws_corrupt = _make_corrupted_wind_state(cfg, ws_true, i)
            da_params = dict(w["true_params"])
            if scenario == "test_s1a":
                b = cfg.s1_param_bias
                da_params["rd"] = da_params["rd"] * (1 - b)
                da_params["rek"] = da_params["rek"] * (1 - b)
                da_model = "qg2l"
            else:
                da_model = "qg1l"
        w["da_model"] = da_model
        w["da_params"] = da_params
        w["wind_state_corrupted"] = ws_corrupt
        w["forcing_true"] = ws_true[:, 0].clone()
        w["forcing_corrupted"] = ws_corrupt[:, 0].clone()
        return w


def make_qg_s0_s1_datasets(cfg: QGConfig, num_test_windows: int | None = None) -> dict:
    n = num_test_windows or cfg.num_windows
    base = QGS01Dataset._generate_truth(cfg, n)
    return {
        "test_s0": QGS01Dataset(cfg, "test_s0", base_windows=base),
        "test_s1a": QGS01Dataset(cfg, "test_s1a", base_windows=base),
        "test_s1b": QGS01Dataset(cfg, "test_s1b", base_windows=base),
    }
