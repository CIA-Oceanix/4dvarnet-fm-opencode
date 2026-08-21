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
        filterfac=cfg.filterfac,
    )


class QGDataset:
    def __init__(self, cfg: QGConfig):
        self.cfg = cfg
        self.device = torch.device("cpu")
        dynamics = _make_qg_dynamics(cfg)

        full_len = (cfg.num_windows - 1) * cfg.window_spacing + cfg.num_steps
        traj, _ = dynamics.generate_full_trajectory(
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
            forcing_true = torch.full((cfg.num_steps,), cfg.U1)
            self.windows.append({
                "true_state": true_state,
                "obs": noisy_obs,
                "obs_mask": obs_mask,
                "forcing_true": forcing_true,
                "forcing_corrupted": forcing_true.clone(),
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
