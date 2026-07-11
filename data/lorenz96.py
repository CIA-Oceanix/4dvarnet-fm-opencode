import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class Lorenz96Config:
    case: int = 1
    dt: float = 0.001
    T_max: float = 3.0
    obs_interval: int = 200
    R_var: float = 0.5
    B_var: float = 2.0
    param_bias: float = 0.0
    num_windows: int = 2000
    window_spacing: int = 2000
    spinup_steps: int = 10000
    seed: int = 42

    NO: int = 8
    J: int = 4
    h: float = 1.0
    hx: float = 1.0
    eps: float = 0.1
    F_true: float = 8.0
    F_da: float = 8.0

    gamma: float = 0.05
    W_L_bar: float = 0.0
    c1: float = 1.0
    c2: float = 0.1
    sigma_0: float = 0.08
    sigma_L: float = 0.20

    tau_eta: float = 5.0
    sigma_eta: float = np.sqrt(0.5)
    forcing_state_bias: float = 0.0
    forcing_coupling: str = "linear"
    coupling_exponent_truth: float = 1.6
    coupling_exponent_da: float = 1.0

    @property
    def num_steps(self) -> int:
        return int(self.T_max / self.dt)

    @property
    def time_grid(self) -> np.ndarray:
        return np.linspace(0, self.T_max, self.num_steps)

    @property
    def state_dim(self) -> int:
        return self.NO + self.NO * self.J

    @property
    def biased_params(self) -> Tuple[float]:
        b = self.param_bias
        return (self.F_true * (1 - b),)

    @property
    def da_params(self) -> Tuple[float]:
        if self.case == 1:
            return (self.F_true,)
        return self.biased_params

    @property
    def use_corrupted_forcing(self) -> bool:
        return self.case == 2


def _generate_observations(
    true_fluid: torch.Tensor, obs_interval: int, R_var: float, seed: int,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_steps = true_fluid.shape[0]
    rng = torch.Generator(device=device).manual_seed(seed)
    obs_indices = np.arange(0, num_steps, obs_interval)
    obs_mask = torch.zeros(num_steps, dtype=torch.bool, device=device)
    obs_mask[obs_indices] = True
    noisy_obs = torch.full_like(true_fluid, float('nan'))
    noisy_obs[obs_indices] = true_fluid[obs_indices] + (
        torch.randn((len(obs_indices), true_fluid.shape[-1]), device=device, generator=rng) * np.sqrt(R_var)
    )
    return noisy_obs, obs_mask


def _make_lorenz96_dynamics(cfg: Lorenz96Config):
    from models.lorenz96_dynamics import Lorenz96Dynamics
    return Lorenz96Dynamics(
        dt=cfg.dt, coupling_exponent=cfg.coupling_exponent_truth,
        c1=cfg.c1, NO=cfg.NO, J=cfg.J, h=cfg.h, hx=cfg.hx, eps=cfg.eps,
        sigma_0=cfg.sigma_0, gamma=cfg.gamma,
        W_L_bar=cfg.W_L_bar, c2=cfg.c2, sigma_L=cfg.sigma_L,
    )


class Lorenz96Dataset:
    def __init__(self, cfg: Lorenz96Config):
        self.cfg = cfg
        self.device = torch.device("cpu")
        dynamics = _make_lorenz96_dynamics(cfg)

        full_traj_len = cfg.spinup_steps + (cfg.num_windows + 2) * cfg.window_spacing
        traj, forcing = dynamics.generate_full_trajectory(
            num_steps=full_traj_len, seed=cfg.seed, F=cfg.F_true,
            coupling_exponent=cfg.coupling_exponent_truth,
        )

        self.windows = []
        start_indices = (
            np.arange(cfg.num_windows) * cfg.window_spacing + cfg.spinup_steps
        ).astype(int)

        for idx in start_indices:
            seg = traj[idx: idx + cfg.num_steps].clone()
            true_fluid = seg
            W_L_true = forcing[idx: idx + cfg.num_steps].clone()

            if cfg.use_corrupted_forcing:
                force_seed = cfg.seed + 2 + idx // (cfg.num_steps + 1)
                corrupted = W_L_true.clone() + cfg.forcing_state_bias * true_fluid[:, 0]
                rng = np.random.RandomState(force_seed)
                eta = np.zeros(cfg.num_steps)
                sqrt_dt = np.sqrt(cfg.dt)
                for et in range(1, cfg.num_steps):
                    d_eta = -(1.0 / cfg.tau_eta) * eta[et - 1] * cfg.dt + cfg.sigma_eta * np.sqrt(2.0 / cfg.tau_eta) * rng.normal(0, sqrt_dt)
                    eta[et] = eta[et - 1] + d_eta
                W_L_star = corrupted + torch.tensor(eta, dtype=true_fluid.dtype, device=self.device)
            else:
                W_L_star = W_L_true.clone()

            noisy_obs, obs_mask = _generate_observations(
                true_fluid, cfg.obs_interval, cfg.R_var, cfg.seed + 1, self.device,
            )

            self.windows.append({
                "true_state": true_fluid,
                "obs": noisy_obs,
                "obs_mask": obs_mask,
                "forcing_true": W_L_true,
                "forcing_corrupted": W_L_star,
            })

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.windows[idx]

    def get_da_forcing(self, idx: int) -> torch.Tensor:
        if self.cfg.use_corrupted_forcing:
            return self.windows[idx]["forcing_corrupted"]
        return self.windows[idx]["forcing_true"]


def make_datasets(cfg: Lorenz96Config) -> Dict[str, Lorenz96Dataset]:
    train_cfg = Lorenz96Config(**{**cfg.__dict__, "seed": 42, "num_windows": 2000})
    val_cfg = Lorenz96Config(**{**cfg.__dict__, "seed": 99, "num_windows": 200})
    test_cfg_cs1 = Lorenz96Config(**{**cfg.__dict__, "seed": 123, "num_windows": 200, "case": 1, "param_bias": 0.0})
    test_cfg_cs2 = Lorenz96Config(**{**cfg.__dict__, "seed": 123, "num_windows": 200, "case": 2, "param_bias": cfg.param_bias})
    return {
        "train": Lorenz96Dataset(train_cfg),
        "val": Lorenz96Dataset(val_cfg),
        "test_cs1": Lorenz96Dataset(test_cfg_cs1),
        "test_cs2": Lorenz96Dataset(test_cfg_cs2),
    }