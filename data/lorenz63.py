import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class Lorenz63Config:
    case: int = 1
    dt: float = 0.01
    T_max: float = 3.0
    obs_interval: int = 20
    R_var: float = 0.5
    B_var: float = 2.0
    param_bias: float = 0.0
    num_windows: int = 2000
    window_spacing: int = 2000
    spinup_steps: int = 10000
    seed: int = 42

    sigma_true: float = 10.0
    rho_true: float = 28.0
    beta_true: float = 8 / 3

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
    def biased_params(self) -> Tuple[float, float, float]:
        b = self.param_bias
        return (
            self.sigma_true * (1 - b),
            self.rho_true * (1 - b),
            self.beta_true * (1 + b),
        )

    @property
    def da_params(self) -> Tuple[float, float, float]:
        if self.case == 1:
            return (self.sigma_true, self.rho_true, self.beta_true)
        return self.biased_params

    @property
    def use_corrupted_forcing(self) -> bool:
        return self.case == 2


def _coupling(W, c1, exponent):
    if exponent == 1.0:
        return c1 * W
    return c1 * torch.sign(W) * torch.abs(W)**exponent


def generate_long_trajectory(
    num_steps: int, dt: float, seed: int,
    sigma: float, rho: float, beta: float,
    gamma: float, W_L_bar: float, c1: float, c2: float,
    sigma_0: float, sigma_L: float,
    device: torch.device = torch.device("cpu"),
    coupling_exponent: float = 1.5,
) -> torch.Tensor:
    num_steps = int(num_steps)
    rng = torch.Generator(device=device).manual_seed(seed)
    trajectory = torch.zeros(num_steps, 4, device=device)
    state = torch.tensor([1.0, 1.0, 20.0, 0.0], device=device)
    trajectory[0] = state

    sqrt_dt = np.sqrt(dt)
    noise = torch.randn((num_steps, 3), device=device, generator=rng) * sqrt_dt

    for t in range(1, num_steps):
        X, Y, Z, W_L = trajectory[t - 1]
        dW1, dW2, dW3 = noise[t]

        dX = sigma * (Y - X) + _coupling(W_L, c1, coupling_exponent)
        dY = X * (rho - Z) - Y
        dZ = X * Y - beta * Z
        dW_L_term = -gamma * (W_L - W_L_bar) + c2 * X

        X_next = X + dX * dt
        Y_next = Y + dY * dt + sigma_0 * Y * dW1
        Z_next = Z + dZ * dt + sigma_0 * Z * dW2
        W_L_next = W_L + dW_L_term * dt + sigma_L * dW3

        trajectory[t] = torch.tensor([X_next, Y_next, Z_next, W_L_next], device=device)

    return trajectory


def generate_corrupted_forcing(
    W_L_true: torch.Tensor, X: torch.Tensor, num_steps: int, dt: float,
    tau_eta: float, sigma_eta: float, seed: int,
    device: torch.device = torch.device("cpu"),
    state_bias: float = 0.0,
) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    eta = np.zeros(num_steps)
    eta[0] = rng.normal(0, sigma_eta)

    sqrt_dt = np.sqrt(dt)
    for t in range(1, num_steps):
        d_eta = -(1.0 / tau_eta) * eta[t - 1] * dt + sigma_eta * np.sqrt(2.0 / tau_eta) * rng.normal(0, sqrt_dt)
        eta[t] = eta[t - 1] + d_eta

    eta_tensor = torch.tensor(eta, dtype=torch.float32, device=device)
    return W_L_true + eta_tensor + state_bias * X


def generate_observations(
    true_fluid: torch.Tensor, obs_interval: int, R_var: float, seed: int,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_steps = true_fluid.shape[0]
    rng = torch.Generator(device=device).manual_seed(seed)
    obs_indices = np.arange(obs_interval, num_steps, obs_interval)
    obs_mask = torch.zeros(num_steps, dtype=torch.bool, device=device)
    obs_mask[obs_indices] = True
    noisy_obs = true_fluid.clone()
    noisy_obs[obs_indices] += (
        torch.randn((len(obs_indices), 3), device=device, generator=rng) * np.sqrt(R_var)
    )
    return noisy_obs, obs_mask


class Lorenz63Dataset:
    def __init__(self, cfg: Lorenz63Config):
        self.cfg = cfg
        self.device = torch.device("cpu")

        traj_seed = cfg.seed
        obs_seed = cfg.seed + 1

        full_steps = cfg.spinup_steps + (cfg.num_windows + 2) * cfg.window_spacing
        long_traj = generate_long_trajectory(
            num_steps=full_steps, dt=cfg.dt, seed=traj_seed,
            sigma=cfg.sigma_true, rho=cfg.rho_true, beta=cfg.beta_true,
            gamma=cfg.gamma, W_L_bar=cfg.W_L_bar,
            c1=cfg.c1, c2=cfg.c2,
            sigma_0=cfg.sigma_0, sigma_L=cfg.sigma_L,
            device=self.device,
            coupling_exponent=cfg.coupling_exponent_truth,
        )

        self.full_trajectory = long_traj

        start_indices = (
            np.arange(cfg.num_windows) * cfg.window_spacing + cfg.spinup_steps
        ).astype(int)

        self.windows = []
        for idx in start_indices:
            seg = long_traj[idx: idx + cfg.num_steps].clone()
            true_fluid = seg[:, :3]
            W_L_true = seg[:, 3]

            if cfg.use_corrupted_forcing:
                force_seed = cfg.seed + 2 + idx // (cfg.num_steps + 1)
                W_L_star = generate_corrupted_forcing(
                    W_L_true, true_fluid[:, 0], cfg.num_steps, cfg.dt,
                    cfg.tau_eta, cfg.sigma_eta, force_seed,
                    self.device, state_bias=cfg.forcing_state_bias,
                )
            else:
                W_L_star = W_L_true.clone()

            noisy_obs, obs_mask = generate_observations(
                true_fluid, cfg.obs_interval, cfg.R_var, obs_seed,
                self.device,
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


def make_datasets(cfg: Lorenz63Config) -> Dict[str, Lorenz63Dataset]:
    train_cfg = Lorenz63Config(**{**cfg.__dict__, "seed": 42, "num_windows": 2000})
    val_cfg = Lorenz63Config(**{**cfg.__dict__, "seed": 99, "num_windows": 200})
    test_cfg_cs1 = Lorenz63Config(**{**cfg.__dict__, "seed": 123, "num_windows": 200, "case": 1, "param_bias": 0.0})
    test_cfg_cs2 = Lorenz63Config(**{**cfg.__dict__, "seed": 123, "num_windows": 200, "case": 2, "param_bias": cfg.param_bias})

    return {
        "train": Lorenz63Dataset(train_cfg),
        "val": Lorenz63Dataset(val_cfg),
        "test_cs1": Lorenz63Dataset(test_cfg_cs1),
        "test_cs2": Lorenz63Dataset(test_cfg_cs2),
    }


def make_mixed_datasets(cfg: Lorenz63Config, *,
                        num_test_windows: int = 10,
                        include_s1_test: bool = False,
                        param_noise: float = 0.2) -> Dict[str, Lorenz63Dataset]:
    from data.random_param_dataset import RandomParamLorenz63Dataset
    base = cfg.__dict__.copy()

    test_s0_cfg = Lorenz63Config(**{**base, "case": 1, "param_bias": 0.0,
        "forcing_state_bias": 0.0, "seed": 123, "num_windows": num_test_windows})
    out = {
        "test_s0": RandomParamLorenz63Dataset(test_s0_cfg, param_noise=param_noise),
    }

    if include_s1_test:
        from data.random_bias_dataset import RandomBiasLorenz63Dataset
        test_s1_cfg = Lorenz63Config(**{**base, "case": 1, "param_bias": 0.15,
            "forcing_state_bias": 0.1, "seed": 131, "num_windows": num_test_windows})
        out["test_s1"] = RandomBiasLorenz63Dataset(
            test_s1_cfg, param_noise=param_noise, bias_mode='fixed')
    return out


def make_s0_s1_trainval(cfg: Lorenz63Config, *,
                        num_train_windows: int = 1000,
                        num_val_windows: int = 100,
                        num_test_windows: int = 200,
                        param_noise: float = 0.2,
                        bias_range: Tuple[float, float] = (0.0, 0.2)) -> Dict[str, "Lorenz63Dataset"]:
    from data.random_param_dataset import RandomParamLorenz63Dataset
    from data.random_bias_dataset import RandomBiasLorenz63Dataset
    base = cfg.__dict__.copy()

    train_cfg = Lorenz63Config(**{**base, "case": 1, "seed": 42,
        "num_windows": num_train_windows, "param_bias": 0.0,
        "forcing_state_bias": 0.0})
    train = RandomBiasLorenz63Dataset(
        train_cfg, param_noise=param_noise, bias_mode='random', bias_range=bias_range)

    val_cfg = Lorenz63Config(**{**base, "case": 1, "seed": 99,
        "num_windows": num_val_windows, "param_bias": 0.0,
        "forcing_state_bias": 0.0})
    val = RandomBiasLorenz63Dataset(
        val_cfg, param_noise=param_noise, bias_mode='random', bias_range=bias_range)

    test_s0_cfg = Lorenz63Config(**{**base, "case": 1, "param_bias": 0.0,
        "forcing_state_bias": 0.0, "seed": 123, "num_windows": num_test_windows})
    test_s0 = RandomParamLorenz63Dataset(test_s0_cfg, param_noise=param_noise)

    test_s1_cfg = Lorenz63Config(**{**base, "case": 1, "param_bias": 0.15,
        "forcing_state_bias": 0.1, "seed": 131, "num_windows": num_test_windows})
    test_s1 = RandomBiasLorenz63Dataset(
        test_s1_cfg, param_noise=param_noise, bias_mode='fixed')

    return {
        "train": train,
        "val": val,
        "test_s0": test_s0,
        "test_s1": test_s1,
    }
