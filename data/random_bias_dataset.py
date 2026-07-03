import torch
from typing import Dict, Tuple
from data.lorenz63 import (
    Lorenz63Config,
    generate_long_trajectory,
    generate_corrupted_forcing,
    generate_observations,
)


class RandomBiasLorenz63Dataset:
    def __init__(self, cfg: Lorenz63Config, param_noise: float = 0.2,
                 bias_range: Tuple[float, float] = (0.0, 0.20)):
        self.cfg = cfg
        self.param_noise = param_noise
        self.bias_range = bias_range
        self.device = torch.device("cpu")
        self.windows = []

        total_steps = cfg.spinup_steps + cfg.num_steps

        for i in range(cfg.num_windows):
            traj_seed = cfg.seed + i * 100
            obs_seed = cfg.seed + i * 100 + 1

            rng = torch.Generator(device=self.device).manual_seed(traj_seed)

            lo = 1.0 - param_noise
            hi = 1.0 + param_noise
            sigma0 = torch.empty(1, device=self.device).uniform_(cfg.sigma_true * lo, cfg.sigma_true * hi, generator=rng).item()
            rho0 = torch.empty(1, device=self.device).uniform_(cfg.rho_true * lo, cfg.rho_true * hi, generator=rng).item()
            beta0 = torch.empty(1, device=self.device).uniform_(cfg.beta_true * lo, cfg.beta_true * hi, generator=rng).item()

            param_bias = torch.empty(1, device=self.device).uniform_(bias_range[0], bias_range[1], generator=rng).item()
            forcing_state_bias = torch.empty(1, device=self.device).uniform_(bias_range[0], bias_range[1], generator=rng).item()

            sigma = sigma0 * (1.0 - param_bias)
            rho = rho0 * (1.0 - param_bias)
            beta = beta0 * (1.0 + param_bias)

            traj = generate_long_trajectory(
                num_steps=total_steps, dt=cfg.dt, seed=traj_seed,
                sigma=sigma, rho=rho, beta=beta,
                gamma=cfg.gamma, W_L_bar=cfg.W_L_bar,
                c1=cfg.c1, c2=cfg.c2,
                sigma_0=cfg.sigma_0, sigma_L=cfg.sigma_L,
                device=self.device,
            )

            seg = traj[-cfg.num_steps:].clone()
            true_fluid = seg[:, :3]
            W_L_true = seg[:, 3]

            if cfg.use_corrupted_forcing:
                W_L_star = generate_corrupted_forcing(
                    W_L_true, true_fluid[:, 0], cfg.num_steps, cfg.dt,
                    cfg.tau_eta, cfg.sigma_eta, traj_seed,
                    self.device, state_bias=forcing_state_bias,
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
                "sigma": sigma,
                "rho": rho,
                "beta": beta,
                "param_bias": param_bias,
                "forcing_state_bias": forcing_state_bias,
            })

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.windows[idx]
