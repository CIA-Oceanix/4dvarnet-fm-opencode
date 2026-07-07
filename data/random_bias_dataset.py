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
                 bias_range: Tuple[float, float] = (0.0, 0.20),
                 bias_mode: str = 'random',
                 cached_windows: list = None,
                 max_window_retries: int = 10):
        self.cfg = cfg
        self.param_noise = param_noise
        self.bias_range = bias_range
        self.bias_mode = bias_mode
        self.device = torch.device("cpu")

        if cached_windows is not None:
            self.windows = cached_windows
            return

        self.windows = []
        total_steps = cfg.spinup_steps + cfg.num_steps

        for i in range(cfg.num_windows):
            base_seed = cfg.seed + i * 100
            for attempt in range(max_window_retries):
                traj_seed = base_seed + attempt
                obs_seed = cfg.seed + i * 100 + 1 + attempt

                rng = torch.Generator(device=self.device).manual_seed(traj_seed)

                lo = 1.0 - param_noise
                hi = 1.0 + param_noise
                sigma0 = torch.empty(1, device=self.device).uniform_(cfg.sigma_true * lo, cfg.sigma_true * hi, generator=rng).item()
                rho0 = torch.empty(1, device=self.device).uniform_(cfg.rho_true * lo, cfg.rho_true * hi, generator=rng).item()
                beta0 = torch.empty(1, device=self.device).uniform_(cfg.beta_true * lo, cfg.beta_true * hi, generator=rng).item()

                if bias_mode == 'fixed':
                    p_bias = cfg.param_bias
                    f_bias = cfg.forcing_state_bias
                    truth_sigma, truth_rho, truth_beta = sigma0, rho0, beta0
                    truth_c1 = 1.0
                    da_c1 = 1.0 * (1.0 - p_bias)
                else:
                    p_bias = torch.empty(1, device=self.device).uniform_(bias_range[0], bias_range[1], generator=rng).item()
                    f_bias = torch.empty(1, device=self.device).uniform_(bias_range[0], bias_range[1], generator=rng).item()
                    truth_sigma = sigma0 * (1.0 - p_bias)
                    truth_rho = rho0 * (1.0 - p_bias)
                    truth_beta = beta0 * (1.0 + p_bias)
                    truth_c1 = cfg.c1
                    da_c1 = cfg.c1

                try:
                    traj = generate_long_trajectory(
                        num_steps=total_steps, dt=cfg.dt, seed=traj_seed,
                        sigma=truth_sigma, rho=truth_rho, beta=truth_beta,
                        gamma=cfg.gamma, W_L_bar=cfg.W_L_bar,
                        c1=truth_c1, c2=cfg.c2,
                        sigma_0=cfg.sigma_0, sigma_L=cfg.sigma_L,
                        device=self.device,
                        coupling_exponent=cfg.coupling_exponent_truth,
                    )
                except RuntimeError:
                    continue

                if torch.isfinite(traj).all():
                    break
            else:
                raise RuntimeError(
                    f"RandomBiasLorenz63Dataset window {i} unstable after "
                    f"{max_window_retries} retries (cfg.seed={cfg.seed})"
                )

            seg = traj[-cfg.num_steps:].clone()
            true_fluid = seg[:, :3]
            W_L_true = seg[:, 3]

            if cfg.use_corrupted_forcing or (bias_mode == 'fixed' and abs(f_bias) > 0):
                W_L_star = generate_corrupted_forcing(
                    W_L_true, true_fluid[:, 0], cfg.num_steps, cfg.dt,
                    cfg.tau_eta, cfg.sigma_eta, traj_seed,
                    self.device, state_bias=f_bias,
                )
            else:
                W_L_star = W_L_true.clone()

            noisy_obs, obs_mask = generate_observations(
                true_fluid, cfg.obs_interval, cfg.R_var, obs_seed,
                self.device,
            )

            da_sigma = sigma0 * (1.0 - p_bias)
            da_rho = rho0 * (1.0 - p_bias)
            da_beta = beta0 * (1.0 + p_bias)

            self.windows.append({
                "true_state": true_fluid,
                "obs": noisy_obs,
                "obs_mask": obs_mask,
                "forcing_true": W_L_true,
                "forcing_corrupted": W_L_star,
                "sigma": da_sigma,
                "rho": da_rho,
                "beta": da_beta,
                "c1": da_c1,
                "true_sigma": truth_sigma,
                "true_rho": truth_rho,
                "true_beta": truth_beta,
                "true_c1": truth_c1,
                "param_bias": p_bias,
                "forcing_state_bias": f_bias,
                "obs_seed": obs_seed,
            })

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        w = self.windows[idx]
        if "obs" not in w or "obs_mask" not in w:
            obs_seed = w.get("obs_seed", self.cfg.obs_interval + idx)
            obs, obs_mask = generate_observations(
                w["true_state"], self.cfg.obs_interval, self.cfg.R_var, obs_seed)
            w["obs"] = obs
            w["obs_mask"] = obs_mask
        return w
