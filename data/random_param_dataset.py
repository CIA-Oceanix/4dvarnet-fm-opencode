import torch
from typing import Dict
from data.lorenz63 import (
    Lorenz63Config,
    generate_long_trajectory,
    generate_corrupted_forcing,
    generate_observations,
)


class RandomParamLorenz63Dataset:
    def __init__(self, cfg: Lorenz63Config, param_noise: float = 0.2,
                 cached_windows: list = None,
                 max_window_retries: int = 10):
        self.cfg = cfg
        self.param_noise = param_noise
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
                sigma = torch.empty(1, device=self.device).uniform_(cfg.sigma_true * lo, cfg.sigma_true * hi, generator=rng).item()
                rho = torch.empty(1, device=self.device).uniform_(cfg.rho_true * lo, cfg.rho_true * hi, generator=rng).item()
                beta = torch.empty(1, device=self.device).uniform_(cfg.beta_true * lo, cfg.beta_true * hi, generator=rng).item()

                try:
                    traj = generate_long_trajectory(
                        num_steps=total_steps, dt=cfg.dt, seed=traj_seed,
                        sigma=sigma, rho=rho, beta=beta,
                        gamma=cfg.gamma, W_L_bar=cfg.W_L_bar,
                        c1=cfg.c1, c2=cfg.c2,
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
                    f"RandomParamLorenz63Dataset window {i} unstable after "
                    f"{max_window_retries} retries (cfg.seed={cfg.seed})"
                )

            seg = traj[-cfg.num_steps:].clone()
            true_fluid = seg[:, :3]
            W_L_true = seg[:, 3]

            if cfg.use_corrupted_forcing:
                W_L_star = generate_corrupted_forcing(
                    W_L_true, true_fluid[:, 0], cfg.num_steps, cfg.dt,
                    cfg.tau_eta, cfg.sigma_eta, traj_seed,
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
                "sigma": sigma,
                "rho": rho,
                "beta": beta,
                "true_sigma": sigma,
                "true_rho": rho,
                "true_beta": beta,
                "true_c1": cfg.c1,
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
