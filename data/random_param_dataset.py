import torch
from typing import Dict
from data.lorenz63 import Lorenz63Config, generate_corrupted_forcing, generate_observations
from models.dynamics import DynamicsBase
from models.lorenz63_dynamics import Lorenz63Dynamics


class RandomParamDataset:
    def __init__(self, cfg, param_noise: float = 0.2,
                 dynamics: DynamicsBase = None,
                 cached_windows: list = None,
                 max_window_retries: int = 10):
        if dynamics is None and isinstance(cfg, Lorenz63Config):
            dynamics = Lorenz63Dynamics(
                dt=cfg.dt, coupling_exponent=cfg.coupling_exponent_truth,
                c1=cfg.c1, sigma_0=cfg.sigma_0,
                gamma=cfg.gamma, W_L_bar=cfg.W_L_bar,
                c2=cfg.c2, sigma_L=cfg.sigma_L,
            )
        self.cfg = cfg if isinstance(cfg, Lorenz63Config) else cfg
        self.dynamics = dynamics
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
                    true_fluid, W_L_true = dynamics.generate_full_trajectory(
                        num_steps=cfg.num_steps, seed=traj_seed,
                        device=self.device,
                        sigma=sigma, rho=rho, beta=beta,
                        spinup_steps=cfg.spinup_steps,
                        coupling_exponent=cfg.coupling_exponent_truth,
                    )
                except RuntimeError:
                    continue

                if torch.isfinite(true_fluid).all():
                    break
            else:
                raise RuntimeError(
                    f"RandomParamDataset window {i} unstable after "
                    f"{max_window_retries} retries (cfg.seed={cfg.seed})"
                )

            use_corrupted = cfg.use_corrupted_forcing
            if use_corrupted:
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


RandomParamLorenz63Dataset = RandomParamDataset