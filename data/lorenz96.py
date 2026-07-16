import torch
import numpy as np
from dataclasses import dataclass, field
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
    obs_var_indices: Tuple[int, ...] = None
    fast_weights: list = field(default_factory=lambda: [1.0, 1.0, 0.1, 0.1])

    @property
    def obs_dim(self) -> int:
        if self.obs_var_indices is not None:
            return len(self.obs_var_indices)
        return self.state_dim

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
    obs_var_indices: np.ndarray = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    true_fluid = true_fluid.to(device=device)
    num_steps, sd = true_fluid.shape[0], true_fluid.shape[-1]
    obs_dim = len(obs_var_indices) if obs_var_indices is not None else sd
    if isinstance(R_var, np.ndarray):
        noise_std = np.sqrt(R_var)
    else:
        noise_std = np.sqrt(R_var)
    rng = torch.Generator(device=device).manual_seed(seed)
    obs_time_indices = np.arange(0, num_steps, obs_interval)
    obs_mask = torch.zeros(num_steps, dtype=torch.bool, device=device)
    obs_mask[obs_time_indices] = True
    noisy_obs = torch.full((num_steps, obs_dim), float('nan'), device=device)
    if obs_var_indices is not None:
        selected = true_fluid[:, obs_var_indices]
        obs_fluid = selected[obs_time_indices]
    else:
        obs_fluid = true_fluid[obs_time_indices]
    noisy_obs[obs_time_indices] = obs_fluid + (
        torch.randn((len(obs_time_indices), obs_dim), device=device, generator=rng) * torch.tensor(noise_std, dtype=torch.float32, device=device)
    )
    return noisy_obs, obs_mask


def estimate_l96_component_variances(NO=8, J=4, dt=0.001, T_max=50.0, F=8.0, seed=42):
    from models.lorenz96_dynamics import Lorenz96Dynamics
    dyn = Lorenz96Dynamics(dt=dt, coupling_exponent=1.6)
    traj, _ = dyn.generate_full_trajectory(
        num_steps=int(T_max / dt), seed=seed, F=F,
        coupling_exponent=1.6, spinup_steps=5000,
    )
    var_per_dim = torch.var(traj, dim=0).numpy()
    return var_per_dim


def _make_lorenz96_dynamics(cfg: Lorenz96Config):
    from models.lorenz96_dynamics import Lorenz96Dynamics
    return Lorenz96Dynamics(
        dt=cfg.dt, coupling_exponent=cfg.coupling_exponent_truth,
        c1=cfg.c1, NO=cfg.NO, J=cfg.J, h=cfg.h, hx=cfg.hx, eps=cfg.eps,
        sigma_0=cfg.sigma_0, gamma=cfg.gamma,
        W_L_bar=cfg.W_L_bar, c2=cfg.c2, sigma_L=cfg.sigma_L,
        fast_weights=cfg.fast_weights,
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
                obs_var_indices=cfg.obs_var_indices,
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


def _make_obs(cfg, true_fluid, obs_seed, device=None):
    device = device or torch.device("cpu")
    return _generate_observations(true_fluid, cfg.obs_interval, cfg.R_var, obs_seed, device,
                                  obs_var_indices=cfg.obs_var_indices)


def _make_corrupted_forcing(cfg, W_L_true, true_fluid, seed, device=None):
    device = device or torch.device("cpu")
    num_steps = cfg.num_steps
    eta = np.zeros(num_steps)
    rng = np.random.RandomState(seed)
    sqrt_dt = np.sqrt(cfg.dt)
    for et in range(1, num_steps):
        d_eta = -(1.0 / cfg.tau_eta) * eta[et - 1] * cfg.dt + cfg.sigma_eta * np.sqrt(2.0 / cfg.tau_eta) * rng.normal(0, sqrt_dt)
        eta[et] = eta[et - 1] + d_eta
    return W_L_true + cfg.forcing_state_bias * true_fluid[:, 0] + torch.tensor(eta, dtype=true_fluid.dtype, device=device)


class RandomParamLorenz96Dataset:
    def __init__(self, cfg: Lorenz96Config, param_noise: float = 0.2,
                 dynamics=None, cached_windows: list = None,
                 max_window_retries: int = 10):
        self.cfg = cfg
        self.param_noise = param_noise
        self.device = torch.device("cpu")
        self.dynamics = dynamics or _make_lorenz96_dynamics(cfg)

        if cached_windows is not None:
            self.windows = cached_windows
            return

        self.windows = []
        for i in range(cfg.num_windows):
            base_seed = cfg.seed + i * 100
            for attempt in range(max_window_retries):
                traj_seed = base_seed + attempt
                obs_seed = cfg.seed + i * 100 + 1 + attempt
                lo = 1.0 - param_noise
                hi = 1.0 + param_noise
                rng_seed = traj_seed
                rng_np = np.random.RandomState(rng_seed)
                F = cfg.F_true * rng_np.uniform(lo, hi)
                try:
                    true_fluid, W_L_true = self.dynamics.generate_full_trajectory(
                        num_steps=cfg.num_steps, seed=traj_seed, F=F,
                        spinup_steps=cfg.spinup_steps,
                        coupling_exponent=cfg.coupling_exponent_truth,
                    )
                except RuntimeError:
                    continue
                if torch.isfinite(true_fluid).all():
                    break
            else:
                raise RuntimeError(f"RandomParamLorenz96Dataset window {i} unstable (seed={cfg.seed})")

            if cfg.use_corrupted_forcing:
                W_L_star = _make_corrupted_forcing(cfg, W_L_true, true_fluid, traj_seed, self.device)
            else:
                W_L_star = W_L_true.clone()

            noisy_obs, obs_mask = _make_obs(cfg, true_fluid, obs_seed, self.device)
            self.windows.append({
                "true_state": true_fluid, "obs": noisy_obs, "obs_mask": obs_mask,
                "forcing_true": W_L_true, "forcing_corrupted": W_L_star,
                "F": F, "true_F": F,
                "obs_seed": obs_seed,
            })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        if "obs" not in w or "obs_mask" not in w:
            obs_seed = w.get("obs_seed", self.cfg.obs_interval + idx)
            obs, obs_mask = _make_obs(self.cfg, w["true_state"], obs_seed, self.device)
            w["obs"] = obs
            w["obs_mask"] = obs_mask
        return w


class RandomBiasLorenz96Dataset:
    def __init__(self, cfg: Lorenz96Config, param_noise: float = 0.2,
                 dynamics=None, cached_windows: list = None,
                 max_window_retries: int = 10):
        self.cfg = cfg
        self.param_noise = param_noise
        self.device = torch.device("cpu")
        self.dynamics = dynamics or _make_lorenz96_dynamics(cfg)

        if cached_windows is not None:
            self.windows = cached_windows
            return

        self.windows = []
        for i in range(cfg.num_windows):
            base_seed = cfg.seed + i * 100
            for attempt in range(max_window_retries):
                traj_seed = base_seed + attempt
                obs_seed = cfg.seed + i * 100 + 1 + attempt
                lo = 1.0 - param_noise
                hi = 1.0 + param_noise
                rng_np = np.random.RandomState(traj_seed)
                F = cfg.F_true * rng_np.uniform(lo, hi)
                try:
                    true_fluid, W_L_true = self.dynamics.generate_full_trajectory(
                        num_steps=cfg.num_steps, seed=traj_seed, F=F,
                        spinup_steps=cfg.spinup_steps,
                        coupling_exponent=cfg.coupling_exponent_truth,
                    )
                except RuntimeError:
                    continue
                if torch.isfinite(true_fluid).all():
                    break
            else:
                raise RuntimeError(f"RandomBiasLorenz96Dataset window {i} unstable (seed={cfg.seed})")

            W_L_star = _make_corrupted_forcing(cfg, W_L_true, true_fluid, traj_seed, self.device)
            noisy_obs, obs_mask = _make_obs(cfg, true_fluid, obs_seed, self.device)
            self.windows.append({
                "true_state": true_fluid, "obs": noisy_obs, "obs_mask": obs_mask,
                "forcing_true": W_L_true, "forcing_corrupted": W_L_star,
                "F": F, "true_F": F,
                "obs_seed": obs_seed,
            })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        if "obs" not in w or "obs_mask" not in w:
            obs_seed = w.get("obs_seed", self.cfg.obs_interval + idx)
            obs, obs_mask = _make_obs(self.cfg, w["true_state"], obs_seed, self.device)
            w["obs"] = obs
            w["obs_mask"] = obs_mask
        return w


def make_l96_s0_s1_datasets(cfg: Lorenz96Config, *,
                            num_test_windows: int = 200) -> Dict:
    dynamics = _make_lorenz96_dynamics(cfg)
    test_s0_cfg = Lorenz96Config(**{**cfg.__dict__, "case": 1, "param_bias": 0.0,
        "forcing_state_bias": 0.0, "seed": 123, "num_windows": num_test_windows})
    test_s1_cfg = Lorenz96Config(**{**cfg.__dict__, "case": 1, "param_bias": 0.15,
        "forcing_state_bias": 0.1, "seed": 131, "num_windows": num_test_windows})
    return {
        "test_s0": RandomParamLorenz96Dataset(test_s0_cfg, param_noise=0.2, dynamics=dynamics),
        "test_s1": RandomBiasLorenz96Dataset(test_s1_cfg, param_noise=0.2, dynamics=dynamics),
    }