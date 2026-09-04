import math
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class Lorenz63Config:
    case: int = 1
    dt: float = 0.01
    T_max: float = 3.0
    obs_interval: int = 30
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


def _coupling_scalar(W: float, c1: float, exponent: float) -> float:
    if exponent == 1.0:
        return c1 * W
    if W == 0.0:
        return 0.0
    return c1 * math.copysign(abs(W) ** exponent, W)


def generate_long_trajectory(
    num_steps: int, dt: float, seed: int,
    sigma: float, rho: float, beta: float,
    gamma: float, W_L_bar: float, c1: float, c2: float,
    sigma_0: float, sigma_L: float,
    device: torch.device = torch.device("cpu"),
    coupling_exponent: float = 1.5,
    max_retries: int = 10,
) -> torch.Tensor:
    num_steps = int(num_steps)
    base_seed = seed
    for attempt in range(max_retries):
        current_seed = base_seed + attempt
        rng = torch.Generator(device=device).manual_seed(current_seed)

        # The Euler-Maruyama recurrence below is inherently sequential (state[t]
        # depends on state[t-1]), so it's kept as a plain Python loop. It uses
        # plain floats + a numpy buffer rather than per-step torch tensor
        # indexing/construction, which previously dominated the runtime with
        # dispatcher/object-creation overhead. Noise is still drawn from
        # torch's RNG (vectorized, so seeding/determinism is unaffected) and
        # only converted to numpy once.
        sqrt_dt = np.sqrt(dt)
        noise = (torch.randn((num_steps, 3), device=device, generator=rng) * sqrt_dt).cpu().numpy()

        trajectory = np.zeros((num_steps, 4), dtype=np.float32)
        X, Y, Z, W_L = 1.0, 1.0, 20.0, 0.0
        trajectory[0] = (X, Y, Z, W_L)

        for t in range(1, num_steps):
            dW1, dW2, dW3 = noise[t]

            dX = sigma * (Y - X) + _coupling_scalar(W_L, c1, coupling_exponent)
            dY = X * (rho - Z) - Y
            dZ = X * Y - beta * Z
            dW_L_term = -gamma * (W_L - W_L_bar) + c2 * X

            X = X + dX * dt
            Y = Y + dY * dt + sigma_0 * Y * dW1
            Z = Z + dZ * dt + sigma_0 * Z * dW2
            W_L = W_L + dW_L_term * dt + sigma_L * dW3

            trajectory[t, 0] = X
            trajectory[t, 1] = Y
            trajectory[t, 2] = Z
            trajectory[t, 3] = W_L

        if np.isfinite(trajectory).all():
            return torch.from_numpy(trajectory).to(device)

    raise RuntimeError(
        f"generate_long_trajectory diverged after {max_retries} retries "
        f"(seed={base_seed}, sigma={sigma:.2f}, rho={rho:.2f}, beta={beta:.2f})"
    )


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
    obs_indices = np.arange(0, num_steps, obs_interval)
    obs_mask = torch.zeros(num_steps, dtype=torch.bool, device=device)
    obs_mask[obs_indices] = True
    noisy_obs = torch.full_like(true_fluid, float('nan'))
    noisy_obs[obs_indices] = true_fluid[obs_indices] + (
        torch.randn((len(obs_indices), true_fluid.shape[-1]), device=device, generator=rng) * np.sqrt(R_var)
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




def _cfg_to_data_dict(cfg: Lorenz63Config) -> dict:
    d = cfg.__dict__.copy()
    d["num_steps"] = cfg.num_steps
    d["num_windows"] = cfg.num_windows
    return d


def _make_lorenz63_dynamics(cfg: Lorenz63Config):
    from models.lorenz63_dynamics import Lorenz63Dynamics
    return Lorenz63Dynamics(
        dt=cfg.dt, coupling_exponent=cfg.coupling_exponent_truth,
        c1=cfg.c1, sigma_0=cfg.sigma_0,
        gamma=cfg.gamma, W_L_bar=cfg.W_L_bar,
        c2=cfg.c2, sigma_L=cfg.sigma_L,
    )


def make_mixed_datasets(cfg: Lorenz63Config, *,
                        num_test_windows: int = 10,
                        include_s1_test: bool = False,
                        param_noise: float = 0.2) -> Dict[str, Lorenz63Dataset]:
    from data.random_param_dataset import RandomParamLorenz63Dataset
    base = _cfg_to_data_dict(cfg)
    dynamics = _make_lorenz63_dynamics(cfg)

    test_s0_cfg = Lorenz63Config(**{**cfg.__dict__, "case": 1, "param_bias": 0.0,
        "forcing_state_bias": 0.0, "seed": 123, "num_windows": num_test_windows})
    out = {
        "test_s0": RandomParamLorenz63Dataset(test_s0_cfg, param_noise=param_noise, dynamics=dynamics),
    }
    if include_s1_test:
        from data.random_bias_dataset import RandomBiasLorenz63Dataset
        test_s1_cfg = Lorenz63Config(**{**cfg.__dict__, "case": 1, "param_bias": 0.15,
            "forcing_state_bias": 0.1, "seed": 131, "num_windows": num_test_windows})
        ds = RandomBiasLorenz63Dataset(
            test_s1_cfg, param_noise=param_noise, dynamics=dynamics, bias_mode='fixed')
        out["test_s1"] = ds
    return out


def _make_s0_s1_cache_key(cfg: Lorenz63Config, *,
                          num_train_windows: int,
                          num_val_windows: int,
                          num_test_windows: int,
                          param_noise: float,
                          bias_range: Tuple[float, float],
                          train_seed: int, val_seed: int,
                          s0_seed: int, s1_seed: int) -> str:
    import hashlib
    key_data = {
        "num_train_windows": num_train_windows,
        "num_val_windows": num_val_windows,
        "num_test_windows": num_test_windows,
        "param_noise": param_noise,
        "bias_range": bias_range,
        "train_seed": train_seed, "val_seed": val_seed,
        "s0_seed": s0_seed, "s1_seed": s1_seed,
        "dt": cfg.dt, "T_max": cfg.T_max,
        "obs_interval": cfg.obs_interval, "R_var": cfg.R_var,
        "spinup_steps": cfg.spinup_steps, "window_spacing": cfg.window_spacing,
        "seed": cfg.seed,
        "sigma_true": cfg.sigma_true, "rho_true": cfg.rho_true, "beta_true": cfg.beta_true,
        "gamma": cfg.gamma, "W_L_bar": cfg.W_L_bar,
        "c1": cfg.c1, "c2": cfg.c2,
        "sigma_0": cfg.sigma_0, "sigma_L": cfg.sigma_L,
        "coupling_exponent_truth": cfg.coupling_exponent_truth,
        "param_bias": cfg.param_bias,
        "forcing_state_bias": cfg.forcing_state_bias,
    }
    return hashlib.sha256(str(sorted(key_data.items())).encode()).hexdigest()


def make_s0_s1_trainval(cfg: Lorenz63Config, *,
                        num_train_windows: int = 1000,
                        num_val_windows: int = 100,
                        num_test_windows: int = 200,
                        param_noise: float = 0.2,
                        bias_range: Tuple[float, float] = (0.0, 0.2),
                        train_seed: int = None, val_seed: int = None,
                        s0_seed: int = None, s1_seed: int = None,
                        require_cache: bool = False) -> Dict[str, "Lorenz63Dataset"]:
    import os
    from data.random_param_dataset import RandomParamLorenz63Dataset
    from data.random_bias_dataset import RandomBiasLorenz63Dataset
    train_seed = 42 if train_seed is None else train_seed
    val_seed = 99 if val_seed is None else val_seed
    s0_seed = 123 if s0_seed is None else s0_seed
    s1_seed = 131 if s1_seed is None else s1_seed
    base = cfg.__dict__.copy()

    train_cfg = Lorenz63Config(**{**base, "case": 1, "seed": train_seed,
        "num_windows": num_train_windows, "param_bias": 0.0,
        "forcing_state_bias": 0.0})
    val_cfg = Lorenz63Config(**{**base, "case": 1, "seed": val_seed,
        "num_windows": num_val_windows, "param_bias": 0.0,
        "forcing_state_bias": 0.0})
    test_s0_cfg = Lorenz63Config(**{**base, "case": 1, "param_bias": 0.0,
        "forcing_state_bias": 0.0, "seed": s0_seed, "num_windows": num_test_windows})
    test_s1_cfg = Lorenz63Config(**{**base, "case": 1, "param_bias": 0.15,
        "forcing_state_bias": 0.1, "seed": s1_seed, "num_windows": num_test_windows})

    cache_dir = os.path.join(os.path.dirname(__file__), "..", "dataset_cache", "l63")
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = _make_s0_s1_cache_key(
        cfg, num_train_windows=num_train_windows,
        num_val_windows=num_val_windows, num_test_windows=num_test_windows,
        param_noise=param_noise, bias_range=bias_range,
        train_seed=train_seed, val_seed=val_seed,
        s0_seed=s0_seed, s1_seed=s1_seed)
    cache_path = os.path.join(cache_dir, f"{cache_key}.pt")

    if require_cache and not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Could not find a cached dataset for data config hash {cache_key[:12]} "
            f"(expected at {cache_path}). Generate it first with:\n"
            f"    python generate_dataset.py\n"
            f"or, for an experiment-specific data config:\n"
            f"    python generate_dataset.py --config-name models/<your_config>"
        )

    if os.path.exists(cache_path):
        print(f"  Loading cached datasets ({cache_key[:12]}...)")
        cached = torch.load(cache_path)
        train = RandomBiasLorenz63Dataset(
            train_cfg, param_noise=param_noise, bias_mode='random',
            bias_range=bias_range, cached_windows=cached["train_windows"])
        val = RandomBiasLorenz63Dataset(
            val_cfg, param_noise=param_noise, bias_mode='random',
            bias_range=bias_range, cached_windows=cached["val_windows"])
        test_s0 = RandomParamLorenz63Dataset(
            test_s0_cfg, param_noise=param_noise,
            cached_windows=cached["test_s0_windows"])
        test_s1 = RandomBiasLorenz63Dataset(
            test_s1_cfg, param_noise=param_noise, bias_mode='fixed',
            cached_windows=cached["test_s1_windows"])
        return {"train": train, "val": val,
                "test_s0": test_s0, "test_s1": test_s1}

    train = RandomBiasLorenz63Dataset(
        train_cfg, param_noise=param_noise, bias_mode='random', bias_range=bias_range)
    val = RandomBiasLorenz63Dataset(
        val_cfg, param_noise=param_noise, bias_mode='random', bias_range=bias_range)
    test_s0 = RandomParamLorenz63Dataset(test_s0_cfg, param_noise=param_noise)
    test_s1 = RandomBiasLorenz63Dataset(
        test_s1_cfg, param_noise=param_noise, bias_mode='fixed')

    def _strip_obs(w):
        w.pop("obs", None)
        w.pop("obs_mask", None)
        return w

    tmp_path = cache_path + ".tmp"
    torch.save({
        "train_windows": [_strip_obs(w) for w in train.windows],
        "val_windows": [_strip_obs(w) for w in val.windows],
        "test_s0_windows": [_strip_obs(w) for w in test_s0.windows],
        "test_s1_windows": [_strip_obs(w) for w in test_s1.windows],
    }, tmp_path)
    os.rename(tmp_path, cache_path)
    print(f"  Cached datasets ({cache_key[:12]}...)")

    return {
        "train": train,
        "val": val,
        "test_s0": test_s0,
        "test_s1": test_s1,
    }
