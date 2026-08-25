import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class Lorenz96Config:
    case: int = 1
    dt: float = 0.001
    T_max: float = 3.0
    obs_interval: int = 100
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
    randomize: Dict[str, Dict] = field(default_factory=dict)

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


def _l96_param_refs(cfg) -> Dict:
    refs = {"F": cfg.F_true, "c1": cfg.c1, "h": cfg.h, "hx": cfg.hx, "eps": cfg.eps}
    refs["fast_weights"] = list(cfg.fast_weights)
    return refs


def _set_window_params(w: Dict, params: Dict, prefix: str = "", suffix: str = "") -> None:
    """Write scalar param keys into a window dict, flattening fast_weights.

    `fast_weights` (list of 4) is exposed as scalar `w1..w4` keys so the
    generic scalar param-extraction path in FlowMatchingDataset can read the
    full L96 8-param vector without list-aware handling. The key for param `k`
    is `{prefix}{k}{suffix}` — e.g. prefix="" (current), prefix="true_"
    (ground truth), suffix="_da" (biased DA params) — mirroring the existing
    `{k}`, `true_{k}`, `{k}_da` conventions; fast_weights becomes `w{j}`.
    """
    for k, v in params.items():
        if k == "fast_weights":
            for j, wj in enumerate(v, start=1):
                w[f"{prefix}w{j}{suffix}"] = wj
        else:
            w[f"{prefix}{k}{suffix}"] = v


def _uses_perparam_randomize(cfg) -> bool:
    return bool(getattr(cfg, "randomize", None) or {})


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


def _make_corrupted_forcing_batch(cfg, W_L_true, true_fluid, seeds, device=None):
    """Vectorized OU corrupted-forcing for a batch of windows.

    Args:
        W_L_true: tensor (B, num_steps) true forcing
        true_fluid: tensor (B, num_steps, D)
        seeds: array (B,) per-window seeds
    Returns:
        tensor (B, num_steps)
    """
    device = device or W_L_true.device
    B = W_L_true.shape[0]
    num_steps = cfg.num_steps
    sq = np.sqrt(2.0 / cfg.tau_eta)
    isinstance_rng = isinstance(seeds, np.ndarray)
    rng = np.random.RandomState(seeds) if isinstance_rng else np.random.RandomState(np.asarray(seeds))
    eta = np.zeros((B, num_steps))
    for et in range(1, num_steps):
        d_eta = -(1.0 / cfg.tau_eta) * eta[:, et - 1] * cfg.dt + cfg.sigma_eta * sq * rng.normal(0.0, np.sqrt(cfg.dt), size=B)
        eta[:, et] = eta[:, et - 1] + d_eta
    eta_t = torch.tensor(eta, dtype=torch.float32, device=device)
    return W_L_true + cfg.forcing_state_bias * true_fluid[:, :, 0] + eta_t


def _draw_l96_params(rng, cfg, param_noise: float = 0.2, bias: float = None,
                      randomize_params: list = None) -> Dict:
    use_perparam = _uses_perparam_randomize(cfg)
    refs = _l96_param_refs(cfg)

    def _draw_scalar(k, ref, spec, rparams):
        if use_perparam:
            spec = spec if spec is not None else {"randomized": False}
            if spec.get("randomized"):
                noise = spec.get("noise", 0.2)
                return ref * rng.uniform(1.0 - noise, 1.0 + noise)
            return ref
        if rparams is not None and k not in rparams:
            return ref
        val = ref * rng.uniform(1.0 - param_noise, 1.0 + param_noise)
        if bias is not None:
            val *= rng.uniform(1.0 - bias, 1.0 + bias)
        return val

    params = {}
    for k, ref in refs.items():
        if k == "fast_weights":
            if use_perparam:
                spec = cfg.randomize.get("fast_weights") or {"randomized": False}
                if spec.get("randomized"):
                    noise = spec.get("noise", 0.2)
                    params[k] = [r * rng.uniform(1.0 - noise, 1.0 + noise) for r in ref]
                else:
                    params[k] = list(ref)
            else:
                if randomize_params is not None and "fast_weights" not in randomize_params:
                    params[k] = list(ref)
                elif randomize_params is not None and "fast_weights" in randomize_params:
                    params[k] = [r * rng.uniform(1.0 - param_noise, 1.0 + param_noise) for r in ref]
                else:
                    params[k] = list(ref)
        else:
            spec = cfg.randomize.get(k) if use_perparam else None
            params[k] = _draw_scalar(k, ref, spec, randomize_params)
    return params


def _param_tensors(params_list, device):
    """Stack per-window param dicts into per-param tensors on `device`."""
    F = torch.tensor([p["F"] for p in params_list], dtype=torch.float32, device=device)
    c1 = torch.tensor([p["c1"] for p in params_list], dtype=torch.float32, device=device)
    h = torch.tensor([p["h"] for p in params_list], dtype=torch.float32, device=device)
    hx = torch.tensor([p["hx"] for p in params_list], dtype=torch.float32, device=device)
    eps = torch.tensor([p["eps"] for p in params_list], dtype=torch.float32, device=device)
    fw = None
    if "fast_weights" in params_list[0]:
        fw = torch.tensor([p["fast_weights"] for p in params_list],
                          dtype=torch.float32, device=device)
    return F, c1, h, hx, eps, fw


def _generate_batch_true(dynamics, cfg, seeds, params_list, max_window_retries,
                         device):
    """Batch-generate true trajectories + forcing for the given windows.

    Returns (traj[B,T,D], forcing[B,T]) float32, all finite. Non-finite windows
    are regenerated individually with fresh seeds/params up to `max_window_retries`.
    """
    B = len(params_list)
    cpu = torch.device("cpu")
    traj_batch = torch.empty(B, cfg.num_steps, dynamics.state_dim,
                             dtype=torch.float32, device=cpu)
    forcing_batch = torch.empty(B, cfg.num_steps, dtype=torch.float32, device=cpu)

    pending = list(range(B))
    for attempt in range(max_window_retries):
        if not pending:
            break
        seeds_i = np.asarray([seeds[i] + attempt for i in pending], dtype=np.int64)
        params_i = [params_list[i] for i in pending]
        F, c1, h, hx, eps, fw = _param_tensors(params_i, device)
        try:
            tg, fo = dynamics.generate_batch_trajectories(
                len(pending), num_steps=cfg.num_steps,
                spinup_steps=cfg.spinup_steps,
                F_values=F, c1_values=c1, h_values=h, hx_values=hx,
                eps_values=eps, fast_weights_values=fw,
                seeds=seeds_i, device=device,
            )
        except RuntimeError:
            tg = None
            fo = None
        if tg is not None:
            if tg.device.type != "cpu":
                tg = tg.cpu()
                fo = fo.cpu()
            finite = torch.isfinite(tg).all(dim=(1, 2))
            still_bad = []
            for j, orig_i in enumerate(pending):
                if finite[j].item():
                    traj_batch[orig_i] = tg[j]
                    forcing_batch[orig_i] = fo[j]
                else:
                    still_bad.append(orig_i)
            pending = still_bad
    if pending:
        raise RuntimeError(
            f"{type(dynamics).__name__} batch: {len(pending)} unstable "
            f"windows after {max_window_retries} retries")

    return traj_batch, forcing_batch


def _build_window(cfg, dynamics, device, params, traj_seed, obs_seed, param_bias,
                  params_da=None):
    """Assemble a single window dict (trajectory split out by the caller).

    All stored tensors are on CPU (matching the legacy behavior); `device` is
    only used as the computation device for observations/forcing.
    """
    cpu = torch.device("cpu")
    true_fluid = params.pop("_traj")
    W_L_true = params.pop("_forcing")
    if cfg.use_corrupted_forcing:
        W_L_star = _make_corrupted_forcing(cfg, W_L_true, true_fluid, traj_seed, cpu)
    else:
        W_L_star = W_L_true.clone()
    noisy_obs, obs_mask = _make_obs(cfg, true_fluid, obs_seed, cpu)
    w = {
        "true_state": true_fluid, "obs": noisy_obs, "obs_mask": obs_mask,
        "forcing_true": W_L_true, "forcing_corrupted": W_L_star,
        "obs_seed": obs_seed,
    }
    if param_bias is not None:
        w["param_bias"] = param_bias
    for k, v in params.items():
        w[k] = v
        w[f"true_{k}"] = v
    _set_window_params(w, params)
    _set_window_params(w, params, "true_")
    if params_da is not None:
        for k, v in params_da.items():
            w[f"{k}_da"] = v
        _set_window_params(w, params_da, suffix="_da")
    return w


def _build_randparam_windows(cfg, dynamics, device, param_noise,
                             randomize_params, max_window_retries, bias):
    B = cfg.num_windows
    traj_seeds = [cfg.seed + i * 100 for i in range(B)]
    obs_seeds = [cfg.seed + i * 100 + 1 for i in range(B)]
    params_list = []
    for i in range(B):
        rng_np = np.random.RandomState(traj_seeds[i])
        params_list.append(_draw_l96_params(rng_np, cfg, param_noise=param_noise,
                                            randomize_params=randomize_params))
    traj_batch, forcing_batch = _generate_batch_true(
        dynamics, cfg, traj_seeds, params_list, max_window_retries, device)
    windows = []
    for i in range(B):
        p = dict(params_list[i])
        p["_traj"] = traj_batch[i]
        p["_forcing"] = forcing_batch[i]
        windows.append(_build_window(
            cfg, dynamics, device, p, traj_seeds[i], obs_seeds[i],
            param_bias=bias))
    return windows


def _build_randbias_windows(cfg, dynamics, device, param_noise,
                            randomize_params, bias_mode, bias_range,
                            max_window_retries):
    B = cfg.num_windows
    traj_seeds = [cfg.seed + i * 100 for i in range(B)]
    obs_seeds = [cfg.seed + i * 100 + 1 for i in range(B)]
    params_list = []
    params_da_list = []
    biases = []
    use_perparam = _uses_perparam_randomize(cfg)
    for i in range(B):
        rng_np = np.random.RandomState(traj_seeds[i])
        b = (rng_np.uniform(bias_range[0], bias_range[1])
             if bias_mode == "random" else cfg.param_bias)
        biases.append(b)
        params_true = _draw_l96_params(rng_np, cfg, param_noise=param_noise,
                                       randomize_params=randomize_params)
        params_da = {}
        for k, v in params_true.items():
            if use_perparam:
                spec = cfg.randomize.get(k) or {"biased": False}
                if spec.get("biased"):
                    bias_val = spec.get("bias", b)
                    params_da[k] = ([x * (1.0 + bias_val) for x in v]
                                    if k == "fast_weights" else v * (1.0 + bias_val))
                else:
                    params_da[k] = v
            elif randomize_params is not None and k not in randomize_params:
                params_da[k] = v
            elif k == "fast_weights":
                params_da[k] = v
            else:
                params_da[k] = v * (1.0 + b)
        params_list.append(params_true)
        params_da_list.append(params_da)

    traj_batch, forcing_batch = _generate_batch_true(
        dynamics, cfg, traj_seeds, params_list, max_window_retries, device)
    windows = []
    for i in range(B):
        p = dict(params_list[i])
        p["_traj"] = traj_batch[i]
        p["_forcing"] = forcing_batch[i]
        windows.append(_build_window(
            cfg, dynamics, device, p, traj_seeds[i], obs_seeds[i], biases[i],
            params_da=params_da_list[i]))
    return windows


class RandomParamLorenz96Dataset:
    def __init__(self, cfg: Lorenz96Config, param_noise: float = 0.2,
                 dynamics=None, cached_windows: list = None,
                 max_window_retries: int = 10,
                 randomize_params: list = None, device=None):
        self.cfg = cfg
        self.param_noise = param_noise
        self.device = torch.device("cpu" if device is None else device)
        self.dynamics = dynamics or _make_lorenz96_dynamics(cfg)
        self.randomize_params = randomize_params

        if cached_windows is not None:
            self.windows = cached_windows
            return

        self.windows = _build_randparam_windows(
            cfg, self.dynamics, self.device, param_noise,
            self.randomize_params, max_window_retries,
            bias=None,
        )

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
                 max_window_retries: int = 10,
                 bias_mode: str = "fixed", bias_range=(0.0, 0.15),
                 randomize_params: list = None, device=None):
        self.cfg = cfg
        self.param_noise = param_noise
        self.device = torch.device("cpu" if device is None else device)
        self.dynamics = dynamics or _make_lorenz96_dynamics(cfg)
        self.bias_mode = bias_mode
        self.bias_range = bias_range
        self.randomize_params = randomize_params

        if cached_windows is not None:
            self.windows = cached_windows
            return

        self.windows = _build_randbias_windows(
            cfg, self.dynamics, self.device, param_noise,
            self.randomize_params, bias_mode, bias_range, max_window_retries,
        )

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
                             num_test_windows: int = 200,
                             randomize_params: list = None,
                             device=None) -> Dict:
    dynamics = _make_lorenz96_dynamics(cfg)
    test_s0_cfg = Lorenz96Config(**{**cfg.__dict__, "case": 1, "param_bias": 0.0,
        "forcing_state_bias": 0.0, "seed": 123, "num_windows": num_test_windows})
    test_s1_cfg = Lorenz96Config(**{**cfg.__dict__, "case": 1, "param_bias": 0.15,
        "forcing_state_bias": 0.1, "seed": 131, "num_windows": num_test_windows})
    return {
        "test_s0": RandomParamLorenz96Dataset(test_s0_cfg, param_noise=0.2, dynamics=dynamics,
                                               randomize_params=randomize_params, device=device),
        "test_s1": RandomBiasLorenz96Dataset(test_s1_cfg, param_noise=0.2, dynamics=dynamics,
                                              randomize_params=randomize_params, device=device),
    }


def make_l96_s0_s1_trainval(cfg: Lorenz96Config, *,
                             num_train_windows: int = 1000,
                             num_val_windows: int = 100,
                             num_test_windows: int = 200,
                             param_noise: float = 0.2,
                             bias_range=(0.0, 0.2),
                             cached_datasets: dict = None,
                             randomize_params: list = None,
                             device=None) -> Dict:
    dynamics = _make_lorenz96_dynamics(cfg)

    def _build(key, cls, cfg_kwargs, **cls_kwargs):
        sub_cfg = Lorenz96Config(**{**cfg.__dict__, **cfg_kwargs})
        if cached_datasets is not None and key in cached_datasets:
            return cls(sub_cfg, cached_windows=cached_datasets[key], dynamics=dynamics, **cls_kwargs)
        return cls(sub_cfg, dynamics=dynamics, device=device, **cls_kwargs)

    train = _build("train", RandomBiasLorenz96Dataset,
                   {"seed": 42, "num_windows": num_train_windows, "case": 1,
                    "param_bias": 0.0, "forcing_state_bias": 0.1},
                   param_noise=param_noise, bias_mode="random", bias_range=bias_range,
                   randomize_params=randomize_params)
    val = _build("val", RandomBiasLorenz96Dataset,


                 {"seed": 99, "num_windows": num_val_windows, "case": 1,
                  "param_bias": 0.0, "forcing_state_bias": 0.1},
                 param_noise=param_noise, bias_mode="random", bias_range=bias_range,
                 randomize_params=randomize_params)
    test_s0 = _build("test_s0", RandomParamLorenz96Dataset,
                     {"seed": 123, "num_windows": num_test_windows, "case": 1,
                      "param_bias": 0.0, "forcing_state_bias": 0.0},
                     param_noise=param_noise,
                     randomize_params=randomize_params)
    test_s1 = _build("test_s1", RandomBiasLorenz96Dataset,
                     {"seed": 131, "num_windows": num_test_windows, "case": 1,
                      "param_bias": 0.1, "forcing_state_bias": 0.1},
                     param_noise=param_noise, bias_mode="fixed",
                     randomize_params=randomize_params)
    return {
        "train": train,
        "val": val,
        "test_s0": test_s0,
        "test_s1": test_s1,
    }