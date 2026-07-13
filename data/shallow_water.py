"""Two-layer rotating shallow water dataset for data assimilation.

State layout: [h1(Nxy), u1(Nxy), v1(Nxy), h2(Nxy), u2(Nxy), v2(Nxy)]
Layer 1 = ocean (slow), Layer 2 = atmosphere (fast).
"""

import torch
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class ShallowWaterConfig:
    """Configuration for the rotating shallow water system."""

    # Physical parameters
    Nx: int = 64
    Ny: int = 64
    dt: float = 0.01
    K: int = 5           # steps per DA window
    tau0: float = 0.08   # wind stress amplitude
    f_cor: float = 0.1   # Coriolis parameter
    g1: float = 0.02     # layer 1 reduced gravity
    g2: float = 0.01     # layer 2 reduced gravity
    coupling: float = 0.05
    friction: float = 0.1
    viscosity: float = 0.001

    # Observation parameters
    obs_noise_std: float = 0.1
    obs_stride_ocean: int = 8     # ocean obs stride (sparser)
    obs_stride_atmos: int = 4     # atmosphere obs stride (denser)

    # Dataset parameters
    spinup_steps: int = 1000
    num_windows: int = 200
    window_steps: int = 500
    seed: int = 42

    # Land mask
    land_mask_type: str = "none"

    @property
    def state_dim(self) -> int:
        return 6 * self.Nx * self.Ny  # [h1,u1,v1,h2,u2,v2]

    @property
    def Nxy(self) -> int:
        return self.Nx * self.Ny


def make_sw_obs_indices(config: ShallowWaterConfig) -> torch.Tensor:
    """Create sparse observation indices for ocean and atmosphere.

    Ocean (layer 1): stride ``obs_stride_ocean`` in both x and y.
    Atmosphere (layer 2): stride ``obs_stride_atmos`` in both x and y.

    Returns
    -------
    obs_indices : Tensor ``(n_obs,)`` of long indices into the flat state vector.
    """
    Nx, Ny = config.Nx, config.Ny
    Nxy = Nx * Ny

    # Spatial grid-point indices observed in each layer
    obs_xy_ocean = []
    for i in range(0, Nx, config.obs_stride_ocean):
        for j in range(0, Ny, config.obs_stride_ocean):
            obs_xy_ocean.append(i * Ny + j)
    obs_xy_ocean = torch.tensor(obs_xy_ocean, dtype=torch.long)

    obs_xy_atmos = []
    for i in range(0, Nx, config.obs_stride_atmos):
        for j in range(0, Ny, config.obs_stride_atmos):
            obs_xy_atmos.append(i * Ny + j)
    obs_xy_atmos = torch.tensor(obs_xy_atmos, dtype=torch.long)

    # Offset into the flat state vector for each variable
    # Layer 1: h1=[0, Nxy), u1=[Nxy, 2*Nxy), v1=[2*Nxy, 3*Nxy)
    # Layer 2: h2=[3*Nxy, 4*Nxy), u2=[4*Nxy, 5*Nxy), v2=[5*Nxy, 6*Nxy)
    idx_ocean = torch.cat([
        obs_xy_ocean,                  # h1
        obs_xy_ocean + Nxy,            # u1
        obs_xy_ocean + 2 * Nxy,        # v1
    ])

    idx_atmos = torch.cat([
        obs_xy_atmos + 3 * Nxy,        # h2
        obs_xy_atmos + 4 * Nxy,        # u2
        obs_xy_atmos + 5 * Nxy,        # v2
    ])

    return torch.cat([idx_ocean, idx_atmos])


def make_sw_obs_mask(config: ShallowWaterConfig) -> torch.Tensor:
    """Return boolean mask of shape ``(state_dim,)`` where True = observed."""
    indices = make_sw_obs_indices(config)
    mask = torch.zeros(config.state_dim, dtype=torch.bool)
    mask[indices] = True
    return mask


def _generate_sw_observations(
    true_state: torch.Tensor,      # (K, state_dim)
    obs_indices: torch.Tensor,     # (n_obs,)
    obs_noise_std: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate noisy spatial observations for one DA window.

    Parameters
    ----------
    true_state : Tensor ``(K, state_dim)``
        Ground-truth trajectory for one window.
    obs_indices : Tensor ``(n_obs,)``
        Indices of observed state components.
    obs_noise_std : float
        Standard deviation of Gaussian observation noise.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    obs : Tensor ``(K, state_dim)`` -- NaN everywhere except observed entries.
    obs_mask : Tensor ``(state_dim,)`` -- Boolean mask (True = observed).
    """
    K, state_dim = true_state.shape
    n_obs = obs_indices.numel()
    rng = torch.Generator().manual_seed(seed)

    obs = torch.full_like(true_state, float("nan"))
    noise = torch.randn(K, n_obs, generator=rng) * obs_noise_std
    obs[:, obs_indices] = true_state[:, obs_indices] + noise

    full_mask = torch.zeros(state_dim, dtype=torch.bool)
    full_mask[obs_indices] = True

    return obs, full_mask


class ShallowWaterDataset:
    """Dataset for rotating shallow water system.

    Each sample is a DA window containing:

    - ``true_state`` : ``(K, state_dim)`` -- ground-truth trajectory
    - ``obs``        : ``(K, state_dim)`` -- noisy observations (NaN where unobserved)
    - ``obs_mask``   : ``(state_dim,)``   -- boolean observation mask
    - ``forcing``    : ``(K, 2)``         -- temporal wind-stress perturbation

    Parameters
    ----------
    config : ShallowWaterConfig
    scenario : str
        ``"S0"`` for the reference scenario, ``"S1"`` for a perturbed scenario
        (uses modified tau0).
    """

    def __init__(self, config: ShallowWaterConfig, scenario: str = "S0"):
        from models.shallow_water_dynamics import ShallowWaterDynamics

        self.config = config
        self.scenario = scenario
        self.obs_indices = make_sw_obs_indices(config)
        self.obs_mask = make_sw_obs_mask(config)

        # Build dynamics -- S1 uses a perturbed tau0
        tau0 = config.tau0
        if scenario == "S1":
            tau0 = config.tau0 * 1.15  # 15% perturbation

        self.dynamics = ShallowWaterDynamics(
            Nx=config.Nx, Ny=config.Ny, dt=config.dt,
            K=config.K, tau0=tau0, f_cor=config.f_cor,
            g1=config.g1, g2=config.g2, coupling=config.coupling,
            friction=config.friction, viscosity=config.viscosity,
            land_mask_type=config.land_mask_type,
        )

        self.windows = self._generate()

    def _generate(self) -> list:
        cfg = self.config
        total_steps = cfg.spinup_steps + cfg.K * cfg.num_windows + 100

        traj, forcing = self.dynamics.generate_full_trajectory(
            num_steps=total_steps,
            seed=cfg.seed,
            spinup_steps=0,  # we manage spinup here
        )

        # Discard the first spinup_steps
        traj = traj[cfg.spinup_steps:]
        forcing = forcing[cfg.spinup_steps:]

        windows = []
        for w in range(cfg.num_windows):
            start = w * cfg.K
            end = start + cfg.K
            seg = traj[start:end].clone()     # (K, state_dim)
            frc = forcing[start:end].clone()  # (K, 2)

            obs, obs_mask = _generate_sw_observations(
                seg, self.obs_indices, cfg.obs_noise_std,
                seed=cfg.seed + w + 1,
            )

            windows.append({
                "true_state": seg,
                "obs": obs,
                "obs_mask": obs_mask,
                "forcing": frc,
            })

        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.windows[idx]


def make_sw_s0_s1_datasets(
    config: ShallowWaterConfig, *,
    num_test_windows: int = 200,
) -> Dict[str, ShallowWaterDataset]:
    """Create S0 (reference) and S1 (perturbed) test datasets.

    Returns
    -------
    Dict with keys ``"test_s0"`` and ``"test_s1"``.
    """
    s0_cfg = ShallowWaterConfig(
        **{**config.__dict__, "num_windows": num_test_windows, "seed": 123},
    )
    s1_cfg = ShallowWaterConfig(
        **{**config.__dict__, "num_windows": num_test_windows, "seed": 131},
    )
    return {
        "test_s0": ShallowWaterDataset(s0_cfg, scenario="S0"),
        "test_s1": ShallowWaterDataset(s1_cfg, scenario="S1"),
    }


def sw_collate_fn(batch):
    """Default collate function for ShallowWaterDataset DataLoader.

    Each item is a dict; this stacks tensors along a new batch dimension.
    ``obs_mask`` is identical across samples so we take the first.
    """
    return {
        "true_state": torch.stack([b["true_state"] for b in batch]),
        "obs":        torch.stack([b["obs"] for b in batch]),
        "obs_mask":   batch[0]["obs_mask"],          # same for all
        "forcing":    torch.stack([b["forcing"] for b in batch]),
    }
