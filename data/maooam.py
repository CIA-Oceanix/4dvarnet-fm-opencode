"""MAOOAM dataset: coupled QG atmosphere + SW ocean via qgs.

State vector: spectral coefficients [psi_a, theta_a, psi_o, delta_T_o].
Physical fields are reconstructed via basis-function expansion in the diagnostics.

References
----------
Demaeyer, De Cruz & Vannitsem (2020). qgs: A flexible Python framework of
    reduced-order multiscale climate models. JOSS, 5(56), 2597.
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class MaooamConfig:
    """Configuration for the MAOOAM coupled model."""

    # --- Time stepping ---
    dt: float = 0.1
    K: int = 5  # steps per DA window

    # --- Atmosphere truncation ---
    atm_nx: int = 4
    atm_ny: int = 4

    # --- Ocean truncation ---
    occ_nx: int = 4
    occ_ny: int = 4

    # --- Atmosphere parameters ---
    kd: float = 0.0290     # bottom friction
    kdp: float = 0.0290    # internal friction
    sigma: float = 0.2     # static stability

    # --- Ocean parameters ---
    r: float = 1e-7        # bottom friction
    h: float = 136.5       # layer depth (m)
    d: float = 1.1e-7      # ocean-atmosphere coupling

    # --- Temperature parameters ---
    eps: float = 0.7       # emissivity
    T0_atm: float = 289.3  # atm ref temperature (K)
    hlambda: float = 15.06 # heat exchange (W/m2/K)
    gamma_oc: float = 5.6e8  # ocean heat capacity (J/m2/K)
    T0_oc: float = 301.46  # ocean ref temperature (K)

    # --- Insolation ---
    C_atm: float = 103.33
    C_oc: float = 310.0

    # --- Domain ---
    scale: float = 5e6     # meridional scale (m)
    f0: float = 1.032e-4   # Coriolis (s-1)
    n_ratio: float = 1.5   # aspect ratio 2*Ly/Lx

    # --- Radiation options ---
    T4: bool = False       # use nonlinear T^4 radiation
    dynamic_T: bool = False  # evolve 0th-order T mode

    # --- Observation parameters ---
    obs_noise_std: float = 0.1
    obs_mode: str = "all"  # "all" = observe all modes, "partial" = subset
    obs_atm_frac: float = 1.0
    obs_oc_frac: float = 1.0

    # --- Dataset parameters ---
    spinup_steps: int = 5000
    num_windows: int = 200
    window_steps: int = 500
    seed: int = 42

    # --- Stochastic forcing (model error for DA) ---
    stochastic_forcing: bool = False
    forcing_amplitude: float = 0.01

    # --- Device ---
    device: str = "cpu"
    compile: bool = True

    @property
    def state_dim(self) -> int:
        from models.maooam_dynamics import _count_atm_modes, _count_oc_modes
        Natm = _count_atm_modes(self.atm_nx, self.atm_ny)
        Noc = _count_oc_modes(self.occ_nx, self.occ_ny)
        return 2 * Natm + 2 * Noc  # psi_a + theta_a + psi_o + dT_o


def make_maooam_obs_indices(config: MaooamConfig) -> torch.Tensor:
    """Create observation indices for the spectral state vector.

    ``obs_mode="all"``: observe every spectral coefficient.
    ``obs_mode="partial"``: observe a fraction of modes per variable.

    Returns
    -------
    obs_indices : Tensor ``(n_obs,)`` of long indices into the flat state vector.
    """
    from models.maooam_dynamics import _count_atm_modes, _count_oc_modes
    Natm = _count_atm_modes(config.atm_nx, config.atm_ny)
    Noc = _count_oc_modes(config.occ_nx, config.occ_ny)
    state_dim = 2 * Natm + 2 * Noc

    if config.obs_mode == "all":
        return torch.arange(state_dim, dtype=torch.long)

    # Partial observation: select fraction of modes per variable block
    rng = torch.Generator().manual_seed(config.seed + 999)

    def _select(n: int, frac: float) -> torch.Tensor:
        k = max(1, int(n * frac))
        perm = torch.randperm(n, generator=rng)
        return perm[:k].sort().values

    idx_a = _select(Natm, config.obs_atm_frac)
    idx_a2 = _select(Natm, config.obs_atm_frac) + Natm
    idx_o = _select(Noc, config.obs_oc_frac) + 2 * Natm
    idx_o2 = _select(Noc, config.obs_oc_frac) + 2 * Natm + Noc

    return torch.cat([idx_a, idx_a2, idx_o, idx_o2])


def make_maooam_obs_mask(config: MaooamConfig) -> torch.Tensor:
    """Return boolean mask of shape ``(state_dim,)`` where True = observed."""
    indices = make_maooam_obs_indices(config)
    mask = torch.zeros(config.state_dim, dtype=torch.bool)
    mask[indices] = True
    return mask


def _generate_maooam_observations(
    true_state: torch.Tensor,      # (K, state_dim)
    obs_indices: torch.Tensor,     # (n_obs,)
    obs_noise_std: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate noisy spectral observations for one DA window."""
    K, state_dim = true_state.shape
    n_obs = obs_indices.numel()
    device = true_state.device
    rng = torch.Generator(device=device).manual_seed(seed)

    obs = torch.full_like(true_state, float("nan"))
    noise = torch.randn(K, n_obs, device=device, generator=rng) * obs_noise_std
    obs[:, obs_indices] = true_state[:, obs_indices] + noise

    full_mask = torch.zeros(state_dim, dtype=torch.bool)
    full_mask[obs_indices] = True

    return obs, full_mask


class MaooamDataset:
    """Dataset for the MAOOAM coupled ocean-atmosphere model.

    Each sample is a DA window containing:

    - ``true_state`` : ``(K, state_dim)`` -- ground-truth spectral trajectory
    - ``obs``        : ``(K, state_dim)`` -- noisy spectral observations (NaN where unobserved)
    - ``obs_mask``   : ``(state_dim,)``   -- boolean observation mask
    - ``forcing``    : ``(K, forcing_dim)`` -- temporal perturbation (zeros if deterministic)

    Parameters
    ----------
    config : MaooamConfig
    scenario : str
        ``"S0"`` for the reference scenario, ``"S1"`` for perturbed coupling.
    """

    def __init__(self, config: MaooamConfig, scenario: str = "S0", dynamics=None):
        self.config = config
        self.scenario = scenario
        self.obs_indices = make_maooam_obs_indices(config)
        self.obs_mask = make_maooam_obs_mask(config)

        # S1 scenario: perturb ocean-atmosphere coupling
        d = config.d
        if scenario == "S1":
            d = config.d * 1.5  # 50% stronger coupling

        # Allow passing a pre-built dynamics instance (avoids qgs tensor re-extraction)
        if dynamics is not None:
            self.dynamics = dynamics
        elif config.device != "cpu":
            from models.maooam_torch import MaooamTorchDynamics
            self.dynamics = MaooamTorchDynamics(
                device=config.device, compile=config.compile,
                dt=config.dt, K=config.K,
                atm_nx=config.atm_nx, atm_ny=config.atm_ny,
                occ_nx=config.occ_nx, occ_ny=config.occ_ny,
                kd=config.kd, kdp=config.kdp, sigma=config.sigma,
                r=config.r, h=config.h, d=d,
                eps=config.eps, T0_atm=config.T0_atm, hlambda=config.hlambda,
                gamma_oc=config.gamma_oc, T0_oc=config.T0_oc,
                C_atm=config.C_atm, C_oc=config.C_oc,
                scale=config.scale, f0=config.f0, n_ratio=config.n_ratio,
                T4=config.T4, dynamic_T=config.dynamic_T,
                stochastic_forcing=config.stochastic_forcing,
                forcing_amplitude=config.forcing_amplitude,
            )
        else:
            from models.maooam_dynamics import MaooamDynamics
            self.dynamics = MaooamDynamics(
                dt=config.dt, K=config.K,
                atm_nx=config.atm_nx, atm_ny=config.atm_ny,
                occ_nx=config.occ_nx, occ_ny=config.occ_ny,
                kd=config.kd, kdp=config.kdp, sigma=config.sigma,
                r=config.r, h=config.h, d=d,
                eps=config.eps, T0_atm=config.T0_atm, hlambda=config.hlambda,
                gamma_oc=config.gamma_oc, T0_oc=config.T0_oc,
                C_atm=config.C_atm, C_oc=config.C_oc,
                scale=config.scale, f0=config.f0, n_ratio=config.n_ratio,
                T4=config.T4, dynamic_T=config.dynamic_T,
                stochastic_forcing=config.stochastic_forcing,
                forcing_amplitude=config.forcing_amplitude,
            )

        self.windows = self._generate()

    def _generate(self) -> list:
        cfg = self.config
        total_steps = cfg.spinup_steps + cfg.K * cfg.num_windows + 100

        traj, forcing = self.dynamics.generate_full_trajectory(
            num_steps=total_steps,
            seed=cfg.seed,
            spinup_steps=0,  # we manage spinup below
        )

        # Discard spinup
        traj = traj[cfg.spinup_steps:]
        forcing = forcing[cfg.spinup_steps:]

        windows = []
        for w in range(cfg.num_windows):
            start = w * cfg.K
            end = start + cfg.K
            seg = traj[start:end].clone()
            frc = forcing[start:end].clone()

            obs, obs_mask = _generate_maooam_observations(
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


def make_maooam_s0_s1_datasets(
    config: MaooamConfig, *,
    num_test_windows: int = 200,
) -> Dict[str, MaooamDataset]:
    """Create S0 (reference) and S1 (perturbed) test datasets.

    Returns
    -------
    Dict with keys ``"test_s0"`` and ``"test_s1"``.
    """
    s0_cfg = MaooamConfig(
        **{**config.__dict__, "num_windows": num_test_windows, "seed": 123},
    )
    s1_cfg = MaooamConfig(
        **{**config.__dict__, "num_windows": num_test_windows, "seed": 131},
    )
    return {
        "test_s0": MaooamDataset(s0_cfg, scenario="S0"),
        "test_s1": MaooamDataset(s1_cfg, scenario="S1"),
    }


def maooam_collate_fn(batch):
    """Default collate function for MaooamDataset DataLoader."""
    return {
        "true_state": torch.stack([b["true_state"] for b in batch]),
        "obs":        torch.stack([b["obs"] for b in batch]),
        "obs_mask":   batch[0]["obs_mask"],
        "forcing":    torch.stack([b["forcing"] for b in batch]),
    }
