from dataclasses import dataclass, field
from typing import List, Tuple, Any
from omegaconf import MISSING


@dataclass
class DataConfig:
    system: str = "lorenz63"
    dt: float = 0.01
    T_max: float = 3.0
    obs_interval: int = 20
    R_var: float = 0.5
    B_var: float = 2.0
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
    sigma_eta: float = 0.7071067811865476
    forcing_state_bias: float = 0.0
    forcing_coupling: str = "linear"
    param_bias: float = 0.0
    case: int = 1
    train_mix: str = "cs1+cs2"
    randomize_params: bool = False
    param_noise: float = 0.2
    test_randparam: bool = True
    test_param_noise: float = 0.2

    # Shallow Water (rotating SW)
    Nx: int = 64
    Ny: int = 64
    tau0: float = 0.0
    f_cor: float = 0.1
    g1: float = 1.0
    g2: float = 4.0
    coupling: float = 0.01
    friction: float = 0.0
    viscosity: float = 0.0001
    obs_stride_ocean: int = 16
    obs_stride_atmos: int = 8
    land_mask_type: str = "none"
    K: int = 5
    window_steps: int = 500
    obs_noise_std: float = 0.1
    spinup_steps: int = 10000
    num_windows: int = 200
    bickley_U: float = 0.5
    bickley_U2: float = 0.3
    bickley_H_ref: float = 10.0
    bickley_perturbation_mode: str = "random_balanced"
    bickley_epsilon: float = 0.01

    # --- MAOOAM parameters ---
    # Atmosphere truncation
    maooam_atm_nx: int = 4
    maooam_atm_ny: int = 4
    # Ocean truncation
    maooam_occ_nx: int = 4
    maooam_occ_ny: int = 4
    # Atmosphere physics
    maooam_kd: float = 0.0290
    maooam_kdp: float = 0.0290
    maooam_sigma: float = 0.2
    # Ocean physics
    maooam_r: float = 1e-7
    maooam_h: float = 136.5
    maooam_d: float = 1.1e-7
    # Temperature
    maooam_eps: float = 0.7
    maooam_T0_atm: float = 289.3
    maooam_hlambda: float = 15.06
    maooam_gamma_oc: float = 5.6e8
    maooam_T0_oc: float = 301.46
    # Insolation
    maooam_C_atm: float = 103.33
    maooam_C_oc: float = 310.0
    # Domain
    maooam_scale: float = 5e6
    maooam_f0: float = 1.032e-4
    maooam_n_ratio: float = 1.5
    # Radiation
    maooam_T4: bool = False
    maooam_dynamic_T: bool = False
    # Observation
    maooam_obs_mode: str = "all"
    maooam_obs_atm_frac: float = 1.0
    maooam_obs_oc_frac: float = 1.0
    # Stochastic forcing
    maooam_stochastic_forcing: bool = False
    maooam_forcing_amplitude: float = 0.01

    @property
    def num_steps(self) -> int:
        return int(self.T_max / self.dt)

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

    def to_lorenz63_config(self) -> Any:
        """Convert to data.lorenz63.Lorenz63Config."""
        from data.lorenz63 import Lorenz63Config as L63C
        return L63C(
            case=self.case, dt=self.dt, T_max=self.T_max,
            obs_interval=self.obs_interval, R_var=self.R_var,
            B_var=self.B_var, param_bias=self.param_bias,
            num_windows=self.num_windows, window_spacing=self.window_spacing,
            spinup_steps=self.spinup_steps, seed=self.seed,
            sigma_true=self.sigma_true, rho_true=self.rho_true,
            beta_true=self.beta_true, gamma=self.gamma,
            W_L_bar=self.W_L_bar, c1=self.c1, c2=self.c2,
            sigma_0=self.sigma_0, sigma_L=self.sigma_L,
            tau_eta=self.tau_eta, sigma_eta=self.sigma_eta,
            forcing_state_bias=self.forcing_state_bias,
            forcing_coupling=self.forcing_coupling,
        )


    def to_shallow_water_config(self):
        """Convert DataConfig to ShallowWaterConfig."""
        from data.shallow_water import ShallowWaterConfig
        return ShallowWaterConfig(
            Nx=self.Nx, Ny=self.Ny, dt=self.dt, K=self.K,
            tau0=self.tau0, f_cor=self.f_cor, g1=self.g1, g2=self.g2,
            coupling=self.coupling, friction=self.friction,
            viscosity=self.viscosity,
            obs_noise_std=self.obs_noise_std,
            obs_stride_ocean=self.obs_stride_ocean,
            obs_stride_atmos=self.obs_stride_atmos,
            spinup_steps=self.spinup_steps,
            num_windows=self.num_windows,
            window_steps=self.window_steps,
            seed=self.seed,
            land_mask_type=self.land_mask_type,
            bickley_U=self.bickley_U,
            bickley_U2=self.bickley_U2,
            bickley_H_ref=self.bickley_H_ref,
            bickley_perturbation_mode=self.bickley_perturbation_mode,
            bickley_epsilon=self.bickley_epsilon,
        )

    def to_maooam_config(self):
        """Convert DataConfig to MaooamConfig."""
        from data.maooam import MaooamConfig
        return MaooamConfig(
            dt=self.dt, K=self.K,
            atm_nx=self.maooam_atm_nx, atm_ny=self.maooam_atm_ny,
            occ_nx=self.maooam_occ_nx, occ_ny=self.maooam_occ_ny,
            kd=self.maooam_kd, kdp=self.maooam_kdp, sigma=self.maooam_sigma,
            r=self.maooam_r, h=self.maooam_h, d=self.maooam_d,
            eps=self.maooam_eps, T0_atm=self.maooam_T0_atm,
            hlambda=self.maooam_hlambda,
            gamma_oc=self.maooam_gamma_oc, T0_oc=self.maooam_T0_oc,
            C_atm=self.maooam_C_atm, C_oc=self.maooam_C_oc,
            scale=self.maooam_scale, f0=self.maooam_f0,
            n_ratio=self.maooam_n_ratio,
            T4=self.maooam_T4, dynamic_T=self.maooam_dynamic_T,
            obs_mode=self.maooam_obs_mode,
            obs_atm_frac=self.maooam_obs_atm_frac,
            obs_oc_frac=self.maooam_obs_oc_frac,
            obs_noise_std=self.obs_noise_std,
            spinup_steps=self.spinup_steps,
            num_windows=self.num_windows,
            window_steps=self.window_steps,
            seed=self.seed,
            stochastic_forcing=self.maooam_stochastic_forcing,
            forcing_amplitude=self.maooam_forcing_amplitude,
        )


@dataclass
class DirectUNetConfig:
    hidden_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    dropout: float = 0.1


@dataclass
class VanillaCFMConfig:
    hidden_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    time_emb_dim: int = 64
    N_outer: int = 10
    sigma_prior: float = 0.5
    dropout: float = 0.1


@dataclass
class JointCFMConfig:
    param_dim: int = 4
    param_loss_weight: float = 0.1
    param_noise_min: float = 0.0
    param_noise_max: float = 0.3
    train_tau_0_only: bool = False


@dataclass
class ModelConfig:
    model_type: str = "tweedie"  # "tweedie" | "direct_unet" | "vanilla_cfm" | "joint_cfm"
    state_dim: int = 3
    hidden_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    time_emb_dim: int = 64
    K_inner: int = 5
    N_outer: int = 10
    nu: float = 1.0
    use_obs: bool = True
    use_energy: bool = True
    dropout: float = 0.1
    direct_unet: DirectUNetConfig = field(default_factory=DirectUNetConfig)
    vanilla_cfm: VanillaCFMConfig = field(default_factory=VanillaCFMConfig)
    joint_cfm: JointCFMConfig = field(default_factory=JointCFMConfig)


@dataclass
class StageConfig:
    epochs: int = 200
    lr: float = 1e-3
    gradient_clip_val: float = 10.0


@dataclass
class LossConfig:
    use_gradient: bool = True
    gradient_weight: float = 0.1


@dataclass
class TrainingConfig:
    stage1: StageConfig = field(default_factory=lambda: StageConfig(epochs=200, lr=1e-3, gradient_clip_val=10.0))
    stage2: StageConfig = field(default_factory=lambda: StageConfig(epochs=400, lr=1e-3, gradient_clip_val=1.0))
    batch_size: int = 32
    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class PathsConfig:
    checkpoint_dir: str = "checkpoints"
    checkpoint_stage1: str = "checkpoints/stage1.pt"
    checkpoint_stage2: str = "checkpoints/stage2.pt"
    outputs_dir: str = "outputs"


@dataclass
class Weak4DVarConfig:
    opt_steps: int = 150
    lr: float = 0.02


@dataclass
class Strong4DVarConfig:
    max_iter: int = 40
    lr: float = 0.1


@dataclass
class EnKFConfig:
    inflation: float = 1.0
    loc_radius: float = -1.0


@dataclass
class ETKFConfig:
    inflation: float = 1.0
    loc_radius: float = -1.0
    loc_mode: str = "square_root"


@dataclass
class BaselinesConfig:
    da_window_steps: int = 300
    N_ensemble: int = 30
    batch_size: int = 128
    weak4dvar: Weak4DVarConfig = field(default_factory=Weak4DVarConfig)
    strong4dvar: Strong4DVarConfig = field(default_factory=Strong4DVarConfig)
    enkf: EnKFConfig = field(default_factory=EnKFConfig)
    etkf: ETKFConfig = field(default_factory=ETKFConfig)


@dataclass
class CaseStudyConfig:
    param_bias: float = 0.0
    forcing_state_bias: float = 0.0
    forcing_coupling: str = "linear"


@dataclass
class CS1Config:
    param_bias: float = 0.0
    forcing_coupling: str = "linear"


@dataclass
class CS2Config:
    param_bias: float = 0.15
    forcing_state_bias: float = 0.15
    forcing_coupling: str = "quartic"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)
    cs1: CS1Config = field(default_factory=CS1Config)
    cs2: CS2Config = field(default_factory=CS2Config)
