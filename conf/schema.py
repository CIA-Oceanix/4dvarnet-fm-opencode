from dataclasses import dataclass, field
from typing import List, Tuple, Any, Dict


@dataclass
class ParamRandomization:
    randomized: bool = True
    noise: float = 0.2
    biased: bool = False
    bias: float = 0.0


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

    # Lorenz96 (two-scale) fields
    NO: int = 8
    J: int = 4
    h: float = 1.0
    hx: float = 1.0
    eps: float = 0.1
    F_true: float = 8.0
    F_da: float = 8.0
    coupling_exponent_truth: float = 1.6
    coupling_exponent_da: float = 1.0
    fast_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 0.1, 0.1])
    obs_j: int = 2
    randomize: Dict[str, ParamRandomization] = field(default_factory=dict)

    # Device
    device: str = "cpu"

    @property
    def num_steps(self) -> int:
        return int(self.T_max / self.dt)

    @property
    def use_corrupted_forcing(self) -> bool:
        return self.case == 2

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

    def to_lorenz96_config(self) -> Any:
        """Convert to data.lorenz96.Lorenz96Config."""
        from data.lorenz96 import Lorenz96Config as L96C
        return L96C(
            case=self.case, dt=self.dt, T_max=self.T_max,
            obs_interval=self.obs_interval, R_var=self.R_var,
            B_var=self.B_var, param_bias=self.param_bias,
            num_windows=self.num_windows, window_spacing=self.window_spacing,
            spinup_steps=self.spinup_steps, seed=self.seed,
            NO=self.NO, J=self.J, h=self.h, hx=self.hx, eps=self.eps,
            F_true=self.F_true, F_da=self.F_da,
            gamma=self.gamma, W_L_bar=self.W_L_bar, c1=self.c1, c2=self.c2,
            sigma_0=self.sigma_0, sigma_L=self.sigma_L,
            tau_eta=self.tau_eta, sigma_eta=self.sigma_eta,
            forcing_state_bias=self.forcing_state_bias,
            forcing_coupling=self.forcing_coupling,
            coupling_exponent_truth=self.coupling_exponent_truth,
            coupling_exponent_da=self.coupling_exponent_da,
            fast_weights=list(self.fast_weights),
            randomize={k: {"randomized": v.randomized, "noise": v.noise,
                           "biased": v.biased, "bias": v.bias}
                       for k, v in self.randomize.items()},
            obs_var_indices=self._compute_obs_var_indices(),
        )

    def _compute_obs_var_indices(self):
        obs_j = self.obs_j
        if obs_j is None or obs_j >= self.J:
            return None
        NO = self.NO
        J = self.J
        X_idx = list(range(NO))
        Y_idx = []
        for k in range(NO):
            for j in range(obs_j):
                Y_idx.append(NO + k * J + j)
        return tuple(X_idx + Y_idx)

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


@dataclass
class DirectUNetConfig:
    hidden_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    dropout: float = 0.1
    cond_extra_dim: int = 0


@dataclass
class VanillaCFMConfig:
    hidden_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    time_emb_dim: int = 64
    N_outer: int = 10
    sigma_prior: float = 0.5
    dropout: float = 0.1
    train_tau_0_only: bool = False
    cond_extra_dim: int = 0


@dataclass
class JointCFMConfig:
    param_dim: int = 4
    param_loss_weight: float = 0.1
    param_noise_min: float = 0.0
    param_noise_max: float = 0.3
    train_tau_0_only: bool = False


@dataclass
class JointDirectUNetConfig:
    param_dim: int = 4
    param_loss_weight: float = 0.1


@dataclass
class PredictStateCFMConfig:
    hidden_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    time_emb_dim: int = 64
    N_outer: int = 10
    sigma_prior: float = 0.5
    dropout: float = 0.1
    cond_extra_dim: int = 0


@dataclass
class TweedieCFMConfig:
    hidden_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    time_emb_dim: int = 64
    K_inner: int = 5
    N_outer: int = 10
    sigma_prior: float = 0.5
    dropout: float = 0.1
    train_tau_0_only: bool = False
    cond_extra_dim: int = 0


@dataclass
class ModelConfig:
    model_type: str = "tweedie"  # "tweedie" | "direct_unet" | "vanilla_cfm" | "joint_cfm" | "predict_state_cfm" | "tweedie_cfm"
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
    joint_direct_unet: JointDirectUNetConfig = field(default_factory=JointDirectUNetConfig)
    predict_state_cfm: PredictStateCFMConfig = field(default_factory=PredictStateCFMConfig)
    tweedie_cfm: TweedieCFMConfig = field(default_factory=TweedieCFMConfig)


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
