import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet import ConvBlock, SinusoidalEmbedding, UNet1D
from models.interpolant import LinearInterpolant


def _make_cond(obs, forcing, params, param_dim=0, cond_extra_dim=0):
    obs_clean = torch.nan_to_num(obs, nan=0.0)
    if cond_extra_dim > 0:
        B, T, D = obs.shape
        cond = torch.cat([obs_clean, forcing.unsqueeze(-1)], dim=-1)
        if param_dim > 0:
            params_t = params.unsqueeze(1).expand(B, T, -1)
            cond = torch.cat([cond, params_t], dim=-1)
    else:
        cond = obs_clean
    return cond


class ParamFlowCNN(nn.Module):
    """CNN flow field on the parameter manifold.

    Learns the param velocity v_phi(obs, forcing, x_hat_1, param_tau, tau) whose
    conditional flow maps param_0 ~ N(0, I) toward true_param as tau -> 1, in
    exact parallel to the state CFM. Inputs are stacked over the time axis and
    reduced to a single (B, param_dim) velocity vector by global average pooling
    (params are a single vector, not a time sequence). tau enters as a sinusoidal
    time embedding per conv block, matching how VanillaCFM conditions on tau.
    """

    def __init__(self, param_dim=4, state_dim=24, hidden_channels=None,
                 time_emb_dim=64, dropout=0.1):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
        self.param_dim = param_dim
        in_c = state_dim + 1 + state_dim + param_dim
        self.time_embed = SinusoidalEmbedding(time_emb_dim)
        self.blocks = nn.ModuleList()
        cin = in_c
        for hc in hidden_channels:
            self.blocks.append(ConvBlock(cin, hc, time_emb_dim, dropout))
            cin = hc
        self.head = nn.Sequential(
            nn.Conv1d(cin, param_dim, 1),
        )

    def forward(self, obs, forcing, x_hat_1, param_tau, tau):
        obs_clean = torch.nan_to_num(obs, nan=0.0)
        B, T, _ = obs_clean.shape
        t_emb = self.time_embed(tau)
        forcing_b = forcing.unsqueeze(-1).expand(B, T, 1)
        x_hat_clean = torch.nan_to_num(x_hat_1, nan=0.0)
        x = torch.cat([obs_clean, forcing_b, x_hat_clean, param_tau], dim=-1)
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x, t_emb)
        x = self.head(x)
        x = x.mean(dim=-1)
        return x


class VanillaCFM(nn.Module):
    def __init__(self, state_dim=3, hidden_channels=None, time_emb_dim=64, N_outer=10, sigma_prior=0.5, dropout=0.1, train_tau_0_only=False, param_dim=4, cond_extra_dim=0):
        super().__init__()
        self.cond_extra_dim = cond_extra_dim
        self.param_dim = param_dim
        self.unet = UNet1D(
            state_dim=state_dim,
            obs_dim=state_dim,
            cond_extra_dim=cond_extra_dim,
            hidden_channels=hidden_channels,
            use_obs=True,
            use_energy=False,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
        )
        self.interpolant = LinearInterpolant(nu=1.0)
        self.N_outer = N_outer
        self.sigma_prior = sigma_prior
        self.state_dim = state_dim
        self.train_tau_0_only = train_tau_0_only

    def forward(self, x_t, batch, tau):
        cond = _make_cond(batch.obs, batch.forcing, batch.params, self.param_dim, self.cond_extra_dim)
        v = self.unet(x_t.transpose(1, 2), cond.transpose(1, 2), tau=tau)
        return v.transpose(1, 2)

    def compute_cfm_loss(self, batch):
        B = batch.obs.shape[0]
        device = batch.obs.device
        tau = torch.zeros(B, device=device) if self.train_tau_0_only else torch.rand(B, device=device)
        x0 = torch.randn_like(batch.states) * self.sigma_prior
        x_tau = self.interpolant.mix(x0, batch.states, tau)
        v_target = batch.states - x0
        v_pred = self.forward(x_tau, batch, tau)
        return F.mse_loss(v_pred, v_target)

    def sample(self, batch, N_outer=None):
        if N_outer is None:
            N_outer = self.N_outer
        obs = batch.obs
        B, T, D = obs.shape
        device = obs.device
        x = torch.randn_like(obs) * self.sigma_prior
        if self.train_tau_0_only:
            v = self.forward(x, batch, tau=torch.zeros(B, device=device))
            return x + v
        dt = 1.0 / N_outer
        for step in range(N_outer):
            tau = torch.full((B,), step / N_outer, device=device)
            v = self.forward(x, batch, tau)
            x = x + dt * v
        return x


class JointCFM(VanillaCFM):
    """Symmetric conditional flow matching on the state AND parameter manifolds.

    State flow: u_theta(x_tau, tau, y, f) conditioned only on [obs, forcing]
    (cond_extra_dim=1); the true parameters are never fed to the state flow.
    Target (as for VanillaCFM): x_1 - x_0 with x_0 ~ N(0, sigma_prior^2).

    Param flow: a separate CNN velocity v_phi(obs, forcing, x_hat_1, param_tau,
    tau) that flows param_0 ~ N(0, I) toward true_param via the interpolant
    param_tau = (1-tau)*param_0 + tau*true_param, target true_param - param_0.
    The state estimate x_hat_1 enters (stop-grad detached) at each tau; coupling
    is state -> param only. true_param appears only as the CFM target in
    training, never as a fixed conditioning input, so no oracle leaks at
    inference.

    Coupled integration: one shared Euler loop advances both flows in lockstep on
    the same tau schedule. At each step the state is advanced FIRST and the
    analytic state estimate x_hat_1(tau_next) = x(tau_next) + (1-tau_next)*u_theta
    is formed; the param velocity then reads this fresh x_hat_1(tau_next) and
    advances param_tau. At tau_next = 1 the analytic estimate snaps to x(1).
    """

    def __init__(self, state_dim=3, param_dim=4, hidden_channels=None, time_emb_dim=64,
                 N_outer=10, sigma_prior=0.5, dropout=0.1, param_loss_weight=0.1,
                 param_flow_channels=None, train_tau_0_only=False):
        super().__init__(state_dim=state_dim, param_dim=param_dim,
                         hidden_channels=hidden_channels,
                         time_emb_dim=time_emb_dim, N_outer=N_outer,
                         sigma_prior=sigma_prior, dropout=dropout,
                         cond_extra_dim=1)
        self.unet = UNet1D(
            state_dim=state_dim,
            obs_dim=state_dim,
            cond_extra_dim=1,
            hidden_channels=hidden_channels,
            use_obs=True,
            use_energy=False,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
            output_dim=state_dim,
        )
        self.param_dim = param_dim
        self.param_loss_weight = param_loss_weight
        self.param_flow = ParamFlowCNN(
            param_dim=param_dim,
            state_dim=state_dim,
            hidden_channels=param_flow_channels,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
        )
        self.train_tau_0_only = train_tau_0_only

    def forward(self, x_t, batch, tau, param_0=None):
        cond = _make_cond(batch.obs, batch.forcing, batch.params, 0, 1)
        v_state = self.unet(x_t.transpose(1, 2), cond.transpose(1, 2), tau=tau)
        v_state = v_state.transpose(1, 2)
        x_hat_1 = x_t + (1.0 - tau).view(-1, 1, 1) * v_state
        v_param = None
        if param_0 is not None:
            Bp, Tp, _ = x_t.shape
            param_tau = ((1.0 - tau).view(-1, 1, 1) * param_0.unsqueeze(1)
                         + tau.view(-1, 1, 1) * batch.true_params.unsqueeze(1))
            param_tau = param_tau.expand(Bp, Tp, -1)
            v_param = self.param_flow(batch.obs, batch.forcing,
                                      x_hat_1.detach(), param_tau, tau)
        return v_state, v_param, x_hat_1

    def _param_target(self, batch, param_0, tau):
        return batch.true_params - param_0

    def compute_cfm_loss(self, batch):
        B = batch.obs.shape[0]
        device = batch.obs.device
        tau = torch.zeros(B, device=device) if self.train_tau_0_only else torch.rand(B, device=device)
        x0 = torch.randn_like(batch.states) * self.sigma_prior
        x_tau = self.interpolant.mix(x0, batch.states, tau)
        v_target = batch.states - x0
        param_0 = torch.randn(B, self.param_dim, device=device)
        v_pred_state, v_pred_param, _ = self.forward(x_tau, batch, tau, param_0)
        loss_cfm = F.mse_loss(v_pred_state, v_target)
        if batch.true_params is not None and self.param_loss_weight > 0:
            param_target = self._param_target(batch, param_0, tau)
            loss_param = F.mse_loss(v_pred_param, param_target)
            return loss_cfm + self.param_loss_weight * loss_param
        return loss_cfm

    def sample(self, batch, N_outer=None, return_params=False):
        if N_outer is None:
            N_outer = self.N_outer
        obs = batch.obs
        B = obs.shape[0]
        device = obs.device
        dt = 1.0 / N_outer
        x = torch.randn_like(obs) * self.sigma_prior
        param = torch.randn(B, self.param_dim, device=device)
        if not return_params:
            if self.train_tau_0_only:
                v_state, _, _ = self.forward(x, batch, tau=torch.zeros(B, device=device))
                return x + v_state
            for step in range(N_outer):
                tau = torch.full((B,), step / N_outer, device=device)
                v_state, _, _ = self.forward(x, batch, tau)
                x = x + dt * v_state
            return x
        if self.train_tau_0_only:
            tau = torch.zeros(B, device=device)
            v_state, v_param, _ = self.forward(x, batch, tau, param)
            x = x + v_state
            param = param + v_param
        else:
            for step in range(N_outer):
                tau = torch.full((B,), step / N_outer, device=device)
                v_state, v_param, _ = self.forward(x, batch, tau, param)
                x = x + dt * v_state
                param = param + dt * v_param
        return x, param


class PredictStateCFM(nn.Module):
    """V3 CFM variant where the network predicts E[x1|xt,y] instead of E[x1-x0|xt,y].

    ODE formulation:
        v = (μ - x) / (1 - τ)  where μ = E[x1 | x_τ, y]
    This represents a backward-drift mechanism that pulls the state towards
    the predicted final state.
    """
    def __init__(self, state_dim=3, hidden_channels=None, time_emb_dim=64,
                 N_outer=10, sigma_prior=0.5, dropout=0.1,
                 train_tau_0_only=False, param_dim=4, cond_extra_dim=0):
        super().__init__()
        self.param_dim = param_dim
        self.cond_extra_dim = cond_extra_dim
        self.hidden_channels = hidden_channels if hidden_channels is not None else [64, 128, 256]
        self.time_emb_dim = time_emb_dim
        self.unet = UNet1D(
            state_dim=state_dim,
            obs_dim=state_dim,
            cond_extra_dim=cond_extra_dim,
            hidden_channels=hidden_channels,
            use_obs=True,
            use_energy=False,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
            output_dim=state_dim,
        )
        self.interpolant = LinearInterpolant(nu=1.0)
        self.N_outer = N_outer
        self.sigma_prior = sigma_prior
        self.state_dim = state_dim
        self.train_tau_0_only = train_tau_0_only

    def forward(self, x_t, batch, tau):
        """Forward pass: predict final state mean μ = E[x1|xt,y]."""
        cond = _make_cond(batch.obs, batch.forcing, batch.params,
                          self.param_dim, self.cond_extra_dim)
        μ = self.unet(x_t.transpose(1, 2), cond.transpose(1, 2), tau=tau)
        return μ.transpose(1, 2)

    def compute_loss(self, batch):
        """Compute CFM loss: MSE(μ, x1) where μ = network prediction."""
        B = batch.obs.shape[0]
        device = batch.obs.device
        tau = torch.zeros(B, device=device) if self.train_tau_0_only else torch.rand(B, device=device)
        x0 = torch.randn_like(batch.states) * self.sigma_prior
        x_tau = self.interpolant.mix(x0, batch.states, tau)
        μ_pred = self.forward(x_tau, batch, tau)
        return F.mse_loss(μ_pred, batch.states)

    def sample(self, batch, N_outer=None):
        """Sample trajectories via forward ODE integration.

        The network predicts μ = E[x_τ=1 | x_τ, y]. We sample by integrating forward:
            x_0 ~ N(0, σ²)
            For τ from 0 to 1: x_τ ← x_τ + dt * (μ_τ - x_τ) / (1 - τ)
        """
        if N_outer is None:
            N_outer = self.N_outer
        obs = batch.obs
        B, T, D = obs.shape
        device = obs.device

        if self.train_tau_0_only:
            x0 = torch.randn_like(obs) * self.sigma_prior
            μ = self.forward(x0, batch, tau=torch.zeros(B, device=device))
            return μ  # single-step: x0 + (μ - x0)/1 = μ

        # Start from random x_0
        x = torch.randn_like(obs) * self.sigma_prior
        dt = 1.0 / N_outer

        # Forward integration with tau as tensor (avoid tau=1 division by zero)
        for step in range(N_outer):
            tau_step = torch.full((B,), step / N_outer, device=device)
            mu = self.forward(x, batch, tau_step)
            v = (mu - x) / (1.0 - tau_step.clamp(max=0.999).view(B, 1, 1).expand(-1, T, -1))
            x = x + dt * v

        return x
class TweedieCFM(nn.Module):
    """Two-stage CFM: MeanEstimatorCell (stage 1) + velocity UNet on residual (stage 2).

    Architecture:
        Stage 1: MeanEstimatorCell assumes obs-only (cond_extra_dim=0)
            x_mean = estimate_mean(obs) = E[x1 | obs]

        Stage 2: Velocity UNet operates in residual space:
            v = E[(x1 - mean) - x0 | x_τ, obs, mean]

        Sampling: mean + CFM_sample(residual)
    """
    def __init__(
        self,
        state_dim: int = 3,
        hidden_channels: list = None,
        time_emb_dim: int = 64,
        K_inner: int = 5,
        N_outer: int = 10,
        sigma_prior: float = 0.5,
        dropout: float = 0.1,
        train_tau_0_only: bool = False,
        cond_extra_dim: int = 0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.K_inner = K_inner
        self.N_outer = N_outer
        self.sigma_prior = sigma_prior
        self.train_tau_0_only = train_tau_0_only

        from models.residual import MeanEstimatorCell
        self.mean_estimator = MeanEstimatorCell(
            state_dim=state_dim,
            hidden_channels=hidden_channels,
            time_emb_dim=time_emb_dim,
            use_obs=True,
            dropout=dropout,
        )
        self.velocity_unet = UNet1D(
            state_dim=state_dim,
            hidden_channels=hidden_channels,
            obs_dim=2 * state_dim,
            cond_extra_dim=cond_extra_dim,
            time_emb_dim=time_emb_dim,
            use_obs=True,
            use_energy=False,
            dropout=dropout,
        )
        self.interpolant = LinearInterpolant(nu=1.0)
        self._stage = 1

    def estimate_mean(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute conditional mean E[x1 | obs] via K_inner iterative refinement."""
        B, T, D = obs.shape
        obs_clean = torch.nan_to_num(obs, nan=0.0)
        x = torch.zeros(B, D, T, device=obs.device)
        for k in range(self.K_inner):
            denom = 1 if self.K_inner == 1 else self.K_inner - 1
            tau = torch.full((B,), k / denom, device=obs.device)
            residual = self.mean_estimator(x, obs_clean.transpose(1, 2), tau)
            x = x + residual
        return x.transpose(1, 2)

    def forward(self, x_t, obs, mean, tau):
        """Predict velocity in residual space.

        Args:
            x_t: Noised residual state (B, T, D)
            obs: Observations (B, T, D)
            mean: Mean estimate (B, T, D)
            tau: Time points (B,) default τ=0 when train_tau_0_only

        Returns:
            v: Predicted velocity (B, T, D)
        """
        if self.train_tau_0_only:
            tau = torch.zeros(obs.shape[0], device=obs.device)
        cond = torch.cat([torch.nan_to_num(obs, nan=0.0), mean], dim=-1)
        v = self.velocity_unet(x_t.transpose(1, 2), cond.transpose(1, 2), tau=tau)
        return v.transpose(1, 2)

    def compute_loss(self, batch):
        """Compute two-stage loss based on current training stage.

        Stage 1: MSE(mean_estimate, x1)  (target = true state, not residual)
        Stage 2: standard CFM loss in residual space
        """
        B = batch.obs.shape[0]
        device = batch.obs.device
        tau = torch.zeros(B, device=device) if self.train_tau_0_only else torch.rand(B, device=device)
        x0 = torch.randn_like(batch.states) * self.sigma_prior
        mean = self.estimate_mean(batch.obs)

        if self._stage == 2:
            x_residue = batch.states - mean
            x_tau_residue = self.interpolant.mix(x0, x_residue, tau)
            v_target = x_residue - x0
            v_pred = self.forward(x_tau_residue, batch.obs, mean, tau)
            return F.mse_loss(v_pred, v_target)
        return F.mse_loss(mean, batch.states)

    def sample(self, batch, N_outer=None):
        """Sample trajectories via Euler integration in residual space.

        Returns: mean + residual_sample
        """
        if N_outer is None:
            N_outer = self.N_outer
        obs = batch.obs
        B, T, D = obs.shape
        device = obs.device

        mean = self.estimate_mean(obs)
        x = torch.randn_like(obs) * self.sigma_prior

        if self.train_tau_0_only:
            v = self.forward(x, obs, mean, tau=torch.zeros(B, device=device))
            x = x + v
        else:
            dt = 1.0 / N_outer
            for step in range(N_outer):
                tau = torch.full((B,), step / N_outer, device=device)
                v = self.forward(x, obs, mean, tau)
                x = x + dt * v

        return mean + x

    def set_stage(self, stage: int):
        """Set the current training stage for compute_loss."""
        self._stage = stage
