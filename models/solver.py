import torch
import torch.nn as nn
from .interpolant import LinearInterpolant
from .residual import IterativeUpdateCell, MeanEstimatorCell


class TweedieSolver(nn.Module):
    def __init__(
        self,
        state_dim: int = 3,
        hidden_channels: list = None,
        time_emb_dim: int = 64,
        use_obs: bool = True,
        use_energy: bool = False,
        nu: float = 1.0,
        K_inner: int = 5,
        N_outer: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.K_inner = K_inner
        self.N_outer = N_outer
        self.state_dim = state_dim
        self.interpolant = LinearInterpolant(nu=nu)

        self.mean_estimator = MeanEstimatorCell(
            state_dim=state_dim,
            hidden_channels=hidden_channels,
            time_emb_dim=time_emb_dim,
            use_obs=use_obs,
            dropout=dropout,
        )

        self.non_gaussian = IterativeUpdateCell(
            state_dim=state_dim,
            hidden_channels=hidden_channels,
            time_emb_dim=time_emb_dim,
            use_obs=use_obs,
            use_energy=use_energy,
            dropout=dropout,
        )

    def estimate_mean(self, obs: torch.Tensor) -> torch.Tensor:
        B, T, D = obs.shape
        x = torch.zeros(B, D, T, device=obs.device)
        for k in range(self.K_inner):
            tau = torch.full((B,), (k + 1) / self.K_inner, device=obs.device)
            residual = self.mean_estimator(x, obs.transpose(1, 2), tau)
            x = x + residual
        return x.transpose(1, 2)

    def energy_terms(
        self,
        x: torch.Tensor,
        obs: torch.Tensor,
        x_tau: torch.Tensor,
        tau: torch.Tensor,
        obs_operator: callable = None,
        prior_operator: callable = None,
        nu: float = 1.0,
    ) -> list:
        B, D, T = x.shape
        device = x.device
        y_diff = torch.zeros(B, D, T, device=device)
        if obs_operator is not None:
            obs_T = obs.shape[1] if obs.dim() == 3 else obs.shape[0]
            if obs.dim() == 3:
                y_pred = obs_operator(x.transpose(1, 2)).transpose(1, 2)
                y_diff = obs.transpose(1, 2) - y_pred
            else:
                y_diff = obs[:, :T] - x

        phi_diff = torch.zeros(B, D, T, device=device)
        if prior_operator is not None:
            phi_x = prior_operator(x.transpose(1, 2)).transpose(1, 2)
            phi_diff = x - phi_x

        a = self.interpolant.alpha(tau)
        b = self.interpolant.beta(tau)
        while a.dim() < x.dim():
            a = a.unsqueeze(-1)
            b = b.unsqueeze(-1)
        bg_diff = (b * x - b ** 2 * x_tau) / (a ** 2 * nu ** 2 + 1e-8)

        return [y_diff, phi_diff, bg_diff]

    def forward(
        self,
        obs: torch.Tensor,
        obs_operator: callable = None,
        prior_operator: callable = None,
    ) -> torch.Tensor:
        B, T, D = obs.shape
        device = obs.device

        x_mean = self.estimate_mean(obs)

        x0 = torch.randn(B, T, D, device=device)
        x = x0.clone()

        dt = 1.0 / self.N_outer
        for n in range(1, self.N_outer + 1):
            tau = torch.full((B,), (n - 1) / self.N_outer, device=device)
            a = self.interpolant.alpha(tau)
            b = self.interpolant.beta(tau)
            K = self.interpolant.gain_matrix(tau)

            while K.dim() < x.dim():
                K = K.unsqueeze(-1)
                a = a.unsqueeze(-1)
                b = b.unsqueeze(-1)

            blended = (1 - K) * x_mean + K * x

            for k in range(1, self.K_inner + 1):
                tau_k = (n - 1 + k) / self.N_outer / self.K_inner
                tau_k = torch.full((B,), tau_k, device=device)

                ng_pre = self.interpolant.ng_prefactor(tau_k)
                while ng_pre.dim() < x.dim():
                    ng_pre = ng_pre.unsqueeze(-1)

                eps = self.energy_terms(
                    blended.transpose(1, 2), obs, x.transpose(1, 2),
                    tau_k, obs_operator, prior_operator,
                )
                residual = self.non_gaussian(
                    blended.transpose(1, 2), obs.transpose(1, 2), x.transpose(1, 2),
                    tau_k, y_diff=eps[0], phi_diff=eps[1], bg_diff=eps[2],
                )
                blended = blended + ng_pre * residual.transpose(1, 2)

            drift = self.interpolant.compute_drift(x, blended, tau)
            x = x + dt * drift

        return x
