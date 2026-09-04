import torch


class LinearInterpolant:
    def __init__(self, nu: float = 1.0):
        self.nu = nu

    def alpha(self, tau: torch.Tensor) -> torch.Tensor:
        return 1.0 - tau

    def beta(self, tau: torch.Tensor) -> torch.Tensor:
        return tau

    def alpha_dot(self, tau: torch.Tensor) -> torch.Tensor:
        return torch.full_like(tau, -1.0)

    def beta_dot(self, tau: torch.Tensor) -> torch.Tensor:
        return torch.full_like(tau, 1.0)

    def mix(self, x0: torch.Tensor, x1: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        a = self.alpha(tau)
        b = self.beta(tau)
        while a.dim() < x0.dim():
            a = a.unsqueeze(-1)
            b = b.unsqueeze(-1)
        return a * x0 + b * x1

    def x1_hat(self, x_tau: torch.Tensor, v: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Tweedie-style posterior-mean estimate x1_hat = x_tau + (1-tau)*v.

        Direct consequence of the linear interpolant (x_tau = (1-tau)x0 + tau*x1,
        v_target = x1 - x0 => x1 = x_tau + (1-tau)*v). Matches the formula
        inlined in ``JointCFM.forward`` (``models/vanilla_cfm.py``).
        """
        b = self.alpha(tau)
        while b.dim() < x_tau.dim():
            b = b.unsqueeze(-1)
        return x_tau + b * v

    def gain_matrix(self, tau: torch.Tensor) -> torch.Tensor:
        a = self.alpha(tau)
        b = self.beta(tau)
        denom = b ** 2 + a ** 2 * self.nu ** 2
        K = b ** 2 / denom
        return K

    def ng_prefactor(self, tau: torch.Tensor) -> torch.Tensor:
        return self.alpha(tau) * self.beta(tau)

    def sample_tau(self, shape, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        return torch.rand(shape, device=device)

    def compute_drift(self, x: torch.Tensor, x_cond_mean: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        a = self.alpha(tau)
        b = self.beta(tau)
        ad = self.alpha_dot(tau)
        bd = self.beta_dot(tau)

        while a.dim() < x.dim():
            a = a.unsqueeze(-1)
            b = b.unsqueeze(-1)
            ad = ad.unsqueeze(-1)
            bd = bd.unsqueeze(-1)

        coeff = bd - b * ad / a
        return (ad / a) * x + coeff * x_cond_mean
