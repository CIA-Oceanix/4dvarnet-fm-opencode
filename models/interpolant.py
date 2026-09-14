import torch


class LinearInterpolant:
    def __init__(self, nu: float = 1.0, tau_sampling: str = "uniform",
                 logit_normal_loc: float = 0.0, logit_normal_scale: float = 1.0,
                 beta_alpha: float = 2.5, beta_beta: float = 1.0):
        self.nu = nu
        if tau_sampling not in ("uniform", "logit_normal", "beta"):
            raise ValueError(f"Unknown tau_sampling: {tau_sampling!r}")
        self.tau_sampling = tau_sampling
        self.logit_normal_loc = logit_normal_loc
        self.logit_normal_scale = logit_normal_scale
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta

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
        if self.tau_sampling == "uniform":
            return torch.rand(shape, device=device)
        if self.tau_sampling == "logit_normal":
            # tau = sigmoid(loc + scale * z), z ~ N(0, 1)  (Stable Diffusion 3 style)
            z = torch.randn(shape, device=device)
            return torch.sigmoid(self.logit_normal_loc + self.logit_normal_scale * z)
        # beta: tau ~ Beta(beta_alpha, beta_beta) (Zheng et al., "Beta-Tuned Timestep
        # Diffusion Model", ECCV 2024). alpha<beta skews toward tau=0 -- in this
        # interpolant that's x0 (noise), the harder/high-corruption end the Euler
        # sampler starts from; alpha>beta skews toward tau=1 (x1, data, easier).
        # alpha=beta<1 gives a U-shaped density biased toward both ends.
        concentration1 = torch.full(shape, self.beta_alpha, device=device)
        concentration0 = torch.full(shape, self.beta_beta, device=device)
        return torch.distributions.Beta(concentration1, concentration0).sample()

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
