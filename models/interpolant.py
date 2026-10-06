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

    def score(self, x_tau: torch.Tensor, v: torch.Tensor, tau: torch.Tensor,
              sigma: float = 1.0) -> torch.Tensor:
        """Closed-form prior score grad_{x_tau} log p(x_tau) from the velocity
        field alone -- no backward pass through the model needed.

        Zhang et al., "Flow Priors for Linear Inverse Problems via Iterative
        Corrupted Trajectory Matching" (ICTM, NeurIPS 2024, see
        ``reports/FlowPrior.pdf`` and ``evaluation/ictm_sampler.py``),
        Proposition 1, adapted to this module's tau=0-noise/tau=1-data
        convention (their beta_t, the noise coefficient, is this module's
        alpha(tau)=1-tau).

        Derivation: x0 ~ N(0, sigma^2 I) and x_tau = alpha(tau)*x0 + beta(tau)*x1
        with v = x1 - x0, so x_tau = x0 + beta(tau)*v, giving the posterior
        mean E[x0|x_tau] = x_tau - beta(tau)*v (the optimal v_theta IS this
        conditional expectation of x1-x0, so this holds exactly, not just
        approximately). The conditional x_tau|x0 is Gaussian with mean
        alpha(tau)*x0 and covariance alpha(tau)^2*sigma^2*I, whose score is
        -(x_tau - alpha(tau)*x0)/(alpha(tau)^2*sigma^2) = -x0/(alpha(tau)*sigma^2);
        by the standard denoising-score-matching identity the marginal score
        equals this conditional score evaluated at the posterior mean of x0.
        Verified against the closed-form score of a Gaussian data distribution
        in ``tests/test_interpolant.py``.

        At tau=0, beta(0)=0 so the ``v`` term drops out entirely and this
        reduces to ``-x_tau/sigma^2`` -- exactly ICTM Algorithm 1's separate
        t=0 closed-form base case (``-log p(x_0)`` for x_0 ~ N(0, sigma^2 I)
        directly), so no special-casing is needed here.
        """
        a = self.alpha(tau)
        b = self.beta(tau)
        while a.dim() < x_tau.dim():
            a = a.unsqueeze(-1)
            b = b.unsqueeze(-1)
        x0_hat = x_tau - b * v
        return -x0_hat / (a * sigma ** 2)

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
