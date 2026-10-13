import pytest
import torch

from evaluation.fm_posterior_sampler import fm_posterior_sample, observation_mask, posterior_state_prediction, tau_grid
from evaluation.sda_sampler import sda_guided_sample
from models.sda import UnconditionalPriorCFM
from models.vanilla_cfm import flow_tau_grid


class _Batch:
    def __init__(self, obs, obs_mask):
        self.obs = obs
        self.obs_mask = obs_mask
        self.batch_size = obs.shape[0]


class GaussianPrior(torch.nn.Module):
    """x1 ~ N(mu, diag(s2)); exact FM velocity of the linear interpolant with x0 ~ N(0, sigma_p^2)."""

    def __init__(self, mu: torch.Tensor, s2: torch.Tensor, sigma_prior: float = 0.5):
        super().__init__()
        self.mu, self.s2, self.sigma_prior = mu, s2, sigma_prior
        self.state_dim = mu.shape[-1]

    def psi(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        t = tau.view(-1, 1, 1)
        a2 = (1 - t) ** 2 * self.sigma_prior ** 2
        return self.mu + t * self.s2 / (t ** 2 * self.s2 + a2) * (x - t * self.mu)

    def forward(self, x, batch, tau):
        return (self.psi(x, tau) - x) / (1 - tau.view(-1, 1, 1))


class TiltedPrior(GaussianPrior):
    """GaussianPrior with a small non-zero Jacobian of Psi at tau = 0, as a trained prior has."""

    def psi(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return super().psi(x, tau) + 1e-2 * x


def _toy(B=2, T=3, D=2, R=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(T, D, generator=g, dtype=torch.float64)
    s2 = 0.5 + torch.rand(T, D, generator=g, dtype=torch.float64)
    prior = GaussianPrior(mu, s2)
    truth = mu + s2.sqrt() * torch.randn(B, T, D, generator=g, dtype=torch.float64)
    obs_mask = torch.zeros(B, T, dtype=torch.bool)
    obs_mask[:, ::2] = True
    obs = truth + R ** 0.5 * torch.randn(B, T, D, generator=g, dtype=torch.float64)
    obs[:, :, 1][:, 2] = float("nan")
    obs = torch.where(obs_mask.unsqueeze(-1), obs, torch.full_like(obs, float("nan")))
    batch = _Batch(obs, obs_mask)
    mask = observation_mask(obs, obs_mask)
    post_var = torch.where(mask > 0, 1 / (1 / s2 + 1 / R), s2.expand(B, T, D))
    post_mean = torch.where(mask > 0, post_var * (mu / s2 + torch.nan_to_num(obs) / R), mu.expand(B, T, D))
    return prior, batch, mask, post_mean, post_var


class TestPosteriorStatePrediction:
    @pytest.mark.parametrize("diag", ["exact", "hutchinson"])
    def test_pigdm_tweedie_is_exact_kalman_update(self, diag):
        R = 0.3
        prior, batch, mask, _, _ = _toy(R=R)
        tau = torch.full((2,), 0.4, dtype=torch.float64)
        x = torch.randn(2, 3, 2, dtype=torch.float64)
        y = torch.nan_to_num(batch.obs)
        _, _, psi_y, _ = posterior_state_prediction(prior, batch, x, tau, y=y, mask=mask, R_norm=R,
                                                    guidance="pigdm", pigdm_diag=diag)
        a2 = (0.6 * prior.sigma_prior) ** 2
        prec = 1 / prior.s2 + 0.16 / a2 + mask / R
        expected = (prior.mu / prior.s2 + 0.4 * x / a2 + mask * y / R) / prec
        assert torch.allclose(psi_y, expected, atol=1e-10)

    def test_dps_guidance_scales_as_inverse_R(self):
        prior, batch, mask, _, _ = _toy()
        tau = torch.full((2,), 0.5, dtype=torch.float64)
        x = torch.randn(2, 3, 2, dtype=torch.float64)
        y = torch.nan_to_num(batch.obs)
        shifts = []
        for R in (0.2, 0.4):
            _, psi, psi_y, _ = posterior_state_prediction(prior, batch, x, tau, y=y, mask=mask, R_norm=R,
                                                          guidance="dps")
            shifts.append(psi_y - psi)
        assert shifts[0].abs().max() > 0
        assert torch.allclose(shifts[0], 2 * shifts[1], atol=1e-12)

    def test_per_channel_R(self):
        prior, batch, mask, _, _ = _toy()
        tau = torch.full((2,), 0.5, dtype=torch.float64)
        x = torch.randn(2, 3, 2, dtype=torch.float64)
        y = torch.nan_to_num(batch.obs)
        kw = dict(y=y, mask=mask, guidance="dps")
        _, psi, psi_y, _ = posterior_state_prediction(prior, batch, x, tau, R_norm=torch.tensor([0.2, 0.4],
                                                      dtype=torch.float64), **kw)
        _, _, psi_a, _ = posterior_state_prediction(prior, batch, x, tau, R_norm=0.2, **kw)
        _, _, psi_b, _ = posterior_state_prediction(prior, batch, x, tau, R_norm=0.4, **kw)
        assert torch.allclose(psi_y[..., 0], psi_a[..., 0]) and torch.allclose(psi_y[..., 1], psi_b[..., 1])

    def test_unobserved_entries_unguided(self):
        prior, batch, mask, _, _ = _toy()
        tau = torch.full((2,), 0.5, dtype=torch.float64)
        x = torch.randn(2, 3, 2, dtype=torch.float64)
        _, psi, psi_y, _ = posterior_state_prediction(prior, batch, x, tau, y=torch.nan_to_num(batch.obs),
                                                      mask=mask, R_norm=0.3, guidance="pigdm")
        assert torch.equal(psi_y[mask == 0], psi[mask == 0])


    @pytest.mark.parametrize("guidance", ["dps", "pigdm"])
    def test_guidance_vanishes_at_tau0_with_nonzero_jacobian(self, guidance):
        prior, batch, mask, _, _ = _toy()
        prior = TiltedPrior(prior.mu, prior.s2)
        x = prior.sigma_prior * torch.randn(2, 3, 2, dtype=torch.float64)
        y = torch.nan_to_num(batch.obs)
        kw = dict(y=y, mask=mask, R_norm=0.3, guidance=guidance)
        _, psi, psi_y, _ = posterior_state_prediction(prior, batch, x, torch.zeros(2, dtype=torch.float64), **kw)
        assert torch.equal(psi_y, psi)
        tau = torch.tensor([0.0, 0.5], dtype=torch.float64)
        _, psi, psi_y, _ = posterior_state_prediction(prior, batch, x, tau, **kw)
        assert torch.equal(psi_y[0], psi[0]) and (psi_y[1] - psi[1]).abs().max() > 0

    @pytest.mark.parametrize("guidance", ["dps", "pigdm"])
    def test_first_step_from_noise_bounded_with_nonzero_jacobian(self, guidance):
        prior, batch, _, post_mean, post_var = _toy()
        prior = TiltedPrior(prior.mu, prior.s2)
        torch.manual_seed(0)
        members, _ = fm_posterior_sample(prior, batch, tau0=0.0, gamma=0.0, guidance=guidance, R_norm=0.3,
                                         n_steps=10, n_members=20)
        assert ((members - post_mean.unsqueeze(-1)) / post_var.sqrt().unsqueeze(-1)).abs().max() < 6


class TestToyPosterior:
    def _run(self, seed, **kw):
        prior, batch, _, post_mean, post_var = _toy(B=1, T=10, D=4)
        torch.manual_seed(seed)
        members, nfe = fm_posterior_sample(prior, batch, guidance="pigdm", R_norm=0.3, n_members=1000,
                                           hutchinson_probes=1, **kw)
        z = ((members.mean(-1) - post_mean) / post_var.sqrt()).abs().max().item()
        return members.var(-1) / post_var, z, nfe, post_mean

    def test_stochastic_pigdm_from_noise_matches_analytic_posterior(self):
        ratio, z, nfe, _ = self._run(0, tau0=0.0, gamma=1.0, n_steps=20)
        assert nfe == 20
        assert z < 0.15
        assert 0.9 < ratio.mean() < 1.1 and ratio.min() > 0.75 and ratio.max() < 1.25

    def test_correctors_restore_posterior_from_mean_warm_start(self):
        post_mean = _toy(B=1, T=10, D=4)[3]
        ratio, z, nfe, _ = self._run(1, x_start=post_mean, tau0=0.7, gamma=1.0, n_steps=20, n_correctors=2,
                                     corrector_snr=0.1)
        assert nfe == 60
        assert z < 0.15
        assert 0.85 < ratio.mean() < 1.15

    def test_deterministic_warm_start_under_disperses(self):
        post_mean = _toy(B=1, T=10, D=4)[3]
        ratio_ode, _, _, _ = self._run(2, x_start=post_mean, tau0=0.7, gamma=0.0, n_steps=10)
        ratio_sde, _, _, _ = self._run(2, x_start=post_mean, tau0=0.7, gamma=1.0, n_steps=10)
        assert ratio_ode.mean() < 0.2
        assert ratio_sde.mean() > 1.5 * ratio_ode.mean()


class TestMatchesSdaGuidedSample:
    def _setup(self):
        torch.manual_seed(0)
        model = UnconditionalPriorCFM(state_dim=3, hidden_channels=[4, 8], N_outer=5).eval()
        B, T = 2, 16
        obs_mask = torch.zeros(B, T, dtype=torch.bool)
        obs_mask[:, ::4] = True
        obs = torch.where(obs_mask.unsqueeze(-1), torch.randn(B, T, 3), torch.tensor(float("nan")))
        obs[0, 4, 1] = float("nan")
        return model, _Batch(obs, obs_mask), torch.randn(B, T, 3)

    @pytest.mark.parametrize("tau0", [0.0, 0.4])
    def test_gamma0_normalized_reproduces_sda(self, tau0):
        model, batch, mean = self._setup()
        torch.manual_seed(5)
        ref, ref_nfe = sda_guided_sample(model, batch, R_var=0.5, N_outer=5, guidance_weight=2.0, n_members=3,
                                         mean_estimate=mean if tau0 > 0 else None, tau0=tau0)
        torch.manual_seed(5)
        n_steps = 5 - int(round(tau0 * 5))
        got, nfe = fm_posterior_sample(model, batch, x_start=mean if tau0 > 0 else None, tau0=tau0,
                                       guidance="normalized", guidance_weight=2.0, R_norm=0.5, n_steps=n_steps,
                                       n_members=3)
        assert nfe == ref_nfe
        assert torch.allclose(got, ref, atol=1e-5)


class TestShapesAndStarts:
    def test_shapes_member_starts_and_tau0_per_window(self):
        prior, batch, _, post_mean, _ = _toy()
        starts = post_mean.unsqueeze(-1) + torch.randn(2, 3, 2, 5, dtype=torch.float64)
        members, _ = fm_posterior_sample(prior, batch, member_starts=starts, tau0=0.5, gamma=0.5,
                                         guidance="dps", R_norm=0.3, n_steps=4, step_power=0.5)
        assert members.shape == (2, 3, 2, 5)
        assert torch.isfinite(members).all()

    def test_tau0_to_one_returns_start(self):
        prior, batch, _, post_mean, _ = _toy()
        start = post_mean + 1.0
        members, _ = fm_posterior_sample(prior, batch, x_start=start, tau0=1.0, gamma=1.0,
                                         guidance="pigdm", R_norm=0.3, n_steps=3, n_members=4)
        assert torch.allclose(members, start.unsqueeze(-1).expand_as(members), atol=1e-4)

    def test_tau0_per_window_matches_scalar_runs(self):
        prior, batch, _, post_mean, _ = _toy()
        kw = dict(x_start=post_mean, gamma=0.0, guidance="pigdm", R_norm=0.3, n_steps=6, n_members=3)
        torch.manual_seed(3)
        mixed, _ = fm_posterior_sample(prior, batch, tau0_per_window=torch.tensor([1 - 1e-7, 0.2]), **kw)
        torch.manual_seed(3)
        scalar, _ = fm_posterior_sample(prior, batch, tau0=0.2, **kw)
        assert torch.allclose(mixed[1], scalar[1], atol=1e-10)
        assert torch.allclose(mixed[0], post_mean[0].unsqueeze(-1).expand_as(mixed[0]), atol=1e-4)

    def test_member_chunk_matches_full_batch_when_deterministic(self):
        prior, batch, _, post_mean, _ = _toy()
        kw = dict(x_start=post_mean, tau0=0.3, gamma=0.0, guidance="dps", R_norm=0.3, n_steps=5, n_members=5)
        torch.manual_seed(4)
        full, _ = fm_posterior_sample(prior, batch, **kw)
        torch.manual_seed(4)
        chunked, _ = fm_posterior_sample(prior, batch, member_chunk=2, **kw)
        assert torch.allclose(full, chunked, atol=1e-12)

    def test_warm_start_requires_start(self):
        prior, batch, _, _, _ = _toy()
        with pytest.raises(ValueError):
            fm_posterior_sample(prior, batch, tau0=0.5, R_norm=0.3, n_members=2)

    def test_R_norm_is_required(self):
        prior, batch, _, _, _ = _toy()
        with pytest.raises(TypeError):
            fm_posterior_sample(prior, batch, n_members=2)

    def test_tau_grid_early_fine(self):
        assert tau_grid(4, 0.5) == flow_tau_grid(4, 0.5)
        assert tau_grid(4, 1.0) == [0.0, 0.25, 0.5, 0.75, 1.0]
