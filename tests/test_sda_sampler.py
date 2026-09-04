import torch

from evaluation.sda_sampler import guided_obs_cost, sda_guided_sample
from models.interpolant import LinearInterpolant
from models.sda import UnconditionalPriorCFM


class _MockBatch:
    def __init__(self, B=2, T=20, D=3, obs_every=4):
        self.states = torch.randn(B, T, D)
        self.obs = torch.randn(B, T, D)
        self.obs_mask = torch.zeros(B, T, dtype=torch.bool)
        self.obs_mask[:, ::obs_every] = True
        self.obs = torch.where(self.obs_mask.unsqueeze(-1), self.obs,
                               torch.full_like(self.obs, float("nan")))
        self.forcing = torch.randn(B, T)
        self.params = torch.randn(B, 4)
        self.batch_size = B


def _make_model():
    return UnconditionalPriorCFM(state_dim=3, hidden_channels=[4, 8], N_outer=3)


class TestLinearInterpolantX1Hat:
    def test_matches_inline_formula(self):
        interpolant = LinearInterpolant(nu=1.0)
        B, T, D = 2, 5, 3
        x_tau = torch.randn(B, T, D)
        v = torch.randn(B, T, D)
        tau = torch.rand(B)
        expected = x_tau + (1.0 - tau).view(-1, 1, 1) * v
        got = interpolant.x1_hat(x_tau, v, tau)
        assert torch.allclose(got, expected)


class TestGuidedObsCost:
    def test_zero_when_matching_at_observed_steps(self):
        B, T, D = 1, 6, 2
        obs_mask = torch.zeros(B, T, dtype=torch.bool)
        obs_mask[:, [1, 3]] = True
        x_hat_1 = torch.randn(B, T, D)
        y = torch.full((B, T, D), float("nan"))
        y[:, [1, 3]] = x_hat_1[:, [1, 3]]
        cost = guided_obs_cost(x_hat_1, y, obs_mask, R_var=0.5)
        assert torch.allclose(cost, torch.tensor(0.0), atol=1e-6)

    def test_ignores_unobserved_steps(self):
        B, T, D = 1, 6, 2
        obs_mask = torch.zeros(B, T, dtype=torch.bool)
        x_hat_1 = torch.randn(B, T, D)
        y = torch.randn(B, T, D) * 100  # would be huge cost if not masked out
        cost = guided_obs_cost(x_hat_1, y, obs_mask, R_var=0.5)
        assert torch.allclose(cost, torch.tensor(0.0), atol=1e-6)

    def test_obs_indices_restricts_which_channels_count(self):
        B, T, D = 1, 6, 3
        obs_mask = torch.zeros(B, T, dtype=torch.bool)
        obs_mask[:, [1, 3]] = True
        x_hat_1 = torch.randn(B, T, D)
        y = x_hat_1.clone()
        y[:, :, 2] += 1000.0  # channel 2 wildly mismatched
        cost = guided_obs_cost(x_hat_1, y, obs_mask, R_var=0.5, obs_indices=[0, 1])
        assert torch.allclose(cost, torch.tensor(0.0), atol=1e-4), \
            "excluded channel's mismatch must not enter the cost"

    def test_obs_indices_all_channels_matches_none(self):
        B, T, D = 1, 6, 3
        obs_mask = torch.zeros(B, T, dtype=torch.bool)
        obs_mask[:, [1, 3]] = True
        x_hat_1 = torch.randn(B, T, D)
        y = torch.randn(B, T, D)
        cost_none = guided_obs_cost(x_hat_1, y, obs_mask, R_var=0.5, obs_indices=None)
        cost_all = guided_obs_cost(x_hat_1, y, obs_mask, R_var=0.5, obs_indices=[0, 1, 2])
        assert torch.allclose(cost_none, cost_all)


class TestSdaGuidedSample:
    def test_guidance_weight_zero_matches_unconditional_sample(self):
        model = _make_model()
        model.eval()
        batch = _MockBatch(B=2, T=20, D=3)

        torch.manual_seed(42)
        with torch.no_grad():
            expected = model.sample(batch, N_outer=3)

        torch.manual_seed(42)
        guided, n_forward = sda_guided_sample(model, batch, R_var=0.1, N_outer=3,
                                              guidance_weight=0.0)
        assert torch.allclose(guided, expected), \
            "guidance_weight=0 must reproduce unconditional sample() exactly"
        assert n_forward == 3

    def test_guidance_reduces_obs_cost(self):
        model = _make_model()
        model.eval()
        batch = _MockBatch(B=2, T=20, D=3, obs_every=2)

        torch.manual_seed(7)
        unguided, _ = sda_guided_sample(model, batch, R_var=0.1, N_outer=5,
                                        guidance_weight=0.0)
        torch.manual_seed(7)
        guided, _ = sda_guided_sample(model, batch, R_var=0.1, N_outer=5,
                                      guidance_weight=5.0)

        cost_unguided = guided_obs_cost(unguided, batch.obs, batch.obs_mask, R_var=0.1)
        cost_guided = guided_obs_cost(guided, batch.obs, batch.obs_mask, R_var=0.1)
        assert cost_guided < cost_unguided, \
            "guided sample should be closer to the observations than the unguided prior sample"

    def test_sample_finite_and_shape(self):
        model = _make_model()
        model.eval()
        batch = _MockBatch(B=1, T=20, D=3)
        samples, n_forward = sda_guided_sample(model, batch, R_var=0.1, N_outer=3,
                                                guidance_weight=1.0)
        assert samples.shape == (1, 20, 3)
        assert torch.isfinite(samples).all()
        assert n_forward == 3

    def test_n_members_stacks_last_dim(self):
        model = _make_model()
        model.eval()
        batch = _MockBatch(B=1, T=20, D=3)
        samples, n_forward = sda_guided_sample(model, batch, R_var=0.1, N_outer=3,
                                                guidance_weight=1.0, n_members=4)
        assert samples.shape == (1, 20, 3, 4)
        assert n_forward == 3

    def test_callable_guidance_schedule(self):
        model = _make_model()
        model.eval()
        batch = _MockBatch(B=1, T=20, D=3)
        torch.manual_seed(1)
        expected, _ = sda_guided_sample(model, batch, R_var=0.1, N_outer=3,
                                        guidance_weight=2.0)
        torch.manual_seed(1)
        got, _ = sda_guided_sample(model, batch, R_var=0.1, N_outer=3,
                                   guidance_weight=lambda _tau: 2.0)
        assert torch.allclose(got, expected)

    def test_obs_indices_restricts_guidance_to_subset_of_channels(self):
        model = _make_model()
        model.eval()
        batch = _MockBatch(B=2, T=20, D=3, obs_every=2)

        torch.manual_seed(7)
        guided_full, _ = sda_guided_sample(model, batch, R_var=0.1, N_outer=5,
                                           guidance_weight=5.0)
        torch.manual_seed(7)
        guided_partial, _ = sda_guided_sample(model, batch, R_var=0.1, N_outer=5,
                                              guidance_weight=5.0, obs_indices=[0])

        # Guiding on all 3 channels vs. only channel 0 must diverge (different
        # gradient direction/normalization at every step).
        assert not torch.allclose(guided_full, guided_partial)

        torch.manual_seed(7)
        unguided, _ = sda_guided_sample(model, batch, R_var=0.1, N_outer=5,
                                        guidance_weight=0.0)

        # Restricting guidance to channel 0 must pull channel 0 closer to its
        # observations than the fully-unguided prior sample, while channel 1
        # (never in obs_indices) is not specifically optimized for.
        cost_partial_ch0 = guided_obs_cost(
            guided_partial, batch.obs, batch.obs_mask, R_var=0.1, obs_indices=[0])
        cost_unguided_ch0 = guided_obs_cost(
            unguided, batch.obs, batch.obs_mask, R_var=0.1, obs_indices=[0])
        assert cost_partial_ch0 < cost_unguided_ch0

    def test_obs_indices_none_unchanged_from_baseline(self):
        """obs_indices=None (the default) must reproduce pre-existing behavior
        exactly -- guards against the state_dim-based `x` init changing
        results for the common (obs_dim == state_dim) case."""
        model = _make_model()
        model.eval()
        batch = _MockBatch(B=1, T=20, D=3)
        torch.manual_seed(3)
        a, _ = sda_guided_sample(model, batch, R_var=0.1, N_outer=3, guidance_weight=1.0)
        torch.manual_seed(3)
        b, _ = sda_guided_sample(model, batch, R_var=0.1, N_outer=3, guidance_weight=1.0,
                                 obs_indices=None)
        assert torch.allclose(a, b)

    def test_runs_under_caller_no_grad(self):
        """The guidance step must work even if the caller wraps the eval loop
        in torch.no_grad(), as evaluation/neural_inference.py's helpers do."""
        model = _make_model()
        model.eval()
        batch = _MockBatch(B=1, T=20, D=3)
        with torch.no_grad():
            samples, _ = sda_guided_sample(model, batch, R_var=0.1, N_outer=3,
                                           guidance_weight=1.0)
        assert torch.isfinite(samples).all()
