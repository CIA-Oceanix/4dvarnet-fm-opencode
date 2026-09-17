"""Unit tests for evaluation/fm_sampler.py -- the configurable FM-prior sampler."""
import pytest
import torch

from evaluation.fm_sampler import (
    Blend,
    Cold,
    Decoupled,
    EnsembleScores,
    SamplerConfig,
    SchemeContext,
    Warm,
    estimate_prior_gain_variance,
    prior_gain,
    reliability_target,
    sample,
)
from models.sda import UnconditionalPriorCFM

SIGMA0 = 0.5


class _MockBatch:
    def __init__(self, B=2, T=8, D=3, obs_every=2):
        self.obs_mask = torch.zeros(B, T, dtype=torch.bool)
        self.obs_mask[:, ::obs_every] = True
        obs = torch.randn(B, T, D)
        self.obs = torch.where(self.obs_mask.unsqueeze(-1), obs,
                               torch.full_like(obs, float("nan")))
        self.states = torch.randn(B, T, D)
        self.forcing = torch.randn(B, T)
        self.params = torch.randn(B, 4)
        self.batch_size = B


def _model(D=3):
    torch.manual_seed(0)
    m = UnconditionalPriorCFM(state_dim=D, hidden_channels=[4, 8], N_outer=3)
    m.eval()
    return m


def _ctx(batch, D=3, **kw):
    B, T = batch.obs.shape[0], batch.obs.shape[1]
    defaults = dict(sigma0=SIGMA0, m_hat=torch.randn(B, T, D),
                    mu_p=torch.randn(B, T, D), p_prior=0.8, p_target=0.3)
    defaults.update(kw)
    return SchemeContext(**defaults)


def _run(scheme, model, batch, ctx, members=2, n_outer=4, seed=0):
    cfg = SamplerConfig(n_outer=n_outer, gamma=1e-2, members=members, seed=seed)
    r_vec = torch.full((batch.obs.shape[-1],), 0.5)
    torch.manual_seed(seed)
    return sample(model, batch, scheme, cfg, ctx, r_vec)


class TestPriorGain:
    def test_zero_at_tau_zero(self):
        assert prior_gain(0.0, 0.8, SIGMA0) == pytest.approx(0.0)

    def test_matches_closed_form(self):
        tau, p = 0.4, 0.8
        expected = tau * p / (((1 - tau) * SIGMA0) ** 2 + tau ** 2 * p)
        assert prior_gain(tau, p, SIGMA0) == pytest.approx(expected, rel=1e-6)

    def test_tends_to_one_over_tau_at_tau_one(self):
        # alpha -> 0, so K -> tau p / (tau^2 p) = 1/tau
        assert prior_gain(1.0, 0.8, SIGMA0) == pytest.approx(1.0, rel=1e-4)


class TestSchemeValidation:
    def test_blend_rejects_p_below_one(self):
        # lam/(1-tau) = lam0 (1-tau)^(p-1) diverges as tau -> 1 for p < 1
        with pytest.raises(ValueError, match="p >= 1"):
            Blend(lam0=0.5, p=0.0)

    def test_blend_rejects_lam0_out_of_range(self):
        with pytest.raises(ValueError, match="lam0"):
            Blend(lam0=1.5, p=2.0)

    def test_decoupled_rejects_unknown_gain(self):
        with pytest.raises(ValueError, match="gain"):
            Decoupled(lam0=0.5, p=2.0, gain="bogus")

    def test_warm_requires_m_hat(self):
        batch = _MockBatch()
        ctx = _ctx(batch, m_hat=None)
        with pytest.raises(ValueError, match="m_hat"):
            _run(Warm(tau0=0.5), _model(), batch, ctx)

    def test_decoupled_target_requires_p_target(self):
        batch = _MockBatch()
        ctx = _ctx(batch, p_target=None)
        with pytest.raises(ValueError, match="p_target"):
            _run(Decoupled(lam0=0.5, p=2.0, gain="target"), _model(), batch, ctx)


class TestSchemeSemantics:
    def test_blend_with_lam0_zero_equals_cold(self):
        batch, model = _MockBatch(), _model()
        ctx = _ctx(batch)
        cold = _run(Cold(), model, batch, ctx)
        blend0 = _run(Blend(lam0=0.0, p=2.0), model, batch, ctx)
        torch.testing.assert_close(cold, blend0)

    def test_blend_is_decoupled_prior_gain_scaled_ng(self):
        """Blend == Decoupled(gain='prior', scale_ng=True) is an exact identity.

        mu~ - c mu_p = lam m_hat, K~ - c K_p = 0, (c-1)x = -lam x, so the general
        affine edit collapses to (1-lam) v + lam (m_hat - x)/(1-tau).
        """
        batch, model = _MockBatch(), _model()
        ctx = _ctx(batch)
        a = _run(Blend(lam0=0.6, p=2.0), model, batch, ctx)
        b = _run(Decoupled(lam0=0.6, p=2.0, gain="prior", scale_ng=True),
                 model, batch, ctx)
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_warm_start_state_is_sdedit_interpolant(self):
        """Warm's initial state is exactly (1-t0) x_0 + t0 m_hat."""
        batch = _MockBatch()
        ctx = _ctx(batch)
        scheme, n_outer = Warm(tau0=0.5), 4
        noise = torch.randn(2, 8, 3)
        state = scheme.initial_state(noise, n_outer, ctx)
        t0 = scheme.start_step(n_outer) / n_outer
        torch.testing.assert_close(state, (1.0 - t0) * noise + t0 * ctx.m_hat)

    def test_warm_skips_the_first_steps(self):
        assert Warm(tau0=0.3).start_step(50) == 15
        assert Cold().start_step(50) == 0


class TestSampleShapeAndDeterminism:
    def test_shape(self):
        batch, model = _MockBatch(), _model()
        out = _run(Cold(), model, batch, _ctx(batch), members=3)
        assert out.shape == (2, 8, 3, 3)

    def test_same_seed_reproduces(self):
        batch, model = _MockBatch(), _model()
        ctx = _ctx(batch)
        torch.testing.assert_close(_run(Cold(), model, batch, ctx, seed=7),
                                   _run(Cold(), model, batch, ctx, seed=7))

    def test_different_seed_differs(self):
        batch, model = _MockBatch(), _model()
        ctx = _ctx(batch)
        a = _run(Cold(), model, batch, ctx, seed=1)
        b = _run(Cold(), model, batch, ctx, seed=2)
        assert not torch.allclose(a, b)

    def test_members_are_distinct_draws(self):
        batch, model = _MockBatch(), _model()
        out = _run(Cold(), model, batch, _ctx(batch), members=2)
        assert not torch.allclose(out[..., 0], out[..., 1])


class TestPriorMeanReproducibility:
    def test_same_seed_reproduces(self):
        """Regression guard: mu_p must not come from the ambient generator.

        Only Decoupled reads mu_p, so an unseeded draw here stays invisible while
        every other row reproduces bit-for-bit -- the pattern that hid this for
        three runs of otherwise identical code.
        """
        from evaluation.fm_sampler import prior_mean
        batch, model = _MockBatch(), _model()
        torch.manual_seed(1234)  # deliberately perturb the ambient state
        a = prior_mean(model, batch, SIGMA0, seed=3)
        torch.manual_seed(9999)
        b = prior_mean(model, batch, SIGMA0, seed=3)
        torch.testing.assert_close(a, b)

    def test_different_seed_differs(self):
        from evaluation.fm_sampler import prior_mean
        batch, model = _MockBatch(), _model()
        a = prior_mean(model, batch, SIGMA0, seed=1)
        b = prior_mean(model, batch, SIGMA0, seed=2)
        assert not torch.allclose(a, b)


class TestPriorGainVariance:
    def test_runs_unguided_and_is_positive(self):
        batch, model = _MockBatch(), _model()
        p = estimate_prior_gain_variance(model, batch, n_outer=4, n_members=3,
                                         sigma0=SIGMA0, seed=0)
        assert p > 0.0

    def test_has_no_guidance_flag(self):
        """Regression guard: the unguided requirement is structural, not optional."""
        import inspect
        params = inspect.signature(estimate_prior_gain_variance).parameters
        assert "guidance" not in params


class TestScoring:
    def test_reliability_target(self):
        assert reliability_target(6) == pytest.approx(0.8452, abs=1e-4)
        assert reliability_target(10) == pytest.approx(0.9045, abs=1e-4)

    def test_perfect_estimate_scores_zero_rmse(self):
        truth = torch.randn(2, 5, 3)
        std, mean = torch.ones(1, 1, 3), torch.zeros(1, 1, 3)
        s = EnsembleScores()
        s.add(truth.unsqueeze(-1), truth, std, mean)
        out = s.summary()
        assert out["rmse_pooled"] == pytest.approx(0.0, abs=1e-6)
        assert out["spread"] == pytest.approx(0.0, abs=1e-6)
        # spread/RMSE is undefined here, and must not raise
        assert out["ratio"] != out["ratio"]

    def test_single_member_has_zero_spread(self):
        members = torch.randn(2, 5, 3, 1)
        s = EnsembleScores()
        s.add(members, torch.randn(2, 5, 3), torch.ones(1, 1, 3), torch.zeros(1, 1, 3))
        assert s.summary()["spread"] == pytest.approx(0.0, abs=1e-6)

    def test_repo_and_pooled_rmse_differ_in_general(self):
        torch.manual_seed(0)
        # per-dimension errors of different magnitude -> mean-of-RMSE < pooled RMSE
        est = torch.zeros(4, 6, 3, 1)
        truth = torch.stack([torch.full((4, 6), s) for s in (0.1, 1.0, 3.0)], dim=-1)
        s = EnsembleScores()
        s.add(est, truth, torch.ones(1, 1, 3), torch.zeros(1, 1, 3))
        out = s.summary()
        assert out["rmse_repo"] < out["rmse_pooled"]
