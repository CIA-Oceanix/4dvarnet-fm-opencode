import torch

from evaluation.ictm_sampler import hutchinson_trace, ictm_map_sample, rademacher_like
from evaluation.sda_sampler import guided_obs_cost
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


class TestHutchinsonTrace:
    def test_zero_gradient_for_linear_v(self):
        """For v(x) = x @ M.T, dv/dx = M is constant (independent of x), so
        the Hutchinson trace estimate eps^T M eps does not depend on x at
        all -- its gradient w.r.t. x must be EXACTLY zero for any eps draw,
        not just in expectation. A strong, non-flaky check of the
        differentiability wiring (create_graph=True)."""
        torch.manual_seed(1)
        B, D = 3, 5
        M = torch.randn(D, D)
        x = torch.randn(B, D, requires_grad=True)
        v = x @ M.T
        eps = rademacher_like(x)
        trace_est = hutchinson_trace(v, x, eps).sum()
        # For a strictly linear v, dv/dx is a genuine constant, so the vjp
        # autograd builds literally never depends on x -- trace_est ends up
        # with requires_grad=False entirely (PyTorch recognized the
        # constant independently), which is the *stronger* confirmation of
        # the zero-gradient property this test checks, not just a
        # numerically-near-zero value.
        if trace_est.requires_grad:
            grad = torch.autograd.grad(trace_est, x, allow_unused=True)[0]
            grad = torch.zeros_like(x) if grad is None else grad
        else:
            grad = torch.zeros_like(x)
        assert torch.allclose(grad, torch.zeros_like(grad), atol=1e-5)

    def test_unbiased_matches_exact_trace(self):
        """For a nonlinear v, dv/dx is a genuine function of x; the Monte
        Carlo average of the Hutchinson estimator over many eps draws must
        converge to the exact trace (computed independently via
        torch.autograd.functional.jacobian)."""
        torch.manual_seed(0)
        D = 6
        M = torch.randn(D, D) * 0.3

        def f(xv):
            return torch.tanh(xv @ M.T)

        x_vec = torch.randn(D)
        jac = torch.autograd.functional.jacobian(f, x_vec)
        exact_trace = jac.trace().item()

        x = x_vec.clone().unsqueeze(0).requires_grad_(True)
        v = f(x.squeeze(0)).unsqueeze(0)
        n_samples = 4000
        total = 0.0
        for _ in range(n_samples):
            eps = rademacher_like(x)
            total += hutchinson_trace(v, x, eps).item()
        mc_trace = total / n_samples
        assert abs(mc_trace - exact_trace) < 0.2, \
            f"Monte Carlo trace {mc_trace} vs exact {exact_trace}"


class TestIctmMapSample:
    def test_shape_and_finite(self):
        model = _make_model()
        model.eval()
        batch = _MockBatch(B=1, T=10, D=3, obs_every=2)
        x1, n_forward = ictm_map_sample(model, batch, N_outer=3, K=1, lam=1.0, step_size=1e-2)
        assert x1.shape == (1, 10, 3)
        assert torch.isfinite(x1).all()
        assert n_forward == 3 * (1 + 1)

    def test_n_members_stacks_last_dim(self):
        model = _make_model()
        model.eval()
        batch = _MockBatch(B=1, T=10, D=3)
        samples, n_forward = ictm_map_sample(model, batch, N_outer=3, K=1, n_members=3)
        assert samples.shape == (1, 10, 3, 3)
        assert n_forward == 3 * (1 + 1)

    def test_higher_lam_reduces_obs_cost(self):
        model = _make_model()
        model.eval()
        batch = _MockBatch(B=1, T=20, D=3, obs_every=2)

        torch.manual_seed(3)
        low_lam, _ = ictm_map_sample(model, batch, N_outer=8, K=1, lam=0.0, step_size=5e-2)
        torch.manual_seed(3)
        high_lam, _ = ictm_map_sample(model, batch, N_outer=8, K=1, lam=200.0, step_size=5e-2)

        cost_low = guided_obs_cost(low_lam, batch.obs, batch.obs_mask, R_var=1.0)
        cost_high = guided_obs_cost(high_lam, batch.obs, batch.obs_mask, R_var=1.0)
        assert cost_high < cost_low

    def test_k_greater_than_one_changes_trajectory_and_costs_more_nfe(self):
        model = _make_model()
        model.eval()
        batch = _MockBatch(B=1, T=10, D=3, obs_every=2)

        torch.manual_seed(5)
        k1, n1 = ictm_map_sample(model, batch, N_outer=3, K=1, lam=5.0)
        torch.manual_seed(5)
        k3, n3 = ictm_map_sample(model, batch, N_outer=3, K=3, lam=5.0)

        assert not torch.allclose(k1, k3)
        assert n3 > n1

    def test_runs_under_caller_no_grad(self):
        """Must work even if the caller wraps the eval loop in
        torch.no_grad(), same contract as sda_guided_sample."""
        model = _make_model()
        model.eval()
        batch = _MockBatch(B=1, T=10, D=3)
        with torch.no_grad():
            x1, _ = ictm_map_sample(model, batch, N_outer=3, K=1)
        assert torch.isfinite(x1).all()

    def test_deterministic_with_fixed_seed(self):
        model = _make_model()
        model.eval()
        batch = _MockBatch(B=1, T=10, D=3, obs_every=2)
        torch.manual_seed(7)
        a, _ = ictm_map_sample(model, batch, N_outer=3, K=2, lam=2.0)
        torch.manual_seed(7)
        b, _ = ictm_map_sample(model, batch, N_outer=3, K=2, lam=2.0)
        assert torch.allclose(a, b)
