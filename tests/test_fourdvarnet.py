import torch
import torch.nn.functional as F

from models.fourdvarnet import FourDVarNetPredictStateCFM, FourDVarNetSolver


class _MockBatch:
    def __init__(self, B=2, T=50, D=3, obs_every=10, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.states = torch.randn(B, T, D)
        obs = torch.randn(B, T, D)
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, ::obs_every] = True
        self.obs = torch.where(mask.unsqueeze(-1), obs, torch.full_like(obs, float("nan")))
        self.obs_mask = mask
        self.batch_size = B


def _make_model(**kwargs):
    defaults = dict(state_dim=3, hidden_channels=[4, 8], N_outer=3)
    defaults.update(kwargs)
    return FourDVarNetSolver(**defaults)


class TestFourDVarNetSolver:
    def test_forward_shape(self):
        model = _make_model()
        batch = _MockBatch(B=2, T=50, D=3)
        out = model(batch)
        assert out.shape == (2, 50, 3)

    def test_forward_finite(self):
        model = _make_model()
        batch = _MockBatch(B=2, T=50, D=3)
        out = model(batch)
        assert torch.isfinite(out).all()

    def test_zero_iterations_returns_init_state(self):
        model = _make_model(N_outer=0)
        batch = _MockBatch(B=2, T=20, D=3)
        out = model(batch)
        assert torch.equal(out, torch.zeros_like(out))

    def test_weight_tied_param_count_independent_of_N_outer(self):
        model_a = _make_model(N_outer=3)
        model_b = _make_model(N_outer=10)
        n_a = sum(p.numel() for p in model_a.parameters())
        n_b = sum(p.numel() for p in model_b.parameters())
        assert n_a == n_b, "param count must not scale with N_outer (weight-tied)"

    def test_output_responds_to_obs(self):
        model = _make_model()
        model.eval()
        batch_a = _MockBatch(B=2, T=50, D=3, seed=0)
        batch_b = _MockBatch(B=2, T=50, D=3, seed=0)
        batch_b.states = batch_a.states.clone()
        batch_b.obs = torch.where(
            batch_b.obs_mask.unsqueeze(-1),
            torch.randn_like(batch_b.obs) + 100.0,
            batch_b.obs,
        )
        with torch.no_grad():
            out_a = model(batch_a)
            out_b = model(batch_b)
        assert not torch.allclose(out_a, out_b), \
            "output must depend on obs (unlike SDA1's unconditional prior)"

    def test_obs_state_differs_from_obs_only(self):
        torch.manual_seed(0)
        model_state = _make_model(update_input="obs+state")
        torch.manual_seed(0)
        model_obs_only = _make_model(update_input="obs-only")
        batch = _MockBatch(B=2, T=50, D=3)
        out_state = model_state(batch)
        out_obs_only = model_obs_only(batch)
        assert not torch.allclose(out_state, out_obs_only)

    def test_unimplemented_update_input_raises(self):
        for mode in ("grad-only", "grad+state", "subgrad+state"):
            try:
                _make_model(update_input=mode)
                raise AssertionError(f"expected NotImplementedError for {mode!r}")
            except NotImplementedError:
                pass

    def test_unknown_update_input_raises_value_error(self):
        try:
            _make_model(update_input="not-a-real-mode")
            raise AssertionError("expected ValueError for an unrecognized update_input")
        except ValueError:
            pass

    def test_compute_loss_matches_final_iteration_mse(self):
        import torch.nn.functional as F
        model = _make_model(dropout=0.0)
        model.eval()
        batch = _MockBatch(B=2, T=50, D=3)
        loss = model.compute_loss(batch)
        expected = F.mse_loss(model(batch), batch.states)
        assert torch.allclose(loss, expected)

    def test_gradients_flow_through_unroll(self):
        model = _make_model(N_outer=4)
        batch = _MockBatch(B=2, T=20, D=3)
        loss = model.compute_loss(batch)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"no gradient reached {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite gradient at {name}"

    def test_deterministic_eval(self):
        model = _make_model(dropout=0.0)
        model.eval()
        batch = _MockBatch(B=1, T=30, D=3)
        with torch.no_grad():
            out_a = model(batch)
            out_b = model(batch)
        assert torch.equal(out_a, out_b)

    def test_state_dim_24(self):
        model = _make_model(state_dim=24, hidden_channels=[8, 16])
        batch = _MockBatch(B=2, T=60, D=24, obs_every=20)
        out = model(batch)
        assert out.shape == (2, 60, 24)
        assert torch.isfinite(out).all()

    def test_more_unroll_iterations_do_not_hurt_fit(self):
        torch.manual_seed(0)
        batch = _MockBatch(B=4, T=20, D=3, obs_every=5)

        def _fit_loss(N_outer, steps=20):
            torch.manual_seed(1)
            model = _make_model(N_outer=N_outer, hidden_channels=[4, 8])
            opt = torch.optim.Adam(model.parameters(), lr=1e-2)
            for _ in range(steps):
                opt.zero_grad()
                loss = model.compute_loss(batch)
                loss.backward()
                opt.step()
            return model.compute_loss(batch).item()

        loss_1 = _fit_loss(N_outer=1)
        loss_5 = _fit_loss(N_outer=5)
        assert loss_5 <= loss_1 + 0.5, (
            f"N_outer=5 fit loss ({loss_5}) should not be much worse than "
            f"N_outer=1 ({loss_1}) after equal optimizer steps"
        )


def _make_cfm_model(**kwargs):
    defaults = dict(state_dim=3, hidden_channels=[4, 8], N_outer=3, K_inner=2)
    defaults.update(kwargs)
    return FourDVarNetPredictStateCFM(**defaults)


class TestFourDVarNetPredictStateCFM:
    def test_forward_shape_and_finite(self):
        model = _make_cfm_model()
        batch = _MockBatch(B=2, T=50, D=3)
        x_tau = torch.randn(2, 50, 3)
        tau = torch.rand(2)
        mu = model(x_tau, batch, tau)
        assert mu.shape == (2, 50, 3)
        assert torch.isfinite(mu).all()

    def test_compute_loss_matches_manual_formula(self):
        model = _make_cfm_model(dropout=0.0)
        model.eval()
        batch = _MockBatch(B=2, T=50, D=3)

        torch.manual_seed(5)
        loss = model.compute_loss(batch)

        torch.manual_seed(5)
        B, device = batch.states.shape[0], batch.states.device
        tau = torch.rand(B, device=device)
        x0 = torch.randn_like(batch.states) * model.sigma_prior
        x_tau = model.interpolant.mix(x0, batch.states, tau)
        mu_pred = model(x_tau, batch, tau)
        expected = F.mse_loss(mu_pred, batch.states)

        assert torch.allclose(loss, expected)

    def test_sample_shape_finite_and_stochastic(self):
        model = _make_cfm_model()
        model.eval()
        batch = _MockBatch(B=2, T=30, D=3)
        with torch.no_grad():
            a = model.sample(batch)
            b = model.sample(batch)
        assert a.shape == (2, 30, 3)
        assert torch.isfinite(a).all()
        assert not torch.allclose(a, b), "sample() must be stochastic (fresh noise init each call)"

    def test_k_inner_one_matches_fourdvarnet_solver_n_outer_one(self):
        """K_inner=1 degenerates to a single UNet call, structurally identical
        to FourDVarNetSolver with N_outer=1 -- same formula, just starting
        from an arbitrary x_t instead of a hardcoded zero state."""
        cfm = _make_cfm_model(K_inner=1, dropout=0.0)
        cfm.eval()
        solver = FourDVarNetSolver(state_dim=3, hidden_channels=[4, 8], N_outer=1, dropout=0.0)
        solver.eval()
        solver.unet.load_state_dict(cfm.unet.state_dict())
        batch = _MockBatch(B=2, T=20, D=3)
        x_t = torch.zeros(2, 20, 3)  # matches FourDVarNetSolver's own x_0 = 0
        tau = torch.rand(2)
        with torch.no_grad():
            mu = cfm(x_t, batch, tau)
            solver_out = solver(batch)
        assert torch.allclose(mu, solver_out)

    def test_unimplemented_update_input_raises(self):
        try:
            _make_cfm_model(update_input="grad-only")
            raise AssertionError("expected NotImplementedError")
        except NotImplementedError:
            pass

    def test_gradients_flow_through_inner_unroll(self):
        model = _make_cfm_model(K_inner=4)
        batch = _MockBatch(B=2, T=20, D=3)
        loss = model.compute_loss(batch)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"no gradient reached {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite gradient at {name}"

    def test_forward_output_bounded_by_clip_range(self):
        """Regression guard for a rare (~1-in-several-thousand ens30 samples)
        divergence found during FDV1-CFM's first full evaluation: an
        out-of-distribution x_t can make the K_inner unroll blow up within a
        handful of steps, corrupting the ensemble-mean point estimate. Force
        the pathological regime with untrained (large-scale) weights and an
        adversarial x_t, and check forward() never exceeds clip_range."""
        torch.manual_seed(0)
        model = FourDVarNetPredictStateCFM(
            state_dim=3, hidden_channels=[4, 8], N_outer=3, K_inner=5, clip_range=50.0,
        )
        with torch.no_grad():
            for p in model.unet.parameters():
                p.mul_(20.0)
        batch = _MockBatch(B=4, T=20, D=3, seed=1)
        x_t = torch.randn(4, 20, 3) * 1000.0
        tau = torch.rand(4)
        mu = model(x_t, batch, tau)
        assert torch.isfinite(mu).all()
        assert mu.abs().max().item() <= 50.0 + 1e-4

    def test_clip_range_inactive_in_distribution(self):
        """The clamp must not perturb ordinary, in-distribution behavior --
        rerunning test_k_inner_one_matches_fourdvarnet_solver_n_outer_one-style
        inputs should give identical output whether or not clip_range binds."""
        model_clipped = _make_cfm_model(dropout=0.0, clip_range=50.0)
        model_unclipped = _make_cfm_model(dropout=0.0, clip_range=1e6)
        model_unclipped.load_state_dict(model_clipped.state_dict())
        model_clipped.eval()
        model_unclipped.eval()
        batch = _MockBatch(B=2, T=20, D=3)
        x_t = torch.randn(2, 20, 3)
        tau = torch.rand(2)
        with torch.no_grad():
            out_clipped = model_clipped(x_t, batch, tau)
            out_unclipped = model_unclipped(x_t, batch, tau)
        assert torch.allclose(out_clipped, out_unclipped)

    def test_sample_mean_estimate_none_reproduces_baseline(self):
        """mean_estimate=None must reproduce the pre-existing sample() exactly
        -- the regression invariant for the FDV1+FDV1-CFM warm-start hybrid."""
        model = _make_cfm_model(dropout=0.0)
        model.eval()
        batch = _MockBatch(B=2, T=20, D=3)
        torch.manual_seed(7)
        out_explicit_none = model.sample(batch, mean_estimate=None)
        torch.manual_seed(7)
        out_default = model.sample(batch)
        assert torch.allclose(out_explicit_none, out_default)

    def test_sample_tau0_zero_matches_baseline_up_to_the_snapped_start(self):
        """tau0=0.0 with a mean_estimate must still take step0=0 (pure noise
        start, no mixing) -- same code path as mean_estimate=None."""
        model = _make_cfm_model(dropout=0.0)
        model.eval()
        batch = _MockBatch(B=2, T=20, D=3)
        mean_estimate = torch.randn(2, 20, 3) * 10.0
        torch.manual_seed(3)
        out_tau0_zero = model.sample(batch, mean_estimate=mean_estimate, tau0=0.0)
        torch.manual_seed(3)
        out_baseline = model.sample(batch, mean_estimate=None)
        assert torch.allclose(out_tau0_zero, out_baseline)

    def test_sample_warm_start_matches_manual_replay(self):
        """tau0>0 must start from interpolant.mix(noise, mean_estimate, tau0)
        (snapped to the nearest step/N_outer) and run only the remaining
        steps -- replay the same noise draw manually and compare."""
        model = _make_cfm_model(N_outer=4, dropout=0.0)
        model.eval()
        batch = _MockBatch(B=2, T=20, D=3)
        mean_estimate = torch.randn(2, 20, 3) * 5.0
        tau0 = 0.5  # step0 = round(0.5*4) = 2
        torch.manual_seed(11)
        out = model.sample(batch, mean_estimate=mean_estimate, tau0=tau0)

        torch.manual_seed(11)
        noise = torch.randn(2, 20, 3) * model.sigma_prior
        x = model.interpolant.mix(noise, mean_estimate, torch.full((2,), 0.5))
        dt = 0.25
        with torch.no_grad():
            for step in range(2, 4):
                tau_val = step / 4
                tau = torch.full((2,), tau_val)
                mu = model(x, batch, tau)
                x = x + dt * (mu - x) / (1 - tau_val)
        assert torch.allclose(out, x)

    def test_sample_tau0_near_one_runs_a_single_step(self):
        model = _make_cfm_model(N_outer=10, dropout=0.0)
        model.eval()
        batch = _MockBatch(B=2, T=20, D=3)
        mean_estimate = torch.randn(2, 20, 3)
        with torch.no_grad():
            out = model.sample(batch, mean_estimate=mean_estimate, tau0=0.95)
        assert torch.isfinite(out).all()

    def test_sample_warm_start_still_stochastic(self):
        """Two calls with the same mean_estimate/tau0 must still differ
        (fresh noise each call) -- an ensemble around the anchor, not a
        single point."""
        model = _make_cfm_model(dropout=0.0)
        model.eval()
        batch = _MockBatch(B=2, T=20, D=3)
        mean_estimate = torch.randn(2, 20, 3)
        with torch.no_grad():
            a = model.sample(batch, mean_estimate=mean_estimate, tau0=0.5)
            b = model.sample(batch, mean_estimate=mean_estimate, tau0=0.5)
        assert not torch.allclose(a, b)
