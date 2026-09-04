import torch

from models.fourdvarnet import FourDVarNetSolver


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
