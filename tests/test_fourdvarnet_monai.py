import pytest
import torch

monai = pytest.importorskip("monai")

# E402 is expected here and must not be "fixed" by moving these up: the
# importorskip above has to run first, or collecting this module raises
# ImportError instead of skipping when monai is absent.
from models.fourdvarnet import FourDVarNetSolver  # noqa: E402
from models.monai_unet_adapter import MonaiUNet1D  # noqa: E402
from models.unet import UNet1D  # noqa: E402


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


def _make_monai_model(**kwargs):
    defaults = dict(
        state_dim=3, hidden_channels=[8, 16], N_outer=3,
        unet_backbone="monai", monai_norm_num_groups=8,
    )
    defaults.update(kwargs)
    return FourDVarNetSolver(**defaults)


class TestUnetBackboneDispatch:
    def test_default_backbone_is_unet1d(self):
        model = FourDVarNetSolver(state_dim=3, hidden_channels=[4, 8], N_outer=3)
        assert isinstance(model.unet, UNet1D)
        assert model.unet_backbone == "unet1d"

    def test_unknown_backbone_raises(self):
        with pytest.raises(ValueError):
            FourDVarNetSolver(state_dim=3, hidden_channels=[4, 8], unet_backbone="resnet")

    def test_monai_backbone_builds_monai_unet(self):
        model = _make_monai_model(update_input="obs+state")
        assert isinstance(model.unet, MonaiUNet1D)
        assert model.prior_unet is None

    def test_monai_backbone_with_prior_tau_conditioning_raises(self):
        with pytest.raises(ValueError):
            _make_monai_model(update_input="grad+state", prior_tau_conditioning=True)

    def test_monai_prior_unet_built_for_grad_modes(self):
        model = _make_monai_model(update_input="grad+state")
        assert isinstance(model.prior_unet, MonaiUNet1D)


class TestMonaiNumResBlocks:
    """monai_num_res_blocks (default 2, MonaiUNet1D's own default) selects a
    capacity tier together with hidden_channels -- see
    project_l96_monai_unet_complexity_tiers memory. Param counts below are
    only exact at the real L96 state_dim=24 / monai_norm_num_groups=32 (the
    tier ladder's own reference config) -- not _make_monai_model's tiny
    smoke-test defaults (state_dim=3)."""

    def test_default_matches_m_tier_param_count(self):
        model = _make_monai_model(
            update_input="obs+state", state_dim=24, hidden_channels=[64, 128, 256],
            monai_norm_num_groups=32,
        )
        n = sum(p.numel() for p in model.unet.parameters())
        assert n == 5_889_048

    def test_num_res_blocks_1_matches_s_tier_param_count(self):
        model = _make_monai_model(
            update_input="obs+state", state_dim=24, hidden_channels=[32, 64, 128],
            monai_norm_num_groups=32, monai_num_res_blocks=1,
        )
        n = sum(p.numel() for p in model.unet.parameters())
        assert n == 1_055_544

    def test_applies_to_prior_unet_too(self):
        """prior_unet's own param count at this tier (1,053,240) differs
        slightly from the main solver unet's S-tier count (1,055,544) above
        -- prior_unet takes state-only input (24ch, no obs concatenation)
        vs. the main unet's obs+state input (48ch), a real difference in
        in_channels, not a bug."""
        model = _make_monai_model(
            update_input="grad+state", state_dim=24, hidden_channels=[32, 64, 128],
            monai_norm_num_groups=32, monai_num_res_blocks=1,
        )
        n = sum(p.numel() for p in model.prior_unet.parameters())
        assert n == 1_053_240


class TestMonaiBackboneForwardFDV1:
    """update_input='obs+state' -- the FDV1 configuration, only self.unet swapped."""

    def test_forward_shape(self):
        model = _make_monai_model(update_input="obs+state")
        batch = _MockBatch(B=2, T=32, D=3)
        out = model(batch)
        assert out.shape == (2, 32, 3)

    def test_forward_finite(self):
        model = _make_monai_model(update_input="obs+state")
        batch = _MockBatch(B=2, T=32, D=3)
        out = model(batch)
        assert torch.isfinite(out).all()

    def test_backward_reaches_unet_params(self):
        model = _make_monai_model(update_input="obs+state")
        batch = _MockBatch(B=2, T=32, D=3)
        loss = model.compute_loss(batch)
        loss.backward()
        grads = [p.grad for p in model.unet.parameters() if p.requires_grad]
        assert len(grads) > 0
        assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


class TestMonaiBackboneForwardFDV2GradState:
    """update_input='grad+state' -- the FDV2 configuration, both self.unet and
    self.prior_unet swapped; prior_unet is always called with tau=None."""

    def test_forward_shape(self):
        model = _make_monai_model(update_input="grad+state")
        batch = _MockBatch(B=2, T=32, D=3)
        out = model(batch)
        assert out.shape == (2, 32, 3)

    def test_forward_finite(self):
        model = _make_monai_model(update_input="grad+state")
        batch = _MockBatch(B=2, T=32, D=3)
        out = model(batch)
        assert torch.isfinite(out).all()

    def test_backward_reaches_unet_and_prior_unet_params(self):
        model = _make_monai_model(update_input="grad+state", aux_var_cost_weight=0.1)
        batch = _MockBatch(B=2, T=32, D=3)
        loss = model.compute_loss(batch)
        loss.backward()
        unet_grads = [p.grad for p in model.unet.parameters() if p.requires_grad]
        prior_grads = [p.grad for p in model.prior_unet.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in unet_grads)
        assert any(g is not None and g.abs().sum() > 0 for g in prior_grads)

    def test_prior_weight_trainable_and_positive(self):
        model = _make_monai_model(update_input="grad+state", trainable_prior_weight=True)
        assert model._prior_weight_raw is not None
        assert model.prior_weight > 0


class TestMonaiBackboneForwardFDV2SubgradState:
    """update_input='subgrad+state' -- the two-residual proxy-gradient FDV2
    variant (never calls torch.autograd.grad); same self.unet/self.prior_unet
    monai swap as grad+state above."""

    def test_forward_shape(self):
        model = _make_monai_model(update_input="subgrad+state")
        batch = _MockBatch(B=2, T=32, D=3)
        out = model(batch)
        assert out.shape == (2, 32, 3)

    def test_forward_finite(self):
        model = _make_monai_model(update_input="subgrad+state")
        batch = _MockBatch(B=2, T=32, D=3)
        out = model(batch)
        assert torch.isfinite(out).all()

    def test_backward_reaches_unet_and_prior_unet_params(self):
        model = _make_monai_model(update_input="subgrad+state", aux_var_cost_weight=0.1)
        batch = _MockBatch(B=2, T=32, D=3)
        loss = model.compute_loss(batch)
        loss.backward()
        unet_grads = [p.grad for p in model.unet.parameters() if p.requires_grad]
        prior_grads = [p.grad for p in model.prior_unet.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in unet_grads)
        assert any(g is not None and g.abs().sum() > 0 for g in prior_grads)

    def test_trainable_prior_weight_is_a_noop(self):
        """subgrad+state is not in _AUTOGRAD_MODES, so trainable_prior_weight
        has no effect (prior_weight stays fixed at 1.0, unlike grad+state)."""
        model = _make_monai_model(update_input="subgrad+state", trainable_prior_weight=True)
        assert model._prior_weight_raw is None
        assert model.prior_weight == 1.0

    def test_monai_prior_unet_built(self):
        model = _make_monai_model(update_input="subgrad+state")
        assert isinstance(model.prior_unet, MonaiUNet1D)


class TestMonaiBackboneForwardFDV2GradsplitState:
    """update_input='gradsplit+state' -- the real-autograd counterpart to
    subgrad+state: same two-residual-plus-state input shape, but g_obs/
    g_prior are each a true torch.autograd.grad of their own cost term
    taken separately (in _AUTOGRAD_MODES, unlike subgrad+state); same
    self.unet/self.prior_unet monai swap as grad+state/subgrad+state
    above."""

    def test_forward_shape(self):
        model = _make_monai_model(update_input="gradsplit+state")
        batch = _MockBatch(B=2, T=32, D=3)
        out = model(batch)
        assert out.shape == (2, 32, 3)

    def test_forward_finite(self):
        model = _make_monai_model(update_input="gradsplit+state")
        batch = _MockBatch(B=2, T=32, D=3)
        out = model(batch)
        assert torch.isfinite(out).all()

    def test_backward_reaches_unet_and_prior_unet_params(self):
        model = _make_monai_model(update_input="gradsplit+state", aux_var_cost_weight=0.1)
        batch = _MockBatch(B=2, T=32, D=3)
        loss = model.compute_loss(batch)
        loss.backward()
        unet_grads = [p.grad for p in model.unet.parameters() if p.requires_grad]
        prior_grads = [p.grad for p in model.prior_unet.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in unet_grads)
        assert any(g is not None and g.abs().sum() > 0 for g in prior_grads)

    def test_prior_weight_trainable_and_positive(self):
        """gradsplit+state IS in _AUTOGRAD_MODES (unlike subgrad+state), so
        trainable_prior_weight behaves like grad+state's, not subgrad+state's
        no-op."""
        model = _make_monai_model(update_input="gradsplit+state", trainable_prior_weight=True)
        assert model._prior_weight_raw is not None
        assert model.prior_weight > 0

    def test_monai_prior_unet_built(self):
        model = _make_monai_model(update_input="gradsplit+state")
        assert isinstance(model.prior_unet, MonaiUNet1D)


class TestPriorOutputInitStd:
    """prior_output_init_std (default 0.0, no-op -- MONAI's own
    zero_module() init on prior_unet's final output conv unchanged).
    Nonzero overrides that exact-zero init with N(0, std) instead -- only
    meaningful alongside prior_residual=True, where exact zero-init is a
    provable permanent dead end (see MonaiUNet1D.__init__'s docstring):
    prior_cost/g_prior are both proportional to this layer's own output, so
    a zero-valued output can never receive a nonzero gradient there,
    confirmed empirically on a real trained checkpoint (job 53509,
    2026-09-14 -- 300+ epochs, final conv weight norm still bit-for-bit
    0.0)."""

    def _final_conv_weight(self, prior_unet):
        final_conv = prior_unet.backbone.out[-1]
        return final_conv.conv.weight if hasattr(final_conv, "conv") else final_conv.weight

    def test_default_preserves_zero_module_init(self):
        model = _make_monai_model(update_input="grad+state", prior_residual=True)
        w = self._final_conv_weight(model.prior_unet)
        assert torch.equal(w, torch.zeros_like(w))
        x = torch.randn(2, 10, 3)
        raw = model.prior_unet(x.transpose(1, 2), tau=None).transpose(1, 2)
        assert torch.equal(raw, torch.zeros_like(raw))

    def test_nonzero_std_breaks_zero_init(self):
        torch.manual_seed(0)
        model = _make_monai_model(update_input="grad+state", prior_residual=True,
                                   prior_output_init_std=0.1)
        w = self._final_conv_weight(model.prior_unet)
        assert not torch.equal(w, torch.zeros_like(w))
        empirical_std = w.std().item()
        assert abs(empirical_std - 0.1) < 0.05
        x = torch.randn(2, 10, 3)
        raw = model.prior_unet(x.transpose(1, 2), tau=None).transpose(1, 2)
        assert not torch.equal(raw, torch.zeros_like(raw))
        assert torch.isfinite(raw).all()

    def test_nonzero_std_gives_prior_cost_a_real_gradient(self):
        """The whole point: with prior_residual=True, prior_cost=||f(x)||^2
        -- its gradient w.r.t. the final conv's weight is 2*f(x)*(upstream
        activation), exactly zero when f(x)=0 (default zero-init) but
        genuinely nonzero once f(x) is nonzero (prior_output_init_std>0)."""
        torch.manual_seed(0)
        model = _make_monai_model(update_input="grad+state", prior_residual=True,
                                   prior_output_init_std=0.1, aux_var_cost_weight=0.1,
                                   dropout=0.0)
        batch = _MockBatch(B=2, T=10, D=3)
        loss = model.compute_loss(batch)
        loss.backward()
        w = self._final_conv_weight(model.prior_unet)
        assert w.grad is not None
        assert w.grad.abs().sum().item() > 0

    def test_unet1d_backbone_ignores_the_flag(self):
        """prior_output_init_std is monai-only -- unet1d backbone (no
        zero_module convention at all) must not raise or otherwise react to
        it being set."""
        model = FourDVarNetSolver(state_dim=3, hidden_channels=[4, 8], N_outer=3,
                                   update_input="grad+state", prior_residual=True,
                                   prior_output_init_std=0.1)
        assert isinstance(model.prior_unet, UNet1D)
