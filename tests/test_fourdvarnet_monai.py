import pytest
import torch

monai = pytest.importorskip("monai")

from models.fourdvarnet import FourDVarNetSolver
from models.monai_unet_adapter import MonaiUNet1D
from models.unet import UNet1D


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
