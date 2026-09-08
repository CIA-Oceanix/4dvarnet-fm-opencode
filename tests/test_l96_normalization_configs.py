"""Tests for the L96 normalization *config wiring* -- component 1 of the plan
in memory project_l96_normalization_integration_plan: every DirectUNet/
VanillaCFM/FDV L96 experiment config must declare ``data.normalize``
explicitly (not rely on an absent key silently defaulting to False), and the
FDV family (previously with no normalization variant at all) gets one.

data/normalization.py's normalize/denormalize/collate wiring itself is
covered by tests/test_l96_normalization.py; this file is about the configs
and train.model_factory, not the normalization math.
"""
import hydra
import pytest
import torch

from train import model_factory

# Canonical DirectUNet/VanillaCFM/FDV L96 configs that must declare
# data.normalize explicitly (true or false) -- no silent default. Excludes
# monai_direct_unet configs (require the separate fdv-monai-proto env, see
# requirements-monai.txt) and non-DirectUNet/VanillaCFM/FDV families
# (joint_*, param_head*, sda_prior*, tweedie_cfm, predict_state_cfm), which
# are out of this plan's scope per the memory.
NORMALIZE_FALSE_CONFIGS = [
    "L1_direct_unet_s0s1", "L1b_direct_unet_s0s1", "L4_direct_unet_s0s1_small",
    "L2_vanilla_cfm_s0s1", "L2b_vanilla_cfm_s0s1", "L3_smoke", "L3_vanilla_cfm_s0s1",
    "L5_vanilla_cfm_s0s1_small_tau0", "L6_vanilla_cfm_s0s1_forcing_cond",
    "FDV1_unrolled_unet_l96", "FDV1CFM_predict_state_l96", "FDV2_grad_state_l96",
    "FDV2_grad_state_l96_fixedw", "FDV2_subgrad_state_l96", "FDV2CFM_grad_state_l96",
]

NORMALIZE_TRUE_CONFIGS = [
    "L1b_direct_unet_s0s1_norm", "L1b_direct_unet_s0s1_norm_lr1e-4",
    "L1b_direct_unet_s0s1_norm_lrfix", "L3_vanilla_cfm_s0s1_norm",
    "L2b_vanilla_cfm_s0s1_norm", "FDV1_unrolled_unet_l96_norm",
    "FDV2_grad_state_l96_norm",
]


def _compose(name):
    with hydra.initialize(config_path="../config", version_base="1.3"):
        return hydra.compose(config_name=f"experiment/{name}")


@pytest.mark.parametrize("name", NORMALIZE_FALSE_CONFIGS)
def test_config_declares_normalize_false_explicitly(name):
    cfg = _compose(name)
    assert "normalize" in cfg.data, f"{name}: data.normalize is not explicit"
    assert cfg.data.normalize is False


@pytest.mark.parametrize("name", NORMALIZE_TRUE_CONFIGS)
def test_config_declares_normalize_true_explicitly(name):
    cfg = _compose(name)
    assert "normalize" in cfg.data, f"{name}: data.normalize is not explicit"
    assert cfg.data.normalize is True


@pytest.mark.parametrize("name", NORMALIZE_FALSE_CONFIGS + [
    "L1b_direct_unet_s0s1_norm", "L3_vanilla_cfm_s0s1_norm",
    "L2b_vanilla_cfm_s0s1_norm", "FDV1_unrolled_unet_l96_norm",
    "FDV2_grad_state_l96_norm",
])
def test_config_instantiates_via_model_factory(name):
    """Every DirectUNet/VanillaCFM/FDV config (normalize true or false) must
    build a real model via train.model_factory on CPU -- the same path
    evaluation.neural_inference.load_model now reuses for auto-discovered
    resolved configs (see tests/test_config_persistence.py).
    """
    cfg = _compose(name)
    model = model_factory(cfg, torch.device("cpu"))
    assert isinstance(model, torch.nn.Module)


class TestFDVModelFactoryOptionalFields:
    """Regression coverage for a bug found while adding the FDV norm configs:
    train.model_factory's fourdvarnet/fourdvarnet_cfm branches read
    R_var/clip_range (and, for fourdvarnet_cfm, obs_weight/min_obs_weight) via
    direct attribute access instead of .get() with a default, unlike the
    other fields on the same fdv/fdv_cfm block. Any config whose fdv block
    predates those fields (e.g. FDV1_unrolled_unet_l96.yaml, written before
    PR #168 added R_var/clip_range to FourDVarNetSolver's constructor) crashed
    model_factory outright -- confirmed against the pre-fix file via `git
    show HEAD`. Fixed to default exactly like the underlying nn.Module
    constructors (models/fourdvarnet.py).
    """

    def test_fourdvarnet_minimal_fdv_block_uses_constructor_defaults(self):
        import hydra as hy

        with hy.initialize(config_path="../config", version_base="1.3"):
            cfg = hy.compose(config_name="experiment/FDV1_unrolled_unet_l96")
        # FDV1's own fdv block has no R_var/clip_range (matches the real file).
        assert "R_var" not in cfg.model.fdv
        assert "clip_range" not in cfg.model.fdv
        model = model_factory(cfg, torch.device("cpu"))
        assert model.R_var == pytest.approx(0.5)
        assert model.clip_range == pytest.approx(50.0)

    def test_fourdvarnet_cfm_minimal_fdv_cfm_block_uses_constructor_defaults(self):
        import hydra as hy

        with hy.initialize(config_path="../config", version_base="1.3"):
            cfg = hy.compose(config_name="experiment/FDV1CFM_predict_state_l96")
        assert "R_var" not in cfg.model.fdv_cfm
        assert "clip_range" not in cfg.model.fdv_cfm
        assert "obs_weight" not in cfg.model.fdv_cfm
        assert "min_obs_weight" not in cfg.model.fdv_cfm
        model = model_factory(cfg, torch.device("cpu"))
        assert model.R_var == pytest.approx(0.5)
        assert model.clip_range == pytest.approx(50.0)
        assert model._obs_weight_fixed == pytest.approx(1.0)
        assert model.min_obs_weight == pytest.approx(1e-3)
