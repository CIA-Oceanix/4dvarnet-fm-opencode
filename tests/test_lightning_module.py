import pytest
import torch

from models.fourdvarnet import FourDVarNetSolver, FourDVarNetPredictStateCFM
from training.lightning_module import LitModel


def _make_lit(model_type="fourdvarnet", use_cosine_scheduler=False, max_epochs=None,
              update_input="grad+state", prior_unet_lr_scale=1.0):
    model = FourDVarNetSolver(state_dim=3, hidden_channels=[4, 8], N_outer=3,
                               update_input=update_input)
    return LitModel(model, model_type=model_type, stage=1, lr=1e-3,
                     use_cosine_scheduler=use_cosine_scheduler, max_epochs=max_epochs,
                     prior_unet_lr_scale=prior_unet_lr_scale)


def _make_lit_cfm(obs_weight_lr_scale=1.0, trainable_obs_weight=True, update_input="grad+state",
                   prior_unet_lr_scale=1.0):
    """FourDVarNetPredictStateCFM is the model that still has a trainable
    obs_weight (_obs_weight_raw) -- FourDVarNetSolver switched to a trainable
    prior_weight instead (see models/fourdvarnet.py)."""
    model = FourDVarNetPredictStateCFM(state_dim=3, hidden_channels=[4, 8], N_outer=3,
                                        update_input=update_input,
                                        trainable_obs_weight=trainable_obs_weight)
    return LitModel(model, model_type="fourdvarnet_cfm", stage=1, lr=1e-3,
                     obs_weight_lr_scale=obs_weight_lr_scale,
                     prior_unet_lr_scale=prior_unet_lr_scale)


class TestCosineScheduler:
    def test_disabled_returns_plain_optimizer(self):
        lit = _make_lit(use_cosine_scheduler=False)
        out = lit.configure_optimizers()
        assert isinstance(out, torch.optim.Optimizer)

    def test_enabled_for_fourdvarnet_returns_scheduler_dict(self):
        lit = _make_lit(model_type="fourdvarnet", use_cosine_scheduler=True, max_epochs=50)
        out = lit.configure_optimizers()
        assert isinstance(out, dict)
        assert isinstance(out["optimizer"], torch.optim.Optimizer)
        assert isinstance(out["lr_scheduler"], torch.optim.lr_scheduler.CosineAnnealingLR)
        assert out["lr_scheduler"].T_max == 50

    def test_enabled_for_fourdvarnet_cfm_returns_scheduler_dict(self):
        lit = _make_lit(model_type="fourdvarnet_cfm", use_cosine_scheduler=True, max_epochs=50)
        out = lit.configure_optimizers()
        assert isinstance(out, dict)
        assert isinstance(out["lr_scheduler"], torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_enabled_but_not_fourdvarnet_type_ignored(self):
        lit = _make_lit(model_type="direct_unet", use_cosine_scheduler=True, max_epochs=50)
        out = lit.configure_optimizers()
        assert isinstance(out, torch.optim.Optimizer)

    def test_enabled_without_max_epochs_raises(self):
        lit = _make_lit(model_type="fourdvarnet", use_cosine_scheduler=True, max_epochs=None)
        with pytest.raises(ValueError):
            lit.configure_optimizers()


class TestObsWeightLRScale:
    """obs_weight_lr_scale only ever applied to FourDVarNetPredictStateCFM's
    trainable obs_weight; FourDVarNetSolver has no _obs_weight_raw at all now
    (trainable prior_weight instead), so this mechanism is a no-op for it."""

    def test_default_scale_uses_single_param_group(self):
        lit = _make_lit_cfm(obs_weight_lr_scale=1.0)
        optimizer = lit.configure_optimizers()
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]["lr"] == 1e-3

    def test_scaled_lr_splits_into_two_param_groups(self):
        lit = _make_lit_cfm(obs_weight_lr_scale=0.1)
        optimizer = lit.configure_optimizers()
        assert len(optimizer.param_groups) == 2
        lrs = sorted(g["lr"] for g in optimizer.param_groups)
        assert lrs[0] == pytest.approx(1e-4)
        assert lrs[1] == pytest.approx(1e-3)
        obs_weight_group = next(g for g in optimizer.param_groups if g["lr"] == pytest.approx(1e-4))
        assert len(obs_weight_group["params"]) == 1
        assert obs_weight_group["params"][0] is lit.model._obs_weight_raw

    def test_scaled_lr_ignored_when_obs_weight_not_trainable(self):
        lit = _make_lit_cfm(obs_weight_lr_scale=0.1, trainable_obs_weight=False)
        optimizer = lit.configure_optimizers()
        assert len(optimizer.param_groups) == 1

    def test_scaled_lr_ignored_for_non_autograd_update_input(self):
        lit = _make_lit_cfm(obs_weight_lr_scale=0.1, update_input="obs+state")
        optimizer = lit.configure_optimizers()
        assert len(optimizer.param_groups) == 1

    def test_scaled_lr_applies_to_prior_weight_for_fourdvarnet_solver(self):
        """obs_weight_lr_scale is generalized: for FourDVarNetSolver (which
        has _prior_weight_raw, not _obs_weight_raw) it scales prior_weight's
        LR instead."""
        lit = _make_lit(update_input="grad+state")
        lit.obs_weight_lr_scale = 0.1
        optimizer = lit.configure_optimizers()
        assert len(optimizer.param_groups) == 2
        lrs = sorted(g["lr"] for g in optimizer.param_groups)
        assert lrs[0] == pytest.approx(1e-4)
        assert lrs[1] == pytest.approx(1e-3)
        scaled_group = next(g for g in optimizer.param_groups if g["lr"] == pytest.approx(1e-4))
        assert scaled_group["params"][0] is lit.model._prior_weight_raw


class TestPriorUnetLRScale:
    def test_default_scale_uses_single_param_group(self):
        lit = _make_lit(update_input="grad+state", prior_unet_lr_scale=1.0)
        optimizer = lit.configure_optimizers()
        assert len(optimizer.param_groups) == 1

    def test_scaled_lr_splits_off_prior_unet_params(self):
        lit = _make_lit(update_input="grad+state", prior_unet_lr_scale=0.5)
        optimizer = lit.configure_optimizers()
        lrs = sorted(g["lr"] for g in optimizer.param_groups)
        assert lrs == [pytest.approx(5e-4), pytest.approx(1e-3)]
        prior_group = next(g for g in optimizer.param_groups if g["lr"] == pytest.approx(5e-4))
        prior_unet_ids = {id(p) for p in lit.model.prior_unet.parameters()}
        assert {id(p) for p in prior_group["params"]} == prior_unet_ids

    def test_scaled_lr_ignored_without_prior_unet(self):
        lit = _make_lit(update_input="obs+state", prior_unet_lr_scale=0.5)
        optimizer = lit.configure_optimizers()
        assert len(optimizer.param_groups) == 1

    def test_combines_with_obs_weight_lr_scale_into_three_groups(self):
        """FourDVarNetPredictStateCFM: trainable obs_weight AND a scaled
        prior_unet -> three distinct param groups, no overlap."""
        lit = _make_lit_cfm(obs_weight_lr_scale=0.1, prior_unet_lr_scale=0.5)
        optimizer = lit.configure_optimizers()
        assert len(optimizer.param_groups) == 3
        lrs = sorted(g["lr"] for g in optimizer.param_groups)
        assert lrs == [pytest.approx(1e-4), pytest.approx(5e-4), pytest.approx(1e-3)]
        all_param_ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
        assert len(all_param_ids) == len(set(all_param_ids)), "no parameter should appear in two groups"
