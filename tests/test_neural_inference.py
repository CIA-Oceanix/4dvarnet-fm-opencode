#!/usr/bin/env python3
"""Tests for neural inference and evaluation."""
import pytest
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from unittest.mock import Mock, patch

from evaluation.neural_inference import (
    load_checkpoint,
    resolve_model_class,
    create_model,
    evaluate_model,
)
from models.direct_unet import DirectUNet
from models.vanilla_cfm import VanillaCFM


class TestNeuralInference:
    """Test neural inference utilities."""

    def test_resolve_model_class_direct_unet(self):
        """Test model class resolution for DirectUNet."""
        cfg = Mock()
        cfg.model = {"type": "DirectUNet"}
        model_class, cfg_model = resolve_model_class(cfg)
        assert model_class == DirectUNet

    def test_resolve_model_class_vanilla_cfm(self):
        """Test model class resolution for VanillaCFM."""
        cfg = Mock()
        cfg.model = {"type": "VanillaCFM"}
        model_class, cfg_model = resolve_model_class(cfg)
        assert model_class == VanillaCFM

    def test_resolve_model_class_unknown_type(self):
        """Test unknown model type raises error."""
        cfg = Mock()
        cfg.model = {"type": "UnknownModel"}
        with pytest.raises(ValueError, match="Unknown model type"):
            resolve_model_class(cfg)

    def test_create_model_direct_unet(self):
        """Test DirectUNet model creation."""
        cfg_dict = {
            "model": {
                "state_dim": 24,
                "hidden_channels": [64, 128, 256],
            }
        }
        cfg = OmegaConf.create(cfg_dict)
        model = create_model(DirectUNet, cfg)
        assert isinstance(model, DirectUNet)
        assert model.state_dim == 24

    def test_create_model_vanilla_cfm_tau0(self):
        """Test VanillaCFM model creation with tau=0."""
        cfg_dict = {
            "model": {
                "state_dim": 24,
                "hidden_channels": [64, 128, 256],
                "time_emb_dim": 64,
                "train_tau_0_only": True,
            }
        }
        cfg = OmegaConf.create(cfg_dict)
        model = create_model(VanillaCFM, cfg)
        assert isinstance(model, VanillaCFM)
        assert model.train_tau_0_only

    def test_state_mse_loss(self):
        """Test MSE computation."""
        preds = torch.randn(10, 24)
        true = torch.randn(10, 24)
        
        mse = torch.mean((preds - true) ** 2)
        assert isinstance(mse, torch.Tensor)
        assert mse.shape == ()
        assert mse > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
