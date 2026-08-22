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
    run_inference,
)
from evaluation.estimate_metrics import evaluate_estimates, evaluate_npz
from models.direct_unet import DirectUNet
from models.vanilla_cfm import VanillaCFM


class _DictBatch:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _build_case_dataloader(truth, preds):
    """Build a single-case dataloader returning [truth] and [preds]."""
    from torch.utils.data import DataLoader, TensorDataset

    class _Collate:
        def __call__(self, batch):
            t = torch.stack([b[0] for b in batch])
            p = torch.stack([b[1] for b in batch])
            return {"true_state": t, "obs": p, "obs_mask": torch.ones(t.shape[0], dtype=torch.bool),
                    "forcing": torch.zeros(t.shape[0], t.shape[1])}


    ds = TensorDataset(truth, preds)
    return DataLoader(ds, batch_size=len(truth), collate_fn=_Collate())


class _IdentityModel:
    """Deterministic model returning the input obs as prediction."""
    def __call__(self, batch):
        return batch.obs


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

    def test_evaluate_estimates_schema_and_values(self):
        """Generic evaluator: pooled RMSE/EV/ES grouped by component."""
        B, T, D = 4, 5, 24
        truth = np.zeros((B, T, D))
        traj = truth + 0.1  # constant 0.1 error
        m = evaluate_estimates(traj, truth)

        assert m["rmse"] == pytest.approx(0.1)
        for grp in ("slow", "obs_fast", "all_obs"):
            assert grp in m["groups"]
            assert grp in m["ev"]["groups"]
            assert grp in m["es"]["groups"]
        # EV = 1 - 0.01/0 = 1 with zero-variance truth is degenerate; check ok
        assert set(m.keys()) >= {"rmse", "groups", "ev", "es", "num_samples"}

    def test_run_inference_returns_estimates_and_truth(self):
        """run_inference returns per-case trajectories/truth arrays (no metrics)."""
        B, T, D = 4, 5, 24

        class StubDirectUNet(DirectUNet):
            """Minimal DirectUNet returning obs + offset (skips UNet init)."""
            def __init__(self):
                super().__init__(state_dim=D, hidden_channels=[4, 8])
                self.offset = 0.0

            def forward(self, batch):
                return batch.obs + self.offset

        s0_truth = torch.zeros(B, T, D)
        s1_truth = torch.zeros(B, T, D)

        model = StubDirectUNet()
        model.offset = 0.0
        dataloaders = {
            "s0": _build_case_dataloader(s0_truth, s0_truth),
            "s1": _build_case_dataloader(s1_truth, s1_truth),
        }

        est = run_inference(model, dataloaders, torch.device("cpu"))
        assert set(est.keys()) == {"s0", "s1"}
        for case in ("s0", "s1"):
            assert set(est[case].keys()) >= {"trajectories", "truth"}
            assert est[case]["trajectories"].shape == (B, T, D)
            assert est[case]["truth"].shape == (B, T, D)
            # metrics come from the generic evaluator
            m = evaluate_estimates(est[case]["trajectories"], est[case]["truth"])
            assert m["rmse"] == pytest.approx(0.0)

    def test_evaluate_npz_roundtrip(self, tmp_path):
        """evaluate_npz loads stored .npz and returns metrics."""
        from evaluation.estimate_metrics import save_estimates

        B, T, D = 3, 4, 24
        traj = np.random.randn(B, T, D)
        truth = np.random.randn(B, T, D)
        path = str(tmp_path / "est.npz")
        save_estimates(path, traj, truth)
        m = evaluate_npz(path)
        # RMSE = mean over dims of per-dim RMSE (same convention as the DA baselines)
        expected = float(np.mean(np.sqrt(np.mean((traj - truth) ** 2, axis=(0, 1)))))
        assert m["rmse"] == pytest.approx(expected)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
