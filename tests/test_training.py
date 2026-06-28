"""Tests for Lightning training pipeline."""
import os
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.solver import TweedieSolver
from training.lightning_module import Lit4DVarNetFM
from training.losses import StateMSELoss
from training.stage1 import train_stage1
from training.stage2 import train_stage2
from data.lorenz63 import Lorenz63Dataset
from data.dataloader import FlowMatchingDataset, collate_fm, FlowMatchingBatch


class _FixedDataset(Dataset):
    """Dataset of identical items for overfitting tests."""
    def __init__(self, states, obs, mask):
        self.states = states
        self.obs = obs
        self.mask = mask

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        return self.states[idx], self.obs[idx], self.mask[idx]


def _make_fixed_loader(B=2, T=50, D=3):
    states = torch.randn(B, T, D)
    obs = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.bool)
    ds = _FixedDataset(states, obs, mask)
    return DataLoader(ds, batch_size=B, collate_fn=collate_fm)


def test_stage1_overfit(tmp_path):
    """Stage 1 training on a single batch should decrease loss."""
    torch.manual_seed(42)
    model = TweedieSolver(state_dim=3, hidden_channels=[4, 8], K_inner=2, N_outer=2)
    loader = _make_fixed_loader(B=2, T=50, D=3)

    with torch.no_grad():
        batch = next(iter(loader))
        pred = model.estimate_mean(batch.obs)
        initial_loss = nn.MSELoss()(pred, batch.states)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        train_stage1(model, loader, loader, epochs=5, lr=1e-2, verbose=False)
    finally:
        os.chdir(old_cwd)

    with torch.no_grad():
        batch = next(iter(loader))
        pred = model.estimate_mean(batch.obs)
        final_loss = nn.MSELoss()(pred, batch.states)

    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.6f} -> {final_loss:.6f}"


def test_stage2_overfit(tmp_path):
    """Stage 2 training on a single batch should decrease loss."""
    torch.manual_seed(42)
    model = TweedieSolver(state_dim=3, hidden_channels=[4, 8], K_inner=2, N_outer=2)
    loader = _make_fixed_loader(B=2, T=50, D=3)

    with torch.no_grad():
        batch = next(iter(loader))
        pred = model(batch.obs)
        initial_loss = nn.MSELoss()(pred, batch.states)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        train_stage2(model, loader, loader, epochs=5, lr=1e-2, verbose=False)
    finally:
        os.chdir(old_cwd)

    with torch.no_grad():
        batch = next(iter(loader))
        pred = model(batch.obs)
        final_loss = nn.MSELoss()(pred, batch.states)

    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.6f} -> {final_loss:.6f}"


def test_state_mse_loss_shape():
    loss_fn = StateMSELoss()
    pred = torch.randn(4, 50, 3)
    target = torch.randn(4, 50, 3)
    loss = loss_fn(pred, target)
    assert loss.ndim == 0


def test_state_mse_loss_value():
    loss_fn = StateMSELoss()
    target = torch.randn(4, 50, 3)
    loss = loss_fn(target, target)
    assert loss.item() == 0.0


def test_state_mse_gradient_loss():
    pred = torch.randn(4, 50, 3)
    target = torch.randn(4, 50, 3)
    loss_no_grad = StateMSELoss(use_gradient_loss=False)(pred, target)
    loss_with_grad = StateMSELoss(use_gradient_loss=True, gradient_weight=0.1)(pred, target)
    assert loss_with_grad > loss_no_grad


def test_lit_module_creation():
    model = TweedieSolver(state_dim=3, hidden_channels=[4, 8], K_inner=2, N_outer=2)
    lit_module = Lit4DVarNetFM(model=model, stage=1, lr=1e-3)
    assert lit_module.stage == 1
    assert lit_module.lr == 1e-3
    assert lit_module.model is model
