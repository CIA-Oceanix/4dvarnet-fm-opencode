"""Tests for UNet1D and components."""
import pytest
import torch
from models.unet import SinusoidalEmbedding, ConvBlock, Down, Up, ConditionEncoder, UNet1D


def test_sinusoidal_embedding_shape():
    B = 4
    dim = 64
    emb = SinusoidalEmbedding(dim)
    t = torch.rand(B)
    out = emb(t)
    assert out.shape == (B, dim)


def test_convblock_shape():
    B, C_in, C_out, L = 2, 3, 16, 50
    block = ConvBlock(C_in, C_out, time_emb_dim=64)
    x = torch.randn(B, C_in, L)
    out = block(x)
    assert out.shape == (B, C_out, L)


def test_convblock_with_time():
    B, C_in, C_out, L = 2, 3, 16, 50
    block = ConvBlock(C_in, C_out, time_emb_dim=8)
    x = torch.randn(B, C_in, L)
    t_emb = SinusoidalEmbedding(8)(torch.rand(B))
    out_no_time = block(x)
    out_with_time = block(x, t_emb)
    assert out_no_time.shape == (B, C_out, L)
    assert out_with_time.shape == (B, C_out, L)


def test_down_shape():
    B, C_in, C_out, L = 2, 3, 16, 100
    down = Down(C_in, C_out, time_emb_dim=64)
    x = torch.randn(B, C_in, L)
    out = down.forward_pool(x)
    assert out.shape == (B, C_out, L // 2)


def test_up_shape():
    B, C_in, C_out, L = 2, 16, 8, 50
    up = Up(C_in, C_out, time_emb_dim=64)
    x = torch.randn(B, C_in, L)
    skip = torch.randn(B, C_out, L * 2)
    out = up(x, skip)
    assert out.shape == (B, C_out, L * 2)


def test_condition_encoder_shape():
    B, D, L = 2, 3, 50
    encoder = ConditionEncoder(state_dim=D, hidden_dim=64, use_obs=True, use_energy=False)
    x = torch.randn(B, D, L)
    obs = torch.randn(B, D, L)
    out = encoder(x, obs)
    assert out.shape == (B, 64, L)


def test_unet1d_forward():
    B, D, L = 2, 3, 50
    model = UNet1D(state_dim=D, hidden_channels=[4, 8], time_emb_dim=8, use_obs=True, use_energy=False)
    x = torch.randn(B, D, L)
    obs = torch.randn(B, D, L)
    tau = torch.rand(B)
    out = model(x, obs=obs, tau=tau)
    assert out.shape == (B, D, L)


def test_unet1d_with_obs():
    B, D, L = 2, 3, 50
    model = UNet1D(state_dim=D, hidden_channels=[4, 8], time_emb_dim=8, use_obs=True, use_energy=False)
    x = torch.randn(B, D, L)
    obs = torch.randn(B, D, L)
    tau = torch.rand(B)
    out = model(x, obs=obs, tau=tau)
    assert out.shape == (B, D, L)


def test_unet1d_with_energy():
    B, D, L = 2, 3, 50
    model = UNet1D(state_dim=D, hidden_channels=[4, 8], time_emb_dim=8, use_obs=True, use_energy=True)
    x = torch.randn(B, D, L)
    obs = torch.randn(B, D, L)
    tau = torch.rand(B)
    energy_terms = [torch.randn(B, D, L) for _ in range(3)]
    out = model(x, obs=obs, tau=tau, energy_terms=energy_terms)
    assert out.shape == (B, D, L)


def test_unet1d_no_obs():
    B, D, L = 2, 3, 50
    model = UNet1D(state_dim=D, hidden_channels=[4, 8], time_emb_dim=8, use_obs=False, use_energy=False)
    x = torch.randn(B, D, L)
    tau = torch.rand(B)
    out = model(x, tau=tau)
    assert out.shape == (B, D, L)
