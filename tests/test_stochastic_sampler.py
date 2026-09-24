"""DDIM-eta sampling step of reports/l96/probe_stochastic_sampler.py."""
import importlib.util
from pathlib import Path

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "probe_stochastic_sampler",
    Path(__file__).resolve().parents[1] / "reports" / "l96" / "probe_stochastic_sampler.py")
pss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pss)

S0 = 0.5


@pytest.mark.parametrize("s,tau", [(0.0, 0.1), (0.3, 0.4), (0.8, 0.9), (0.9, 1.0)])
def test_eta_zero_is_the_euler_ode_step(s, tau):
    torch.manual_seed(0)
    x, d = torch.randn(64, 5), torch.randn(64, 5)
    euler = x + (tau - s) * (d - x) / (1.0 - s)
    assert torch.allclose(pss.eta_step(x, d, s, tau, S0, 0.0), euler, atol=1e-6)


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("s,tau", [(0.0, 0.2), (0.3, 0.5), (0.6, 0.9)])
def test_every_eta_keeps_the_path_marginal_with_an_exact_denoiser(eta, s, tau):
    n, x1 = 400_000, 1.3
    torch.manual_seed(1)
    x_s = s * x1 + (1 - s) * S0 * torch.randn(n)
    out = pss.eta_step(x_s, torch.full((n,), x1), s, tau, S0, eta)
    assert out.mean().item() == pytest.approx(tau * x1, abs=3e-3)
    assert out.var().item() == pytest.approx(((1 - tau) * S0) ** 2, rel=2e-2)


def test_ancestral_step_from_zero_forgets_x0():
    torch.manual_seed(2)
    d = torch.full((200_000,), 0.7)
    a = pss.eta_step(torch.randn(200_000) * S0, d, 0.0, 0.1, S0, 1.0)
    b = pss.eta_step(torch.randn(200_000) * S0, d, 0.0, 0.1, S0, 1.0)
    assert abs(torch.corrcoef(torch.stack([a, b]))[0, 1].item()) < 0.01


def test_last_step_returns_the_denoiser():
    x, d = torch.randn(8, 3), torch.randn(8, 3)
    assert torch.equal(pss.eta_step(x, d, 0.9, 1.0, S0, 1.0), d)


def test_fair_crps_matches_brute_force():
    torch.manual_seed(3)
    M = 7
    mem, y = torch.randn(M, 11), torch.randn(11)
    brute = (mem - y).abs().mean(0) - (mem.unsqueeze(0) - mem.unsqueeze(1)).abs().sum((0, 1)) / (2 * M * (M - 1))
    assert torch.allclose(pss.fair_crps_sum(mem, y), brute, atol=1e-6)


def test_tau_grid_uniform_and_late_schedules():
    assert pss.tau_grid(4) == [0.0, 0.25, 0.5, 0.75, 1.0]
    late = pss.tau_grid(4, 2.0)
    assert late[0] == 0.0 and late[-1] == 1.0
    gaps = [b - a for a, b in zip(late[:-1], late[1:])]
    assert all(g1 > g2 for g1, g2 in zip(gaps[:-1], gaps[1:]))
