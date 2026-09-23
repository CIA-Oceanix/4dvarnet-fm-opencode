"""The NS1b / T2 pair construction x_s = pair_from_tau(x_tau, tau, s): given x1 it
must have law N(s x1, b_s^2 I) with b_s = (1 - s) s0, and at s = 0 carry no x1."""
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "reports/l96/probe_tau_consistency_truth_marginal.py"


def _load():
    spec = importlib.util.spec_from_file_location("probe_tau_truth_marginal", PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("s,tau", [(0.0, 0.3), (0.1, 0.2), (0.2, 0.5), (0.3, 0.9)])
def test_pair_has_forward_noising_law(s, tau):
    pair_from_tau = _load().pair_from_tau
    torch.manual_seed(0)
    s0, n = 0.5, 400_000
    x1 = torch.full((n,), 1.7)
    x_tau = tau * x1 + (1.0 - tau) * s0 * torch.randn(n)
    x_s = pair_from_tau(x_tau, tau, s, s0)
    b_s = (1.0 - s) * s0
    assert abs(x_s.mean().item() - s * 1.7) < 5e-3
    assert abs(x_s.std().item() / b_s - 1.0) < 5e-3


def test_pair_is_noisier_copy_of_x_tau():
    pair_from_tau = _load().pair_from_tau
    torch.manual_seed(1)
    s0, s, tau, n = 0.5, 0.1, 0.4, 400_000
    x1 = torch.randn(n)
    x_tau = tau * x1 + (1.0 - tau) * s0 * torch.randn(n)
    x_s = pair_from_tau(x_tau, tau, s, s0)
    resid = x_s - (s / tau) * x_tau
    assert abs(torch.corrcoef(torch.stack([resid, x_tau]))[0, 1].item()) < 5e-3
