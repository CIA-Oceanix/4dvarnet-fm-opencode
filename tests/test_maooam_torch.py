"""Tests for PyTorch-native MAOOAM dynamics."""

import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from models.maooam_torch import MaooamTorchDynamics


# Shared instance to avoid repeated qgs tensor extraction
_dynamics = None


def _get_dynamics(device="cpu"):
    global _dynamics
    if _dynamics is None:
        _dynamics = MaooamTorchDynamics(device=device, dt=0.1, K=5)
    return _dynamics


class TestMaooamTorchDynamics:
    def test_state_dim(self):
        d = _get_dynamics()
        assert d.state_dim == 104
        assert d.Natm == 36
        assert d.Npsi_o == 16

    def test_rhs_shape_and_finite(self):
        d = _get_dynamics()
        x = torch.randn(d.state_dim, dtype=torch.float64)
        rhs = d._rhs(x)
        assert rhs.shape == (d.state_dim,)
        assert torch.isfinite(rhs).all()

    def test_step_shape_and_finite(self):
        d = _get_dynamics()
        x = torch.randn(1, d.state_dim)
        f = torch.zeros(1, 1)
        out = d.step(x, f)
        assert out.shape == (1, d.state_dim)
        assert torch.isfinite(out).all()

    def test_step_uses_rk4(self):
        d = _get_dynamics()
        x = torch.randn(d.state_dim, dtype=torch.float64)
        # Single RK4 step
        out = d.step(x.unsqueeze(0), torch.zeros(1, 1))
        # Euler step: x + dt * rhs(x)
        euler = x + d.dt * d._rhs(x)
        # RK4 is not Euler — the difference should be O(dt^2)
        euler_diff = (out[0].float() - euler.float()).abs().max().item()
        assert euler_diff > 1e-3  # RK4 != Euler
        # But trajectory should be smooth
        assert torch.isfinite(out).all()

    def test_trajectory_stable(self):
        d = _get_dynamics()
        traj, forcing = d.generate_full_trajectory(1000, seed=42, spinup_steps=200)
        assert traj.shape == (1000, d.state_dim)
        assert torch.isfinite(traj).all()
        assert traj.abs().max().item() < 10.0

    def test_jacobian_shape(self):
        d = _get_dynamics()
        x = torch.randn(d.state_dim, dtype=torch.float64)
        J = d._jacobian(x)
        assert J.shape == (d.state_dim, d.state_dim)
        assert torch.isfinite(J).all()

    def test_autograd(self):
        d = _get_dynamics()
        x = torch.randn(d.state_dim, dtype=torch.float64, requires_grad=True)
        rhs = d._rhs(x)
        loss = rhs.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_spectral_to_physical(self):
        d = _get_dynamics()
        state = np.random.randn(d.state_dim).astype(np.float64) * 0.01
        phys = d.spectral_to_physical(state)
        for k in ["psi_upper", "psi_lower", "psi_oc", "T_atm", "T_oc"]:
            assert k in phys
            assert phys[k].ndim == 2

    def test_spectral_to_physical_interp(self):
        d = _get_dynamics()
        state = np.random.randn(d.state_dim).astype(np.float64) * 0.01
        phys = d.spectral_to_physical(state, interp_size=64)
        assert phys["psi_upper"].shape == (64, 64)


class TestMaooamTorchGPU:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_equivalence(self):
        cpu_dyn = _get_dynamics("cpu")
        gpu_dyn = _get_dynamics("cuda")
        gpu_dyn.load_state_dict(cpu_dyn.state_dict())

        x = torch.randn(cpu_dyn.state_dim, dtype=torch.float64)
        rhs_cpu = cpu_dyn._rhs(x)
        rhs_gpu = gpu_dyn._rhs(x.to("cuda")).cpu()

        max_diff = (rhs_cpu - rhs_gpu).abs().max().item()
        assert max_diff < 1e-8


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])