"""Verify PyTorch MAOOAM RHS reproduces numba version exactly.

Tests:
1. Single RHS evaluation: max diff < 1e-10 (C6)
2. Multi-step trajectory: max diff < 1e-6 (C7)  (after RK4 bug fix)
3. Gradient verification: autograd works (C8)
4. GPU equivalence: CPU vs CUDA match (C9)

Usage:
    python reports/verify_maooam_torch.py
"""

import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from models.maooam_dynamics import MaooamDynamics
from models.maooam_torch import MaooamTorchDynamics


def test_rhs_equivalence(torch_dyn, numba_dyn, seed=42):
    """C6: Single RHS evaluation — max diff < 1e-10."""
    rng = np.random.RandomState(seed)
    x_np = rng.randn(torch_dyn.state_dim).astype(np.float64) * 0.01
    x_torch = torch.tensor(x_np, dtype=torch.float64)

    rhs_numba = numba_dyn._f(0.0, x_np)
    rhs_torch = torch_dyn._rhs(x_torch).numpy()

    max_diff = np.abs(rhs_numba - rhs_torch).max()
    assert max_diff < 1e-10, f"RHS mismatch: {max_diff:.2e}"
    print(f"  C6 PASSED: max RHS diff = {max_diff:.2e}")
    return True


def test_trajectory_equivalence(torch_dyn, numba_dyn, seed=42, steps=1000):
    """C7: Multi-step trajectory — max diff < 1e-6 after RK4 bug fix."""
    rng = np.random.RandomState(seed)
    x_np = rng.randn(torch_dyn.state_dim).astype(np.float64) * 0.01
    x_torch = torch.tensor(x_np, dtype=torch.float64)

    x_n = x_np.copy()
    x_t = x_torch.clone()
    for s in range(steps):
        x_n = numba_dyn._rk4_numpy(x_n)  # now fixed: k4/6.0
        x_t = torch_dyn._rk4_step(x_t)
        if s in [0, steps//4, steps//2, 3*steps//4, steps-1]:
            diff = np.abs(x_n - x_t.numpy()).max()
            print(f"  step {s+1}: max diff = {diff:.2e}")

    max_diff = np.abs(x_n - x_t.numpy()).max()
    assert max_diff < 1e-6, f"Trajectory mismatch at step {steps}: {max_diff:.2e}"
    print(f"  C7 PASSED: trajectory diff after {steps} steps = {max_diff:.2e}")
    return True


def test_gradient(torch_dyn, seed=42):
    """C8: torch.autograd works on PyTorch RHS."""
    x = torch.randn(torch_dyn.state_dim, dtype=torch.float64, requires_grad=True)
    rhs = torch_dyn._rhs(x)
    loss = rhs.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    print(f"  C8 PASSED: gradient shape={x.grad.shape}, max={x.grad.abs().max():.2e}")
    return True


def test_gpu_equivalence(seed=42):
    """C9: CPU and CUDA produce identical results."""
    if not torch.cuda.is_available():
        print("  C9 SKIPPED: no CUDA available")
        return True

    cpu_dyn = MaooamTorchDynamics(device="cpu")
    gpu_dyn = MaooamTorchDynamics(device="cuda")
    gpu_dyn.load_state_dict(cpu_dyn.state_dict())

    x_cpu = torch.randn(cpu_dyn.state_dim, dtype=torch.float64)
    x_gpu = x_cpu.to("cuda")

    rhs_cpu = cpu_dyn._rhs(x_cpu)
    rhs_gpu = gpu_dyn._rhs(x_gpu).cpu()

    max_diff = (rhs_cpu - rhs_gpu).abs().max().item()
    assert max_diff < 1e-8, f"CPU/GPU mismatch: {max_diff:.2e}"
    print(f"  C9 PASSED: max CPU/GPU diff = {max_diff:.2e}")
    return True


def main():
    print("Building dynamics (this takes ~4 min for JIT + tensor extraction)...")
    torch_dyn = MaooamTorchDynamics(dt=0.1, K=5)
    numba_dyn = MaooamDynamics(dt=0.1, K=5)
    print(f"  state_dim: {torch_dyn.state_dim}")

    print("\n--- C6: RHS equivalence ---")
    test_rhs_equivalence(torch_dyn, numba_dyn)

    print("\n--- C7: Trajectory equivalence ---")
    test_trajectory_equivalence(torch_dyn, numba_dyn, steps=1000)

    print("\n--- C8: Gradient check ---")
    test_gradient(torch_dyn)

    print("\n--- C9: GPU equivalence ---")
    test_gpu_equivalence()

    print("\nAll verification tests PASSED.")


if __name__ == "__main__":
    main()