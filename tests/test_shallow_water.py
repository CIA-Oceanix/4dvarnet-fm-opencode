"""Dynamics-level tests for the two-layer shallow water model (CPU-only)."""
import pytest
import torch

from models.shallow_water_dynamics import ShallowWaterDynamics


class _DotDict(dict):
    """Dict subclass that supports attribute access (mimics Hydra DictConfig)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


def _balanced_ic(dyn: ShallowWaterDynamics) -> torch.Tensor:
    """Geostrophically balanced Bickley jet IC with h near 1.0."""
    s0 = dyn._init_bickley_jet(seed=42, H_ref=1.0)
    return s0.unsqueeze(0)


def test_dynamics_step_finite():
    """Single step produces finite values of preserved shape."""
    dyn = ShallowWaterDynamics(Nx=8, Ny=8)
    x0 = torch.randn(1, dyn.state_dim) * 0.1 + 1.0
    forcing = torch.randn(1, 2)
    x1 = dyn.step(x0, forcing)
    assert x1.shape == x0.shape
    assert x1.isfinite().all()


def test_get_dynamics_factory():
    """get_dynamics() returns a ShallowWaterDynamics for shallow_water type."""
    from models.dynamics import get_dynamics
    cfg = _DotDict(data=_DotDict(
        system="shallow_water",
        Nx=8, Ny=8, K=3, dt=0.01,
        tau0=0.08, f_cor=0.1, g1=0.02, g2=0.01,
        coupling=0.05, friction=0.1, viscosity=0.001,
        land_mask_type="none",
    ))
    dyn = get_dynamics(cfg)
    assert type(dyn).__name__ == "ShallowWaterDynamics"
    assert dyn.Nx == 8
    assert dyn.Ny == 8


def test_get_dynamics_rejects_unknown():
    """Unknown system names still raise ValueError."""
    from models.dynamics import get_dynamics
    cfg = _DotDict(data=_DotDict(system="not_a_system"))
    with pytest.raises(ValueError):
        get_dynamics(cfg)


def test_state_dim():
    """state_dim == 6 * Nx * Ny for an 8x8 grid."""
    dyn = ShallowWaterDynamics(Nx=8, Ny=8)
    assert dyn.state_dim == 6 * 8 * 8
    assert dyn.forcing_dim == 2


def test_short_rollout_bounded():
    """20 steps from the balanced IC keep |h| finite and bounded."""
    dyn = ShallowWaterDynamics(Nx=8, Ny=8)
    x = _balanced_ic(dyn)
    forcing = torch.zeros(20, 2)
    max_h = 0.0
    for t in range(20):
        max_h = max(max_h, x[0, : dyn.Nx * dyn.Ny].abs().max().item())
        x = dyn.step(x, forcing[t : t + 1])
    max_h = max(max_h, x[0, : dyn.Nx * dyn.Ny].abs().max().item())
    assert torch.isfinite(x).all()
    assert max_h < 2.0
