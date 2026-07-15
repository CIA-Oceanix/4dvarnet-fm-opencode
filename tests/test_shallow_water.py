"""Tests for rotating shallow water case study."""
import numpy as np
import pytest
import torch


# ── Helper for get_dynamics tests ───────────────────────────────────

class _DotDict(dict):
    """Dict subclass that supports attribute access (mimics Hydra DictConfig)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


# ── CPU-only unit tests ─────────────────────────────────────────────

@pytest.mark.unit
def test_sw_dynamics_step_finite(sw_dynamics):
    """Single step produces finite values."""
    state_dim = sw_dynamics.state_dim
    x0 = torch.randn(1, state_dim) * 0.1 + 1.0  # h around 1.0
    forcing = torch.randn(1, 2)
    x1 = sw_dynamics.step(x0, forcing)
    assert x1.shape == x0.shape
    assert x1.isfinite().all()


@pytest.mark.unit
def test_sw_dataset_shapes(sw_config):
    """Dataset windows have correct shapes."""
    from data.shallow_water import ShallowWaterDataset
    ds = ShallowWaterDataset(sw_config, scenario="S0")
    # Dataset generation should complete without error
    assert ds.config.Nx == 8
    assert ds.config.state_dim == 6 * 8 * 8


@pytest.mark.unit
def test_explained_variance_perfect():
    """EV = 1.0 for perfect reconstruction."""
    from evaluation.metrics import explained_variance
    rng = np.random.default_rng(0)
    truth = rng.standard_normal((100, 10))
    ev = explained_variance(truth, truth)
    np.testing.assert_allclose(ev, 1.0, atol=1e-10)


@pytest.mark.unit
def test_explained_variance_mean():
    """EV = 0 when analysis equals climatological mean."""
    from evaluation.metrics import explained_variance
    truth = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    analysis = np.full_like(truth, truth.mean(axis=0))
    ev = explained_variance(analysis, truth)
    np.testing.assert_allclose(ev, 0.0, atol=1e-10)


@pytest.mark.unit
def test_explained_variance_negative():
    """EV < 0 when analysis worse than climatological mean."""
    from evaluation.metrics import explained_variance
    truth = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    # analysis is constant at 100.0 — far from truth
    analysis = np.full_like(truth, 100.0)
    ev = explained_variance(analysis, truth)
    assert (ev < 0).all()


@pytest.mark.unit
def test_sw_component_metrics(sw_config):
    """Per-component metrics have correct structure."""
    from evaluation.metrics import compute_sw_component_metrics
    Nxy = sw_config.Nx * sw_config.Ny
    rng = np.random.default_rng(42)
    T = 5
    analysis = rng.standard_normal((T, 6 * Nxy))
    truth = rng.standard_normal((T, 6 * Nxy))
    result = compute_sw_component_metrics(analysis, truth, sw_config.Nx, sw_config.Ny)
    assert "ocean" in result
    assert "atmosphere" in result
    assert "h" in result["ocean"]
    assert "u" in result["ocean"]
    assert "v" in result["ocean"]
    assert "aggregate" in result["ocean"]
    assert "rmse" in result["ocean"]["h"]
    assert "ev" in result["ocean"]["h"]
    # EV should be a finite number
    assert np.isfinite(result["ocean"]["h"]["ev"])


@pytest.mark.unit
def test_validate_ev_targets():
    """validate_ev_targets correctly checks thresholds."""
    from evaluation.metrics import validate_ev_targets
    metrics = {
        "ocean": {"aggregate": {"ev": 0.96}},
        "atmosphere": {"aggregate": {"ev": 0.98}},
    }
    targets = {"ocean": 0.95, "atmosphere": 0.95}
    result = validate_ev_targets(metrics, targets, "S0")
    assert result["ocean"]["passed"] is True
    assert result["atmosphere"]["passed"] is True


@pytest.mark.unit
def test_validate_ev_targets_fail():
    """validate_ev_targets detects when targets not met."""
    from evaluation.metrics import validate_ev_targets
    metrics = {
        "ocean": {"aggregate": {"ev": 0.60}},
        "atmosphere": {"aggregate": {"ev": 0.90}},
    }
    targets = {"ocean": 0.70, "atmosphere": 0.85}
    result = validate_ev_targets(metrics, targets, "S1")
    assert result["ocean"]["passed"] is False
    assert result["atmosphere"]["passed"] is True


@pytest.mark.unit
def test_sw_config_hydra():
    """SW config loads from YAML defaults."""
    from data.shallow_water import ShallowWaterConfig
    config = ShallowWaterConfig()
    assert config.Nx == 64
    assert config.state_dim == 6 * 64 * 64
    assert config.land_mask_type == "none"
    assert config.f_cor == 0.1
    assert config.bickley_perturbation_mode == "random_balanced"


@pytest.mark.unit
def test_sw_factory():
    """get_dynamics() returns ShallowWaterDynamics for shallow_water type."""
    from models.dynamics import get_dynamics
    cfg = _DotDict(data=_DotDict(
        system="shallow_water",
        Nx=8, Ny=8, K=3, dt=0.01,
        tau0=0.0, f_cor=1.0, g1=1.0, g2=4.0,
        coupling=0.01, friction=0.0, viscosity=0.0001,
        land_mask_type="none",
    ))
    dyn = get_dynamics(cfg)
    assert type(dyn).__name__ == "ShallowWaterDynamics"


@pytest.mark.unit
def test_make_sw_obs_indices():
    """Obs indices map correctly to state vector."""
    from data.shallow_water import ShallowWaterConfig, make_sw_obs_indices
    Nx, Ny = 16, 16
    cfg = ShallowWaterConfig(
        Nx=Nx, Ny=Ny,
        obs_stride_ocean=8, obs_stride_atmos=4,
    )
    indices = make_sw_obs_indices(cfg)
    Nxy = Nx * Ny
    # Ocean layer: stride=8 -> (16/8)^2 = 4 points per variable
    # Atmosphere layer: stride=4 -> (16/4)^2 = 16 points per variable
    ocean_h_obs = len(indices[indices < Nxy])
    assert ocean_h_obs == 4
    atmos_h_obs = len(indices[(indices >= 3 * Nxy) & (indices < 4 * Nxy)])
    assert atmos_h_obs == 16


@pytest.mark.unit
def test_sw_sinusoidal_perturbation():
    """Sinusoidal perturbation seeds x-variation in initial condition."""
    from models.shallow_water_dynamics import ShallowWaterDynamics
    Nx, Ny = 16, 16
    dyn = ShallowWaterDynamics(Nx=Nx, Ny=Ny, f_cor=1.0, g1=1.0, g2=4.0)
    # Generate IC with sinusoidal perturbation
    state = dyn._init_bickley_jet(U=1.0, U2=0.6, H_ref=1.0,
                                   perturbation_mode="sinusoidal")
    Nxy = Nx * Ny
    # Extract u1 and reshape to (Nx, Ny)
    u1 = state[Nxy : 2 * Nxy].reshape(Nx, Ny)
    # Row means should vary across rows (x-direction)
    row_means = u1.mean(dim=1)
    x_variation = row_means.std().item()
    assert x_variation > 1e-6, f"x-variation too small: {x_variation:.8f}"
    # Columns should also have structure (y-direction from jet)
    col_means = u1.mean(dim=0)
    y_variation = col_means.std().item()
    assert y_variation > 0.01, f"y-variation too small: {y_variation:.8f}"


@pytest.mark.unit
def test_sw_random_perturbation():
    """Random perturbation produces IC but no systematic x-structure."""
    from models.shallow_water_dynamics import ShallowWaterDynamics
    Nx, Ny = 16, 16
    dyn = ShallowWaterDynamics(Nx=Nx, Ny=Ny, f_cor=1.0, g1=1.0, g2=4.0)
    state = dyn._init_bickley_jet(U=1.0, U2=0.6, H_ref=1.0,
                                   perturbation_mode="random")
    Nxy = Nx * Ny
    u1 = state[Nxy : 2 * Nxy].reshape(Nx, Ny)
    # Random pert: x-variation should be tiny (same noise added to all columns)
    row_means = u1.mean(dim=1)
    x_variation = row_means.std().item()
    # The jet is y-only, so row means should all be ~0 (mean of sech²)
    assert x_variation < 0.01, f"Random pert x-variation too large: {x_variation:.4f}"


# ── GPU tests (mark as slow) ────────────────────────────────────────

@pytest.mark.slow
def test_sw_dynamics_trajectory_stable(sw_dynamics):
    """Trajectory via generate_full_trajectory remains bounded."""
    Nxy = sw_dynamics.Nx * sw_dynamics.Ny
    traj, forcing = sw_dynamics.generate_full_trajectory(
        num_steps=200, seed=42,
    )
    assert traj.isfinite().all()
    h1 = traj[:, :Nxy]
    assert h1.min() > 0.0, f"Negative h1: {h1.min():.4f}"
    assert h1.max() < 20.0, f"h1 too large: {h1.max():.4f}"


@pytest.mark.slow
def test_sw_dynamics_mass_conservation(sw_dynamics):
    """Total mass approximately conserved (no land, no friction)."""
    Nxy = sw_dynamics.Nx * sw_dynamics.Ny
    state_dim = sw_dynamics.state_dim
    # Use zero friction for true conservation
    from models.shallow_water_dynamics import ShallowWaterDynamics
    dyn = ShallowWaterDynamics(
        Nx=sw_dynamics.Nx, Ny=sw_dynamics.Ny,
        friction=0.0, viscosity=0.0,
    )
    x0 = torch.randn(1, state_dim) * 0.1 + 1.0
    forcing = torch.randn(100, 2)
    traj = [x0]
    for t in range(100):
        traj.append(dyn.step(traj[-1], forcing[t:t + 1]))
    traj = torch.cat(traj, dim=0)
    total_mass = traj[:, :Nxy].sum(dim=1)
    # Mass should be roughly conserved (within 5%)
    rel_change = (total_mass - total_mass[0]).abs() / total_mass[0].abs()
    assert rel_change.max() < 0.05, f"Mass changed by {rel_change.max():.2%}"


@pytest.mark.slow
def test_sw_s0_ev_ocean(sw_s0_dataset, device):
    """S0 ocean EV >= 95% with Weak-4DVar (smoke test)."""
    pytest.skip("Requires GPU and full dataset — run via sbatch")


@pytest.mark.slow
def test_sw_s0_ev_atmosphere(sw_s0_dataset, device):
    """S0 atmosphere EV >= 95% with Weak-4DVar (smoke test)."""
    pytest.skip("Requires GPU and full dataset — run via sbatch")


@pytest.mark.slow
def test_sw_s1_ev_ocean(sw_s1_dataset, device):
    """S1 ocean EV >= 70% with Weak-4DVar (smoke test)."""
    pytest.skip("Requires GPU and full dataset — run via sbatch")


@pytest.mark.slow
def test_sw_s1_ev_atmosphere(sw_s1_dataset, device):
    """S1 atmosphere EV >= 85% with Weak-4DVar (smoke test)."""
    pytest.skip("Requires GPU and full dataset — run via sbatch")


@pytest.mark.slow
def test_sw_all_baselines_run(sw_s0_dataset, device):
    """All 4 DA methods complete without errors (smoke test)."""
    pytest.skip("Requires GPU and full dataset — run via sbatch")
