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


# ── Data-level tests (SW data pipeline) ─────────────────────────────

def test_sw_dataset_window_shapes(sw_config):
    """Dataset window dicts have the expected keys and shapes."""
    from data.shallow_water import ShallowWaterDataset
    ds = ShallowWaterDataset(sw_config, scenario="S0")
    assert len(ds) == sw_config.num_windows
    window = ds[0]
    assert set(window.keys()) == {"true_state", "obs", "obs_mask", "forcing"}
    state_dim = 6 * sw_config.Nx * sw_config.Ny
    assert window["true_state"].shape == (sw_config.K, state_dim)
    assert window["obs"].shape == (sw_config.K, state_dim)
    assert window["obs_mask"].shape == (state_dim,)
    assert window["forcing"].shape == (sw_config.K, 2)
    assert window["obs_mask"].dtype == torch.bool


def test_sw_obs_indices_unique_inrange(sw_config):
    """make_sw_obs_indices yields unique, in-range long indices."""
    from data.shallow_water import ShallowWaterConfig, make_sw_obs_indices
    cfg = ShallowWaterConfig(
        Nx=8, Ny=8, obs_stride_ocean=4, obs_stride_atmos=2,
    )
    indices = make_sw_obs_indices(cfg)
    assert indices.dtype == torch.long
    assert indices.numel() == len(torch.unique(indices))
    assert indices.min().item() >= 0
    assert indices.max().item() < cfg.state_dim


def test_make_sw_obs_indices_count(sw_config):
    """Observed-point counts match stride expectations for each layer."""
    from data.shallow_water import ShallowWaterConfig, make_sw_obs_indices
    Nx, Ny = 16, 16
    cfg = ShallowWaterConfig(
        Nx=Nx, Ny=Ny, obs_stride_ocean=8, obs_stride_atmos=4,
    )
    indices = make_sw_obs_indices(cfg)
    Nxy = Nx * Ny
    ocean_h_obs = len(indices[indices < Nxy])
    atmos_h_obs = len(indices[(indices >= 3 * Nxy) & (indices < 4 * Nxy)])
    assert ocean_h_obs == 4
    assert atmos_h_obs == 16


def test_sw_obs_noise_std_structure(sw_config):
    """make_sw_obs_noise_std length matches obs count and follows indices ordering."""
    from data.shallow_water import ShallowWaterConfig, make_sw_obs_noise_std, make_sw_obs_indices
    cfg = ShallowWaterConfig(
        Nx=8, Ny=8, obs_stride_ocean=4, obs_stride_atmos=2,
    )
    indices = make_sw_obs_indices(cfg)
    noise = make_sw_obs_noise_std(cfg)
    assert noise.shape == (indices.numel(),)
    assert (noise > 0).all()
    Nxy = cfg.Nx * cfg.Ny
    comp = indices // Nxy
    expected = torch.tensor([cfg.obs_state_stds[c] * cfg.obs_noise_pct for c in comp])
    torch.testing.assert_close(noise, expected, rtol=1e-6, atol=1e-7)


def test_sw_s0_reproducible(sw_config):
    """S0 dataset regenerated with the same seed is identical."""
    from data.shallow_water import ShallowWaterDataset, ShallowWaterConfig
    cfg_a = ShallowWaterConfig(**{**sw_config.__dict__, "seed": 7})
    cfg_b = ShallowWaterConfig(**{**sw_config.__dict__, "seed": 7})
    ds_a = ShallowWaterDataset(cfg_a, scenario="S0")
    ds_b = ShallowWaterDataset(cfg_b, scenario="S0")
    for a, b in zip(ds_a.windows, ds_b.windows):
        for key in ("true_state", "obs", "obs_mask", "forcing"):
            torch.testing.assert_close(a[key], b[key], equal_nan=True)


def test_sw_s1_differs_from_s0(sw_config):
    """S1 (perturbed jet) produces trajectories that differ from S0."""
    from data.shallow_water import ShallowWaterDataset
    ds_s0 = ShallowWaterDataset(sw_config, scenario="S0")
    ds_s1 = ShallowWaterDataset(sw_config, scenario="S1")
    assert ds_s0.config.seed == ds_s1.config.seed
    d = (ds_s1[0]["true_state"] - ds_s0[0]["true_state"]).abs().max().item()
    assert d > 0.0


def test_sw_config_defaults():
    """ShallowWaterConfig defaults describe the reference 64x64 grid."""
    from data.shallow_water import ShallowWaterConfig
    config = ShallowWaterConfig()
    assert config.Nx == 64
    assert config.state_dim == 6 * 64 * 64
    assert config.land_mask_type == "none"


# ── SW/EV metrics tests (not covered by tests/test_metrics.py) ──────

def test_explained_variance_perfect():
    """EV = 1.0 for a perfect reconstruction."""
    import numpy as np
    from evaluation.metrics import explained_variance
    rng = np.random.default_rng(0)
    truth = rng.standard_normal((100, 10))
    ev = explained_variance(truth, truth)
    np.testing.assert_allclose(ev, 1.0, atol=1e-10)


def test_explained_variance_mean():
    """EV = 0 when the analysis equals the climatological mean."""
    import numpy as np
    from evaluation.metrics import explained_variance
    truth = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    analysis = np.full_like(truth, truth.mean(axis=0))
    ev = explained_variance(analysis, truth)
    np.testing.assert_allclose(ev, 0.0, atol=1e-10)


def test_explained_variance_negative():
    """EV < 0 when the analysis is worse than the climatological mean."""
    import numpy as np
    from evaluation.metrics import explained_variance
    truth = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    analysis = np.full_like(truth, 100.0)
    ev = explained_variance(analysis, truth)
    assert (ev < 0).all()


def test_sw_component_metrics(sw_config):
    """Per-component metrics have a correct structure."""
    import numpy as np
    from evaluation.metrics import compute_sw_component_metrics
    Nxy = sw_config.Nx * sw_config.Ny
    rng = np.random.default_rng(42)
    T = 5
    analysis = rng.standard_normal((T, 6 * Nxy))
    truth = rng.standard_normal((T, 6 * Nxy))
    result = compute_sw_component_metrics(analysis, truth, sw_config.Nx, sw_config.Ny)
    assert "ocean" in result and "atmosphere" in result
    for group in ("ocean", "atmosphere"):
        for field in ("h", "u", "v"):
            assert field in result[group]
            assert "rmse" in result[group][field]
            assert "ev" in result[group][field]
        assert "aggregate" in result[group]
    assert np.isfinite(result["ocean"]["h"]["ev"])


def test_validate_ev_targets():
    """validate_ev_targets correctly reports when targets are met."""
    from evaluation.metrics import validate_ev_targets
    metrics = {
        "ocean": {"aggregate": {"ev": 0.96}},
        "atmosphere": {"aggregate": {"ev": 0.98}},
    }
    targets = {"ocean": 0.95, "atmosphere": 0.95}
    result = validate_ev_targets(metrics, targets, "S0")
    assert result["ocean"]["passed"] is True
    assert result["atmosphere"]["passed"] is True


def test_validate_ev_targets_fail():
    """validate_ev_targets detects when targets are not met."""
    from evaluation.metrics import validate_ev_targets
    metrics = {
        "ocean": {"aggregate": {"ev": 0.60}},
        "atmosphere": {"aggregate": {"ev": 0.90}},
    }
    targets = {"ocean": 0.70, "atmosphere": 0.85}
    result = validate_ev_targets(metrics, targets, "S1")
    assert result["ocean"]["passed"] is False
    assert result["atmosphere"]["passed"] is True


def test_dataconfig_to_shallow_water_config():
    """DataConfig.to_shallow_water_config() round-trips and applies SW defaults."""
    from conf.schema import DataConfig
    from data.shallow_water import ShallowWaterConfig
    dc = DataConfig(system="shallow_water", dt=0.1, num_windows=7, spinup_steps=10, seed=3)
    cfg = dc.to_shallow_water_config()
    assert isinstance(cfg, ShallowWaterConfig)
    assert cfg.dt == 0.1
    assert cfg.num_windows == 7
    assert cfg.spinup_steps == 10
    assert cfg.seed == 3
    assert cfg.tau0 == 0.0
    assert cfg.f_cor == 0.1
    assert cfg.g1 == 0.5
    assert cfg.g2 == 2.0
    assert cfg.coupling == 0.01
    assert cfg.Nx == 64
    assert cfg.Ny == 64
    assert cfg.obs_state_stds == (0.134, 0.069, 0.005, 0.036, 0.071, 0.001)
    assert cfg.state_dim == 6 * 64 * 64


def test_sw_yaml_composes():
    """config/case_study/shallow_water.yaml composes over the base config."""
    import hydra
    import os
    config_dir = os.path.join(os.path.dirname(__file__), "..", "config")
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose("lorenz63_default", overrides=["+case_study=shallow_water"])
    assert cfg.data.system == "shallow_water"
    assert cfg.data.dt == 0.1
    assert cfg.data.tau0 == 0.0
    assert cfg.data.g2 == 2.0
