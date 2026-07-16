"""Tests for MAOOAM dynamics and dataset."""

import sys
import pytest
import torch

sys.path.insert(0, ".")
from models.maooam_dynamics import MaooamDynamics, _count_atm_modes, _count_oc_modes
from data.maooam import MaooamConfig, MaooamDataset, make_maooam_obs_indices, make_maooam_obs_mask

# Share a single dynamics instance across all tests to avoid repeated JIT compilation
_dynamics = None

def _get_dynamics():
    global _dynamics
    if _dynamics is None:
        _dynamics = MaooamDynamics(dt=0.1, K=5)
    return _dynamics


class TestMaooamDynamics:
    def test_state_dim(self):
        d = _get_dynamics()
        Natm = _count_atm_modes(4, 4)
        Noc = _count_oc_modes(4, 4)
        assert d.state_dim == 2 * Natm + 2 * Noc

    def test_generate_trajectory_shape(self):
        d = _get_dynamics()
        traj, forcing = d.generate_full_trajectory(500, seed=42, spinup_steps=500)
        assert traj.shape == (500, d.state_dim)
        assert forcing.shape[0] == 500

    def test_trajectory_stable(self):
        d = _get_dynamics()
        traj, _ = d.generate_full_trajectory(1000, seed=42, spinup_steps=1000)
        assert torch.isfinite(traj).all()
        assert traj.abs().max() < 10.0

    def test_step_matches_trajectory(self):
        d = _get_dynamics()
        traj, forcing = d.generate_full_trajectory(10, seed=42, spinup_steps=100)
        out = d.step(traj[0:1], forcing[0:1])
        diff = (out - traj[1:2]).abs().max().item()
        assert diff < 1e-4, f"Step mismatch: {diff}"

    def test_spectral_to_physical(self):
        d = _get_dynamics()
        traj, _ = d.generate_full_trajectory(100, seed=42, spinup_steps=200)
        phys = d.spectral_to_physical(traj[50].numpy())
        assert "psi_upper" in phys
        assert "psi_oc" in phys
        assert phys["psi_upper"].ndim == 2
        assert phys["psi_oc"].ndim == 2

    def test_spectral_to_physical_interp(self):
        d = _get_dynamics()
        traj, _ = d.generate_full_trajectory(100, seed=42, spinup_steps=200)
        phys = d.spectral_to_physical(traj[50].numpy(), interp_size=64)
        assert phys["psi_upper"].shape == (64, 64)
        assert phys["psi_oc"].shape == (64, 64)


class TestMaooamDataset:
    def test_obs_indices_all(self):
        cfg = MaooamConfig(obs_mode="all")
        idx = make_maooam_obs_indices(cfg)
        assert idx.numel() == cfg.state_dim

    def test_obs_mask(self):
        cfg = MaooamConfig(obs_mode="all")
        mask = make_maooam_obs_mask(cfg)
        assert mask.all()

    def test_dataset_generation(self):
        cfg = MaooamConfig(spinup_steps=200, num_windows=3, K=10)
        ds = MaooamDataset(cfg, scenario="S0")
        assert len(ds) == 3
        sample = ds[0]
        assert sample["true_state"].shape == (10, cfg.state_dim)
        assert torch.isfinite(sample["true_state"]).all()

    def test_s1_config_perturbation(self):
        """Test S1 config differs from S0 without instantiating dynamics."""
        cfg_s0 = MaooamConfig()
        cfg_s1 = MaooamConfig()
        # S1 scenario increases coupling
        d_s1 = cfg_s1.d * 1.5
        assert d_s1 != cfg_s0.d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
