"""Tests for MAOOAM dynamics and dataset."""

import sys
import pytest
import torch

sys.path.insert(0, ".")
from models.maooam_dynamics import MaooamDynamics, _count_atm_modes, _count_oc_modes
from data.maooam import MaooamConfig, MaooamDataset, make_maooam_obs_indices, make_maooam_obs_mask


class TestMaooamDynamics:
    def test_state_dim(self):
        d = MaooamDynamics(atm_nx=2, atm_ny=2, occ_nx=2, occ_ny=4)
        Natm = _count_atm_modes(2, 2)
        Noc = _count_oc_modes(2, 4)
        assert d.state_dim == 2 * Natm + 2 * Noc

    def test_generate_trajectory_shape(self):
        d = MaooamDynamics(dt=0.1, K=5)
        traj, forcing = d.generate_full_trajectory(500, seed=42, spinup_steps=500)
        assert traj.shape == (500, 36)
        assert forcing.shape[0] == 500

    def test_trajectory_stable(self):
        d = MaooamDynamics(dt=0.1, K=5)
        traj, _ = d.generate_full_trajectory(1000, seed=42, spinup_steps=1000)
        assert torch.isfinite(traj).all()
        assert traj.abs().max() < 10.0

    def test_step_matches_trajectory(self):
        d = MaooamDynamics(dt=0.1, K=5)
        traj, forcing = d.generate_full_trajectory(10, seed=42, spinup_steps=100)
        # Manual single step
        out = d.step(traj[0:1], forcing[0:1])
        # Should be close to traj[1]
        diff = (out - traj[1:2]).abs().max().item()
        assert diff < 1e-4, f"Step mismatch: {diff}"

    def test_spectral_to_physical(self):
        d = MaooamDynamics(dt=0.1, K=5)
        traj, _ = d.generate_full_trajectory(100, seed=42, spinup_steps=200)
        phys = d.spectral_to_physical(traj[50].numpy())
        assert "psi_upper" in phys
        assert "psi_oc" in phys
        assert phys["psi_upper"].ndim == 2
        assert phys["psi_oc"].ndim == 2


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

    def test_s1_scenario(self):
        cfg = MaooamConfig(spinup_steps=200, num_windows=2, K=10)
        ds = MaooamDataset(cfg, scenario="S1")
        assert len(ds) == 2
        assert torch.isfinite(ds[0]["true_state"]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
