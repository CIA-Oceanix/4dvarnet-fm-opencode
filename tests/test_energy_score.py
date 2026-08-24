import numpy as np
import pytest
import torch

from evaluation.metrics import energy_score
from evaluation.baselines import EnKF, Strong4DVar, BaselineResult


class TestEnergyScoreMetric:
    def test_perfect_deterministic_ensemble_zero(self):
        T, D, N = 50, 3, 5
        truth = np.random.randn(T, D)
        ensemble = np.stack([truth] * N, axis=0)
        es = energy_score(ensemble, truth)
        assert es.shape == (D,)
        np.testing.assert_allclose(es, 0.0, atol=1e-10)

    def test_spread_only_positive(self):
        rng = np.random.RandomState(0)
        T, D, N = 50, 2, 10
        truth = rng.randn(T, D)
        ensemble = np.stack([truth + rng.randn(T, D) * 2.0 for _ in range(N)], axis=0)
        es = energy_score(ensemble, truth)
        assert es.shape == (D,)
        # A spread ensemble around the truth must have positive ES
        assert np.all(es > 0.0)

    def test_tighter_ensemble_lower_es(self):
        rng = np.random.RandomState(1)
        T, D, N = 60, 3, 10
        truth = rng.randn(T, D)
        wide = np.stack([truth + rng.randn(T, D) * 3.0 for _ in range(N)], axis=0)
        tight = np.stack([truth + rng.randn(T, D) * 0.3 for _ in range(N)], axis=0)
        es_wide = energy_score(wide, truth)
        es_tight = energy_score(tight, truth)
        assert np.all(es_tight < es_wide)

    def test_bimodal_es_lower_than_wide(self):
        """A properly sharp ensemble centered near the truth should beat a vague one."""
        rng = np.random.RandomState(2)
        T, D, N = 40, 2, 8
        truth = rng.randn(T, D)
        sharp = np.stack([truth + rng.randn(T, D) * 0.1 for _ in range(N)], axis=0)
        vague = rng.randn(N, T, D) * 5.0
        # Force vague to be centered on the truth mean so it's purely a spread issue
        vague += (truth[np.newaxis] - vague.mean(axis=0, keepdims=True))
        assert np.all(energy_score(sharp, truth) < energy_score(vague, truth))


class TestESAccumulator:
    def _acc(self, N, sd, T):
        from evaluation.baselines import _ESAccumulator
        return _ESAccumulator(T, sd, N)

    def test_matches_energy_score_stepwise(self):
        """Feeding ensembles timestep-by-timestep must reproduce energy_score."""
        rng = np.random.RandomState(7)
        N, T, D = 6, 20, 3
        ens = rng.randn(N, T, D)
        truth = rng.randn(T, D)

        acc = self._acc(N, D, T)
        for t in range(T):
            acc.step(torch.from_numpy(ens[:, t]), torch.from_numpy(truth[t]))
        np.testing.assert_allclose(acc.es(), energy_score(ens, truth), rtol=1e-12)

    def test_identical_members_equal_mae_any_n(self):
        rng = np.random.RandomState(3)
        N, T, D = 5, 15, 2
        traj = rng.randn(T, D)
        truth = rng.randn(T, D)
        acc = self._acc(N, D, T)
        for t in range(T):
            acc.step(torch.from_numpy(np.stack([traj[t]] * N)), torch.from_numpy(truth[t]))
        mae = np.mean(np.abs(traj - truth), axis=0)
        # Spread term vanishes -> ES == MAE of the (shared) trajectory
        np.testing.assert_allclose(acc.es(), mae, atol=1e-12)

    def test_single_member_mae_proxy(self):
        rng = np.random.RandomState(11)
        T, D = 12, 3
        traj = rng.randn(T, D)
        truth = rng.randn(T, D)
        acc = self._acc(1, D, T)
        for t in range(T):
            acc.step(torch.from_numpy(traj[t][None]), torch.from_numpy(truth[t]))
        np.testing.assert_allclose(acc.es(), np.mean(np.abs(traj - truth), axis=0), atol=1e-12)


class TestEnKFEnergyScore:
    def test_batch_reports_es_field(self, device):
        torch = pytest.importorskip("torch")
        from models.lorenz63_dynamics import Lorenz63Dynamics
        enkf = EnKF(N_ensemble=8, dt=0.01, device=device, dynamics=Lorenz63Dynamics(dt=0.01))
        T, D = 30, 3
        obs = torch.randn(T, D, device=device)
        mask = torch.zeros(T, dtype=torch.bool, device=device)
        mask[::3] = True
        force = torch.zeros(T, device=device)
        truth = torch.randn(T, D, device=device)
        result = enkf.assimilate(obs, mask, force, true_state=truth)
        assert isinstance(result, BaselineResult)
        assert result.es is not None
        assert result.es.shape == (D,)
        assert np.all(np.isfinite(result.es))

    def test_es_zero_when_truth_absent(self, device):
        torch = pytest.importorskip("torch")
        from models.lorenz63_dynamics import Lorenz63Dynamics
        enkf = EnKF(N_ensemble=8, dt=0.01, device=device, dynamics=Lorenz63Dynamics(dt=0.01))
        T, D = 20, 3
        obs = torch.randn(T, D, device=device)
        mask = torch.zeros(T, dtype=torch.bool, device=device)
        mask[::3] = True
        force = torch.zeros(T, device=device)
        result = enkf.assimilate(obs, mask, force)
        assert result.es is None


class TestStrong4DVarBatchES:
    def test_batch_reports_mae_es(self, device):
        pytest.importorskip("torch")
        from models.lorenz63_dynamics import Lorenz63Dynamics
        method = Strong4DVar(
            da_window_steps=5, max_iter=2, lr=0.1, dt=0.01,
            device=device, dynamics=Lorenz63Dynamics(dt=0.01),
        )
        B, T, D = 2, 10, 3
        obs = torch.randn(B, T, D, device=device)
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        mask[:, ::3] = True
        force = torch.zeros(B, T, device=device)
        truth = torch.randn(B, T, D, device=device)
        results = method.assimilate_batch(obs, mask, force, true_state=truth)
        assert len(results) == B
        for b, result in enumerate(results):
            assert result.es is not None
            assert result.es.shape == (D,)
            mae = np.mean(np.abs(result.trajectory - truth[b].cpu().numpy()), axis=0)
            np.testing.assert_allclose(result.es, mae, rtol=1e-5)

    def test_batch_es_none_when_truth_absent(self, device):
        pytest.importorskip("torch")
        from models.lorenz63_dynamics import Lorenz63Dynamics
        method = Strong4DVar(
            da_window_steps=5, max_iter=2, lr=0.1, dt=0.01,
            device=device, dynamics=Lorenz63Dynamics(dt=0.01),
        )
        B, T, D = 1, 5, 3
        obs = torch.randn(B, T, D, device=device)
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        mask[:, ::3] = True
        force = torch.zeros(B, T, device=device)
        results = method.assimilate_batch(obs, mask, force)
        assert all(r.es is None for r in results)
