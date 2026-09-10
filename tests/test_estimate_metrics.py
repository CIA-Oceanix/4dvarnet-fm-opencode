"""Unit tests for the additional joint metrics in ``evaluation/estimate_metrics.py``.

Covers ``nrmse_param`` (normalized parameter RMSE) and ``trajectory_forecast_skill``
(short-term forecast divergence between true- and estimated-parameter rollouts).
Uses the real L96 dynamics on CPU; small window counts to stay fast.
"""
import numpy as np
import pytest

from evaluation.estimate_metrics import (
    L96_PARAM_ORDER,
    evaluate_ensemble_estimates,
    nrmse_param,
    per_window_deterministic_crps,
    per_window_ensemble_crps,
    per_window_rmse_ev,
    trajectory_forecast_skill,
)
from evaluation.run_l96 import make_obs_j_indices
from models.lorenz96_dynamics import Lorenz96Dynamics

TRUE_PARAMS = np.array([8.0, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 0.1])  # F,c1,hx,eps,w1..w4


@pytest.fixture(scope="module")
def dyn():
    return Lorenz96Dynamics(
        dt=0.001, coupling_exponent=1.6, NO=8, J=4,
        h=1.0, hx=1.0, eps=0.1, fast_weights=[1.0, 1.0, 0.1, 0.1],
    )


@pytest.fixture(scope="module")
def batch(dyn):
    """A small batch (W=2) of x0/forcing/params for the forecast-skill metric."""
    traj, forcing = dyn.generate_full_trajectory(num_steps=400, seed=42, F=8.0)
    x0 = traj[0].numpy()
    forces = np.stack([forcing.numpy(), forcing.numpy()], axis=0)
    x0s = np.stack([x0, x0 + 0.1], axis=0)
    tp = np.tile(TRUE_PARAMS, (2, 1))
    return x0s, forces, tp


OBS_IDX = make_obs_j_indices(NO=8, J_truth=4, J_obs=2)


class TestNrmseParam:
    def test_identical_params_zero(self):
        res = nrmse_param(TRUE_PARAMS, TRUE_PARAMS)
        assert res["mean"] == pytest.approx(0.0, abs=1e-12)
        assert np.allclose(res["per_param"], 0.0, atol=1e-12)

    def test_scale_invariance_of_normalization(self):
        # Same relative error on a large (F) and small (eps) param should give
        # comparable NRMSE even though raw RMSE is dominated by F.
        # perturb F by 10% and eps by 10%
        pred = TRUE_PARAMS.copy()
        pred[0] = TRUE_PARAMS[0] * 1.1
        pred[3] = TRUE_PARAMS[3] * 1.1
        res = nrmse_param(pred, TRUE_PARAMS)
        # 10% error -> NRMSE ~0.10 for both (approx; mean over 8 params dilutes)
        assert res["per_param"][0] == pytest.approx(0.1, abs=1e-3)
        assert res["per_param"][3] == pytest.approx(0.1, abs=1e-3)

    def test_shape_and_mean(self):
        pred = TRUE_PARAMS + np.array([0.5, 0.5, 0.5, 0.05, 0.5, 0.5, 0.05, 0.05])
        res = nrmse_param(pred, TRUE_PARAMS)
        assert res["per_param"].shape == (8,)
        expected_mean = float(np.mean(res["per_param"]))
        assert res["mean"] == pytest.approx(expected_mean)

    def test_batch_axis(self):
        tp = np.tile(TRUE_PARAMS, (3, 1))
        pred = tp * 1.05
        res = nrmse_param(pred, tp)
        assert np.allclose(res["per_param"], 0.05, atol=1e-6)


class TestTrajectoryForecastSkill:
    def test_identical_params_perfect(self, dyn, batch):
        x0s, forces, tp = batch
        res = trajectory_forecast_skill(
            dyn, x0s, forces, tp, tp, n_steps=200, obs_var_indices=OBS_IDX)
        assert res["ev"]["mean"] == pytest.approx(1.0, abs=1e-6)
        assert res["rmse"]["mean"] == pytest.approx(0.0, abs=1e-9)

    def test_grouped_structure(self, dyn, batch):
        x0s, forces, tp = batch
        good = tp.copy()
        good[0, 0] = tp[0, 0] * 1.05
        res = trajectory_forecast_skill(
            dyn, x0s, forces, tp, good, n_steps=200, obs_var_indices=OBS_IDX)
        for key in ("slow", "obs_fast", "all_obs"):
            assert key in res["rmse"]["groups"]
            assert key in res["ev"]["groups"]

    def test_good_better_than_poor(self, dyn, batch):
        x0s, forces, tp = batch
        n_steps = 300
        good = tp.copy()
        good[:, :] = tp[:, :] * np.array([1.01, 1.02, 1.02, 1.01, 1.01, 1.01, 1.01, 1.01])
        poor = tp.copy()
        poor[:, :] = tp[:, :] * np.array([1.10, 0.90, 1.10, 1.20, 1.10, 0.90, 1.20, 0.80])
        res_good = trajectory_forecast_skill(
            dyn, x0s, forces, tp, good, n_steps=n_steps, obs_var_indices=OBS_IDX)
        res_poor = trajectory_forecast_skill(
            dyn, x0s, forces, tp, poor, n_steps=n_steps, obs_var_indices=OBS_IDX)
        assert res_good["ev"]["mean"] > res_poor["ev"]["mean"]
        assert res_good["rmse"]["mean"] < res_poor["rmse"]["mean"]

    def test_n_steps_recorded(self, dyn, batch):
        x0s, forces, tp = batch
        res = trajectory_forecast_skill(
            dyn, x0s, forces, tp, tp, n_steps=150, obs_var_indices=OBS_IDX)
        assert res["n_steps"] == 150


class TestPerWindowMetrics:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(0)

    def test_perfect_reconstruction_zero_error(self, rng):
        truth = rng.normal(size=(5, 20, 24))
        pw = per_window_rmse_ev(truth.copy(), truth)
        for group in ("slow", "obs_fast", "all_obs"):
            assert pw["rmse"][group]["mean"] == pytest.approx(0.0, abs=1e-8)
            assert pw["rmse"][group]["std"] == pytest.approx(0.0, abs=1e-8)
            assert pw["ev"][group]["mean"] == pytest.approx(1.0, abs=1e-6)

    def test_grouped_keys_present(self, rng):
        traj = rng.normal(size=(6, 15, 24))
        truth = rng.normal(size=(6, 15, 24))
        pw = per_window_rmse_ev(traj, truth)
        for group in ("slow", "obs_fast", "all_obs"):
            assert group in pw["rmse"] and {"mean", "std"} <= pw["rmse"][group].keys()
            assert group in pw["ev"] and {"mean", "std"} <= pw["ev"][group].keys()

    def test_per_window_crps_mean_matches_pooled_es(self, rng):
        # Equal-length windows -> mean of per-window CRPS must equal the
        # pooled Energy Score (evaluate_ensemble_estimates), since averaging
        # a per-window mean over equal-sized groups is the same as pooling.
        truth = rng.normal(size=(8, 10, 24))
        members = truth[:, :, :, None] + rng.normal(scale=0.3, size=(8, 10, 24, 12))
        pooled = evaluate_ensemble_estimates(members, truth)
        pw_crps = per_window_ensemble_crps(members, truth)
        for group in ("slow", "obs_fast", "all_obs"):
            assert pw_crps[group]["mean"] == pytest.approx(
                pooled["ensemble"]["es"]["groups"][group], abs=1e-6)
            assert pw_crps[group]["std"] >= 0.0

    def test_perfect_ensemble_zero_crps(self, rng):
        truth = rng.normal(size=(4, 10, 24))
        members = np.repeat(truth[:, :, :, None], 5, axis=-1)
        pw_crps = per_window_ensemble_crps(members, truth)
        for group in ("slow", "obs_fast", "all_obs"):
            assert pw_crps[group]["mean"] == pytest.approx(0.0, abs=1e-8)

    def test_deterministic_crps_equals_mae_and_matches_ensemble_n1(self, rng):
        traj = rng.normal(size=(6, 12, 24))
        truth = rng.normal(size=(6, 12, 24))
        det = per_window_deterministic_crps(traj, truth)
        ens = per_window_ensemble_crps(traj[:, :, :, None], truth)
        for group in ("slow", "obs_fast", "all_obs"):
            assert det[group]["mean"] == pytest.approx(ens[group]["mean"], abs=1e-6)
            assert det[group]["std"] == pytest.approx(ens[group]["std"], abs=1e-6)

    def test_deterministic_crps_zero_for_perfect(self, rng):
        truth = rng.normal(size=(4, 10, 24))
        det = per_window_deterministic_crps(truth.copy(), truth)
        for group in ("slow", "obs_fast", "all_obs"):
            assert det[group]["mean"] == pytest.approx(0.0, abs=1e-8)
