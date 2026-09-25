"""Opt-in ETKF/EnKF member storage (``store_ensemble``) and the rank-histogram probe
(reports/l96/probe_rank_histograms.py)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from evaluation.baselines import EnKF, ETKF, ObsOperator
from models.lorenz96_dynamics import Lorenz96Dynamics

_spec = importlib.util.spec_from_file_location(
    "probe_rank_histograms", Path(__file__).resolve().parents[1] / "reports" / "l96" / "probe_rank_histograms.py")
prh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prh)

NO, J, WINDOW, DT, B = 2, 4, 20, 0.01, 2
STATE_DIM = NO + NO * J


def _build(cls, store: bool):
    indices = list(range(NO)) + [NO + k * J for k in range(NO)]
    obs_op = ObsOperator(STATE_DIM, indices)
    dyn = Lorenz96Dynamics(dt=DT, NO=NO, J=J, h=1.0, coupling_exponent=1.6, clip_range=50.0)
    m = cls(N_ensemble=10, inflation=1.0, NO=NO, J=J, dt=DT, dynamics=dyn, obs_operator=obs_op)
    if store:
        m.store_ensemble = True
    return m, obs_op


def _batch(obs_op):
    torch.manual_seed(0)
    n_obs = len(obs_op.indices)
    truth = torch.randn(B, WINDOW, STATE_DIM) * 0.5
    obs = torch.zeros(B, WINDOW, n_obs)
    mask = torch.zeros(B, WINDOW, dtype=torch.bool)
    for t in range(0, WINDOW, 5):
        obs[:, t] = truth[:, t][:, obs_op.indices] + 0.1 * torch.randn(B, n_obs)
        mask[:, t] = True
    return truth, obs, mask, torch.zeros(B, WINDOW)


@pytest.mark.parametrize("cls", [ETKF, EnKF])
def test_store_ensemble_keeps_members_whose_mean_is_the_analysis(cls):
    m, obs_op = _build(cls, store=True)
    truth, obs, mask, forcing = _batch(obs_op)
    torch.manual_seed(1)
    res = m.assimilate_batch(obs, mask, forcing, truth, F=8.0, c1=1.0, h=1.0, hx=1.0, eps=0.1)
    for r in res:
        assert r.ensemble.shape == (10, WINDOW, STATE_DIM)
        assert np.allclose(r.ensemble.mean(0), r.trajectory, atol=1e-4)
        assert np.abs(r.ensemble).sum() > 0


@pytest.mark.parametrize("cls", [ETKF, EnKF])
def test_store_ensemble_does_not_change_the_analysis(cls):
    outs = []
    for store in (False, True):
        m, obs_op = _build(cls, store=store)
        truth, obs, mask, forcing = _batch(obs_op)
        torch.manual_seed(1)
        outs.append(m.assimilate_batch(obs, mask, forcing, truth, F=8.0, c1=1.0, h=1.0, hx=1.0, eps=0.1))
    for a, b in zip(*outs):
        assert np.array_equal(a.trajectory, b.trajectory)
    assert all(np.abs(r.ensemble).sum() == 0 for r in outs[0])


def _write(tmp_path, members, truth, name):
    p = tmp_path / f"{name}.npz"
    np.savez(p, members=members.astype(np.float32), truth=truth.astype(np.float32))
    return p


def _synthetic(rng, W=4, T=200, D=24, M=30, spread=1.0, skew=False):
    mu = rng.normal(size=(W, T, D))
    if skew:
        truth = mu + (rng.exponential(1.0, size=(W, T, D)) - 1.0)
    else:
        truth = mu + rng.normal(size=(W, T, D))
    members = mu[..., None] + spread * rng.normal(size=(W, T, D, M))
    return members, truth


def test_calibrated_ensemble_is_flat(tmp_path):
    rng = np.random.default_rng(0)
    r = prh.analyse("cal", _write(tmp_path, *_synthetic(rng), "cal"), seed=0)["all_obs"]
    assert r["ri"] < 0.05 and abs(r["extreme_ratio"] - 1) < 0.1 and abs(r["rank_bias"]) < 0.01
    assert abs(r["spread_over_rmse"] - 1) < 0.05


def test_under_dispersion_is_u_shaped_and_removed_by_the_spread_correction(tmp_path):
    rng = np.random.default_rng(1)
    r = prh.analyse("under", _write(tmp_path, *_synthetic(rng, spread=0.5), "under"), seed=0)["all_obs"]
    assert r["extreme_ratio"] > 3 and r["ri"] > 0.3
    assert r["ri_shape"] < 0.05 and abs(r["extreme_ratio_shape"] - 1) < 0.15


def test_skewed_truth_survives_the_spread_correction(tmp_path):
    rng = np.random.default_rng(2)
    r = prh.analyse("skew", _write(tmp_path, *_synthetic(rng, skew=True), "skew"), seed=0)["all_obs"]
    assert r["ri_shape"] > 0.08


def test_ties_are_broken_uniformly():
    rng = np.random.default_rng(3)
    members = np.zeros((1, 1000, 1, 9))
    truth = np.zeros((1, 1000, 1))
    counts = np.bincount(prh.ranks(members, truth, rng).ravel(), minlength=10)
    assert counts.min() > 50


def test_mean_bias_is_a_ramp_removed_by_debiasing(tmp_path):
    rng = np.random.default_rng(4)
    members, truth = _synthetic(rng)
    r = prh.analyse("bias", _write(tmp_path, members, truth + 0.8, "bias"), seed=0)["all_obs"]
    assert r["rank_bias"] > 0.15 and r["ri_shape"] > 0.3
    assert r["ri_debiased"] < 0.06 and abs(r["rank_bias_debiased"]) < 0.02
