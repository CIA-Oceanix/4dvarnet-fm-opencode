"""Tests for the L96 DA obs-count sweep helpers (eval_da_obs_count_l96.py)."""
import numpy as np
import pytest
import torch

from eval_da_obs_count_l96 import (
    build_cell,
    observe_at_times,
    per_window_metrics,
    random_obs_times,
)
from evaluation.run_l96 import make_obs_j_indices

IDX = make_obs_j_indices(8, 4, 2)


def test_random_obs_times_first_step_always_observed():
    rng = np.random.default_rng(0)
    for n in (1, 2, 10, 300):
        t = random_obs_times(3000, n, rng)
        assert t[0] == 0 and len(t) == n
        assert len(np.unique(t)) == n and np.all(np.diff(t) > 0)
        assert t.max() < 3000


def test_random_obs_times_rejects_bad_count():
    with pytest.raises(ValueError):
        random_obs_times(10, 11, np.random.default_rng(0))


def test_observe_at_times_noise_and_mask():
    truth = torch.randn(3000, 40)
    times = random_obs_times(3000, 1000, np.random.default_rng(1))
    obs, mask = observe_at_times(truth, times, IDX, 0.5, np.random.default_rng(2))
    assert obs.shape == (3000, 24) and mask.sum() == 1000
    assert torch.isnan(obs[~mask]).all() and not torch.isnan(obs[mask]).any()
    resid = obs[mask] - truth[:, list(IDX)][mask]
    assert abs(resid.var().item() - 0.5) < 0.03


def _fake_datasets(n=20):
    g = torch.Generator().manual_seed(0)
    mk = lambda: [{"true_state": torch.randn(300, 40, generator=g),  # noqa: E731
                   "obs": torch.zeros(300, 24), "obs_mask": torch.zeros(300, dtype=torch.bool)}
                  for _ in range(n)]
    return {"test_s0": mk(), "test_s1": mk()}


def test_build_cell_is_deterministic_and_draws_differ():
    ds = _fake_datasets()
    a = build_cell(ds, [0, 5], "7", 3, IDX)
    b = build_cell(ds, [0, 5], "7", 3, IDX)
    assert len(a["test_s0"]) == 6
    for x, y in zip(a["test_s0"], b["test_s0"]):
        assert torch.equal(x["obs_mask"], y["obs_mask"])
    masks = [w["obs_mask"] for w in a["test_s0"][:3]]
    assert not torch.equal(masks[0], masks[1])
    assert all(m.sum() == 7 and m[0] for m in masks)
    assert not torch.equal(a["test_s0"][0]["obs_mask"], a["test_s1"][0]["obs_mask"])
    assert ds["test_s0"][0]["obs_mask"].sum() == 0


def test_build_cell_reg30_keeps_cached_obs():
    ds = _fake_datasets()
    cell = build_cell(ds, [0, 5], "reg30", 3, IDX)
    assert len(cell["test_s1"]) == 2
    assert cell["test_s1"][1]["obs"] is ds["test_s1"][5]["obs"]


def test_per_window_metrics_indexes_full_state_and_matches_formulas():
    idx = np.array(IDX)
    truth = np.random.default_rng(0).standard_normal((4, 50, 24))
    traj_full = np.zeros((4, 50, 40))
    traj_full[..., idx] = truth + 0.5
    var_full = np.full((4, 50, 40), 0.25)
    m = per_window_metrics(traj_full, truth, var_full, idx)
    np.testing.assert_allclose(m["rmse"]["all_obs"], 0.5)
    np.testing.assert_allclose(m["mae"]["slow"], 0.5)
    np.testing.assert_allclose(m["spread"]["obs_fast"], 0.5)
    assert "spread" not in per_window_metrics(traj_full, truth, None, idx)


def test_expand_batch_params_repeats_scalars_and_per_window_vectors():
    from evaluation.baselines import _expand_batch_params
    fw = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = _expand_batch_params({"F": torch.tensor([8.0, 9.0]), "fast_weights": fw, "h": 1.0}, 2, 3)
    assert torch.equal(out["F"], torch.tensor([8.0, 8.0, 8.0, 9.0, 9.0, 9.0]))
    assert out["fast_weights"].shape == (6, 2)
    assert torch.equal(out["fast_weights"][3], fw[1])
    assert out["h"] == 1.0


def test_batched_etkf_accepts_per_window_fast_weights():
    from evaluation.baselines import ETKF, ObsOperator
    from models.lorenz96_dynamics import Lorenz96Dynamics
    torch.manual_seed(0)
    T, B = 20, 2
    etkf = ETKF(N_ensemble=5, dt=0.001, inflation=2.0, dynamics=Lorenz96Dynamics(dt=0.001),
                obs_operator=ObsOperator(40, IDX), NO=8, J=4)
    obs = torch.full((B, T, 24), float("nan"))
    mask = torch.zeros(B, T, dtype=torch.bool)
    obs[:, [0, 7, 15]] = torch.randn(B, 3, 24)
    mask[:, [0, 7, 15]] = True
    fw = torch.tensor([[1.0, 1.0, 0.1, 0.1], [1.1, 0.9, 0.12, 0.08]])
    kw = {k: torch.full((B,), v) for k, v in dict(F=8.0, c1=1.0, h=1.0, hx=1.0, eps=0.1).items()}
    res = etkf.assimilate_batch(obs, mask, torch.zeros(B, T), torch.randn(B, T, 40), fast_weights=fw, **kw)
    assert len(res) == B and all(np.isfinite(r.trajectory).all() for r in res)
