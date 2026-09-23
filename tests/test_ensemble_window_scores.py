"""ensemble_window_scores / save_members_or_scores must reproduce the P1
benchmark's ensemble metrics without storing the members."""
import sys
from pathlib import Path

import numpy as np
import pytest

from evaluation.estimate_metrics import _groups_from_per_window, ensemble_window_scores, save_members_or_scores

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reports" / "l96"))
from generate_p1_l96_benchmark import generative  # noqa: E402


def _ensemble(seed=0, W=5, T=40, D=24, M=7):
    g = np.random.default_rng(seed)
    truth = g.normal(size=(W, T, D))
    members = truth[..., None] + g.normal(scale=0.7, size=(W, T, D, M)) + 0.2
    return members.astype(np.float32), truth.astype(np.float32)


def test_scores_match_p1_generative(tmp_path):
    members, truth = _ensemble()
    ref_path = save_members_or_scores(tmp_path, "s0", members, truth, "full")
    ref = generative(Path(ref_path))
    s = np.load(save_members_or_scores(tmp_path, "s0", members, truth, "scores"))
    for key in ("rmse", "crps", "spread"):
        np.testing.assert_allclose(_groups_from_per_window(s[key])["all_obs"], ref[key], rtol=1e-12)
    assert int(s["n_members"]) == members.shape[-1]


def test_crps_of_identical_members_is_mae():
    members, truth = _ensemble(M=4)
    same = np.repeat(members[..., :1], 4, axis=-1)
    s = ensemble_window_scores(same, truth)
    np.testing.assert_allclose(s["crps"], np.abs(same[..., 0] - truth).mean(axis=1), rtol=1e-6)
    np.testing.assert_allclose(s["spread"], 0.0, atol=1e-12)


def test_unknown_mode_raises(tmp_path):
    members, truth = _ensemble()
    with pytest.raises(ValueError):
        save_members_or_scores(tmp_path, "s0", members, truth, "bogus")
