import numpy as np
import pytest
import torch

from data.qg import QGConfig
from data.qg_neural import psi_daily, steps_per_day
from evaluation.run_qg_specwind_da import FIELDS
from evaluation.run_qg_specwind_neural import bin_centres, daily_to_steps, score_window

SPD = 12


def test_bin_centres_are_the_mean_step_of_each_day():
    assert np.allclose(bin_centres(3, SPD), [5.5, 17.5, 29.5])


def test_cubic_reproduces_a_cubic_in_time_exactly():
    t = np.arange(30 * SPD, dtype=float)
    poly = lambda x: 2e-6 * x ** 3 - 1e-3 * x ** 2 + 0.5 * x - 3.0  # noqa: E731
    daily = poly(bin_centres(30, SPD))[:, None] * np.ones((1, 5))
    out = daily_to_steps(daily, SPD, len(t), "cubic")
    assert np.allclose(out, poly(t)[:, None], rtol=1e-9, atol=1e-8)


@pytest.mark.parametrize("kind", ["cubic", "linear"])
def test_constant_and_linear_series_are_kept(kind):
    n = 30 * SPD
    const = np.full((2, 30, 4), 7.0)
    assert np.allclose(daily_to_steps(const, SPD, n, kind), 7.0)
    lin = (0.3 * bin_centres(30, SPD) - 1.0)[:, None] * np.ones((1, 4))
    assert np.allclose(daily_to_steps(lin, SPD, n, kind), (0.3 * np.arange(n) - 1.0)[:, None])


def test_unknown_interpolation_is_refused():
    with pytest.raises(ValueError):
        daily_to_steps(np.zeros((30, 2)), SPD, 360, "nearest")


def _window(cfg: QGConfig, n_days: int = 3) -> dict:
    g = torch.Generator().manual_seed(0)
    q0 = torch.randn(2, cfg.ny * cfg.nx, generator=g) * 1e-5
    q0 = (q0 - q0.mean(1, keepdim=True)).reshape(-1)
    tp = {"U1": 0.0, "U2": 0.0, "rd": 4.0e4, "rek": 1e-7, "beta": cfg.beta}
    return {"true_state": q0.repeat(n_days * steps_per_day(cfg), 1), "true_params": tp}


def test_a_perfect_daily_estimate_of_a_steady_field_scores_one():
    cfg = QGConfig(nx=16)
    w = _window(cfg)
    psi = psi_daily(w, cfg).numpy()
    m = score_window(psi[None], w, cfg, torch.device("cpu"))
    for f in FIELDS:
        assert m[f"ev_da_{f}"] == pytest.approx(1.0, abs=1e-6)
        assert m[f"daily_ev_da_{f}"] == pytest.approx(1.0, abs=1e-6)
    assert m["crps"] < 1e-5 * float(w["true_state"].std())
    assert "spread_ratio_q1" not in m
    assert not any(k.startswith("ev_free") for k in m)


def test_ensemble_scores_carry_crps_and_spread():
    cfg = QGConfig(nx=16)
    w = _window(cfg)
    psi = psi_daily(w, cfg).numpy()
    rng = np.random.default_rng(0)
    members = psi[None] + rng.normal(scale=0.1 * psi.std(), size=(5, *psi.shape))
    m = score_window(members, w, cfg, torch.device("cpu"))
    assert m["crps"] > 0.0
    assert m["spread_ratio_q1"] > 0.0 and m["spread_ratio_q2"] > 0.0
    assert m["score"] < 1.0
