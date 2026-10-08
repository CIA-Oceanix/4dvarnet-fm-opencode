import random
from collections import Counter

import numpy as np
import pytest
import torch

from data.qg import QGConfig, ensure_truth_only_cache
from data.qg_neural import (
    QGBatch,
    QGNeuralDataset,
    num_days,
    slice_days,
    window_starts,
    windowed_estimate,
)

CACHE = "/tmp/qg_neural_test_cache"


def _cfg(days: float = 4.0) -> QGConfig:
    return QGConfig(nx=8, window_days=days, spinup_years=0.02, num_windows=1,
                    obs_geometry="random_columns", cols_per_day=1, R_var=1e-12)


def _batch(b: int = 2, days: int = 30, d: int = 6) -> QGBatch:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(b, days, d, generator=g)
    return QGBatch(x, x + 1.0, torch.ones(b, days, d, dtype=torch.bool), torch.zeros(b, days, 2, 2),
                   x * 2.0, torch.full((b,), 1.5e4))


def _draws(power, n=20000):
    windows = ensure_truth_only_cache(_cfg(), 1, CACHE)
    ds = QGNeuralDataset(windows, _cfg(), on_the_fly_obs=True, cols_per_day_range=(3, 30),
                         cols_sampling="uniform_slots", cols_per_day_power=power)
    random.seed(0)
    return np.array([ds._draw_cols_per_day() for _ in range(n)])


def test_power_law_density_draw_has_median_six_at_power_one_and_a_half():
    k = _draws(1.5)
    assert k.min() >= 3 and k.max() <= 30
    assert np.median(k) == 6
    weights = np.arange(3, 31) ** -1.5
    assert np.mean(k == 3) == pytest.approx(weights[0] / weights.sum(), abs=0.01)


def test_density_draw_without_power_stays_uniform():
    counts = Counter(_draws(None))
    assert set(counts) == set(range(3, 31))
    assert max(counts.values()) / min(counts.values()) < 1.3


def test_power_needs_a_density_range():
    windows = ensure_truth_only_cache(_cfg(), 1, CACHE)
    with pytest.raises(ValueError, match="cols_per_day_range"):
        QGNeuralDataset(windows, _cfg(), on_the_fly_obs=True, cols_per_day_power=1.5)


def test_window_starts_cover_the_window_and_end_flush():
    assert window_starts(30, 15, 5) == [0, 5, 10, 15]
    assert window_starts(30, 15, 15) == [0, 15]
    assert window_starts(30, 15, 7) == [0, 7, 14, 15]
    assert window_starts(30, 30, 5) == [0]
    with pytest.raises(ValueError):
        window_starts(10, 15, 5)


def test_slice_days_cuts_every_per_day_field():
    b = slice_days(_batch(), 5, 20)
    assert b.T == 15 and b.obs.shape[1] == 15 and b.forcing.shape[1] == 15
    assert torch.equal(b.states_q, _batch().states_q[:, 5:20])


def test_windowed_estimate_is_exact_for_a_day_local_estimator():
    batch = _batch()
    calls = []

    def estimate(sub):
        calls.append(sub.T)
        return torch.stack([sub.obs * 3.0, sub.obs])

    out = windowed_estimate(estimate, batch, 15, 5)
    assert calls == [15] * 4
    assert out.shape == (2, *batch.obs.shape)
    assert torch.allclose(out[0], batch.obs * 3.0, atol=1e-6)


def test_windowed_estimate_weights_mid_sub_window_days_most():
    batch = _batch(b=1, days=6, d=1)

    def estimate(sub):
        return torch.full_like(sub.obs, float(sub.states[0, 0, 0] == batch.states[0, 0, 0]))

    out = windowed_estimate(estimate, batch, 4, 2)
    assert out[0, 0, 0] == 1.0 and out[0, -1, 0] == 0.0
    assert 0.0 < out[0, 2, 0] < 1.0


def test_windowed_estimate_is_the_plain_call_on_a_full_window():
    batch = _batch(days=15)
    assert torch.equal(windowed_estimate(lambda b: b.obs, batch, 15, 5), batch.obs)


def test_dataset_crops_tile_the_full_items():
    cfg = _cfg(days=4.0)
    windows = ensure_truth_only_cache(cfg, 1, CACHE)
    from data.qg_specwind_neural import with_fixed_obs
    windows = with_fixed_obs(windows, cfg)
    full = QGNeuralDataset(windows, cfg)
    tiled = QGNeuralDataset(windows, cfg, crop_days=2, crop_starts=[0, 2])
    assert len(tiled) == 2
    ref = full[0]
    for i, s in enumerate([0, 2]):
        item = tiled[i]
        for a, b in zip(item[:5], ref[:5]):
            assert torch.equal(a, b[s:s + 2])
    random.seed(1)
    crops = QGNeuralDataset(windows, cfg, crop_days=3)
    assert len(crops) == 1 and crops[0][0].shape[0] == 3


def test_crop_is_validated():
    cfg = _cfg(days=4.0)
    windows = ensure_truth_only_cache(cfg, 1, CACHE)
    with pytest.raises(ValueError):
        QGNeuralDataset(windows, cfg, crop_days=num_days(cfg) + 1)
    with pytest.raises(ValueError):
        QGNeuralDataset(windows, cfg, crop_days=2, crop_starts=[3])
    with pytest.raises(ValueError):
        QGNeuralDataset(windows, cfg, crop_starts=[0])


def test_load_run_weights_strips_the_lightning_prefix(tmp_path):
    from train_qg_neural import load_run_weights
    (tmp_path / "checkpoints").mkdir()
    torch.save({"state_dict": {"model.w": torch.ones(2)}}, tmp_path / "checkpoints" / "stage1_best.ckpt")
    assert list(load_run_weights(str(tmp_path))) == ["w"]
    with pytest.raises(FileNotFoundError):
        load_run_weights(str(tmp_path / "missing"))
