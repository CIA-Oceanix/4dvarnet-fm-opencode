"""data.train_cache: cached L96 train/val windows (make_l96_s0_s1_trainval_cached)."""
import pytest
import torch

from data.lorenz96 import (Lorenz96Config, _trainval_cache_meta, make_l96_s0_s1_trainval,
                           make_l96_s0_s1_trainval_cached)

KW = dict(num_train_windows=2, num_val_windows=1, num_test_windows=1, param_noise=0.2,
          bias_range=(0.0, 0.2), train_forcing_state_bias=0.1)


def _cfg(**over):
    return Lorenz96Config(T_max=0.2, spinup_steps=50, window_spacing=300, **over)


def test_meta_changes_with_settings():
    assert _trainval_cache_meta(_cfg(), KW) != _trainval_cache_meta(_cfg(seed=7), KW)
    assert _trainval_cache_meta(_cfg(), KW) != _trainval_cache_meta(_cfg(), {**KW, "num_train_windows": 3})
    assert _trainval_cache_meta(_cfg(), KW) == _trainval_cache_meta(_cfg(), {**KW, "cached_datasets": {"x": 1}})


def test_mismatched_cache_is_refused(tmp_path):
    path = tmp_path / "trainval.pt"
    torch.save({"meta": _trainval_cache_meta(_cfg(seed=7), KW), "train": [], "val": []}, path)
    with pytest.raises(ValueError, match="different settings"):
        make_l96_s0_s1_trainval_cached(_cfg(), train_cache=str(path), **KW)


@pytest.mark.slow
def test_cache_roundtrip_reproduces_generation(tmp_path):
    path = tmp_path / "trainval.pt"
    first = make_l96_s0_s1_trainval_cached(_cfg(), train_cache=str(path), **KW)
    assert path.exists() and not list(tmp_path.glob("*.tmp*"))
    second = make_l96_s0_s1_trainval_cached(_cfg(), train_cache=str(path), **KW)
    fresh = make_l96_s0_s1_trainval(_cfg(), **KW)
    for split in ("train", "val"):
        assert len(second[split]) == len(fresh[split])
        for a, b, c in zip(first[split].windows, second[split].windows, fresh[split].windows):
            assert torch.equal(a["true_state"], b["true_state"]) and torch.equal(b["true_state"], c["true_state"])
            assert torch.equal(b["obs"].nan_to_num(), c["obs"].nan_to_num())
            assert b.get("F_da") == c.get("F_da")
