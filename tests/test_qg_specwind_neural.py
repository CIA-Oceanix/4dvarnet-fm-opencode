import os
import shutil
from types import SimpleNamespace

import pytest
import torch

from data.qg import QGConfig
from data.qg_datasets import (
    QGDatasetSpec,
    assemble_split,
    generate_windows,
    split_dir,
    write_shard,
)
from data.qg_neural import QGNeuralDataset, qg_collate
from data.qg_specwind_neural import (
    REGEN_BASE_INDEX,
    RegenerateTrainWindows,
    SpecWindTrainSource,
    check_compatible,
    legacy_windows,
    materialize_split,
    regeneration_rounds_disjoint,
    train_norm_stats,
    with_fixed_obs,
)

TINY = QGDatasetSpec(name="tiny_g5", n_train=4, n_val=3, n_test=3, nx=16, spinup_days=1.0,
                     lead_days=0.5, window_days=1.0, burnin_units=2.0)
CFG = QGConfig(nx=16, window_days=1.0, init_lead_days=0.5, obs_geometry="random_columns",
               cols_per_day=2, obs_noise_std_frac=0.01, init_lag_days=0.25)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("g5"))
    for split in ("train", "val", "test"):
        write_shard(TINY, split, root, 0, 1, batch_size=2)
        assemble_split(TINY, split, root)
    yield root
    d = split_dir(root, TINY, "test")
    for f in os.listdir(d):
        os.chmod(os.path.join(d, f), 0o644)
    shutil.rmtree(root, ignore_errors=True)


def test_spec_and_config_compatibility_is_checked():
    check_compatible(TINY, CFG)
    with pytest.raises(ValueError):
        check_compatible(TINY, QGConfig(nx=32, window_days=1.0, init_lead_days=0.5))


def test_legacy_layout_matches_the_qg_convention():
    res = generate_windows(TINY, "train", [0, 1], keep_every=1)
    ws = legacy_windows(res, TINY, with_wind_curl=True)
    lead = TINY.n_lead + 1
    frames = res["true_state"][0].reshape(res["true_state"].shape[1], -1)
    assert ws[0]["true_state"].shape == (TINY.n_window, 2 * 16 * 16)
    assert ws[0]["init_lead_truth"].shape == (lead, 2 * 16 * 16)
    torch.testing.assert_close(ws[0]["true_state"][0], frames[lead])
    torch.testing.assert_close(ws[0]["init_lead_truth"][-1], frames[lead - 1])
    assert ws[0]["wind_curl"].shape == (TINY.n_window, 16, 16)
    assert set(ws[0]["true_params"]) == {"U1", "rd", "rek", "beta", "U2"}
    with pytest.raises(ValueError):
        legacy_windows(generate_windows(TINY, "train", [0]), TINY)


@pytest.mark.parametrize("cond_mode", ["none", "true"])
def test_qg_neural_dataset_consumes_regenerated_windows(cond_mode):
    src = SpecWindTrainSource(TINY, 3, batch_size=3, with_wind_curl=(cond_mode == "true"))
    ws = src.draw(0)
    stats = train_norm_stats(ws, CFG, with_forcing=(cond_mode == "true"))
    kw = {}
    if cond_mode == "true":
        kw = {"param_norm_stats": {"mean": torch.zeros(3), "std": torch.ones(3)},
              "forcing_norm_stats": stats["forcing"]}
    ds = QGNeuralDataset(ws, CFG, stats["psi"], on_the_fly_obs=True, cond_mode=cond_mode, **kw)
    batch = qg_collate([ds[i] for i in range(len(ds))])
    days = int(CFG.window_days)
    assert batch.states.shape[:2] == (3, days)
    assert torch.isfinite(batch.states).all()
    assert torch.isfinite(batch.forcing).all()


def test_regeneration_rounds_are_fresh_disjoint_and_deterministic():
    src = SpecWindTrainSource(TINY, 2, batch_size=2)
    assert src.indices(0)[0] == REGEN_BASE_INDEX
    assert regeneration_rounds_disjoint(src, 5, stored_n=TINY.n_train)
    a, b, c = src.draw(0), src.draw(0), src.draw(1)
    torch.testing.assert_close(a[0]["true_state"], b[0]["true_state"], rtol=0, atol=0)
    assert not torch.equal(a[0]["true_state"], c[0]["true_state"])
    assert src.factors(0)["rd"][0] != src.factors(1)["rd"][0]


def test_val_materializes_at_full_resolution_and_matches_stored_frames(built):
    windows, rep = materialize_split(TINY, "val", built)
    assert rep["regenerated"] and len(windows) == TINY.n_val
    assert rep["max_abs_diff"] == 0.0
    assert windows[0]["true_state"].shape[0] == TINY.n_window


def test_test_split_is_read_from_full_resolution_storage(built):
    windows, rep = materialize_split(TINY, "test", built)
    assert not rep["regenerated"] and len(windows) == TINY.n_test
    fixed = with_fixed_obs(windows, CFG)
    assert all(k in fixed[0] for k in ("obs", "obs_mask", "init_state"))
    again = with_fixed_obs(windows, CFG)
    torch.testing.assert_close(fixed[0]["obs"], again[0]["obs"], equal_nan=True)


def test_callback_swaps_windows_every_k_epochs(tmp_path):
    src = SpecWindTrainSource(TINY, 2, batch_size=2)
    ds = SimpleNamespace(windows=src.draw(0))
    first = ds.windows
    cb = RegenerateTrainWindows(ds, src, every=2, log_path=str(tmp_path / "regen.jsonl"))
    cb.on_train_epoch_start(SimpleNamespace(current_epoch=1), None)
    assert ds.windows is first
    cb.on_train_epoch_start(SimpleNamespace(current_epoch=2), None)
    assert ds.windows is not first and len(ds.windows) == 2
    assert (tmp_path / "regen.jsonl").read_text().count("\n") == 1


def _eval_args(root, spec_name="tiny_g5"):
    return SimpleNamespace(specwind_spec=spec_name, specwind_root=root, regen_windows=2,
                           regen_every=1, regen_batch_size=2, eval_only="stage1_best.pt")


def test_eval_only_reloads_the_stats_saved_by_training(built, tmp_path, monkeypatch):
    import train_qg_neural
    from data.normalization import save_norm_stats
    from data.qg_datasets import SPECS

    monkeypatch.setitem(SPECS, "tiny_g5", TINY)
    exp = str(tmp_path)
    psi = {"mean": torch.tensor([1.0, 2.0]), "std": torch.tensor([3.0, 4.0])}
    save_norm_stats(os.path.join(exp, "specwind_psi_norm_stats.pt"), psi)
    save_norm_stats(os.path.join(exp, "specwind_forcing_norm_stats.pt"),
                    {"mean": torch.tensor([0.5]), "std": torch.tensor([2.0])})
    save_norm_stats(os.path.join(exp, "specwind_param_norm_stats.pt"),
                    {"mean": torch.zeros(3), "std": torch.ones(3)})
    out = train_qg_neural._build_specwind_data(
        _eval_args(built), CFG, exp, "cpu", "true", False, None, 1.5, None, None, None, True,
        os.path.join(exp, "specwind_psi_norm_stats.pt"))
    test_windows, train_ds, val_ds, callback, norm, forcing_norm, param_norm = out
    assert train_ds is None and callback is None and len(test_windows) == TINY.n_test
    torch.testing.assert_close(norm["std"], psi["std"])
    torch.testing.assert_close(forcing_norm["mean"], torch.tensor([0.5]))
    assert param_norm is not None


def test_eval_only_fails_loudly_without_saved_stats(built, tmp_path, monkeypatch):
    import train_qg_neural
    from data.qg_datasets import SPECS

    monkeypatch.setitem(SPECS, "tiny_g5", TINY)
    with pytest.raises(FileNotFoundError):
        train_qg_neural._build_specwind_data(
            _eval_args(built), CFG, str(tmp_path), "cpu", "none", False, None, 1.5, None, None,
            None, True, os.path.join(str(tmp_path), "specwind_psi_norm_stats.pt"))
