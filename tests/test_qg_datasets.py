import json
import os
import shutil

import numpy as np
import pytest
import torch

from data.qg_datasets import (
    SPLIT_IDS,
    QGDatasetSpec,
    QGGeneratedSplit,
    assemble_split,
    check_independence,
    design_factors,
    diversity_report,
    generate_windows,
    shard_indices,
    spec_with,
    split_dir,
    window_seed,
    write_shard,
)

TINY = QGDatasetSpec(name="tiny", n_train=8, n_val=4, n_test=4, nx=16, spinup_days=1.0,
                     lead_days=0.5, window_days=1.0, burnin_units=2.0)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("qgds"))
    for split in ("train", "val", "test"):
        for shard in range(2):
            write_shard(TINY, split, root, shard, 2, batch_size=3)
        assemble_split(TINY, split, root)
    yield root
    for split in ("test",):
        d = split_dir(root, TINY, split)
        for f in os.listdir(d):
            os.chmod(os.path.join(d, f), 0o644)
    shutil.rmtree(root, ignore_errors=True)


def test_window_seeds_are_disjoint_across_splits():
    seeds = {s: {window_seed(TINY, s, i) for i in range(TINY.n_windows(s))} for s in SPLIT_IDS}
    assert not (seeds["train"] & seeds["val"])
    assert not (seeds["train"] & seeds["test"])
    assert not (seeds["val"] & seeds["test"])


def test_test_split_does_not_depend_on_train_or_val_sizes():
    bigger = spec_with(TINY, n_train=20, n_val=9)
    for i in range(TINY.n_test):
        assert window_seed(TINY, "test", i) == window_seed(bigger, "test", i)
    fa, fb = design_factors(TINY, "test"), design_factors(bigger, "test")
    for k in fa:
        np.testing.assert_array_equal(fa[k], fb[k])
    assert TINY.split_hash("test") == bigger.split_hash("test")
    assert TINY.split_hash("train") != bigger.split_hash("train")
    a = generate_windows(TINY, "test", [0, 1])
    b = generate_windows(bigger, "test", [0, 1])
    torch.testing.assert_close(a["true_state"], b["true_state"], rtol=0, atol=0)


def test_factor_design_covers_ranges_and_calm_stratum():
    spec = spec_with(TINY, n_train=500)
    f = design_factors(spec, "train")
    assert f["rd"].min() >= spec.rd[0] and f["rd"].max() <= spec.rd[1]
    assert f["time_unit_days"].min() >= 30.0 and f["time_unit_days"].max() <= 90.0
    assert abs(float((f["level"] == 0).mean()) - spec.calm_fraction) < 0.01
    for j in range(f["unit"].shape[1]):
        counts = np.histogram(f["unit"][:, j], bins=10, range=(0, 1))[0]
        assert counts.min() == counts.max() == 50


def test_shard_indices_are_contiguous_and_cover_the_split():
    parts = [shard_indices(10, k, 3) for k in range(3)]
    assert sum(parts, []) == list(range(10))
    assert all(p == list(range(p[0], p[-1] + 1)) for p in parts)


def test_sharding_does_not_change_windows():
    one = generate_windows(TINY, "train", list(range(4)), batch_size=4)
    two = generate_windows(TINY, "train", [2, 3], batch_size=2)
    assert one["window_seed"][2:] == two["window_seed"]
    torch.testing.assert_close(one["wind_amplitudes"][2:], two["wind_amplitudes"], rtol=0, atol=0)
    torch.testing.assert_close(one["true_state"][2:], two["true_state"], rtol=1e-4, atol=1e-12)


def test_storage_resolution_per_split():
    tr = generate_windows(TINY, "train", [0])
    te = generate_windows(TINY, "test", [0])
    n = TINY.n_lead + TINY.n_window
    assert tr["true_state"].shape[1] == n // TINY.keep_every_trainval + 1
    assert te["true_state"].shape[1] == n + 1
    assert tr["wind_amplitudes"].shape[1] == n
    assert tr["window_start_frame"] == TINY.n_lead // TINY.keep_every_trainval


def test_manifest_and_loader(built):
    tr = QGGeneratedSplit(split_dir(built, TINY, "train"), purpose="train")
    assert len(tr) == TINY.n_train
    assert [w["index"] for w in tr.windows] == list(range(TINY.n_train))
    item = tr[5]
    assert item["true_state"].shape[-3:] == (2, 16, 16)
    psi = tr.streamfunction(5)
    assert psi.shape == item["true_state"].shape
    assert tr.curl_field(5).shape[-2:] == (16, 16)
    with open(os.path.join(split_dir(built, TINY, "train"), "manifest.json")) as fh:
        m = json.load(fh)
    assert m["split_hash"] == TINY.split_hash("train")


def test_test_split_is_read_only_and_refused_outside_test(built):
    d = split_dir(built, TINY, "test")
    assert not os.access(os.path.join(d, "manifest.json"), os.W_OK)
    with pytest.raises(PermissionError):
        QGGeneratedSplit(d, purpose="train")
    with pytest.raises(PermissionError):
        QGGeneratedSplit(d, purpose="eval")
    assert len(QGGeneratedSplit(d, purpose="test")) == TINY.n_test
    with pytest.raises(PermissionError):
        QGGeneratedSplit(split_dir(built, TINY, "val"), purpose="test")


def test_independence_checks_pass_for_clean_splits(built):
    tr = QGGeneratedSplit(split_dir(built, TINY, "train"), purpose="train")
    for split, purpose in (("val", "eval"), ("test", "test")):
        rep = check_independence(tr, QGGeneratedSplit(split_dir(built, TINY, split), purpose=purpose))
        assert rep["checks"]["seeds_disjoint"] and rep["checks"]["state_hashes_disjoint"]
        assert rep["checks"]["no_duplicate_factors"]


def test_independence_checks_catch_a_leaked_window(built, tmp_path):
    tr = QGGeneratedSplit(split_dir(built, TINY, "train"), purpose="train")
    leaked_dir = tmp_path / "val_leaked"
    shutil.copytree(split_dir(built, TINY, "val"), leaked_dir)
    rec = dict(tr.windows[3])
    torch.save({"true_state": tr[3]["true_state"].unsqueeze(0)}, leaked_dir / "leak.pt")
    with open(leaked_dir / "manifest.json") as fh:
        m = json.load(fh)
    m["windows"].append({**rec, "file": "leak.pt", "row": 0})
    with open(leaked_dir / "manifest.json", "w") as fh:
        json.dump(m, fh)
    rep = check_independence(tr, QGGeneratedSplit(str(leaked_dir), purpose="eval"))
    assert not rep["passed"]
    assert not rep["checks"]["seeds_disjoint"]
    assert not rep["checks"]["state_hashes_disjoint"]
    assert not rep["checks"]["no_duplicate_factors"]
    assert not rep["checks"]["window_start_q1_max_abs_corr_below_0.99"]


def test_diversity_report_summaries(built):
    splits = {"train": QGGeneratedSplit(split_dir(built, TINY, "train"), purpose="train"),
              "val": QGGeneratedSplit(split_dir(built, TINY, "val"), purpose="eval")}
    rep = diversity_report(splits)
    assert sum(rep["train"]["regime_counts"].values()) == TINY.n_train
    assert "ks_pvalue_vs_train" in rep["val"]
    calm = [w["factors"]["level"] == 0.0 for w in splits["train"].windows]
    assert all((c and v == 0.0) or (not c and v > 0.0) for c, v in zip(calm, rep["train"]["rms_curl"]))
