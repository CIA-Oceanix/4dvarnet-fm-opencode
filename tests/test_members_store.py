from pathlib import Path

import pytest

from evaluation import members_store

ROOT = Path(__file__).resolve().parents[1]
WRITERS = [
    "eval_neural_l96.py",
    "eval_sda_l96.py",
    "eval_sda_mean_hybrid_l96.py",
    "eval_sda_fdv1_hybrid_l96.py",
    "eval_sda_directunet_hybrid_l96.py",
    "eval_fdv1_fdv1cfm_hybrid_l96.py",
]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    monkeypatch.delenv(members_store.KEEP_ENV, raising=False)
    monkeypatch.setenv(members_store.ROOT_ENV, str(tmp_path / "node_tmp"))
    return tmp_path


def test_default_goes_to_node_local_root(clean_env):
    out = clean_env / "experiments" / "run" / "ens30_no10"
    p = members_store.members_path(out, "s0")
    assert p.name == "members_s0.npz"
    assert p.is_relative_to(clean_env / "node_tmp")
    assert p.parent.is_dir()
    assert str(out.resolve()).lstrip("/") in str(p)


def test_keep_flag_writes_next_to_output(clean_env):
    out = clean_env / "run"
    assert members_store.members_path(out, "s1", keep=True) == out / "members_s1.npz"


@pytest.mark.parametrize("value,kept", [("1", True), ("true", True), ("0", False), ("", False), ("false", False)])
def test_keep_env(clean_env, monkeypatch, value, kept):
    monkeypatch.setenv(members_store.KEEP_ENV, value)
    out = clean_env / "run"
    assert (members_store.members_path(out, "s0") == out / "members_s0.npz") is kept


def test_default_root_is_per_job_under_tmp(monkeypatch):
    monkeypatch.delenv(members_store.ROOT_ENV, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    root = members_store.node_local_root()
    assert root.parts[:2] == ("/", "tmp")
    assert root.name == "members_12345"


@pytest.mark.parametrize("script", WRITERS)
def test_writers_route_through_members_store(script):
    src = (ROOT / script).read_text()
    assert 'output_path.parent / f"members_{case}.npz"' not in src
    assert "members_store.members_path(" in src
    assert '"--keep-members"' in src
