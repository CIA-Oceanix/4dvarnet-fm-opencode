"""Tests for evaluation.archive -- artifact resolution for archived L96 runs.

These are pure-logic tests against a synthetic tree (tmp_path), deliberately
independent of the real ``experiments/`` directory: that directory is gitignored
machine-local data, so a test that required it would pass here and fail in CI.
The one test that does touch the real tree skips when it is absent.
"""
import json

import pytest

from evaluation import archive


@pytest.fixture
def fake_tree(tmp_path, monkeypatch):
    """experiments/ with a canonical archive dir and a legacy flat dir."""
    exp = tmp_path / "experiments"
    (exp / "l96" / "runA" / "checkpoints").mkdir(parents=True)
    (exp / "runB").mkdir(parents=True)
    monkeypatch.setattr(archive, "ROOT", tmp_path)
    monkeypatch.setattr(archive, "EXPERIMENTS", exp)
    monkeypatch.setattr(archive, "ARCHIVE", exp / "l96")
    return exp


class TestResolution:
    def test_canonical_preferred_over_legacy(self, fake_tree):
        (fake_tree / "l96" / "runA" / "resolved_config.yaml").write_text("a: 1")
        (fake_tree / "runA").mkdir()
        (fake_tree / "runA" / "resolved_config.yaml").write_text("a: 2")
        got = archive.resolve_config("runA")
        assert got == fake_tree / "l96" / "runA" / "resolved_config.yaml"

    def test_falls_back_to_legacy(self, fake_tree):
        (fake_tree / "runB" / "estimates_s0.npz").write_bytes(b"x")
        assert archive.resolve_estimates("runB", "s0") == fake_tree / "runB" / "estimates_s0.npz"

    def test_artifacts_resolve_independently(self, fake_tree):
        """A run's artifacts may legitimately straddle both layouts -- the
        2026-09-10 consolidation moved configs but not the estimate arrays."""
        (fake_tree / "l96" / "runA" / "resolved_config.yaml").write_text("a: 1")
        (fake_tree / "runA").mkdir()
        (fake_tree / "runA" / "estimates_s0.npz").write_bytes(b"x")
        assert archive.resolve_config("runA").parent.parent.name == "l96"
        assert archive.resolve_estimates("runA", "s0").parent.name == "runA"
        assert archive.resolve_estimates("runA", "s0").parent.parent.name == "experiments"

    def test_missing_raises_with_both_paths_named(self, fake_tree):
        with pytest.raises(FileNotFoundError) as e:
            archive.resolve_estimates("nope", "s0")
        msg = str(e.value)
        assert "l96/nope" in msg and "experiments/nope" in msg

    def test_missing_returns_none_when_not_required(self, fake_tree):
        assert archive.resolve_estimates("nope", "s0", required=False) is None

    def test_nested_run_names_supported(self, fake_tree):
        """ens30 entries look like 'L3_vanilla_cfm_s0s1/ens30_no10'."""
        d = fake_tree / "l96" / "L3/ens30_no10"
        d.mkdir(parents=True)
        (d / "estimates_s0.npz").write_bytes(b"x")
        assert archive.resolve_estimates("L3/ens30_no10", "s0") == d / "estimates_s0.npz"

    def test_bad_case_rejected(self, fake_tree):
        with pytest.raises(ValueError):
            archive.resolve_estimates("runA", "s2")


class TestCheckpointResolution:
    def test_prefers_requested_then_falls_back(self, fake_tree):
        ck = fake_tree / "l96" / "runA" / "checkpoints"
        (ck / "stage1_best.ckpt").write_bytes(b"x")
        assert archive.resolve_checkpoint("runA").name == "stage1_best.ckpt"
        (ck / "stage1.pt").write_bytes(b"x")
        assert archive.resolve_checkpoint("runA").name == "stage1.pt"
        assert archive.resolve_checkpoint("runA", prefer="stage1_best.ckpt").name == "stage1_best.ckpt"


class TestNormStats:
    def test_config_path_wins(self, fake_tree):
        stats = fake_tree / "custom_stats.pt"
        stats.write_bytes(b"x")
        (fake_tree / archive.DEFAULT_NORM_STATS).write_bytes(b"y")

        class Cfg:
            data = {"norm_stats_path": "experiments/custom_stats.pt"}
        assert archive.resolve_norm_stats(Cfg()) == stats

    def test_falls_back_to_shared_default(self, fake_tree):
        shared = fake_tree / archive.DEFAULT_NORM_STATS
        shared.write_bytes(b"y")
        assert archive.resolve_norm_stats(None) == shared

    def test_run_local_stats_preferred_over_shared(self, fake_tree):
        (fake_tree / archive.DEFAULT_NORM_STATS).write_bytes(b"y")
        local = fake_tree / "l96" / "runA" / archive.DEFAULT_NORM_STATS
        local.write_bytes(b"z")
        assert archive.resolve_norm_stats(None, name="runA") == local

    def test_absent_returns_none_or_raises(self, fake_tree):
        assert archive.resolve_norm_stats(None) is None
        with pytest.raises(FileNotFoundError):
            archive.resolve_norm_stats(None, required=True)

    def test_never_derived_from_checkpoint_depth(self, fake_tree):
        """Regression: the old code did
        ``Path(ckpt).resolve().parents[2] / "l96_norm_stats_obsj2.pt"``, which
        silently pointed into experiments/l96/ once archived runs gained a
        directory level. Resolution must not depend on how deep the run sits."""
        shared = fake_tree / archive.DEFAULT_NORM_STATS
        shared.write_bytes(b"y")
        deep = fake_tree / "l96" / "a/b/c/runDeep"
        deep.mkdir(parents=True)
        assert archive.resolve_norm_stats(None, name="a/b/c/runDeep") == shared


class TestAudit:
    def test_config_not_required_by_default(self, fake_tree):
        """Runs predating PR #171 have no resolved_config.yaml but are still
        scoreable, so the default audit must not flag them."""
        (fake_tree / "runB" / "estimates_s0.npz").write_bytes(b"x")
        (fake_tree / "runB" / "estimates_s1.npz").write_bytes(b"x")
        assert archive.missing_artifacts("runB") == []
        assert archive.missing_artifacts("runB", need_config=True) == ["resolved_config.yaml"]

    def test_reports_each_missing_case(self, fake_tree):
        (fake_tree / "runB" / "estimates_s0.npz").write_bytes(b"x")
        assert archive.missing_artifacts("runB") == ["estimates_s1.npz"]

    def test_audit_only_lists_incomplete(self, fake_tree):
        (fake_tree / "runB" / "estimates_s0.npz").write_bytes(b"x")
        (fake_tree / "runB" / "estimates_s1.npz").write_bytes(b"x")
        out = archive.audit(["runB", "missingRun"])
        assert set(out) == {"missingRun"}


class TestManifest:
    def test_roundtrip(self, fake_tree):
        payload = {"run": "runA", "artifacts": {"estimates_s0.npz": {"bytes": 1}}}
        archive.write_manifest("runA", payload)
        assert archive.read_manifest("runA") == payload

    def test_read_absent_is_none(self, fake_tree):
        assert archive.read_manifest("runA") is None

    def test_written_where_the_run_resolves(self, fake_tree):
        archive.write_manifest("runA", {"run": "runA"})
        written = fake_tree / "l96" / "runA" / "manifest.json"
        assert json.loads(written.read_text())["run"] == "runA"


class TestQGLayout:
    """The QG archive is the same resolver at a different set of conventions.

    Kept in one module rather than two because the conventions are the part that
    differs -- where the weights sit, which cases exist, which stats file --
    and duplicating the resolver per case study is how the two drift apart.
    """

    @pytest.fixture
    def qg_tree(self, tmp_path, monkeypatch):
        exp = tmp_path / "experiments"
        (exp / "qg" / "runQ" / "checkpoints").mkdir(parents=True)
        monkeypatch.setattr(archive, "ROOT", tmp_path)
        monkeypatch.setattr(archive, "EXPERIMENTS", exp)
        return exp

    def test_archive_dir_is_per_system(self, qg_tree):
        assert archive.QG.archive_dir == qg_tree / "qg"
        assert archive.L96.archive_dir == qg_tree / "l96"

    def test_finished_run_prefers_the_bare_state_dict(self, qg_tree):
        run = qg_tree / "qg" / "runQ"
        (run / "checkpoints" / "stage1_best.ckpt").write_bytes(b"x")
        (run / "stage1_best.pt").write_bytes(b"x")
        assert archive.QG.resolve_checkpoint("runQ").name == "stage1_best.pt"

    def test_interrupted_run_falls_back_to_lightnings_own(self, qg_tree):
        (qg_tree / "qg" / "runQ" / "checkpoints" / "stage1_last.ckpt").write_bytes(b"x")
        assert archive.QG.resolve_checkpoint("runQ").name == "stage1_last.ckpt"

    def test_only_s0_estimates_exist(self, qg_tree):
        """QG runs evaluate S0 in-training; the S0/S1 table comes from a
        separate cross-scenario evaluation, not from per-run arrays."""
        with pytest.raises(ValueError):
            archive.QG.resolve_estimates("runQ", "s1", required=False)

    def test_stats_kept_with_the_archive_are_found(self, qg_tree):
        """The breakage this fixed: checkpoints reachable from master, and the
        normalization they are meaningless without reachable only from the
        worktree that trained them."""
        stats = qg_tree / "qg" / "qg_psi_norm_stats.pt"
        stats.write_bytes(b"x")
        assert archive.QG.resolve_norm_stats(None) == stats

    def test_stats_key_selects_which_file(self, qg_tree):
        (qg_tree / "custom_forcing.pt").write_bytes(b"x")

        class Cfg:
            data = {"norm_stats_path": "experiments/psi.pt",
                    "forcing_norm_stats_path": "experiments/custom_forcing.pt"}
        got = archive.QG.resolve_norm_stats(Cfg(), key="forcing_norm_stats_path")
        assert got == qg_tree / "custom_forcing.pt"

    def test_l96_module_functions_still_resolve_l96(self, qg_tree):
        """The module-level API stayed L96's, so no existing caller changed."""
        (qg_tree / "l96" / "runL").mkdir(parents=True)
        (qg_tree / "l96" / "runL" / "estimates_s0.npz").write_bytes(b"x")
        assert archive.resolve_estimates("runL", "s0").parent.parent.name == "l96"
