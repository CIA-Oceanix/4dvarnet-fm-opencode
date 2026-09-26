"""L96 report generators read their inputs only through reports/l96/_inputs.py."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reports" / "l96"))
import _inputs  # noqa: E402

WORKTREE_PATH = re.compile(r"4dvarnet-fm-[a-z0-9-]+/|/Odyssey/private/rfablet/Python/")
EXEMPT = ("reports/qg/",)


def _report_scripts():
    out = subprocess.run(["git", "ls-files", "reports/*.py"], cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout.split()
    return [p for p in out if not p.startswith(EXEMPT)]


@pytest.mark.parametrize("script", _report_scripts())
def test_no_worktree_or_absolute_repo_paths(script):
    hits = [f"{script}:{i}: {line.strip()}" for i, line in
            enumerate((ROOT / script).read_text(encoding="utf-8").splitlines(), 1) if WORKTREE_PATH.search(line)]
    assert not hits, ("report scripts must resolve inputs through reports/l96/_inputs.py, not worktree or "
                      "absolute repo paths:\n" + "\n".join(hits))


def test_roots_default_to_the_report_bundle(monkeypatch):
    monkeypatch.delenv(_inputs.ROOTS_ENV, raising=False)
    for report, names in _inputs.REPORTS.items():
        for name in names:
            assert _inputs.root(report, name) == _inputs.bundle(report) / name
    assert _inputs.bundle("p1_benchmark").is_relative_to(_inputs.shared())


def test_override_and_unknown_root(monkeypatch, tmp_path):
    monkeypatch.setenv(_inputs.ROOTS_ENV, json.dumps({"here": str(tmp_path)}))
    assert _inputs.root("p1_benchmark", "here") == tmp_path
    with pytest.raises(KeyError):
        _inputs.root("p1_benchmark", "bench")


def test_bundler_covers_every_report_root():
    sys.path.insert(0, str(ROOT / "scripts"))
    import bundle_report_inputs as b
    assert set(b.GENERATORS) == set(_inputs.REPORTS) == set(b.LEGACY)
    for report, names in _inputs.REPORTS.items():
        assert set(b.LEGACY[report]) == set(names)
        assert (ROOT / b.GENERATORS[report]).is_file()


def test_add_links_a_new_result_dir_and_records_it(monkeypatch, tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import bundle_report_inputs as b
    bundle = tmp_path / "bundle"
    (bundle / "here").mkdir(parents=True)
    (bundle / "MANIFEST.json").write_text(json.dumps({"roots": {"here": {"built_from": "x", "files": ["a/f.npz"]}}}))
    src = tmp_path / "run" / "ens30_gw20"
    src.mkdir(parents=True)
    for name in ("members_s0.npz", "sda_eval.json"):
        (src / name).write_text(name)
    monkeypatch.setattr(b._inputs, "bundle", lambda report: bundle)
    assert b.add("p1_benchmark", "here", src, "newrun/ens30_gw20", apply=False) == 0
    assert not (bundle / "here" / "newrun").exists()
    assert b.add("p1_benchmark", "here", src, "newrun/ens30_gw20", apply=True) == 0
    dst = bundle / "here" / "newrun" / "ens30_gw20" / "sda_eval.json"
    assert dst.samefile(src / "sda_eval.json")
    manifest = json.loads((bundle / "MANIFEST.json").read_text())
    assert "newrun/ens30_gw20/members_s0.npz" in manifest["roots"]["here"]["files"]
    assert "a/f.npz" in manifest["roots"]["here"]["files"] and manifest["added"][0]["dest"] == "newrun/ens30_gw20"
    assert b.add("p1_benchmark", "here", src, "newrun/ens30_gw20", apply=True) == 0
    with pytest.raises(KeyError):
        b.add("p1_benchmark", "bench", src, "x", apply=False)
