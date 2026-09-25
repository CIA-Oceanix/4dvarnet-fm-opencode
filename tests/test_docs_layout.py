"""docs/ layout contract (see docs/README.md).

Every doc under ``docs/plans/`` and ``docs/results/`` declares a ``**Status:**``
whose first word is in its folder's vocabulary, and every ``docs/...`` path cited
anywhere in the tracked tree exists. CHANGELOG history is exempt: it records
paths as they were.
"""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

VOCAB = {
    "plans/paper/superseded": {"SUPERSEDED"},
    "plans/paper": {"SCOPING", "SUPERSEDED"},
    "plans/analysis": {"PLAN", "DESIGN", "DRAFT", "CLOSED", "SUPERSEDED"},
    "plans/case_study": {"PROPOSAL", "SCOPING", "CLOSED", "SUPERSEDED"},
    "plans/tech": {"DESIGN", "DRAFT", "PROPOSAL", "SCOPING", "NOTES"},
    "results": {"RESULTS", "NOTES"},
}
STATUS = re.compile(r"^\*\*Status:\*\*\s*(?:\*\*)?([A-Z]+)")
CITED = re.compile(r"docs/[A-Za-z0-9_./-]+\.(?:md|tex|bib|pdf)")
TEXT_SUFFIXES = {".md", ".py", ".tex", ".yaml", ".yml", ".sh", ".sbatch", ".slurm", ".txt", ".toml", ".ini", ".cfg"}
DELIBERATELY_ABSENT = {
    "docs/plans/analysis/foo.md",
    "docs/phase_D_l96_sda.md",
    "docs/joint_estimation_design.md",
}


def _docs():
    return sorted(p for p in (DOCS / "plans").rglob("*.md")) + sorted((DOCS / "results").glob("*.md"))


def _vocab_for(path: Path) -> set[str]:
    rel = path.parent.relative_to(DOCS).as_posix()
    for folder in sorted(VOCAB, key=len, reverse=True):
        if rel == folder or rel.startswith(folder + "/"):
            return VOCAB[folder]
    raise AssertionError(f"{path} is not in a folder docs/README.md defines")


@pytest.mark.parametrize("doc", _docs(), ids=lambda p: p.relative_to(DOCS).as_posix())
def test_status_line_matches_folder(doc):
    head = doc.read_text(encoding="utf-8").splitlines()[:8]
    words = [m.group(1) for m in map(STATUS.match, head) if m]
    assert words, f"{doc.relative_to(ROOT)}: no **Status:** line in the first 8 lines"
    assert words[0] in _vocab_for(doc), (
        f"{doc.relative_to(ROOT)}: Status {words[0]!r} not in {sorted(_vocab_for(doc))}")


def test_no_docs_outside_the_layout():
    allowed_top = {"README.md", "CONTRIBUTING.md"}
    stray = [p.relative_to(DOCS).as_posix() for p in DOCS.glob("*.md") if p.name not in allowed_top]
    stray += [p.relative_to(DOCS).as_posix() for p in DOCS.iterdir()
              if p.is_dir() and p.name not in {"plans", "results", "papers"}]
    assert not stray, f"outside the docs/ layout: {stray}"


def _tracked_text_files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    for name in out.splitlines():
        if name == "CHANGELOG.md" or name.startswith("CHANGELOG.d/"):
            continue
        p = ROOT / name
        if p.suffix in TEXT_SUFFIXES and p.is_file():
            yield p


def test_cited_doc_paths_exist():
    missing = []
    for p in _tracked_text_files():
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for cited in set(CITED.findall(text)):
            cited = cited.rstrip(".")
            if cited not in DELIBERATELY_ABSENT and not (ROOT / cited).exists():
                missing.append(f"{p.relative_to(ROOT)} -> {cited}")
    assert not missing, "cited docs/ paths that do not exist:\n" + "\n".join(sorted(missing))
