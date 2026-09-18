"""Internal consistency of the L96 consolidated report's declarations.

Pure code inspection -- no `experiments/` data, so this runs in CI where that
gitignored directory does not exist.

These guard the drift that made the report unreproducible: between PR #183 and
PR #204 the checked-in `l96_consolidated_benchmark.md` gained seven methods that
the generator had never heard of, because rows were added to the markdown by
hand. `docs/scoping/README.md` already forbids that ("fix the generator **and**
the checked-in output together"); nothing enforced it.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "reports/l96/generate_l96_consolidated_report.py"
OUT = ROOT / "reports/l96/outputs/l96_consolidated_benchmark.md"


@pytest.fixture(scope="module")
def src() -> str:
    return GEN.read_text()


def _block(src: str, pattern: str) -> str:
    m = re.search(pattern, src, re.S)
    assert m, f"could not find {pattern!r} in the generator"
    return m.group(1)


def neural_dirs(src: str) -> list[str]:
    return re.findall(r'"([^"]+)"', _block(src, r"NEURAL_EXP_DIRS = \[(.*?)\n\]"))


def scheme_ids(src: str) -> list[str]:
    blk = _block(src, r"SCHEME_DESCRIPTIONS: list\[tuple\[str, str, str\]\] = \[(.*?)\n\]\n")
    return re.findall(r'\("([^"]+)",\s*"', blk)


def monai_rows(src: str) -> list[str]:
    return re.findall(r'"([^"]+)"', _block(src, r"MONAI_ROWS = \[(.*?)\n\]"))


def deterministic_methods(src: str) -> list[str]:
    return re.findall(
        r'"([^"]+)"',
        _block(src, r"DETERMINISTIC_METHODS: frozenset\[str\] = frozenset\(\{(.*?)\}\)"))


def short_names(src: str) -> dict[str, str]:
    blk = _block(src, r"MONAI_SHORT_NAMES = \{(.*?)\n    \}")
    return dict(re.findall(r'"([^"]+)":\s*"([^"]+)"', blk))


class TestDeclarationsAgree:
    def test_every_method_has_a_description(self, src):
        missing = [d for d in neural_dirs(src) if d not in scheme_ids(src)]
        assert not missing, f"methods with no SCHEME_DESCRIPTIONS entry: {missing}"

    def test_every_method_has_a_short_name(self, src):
        missing = [d for d in neural_dirs(src) if d not in short_names(src)]
        assert not missing, f"methods with no MONAI_SHORT_NAMES entry: {missing}"

    def test_no_orphan_descriptions(self, src):
        da = {"Strong-4DVar", "Weak-4DVar", "EnKF", "ETKF"}
        orphans = [i for i in scheme_ids(src) if i not in neural_dirs(src) and i not in da]
        assert not orphans, f"descriptions for methods no longer listed: {orphans}"

    def test_no_duplicate_methods(self, src):
        dirs = neural_dirs(src)
        dupes = {d for d in dirs if dirs.count(d) > 1}
        assert not dupes, f"duplicated entries in NEURAL_EXP_DIRS: {dupes}"

    def test_short_names_are_unique(self, src):
        names = list(short_names(src).values())
        dupes = {n for n in names if names.count(n) > 1}
        assert not dupes, f"two methods share a display label: {dupes}"


class TestOverrideMapsReferenceRealMethods:
    def test_estimate_filename_overrides_are_listed_methods(self, src):
        blk = _block(src, r"ESTIMATE_FILENAMES: dict\[str, str\] = \{(.*?)\n\}")
        for name in re.findall(r'"([^"]+)":\s*\n?\s*"', blk):
            assert name in neural_dirs(src), f"{name} has a filename override but is not a method"

    def test_unavailable_methods_are_listed_methods(self, src):
        blk = _block(src, r"UNAVAILABLE_METHODS: frozenset\[str\] = frozenset\(\{(.*?)\}\)")
        for name in re.findall(r'"([^"]+)"', blk):
            assert name in neural_dirs(src), f"{name} marked unavailable but is not a method"

    def test_filename_overrides_carry_a_case_placeholder(self, src):
        blk = _block(src, r"ESTIMATE_FILENAMES: dict\[str, str\] = \{(.*?)\n\}")
        for tmpl in re.findall(r':\s*\n?\s*"([^"]+)"', blk):
            assert "{case}" in tmpl, f"override {tmpl!r} cannot vary by case"


@pytest.mark.skipif(not OUT.exists(), reason="checked-in report not present")
class TestPublishedTableMatchesGenerator:
    def test_every_published_row_is_a_declared_method(self, src):
        """The check that would have caught the #188-#204 drift.

        Every labelled row in the published metric tables must correspond to a
        method the generator knows, otherwise the markdown has been edited by
        hand and regenerating it would silently delete results.
        """
        labels = set(short_names(src).values()) | {
            "Strong-4DVar", "Weak-4DVar", "EnKF", "ETKF", "Free forecast", "Obs"}
        published = set()
        for line in OUT.read_text().splitlines():
            m = re.match(r"^\|\s*([^|]+?)\s*\|\s*(0\.\d+|—|n/a)\s*\|", line)
            if m:
                published.add(m.group(1))
        unknown = sorted(published - labels)
        assert not unknown, (
            "rows in the published table with no generator path (hand-edited?): "
            f"{unknown}")


class TestParallelMethodListsAgree:
    """The report keeps several method lists that must not drift apart.

    `NEURAL_EXP_DIRS` drives the pooled tables; `MONAI_ROWS` drives the
    per-window ones. A method added to only the first silently appears in three
    of the six metric tables -- which is exactly what happened to the seven
    methods ported on 2026-09-18, and was noticed by eye rather than by test.
    """

    def test_monai_rows_are_declared_methods(self, src):
        unknown = [r for r in monai_rows(src) if r not in neural_dirs(src)]
        assert not unknown, f"MONAI_ROWS entries that are not methods: {unknown}"

    def test_every_method_appears_in_per_window_tables(self, src):
        missing = [d for d in neural_dirs(src) if d not in monai_rows(src)]
        assert not missing, (
            "methods absent from MONAI_ROWS, so they would be missing from the "
            f"per-window tables: {missing}")

    def test_deterministic_methods_are_declared(self, src):
        da = {"Strong-4DVar", "Weak-4DVar", "EnKF", "ETKF"}
        unknown = [m for m in deterministic_methods(src)
                   if m not in neural_dirs(src) and m not in da]
        assert not unknown, f"DETERMINISTIC_METHODS entries that are not methods: {unknown}"

    def test_no_n1_proxy_fallback_remains(self, src):
        """The ES/CRPS columns must not mix two scoring formulas.

        A proper ensemble score credits spread; the N=1 MAE proxy cannot. When
        both appeared in one column the two rows that had real members looked
        better partly by convention, which is exactly the apples-to-apples
        failure this table exists to avoid.
        """
        assert "N1_ES_METHODS" not in src
        assert "per_window_deterministic_crps" not in src
