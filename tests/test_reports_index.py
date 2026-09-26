"""reports/README.md must index every checked-in report (see its header)."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "reports" / "README.md"
STATUSES = {"CURRENT", "FROZEN", "STALE", "SUPERSEDED"}
ROW = re.compile(r"^\| `([^`/]+\.md)` \|(.*)\|\s*$")


def _index_rows() -> dict[str, list[str]]:
    rows = {}
    for line in INDEX.read_text(encoding="utf-8").splitlines():
        m = ROW.match(line)
        if m:
            rows[m.group(1)] = [c.strip() for c in m.group(2).split("|")]
    return rows


def _reports() -> list[str]:
    out = subprocess.run(["git", "ls-files", "reports/*/outputs/*.md"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split()
    return [p for p in out if p.count("/") == 3]


def test_every_report_is_indexed():
    indexed = set(_index_rows())
    missing = [p for p in _reports() if Path(p).name not in indexed]
    assert not missing, f"not in reports/README.md: {missing}"


def test_every_indexed_report_exists():
    names = {Path(p).name for p in _reports()}
    stale = [n for n in _index_rows() if n not in names]
    assert not stale, f"indexed in reports/README.md but not a report: {stale}"


def test_rows_have_a_known_status():
    bad = {}
    for name, cells in _index_rows().items():
        m = re.search(r"\*\*([A-Z]+)\*\*", cells[1]) if len(cells) >= 4 else None
        if not m or m.group(1) not in STATUSES:
            bad[name] = cells[1] if len(cells) > 1 else cells
    assert not bad, f"rows without a status in {sorted(STATUSES)}: {bad}"


def test_listed_generators_exist():
    cited = set(re.findall(r"`(reports/[A-Za-z0-9_./-]+\.py)`", INDEX.read_text(encoding="utf-8")))
    missing = sorted(p for p in cited if not (ROOT / p).is_file())
    assert not missing, f"generators listed in reports/README.md that do not exist: {missing}"


FRESHNESS_EXEMPT_CASES = {"qg", "l63"}
EXPERIMENT_DIR = ROOT / "config" / "experiment"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def _load_yaml(text: str):
    import yaml
    return yaml.safe_load(text)


def _defaults_closure(config: Path, seen: set[Path]) -> None:
    if config in seen or not config.is_file():
        return
    seen.add(config)
    for entry in (_load_yaml(config.read_text(encoding="utf-8")) or {}).get("defaults", []) or []:
        if isinstance(entry, str) and entry.startswith("/"):
            _defaults_closure(ROOT / "config" / f"{entry[1:]}.yaml", seen)


def _config_deps(generator: Path) -> list[Path]:
    src = generator.read_text(encoding="utf-8")
    direct = {c for c in EXPERIMENT_DIR.glob("*.yaml")
              if re.search(rf"(?<![A-Za-z0-9_]){re.escape(c.stem)}(?![A-Za-z0-9])", src)}
    direct |= {ROOT / m for m in re.findall(r"config/[A-Za-z0-9_/]+\.yaml", src) if (ROOT / m).is_file()}
    closure: set[Path] = set()
    for c in direct:
        _defaults_closure(c, closure)
    return sorted(closure)


def _current_reports_with_generators():
    for name, cells in _index_rows().items():
        m = re.search(r"\*\*([A-Z]+)\*\*", cells[1]) if len(cells) >= 4 else None
        gens = re.findall(r"`(reports/[A-Za-z0-9_./-]+\.py)`", cells[-1]) if cells else []
        if not m or m.group(1) != "CURRENT" or not gens:
            continue
        case = gens[0].split("/")[1]
        if case in FRESHNESS_EXEMPT_CASES:
            continue
        yield ROOT / "reports" / case / "outputs" / name, ROOT / gens[0]


def test_current_reports_are_newer_than_their_model_configs():
    if _git("rev-parse", "--is-shallow-repository").strip() == "true":
        import pytest
        pytest.skip("needs full git history (CI checks out with fetch-depth: 0)")
    stale = []
    for report, generator in _current_reports_with_generators():
        rev = _git("log", "-1", "--format=%H", "--", str(report.relative_to(ROOT))).strip()
        if not rev:
            continue
        for config in _config_deps(generator):
            rel = config.relative_to(ROOT).as_posix()
            try:
                then = _load_yaml(_git("show", f"{rev}:{rel}"))
            except subprocess.CalledProcessError:
                then = None
            if then != _load_yaml(config.read_text(encoding="utf-8")):
                stale.append(f"{report.relative_to(ROOT)}: {rel} changed after the report was last regenerated")
    assert not stale, ("CURRENT reports older than their model configs -- regenerate them, or mark them "
                       "FROZEN in reports/README.md:\n" + "\n".join(stale))
