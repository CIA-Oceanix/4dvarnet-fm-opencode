#!/usr/bin/env python3
"""Gather scattered L96 run artifacts into the canonical archive.

The 2026-09-10 consolidation moved ``checkpoints/`` and ``resolved_config.yaml``
into ``experiments/l96/<run>/`` but deliberately left the large
``estimates_*.npz`` behind in whichever worktree trained the run. The result is
that ``reports/l96/generate_l96_consolidated_report.py`` cannot run from the
master worktree at all -- it dies on the first row whose estimates live
somewhere else.

This script finds those artifacts across the sibling topic worktrees and links
them into the canonical archive. It uses **hard links**: the worktrees share one
filesystem, so this costs no additional space (the arrays are ~100 MB each and
the volume runs near full), and unlike symlinks a hard link survives the
original worktree being pruned and is immune to ``Path.resolve()`` walking off
somewhere unexpected -- the specific way the old norm-stats path arithmetic
broke.

Dry-run by default; pass --apply to make changes. Idempotent.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import archive  # noqa: E402

ROOT = archive.ROOT
WANTED_FILES = ("estimates_s0.npz", "estimates_s1.npz", "resolved_config.yaml",
                "neural_eval.json", "results.json")

# Runs that do not use the standard estimates_<case>.npz name. The obs-density
# hybrids write one file per retained fast-Y count; keep16 is the full-density
# one the consolidated benchmark reports.
EXTRA_FILES = {
    "l96_obs_density_directunet_aug_sda3_hybrid": (
        "estimates_directunet_sda3_s0_keep16.npz",
        "estimates_directunet_sda3_s1_keep16.npz",
    ),
}


def report_estimate_filenames() -> dict[str, str]:
    """The report's ESTIMATE_FILENAMES overrides, for runs not using the
    standard ``estimates_<case>.npz`` name."""
    src = (ROOT / "reports/l96/generate_l96_consolidated_report.py").read_text()
    m = re.search(r"ESTIMATE_FILENAMES: dict\[str, str\] = \{(.*?)\n\}", src, re.S)
    if not m:
        return {}
    return dict(re.findall(r'"([^"]+)":\s*\n?\s*"([^"]+)"', m.group(1)))


def report_unavailable() -> set[str]:
    """Methods the report deliberately publishes with no result (em-dash row),
    so absent estimates are expected rather than a broken artifact."""
    src = (ROOT / "reports/l96/generate_l96_consolidated_report.py").read_text()
    m = re.search(r"UNAVAILABLE_METHODS: frozenset\[str\] = frozenset\(\{(.*?)\}\)", src, re.S)
    return set(re.findall(r'"([^"]+)"', m.group(1))) if m else set()


def report_methods() -> dict[str, dict[str, list[str]]]:
    """method -> case -> candidate run dirs, in the report's own priority order.

    Mirrors ``collect_estimates``: an ens30 method reads its per-case ens30
    subdirectory and falls back to the plain run dir; every other method reads
    the plain run dir only. Auditing per method+case (rather than demanding both
    cases from every directory) is what makes the result meaningful --
    ``L3_vanilla_cfm_s0s1/ens30_no10`` supplies s0 and is *supposed* to have no
    s1, so a naive check reports it as broken forever.
    """
    src = (ROOT / "reports/l96/generate_l96_consolidated_report.py").read_text()
    methods: list[str] = []
    m = re.search(r"NEURAL_EXP_DIRS = \[(.*?)\]", src, re.S)
    if m:
        methods = re.findall(r'"([^"]+)"', m.group(1))
    ens30 = {name: {"s0": s0, "s1": s1} for name, s0, s1 in re.findall(
        r'"([^"]+)":\s*\{\s*"s0":\s*"([^"]+)",\s*"s1":\s*"([^"]+)"', src)}
    out: dict[str, dict[str, list[str]]] = {}
    for name in methods:
        out[name] = {c: ([ens30[name][c], name] if name in ens30 else [name])
                     for c in archive.CASES}
    return out


def report_run_names() -> list[str]:
    """Every run dir the report may read, for linking purposes."""
    names = set()
    for cases in report_methods().values():
        for candidates in cases.values():
            names.update(candidates)
    return sorted(names)


def audit_methods() -> dict[str, list[str]]:
    """method -> cases whose estimates resolve from no candidate directory."""
    broken = {}
    fnames = report_estimate_filenames()
    unavailable = report_unavailable()
    for method, cases in report_methods().items():
        if method in unavailable:
            continue
        missing = [c for c, candidates in cases.items()
                   if all(archive.resolve_estimates(
                       n, c, filename=fnames.get(n), required=False) is None
                       for n in candidates)]
        if missing:
            broken[method] = missing
    return broken


def sibling_worktrees() -> list[Path]:
    return sorted(p for p in ROOT.glob("4dvarnet-fm-*") if (p / "experiments").is_dir())


def find_source(name: str, filename: str) -> Path | None:
    """Locate <name>/<filename> in a sibling worktree's experiments/."""
    for wt in sibling_worktrees():
        cand = wt / "experiments" / name / filename
        if cand.is_file():
            return cand
    return None


def sha256(path: Path, limit: int = 1 << 20) -> str:
    """Hash the first ``limit`` bytes -- enough to detect a swapped artifact
    without reading gigabytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(limit))
    return h.hexdigest()


def link(src: Path, dst: Path, apply: bool) -> str:
    if dst.exists():
        return "exists"
    if not apply:
        return "would-link"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        import shutil
        shutil.copy2(src, dst)
        return "copied"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="make changes (default: dry run)")
    ap.add_argument("--manifest", action="store_true",
                    help="write manifest.json per run (provenance of each artifact)")
    ap.add_argument("--names", nargs="*", help="limit to these runs")
    ap.add_argument("--check", action="store_true",
                    help="audit only: exit non-zero if any run the report needs "
                         "is unresolvable. Nothing is linked or written.")
    args = ap.parse_args()

    if args.check:
        methods = report_methods()
        incomplete = audit_methods()
        print(f"methods the report needs:     {len(methods)}")
        print(f"methods missing estimates:    {len(incomplete)}")
        for n, miss in sorted(incomplete.items()):
            print(f"  {n:52s} {miss}")
        if incomplete:
            print("\nRun `scripts/consolidate_l96_archive.py --apply` to link them "
                  "in from the sibling worktrees.")
        return 1 if incomplete else 0

    names = args.names or report_run_names()
    actions: dict[str, int] = {}
    still_missing: dict[str, list[str]] = {}

    for name in names:
        dest_dir = archive.ARCHIVE / name
        provenance = {}
        for filename in WANTED_FILES + EXTRA_FILES.get(name, ()):
            if archive.resolve_artifact(name, filename, required=False) is not None:
                actions["already-resolvable"] = actions.get("already-resolvable", 0) + 1
                continue
            src = find_source(name, filename)
            if src is None:
                if filename.startswith("estimates_"):
                    still_missing.setdefault(name, []).append(filename)
                continue
            status = link(src, dest_dir / filename, args.apply)
            actions[status] = actions.get(status, 0) + 1
            print(f"  {status:12s} {name}/{filename}  <- {src.relative_to(ROOT)}")
            provenance[filename] = {"source": str(src.relative_to(ROOT))}

        if args.manifest and args.apply and archive.resolve_run_dir(name, required=False):
            for filename in WANTED_FILES:
                p = archive.resolve_artifact(name, filename, required=False)
                if p is not None:
                    provenance.setdefault(filename, {})["path"] = str(p.relative_to(ROOT))
                    provenance[filename]["sha256_head"] = sha256(p)
                    provenance[filename]["bytes"] = p.stat().st_size
            archive.write_manifest(name, {
                "run": name,
                "written_at": datetime.now(timezone.utc).isoformat(),
                "artifacts": provenance,
            })

    print("\n=== summary ===")
    for k, v in sorted(actions.items()):
        print(f"  {k:20s} {v}")
    incomplete = archive.audit(names)
    print(f"\nruns the report needs:        {len(names)}")
    print(f"runs still missing artifacts: {len(incomplete)}")
    for n, miss in sorted(incomplete.items()):
        print(f"  {n:52s} {miss}")
    if not args.apply:
        print("\n(dry run -- pass --apply to make changes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
