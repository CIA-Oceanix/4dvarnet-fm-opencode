#!/usr/bin/env python3
"""Gather scattered QG run artifacts into the canonical archive.

The QG counterpart of ``consolidate_l96_archive.py``, for the same failure:
``reports/qg/generate_qg_neural_report.py`` publishes a table whose neural half
could not be regenerated outside the worktree that trained the models. Three
things lived only there --

1. the cross-scenario evaluation JSON the report reads (now tracked in git),
2. the ``qg_*_norm_stats.pt`` files the checkpoints are meaningless without,
3. the checkpoints of any run not yet moved into ``experiments/qg/``.

-- so the archived Q1-Q4 checkpoints were reachable from master and still
unusable there.

Hard links, not symlinks: the worktrees share one filesystem, so this costs no
additional space, and a hard link survives the source worktree being pruned.
The 2026-09-14 QG consolidation used symlinks, which do not.

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

from evaluation.archive import QG, ROOT  # noqa: E402

EVAL_SCRIPT = ROOT / "eval_qg_neural_s0_s1.py"
REPORT_GENERATOR = ROOT / "reports/qg/generate_qg_neural_report.py"

# Per run. `estimates_s0.npz` is the in-training S0 evaluation's arrays; the
# S0/S1 table itself comes from the cross-scenario JSON, not from these.
WANTED_FILES = ("resolved_config.yaml", "config.yaml", "results.json",
                "estimates_s0.npz", "stage1_best.pt",
                "checkpoints/stage1_best.ckpt")

# Needed to interpret any QG checkpoint at all: psi z-scoring is baked into
# what the model predicts, and the conditioned schemes normalize their
# forcing/param inputs the same way.
SHARED_STATS = QG.shared_stats


def report_run_names() -> list[str]:
    """Run directories the published report's neural rows depend on.

    Parsed from ``eval_qg_neural_s0_s1.py``'s SCHEMES rather than duplicated
    here, so adding a scheme cannot leave this script behind.
    """
    src = EVAL_SCRIPT.read_text()
    m = re.search(r"SCHEMES = \{(.*?)\n\}", src, re.S)
    if not m:
        return []
    return re.findall(r'"run":\s*"([^"]+)"', m.group(1))


def report_json_inputs() -> list[Path]:
    """JSON files the report generator reads.

    The DA halves (``qg_repro_validation{,_s1}/<method>.json``) have always been
    tracked in git; the neural half was not, which is why the published table's
    four neural rows could not be regenerated from master. Both are parsed out of
    the generator rather than restated, so a renamed input surfaces here.
    """
    src = REPORT_GENERATOR.read_text()
    out = ROOT / "reports/qg/outputs"
    paths = []
    methods = re.findall(r'\("[^"]+",\s*"([^"]+)"\)', _block(src, r"DA_METHODS = \[(.*?)\]"))
    for scenario_dir in re.findall(r'load_da_summary\(out_root, "([^"]+)"', src):
        paths += [out / scenario_dir / f"{m}.json" for m in methods]
    m = re.search(r'root / "(qg_neural_s0_s1_cross_scenario)" / "([^"]+)"', src)
    if m:
        paths.append(out / m.group(1) / m.group(2))
    return sorted(set(paths))


def _block(src: str, pattern: str) -> str:
    m = re.search(pattern, src, re.S)
    if not m:
        raise SystemExit(f"could not find {pattern!r} in {REPORT_GENERATOR}")
    return m.group(1)


def sibling_worktrees() -> list[Path]:
    return sorted(p for p in ROOT.parent.glob("4dvarnet-fm-*")
                  if (p / "experiments").is_dir())


def find_source(relative: str) -> Path | None:
    """Locate ``experiments/<relative>`` in this or a sibling worktree.

    Follows symlinks only to test existence -- ``os.link`` then links the real
    file, which is how a symlinked archive entry becomes a hard-linked one.
    """
    for wt in [ROOT] + sibling_worktrees():
        cand = wt / "experiments" / relative
        if cand.is_file():
            return cand.resolve()
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


def audit(portable: bool = False) -> dict[str, list[str]]:
    """What the published report needs and cannot resolve. Empty == reproducible.

    ``portable`` restricts resolution to the canonical archive, answering the
    question that actually matters -- *could another worktree regenerate this
    table?* -- rather than the one the training worktree always answers yes to.
    An artifact reachable only through this worktree's own ``experiments/<run>``
    is exactly the state that made the QG neural rows unreproducible: it looks
    fine from here and does not exist anywhere else.
    """
    def has(name: str, filename: str) -> bool:
        if portable:
            return (QG.archive_dir / name / filename).is_file()
        return QG.resolve_artifact(name, filename, required=False) is not None

    problems: dict[str, list[str]] = {}
    for name in report_run_names():
        missing = []
        if not any(has(name, c) for c in QG.checkpoints):
            missing.append("checkpoint")
        if not (has(name, "resolved_config.yaml") or has(name, "results.json")):
            missing.append("resolved_config.yaml|results.json")
        if missing:
            problems[name] = missing
    stats_missing = [s for s in SHARED_STATS
                     if not ((QG.archive_dir / s).is_file() if portable
                             else QG.resolve_norm_stats(None, filename=s) is not None)]
    if stats_missing:
        problems["<shared norm stats>"] = stats_missing
    absent_json = [str(p.relative_to(ROOT)) for p in report_json_inputs() if not p.is_file()]
    if absent_json:
        problems["<report inputs>"] = absent_json
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="make changes (default: dry run)")
    ap.add_argument("--manifest", action="store_true",
                    help="write manifest.json per run (provenance of each artifact)")
    ap.add_argument("--names", nargs="*", help="limit to these runs")
    ap.add_argument("--check", action="store_true",
                    help="audit only: exit non-zero if anything the published "
                         "report needs is unresolvable. Nothing is linked.")
    ap.add_argument("--portable", action="store_true",
                    help="with --check: resolve only through the canonical "
                         "archive, i.e. ask whether another worktree could "
                         "regenerate the report, not just this one.")
    args = ap.parse_args()

    if args.check:
        problems = audit(portable=args.portable)
        names = report_run_names()
        print(f"runs the report needs:        {len(names)}")
        print(f"unresolvable:                 {len(problems)}")
        for n, miss in sorted(problems.items()):
            print(f"  {n:46s} {miss}")
        if problems:
            print("\nRun `scripts/consolidate_qg_archive.py --apply` to link them "
                  "in from the sibling worktrees.")
        return 1 if problems else 0

    names = args.names or report_run_names()
    actions: dict[str, int] = {}

    for stats in SHARED_STATS:
        if (QG.archive_dir / stats).is_file():
            actions["already-archived"] = actions.get("already-archived", 0) + 1
            continue
        src = find_source(stats)
        if src is None:
            print(f"  MISSING      {stats} (not in any worktree)")
            continue
        # Into the archive dir, not experiments/ root: the stats must travel
        # with the checkpoints, or another worktree inherits the same problem.
        status = link(src, QG.archive_dir / stats, args.apply)
        actions[status] = actions.get(status, 0) + 1
        print(f"  {status:12s} qg/{stats}  <- {src}")

    for name in names:
        provenance = {}
        for filename in WANTED_FILES:
            if (QG.archive_dir / name / filename).is_file():
                actions["already-archived"] = actions.get("already-archived", 0) + 1
                continue
            src = find_source(f"{name}/{filename}")
            if src is None:
                continue
            status = link(src, QG.archive_dir / name / filename, args.apply)
            actions[status] = actions.get(status, 0) + 1
            print(f"  {status:12s} {name}/{filename}  <- {src}")
            provenance[filename] = {"source": str(src)}

        if args.manifest and args.apply and QG.resolve_run_dir(name, required=False):
            for filename in WANTED_FILES:
                p = QG.resolve_artifact(name, filename, required=False)
                if p is not None:
                    # No absolute path recorded: the entry is keyed by its
                    # archive-relative filename already, and an absolute one
                    # would bake in whichever worktree happened to run this.
                    provenance.setdefault(filename, {})["sha256_head"] = sha256(p)
                    provenance[filename]["bytes"] = p.stat().st_size
            QG.write_manifest(name, {
                "run": name,
                "written_at": datetime.now(timezone.utc).isoformat(),
                "artifacts": provenance,
            })

    print("\n=== summary ===")
    for k, v in sorted(actions.items()):
        print(f"  {k:20s} {v}")
    problems = audit(portable=True)
    print(f"\nruns the report needs:        {len(names)}")
    print(f"unresolvable from the archive: {len(problems)}")
    for n, miss in sorted(problems.items()):
        print(f"  {n:46s} {miss}")
    if not args.apply:
        print("\n(dry run -- pass --apply to make changes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
