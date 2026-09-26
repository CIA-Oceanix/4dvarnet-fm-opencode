#!/usr/bin/env python3
"""Build the input bundles the L96 benchmark reports regenerate from.

``reports/l96/_inputs.py`` resolves every input root of a report to
``experiments/l96/report_inputs/<report>/<root>/``. This script fills those
bundles:

  record  run a report's generator against the topic-worktree directories below
          (via ``FDV_REPORT_INPUT_ROOTS``) under an audit hook, and log every file
          it opens;
  link    hard-link each logged file under a legacy root into the bundle at the
          same relative path (same filesystem: no extra space, survives pruning),
          plus the ``*_eval.json`` sidecars beside it -- generators test a result
          directory's completeness by their existence without opening them,
          write ``MANIFEST.json`` and flag eval sidecars whose recorded dataset
          path points into a worktree;
  run     run the generator against the bundle only (the default resolution).

A report is migrated when ``run`` reproduces the checked-in output byte for byte.
``LEGACY`` is the one place that still names the topic worktrees; it records
where the bundles were built from and is not read by any generator.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reports" / "l96"))
import _inputs  # noqa: E402

GENERATORS = {
    "p1_benchmark": "reports/l96/generate_p1_l96_benchmark.py",
    "benchmark_default": "reports/l96/generate_l96_benchmark_default_report.py",
    "benchmark_extended": "reports/l96/generate_l96_benchmark_extended_report.py",
}
_WT = {
    "sda_nanfix": "4dvarnet-fm-sda-nanfix", "bench_default": "4dvarnet-fm-bench-default",
    "fdv_tau_aware": "4dvarnet-fm-fdv-tau-aware", "bench_n20": "4dvarnet-fm-bench-n20",
    "da_random_layout": "4dvarnet-fm-da-random-layout", "random_obs_times": "4dvarnet-fm-random-obs-times",
}
LEGACY = {
    "p1_benchmark": {"here": "fdv_tau_aware"},
    "benchmark_default": {"here": "bench_n20", "bench": "bench_default", "p1": "fdv_tau_aware",
                          "da_random_layout": "da_random_layout", "random_obs_times": "random_obs_times"},
    "benchmark_extended": {"here": "sda_nanfix", "bench": "bench_default", "p1": "fdv_tau_aware",
                           "n20": "bench_n20", "da_random_layout": "da_random_layout",
                           "random_obs_times": "random_obs_times"},
}
RUNNER = r"""
import os, runpy, sys
log = open(sys.argv[1], "w")
def hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        log.write(os.path.abspath(os.fsdecode(args[0])) + "\n")
sys.addaudithook(hook)
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
log.close()
"""


def legacy_roots(report: str) -> dict[str, Path]:
    return {name: _inputs.main_checkout() / _WT[wt] / "experiments" for name, wt in LEGACY[report].items()}


def _run(report: str, env_roots: dict[str, Path] | None, log: Path | None) -> None:
    env = dict(os.environ)
    env.pop(_inputs.ROOTS_ENV, None)
    if env_roots is not None:
        env[_inputs.ROOTS_ENV] = json.dumps({k: str(v) for k, v in env_roots.items()})
    gen = str(ROOT / GENERATORS[report])
    cmd = [sys.executable, "-c", RUNNER, str(log), gen] if log else [sys.executable, gen]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _log_path(report: str, workdir: Path) -> Path:
    return workdir / f"opened_{report}.txt"


def record(report: str, workdir: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    _run(report, legacy_roots(report), _log_path(report, workdir))


def link(report: str, workdir: Path, apply: bool) -> int:
    roots = legacy_roots(report)
    opened = {line.strip() for line in _log_path(report, workdir).read_text().splitlines() if line.strip()}
    opened |= {str(j) for p in list(opened) for j in Path(p).parent.glob("*_eval.json")}
    opened = sorted(opened)
    dest_root = _inputs.bundle(report)
    counts: dict[str, int] = {}
    files: dict[str, list[str]] = {name: [] for name in roots}
    problems: list[str] = []
    for p in opened:
        path = Path(p)
        owner = next((n for n, r in roots.items() if path.is_relative_to(r)), None)
        if owner is None or not path.is_file():
            continue
        rel = path.relative_to(roots[owner])
        files[owner].append(rel.as_posix())
        dst = dest_root / owner / rel
        src = Path(os.path.realpath(path))
        if dst.exists():
            status = "same" if os.path.samefile(src, dst) else "CONFLICT"
        else:
            status = "linked" if apply else "would-link"
            if apply:
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.link(src, dst)
        counts[status] = counts.get(status, 0) + 1
        if status == "CONFLICT":
            problems.append(f"CONFLICT {dst}")
        if path.suffix == ".json" and path.name.endswith("_eval.json"):
            ds = (json.loads(path.read_text()).get("dataset") or {}).get("path")
            if ds and any(w in ds for w in _WT.values()):
                problems.append(f"WORKTREE-DATASET {path}: dataset.path = {ds}")
    if apply:
        dest_root.mkdir(parents=True, exist_ok=True)
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        manifest = {"report": report, "generator": GENERATORS[report], "built_at": datetime.now(timezone.utc).isoformat(),
                    "built_from_commit": sha,
                    "roots": {n: {"built_from": str(r), "files": sorted(files[n])} for n, r in roots.items()}}
        (dest_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    for p in problems:
        print(p)
    print(report, dict(sorted(counts.items())), "(dry run)" if not apply else "")
    return sum(1 for p in problems if p.startswith("CONFLICT"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["record", "link", "run"])
    ap.add_argument("reports", nargs="*", default=list(GENERATORS))
    ap.add_argument("--workdir", type=Path, default=ROOT / "experiments" / "report_inputs_logs")
    ap.add_argument("--apply", action="store_true", help="link: actually create the hard links")
    args = ap.parse_args()
    conflicts = 0
    for report in args.reports:
        if args.action == "record":
            record(report, args.workdir)
        elif args.action == "link":
            conflicts += link(report, args.workdir, args.apply)
        else:
            _run(report, None, None)
    sys.exit(1 if conflicts else 0)


if __name__ == "__main__":
    main()
