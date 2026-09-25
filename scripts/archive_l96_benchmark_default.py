#!/usr/bin/env python3
"""Hard-link the runs behind the L96 benchmark reports -- ``p1_l96_benchmark.md``,
``l96_benchmark_default.md`` and ``l96_benchmark_extended.md`` -- into the canonical
L96 archive (``<repo>/experiments/l96/<run>/``), in the layout
``scripts/consolidate_l96_archive.py`` uses: ``checkpoints/``, ``resolved_config.yaml``,
``results.json``, regular-grid ``estimates_{s0,s1}.npz`` + eval json, plus the canonical
random-set estimates as ``estimates_rlayout_{s0,s1}.npz``. Hybrids have no checkpoint of
their own; their estimates are archived under the hybrid name and the manifest names
the two component runs. The canonical test set and its manifest are linked under
``experiments/l96/testsets/``. Runs that only the P1 report uses (S+/L tiers) were never
scored on the random set, so only their regular-grid results are archived.

Hard links: same filesystem, no extra space (the volume runs near full), and they
survive the source worktree being pruned. Members files are NOT archived (1.6 GB each).
Dry-run by default; ``--apply`` to link. ``--refresh`` replaces archived estimates /
eval json that differ from the report's current source (e.g. flows re-scored with the
#257 sampler, SDA2/SDA3 re-evaluated after the S1 params fix); checkpoints are never
replaced -- a checkpoint conflict is reported and left alone. Idempotent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Optional

REPO = Path("/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode")
ARCHIVE = REPO / "experiments" / "l96"
BENCH = REPO / "4dvarnet-fm-bench-default/experiments"
HERE = REPO / "4dvarnet-fm-sda-nanfix/experiments"
P1 = REPO / "4dvarnet-fm-fdv-tau-aware/experiments"
REGULAR_EVAL = HERE / "eval_regular"
CANON_EVAL = HERE / "eval_rlayout_n10-100_k4-16_w200"
CANON_EVAL_FLOW = REPO / "4dvarnet-fm-bench-n20/experiments/eval_rlayout_n10-100_k4-16_w200"
TESTSET = REPO / "experiments" / "l96_testset_rlayout_n10-100_k4-16_w200_d1.pt"
REPORTS = {"p1": "reports/l96/outputs/p1_l96_benchmark.md",
           "default": "reports/l96/outputs/l96_benchmark_default.md",
           "extended": "reports/l96/outputs/l96_benchmark_extended.md"}

DET, FLOW, SDA20, SDA25, HYB = "ens1_no1", "ens30_no20", "ens30_gw20", "ens30_gw25", "tau0.1_gw2"


class Run(NamedTuple):
    name: str
    run_dir: Optional[Path]
    regular: Path
    canonical: Optional[Path]
    reports: tuple
    components: tuple = ()


def _std(name: str, src: Path, sub: str, reports: tuple) -> Run:
    can_root = CANON_EVAL_FLOW if sub == FLOW else CANON_EVAL
    canonical = None if reports == ("p1",) else can_root / name / sub
    return Run(name, src / name, src / name / sub, canonical, reports)


def _sda_seed(name: str, run_dir: Path) -> Run:
    return Run(name, run_dir, REGULAR_EVAL / name / SDA25, CANON_EVAL / name / SDA25, ("extended",))


def _hybrid(name: str, mean: str, prior: str) -> Run:
    return Run(name, None, REGULAR_EVAL / name / HYB, CANON_EVAL / name / HYB, ("extended",), (mean, prior))


P1_RUNS = [(n, DET) for n in ("P1_directunet_monaiSplus_noaug_l96", "P1_directunet_monaiM_noaug_l96",
                              "P1_directunet_monaiL_noaug_l96")] \
    + [(n, FLOW) for n in ("A1_vanillacfm_monaiSplus_l96", "A1_vanillacfm_monaiM_l96", "A1_vanillacfm_monaiL_l96",
                           "A2_predictstatecfm_monaiSplus_l96", "A2_predictstatecfm_monaiM_l96",
                           "A2_predictstatecfm_monaiL_lr3e4_l96", "A2_predictstatecfm_monaiL_lr5e4_l96")] \
    + [(n, SDA20) for n in ("B4_sda1_monaiSplus_l96", "B4_sda1_monaiM_l96", "B4_sda1_monaiL_l96",
                            "A3_sda2_monaiM_l96", "A3_sda3_monaiM_l96")]
IN_DEFAULT = {"P1_directunet_monaiM_noaug_l96", "A1_vanillacfm_monaiM_l96", "A2_predictstatecfm_monaiM_l96",
              "B4_sda1_monaiSplus_l96", "B4_sda1_monaiM_l96", "B4_sda1_monaiL_l96", "A3_sda2_monaiM_l96",
              "A3_sda3_monaiM_l96"}

RUNS = (
    [_std(n, P1, sub, ("p1", "default") if n in IN_DEFAULT else ("p1",)) for n, sub in P1_RUNS]
    + [_std(f"L96B_directunet_monaiM_seed{s}", BENCH, DET, ("default", "extended")) for s in (1, 2, 3)]
    + [_std(f"L96B_vanillacfm_monaiM_seed{s}", BENCH, FLOW, ("default", "extended")) for s in (1, 2, 3)]
    + [_std(f"L96B_predictstatecfm_monaiM_seed{s}", HERE, FLOW, ("default", "extended")) for s in (1, 2, 3)]
    + [_std(f"L96B_directunet_monaiM_ep1200_seed{s}", HERE, DET, ("extended",)) for s in (1, 2, 3)]
    + [Run(f"L96B_{f}_monaiM_ep1200_seed{s}", HERE / f"L96B_{f}_monaiM_ep1200_seed{s}",
           HERE / f"L96B_{f}_monaiM_ep1200_seed{s}" / FLOW, CANON_EVAL / f"L96B_{f}_monaiM_ep1200_seed{s}" / FLOW,
           ("extended",)) for f in ("vanillacfm", "predictstatecfm") for s in (1, 2, 3)]
    + [_std("L96B_directunet_monaiM_ntrain3000_seed1", HERE, DET, ("extended",))]
    + [_sda_seed(f"{e}_seed{s}", HERE / f"{e}_seed{s}") for e in ("B4_sda1_monaiM_l96", "A3_sda2_monaiM_l96")
       for s in (2, 3)]
    + [_sda_seed(f"A3_sda3fix_monaiM_l96_seed{s}", HERE / f"A3_sda3fix_monaiM_l96_seed{s}") for s in (1, 2, 3)]
    + [_hybrid(f"hybrid_DU1200s{s}_A3_sda3fix_monaiM_l96", f"L96B_directunet_monaiM_ep1200_seed{s}",
               "A3_sda3fix_monaiM_l96_seed1") for s in (1, 2, 3)]
    + [_hybrid(f"hybrid_DU1200s{s}_A3_sda2_monaiM_l96", f"L96B_directunet_monaiM_ep1200_seed{s}",
               "A3_sda2_monaiM_l96") for s in (1, 2, 3)]
)


def sha256_head(path: Path, limit: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(limit))
    return h.hexdigest()


def link(src: Path, dst: Path, apply: bool, refresh: bool) -> str:
    if dst.exists():
        if os.path.samefile(src, dst):
            return "same"
        if not refresh or dst.parent.name == "checkpoints":
            return "CONFLICT"
        if apply:
            dst.unlink()
            os.link(src, dst)
            return "refreshed"
        return "would-refresh"
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dst)
        return "linked"
    return "would-link"


def _eval_json(d: Path) -> str:
    return next((f for f in ("hybrid_eval.json", "sda_eval.json", "neural_eval.json") if (d / f).exists()),
                "neural_eval.json")


def plan(run: Run) -> list[tuple[Path, str]]:
    items = []
    if run.run_dir is not None:
        items += [(p, f"checkpoints/{p.name}") for p in sorted((run.run_dir / "checkpoints").glob("*.ckpt"))]
        items += [(run.run_dir / f, f) for f in ("resolved_config.yaml", "results.json") if (run.run_dir / f).exists()]
    for d, prefix in ((run.regular, ""), (run.canonical, "rlayout_")):
        if d is None:
            continue
        tag = "estimates_rlayout" if prefix else "estimates"
        items += [(d / f"estimates_{c}.npz", f"{tag}_{c}.npz") for c in ("s0", "s1")]
        j = _eval_json(d)
        items.append((d / j, f"{prefix}{j}"))
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    counts: dict[str, int] = {}
    missing, conflicts = [], []
    for run in RUNS:
        provenance = {}
        for s, rel in plan(run):
            if not s.exists():
                missing.append(str(s))
                continue
            status = link(s, ARCHIVE / run.name / rel, args.apply, args.refresh)
            counts[status] = counts.get(status, 0) + 1
            if status == "CONFLICT":
                conflicts.append(f"{run.name}/{rel}")
            provenance[rel] = {"source": str(s.relative_to(REPO)), "bytes": s.stat().st_size,
                               "sha256_head": sha256_head(s)}
        if args.apply:
            manifest = {"run": run.name, "written_at": datetime.now(timezone.utc).isoformat(),
                        "benchmarks": [REPORTS[r] for r in run.reports], "artifacts": provenance}
            if run.components:
                manifest["components"] = {"mean": run.components[0], "prior": run.components[1]}
            (ARCHIVE / run.name).mkdir(parents=True, exist_ok=True)
            (ARCHIVE / run.name / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    for f in (TESTSET, Path(str(TESTSET) + ".manifest.json")):
        status = link(f, ARCHIVE / "testsets" / f.name, args.apply, False)
        counts[status] = counts.get(status, 0) + 1
    for m in missing:
        print("MISSING", m)
    for c in conflicts:
        print("CONFLICT", c)
    print({k: v for k, v in sorted(counts.items())}, "(dry run)" if not args.apply else "")


if __name__ == "__main__":
    main()
