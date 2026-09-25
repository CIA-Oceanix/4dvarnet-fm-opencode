#!/usr/bin/env python3
"""Hard-link the runs behind ``reports/l96/outputs/l96_benchmark_default.md``
into the canonical L96 archive (``<repo>/experiments/l96/<run>/``), in the
layout ``scripts/consolidate_l96_archive.py`` uses: ``checkpoints/``,
``resolved_config.yaml``, ``results.json``, regular-grid ``estimates_{s0,s1}.npz``
+ eval json, plus the canonical random-set estimates as
``estimates_rlayout_{s0,s1}.npz``. The canonical test set and its manifest are
linked under ``experiments/l96/testsets/``.

Hard links: same filesystem, no extra space (the volume runs near full), and
they survive the source worktree being pruned. Members files are NOT archived
(1.6 GB each). Dry-run by default; ``--apply`` to link. Idempotent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode")
ARCHIVE = REPO / "experiments" / "l96"
BENCH = REPO / "4dvarnet-fm-bench-default/experiments"
HERE = REPO / "4dvarnet-fm-sda-nanfix/experiments"
P1 = REPO / "4dvarnet-fm-fdv-tau-aware/experiments"
CANON_EVAL = HERE / "eval_rlayout_n10-100_k4-16_w200"
CANON_EVAL_FLOW = REPO / "4dvarnet-fm-bench-n20/experiments/eval_rlayout_n10-100_k4-16_w200"
TESTSET = REPO / "experiments" / "l96_testset_rlayout_n10-100_k4-16_w200_d1.pt"

DET, FLOW, SDA = "ens1_no1", "ens30_no20", "ens30_gw20"
RUNS = (
    [(f"L96B_directunet_monaiM_seed{s}", BENCH, DET) for s in (1, 2, 3)]
    + [(f"L96B_vanillacfm_monaiM_seed{s}", BENCH, FLOW) for s in (1, 2, 3)]
    + [(f"L96B_predictstatecfm_monaiM_seed{s}", HERE, FLOW) for s in (1, 2, 3)]
    + [("P1_directunet_monaiM_noaug_l96", P1, DET), ("A1_vanillacfm_monaiM_l96", P1, FLOW),
       ("A2_predictstatecfm_monaiM_l96", P1, FLOW)]
    + [(n, P1, SDA) for n in ("B4_sda1_monaiSplus_l96", "B4_sda1_monaiM_l96", "B4_sda1_monaiL_l96",
                              "A3_sda2_monaiM_l96", "A3_sda3_monaiM_l96")]
)


def sha256_head(path: Path, limit: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(limit))
    return h.hexdigest()


def link(src: Path, dst: Path, apply: bool) -> str:
    if dst.exists():
        return "same" if os.path.samefile(src, dst) else "CONFLICT"
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dst)
        return "linked"
    return "would-link"


def plan(name: str, src: Path, sub: str) -> list[tuple[Path, str]]:
    run = src / name
    eval_json = "sda_eval.json" if sub == SDA else "neural_eval.json"
    items = [(p, f"checkpoints/{p.name}") for p in sorted((run / "checkpoints").glob("*")) if p.is_file()]
    items += [(run / f, f) for f in ("resolved_config.yaml", "results.json") if (run / f).exists()]
    items += [(run / sub / f"estimates_{c}.npz", f"estimates_{c}.npz") for c in ("s0", "s1")]
    items.append((run / sub / eval_json, eval_json))
    canon = CANON_EVAL_FLOW if sub == FLOW else CANON_EVAL
    items += [(canon / name / sub / f"estimates_{c}.npz", f"estimates_rlayout_{c}.npz") for c in ("s0", "s1")]
    items.append((canon / name / sub / eval_json, f"rlayout_{eval_json}"))
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    counts: dict[str, int] = {}
    missing = []
    for name, src, sub in RUNS:
        provenance = {}
        for s, rel in plan(name, src, sub):
            if not s.exists():
                missing.append(str(s))
                continue
            status = link(s, ARCHIVE / name / rel, args.apply)
            counts[status] = counts.get(status, 0) + 1
            provenance[rel] = {"source": str(s.relative_to(REPO)), "bytes": s.stat().st_size,
                               "sha256_head": sha256_head(s)}
        if args.apply:
            (ARCHIVE / name / "manifest.json").write_text(json.dumps(
                {"run": name, "written_at": datetime.now(timezone.utc).isoformat(),
                 "benchmark": "reports/l96/outputs/l96_benchmark_default.md", "artifacts": provenance},
                indent=2, sort_keys=True))
    for f in (TESTSET, Path(str(TESTSET) + ".manifest.json")):
        status = link(f, ARCHIVE / "testsets" / f.name, args.apply)
        counts[status] = counts.get(status, 0) + 1
    for m in missing:
        print("MISSING", m)
    print({k: v for k, v in sorted(counts.items())}, "(dry run)" if not args.apply else "")


if __name__ == "__main__":
    main()
