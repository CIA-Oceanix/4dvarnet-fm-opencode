"""Generate, assemble and check a QG spectral-wind dataset (G2).

    generate  one shard of one split (a SLURM array task runs one shard)
    assemble  build manifests, then write the independence and diversity reports
    all       generate every split in one process, then assemble (small specs)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402

from data.qg_datasets import (  # noqa: E402
    SPECS,
    QGGeneratedSplit,
    assemble_split,
    check_independence,
    diversity_report,
    estimate_storage_gb,
    split_dir,
    write_shard,
)

PURPOSE = {"train": "train", "val": "eval", "test": "test"}


def _assemble(spec, root: str) -> dict:
    manifests = {s: assemble_split(spec, s, root) for s in ("train", "val", "test")
                 if os.path.isdir(split_dir(root, spec, s))}
    splits = {s: QGGeneratedSplit(split_dir(root, spec, s), purpose=PURPOSE[s]) for s in manifests}
    reports = {"independence": [], "passed": True}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        if a in splits and b in splits:
            rep = check_independence(splits[a], splits[b])
            reports["independence"].append(rep)
            reports["passed"] = reports["passed"] and rep["passed"]
    out_dir = os.path.join(root, spec.name, "reports")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "independence.json"), "w") as fh:
        json.dump(reports, fh, indent=1)
    with open(os.path.join(out_dir, "diversity.json"), "w") as fh:
        json.dump(diversity_report(splits), fh)
    summary = {"spec": spec.name, "n": {s: m["n"] for s, m in manifests.items()},
               "generation_seconds": {s: round(m["generation_seconds"], 1) for s, m in manifests.items()},
               "independence_passed": reports["passed"]}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=("generate", "assemble", "all", "estimate"))
    p.add_argument("--spec", default="qg_specwind_gyrostat_demo", choices=sorted(SPECS))
    p.add_argument("--split", choices=("train", "val", "test"))
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--root", default="experiments/qg_datasets")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    spec = SPECS[args.spec]
    if args.command == "estimate":
        print(json.dumps({"storage_gb": estimate_storage_gb(spec), "steps_per_window": spec.n_total}))
        return
    if args.command == "generate":
        if args.split is None:
            p.error("generate needs --split")
        t0 = time.time()
        path = write_shard(spec, args.split, args.root, args.shard, args.n_shards,
                           device=args.device, batch_size=args.batch_size)
        print(f"{args.split} shard {args.shard}/{args.n_shards}: {path} ({time.time() - t0:.0f}s)")
        return
    if args.command == "all":
        for split in ("train", "val", "test"):
            t0 = time.time()
            write_shard(spec, split, args.root, 0, 1, device=args.device, batch_size=args.batch_size)
            print(f"{split}: {spec.n_windows(split)} windows in {time.time() - t0:.0f}s", flush=True)
    print(json.dumps(_assemble(spec, args.root), indent=1))


if __name__ == "__main__":
    main()
