"""Localized ETKF update check on val: legacy `square_root` vs exact EnSRF vs EnKF, S0 and S1.

The legacy localized update (`ETKF(loc_mode="square_root")`) builds its gain
from sample covariances without the 1/(N-1) factor and applies the full gain
to the anomalies, so it both over-trusts the observations and over-contracts
the spread. `loc_mode="ensrf"` is the exact localized EnSRF
(`evaluation.baselines._ensrf_localized_analysis`). Each task runs one
configuration on the first 20 val windows of the forced dataset, in S0 or in
the realistic S1 (base), and records per-window EVs and spread/RMSE ratios.
Tasks 16-25 run the localized EnKS (`EnKS`, smoother on the EnSRF ETKF,
radius 8, ridge 0.1) at lags 3, 6, 12, 24 analyses and the whole window.

    python -m evaluation.qg_specwind_etkf_check --list
    python -m evaluation.qg_specwind_etkf_check --task 3
    python -m evaluation.qg_specwind_etkf_check --summarize
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np
import torch

from data.qg_datasets import SPECS
from evaluation.qg_specwind_s1 import REALISTIC_VARIANTS
from evaluation.run_qg_specwind_da import FIELDS, apply_s1, build_cfg, da_cfg, evaluate, s0_windows

SPEC = "qg_specwind_gyrostat_v1"


def configs() -> list[dict]:
    out = []
    for scen in ("s0", "s1"):
        out.append({"scen": scen, "method": "etkf", "mode": "square_root", "loc": 8.0, "ridge": 1.0})
        out.append({"scen": scen, "method": "enkf", "mode": "-", "loc": 6.0, "ridge": 0.1})
        for loc, ridge in itertools.product((4.0, 6.0, 8.0), (0.0, 0.1)):
            out.append({"scen": scen, "method": "etkf", "mode": "ensrf", "loc": loc, "ridge": ridge})
    for scen in ("s0", "s1"):
        for lag in (3, 6, 12, 24, None):
            out.append({"scen": scen, "method": "enks", "mode": "ensrf", "loc": 8.0, "ridge": 0.1,
                        "lag": lag})
    return out


def name(c: dict) -> str:
    base = f"{c['scen']}_{c['method']}_{c['mode']}_loc{c['loc']:g}_r{c['ridge']:g}"
    if c["method"] == "enks":
        base += f"_lag{c['lag'] if c['lag'] is not None else 'all'}"
    return base


def run_task(task: int, out_dir: str, root: str, n_windows: int, device: torch.device) -> str:
    c = configs()[task]
    path = os.path.join(out_dir, f"{name(c)}.json")
    if os.path.exists(path):
        return path
    spec = SPECS[SPEC]
    cfg = build_cfg(spec, 3, 0.05, 5.0)
    windows, report = s0_windows(spec, "val", root, list(range(n_windows)), cfg, device)
    levels = REALISTIC_VARIANTS["base"] if c["scen"] == "s1" else None
    if levels is not None:
        windows = apply_s1(windows, spec, levels)
    t0 = time.time()
    _, per_window = evaluate(windows, da_cfg(cfg, levels), c["method"], device, loc_radius=c["loc"],
                             etkf_ridge=c["ridge"],
                             etkf_loc_mode=c["mode"] if c["method"] in ("etkf", "enks") else "square_root",
                             enks_lag=c.get("lag"))
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"config": c, "name": name(c), "load": report, "da_seconds": round(time.time() - t0, 1),
                   "per_window": per_window}, fh, indent=1)
    return path


def summarize(out_dir: str) -> list[dict]:
    rows = []
    for f in sorted(os.listdir(out_dir)):
        if not f.endswith(".json") or f == "summary.json":
            continue
        r = json.load(open(os.path.join(out_dir, f)))
        pw = r["per_window"]
        row = {"name": r["name"], **r["config"]}
        for k in FIELDS:
            row[f"ev_{k}"] = float(np.mean([w[f"ev_da_{k}"] for w in pw]))
        row["score"] = float(np.mean([w["score"] for w in pw]))
        for k in ("q1", "q2"):
            row[f"spread_{k}"] = float(np.median([w[f"spread_ratio_{k}"] for w in pw]))
        rows.append(row)
    rows.sort(key=lambda x: (x["scen"], -x["score"]))
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--list", action="store_true")
    p.add_argument("--task", type=int)
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--out-dir", default="experiments/qg_specwind_etkf_check")
    p.add_argument("--root", default="experiments/qg_datasets")
    p.add_argument("--n-windows", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.list:
        for k, c in enumerate(configs()):
            print(k, name(c))
    elif args.summarize:
        for r in summarize(args.out_dir):
            print(f"{r['name']:34s} score {r['score']:.3f}  psi1 {r['ev_psi1']:.3f}  psi2 {r['ev_psi2']:.3f}  "
                  f"q1 {r['ev_q1']:.3f}  q2 {r['ev_q2']:.3f}  spread/RMSE q1 {r['spread_q1']:.2f}  "
                  f"q2 {r['spread_q2']:.2f}")
    else:
        print(run_task(args.task, args.out_dir, args.root, args.n_windows, torch.device(args.device)))


if __name__ == "__main__":
    main()
