"""S1 tuning on val for the QG gyrostat DA: ETKF, EnKF and localized EnKS under the realistic S1.

The benchmark settings were tuned in S0 (`docs/results/qg_specwind_da2_val_tuning.md`,
`docs/results/qg_specwind_etkf_loc_update.md`, `docs/results/qg_specwind_enks.md`).
Under model error the usual levers are multiplicative inflation and a
larger observation-error variance (R scale). The localized EnKS needs
inflation 1, so it is tuned on the R scale, radius and lag only. Each task
runs one configuration on the first 20 val windows of the forced dataset
under the realistic S1 (base). Tasks 44-55 extend the R scale to 4-8 at each
method's best radius (the first grid peaked at its edge, R x 3).

    python -m evaluation.qg_specwind_s1_tuning --list
    python -m evaluation.qg_specwind_s1_tuning --task 3
    python -m evaluation.qg_specwind_s1_tuning --summarize
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
DEFAULTS = {"etkf": {"loc": 8.0, "infl": 1.0, "r": 1.0, "lag": None},
            "enkf": {"loc": 6.0, "infl": 1.0, "r": 1.0, "lag": None},
            "enks": {"loc": 8.0, "infl": 1.0, "r": 1.0, "lag": 12}}


def configs() -> list[dict]:
    out = []
    for method in ("etkf", "enkf"):
        for infl, loc in itertools.product((1.0, 1.05, 1.1, 1.2), (6.0, 8.0)):
            out.append({"method": method, "loc": loc, "infl": infl, "r": 1.0, "lag": None})
        for r, loc in itertools.product((1.5, 2.0, 3.0), (6.0, 8.0)):
            out.append({"method": method, "loc": loc, "infl": 1.0, "r": r, "lag": None})
    for r, loc, lag in itertools.product((1.0, 1.5, 2.0, 3.0), (6.0, 8.0), (6, 12)):
        out.append({"method": "enks", "loc": loc, "infl": 1.0, "r": r, "lag": lag})
    for r in (4.0, 6.0, 8.0):
        out.append({"method": "etkf", "loc": 8.0, "infl": 1.0, "r": r, "lag": None})
        out.append({"method": "enkf", "loc": 6.0, "infl": 1.0, "r": r, "lag": None})
        for lag in (6, 12):
            out.append({"method": "enks", "loc": 8.0, "infl": 1.0, "r": r, "lag": lag})
    return out


def name(c: dict) -> str:
    base = f"{c['method']}_loc{c['loc']:g}_infl{c['infl']:g}_r{c['r']:g}"
    return base + (f"_lag{c['lag']}" if c["method"] == "enks" else "")


def run_task(task: int, out_dir: str, root: str, n_windows: int, device: torch.device) -> str:
    c = configs()[task]
    path = os.path.join(out_dir, f"{name(c)}.json")
    if os.path.exists(path):
        return path
    spec = SPECS[SPEC]
    cfg = build_cfg(spec, 3, 0.05, 5.0)
    levels = REALISTIC_VARIANTS["base"]
    windows, report = s0_windows(spec, "val", root, list(range(n_windows)), cfg, device)
    windows = apply_s1(windows, spec, levels)
    t0 = time.time()
    _, per_window = evaluate(windows, da_cfg(cfg, levels), c["method"], device, inflation=c["infl"],
                             loc_radius=c["loc"], etkf_ridge=0.1, etkf_loc_mode="ensrf",
                             enks_lag=c["lag"], r_scale=c["r"])
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"config": c, "name": name(c), "load": report, "da_seconds": round(time.time() - t0, 1),
                   "per_window": per_window}, fh, indent=1)
    return path


def _boot(x: np.ndarray, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(out_dir: str) -> list[dict]:
    recs = {}
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".json") and f != "summary.json":
            r = json.load(open(os.path.join(out_dir, f)))
            recs[r["name"]] = r
    rows = []
    for nm, r in recs.items():
        c, pw = r["config"], r["per_window"]
        row = {"name": nm, **c, "score": float(np.mean([w["score"] for w in pw]))}
        for k in FIELDS:
            row[f"ev_{k}"] = float(np.mean([w[f"ev_da_{k}"] for w in pw]))
        row["spread_q1"] = float(np.median([w["spread_ratio_q1"] for w in pw]))
        base = recs.get(name({"method": c["method"], **DEFAULTS[c["method"]]}))
        if base is not None:
            d = np.array([w["score"] for w in pw]) - np.array([w["score"] for w in base["per_window"]])
            row["d_default"], row["d_default_ci"] = float(d.mean()), _boot(d)
        rows.append(row)
    rows.sort(key=lambda x: (x["method"], -x["score"]))
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--list", action="store_true")
    p.add_argument("--task", type=int)
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--out-dir", default="experiments/qg_specwind_s1_tuning")
    p.add_argument("--root", default="experiments/qg_datasets")
    p.add_argument("--n-windows", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.list:
        for k, c in enumerate(configs()):
            print(k, name(c))
    elif args.summarize:
        for r in summarize(args.out_dir):
            ci = r.get("d_default_ci")
            print(f"{r['name']:34s} score {r['score']:.3f}  psi1 {r['ev_psi1']:.3f}  psi2 {r['ev_psi2']:.3f}  "
                  f"q1 {r['ev_q1']:.3f}  q2 {r['ev_q2']:.3f}  spread q1 {r['spread_q1']:.2f}"
                  + (f"  d_default {r['d_default']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""))
    else:
        print(run_task(args.task, args.out_dir, args.root, args.n_windows, torch.device(args.device)))


if __name__ == "__main__":
    main()
