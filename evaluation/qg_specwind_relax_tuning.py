"""Relaxation inflation (RTPS / RTPP) on val for the QG gyrostat ETKF and EnKF.

Multiplicative inflation diverges under the realistic S1, so the S1-tuned filters
absorb model error with R × 6 instead (`docs/results/qg_specwind_s1_tuning.md`).
This sweep tests the standard alternative: relaxation of the analysis anomalies
toward the forecast ones (RTPP) or of the analysis spread toward the forecast
spread (RTPS), applied once per analysis.

- Tasks 0-7: baselines without relaxation (S0 R × 1; S1 R × 1, 3, 6).
- Tasks 8-23: S1 at R × 1, RTPS and RTPP, α ∈ {0.25, 0.5, 0.75, 0.9}.
- Tasks 24-35: S1 at R × 3 and R × 6, RTPS α ∈ {0.25, 0.5, 0.75} (relaxation on top of R).
- Tasks 36-43: S0, RTPS and RTPP, α ∈ {0.25, 0.5} (spread is under-dispersed at N = 80).

Each for the ETKF (EnSRF, radius 8, ridge 0.1) and the EnKF (radius 6); 20 val windows
of the forced dataset; S1 is the realistic base.

    python -m evaluation.qg_specwind_relax_tuning --list
    python -m evaluation.qg_specwind_relax_tuning --task 3
    python -m evaluation.qg_specwind_relax_tuning --summarize
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from data.qg_datasets import SPECS
from evaluation.qg_specwind_s1 import REALISTIC_VARIANTS
from evaluation.run_qg_specwind_da import FIELDS, apply_s1, build_cfg, da_cfg, evaluate, s0_windows

SPEC = "qg_specwind_gyrostat_v1"
LOC = {"etkf": 8.0, "enkf": 6.0}
METHODS = ("etkf", "enkf")


def configs() -> list[dict]:
    out = []
    for scen, r in (("s0", 1.0), ("s1", 1.0), ("s1", 3.0), ("s1", 6.0)):
        out += [{"scen": scen, "method": m, "r": r, "relax": None, "alpha": 0.0} for m in METHODS]
    for relax in ("rtps", "rtpp"):
        for alpha in (0.25, 0.5, 0.75, 0.9):
            out += [{"scen": "s1", "method": m, "r": 1.0, "relax": relax, "alpha": alpha} for m in METHODS]
    for r in (3.0, 6.0):
        for alpha in (0.25, 0.5, 0.75):
            out += [{"scen": "s1", "method": m, "r": r, "relax": "rtps", "alpha": alpha} for m in METHODS]
    for relax in ("rtps", "rtpp"):
        for alpha in (0.25, 0.5):
            out += [{"scen": "s0", "method": m, "r": 1.0, "relax": relax, "alpha": alpha} for m in METHODS]
    return out


def name(c: dict) -> str:
    relax = f"_{c['relax']}{c['alpha']:g}" if c["relax"] else ""
    return f"{c['scen']}_{c['method']}_R{c['r']:g}{relax}"


def base_name(c: dict) -> str:
    return name({**c, "relax": None})


def run_task(task: int, out_dir: str, root: str, n_windows: int, device: torch.device) -> str:
    c = configs()[task]
    path = os.path.join(out_dir, f"{name(c)}.json")
    if os.path.exists(path):
        return path
    spec = SPECS[SPEC]
    cfg = build_cfg(spec, 3, 0.05, 5.0)
    levels = REALISTIC_VARIANTS["base"] if c["scen"] == "s1" else None
    windows, report = s0_windows(spec, "val", root, list(range(n_windows)), cfg, device)
    if levels is not None:
        windows = apply_s1(windows, spec, levels)
    t0 = time.time()
    relax = {"relax": c["relax"], "relax_alpha": c["alpha"]} if c["relax"] else {}
    _, per_window = evaluate(windows, da_cfg(cfg, levels), c["method"], device, N=80,
                             loc_radius=LOC[c["method"]], etkf_ridge=0.1, etkf_loc_mode="ensrf",
                             r_scale=c["r"], **relax)
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"config": c, "name": name(c), "load": report, "da_seconds": round(time.time() - t0, 1),
                   "per_window": per_window}, fh, indent=1)
    return path


def summarize(out_dir: str) -> list[dict]:
    recs = {}
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".json") and f != "summary.json":
            r = json.load(open(os.path.join(out_dir, f)))
            recs[r["name"]] = r
    rng = np.random.default_rng(0)
    rows = []
    for nm, r in recs.items():
        c, pw = r["config"], r["per_window"]
        score = np.array([w["score"] for w in pw])
        row = {"name": nm, **c, "score": float(score.mean()),
               "crps": float(np.mean([w["crps"] for w in pw])),
               "spread_q1": float(np.median([w["spread_ratio_q1"] for w in pw]))}
        for k in FIELDS:
            row[f"ev_{k}"] = float(np.mean([w[f"ev_da_{k}"] for w in pw]))
        for ref_name, key in ((base_name(c), "d_base"),
                              (name({**c, "r": 6.0, "relax": None}) if c["scen"] == "s1" else None, "d_r6")):
            ref = recs.get(ref_name) if ref_name else None
            if ref is None or ref_name == nm:
                continue
            d = score - np.array([w["score"] for w in ref["per_window"]])
            m = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
            row[key] = float(d.mean())
            row[f"{key}_ci"] = [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
        rows.append(row)
    rows.sort(key=lambda x: (x["scen"], x["method"], -x["score"]))
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--list", action="store_true")
    p.add_argument("--task", type=int)
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--out-dir", default="experiments/qg_specwind_relax_tuning")
    p.add_argument("--root", default="experiments/qg_datasets")
    p.add_argument("--n-windows", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.list:
        for k, c in enumerate(configs()):
            print(k, name(c))
    elif args.summarize:
        for r in summarize(args.out_dir):
            extra = "".join(f"  {k} {r[k]:+.3f} [{r[k + '_ci'][0]:+.3f}, {r[k + '_ci'][1]:+.3f}]"
                            for k in ("d_base", "d_r6") if k in r)
            print(f"{r['name']:26s} score {r['score']:.3f}  psi1 {r['ev_psi1']:.3f}  psi2 {r['ev_psi2']:.3f}  "
                  f"q1 {r['ev_q1']:.3f}  q2 {r['ev_q2']:.3f}  crps {r['crps']:.3e}  spread q1 {r['spread_q1']:.2f}"
                  + extra)
    else:
        print(run_task(args.task, args.out_dir, args.root, args.n_windows, torch.device(args.device)))


if __name__ == "__main__":
    main()
