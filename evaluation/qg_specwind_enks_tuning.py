"""EnKS lag and time-taper tuning on val for the QG gyrostat DA (S0 and S1-tuned).

The benchmark EnKS uses a hard lag of 12 analyses, chosen at R x 1
(`docs/results/qg_specwind_enks.md`). This sweep:
- re-tunes the lag under the S1-tuned R x 6 (`docs/results/qg_specwind_s1_tuning.md`);
- tests the time taper (`EnKS(taper_steps=...)`: Gaspari–Cohn in time, zero
  beyond 2τ) instead of a hard lag.

Radius 8, ridge 0.1, EnSRF, 20 val windows of the forced dataset; S1 is the
realistic base.

    python -m evaluation.qg_specwind_enks_tuning --list
    python -m evaluation.qg_specwind_enks_tuning --task 3
    python -m evaluation.qg_specwind_enks_tuning --summarize
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
R_BY_SCEN = {"s0": 1.0, "s1": 6.0}


def configs() -> list[dict]:
    out = []
    for scen in ("s0", "s1"):
        for lag in (12, 24):
            out.append({"scen": scen, "lag": lag, "tau": None})
        if scen == "s1":
            out.append({"scen": scen, "lag": None, "tau": None})
        for tau in (2.0, 4.0, 8.0, 16.0):
            out.append({"scen": scen, "lag": None, "tau": tau})
    return out


def name(c: dict) -> str:
    lag = "all" if c["lag"] is None else c["lag"]
    return f"{c['scen']}_enks_lag{lag}" + (f"_tau{c['tau']:g}d" if c["tau"] is not None else "")


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
    _, per_window = evaluate(windows, da_cfg(cfg, levels), "enks", device, loc_radius=8.0, etkf_ridge=0.1,
                             etkf_loc_mode="ensrf", enks_lag=c["lag"], enks_taper_days=c["tau"],
                             r_scale=R_BY_SCEN[c["scen"]])
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
        row = {"name": nm, **c, "score": float(np.mean([w["score"] for w in pw])),
               "crps": float(np.mean([w["crps"] for w in pw])),
               "spread_q1": float(np.median([w["spread_ratio_q1"] for w in pw]))}
        for k in FIELDS:
            row[f"ev_{k}"] = float(np.mean([w[f"ev_da_{k}"] for w in pw]))
        base = recs.get(f"{c['scen']}_enks_lag12")
        if base is not None:
            d = np.array([w["score"] for w in pw]) - np.array([w["score"] for w in base["per_window"]])
            m = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
            row["d_lag12"] = float(d.mean())
            row["d_lag12_ci"] = [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
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
    p.add_argument("--out-dir", default="experiments/qg_specwind_enks_tuning")
    p.add_argument("--root", default="experiments/qg_datasets")
    p.add_argument("--n-windows", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.list:
        for k, c in enumerate(configs()):
            print(k, name(c))
    elif args.summarize:
        for r in summarize(args.out_dir):
            ci = r.get("d_lag12_ci")
            print(f"{r['name']:24s} score {r['score']:.3f}  psi1 {r['ev_psi1']:.3f}  psi2 {r['ev_psi2']:.3f}  "
                  f"q1 {r['ev_q1']:.3f}  q2 {r['ev_q2']:.3f}  crps {r['crps']:.3e}  spread q1 {r['spread_q1']:.2f}"
                  + (f"  d_lag12 {r['d_lag12']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""))
    else:
        print(run_task(args.task, args.out_dir, args.root, args.n_windows, torch.device(args.device)))


if __name__ == "__main__":
    main()
