"""S1 calibration and attribution on val for the spectral-wind QG DA (ETKF/EnKF, no 4D-Var).

Tasks run the val-tuned DA (`docs/results/qg_specwind_da2_val_tuning.md`) on
the first `--n-windows` val windows of the forced dataset under S1 errors
(`evaluation/qg_specwind_s1.py`) at intensity kappa x `REFERENCE`.

* `--phase calib`: S0; each component alone at kappa 1, 2, 4; all components
  at kappa 0.5, 1, 1.5, 2, 3, 4. Used to pick kappa so that the upper-layer
  analysis EVs are typically q1 in [0, 0.25] and psi1 in [0.7, 0.9].
* `--phase shapley --kappa K`: all 16 on/off combinations of the four
  components at kappa K, for an exact additive (Shapley) attribution of the
  degradation from S0 to full S1.
* `--phase realistic_calib`: the realism-anchored scenario (`REALISTIC`,
  groups forcing / rd / drag / obs / res) at the variants in `VARIANTS`
  (realistic ranges: lower edge, base, upper edge, DA grid 64/48/32).
* `--phase realistic_shapley --variant V`: all 32 on/off combinations of the
  five groups at variant V.

    python -m evaluation.qg_specwind_s1_sweep --phase calib --list
    python -m evaluation.qg_specwind_s1_sweep --phase calib --task 3
    python -m evaluation.qg_specwind_s1_sweep --summarize [--kappa K]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time

import numpy as np
import torch

from data.qg_datasets import SPECS
from evaluation.qg_specwind_s1 import COMPONENTS, REALISTIC_GROUPS, REALISTIC_VARIANTS, S1Levels
from evaluation.run_qg_specwind_da import (
    FIELDS,
    apply_s1,
    build_cfg,
    da_cfg,
    evaluate,
    s0_windows,
    s1_levels_from,
)

SPEC = "qg_specwind_gyrostat_v1"
DA = {"method": "etkf", "loc_radius": 8.0, "etkf_ridge": 1.0, "loc_cross_layer": 1.0,
      "init_ensemble": "bred"}
TARGETS = {"q1": (0.0, 0.25), "psi1": (0.7, 0.9)}


VARIANTS = REALISTIC_VARIANTS


def realistic_tasks(phase: str, variant: str | None = None) -> list[dict]:
    if phase == "realistic_calib":
        return [{"name": f"s1r_{v}", "levels": lv, "components": list(REALISTIC_GROUPS)}
                for v, lv in VARIANTS.items()]
    if variant not in VARIANTS:
        raise ValueError(f"--phase realistic_shapley needs --variant from {sorted(VARIANTS)}")
    out = []
    for r in range(len(REALISTIC_GROUPS) + 1):
        for comps in itertools.combinations(REALISTIC_GROUPS, r):
            out.append({"name": realistic_name(variant, comps),
                        "levels": VARIANTS[variant].only(comps), "components": list(comps)})
    return out


def realistic_name(variant: str, comps) -> str:
    if not comps:
        return "s0"
    if tuple(comps) == REALISTIC_GROUPS:
        return f"s1r_{variant}"
    return f"s1r_{variant}_{'+'.join(comps)}"


def tasks(phase: str, kappa: float | None = None) -> list[tuple[float, tuple[str, ...]]]:
    if phase == "calib":
        out = [(0.0, ())]
        out += [(k, (c,)) for c in COMPONENTS for k in (1.0, 2.0, 4.0)]
        out += [(k, COMPONENTS) for k in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)]
        return out
    if phase == "shapley":
        if kappa is None:
            raise ValueError("--phase shapley needs --kappa")
        return [(kappa if comps else 0.0, comps) for r in range(len(COMPONENTS) + 1)
                for comps in itertools.combinations(COMPONENTS, r)]
    raise ValueError(f"unknown phase {phase!r}")


def task_name(kappa: float, comps: tuple[str, ...], disp_frac: float) -> str:
    base = "s0" if not comps or kappa == 0.0 else f"s1_k{kappa:g}_{'+'.join(comps)}"
    return base + (f"_d{disp_frac:g}" if disp_frac != 1.0 else "")


def run_task(kappa: float, comps: tuple[str, ...], out_dir: str, root: str, n_windows: int,
             device: torch.device, disp_frac: float, method: str, loc_radius: float) -> str:
    name = task_name(kappa, comps, disp_frac)
    levels = s1_levels_from(kappa, comps)
    return run_levels(name, levels if comps and kappa else None, kappa, list(comps), out_dir,
                      root, n_windows, device, disp_frac, method, loc_radius)


def run_levels(name: str, levels: S1Levels | None, kappa: float, comps: list[str], out_dir: str,
               root: str, n_windows: int, device: torch.device, disp_frac: float, method: str,
               loc_radius: float) -> str:
    path = os.path.join(out_dir, f"{name}.json")
    if os.path.exists(path):
        return path
    spec = SPECS[SPEC]
    cfg = build_cfg(spec, 3, 0.05, 5.0)
    idx = list(range(n_windows))
    t0 = time.time()
    windows, report = s0_windows(spec, "val", root, idx, cfg, device)
    if levels is not None:
        windows = apply_s1(windows, spec, levels)
    load_s = time.time() - t0
    da = {**DA, "method": method, "loc_radius": loc_radius}
    t0 = time.time()
    _, per_window = evaluate(windows, da_cfg(cfg, levels), da["method"], device,
                             loc_radius=da["loc_radius"],
                             etkf_ridge=da["etkf_ridge"], loc_cross_layer=da["loc_cross_layer"],
                             init_ensemble=da["init_ensemble"], disp_frac=disp_frac)
    rec = {"name": name, "kappa": kappa, "components": list(comps),
           "levels": levels.as_dict() if levels is not None else None,
           "da": {**da, "disp_frac": disp_frac}, "indices": idx, "load": report,
           "load_seconds": round(load_s, 1), "da_seconds": round(time.time() - t0, 1),
           "per_window": per_window}
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1)
    return path


def _boot(x: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _load(out_dir: str) -> dict:
    recs = {}
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".json") and not f.startswith("summary"):
            with open(os.path.join(out_dir, f)) as fh:
                r = json.load(fh)
            recs[r["name"]] = r
    return recs


def _metric(rec: dict, key: str) -> np.ndarray:
    if key == "score":
        return np.array([w["score"] for w in rec["per_window"]])
    return np.array([w[f"ev_da_{key}"] for w in rec["per_window"]])


def shapley(recs: dict, kappa: float, disp_frac: float = 1.0) -> dict | None:
    """Per-metric Shapley attribution of the S0 -> full-S1 EV loss at kappa (None if incomplete)."""
    return shapley_over(recs, COMPONENTS,
                        lambda c: task_name(kappa if c else 0.0, c, disp_frac))


def shapley_over(recs: dict, components, name_of) -> dict | None:
    """Shapley attribution over `components`; `name_of(subset)` names each subset's record."""
    COMPONENTS = tuple(components)  # noqa: N806
    n = len(COMPONENTS)
    subsets = [c for r in range(n + 1) for c in itertools.combinations(COMPONENTS, r)]
    names = {c: name_of(c) for c in subsets}
    if any(nm not in recs for nm in names.values()):
        return None
    out = {}
    for key in ("score",) + FIELDS:
        v = {c: _metric(recs[names[c]], key) for c in subsets}
        loss_full = v[()] - v[tuple(COMPONENTS)]
        comp = {}
        for i, ci in enumerate(COMPONENTS):
            phi = np.zeros_like(loss_full)
            others = [c for c in COMPONENTS if c != ci]
            for r in range(n):
                for s in itertools.combinations(others, r):
                    with_i = tuple(c for c in COMPONENTS if c in s or c == ci)
                    wgt = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                    phi += wgt * (v[s] - v[with_i])
            alone = v[()] - v[(ci,)]
            comp[ci] = {"shapley": float(phi.mean()), "shapley_ci": _boot(phi),
                        "alone": float(alone.mean()), "alone_ci": _boot(alone),
                        "share": float(phi.mean() / loss_full.mean()) if loss_full.mean() else None}
        interaction = loss_full - sum(v[()] - v[(c,)] for c in COMPONENTS)
        out[key] = {"s0": float(v[()].mean()), "s1": float(v[tuple(COMPONENTS)].mean()),
                    "loss": float(loss_full.mean()), "loss_ci": _boot(loss_full),
                    "interaction": float(interaction.mean()), "components": comp}
    return out


def summarize(out_dir: str, kappa: float | None, disp_frac: float,
              variant: str | None = None) -> dict:
    recs = _load(out_dir)
    rows = []
    for name, r in recs.items():
        row = {"name": name, "kappa": r["kappa"], "components": r["components"],
               "n": len(r["per_window"])}
        for key in FIELDS + ("score",):
            x = _metric(r, key)
            row[f"mean_{key}"] = float(x.mean())
            row[f"median_{key}"] = float(np.median(x))
            row[f"iqr_{key}"] = [float(np.percentile(x, 25)), float(np.percentile(x, 75))]
        for key, (lo, hi) in TARGETS.items():
            x = _metric(r, key)
            row[f"in_target_{key}"] = float(((x >= lo) & (x <= hi)).mean())
        rows.append(row)
    rows.sort(key=lambda x: (len(x["components"]) != len(COMPONENTS), x["kappa"], x["name"]))
    out = {"rows": rows, "targets": TARGETS}
    if kappa is not None:
        out["shapley"] = shapley(recs, kappa, disp_frac)
    if variant is not None:
        out["shapley"] = shapley_over(recs, REALISTIC_GROUPS,
                                      lambda c: realistic_name(variant, c))
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--phase", default="calib",
                   choices=("calib", "shapley", "realistic_calib", "realistic_shapley"))
    p.add_argument("--variant", default=None, choices=sorted(VARIANTS))
    p.add_argument("--kappa", type=float, default=None)
    p.add_argument("--list", action="store_true")
    p.add_argument("--task", type=int)
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--out-dir", default="experiments/qg_specwind_s1")
    p.add_argument("--root", default="experiments/qg_datasets")
    p.add_argument("--n-windows", type=int, default=20)
    p.add_argument("--disp-frac", type=float, default=1.0)
    p.add_argument("--method", default=DA["method"], choices=("etkf", "enkf"))
    p.add_argument("--loc-radius", type=float, default=DA["loc_radius"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.summarize:
        s = summarize(args.out_dir, args.kappa, args.disp_frac, args.variant)
        for r in s["rows"]:
            print(f"{r['name']:40s} psi1 med {r['median_psi1']:.3f} [{r['iqr_psi1'][0]:.2f},"
                  f"{r['iqr_psi1'][1]:.2f}] in {r['in_target_psi1']:.0%}  q1 med {r['median_q1']:.3f} "
                  f"[{r['iqr_q1'][0]:.2f},{r['iqr_q1'][1]:.2f}] in {r['in_target_q1']:.0%}  "
                  f"psi2 {r['mean_psi2']:.3f} q2 {r['mean_q2']:.3f} score {r['mean_score']:.3f}")
        if s.get("shapley"):
            for key, d in s["shapley"].items():
                parts = "  ".join(f"{c} {v['shapley']:+.3f} ({v['share']:.0%})"
                                  for c, v in d["components"].items())
                print(f"shapley {key:5s} S0 {d['s0']:.3f} -> S1 {d['s1']:.3f} (loss {d['loss']:.3f}): "
                      f"{parts}  interaction {d['interaction']:+.3f}")
        return
    out_dir = args.out_dir if args.method == "etkf" else os.path.join(args.out_dir, args.method)
    if args.phase.startswith("realistic"):
        todo_r = realistic_tasks(args.phase, args.variant)
        if args.list:
            for k, t in enumerate(todo_r):
                print(k, t["name"])
            return
        t = todo_r[args.task]
        print(run_levels(t["name"], t["levels"] if t["components"] else None, 1.0, t["components"],
                         out_dir, args.root, args.n_windows, torch.device(args.device),
                         args.disp_frac, args.method, args.loc_radius))
        return
    todo = tasks(args.phase, args.kappa)
    if args.list:
        for k, (kap, comps) in enumerate(todo):
            print(k, task_name(kap, comps, args.disp_frac))
        return
    kap, comps = todo[args.task]
    print(run_task(kap, comps, out_dir, args.root, args.n_windows, torch.device(args.device),
                   args.disp_frac, args.method, args.loc_radius))


if __name__ == "__main__":
    main()
