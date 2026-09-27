"""Report: S0/S1 ETKF/EnKF on the spectral-wind QG test sets (DA-3/DA-4).

Reads the per-shard `*_meta.json` written by `evaluation.run_qg_specwind_da`
(per-window EVs computed with each window's own parameters) under
`<root>/<spec>/test/<config>/`, joins the dataset manifests for the window
factors, and writes `reports/qg/outputs/qg_specwind_da_report.md`.

Tables: main (3 columns/day) and bridge (4 columns/day) per dataset and
method with the free forecast; the forced-vs-coupled difference (D2); the
density curve (D4); factor strata (D3); S1 against S0.

    python reports/qg/generate_qg_specwind_da_report.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

FIELDS = ("psi1", "psi2", "q1", "q2")
LABEL = {"psi1": "ψ₁", "psi2": "ψ₂", "q1": "q₁", "q2": "q₂", "score": "score"}
SPECS = {"qg_specwind_gyrostat_v1": "forced", "qg_coupled_gyrostat_v1": "coupled"}


def load_runs(root: str) -> list[dict]:
    groups = defaultdict(list)
    for f in glob.glob(os.path.join(root, "*", "test", "*", "shard_*_meta.json")):
        groups[os.path.dirname(f)].append(f)
    runs = []
    for d, files in sorted(groups.items()):
        metas = [json.load(open(f)) for f in sorted(files)]
        if any("per_window" not in m for m in metas):
            continue
        m0 = metas[0]
        n_shards = int(os.path.basename(files[0]).split("_of_")[1].split("_")[0])
        pw = sorted((w for m in metas for w in m["per_window"]), key=lambda w: w["index"])
        runs.append({"dir": d, "spec": m0["spec"], "method": m0["method"],
                     "cols": m0["cols_per_day"], "kappa": m0.get("s1_kappa", 0.0) or 0.0,
                     "s1": (f"realistic {m0['s1_variant']}" if m0.get("s1_variant")
                            else (f"κ = {m0['s1_kappa']:g}" if m0.get("s1_kappa") else "S0")),
                     "loc": m0["loc_radius"], "complete": len(metas) == n_shards,
                     "n_shards": n_shards, "shards": len(metas), "per_window": pw,
                     "da_seconds": sum(m["da_seconds"] for m in metas)})
    return runs


def manifest_factors(ds_root: str) -> dict:
    out = {}
    for spec in SPECS:
        path = os.path.join(ds_root, spec, "test", "manifest.json")
        if os.path.exists(path):
            m = json.load(open(path))
            out[spec] = {w["index"]: w for w in m["windows"]}
    return out


def _boot_mean(x: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _values(run: dict, tag: str, key: str) -> np.ndarray:
    if key == "score":
        return np.array([np.mean([w[f"ev_{tag}_{f}"] for f in FIELDS]) for w in run["per_window"]])
    return np.array([w[f"ev_{tag}_{key}"] for w in run["per_window"]])


def _rank_marks(cells: list[list[float | None]]) -> list[list[str]]:
    """Bold the best and italicize the second-best value per column (higher is better)."""
    out = [["" for _ in row] for row in cells]
    for j in range(len(cells[0])):
        col = sorted({round(r[j], 3) for r in cells if r[j] is not None}, reverse=True)
        for i, r in enumerate(cells):
            if r[j] is None:
                continue
            v = round(r[j], 3)
            out[i][j] = "**" if col and v == col[0] else ("*" if len(col) > 1 and v == col[1] else "")
    return out


def _table(header: list[str], rows: list[tuple[str, list[float | None], list[str]]]) -> list[str]:
    marks = _rank_marks([r[1] for r in rows])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for (name, vals, extras), mk in zip(rows, marks):
        cells = [f"{mk[j]}{v:.3f}{mk[j]}" if v is not None else "—" for j, v in enumerate(vals)]
        lines.append("| " + " | ".join([name] + cells + extras) + " |")
    return lines


def _find(runs, spec, method, cols, kappa=0.0):
    hits = [r for r in runs if r["spec"] == spec and r["method"] == method and r["cols"] == cols
            and abs(r["kappa"] - kappa) < 1e-9 and r["complete"]]
    return hits[0] if hits else None


def main_table(runs, cols: int, kappa: float) -> list[str]:
    rows = []
    for spec, lab in SPECS.items():
        base = None
        for method in ("etkf", "enkf"):
            r = _find(runs, spec, method, cols, kappa)
            if r is None:
                continue
            base = base or r
            vals = [float(_values(r, "da", k).mean()) for k in FIELDS + ("score",)]
            ci = _boot_mean(_values(r, "da", "score"))
            rows.append((f"{lab} · {method.upper()} (radius {r['loc']:g})", vals,
                         [f"[{ci[0]:.3f}, {ci[1]:.3f}]", str(len(r["per_window"]))]))
        if base is not None:
            vals = [float(_values(base, "free", k).mean()) for k in FIELDS + ("score",)]
            rows.append((f"{lab} · free forecast", vals, ["", str(len(base["per_window"]))]))
    if not rows:
        return ["_No complete runs yet._"]
    return _table(["run"] + [LABEL[k] for k in FIELDS] + ["score", "score 95% CI", "windows"], rows)


def d2_table(runs, cols: int, kappa: float) -> list[str]:
    lines = ["| method | field | forced | coupled | forced − coupled [95% CI] |", "|---|---|---|---|---|"]
    rng = np.random.default_rng(1)
    any_row = False
    for method in ("etkf", "enkf"):
        a = _find(runs, "qg_specwind_gyrostat_v1", method, cols, kappa)
        b = _find(runs, "qg_coupled_gyrostat_v1", method, cols, kappa)
        if a is None or b is None:
            continue
        for k in FIELDS + ("score",):
            x, y = _values(a, "da", k), _values(b, "da", k)
            d = (x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
                 - y[rng.integers(0, len(y), (10000, len(y)))].mean(1))
            lines.append(f"| {method.upper()} | {LABEL[k]} | {x.mean():.3f} | {y.mean():.3f} | "
                         f"{x.mean() - y.mean():+.3f} [{np.percentile(d, 2.5):+.3f}, "
                         f"{np.percentile(d, 97.5):+.3f}] |")
            any_row = True
    return lines if any_row else ["_Needs complete forced and coupled runs._"]


def density_table(runs, kappa: float) -> list[str]:
    rows = []
    for cols in sorted({r["cols"] for r in runs}):
        r = _find(runs, "qg_specwind_gyrostat_v1", "etkf", cols, kappa)
        if r is None:
            continue
        vals = [float(_values(r, "da", k).mean()) for k in FIELDS + ("score",)]
        rows.append((f"{cols} ({100 * cols / 64:.1f}% of columns)", vals, [str(len(r["per_window"]))]))
    if not rows:
        return ["_No complete runs yet._"]
    return _table(["columns per day"] + [LABEL[k] for k in FIELDS] + ["score", "windows"], rows)


def strata_table(runs, factors, cols: int, kappa: float) -> list[str]:
    r = _find(runs, "qg_specwind_gyrostat_v1", "etkf", cols, kappa)
    if r is None or "qg_specwind_gyrostat_v1" not in factors:
        return ["_Needs the complete forced ETKF run._"]
    man = factors["qg_specwind_gyrostat_v1"]
    pw = r["per_window"]
    level = np.array([man[w["index"]]["factors"]["level"] for w in pw])
    unit = np.array([man[w["index"]]["factors"]["time_unit_days"] for w in pw])
    regime = np.array([man[w["index"]]["regime"] for w in pw])
    windy = level > 0
    lt = np.percentile(level[windy], [100 / 3, 200 / 3]) if windy.sum() >= 3 else [np.inf, np.inf]
    ut = np.percentile(unit, [100 / 3, 200 / 3])
    strata = [("calm (level 0)", ~windy), ("windy", windy),
              ("windy, level tercile 1", windy & (level <= lt[0])),
              ("windy, level tercile 2", windy & (level > lt[0]) & (level <= lt[1])),
              ("windy, level tercile 3", windy & (level > lt[1])),
              ("time unit tercile 1 (fast)", unit <= ut[0]),
              ("time unit tercile 2", (unit > ut[0]) & (unit <= ut[1])),
              ("time unit tercile 3 (slow)", unit > ut[1]),
              ("regime 0 or 15", np.isin(regime, (0, 15))), ("other regimes", ~np.isin(regime, (0, 15)))]
    lines = ["| stratum | n | " + " | ".join(f"{LABEL[k]} DA (free)" for k in FIELDS) + " |",
             "|---|---|" + "---|" * len(FIELDS)]
    for name, mask in strata:
        if mask.sum() == 0:
            continue
        cells = []
        for k in FIELDS:
            da = np.array([w[f"ev_da_{k}"] for w in pw])[mask]
            fr = np.array([w[f"ev_free_{k}"] for w in pw])[mask]
            cells.append(f"{da.mean():.3f} ({fr.mean():.3f})")
        lines.append(f"| {name} | {int(mask.sum())} | " + " | ".join(cells) + " |")
    return lines


def s1_table(runs, cols: int) -> list[str]:
    lines = ["| dataset · method | S1 | " + " | ".join(f"{LABEL[k]} S0 → S1" for k in FIELDS)
             + " | S1 windows in target (q₁ ∈ [0, 0.25], ψ₁ ∈ [0.7, 0.9]) |",
             "|---|---|" + "---|" * len(FIELDS) + "---|"]
    any_row = False
    for spec, lab in SPECS.items():
        for method in ("etkf", "enkf"):
            s0 = _find(runs, spec, method, cols, 0.0)
            for r in runs:
                if (r["spec"], r["method"], r["cols"]) != (spec, method, cols) or not r["kappa"] \
                        or not r["complete"] or s0 is None:
                    continue
                cells = [f"{_values(s0, 'da', k).mean():.3f} → {_values(r, 'da', k).mean():.3f}"
                         for k in FIELDS]
                q1, p1 = _values(r, "da", "q1"), _values(r, "da", "psi1")
                cells.append(f"q₁ {np.mean((q1 >= 0) & (q1 <= 0.25)):.0%}, "
                             f"ψ₁ {np.mean((p1 >= 0.7) & (p1 <= 0.9)):.0%}")
                lines.append(f"| {lab} · {method.upper()} | {r['s1']} | " + " | ".join(cells) + " |")
                any_row = True
    return lines if any_row else ["_No complete S1 test runs yet._"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="experiments/qg_specwind_da")
    p.add_argument("--datasets", default="experiments/qg_datasets")
    p.add_argument("--out", default="reports/qg/outputs/qg_specwind_da_report.md")
    p.add_argument("--s1-kappa", type=float, default=None,
                   help="S1 intensity whose test runs feed the S1 sections (default: none)")
    args = p.parse_args()
    runs = load_runs(args.root)
    factors = manifest_factors(args.datasets)
    incomplete = [f"`{os.path.relpath(r['dir'], args.root)}` ({r['shards']}/{r['n_shards']})"
                  for r in runs if not r["complete"]]
    md = ["# S0/S1 ETKF/EnKF on the spectral-wind QG test sets", "",
          "Generated by `reports/qg/generate_qg_specwind_da_report.py` from "
          f"`{args.root}` (per-window metrics from `evaluation/run_qg_specwind_da.py`). "
          "Design: `docs/plans/analysis/qg_specwind_da_s0.md`, `docs/plans/analysis/qg_specwind_da_s1.md`.",
          "", "EV = 1 − MSE / Var(truth) per window over its 30 days, averaged over windows; "
          "ψ inverted with each window's own parameters. score = mean of the four EVs. "
          "Bold: best per column; italics: second best. Observations: upper-layer ψ₁ on random "
          "meridional columns, each once per day, 5% noise; initial state = truth lagged about 5 days; "
          "ETKF radius 8 / ridge 1, EnKF radius 6; cross-layer weight 1; bred initial ensemble; N = 80.", ""]
    if incomplete:
        md += ["**Incomplete runs (excluded):** " + ", ".join(incomplete), ""]
    md += ["## S0 — main runs (3 columns per day, 4.7%)", ""] + main_table(runs, 3, 0.0)
    md += ["", "## S0 — bridge density (4 columns per day, 6.3%)", ""] + main_table(runs, 4, 0.0)
    md += ["", "## D2 — forced vs coupled (S0, 3 columns per day)", "",
           "Independent samples (paired windows share seeds but not trajectories), two-sample bootstrap.",
           ""] + d2_table(runs, 3, 0.0)
    md += ["", "## D4 — observation density (S0, forced, ETKF)", ""] + density_table(runs, 0.0)
    md += ["", "## D3 — factor strata (S0, forced, ETKF, 3 columns per day)", ""] + \
        strata_table(runs, factors, 3, 0.0)
    if args.s1_kappa is not None:
        s1_label = next((r["s1"] for r in runs if abs(r["kappa"] - args.s1_kappa) < 1e-9
                         and r["kappa"]), f"κ = {args.s1_kappa:g}")
        md += ["", f"## S1 ({s1_label}) — main runs (3 columns per day)", ""] + \
            main_table(runs, 3, args.s1_kappa)
        md += ["", "## S1 against S0", ""] + s1_table(runs, 3)
        md += ["", f"## D3 — factor strata (S1 {s1_label}, forced, ETKF)", ""] + \
            strata_table(runs, factors, 3, args.s1_kappa)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
