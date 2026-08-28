#!/usr/bin/env python3
"""L96 joint state-parameter DA benchmark report builder.

Consumes the JSON written by ``eval_joint_comparison_l96.py``:
``experiments/l96_joint_comparison.json`` with schema
``results[label][method] = {state_rmse{slow,obs_fast,mean}, ev{...}, es{...},
param_rmse{F,c1,hx,eps,w1..w4}}``.

Generates ``reports/l96/outputs/l96_joint_da_benchmark.md``: state RMSE/EV/ES
per case and group for every benchmarked method, per-param RMSE (+NRMSE) for the
joint methods, and context anchors against the L9 joint neural baseline (ens30x10).
Missing values render as ``--`` without crashing. Method rows are read from the
JSON, so adding e.g. Joint-EnKF / Joint-Strong-4DVar to the comparator is enough
for them to appear in the report.
"""
import argparse
import json
import logging
import math
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = ROOT / "experiments/l96_joint_comparison.json"
DEFAULT_OUT = ROOT / "reports/l96/outputs/l96_joint_da_benchmark.md"

PARAM_NAMES = ["F", "c1", "hx", "eps", "w1", "w2", "w3", "w4"]
CASES = ["S0", "S1"]
GROUPS = ["slow", "obs_fast", "mean"]
ESTIMATED_PARAMS = {"w1", "w2", "w3", "w4"}


def _method_kind(m):
    """Return ('joint'|'vanilla', 'EnKF'|'ETKF'|'Strong-4DVar') from a method name."""
    kind = "Strong-4DVar" if "4DVar" in m or "Strong" in m else ("EnKF" if "EnKF" in m else "ETKF")
    return ("joint" if "Joint" in m else "vanilla", kind)


def load_json(path: Path):
    if not path.exists():
        logger.warning("JSON not found: %s", path)
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read %s: %s", path, e)
        return None


def fmt_num(x, missing="--", ndigits=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return missing
    return f"{x:.{ndigits}f}"


def _case_entry(data, case):
    if data is None:
        return None
    return data.get(case)


def _da_best(data, case, metric, group="mean", higher_is_better=False):
    vals = []
    for method, entry in (data.get(case) or {}).items():
        band = (entry or {}).get(metric) or {}
        v = band.get(group)
        if isinstance(v, (int, float)):
            vals.append((method, float(v)))
    if not vals:
        return None
    vals.sort(key=lambda x: x[1], reverse=higher_is_better)
    return vals[0][0]


def _neural_anchor(exp_dir: Path, exp_name: str, case: str, key_rmse: str = "rmse",
                   key_params: str = "param_rmse"):
    """Return (state_rmse, param_rmse_mean) for a neural model from its single-sample JSON."""
    p = exp_dir / exp_name / "joint_neural_eval.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    m = (data.get("metrics") or {}).get(case.lower())
    if m is None:
        return None
    params = m.get(key_params) or {}
    pmean = params.get("mean")
    if pmean is None and params:
        pvals = [v for v in params.values() if isinstance(v, (int, float))]
        pmean = float(sum(pvals) / len(pvals)) if pvals else None
    return (m.get(key_rmse), pmean)


def write_report(exp_dir: Path, json_path: Path, output_path: Path) -> None:
    data = load_json(json_path)
    md = []

    md.append("# L96 Joint State-Parameter DA Benchmark")
    md.append("")
    md.append("**System:** Lorenz-96 two-scale (NO=8, J=4), observed subspace state_dim=24 "
              "(Obs30, obs_interval=100), 200 shared test windows.")
    md.append("")
    md.append("**Models:** joint ensemble filters (Joint-EnKF / Joint-ETKF) estimate the 24D "
              "observed state **and** the 8 model parameters `F, c1, hx, eps, w1..w4` (fast "
              "weights per index; h fixed) via an augmented-state ensemble, benchmarked "
              "against their state-only vanilla counterparts on the same cached S0/S1 test "
              "set. On S1 (reduced 24D J=2 dynamics) the DA only carries `w1,w2`; `w3,w4` "
              "default to the reference prior `[1.0, 0.1]` and are marked with a `†` (not "
              "estimated).")
    md.append("")
    md.append("---")
    md.append("")

    if data is None:
        md.append("_No `l96_joint_comparison.json` found; nothing to render._")
        md.append("")
    else:
        methods = []
        seen = set()
        for c in CASES:
            for m in (data.get(c) or {}):
                if m not in seen:
                    seen.add(m)
                    methods.append(m)

        # Methods table
        md.append("## Benchmarked methods")
        md.append("")
        md.append("| Method | Type | Describes state + 8 params? |")
        md.append("|---|---|---|")
        for m in methods:
            scope, kind = _method_kind(m)
            md.append(f"| {m} | {scope} {kind} | "
                      f"{'yes' if scope == 'joint' else 'no (state only)'} |")
        md.append("")
        md.append("---")
        md.append("")

        # State RMSE table per case
        md.append("## State RMSE (per case)")
        md.append("")
        md.append("Pooled RMSE over the observed subspace, grouped slow (8D) / obs_fast (16D) / "
                  "mean (24D). Lower is better.")
        md.append("")
        md.append("| Method | S0 slow | S0 obs_fast | S0 mean | S1 slow | S1 obs_fast | S1 mean |")
        md.append("|---|---|---|---|---|---|---|")
        for m in methods:
            cells = [f"| {m} |"]
            for case in CASES:
                e = _case_entry(data, case) or {}
                band = (e.get(m) or {}).get("state_rmse") or {}
                for g in GROUPS:
                    cells.append(f" {fmt_num(band.get(g))} |")
            md.append("".join(cells))
        md.append("")
        md.append("*Best is the lowest per column; rendered from the comparator JSON.*")
        md.append("")
        md.append("---")
        md.append("")

        # ES table per case (whole-group mean only, since the comparator stores group es)
        md.append("## Energy Score (ES, per case)")
        md.append("")
        md.append("N=30 ensemble Energy Score on the observed subspace (subsampled to 24D). "
                  "Lower is better.")
        md.append("")
        md.append("| Method | S0 ES | S1 ES |")
        md.append("|---|---|---|")
        for m in methods:
            cells = [f"| {m} |"]
            for case in CASES:
                e = _case_entry(data, case) or {}
                band = (e.get(m) or {}).get("es") or {}
                cells.append(f" {fmt_num(band.get('mean'))} |")
            md.append("".join(cells))
        md.append("")
        md.append("---")
        md.append("")

        # EV table per case
        md.append("## Explained variance (EV, per case)")
        md.append("")
        md.append("Pooled explained variance over the observed subspace, grouped slow (8D) / "
                  "obs_fast (16D) / mean (24D). Higher is better.")
        md.append("")
        md.append("| Method | S0 slow | S0 obs_fast | S0 mean | S1 slow | S1 obs_fast | S1 mean |")
        md.append("|---|---|---|---|---|---|---|")
        for m in methods:
            cells = [f"| {m} |"]
            for case in CASES:
                e = _case_entry(data, case) or {}
                band = (e.get(m) or {}).get("ev") or {}
                for g in GROUPS:
                    cells.append(f" {fmt_num(band.get(g))} |")
            md.append("".join(cells))
        md.append("")
        md.append("---")
        md.append("")

        # Parameter RMSE table (joint only)
        joint_methods = [m for m in methods if "Joint" in m]
        if joint_methods:
            for case in CASES:
                md.append(f"## Parameter RMSE — {case}")
                md.append("")
                md.append("Per-parameter RMSE (`F, c1, hx, eps, w1..w4`) against the per-window "
                          "true params. `†` marks fast weights defaulted to the reference prior "
                          "(not estimated) on the J=2 S1 dynamics.")
                md.append("")
                header = "| Method | " + " | ".join(PARAM_NAMES) + " | mean |"
                sep = "|---|" + "---|" * len(PARAM_NAMES) + "---|"
                md.append(header)
                md.append(sep)
                for m in joint_methods:
                    e = _case_entry(data, case) or {}
                    prmse = (e.get(m) or {}).get("param_rmse") or {}
                    pvals = [v for v in prmse.values() if isinstance(v, (int, float))]
                    pmean = float(sum(pvals) / len(pvals)) if pvals else None
                    cells = [f"| {m} |"]
                    for i, p in enumerate(PARAM_NAMES):
                        v = prmse.get(p)
                        dagger = "†" if (case == "S1" and p in ESTIMATED_PARAMS) else ""
                        cells.append(f" {fmt_num(v)}{dagger} |")
                    cells.append(f" {fmt_num(pmean)} |")
                    md.append("".join(cells))
                md.append("")
                md.append("---")
                md.append("")

        # Context anchors vs L9 neural (ens30x10)
        md.append("## Context: L9 joint neural baseline (ens30 × 10)")
        md.append("")
        md.append("Single-sample L9 `JointCFM` multi-tau state RMSE and param-RMSE mean, for "
                  "reference against the DA rows. (Full neural tables live in ")
        md.append("`l96_joint_neural_benchmark.md`.)")
        md.append("")
        md.append("| Case | L9 state RMSE | L9 param-RMSE mean | Best DA state RMSE |")
        md.append("|---|---|---|---|")
        for case in CASES:
            l9 = _neural_anchor(exp_dir, "L9_joint_cfm_s0s1_multitau", case.casefold())
            best_da = None
            band = None
            for m in methods:
                e = _case_entry(data, case) or {}
                b = (e.get(m) or {}).get("state_rmse") or {}
                v = b.get("mean")
                if isinstance(v, (int, float)) and (band is None or v < band):
                    band = float(v)
                    best_da = m
            l9s = fmt_num(l9[0]) if l9 else "--"
            l9p = fmt_num(l9[1]) if l9 else "--"
            md.append(f"| {case} | {l9s} | {l9p} | "
                      f"{fmt_num(band)} ({best_da if best_da else '--'}) |")
        md.append("")
        md.append("*Best DA state RMSE is the minimum across the benchmarked methods for that case.*")
        md.append("")
        md.append("---")
        md.append("")

    md.append("## Consistency check")
    md.append("")
    md.append("The comparator loads the cached `l96_datasets_obsj2_int100_nwin200.pt` and runs "
              "`evaluate_baseline` (the same code path as the vanilla DA caches). State RMSE/EV/"
              "ES are pooled over all 200 windows and subsampled to `obs_var_indices` (24D); "
              "param RMSE compares each joint filter's 8-wide estimate against `true_*` "
              "(padded to 8 with the reference prior on S1).")
    md.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(md))
    logger.info("Report written to %s", output_path)
    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate the L96 joint DA benchmark report")
    parser.add_argument("--json", type=str, default=str(DEFAULT_JSON),
                        help="Path to the comparator JSON (default: experiments/l96_joint_comparison.json)")
    parser.add_argument("--exp-dir", type=str, default=str(ROOT / "experiments"),
                        help="Experiments dir (for neural anchor JSONs)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUT),
                        help="Output markdown report path")
    args = parser.parse_args()

    json_path = Path(args.json)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_report(Path(args.exp_dir), json_path, out)


if __name__ == "__main__":
    main()
