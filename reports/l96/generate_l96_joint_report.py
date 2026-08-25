#!/usr/bin/env python3
"""L96 joint state-parameter benchmark report: neural vs DA.

Merges the two joint-estimation artifacts into a single markdown table:

- joint DA baselines  ``experiments/l96_joint_comparison.json``
  (schema: ``{S0|S1: {method: {state_rmse:{slow,obs_fast,mean}, ev, es,
  param_rmse:{F,c1,hx,eps,w1,w2,w3,w4}|None}}}``), produced by
  ``eval_joint_comparison_l96.py`` (EnKF/Joint-EnKF/ETKF/Joint-ETKF/
  Strong-4DVar/Joint-Strong-4DVar).
- joint neural models ``experiments/L{7,8,9}_*/joint_neural_eval.json``
  (schema: ``{metrics: {s0|s1: {rmse, groups, ev, es, param_rmse,
  param_rmse_mean}}}``), produced by ``eval_joint_neural_l96.py``.

Outputs ``reports/l96/outputs/l96_joint_benchmark.md`` with, per case (S0/S1),
a state RMSE / EV / ES table across all methods and a separate param-RMSE table
for the joint methods only, plus S1/S0 state-degradation.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

NEURAL = {
    "L7_joint_cfm_s0s1": ("JointCFM τ=0", "joint_neural_eval.json"),
    "L8_joint_direct_unet_s0s1": ("JointDirectUNet", "joint_neural_eval.json"),
    "L9_joint_cfm_s0s1_multitau": ("JointCFM multi-τ", "joint_neural_eval.json"),
}
PARAMS = ["F", "c1", "hx", "eps", "w1", "w2", "w3", "w4"]


def _load_da(path: Path) -> dict:
    if not path.exists():
        logger.warning("DA comparison JSON not found: %s", path)
        return {}
    return json.load(open(path))


def _load_neural(exp: str, exp_dir: Path) -> dict:
    jpath = exp_dir / exp / NEURAL[exp][1]
    if not jpath.exists():
        logger.warning("Neural eval JSON not found: %s (training/eval incomplete?)", jpath)
        return {}
    return json.load(open(jpath)).get("metrics", {})


def _glyph(v):
    return "--" if v is None else f"{v:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--da-json", default="experiments/l96_joint_comparison.json")
    parser.add_argument("--experiments-dir", default="experiments",
                        help="Experiment root holding L{7,8,9}_*/joint_neural_eval.json "
                             "(relative to repo root, or absolute)")
    parser.add_argument("--out-dir", default="reports/l96/outputs")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "l96_joint_benchmark.md"

    exp_dir = Path(args.experiments_dir)
    if not exp_dir.is_absolute():
        exp_dir = ROOT / args.experiments_dir

    da = _load_da(ROOT / args.da_json)
    neural = {exp: _load_neural(exp, exp_dir) for exp in NEURAL}

    lines = []
    lines.append("# L96 Joint State-Parameter Estimation Benchmark\n")
    lines.append("Neural (L7/L8/L9) vs DA (EnKF/ETKF/Strong-4DVar and their joint "
                 "variants) on the shared cached S0/S1 test set (Obs30, 200 windows, "
                 "24D observed subspace). Each joint method estimates the 24D state "
                 "**and** 8 parameters (F, c1, hx, eps + 4 fast_weights; h fixed).\n")

    # --- State table (RMSE / EV / ES) ---
    lines.append("## State metrics\n")
    lines.append("| Method | Type | S0 RMSE | S1 RMSE | S1/S0 | S0 EV | S1 EV | S0 ES | S1 ES |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    def da_state(case, method, field):
        ent = da.get(case, {}).get(method, {})
        val = ent.get(field)
        if isinstance(val, dict):
            return val.get("mean")
        return val

    def nn_state(case, exp, metric_key):
        m = neural.get(exp, {}).get(case, {})
        if metric_key == "rmse":
            return m.get("rmse")
        return m.get(metric_key, {}).get("groups", {}).get("all_obs")

    # neural rows
    nn_rows = []
    for exp, (label, _) in NEURAL.items():
        s0_rmse = nn_state("s0", exp, "rmse")
        s1_rmse = nn_state("s1", exp, "rmse")
        s0_ev = nn_state("s0", exp, "ev")
        s1_ev = nn_state("s1", exp, "ev")
        s0_es = nn_state("s0", exp, "es")
        s1_es = nn_state("s1", exp, "es")
        deg = (s1_rmse / s0_rmse) if (s0_rmse and s1_rmse and s0_rmse > 0) else None
        nn_rows.append((f"**{label}**", "neural", s0_rmse, s1_rmse, deg, s0_ev, s1_ev, s0_es, s1_es))

    # DA rows (state names in da: EnKF, Joint-EnKF, ...)
    DA_METHODS = ["EnKF", "Joint-EnKF", "ETKF", "Joint-ETKF", "Strong-4DVar", "Joint-Strong-4DVar"]
    da_rows = []
    for method in DA_METHODS:
        s0_rmse = da_state("S0", method, "state_rmse")
        s1_rmse = da_state("S1", method, "state_rmse")
        s0_ev = da_state("S0", method, "ev")
        s1_ev = da_state("S1", method, "ev")
        s0_es = da_state("S0", method, "es")
        s1_es = da_state("S1", method, "es")
        deg = (s1_rmse / s0_rmse) if (s0_rmse and s1_rmse and s0_rmse > 0) else None
        da_rows.append((method, "DA", s0_rmse, s1_rmse, deg, s0_ev, s1_ev, s0_es, s1_es))

    all_rows = nn_rows + da_rows
    # sort: neural first then DA by name, but keep joint/vanilla grouping simple
    for label, typ, s0r, s1r, deg, s0ev, s1ev, s0es, s1es in all_rows:
        lines.append(f"| {label} | {typ} | {_glyph(s0r)} | {_glyph(s1r)} | {_glyph(deg)} "
                     f"| {_glyph(s0ev)} | {_glyph(s1ev)} | {_glyph(s0es)} | {_glyph(s1es)} |")
    lines.append("")

    # --- Param table (joint methods only) ---
    lines.append("## Parameter RMSE (mean over 8 params; joint methods only)\n")
    lines.append("| Method | Type | S0 param-RMSE | S1 param-RMSE |")
    lines.append("|---|---|---|---|")

    for exp, (label, _) in NEURAL.items():
        p0 = neural.get(exp, {}).get("s0", {}).get("param_rmse_mean")
        p1 = neural.get(exp, {}).get("s1", {}).get("param_rmse_mean")
        lines.append(f"| **{label}** | neural | {_glyph(p0)} | {_glyph(p1)} |")

    for method in DA_METHODS:
        if not method.startswith("Joint-"):
            continue
        p0 = da.get("S0", {}).get(method, {}).get("param_rmse")
        p1 = da.get("S1", {}).get(method, {}).get("param_rmse")
        p0m = (sum(p0.values()) / len(p0)) if p0 else None
        p1m = (sum(p1.values()) / len(p1)) if p1 else None
        lines.append(f"| {method} | DA | {_glyph(p0m)} | {_glyph(p1m)} |")
    lines.append("")

    report_path.write_text("\n".join(lines))
    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
