#!/usr/bin/env python3
"""L96 DA-baseline observation-density report builder.

Compares two observation configurations over the SAME 200-window cached S0/S1
test set (identical dynamics / truth / params; only ``obs`` changes):

* **obsj2** (canonical): all 24D observed (8 slow X + 16 fast Y1,Y2 per node).
* **slow-only obsj0**: only the 8 slow X observed (no fast Y).

Both are evaluated on the **same 24D eval subspace** (slow + first-2-fast), so
the slow / obs_fast / all_obs metric groups are directly comparable — the
"obs_fast" group on the slow-only config reflects fast variables NOT observed by
the DA (a stress test of slow-only observation).

Consumes (all produced by this session's runs with the S1 ``case=2`` corrupted-
forcing fix):
* state-only DA: ``l96_baselines_dws500_*_obsj{2,0}_int100*.json``
* joint  DA:     ``l96_joint_comparison{,_slowobs}.json``
* SDA (score-based DA, guidance-only): ``{SDA1_prior_l96,SDA2_cond_mixed_l96,
  SDA2_cond_nominal_l96}{,_slowobs}/ens30_no10/neural_eval.json`` -- no
  retraining or new dataset needed for the slow-only variant (see
  ``evaluation/sda_sampler.py``'s ``obs_indices``): the same obsj2 checkpoint
  and cached test set are reused, restricting only which channels the DPS
  guidance cost is allowed to see.

Generates ``reports/l96/outputs/l96_obs_density_da_baselines.md``.
"""
import json
import logging
import math
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Canonical (obsj2) artifacts
CANON_STATE_JSON = ROOT / "experiments/l96_baselines_dws500_s0c_inf2.0_etkf_inf2.0_obsj2_int100_fw.json"
CANON_JOINT_JSON = ROOT / "experiments/l96_joint_comparison.json"
# Slow-only (obsj0) artifacts
SLOW_STATE_JSON = ROOT / "experiments/l96_baselines_dws500_slowobs_inf2.0_etkf_inf2.0_obsj0_s1j2_int100.json"
SLOW_JOINT_JSON = ROOT / "experiments/l96_joint_comparison_slowobs.json"

DEFAULT_OUT = ROOT / "reports/l96/outputs/l96_obs_density_da_baselines.md"

PARAM_NAMES = ["F", "c1", "hx", "eps", "w1", "w2", "w3", "w4"]
STATE_ONLY_METHODS = ["Strong-4DVar", "EnKF", "ETKF"]
JOINT_METHODS = ["Joint-EnKF", "Joint-ETKF", "Joint-Strong-4DVar"]

# SDA (score-based DA): (display name, canonical/obsj2 exp dir, slow-only/obsj0 exp dir)
NEURAL_METHODS = [
    ("SDA1", "SDA1_prior_l96", "SDA1_prior_l96_slowobs"),
    ("SDA2-mixed", "SDA2_cond_mixed_l96", "SDA2_cond_mixed_l96_slowobs"),
    ("SDA2-nominal", "SDA2_cond_nominal_l96", "SDA2_cond_nominal_l96_slowobs"),
]


def load_json(path: Path):
    if not path.exists():
        logger.warning("JSON not found: %s", path)
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not read %s: %s", path, e)
        return None


def fmt_num(x, missing="--", ndigits=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return missing
    return f"{x:.{ndigits}f}"


def state_row(state_data, case, method):
    """Returns (mean, slow, obs_fast) from a run_and_cache_baselines state-only JSON."""
    entry = (state_data or {}).get(case.lower(), {}).get(method)
    if not entry:
        return None
    g = entry.get("groups", {})
    return (entry.get("mean"), g.get("slow"), g.get("obs_fast"))


def joint_state_row(joint_data, case, method):
    """Returns (mean, slow, obs_fast, ev_mean, param_mean) from the comparator JSON."""
    entry = (joint_data or {}).get(case, {}).get(method)
    if not entry:
        return None
    sr = entry.get("state_rmse", {})
    ev = entry.get("ev", {})
    pr = entry.get("param_rmse")
    pmean = (sum(pr.values()) / len(pr)) if pr else None
    return (sr.get("mean"), sr.get("slow"), sr.get("obs_fast"), ev.get("mean"), pmean)


def neural_row(neural_data, case):
    """Returns (rmse, slow, obs_fast) from an eval_sda_l96.py ens30 neural_eval.json."""
    entry = (neural_data or {}).get("metrics", {}).get(case)
    if not entry:
        return None
    g = entry.get("groups", {})
    return (entry.get("rmse"), g.get("slow"), g.get("obs_fast"))


def build_neural_table():
    lines = []
    lines.append("## Neural (SDA, score-based DA) — same obsj2 checkpoint, guidance-only restriction")
    lines.append("")
    lines.append("No retraining and no new dataset: the slow-only column restricts which channels "
                 "the DPS guidance cost may see (`evaluation/sda_sampler.py`'s `obs_indices`), reusing "
                 "the identical obsj2-trained checkpoint and cached test set. All rows are the "
                 "30-member-ensemble (`ens30×10`) convention, same as the consolidated benchmark.")
    lines.append("")
    lines.append("| Case | Method | obsj2 mean | obsj0 mean | Δ mean | obsj2 slow | obsj0 slow | obsj2 obs_fast | obsj0 obs_fast |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for case_label, case_key in (("S0", "s0"), ("S1", "s1")):
        for name, canon_dir, slow_dir in NEURAL_METHODS:
            canon = load_json(ROOT / f"experiments/{canon_dir}/ens30_no10/neural_eval.json")
            slow = load_json(ROOT / f"experiments/{slow_dir}/ens30_no10/neural_eval.json")
            c = neural_row(canon, case_key)
            s = neural_row(slow, case_key)
            if c is None and s is None:
                continue
            delta = (s[0] - c[0]) if (c and s) else None
            lines.append(
                f"| {case_label} | {name} | {fmt_num(c[0]) if c else '--'} | "
                f"{fmt_num(s[0]) if s else '--'} | {fmt_num(delta) if delta is not None else '--'} | "
                f"{fmt_num(c[1]) if c else '--'} | {fmt_num(s[1]) if s else '--'} | "
                f"{fmt_num(c[2]) if c else '--'} | {fmt_num(s[2]) if s else '--'} |"
            )
    lines.append("")
    return lines


def build_state_table(canon, slow):
    lines = []
    lines.append("## State-only DA baselines (S0/S1)")
    lines.append("")
    lines.append("| Case | Method | obsj2 mean | obsj0 mean | Δ mean | obsj2 slow | obsj0 slow | obsj2 obs_fast | obsj0 obs_fast |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for case in ("s0", "S0"):
        pass
    cases = [("S0", "s0"), ("S1", "s1")]
    for case_label, case_key in cases:
        for m in STATE_ONLY_METHODS:
            c = state_row(canon, case_key, m)
            s = state_row(slow, case_key, m)
            if c is None and s is None:
                continue
            delta = (s[0] - c[0]) if (c and s) else None
            lines.append(
                f"| {case_label} | {m} | {fmt_num(c[0]) if c else '--'} | "
                f"{fmt_num(s[0]) if s else '--'} | {fmt_num(delta) if delta is not None else '--'} | "
                f"{fmt_num(c[1]) if c else '--'} | {fmt_num(s[1]) if s else '--'} | "
                f"{fmt_num(c[2]) if c else '--'} | {fmt_num(s[2]) if s else '--'} |"
            )
    lines.append("")
    return lines


def build_joint_state_table(canon, slow):
    lines = []
    lines.append("## Joint state-parameter DA baselines (S0/S1)")
    lines.append("")
    lines.append("| Case | Method | obsj2 mean | obsj0 mean | Δ mean | obsj2 EV | obsj0 EV | obsj2 param* | obsj0 param* |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for case in ("S0", "S1"):
        for m in JOINT_METHODS:
            c = joint_state_row(canon, case, m)
            s = joint_state_row(slow, case, m)
            if c is None and s is None:
                continue
            delta = (s[0] - c[0]) if (c and s) else None
            lines.append(
                f"| {case} | {m} | {fmt_num(c[0]) if c else '--'} | "
                f"{fmt_num(s[0]) if s else '--'} | {fmt_num(delta) if delta is not None else '--'} | "
                f"{fmt_num(c[3]) if c else '--'} | {fmt_num(s[3]) if s else '--'} | "
                f"{fmt_num(c[4]) if c else '--'} | {fmt_num(s[4]) if s else '--'} |"
            )
    lines.append("*param = mean of the (identifiable) per-parameter RMSE "
                 "(8 on S0, 6 on S1 — w3/w4 pinned to the reference prior at J=2).*")
    lines.append("")
    return lines


def build_joint_param_table(canon, slow):
    lines = []
    lines.append("## Joint-DA per-parameter RMSE (S0/S1)")
    lines.append("")
    for case in ("S0", "S1"):
        lines.append(f"### {case} — per-parameter RMSE")
        lines.append("")
        lines.append(f"| Method / config | mean | {' | '.join(PARAM_NAMES)} |")
        lines.append(f"|---|---|{'---|' * len(PARAM_NAMES)}")
        for m in JOINT_METHODS:
            for label, data in (("obsj2", canon), ("obsj0", slow)):
                entry = (data or {}).get(case, {}).get(m)
                pr = (entry or {}).get("param_rmse")
                if not pr:
                    continue
                pmean = sum(pr.values()) / len(pr)
                cells = " | ".join(fmt_num(pr.get(p)) for p in PARAM_NAMES)
                lines.append(f"| {m} / {label} | {pmean:.4f} | {cells} |")
        lines.append("")
    return lines


def write_report(canon_state, canon_joint, slow_state, slow_joint, output_path: Path) -> None:
    md = []
    md.append("# L96 DA-Baseline Observation Density: obsj2 (24D) vs slow-only obsj0 (8D)")
    md.append("")
    md.append("**System:** Lorenz-96 two-scale (NO=8, J=4), 200 shared cached S0/S1 test windows, "
              "Obs30 (obs_interval=100). Same dynamics/truth/params; only the observation changes.")
    md.append("")
    md.append("**Configurations:**")
    md.append("* **obsj2 (canonical):** 24D observed = 8 slow X + 16 fast Y1,Y2 per node.")
    md.append("* **slow-only obsj0:** only the **8 slow X** observed (no fast Y); S1 reduced dynamics "
              "kept at J=2 (24D state).")
    md.append("")
    md.append("**Eval subspace:** both are scored on the identical 24D group (slow + first-2-fast), so "
              "the metrics are directly comparable. On obsj0 the `obs_fast` group reflects fast "
              "variables **not observed** by the DA (slow-only stress test).")
    md.append("")
    md.append("**S1 forcings:** all rows use the corrected `case=2` config, i.e. the DA is fed the "
              "**corrupted** forcing `forcing_corrupted` on S1 (and `forcing_true` on S0).")
    md.append("")
    md.append("---")
    md.append("")
    md.extend(build_state_table(canon_state, slow_state))
    md.append("---")
    md.append("")
    md.extend(build_joint_state_table(canon_joint, slow_joint))
    md.append("---")
    md.append("")
    md.extend(build_joint_param_table(canon_joint, slow_joint))
    md.append("---")
    md.append("")
    md.extend(build_neural_table())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(md) + "\n")
    logger.info("Report written to %s", output_path)


def main():
    canon_state = load_json(CANON_STATE_JSON)
    canon_joint = load_json(CANON_JOINT_JSON)
    slow_state = load_json(SLOW_STATE_JSON)
    slow_joint = load_json(SLOW_JOINT_JSON)
    write_report(canon_state, canon_joint, slow_state, slow_joint, DEFAULT_OUT)


if __name__ == "__main__":
    main()
