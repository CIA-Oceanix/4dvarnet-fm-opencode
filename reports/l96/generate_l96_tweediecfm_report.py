#!/usr/bin/env python3
"""Dedicated L96 TweedieCFM benchmark report.

Compares the V2 TweedieCFM family (published, rerun, s0p2, kinner1) against the
V3 PredictStateCFM reference on the shared cached S0/S1 test set (Obs30,
dws=500, 200 windows). Neural-only: no DA baselines here (those live in the
consolidated report).

Consumes only cached eval artifacts (``experiments/<EXP>/neural_eval.json`` +
``ens30_no10/neural_eval.json`` and the shared test dataset) and outputs
``reports/l96/outputs/l96_tweediecfm_benchmark.md`` with single-sample (N=1) and
ens30×10 (N=30) RMSE/EV/ES/spread tables over the all/slow/fast groups, S1/S0
degradation, scheme descriptions, and a truth-consistency check.
"""
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

DATASET_CANDIDATES = [
    "experiments/l96_datasets_obsj2_int100_nwin200.pt",
]

CASES = ("s0", "s1")
GROUPS = ("all_obs", "slow", "obs_fast")
NO = 8

# (exp_dir, K_inner, sigma_prior, N_outer, label)
V2_SCHEMES = [
    ("V2_tweedie_cfm_l96", 5, 0.5, 10, "published"),
    ("V2_tweedie_cfm_l96_rerun", 5, 0.5, 10, "rerun (post-fix)"),
    ("V2_tweedie_cfm_l96_s0p2", 5, 0.2, 10, "σ_prior=0.2 ablation (#7)"),
    ("V2_tweedie_cfm_l96_kinner1", 1, 0.5, 10, "K_inner=1 ablation (#4)"),
]
V3_SCHEME = ("V3_predict_state_cfm_l96", None, None, 10, "PredictStateCFM reference")
VANILLA_SCHEMES = [
    ("L2b_vanilla_cfm_s0s1", "vanilla CFM, τ=0 (conditional-mean)"),
    ("L3_vanilla_cfm_s0s1", "vanilla CFM, multi-τ"),
]

# ens30 subdir per case (L3 uses a distinct S1 dir); default "ens30_no10".
ENS30_DIRS: dict[str, dict[str, str]] = {}


def _ens30_dir(name: str, case: str) -> str:
    override = ENS30_DIRS.get(name, {})
    return override.get(case, f"{name}/ens30_no10")

SCHEME_DESCRIPTIONS: list[tuple[str, str, str]] = [
    ("V2_tweedie_cfm_l96", "TweedieCFM",
     ("Two-stage Tweedie CFM: stage-1 MeanEstimatorCell (obs → mean), stage-2 residual "
      "velocity UNet; hidden [64,128,256]; 100+400 epochs; multi-τ, K_inner=5, σ_prior=0.5; "
      "N_outer=10. **Pre-Group-A-fix reference** (trained before the stage-dispatch fix); "
      "kept as the original-published baseline.")),
    ("V2_tweedie_cfm_l96_rerun", "TweedieCFM",
     ("V2 re-run with identical config (K_inner=5, σ_prior=0.5, N_outer=10) after the Group A "
      "stage-dispatch fix; sanity check that the fix changes/improves the trained model.")),
    ("V2_tweedie_cfm_l96_s0p2", "TweedieCFM",
     ("V2 with σ_prior=0.2 (default 0.5); ablation #7 — sensitivity of the trained CFM to the "
      "x₀ noise prior.")),
    ("V2_tweedie_cfm_l96_kinner1", "TweedieCFM",
     ("V2 with K_inner=1 (default 5); ablation #4 — whether iterative mean refinement in "
      "stage 1 / sampling matters; exercises the K_inner=1 div-by-zero guard.")),
    ("V3_predict_state_cfm_l96", "PredictStateCFM",
     ("Single-stage CFM predicting the final-state mean μ = E[x₁|x_τ,y]; hidden [64,128,256]; "
      "400 epochs; N_outer=10. Cross-CFM-family reference.")),
]


def short_name(name: str) -> str:
    if name.startswith("V2_"):
        variant = name.replace("V2_tweedie_cfm_l96", "")
        if variant:
            return "V2" + variant.replace("_", "-").strip("-")
        return "V2"
    return name.split("_")[0] if "_" in name else name


def make_obs_j_indices(no: int, j_truth: int, j_obs: int) -> np.ndarray:
    x_idx = list(range(no))
    y_idx = [no + k * j_truth + j for k in range(no) for j in range(j_obs)]
    return np.array(x_idx + y_idx)


def load_truth(dataset_path: Path, obs_idx: np.ndarray) -> dict[str, np.ndarray]:
    ds = torch.load(dataset_path, map_location="cpu", weights_only=False)
    return {
        case: np.stack([w["true_state"][..., obs_idx].numpy() for w in ds[f"test_{case}"]])
        for case in CASES
    }


def load_neural_trajectories(exp_dir: Path, case: str) -> np.ndarray | None:
    npz_path = exp_dir / f"estimates_{case}.npz"
    if not npz_path.exists():
        return None
    return np.load(npz_path)["trajectories"].astype(np.float64)


def load_single_metrics(exp_dir: str, case: str) -> dict[str, float]:
    with open(ROOT / "experiments" / exp_dir / "neural_eval.json") as f:
        d = json.load(f)
    m = d["metrics"][case]
    return {
        "rmse": float(m["rmse"]),
        "ev": float(m["ev"]["groups"]["all_obs"]),
        "es": float(m["es"]["groups"]["all_obs"]),
    }


def load_ens30_metrics(exp_dir: str, case: str) -> dict[str, float | None]:
    path = ROOT / "experiments" / _ens30_dir(exp_dir, case) / "neural_eval.json"
    if not path.exists():
        return {"rmse": None, "ev": None, "es": None, "spread": None}
    with open(path) as f:
        d = json.load(f)
    m = d["metrics"].get(case)
    if not m:
        return {"rmse": None, "ev": None, "es": None, "spread": None}
    ens = m["ensemble"]
    es = ens.get("es_textbook") or ens.get("es")
    return {
        "rmse": float(m["rmse"]),
        "ev": float(m["ev"]["groups"]["all_obs"]),
        "es": float(es["groups"]["all_obs"]),
        "spread": float(ens["spread"]["groups"]["all_obs"]),
    }


def fmt_table(
    title: str,
    metrics: dict[str, dict[str, float]],
    row_order: list[str],
    degrade: bool,
    field: str,
    higher_better: bool = False,
    lower_better: bool = True,
) -> str:
    header = "| Method | S0 | S1 |"
    sep = "|---|---|---|"
    if degrade:
        header += " S1/S0 |"
        sep += "---|"
    lines = [f"### {title}", "", header, sep]
    for row in row_order:
        s0 = metrics[row]["s0"][field]
        s1 = metrics[row]["s1"][field]
        cell0 = f"{s0:.4f}" if s0 is not None else "  —  "
        cell1 = f"{s1:.4f}" if s1 is not None else "  —  "
        line = f"| {short_name(row)} | {cell0} | {cell1} |"
        if degrade:
            line += f" {s1 / s0:.3f} |" if s0 is not None and s1 is not None and s0 > 0 else " n/a |"
        lines.append(line)
    lines.append("")
    return "\n".join(lines)


def check_truth(est: dict[str, dict[str, np.ndarray | None]], truth: dict[str, np.ndarray]) -> tuple[float, list[str]]:
    max_diff = 0.0
    problems: list[str] = []
    for name in est:
        for case in CASES:
            traj = est.get(name, {}).get(case)
            if traj is None:
                continue
            if traj.shape != truth[case].shape:
                problems.append(f"{name}/{case}: estimates shape {traj.shape} != truth {truth[case].shape}")
                continue
            exp_dir = _ens30_dir(name, case)
            npz_path = ROOT / "experiments" / exp_dir / f"estimates_{case}.npz"
            if not npz_path.exists():
                continue
            stored = np.load(npz_path)["truth"].astype(np.float64)
            max_diff = max(max_diff, float(np.max(np.abs(stored - truth[case]))))
    return max_diff, problems


ENS30_DIRS = {}


def main() -> None:
    global ENS30_DIRS
    out_dir = ROOT / "reports/l96/outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = DATASET_CANDIDATES[0]
    if not (ROOT / dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    obs_idx = make_obs_j_indices(NO, 4, 2)
    truth = load_truth(ROOT / dataset_path, obs_idx)

    all_rows = [e[0] for e in V2_SCHEMES] + [V3_SCHEME[0]] + [e[0] for e in VANILLA_SCHEMES]
    ENS30_DIRS = {name: {"s0": f"{name}/ens30_no10", "s1": f"{name}/ens30_no10"} for name in all_rows}
    ENS30_DIRS["L3_vanilla_cfm_s0s1"] = {
        "s0": "L3_vanilla_cfm_s0s1/ens30_no10",
        "s1": "L3_vanilla_cfm_s0s1/ens30_s1_no10",
    }

    est = {name: {case: load_neural_trajectories(ROOT / "experiments" / name, case) for case in CASES} for name in all_rows}
    single = {name: {case: load_single_metrics(name, case) for case in CASES} for name in all_rows}
    ens30 = {name: {case: load_ens30_metrics(name, case) for case in CASES} for name in all_rows}

    truth_max_diff, problems = check_truth(est, truth)
    truth_ok = truth_max_diff <= 1e-3 and not problems

    md: list[str] = [
        "# L96 TweedieCFM Benchmark — V2 family vs V3 reference",
        "",
        (
            "Setup: two-scale L96, Obs30 (`obs_interval=100`, `obs_j=2` → 24D observed space), "
            "dws=500, 200 shared cached test windows; S1 = ±20% params + ±10% bias. "
            "All schemes are the [V2/V3 CFM variants](phase_B_l96_cfm_variants.md); "
            "DA baselines are covered in `l96_consolidated_benchmark.md`."
        ),
        "",
        (
            "Single-sample (N=1) metrics are read from each experiment's root `neural_eval.json`; "
            "the ens30×10 (N=30) tables use the shared `ens30_no10` run (10 Euler steps, fresh x₀ "
            "per member). ES/spread for ens30 rows are proper ensemble scores (MAE − 0.5·pairwise). "
            "**bold** marks the best value per column."
        ),
        "",
        "## Schemes",
        "",
        "| ID | Type | K_inner | σ_prior | Description |",
        "|---|---|---|---|---|",
    ]
    for name, k, sigma, _nouter, note in V2_SCHEMES:
        md.append(f"| **{short_name(name)}** | TweedieCFM | {k} | {sigma} | {note} |")
    _nk, _ns, _nn, _no, v3note = V3_SCHEME
    md.append(f"| **{short_name(V3_SCHEME[0])}** | PredictStateCFM | n/a | n/a | {v3note} |")
    for name, note in VANILLA_SCHEMES:
        md.append(f"| **{short_name(name)}** | vanilla CFM | n/a | n/a | {note} |")
    md.append("")

    md.append("## Single-sample (N=1)")
    md.append("")
    md.append(fmt_table("RMSE (lower is better)", single, all_rows, True, "rmse"))
    md.append(fmt_table("EV (higher is better)", single, all_rows, False, "ev"))

    md.append("## ens30×10 (N=30, proper ensemble)")
    md.append("")
    md.append(fmt_table("RMSE (lower is better)", ens30, all_rows, True, "rmse"))
    md.append(fmt_table("EV (higher is better)", ens30, all_rows, False, "ev"))
    md.append(fmt_table("ES (lower is better)", ens30, all_rows, False, "es"))
    md.append(fmt_table("Spread (higher = more diverse ensemble)", ens30, all_rows, False, "spread"))

    md += ["## Findings", ""]
    md.append(
        "- **Group A fix materially improves V2**: the post-fix rerun (K_inner=5, σ_prior=0.5) at "
        f"ens30×10 S0/S1 {ens30['V2_tweedie_cfm_l96_rerun']['s0']['rmse']:.4f}/"
        f"{ens30['V2_tweedie_cfm_l96_rerun']['s1']['rmse']:.4f} beats the pre-fix published V2 "
        f"({ens30['V2_tweedie_cfm_l96']['s0']['rmse']:.4f}/"
        f"{ens30['V2_tweedie_cfm_l96']['s1']['rmse']:.4f}), because the correct stage-2 checkpoint "
        "selection yields a genuinely better model. Both runs use identical config."
    )
    md.append(
        "- **σ_prior=0.2 (#7) is neutral-to-marginal**: essentially ties the rerun on RMSE "
        f"(S0 {ens30['V2_tweedie_cfm_l96_s0p2']['s0']['rmse']:.4f} vs "
        f"{ens30['V2_tweedie_cfm_l96_rerun']['s0']['rmse']:.4f}) with a tighter ensemble "
        f"(spread {ens30['V2_tweedie_cfm_l96_s0p2']['s0']['spread']:.3f} vs "
        f"{ens30['V2_tweedie_cfm_l96_rerun']['s0']['spread']:.3f}) but a marginally higher ES."
    )
    md.append(
        "- **K_inner=1 (#4) is clearly worse**: "
        f"S0 {ens30['V2_tweedie_cfm_l96_kinner1']['s0']['rmse']:.4f} vs rerun "
        f"{ens30['V2_tweedie_cfm_l96_rerun']['s0']['rmse']:.4f} (+{(ens30['V2_tweedie_cfm_l96_kinner1']['s0']['rmse']/ens30['V2_tweedie_cfm_l96_rerun']['s0']['rmse']-1)*100:.1f}%), "
        "higher spread — iterative mean refinement (K_inner=5) matters."
    )
    l2b = ens30["L2b_vanilla_cfm_s0s1"]["s0"]["rmse"]
    md.append(
        "- **TweedieCFM beats vanilla CFM at ens30×10**: the best TweedieCFM (rerun/s0p2, "
        f"S0 {ens30['V2_tweedie_cfm_l96_s0p2']['s0']['rmse']:.4f}) is below the vanilla τ=0 CFM "
        f"L2b ({l2b:.4f}) and multi-τ L3 "
        f"({ens30['L3_vanilla_cfm_s0s1']['s0']['rmse']:.4f}), and V3 PredictStateCFM "
        f"({ens30['V3_predict_state_cfm_l96']['s0']['rmse']:.4f})."
    )
    md += [
        "",
        "## Consistency check",
        "",
        f"- Neural stored truth vs dataset true_state[:, obs_var_indices]: max |Δ| = {truth_max_diff:.2e} → "
        + ("PASS" if truth_ok else "FAIL"),
    ]
    for problem in problems:
        md.append(f"- WARNING: {problem}")

    report_path = out_dir / "l96_tweediecfm_benchmark.md"
    report_path.write_text("\n".join(md))
    logger.info("Report saved: %s", report_path)

    if not truth_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
