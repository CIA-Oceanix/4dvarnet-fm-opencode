#!/usr/bin/env python3
"""Offline per-parameter diagnosis for L96 joint state-parameter neural models.

Reads the eval outputs stored by ``eval_joint_neural_l96.py`` (which itself
re-runs on the canonical cached S0/S1 test set) and recomputes the detailed
per-parameter diagnostics from the stored arrays, WITHOUT re-running inference:

- ``joint_estimates_{case}.npz``      single-sample (n_members=1)
- ``joint_estimates_{case}_ens30.npz`` ens30 (n_members=30; params_pred is the
  member-mean (W,8))

Each npz holds ``params_pred (W,8)``, ``params_true (W,8)``, ``x0 (W,40)`` and
``forcing_true (W,3000)``, so everything below — per-parameter RMSE / EV / NRMSE
and the free-forecast (true-vs-estimated params, same x0 + forcing) — is
recomputed purely from disk, independently of the JSONs the benchmark report
consumes. This is the ground-truth cross-check for the "L9 multi-tau params
fail" concern.

Per-parameter EV uses the same pooled convention as the state EV:
``EV_p = 1 - mean_w[(pred-true)^2] / var_w[true]`` (negative => worse than a
time-constant mean prediction; +1 => perfect).
"""
import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.estimate_metrics import nrmse_param, trajectory_forecast_skill
from evaluation.run_l96 import make_obs_j_indices
from models.lorenz96_dynamics import Lorenz96Dynamics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

L96_JOINT_PARAM_NAMES = ("F", "c1", "hx", "eps", "w1", "w2", "w3", "w4")
CASES = ["s0", "s1"]
N_COMPARE_STEPS = 300
NO = 8
J_TRUTH = 4
J_OBS = 2

MODELS = [
    ("L7_joint_cfm_s0s1", "JointCFM tau=0"),
    ("L8_joint_direct_unet_s0s1", "JointDirectUNet"),
    ("L9_joint_cfm_s0s1_multitau", "JointCFM multi-tau"),
]
ENS_K_STEPS = [1, 10]


def load_params(npz_path: Path):
    data = np.load(npz_path)
    return data["params_pred"], data["params_true"], data


def per_param_rmse(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((pred - true) ** 2, axis=0))


def per_param_ev(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    mse = np.mean((pred - true) ** 2, axis=0)
    var_true = np.var(true, axis=0)
    return 1.0 - mse / np.maximum(var_true, 1e-12)


def free_forecast(dyn, x0, forcing, true, est, obs_idx):
    skill = trajectory_forecast_skill(
        dyn, x0, forcing, true, est,
        n_steps=N_COMPARE_STEPS, obs_var_indices=obs_idx,
    )
    return skill["rmse"]["mean"], skill["ev"]["mean"]


def load_json(path: Path):
    if not path.exists():
        return None
    import json

    with open(path) as f:
        return json.load(f)


def forecast_from_json(exp_dir: Path, case: str, ens: bool, k: int):
    """Prefer the free-forecast already stored by the eval script (identical
    computation); recomputed offline (~0.7 s/window, ~140 s per 200-window run)
    is only triggered by ``--recompute-forecast``."""
    if ens:
        fname = f"joint_neural_eval_ens30_m30_k{k}.json"
        key = "traj_forecast"
    else:
        fname = "joint_neural_eval.json"
        key = "traj_forecast"
    js = load_json(exp_dir / fname)
    if js is None:
        return None
    m = (js.get("metrics") or {}).get(case)
    if not m or key not in m:
        return None
    tf = m[key]
    return tf["rmse"]["mean"], tf["ev"]["mean"]


def fmt(x, missing="--", ndigits=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return missing
    return f"{x:.{ndigits}f}"


def build_model_section(md, exp_name, desc, exp_dir, dyn, obs_idx, recompute_forecast):
    md.append(f"## {exp_name} — {desc}")
    md.append("")

    # ---- Single-sample per-param table ----
    md.append("### Single-sample (n_members=1) per-parameter metrics")
    md.append("")
    md.append("Pooled over the 200 windows. RMSE = `sqrt(mean((pred-true)^2))`; "
              "EV = `1 - mean((pred-true)^2)/var(true)`; NRMSE = RMSE / mean(|true|).")
    md.append("")
    header = "| Case | P | " + " | ".join(L96_JOINT_PARAM_NAMES) + " | mean |"
    sep = "|---|---|" + "---|" * len(L96_JOINT_PARAM_NAMES) + "---|"
    md.append(header)
    md.append(sep)
    for case in CASES:
        npz = exp_dir / f"joint_estimates_{case}.npz"
        if not npz.exists():
            md.append(f"| {case.upper()} | RMSE | " + " | ".join(["--"] * 8) + " | -- |")
            continue
        pred, true, _ = load_params(npz)
        rmse = per_param_rmse(pred, true)
        ev = per_param_ev(pred, true)
        nrmse = nrmse_param(pred, true)["per_param"]
        for label, vec in (("RMSE", rmse), ("EV", ev), ("NRMSE", nrmse)):
            cells = [f"| {case.upper()} | {label} |"]
            for v in vec:
                cells.append(f" {fmt(v)} |")
            cells.append(f" {fmt(np.mean(vec))} |")
            md.append("".join(cells))
    md.append("")
    md.append("---")
    md.append("")

    # ---- Free forecast (single-sample) ----
    md.append("### Free forecast (single-sample, 300-step)")
    md.append("")
    md.append("Free forecast RMSE / EV between a rollout with the **estimated** params and one "
              "with the **true** params, from the same x0 and forcing (observed subspace). "
              "High RMSE / negative EV => parameter error destroys short-term forecast skill.")
    md.append("")
    md.append("| Case | RMSE | EV |")
    md.append("|---|---|---|")
    for case in CASES:
        r = e = None
        if recompute_forecast:
            npz = exp_dir / f"joint_estimates_{case}.npz"
            if npz.exists():
                pred, true, data = load_params(npz)
                r, e = free_forecast(dyn, data["x0"], data["forcing_true"], true, pred, obs_idx)
        else:
            fe = forecast_from_json(exp_dir, case, ens=False, k=1)
            if fe is not None:
                r, e = fe
        md.append(f"| {case.upper()} | {fmt(r)} | {fmt(e)} |")
    md.append("")
    md.append("---")
    md.append("")

    # ---- ens30 per-param table ----
    for k in ENS_K_STEPS:
        md.append(f"### ens30 (n_members=30, k={k}) per-parameter metrics")
        md.append("")
        md.append("`params_pred` is the member-mean across the 30 members.")
        md.append("")
        md.append(header)
        md.append(sep)
        for case in CASES:
            npz = exp_dir / f"joint_estimates_{case}_ens30.npz"
            if not npz.exists():
                md.append(f"| {case.upper()} | RMSE | " + " | ".join(["--"] * 8) + " | -- |")
                continue
            pred, true, _ = load_params(npz)
            rmse = per_param_rmse(pred, true)
            ev = per_param_ev(pred, true)
            nrmse = nrmse_param(pred, true)["per_param"]
            for label, vec in (("RMSE", rmse), ("EV", ev), ("NRMSE", nrmse)):
                cells = [f"| {case.upper()} | {label} |"]
                for v in vec:
                    cells.append(f" {fmt(v)} |")
                cells.append(f" {fmt(np.mean(vec))} |")
                md.append("".join(cells))
        md.append("")
        md.append("---")
        md.append("")

    # ---- Free forecast (ens30) ----
    for k in ENS_K_STEPS:
        md.append(f"### Free forecast (ens30, k={k}, 300-step)")
        md.append("")
        md.append("| Case | RMSE | EV |")
        md.append("|---|---|---|")
        for case in CASES:
            r = e = None
            if recompute_forecast:
                npz = exp_dir / f"joint_estimates_{case}_ens30.npz"
                if npz.exists():
                    pred, true, data = load_params(npz)
                    r, e = free_forecast(dyn, data["x0"], data["forcing_true"], true, pred, obs_idx)
            else:
                fe = forecast_from_json(exp_dir, case, ens=True, k=k)
                if fe is not None:
                    r, e = fe
            md.append(f"| {case.upper()} | {fmt(r)} | {fmt(e)} |")
        md.append("")
        md.append("---")
        md.append("")


def main():
    parser = argparse.ArgumentParser(description="Offline per-parameter diagnostic for L96 joint neural models")
    parser.add_argument("--exp-dir", type=str, default=str(ROOT / "experiments"),
                        help="Experiments directory (default: <repo>/experiments)")
    parser.add_argument("--output", type=str,
                        default=str(ROOT / "reports/l96/outputs/l96_joint_param_diagnostic.md"),
                        help="Output markdown diagnostic path")
    parser.add_argument("--recompute-forecast", action="store_true",
                        help="Recompute the free-forecast RMSE/EV offline from the stored arrays "
                             "rather than reading the eval JSONs (slow: ~0.7 s/window, ~140 s per "
                             "200-window run)")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.is_dir():
        logger.error("Experiments dir does not exist: %s", exp_dir)
        raise SystemExit(1)

    truth_dyn = Lorenz96Dynamics(
        dt=0.001, coupling_exponent=1.6, NO=NO, J=J_TRUTH,
        h=1.0, hx=1.0, eps=0.1, fast_weights=[1.0, 1.0, 0.1, 0.1],
    )
    obs_idx = make_obs_j_indices(NO, J_TRUTH, J_OBS)

    md = []
    md.append("# L96 joint state-parameter — per-parameter diagnostic (offline recompute)")
    md.append("")
    md.append("**Source:** the eval arrays stored by `eval_joint_neural_l96.py` "
              "(`joint_estimates_{case}.npz` / `..._ens30.npz`), recomputed offline — no "
              "inference re-run. Pooled over the 200 cached windows (Obs30, observed subspace 24D).")
    md.append("")
    md.append("**Metrics:** per-parameter `RMSE = sqrt(mean((pred-true)^2))`, "
              "`EV = 1 - mean((pred-true)^2)/var(true)`, `NRMSE = RMSE / mean(|true|)`; free "
              "forecast = 300-step rollout RMSE/EV between estimated- and true-parameter "
              "trajectories from the same x0 + forcing (observed subspace).")
    md.append("")
    md.append("*Reading note:* per-parameter EV is dominated by the parameter's own scale — the "
              "D-subsystem params (`eps, w3, w4`) have very small true variance (~0.1 vs F~8), so "
              "even small absolute errors yield large negative EV there. **NRMSE is the fair "
              "cross-parameter comparison** (normalizes by mean(|true|)); free-forecast EV is the "
              "most physically meaningful summary of a parameter block.")
    md.append("")
    md.append("---")
    md.append("")

    for exp_name, desc in MODELS:
        edir = exp_dir / exp_name
        if not edir.is_dir():
            md.append(f"## {exp_name} — {desc} (no eval dir)")
            md.append("")
            md.append("---")
            md.append("")
            continue
        build_model_section(md, exp_name, desc, edir, truth_dyn, obs_idx, args.recompute_forecast)
        md.append("")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(md))
    logger.info("Diagnostic written to %s", output_path)
    print(f"Diagnostic saved to: {output_path}")


if __name__ == "__main__":
    main()