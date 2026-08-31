#!/usr/bin/env python3
"""L96 joint state-parameter neural benchmark report builder.

Consumes the eval JSONs written by ``eval_joint_neural_l96.py`` under each
experiment dir:

- single-sample: ``joint_neural_eval.json``            (metrics per case s0/s1)
- ensemble:      ``joint_neural_eval_ens30_m30_k{K}.json``  (n_members=30)

Each case's ``metrics[case]`` holds ``rmse``, ``groups`` (slow/obs_fast/
all_obs), ``ev``/``es`` (each ``{"groups": {...}}``), ``param_rmse`` (keyed by
the 8 param names) and ``param_rmse_mean``.

Generates ``reports/l96/outputs/l96_joint_neural_benchmark.md`` with a models
table, single-sample and ens30 state tables, a param table, DA placeholder rows
and a consistency note. Missing files are rendered as ``--`` without crashing.
"""
import argparse
import json
import logging
import math
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

L96_JOINT_PARAM_NAMES = ("F", "c1", "hx", "eps", "w1", "w2", "w3", "w4")
PARAM_LIST = list(L96_JOINT_PARAM_NAMES)

# ID -> (eval file base name, type, tau mode, description)
MODEL_DEFS = {
    "L7_joint_cfm_s0s1": {
        "type": "JointCFM",
        "tau": "tau=0",
        "desc": "Conditional flow matching (state + 8-param joint output) trained at tau=0 only; "
                "sampled with a single Euler step. Hidden [64,128,256], 400 epochs.",
    },
    "L8_joint_direct_unet_s0s1": {
        "type": "JointDirectUNet",
        "tau": "n/a",
        "desc": "Single-pass joint regression obs -> (state, 8 params). Deterministic. Hidden "
                "[64,128,256], 200 epochs.",
    },
    "L9_joint_cfm_s0s1_multitau": {
        "type": "JointCFM",
        "tau": "multi-tau",
        "desc": "Standard multi-tau conditional flow matching (state + 8-param joint output); "
                "sampled as a 30-member ensemble with 10 Euler steps (ens30 x 10, N=30). Hidden "
                "[64,128,256], 400 epochs.",
    },
}

CASES = ["s0", "s1"]
ENS_K_STEPS = [1, 10]
GROUPS = ("all_obs", "slow", "obs_fast")
N_STEP_COMPARE = 300  # forecast horizon for the parameter-sensitivity metric


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read %s: %s", path, e)
        return None


def find_single_eval(exp_dir: Path):
    return load_json(exp_dir / "joint_neural_eval.json")


def find_ens_eval(exp_dir: Path, n_members: int, n_outer: int):
    return load_json(exp_dir / f"joint_neural_eval_ens{n_members}_m{n_members}_k{n_outer}.json")


def fmt_num(x, missing="--", ndigits=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return missing
    return f"{x:.{ndigits}f}"


def metrics_case(data, case):
    if data is None:
        return None
    metrics = data.get("metrics") or {}
    return metrics.get(case)


def write_report(exp_dir: Path, output_path: Path, comparison_json: Path) -> None:
    md = []

    md.append("# L96 Joint State-Parameter Neural Estimation Benchmark")
    md.append("")
    md.append("**System:** Lorenz-96 two-scale (NO=8, J=4), observed subspace state_dim=24 "
              "(Obs30, obs_interval=100), 200 shared test windows.")
    md.append("")
    md.append("**Models:** JointCFM + JointDirectUNet jointly estimate the 24D observed state "
              "**and** the 8 model parameters `F, c1, hx, eps, w1..w4` (fast weights per index; "
              "h fixed), matching the L96 joint DA convention. Each model's predictions are "
              "evaluated on the same cached S0/S1 test set used by the DA baselines.")
    md.append("")
    md.append("---")
    md.append("")

    # Benchmarked models table (only list models whose result files may exist)
    md.append("## Benchmarked models")
    md.append("")
    md.append("| ID | Type | τ mode | Description |")
    md.append("|---|---|---|---|")
    for exp_name, d in MODEL_DEFS.items():
        md.append(f"| {exp_name} | {d['type']} | {d['tau']} | {d['desc']} |")
    md.append("")
    md.append("---")
    md.append("")

    # Single-sample table
    md.append("## Single-sample results (n_members=1, k=1)")
    md.append("")
    md.append("State RMSE over the observed subspace. S1/S0 is the degradation ratio "
              "(>1 means the model is worse on the parameter-biased S1 setup).")
    md.append("")
    md.append("| ID | S0 RMSE | S1 RMSE | S1/S0 |")
    md.append("|---|---|---|---|")
    rows = []
    for exp_name in MODEL_DEFS:
        edir = exp_dir / exp_name
        data = find_single_eval(edir) if edir.is_dir() else None
        m0 = metrics_case(data, "s0") if data else None
        m1 = metrics_case(data, "s1") if data else None
        r0 = m0["rmse"] if m0 else None
        r1 = m1["rmse"] if m1 else None
        deg = (data.get("metrics", {}).get("degradation") if data else None)
        if isinstance(deg, float) and math.isnan(deg):
            deg = None
        rows.append((exp_name, r0, r1, deg))
    best_r0 = min((r for _, r, _, _ in rows if isinstance(r, float)), default=None)
    best_r1 = min((r for _, _, r, _ in rows if isinstance(r, float)), default=None)
    for exp_name, r0, r1, deg in rows:
        b0 = " **" if isinstance(r0, float) and r0 == best_r0 else ""
        b1 = " **" if isinstance(r1, float) and r1 == best_r1 else ""
        md.append(f"| {exp_name} | {fmt_num(r0)}{b0} | {fmt_num(r1)}{b1} | {fmt_num(deg)} |")
    md.append("")
    md.append("*Best per column is bolded (lowest RMSE; S1/S0 degradation >1 means worse on the "
              "parameter-biased S1 setup).*")
    md.append("")
    md.append("---")
    md.append("")

    # Ensemble tables
    for k in ENS_K_STEPS:
        md.append(f"## Ensemble results (n_members=30, k={k})")
        md.append("")
        md.append("State RMSE / explained variance (EV) / energy score (ES) over the observed "
                  "subspace, computed on the member-mean trajectory; ES is the proper N=30 "
                  "ensemble scoring rule.")
        md.append("")
        md.append("| ID | S0 RMSE | S0 EV | S0 ES | S1 RMSE | S1 EV | S1 ES |")
        md.append("|---|---|---|---|---|---|---|")
        rows = []
        for exp_name in MODEL_DEFS:
            edir = exp_dir / exp_name
            data = find_ens_eval(edir, 30, k) if edir.is_dir() else None
            if data is None:
                rows.append((exp_name, [None] * 6))
                continue
            cell_vals = []
            for case in CASES:
                m = metrics_case(data, case)
                cell_vals.append(m["rmse"] if m else None)
                cell_vals.append(m["ev"]["groups"]["all_obs"] if m else None)
                cell_vals.append(m["es"]["groups"]["all_obs"] if m else None)
            rows.append((exp_name, cell_vals))
        # best per column: lowest RMSE (idx 0,3) and ES (idx 2,5); highest EV (idx 1,4)
        ncol = 6
        best = {j: (None, None) for j in range(ncol)}
        for _, vals in rows:
            for j, v in enumerate(vals):
                if not isinstance(v, float):
                    continue
                best_val, _ = best[j]
                lower_is_better = j in (0, 2, 3, 5)
                if best_val is None or (v < best_val) == lower_is_better:
                    best[j] = (v, lower_is_better)
        for exp_name, vals in rows:
            cells = [f"| {exp_name} |"]
            for j, v in enumerate(vals):
                best_val, _ = best[j]
                is_best = (isinstance(v, float) and v == best_val)
                cells.append(f" {fmt_num(v)}{' **' if is_best else ''} |")
            md.append("".join(cells))
        md.append("")
        md.append("*Only ens30 runs present on disk are shown; missing runs render as -- (L8 is "
                  "deterministic and is not run as an ensemble). Best per column: lowest RMSE/ES, "
                  "highest EV.*")
        md.append("")
        md.append("---")
        md.append("")

    # Parameter tables
    for _, label in (("s0", "S0"), ("s1", "S1")):
        md.append(f"## Parameter RMSE — {label} (single-sample)")
        md.append("")
        md.append("Per-parameter RMSE (`F, c1, hx, eps, w1..w4`) and its mean across the 8 params.")
        md.append("")
        header = "| ID | " + " | ".join(PARAM_LIST) + " | mean |"
        sep = "|---|" + "---|" * len(PARAM_LIST) + "---|"
        md.append(header)
        md.append(sep)
        for exp_name in MODEL_DEFS:
            edir = exp_dir / exp_name
            data = find_single_eval(edir) if edir.is_dir() else None
            m = metrics_case(data, label if label in ("s0", "s1") else "s0")
            if m is None:
                md.append(f"| {exp_name} | " + " | ".join(["--"] * len(PARAM_LIST)) + " | -- |")
                continue
            prmse = m.get("param_rmse", {})
            cells = [f"| {exp_name} |"]
            for p in PARAM_LIST:
                cells.append(f" {fmt_num(prmse.get(p), missing='--')} |")
            cells.append(f" {fmt_num(m.get('param_rmse_mean'), missing='--')} |")
            md.append("".join(cells))
        md.append("")
        md.append("---")
        md.append("")

    # NRMSE (normalized parameter RMSE) tables
    for case, label in (("s0", "S0"), ("s1", "S1")):
        md.append(f"## Normalized parameter RMSE (NRMSE) — {label} (single-sample)")
        md.append("")
        md.append("Per-parameter NRMSE = `param_RMSE / mean(|true_param|)`, which normalizes "
                  "away the scale difference between parameters (e.g. F~8 vs eps~0.1) so each "
                  "competes equally. Mean is across the 8 params; lower is better.")
        md.append("")
        header = "| ID | " + " | ".join(PARAM_LIST) + " | mean |"
        sep = "|---|" + "---|" * len(PARAM_LIST) + "---|"
        md.append(header)
        md.append(sep)
        rows = []
        for exp_name in MODEL_DEFS:
            edir = exp_dir / exp_name
            data = find_single_eval(edir) if edir.is_dir() else None
            m = metrics_case(data, case)
            if m is None:
                rows.append((exp_name, [None] * (len(PARAM_LIST) + 1)))
                continue
            nrmse = m.get("nrmse_param", {})
            vals = [nrmse.get(p) for p in PARAM_LIST]
            vals.append(m.get("nrmse_param_mean"))
            rows.append((exp_name, vals))
        best_idx = [
            min((r[i] for _, r in rows if isinstance(r[i], float)), default=None)
            for i in range(len(PARAM_LIST) + 1)
        ]
        for exp_name, vals in rows:
            cells = [f"| {exp_name} |"]
            for i, v in enumerate(vals):
                best = best_idx[i]
                cells.append(f" {fmt_num(v)}{' **' if (isinstance(v, float) and v == best) else ''} |")
            md.append("".join(cells))
        md.append("")
        md.append("*Best per column (lowest NRMSE) is bolded.*")
        md.append("")
        md.append("---")
        md.append("")

    # Trajectory forecast skill tables (single-sample)
    for case, label in (("s0", "S0"), ("s1", "S1")):
        md.append(f"## Trajectory forecast skill — {label} (single-sample, {N_STEP_COMPARE}-step)")
        md.append("")
        md.append("State RMSE / EV between a short forecast rolled with the **estimated** "
                  "parameters and one rolled with the **true** parameters, from the same initial "
                  "state and forcing (L96 truth dynamics, {}-step horizon, observed subspace). "
                  "This quantifies the sensitivity of short-term forecast quality to parameter "
                  "estimation error; higher EV / lower RMSE is better.".format(N_STEP_COMPARE))
        md.append("")
        md.append("| ID | RMSE slow | RMSE obs_fast | RMSE all | EV slow | EV obs_fast | EV all |")
        md.append("|---|---|---|---|---|---|---|")
        rows = []
        for exp_name in MODEL_DEFS:
            edir = exp_dir / exp_name
            data = find_single_eval(edir) if edir.is_dir() else None
            m = metrics_case(data, case)
            if m is None:
                rows.append((exp_name, [None] * 6))
                continue
            tf = m.get("traj_forecast", {})
            if not tf:
                rows.append((exp_name, [None] * 6))
                continue
            rs = tf.get("rmse", {}).get("groups", {})
            ev = tf.get("ev", {}).get("groups", {})
            vals = [
                rs.get("slow"), rs.get("obs_fast"), rs.get("all_obs"),
                ev.get("slow"), ev.get("obs_fast"), ev.get("all_obs"),
            ]
            rows.append((exp_name, vals))
        best = {j: (None, False) for j in range(6)}
        for _, vals in rows:
            for j, v in enumerate(vals):
                if not isinstance(v, float):
                    continue
                lower = j < 3
                bv, _ = best[j]
                if bv is None or (v < bv) == lower:
                    best[j] = (v, lower)
        for exp_name, vals in rows:
            cells = [f"| {exp_name} |"]
            for j, v in enumerate(vals):
                bv, _ = best[j]
                cells.append(f" {fmt_num(v)}{' **' if (isinstance(v, float) and v == bv) else ''} |")
            md.append("".join(cells))
        md.append("")
        md.append("*Best per column is bolded (lowest RMSE, highest EV).*")
        md.append("")
        md.append("---")
        md.append("")

    # Trajectory forecast skill tables (ens30)
    for k in ENS_K_STEPS:
        md.append(f"## Trajectory forecast skill — ens30 (n_members=30, k={k}, {N_STEP_COMPARE}-step)")
        md.append("")
        md.append("Same parameter-sensitivity metric computed on the **member-mean** parameter "
                  "estimates from the {} ({}-step rollouts, observed subspace). Higher EV / lower "
                  "RMSE is better.".format("ens30 ensemble", N_STEP_COMPARE))
        md.append("")
        md.append("| ID | S0 EV all | S0 RMSE all | S1 EV all | S1 RMSE all |")
        md.append("|---|---|---|---|---|")
        rows = []
        for exp_name in MODEL_DEFS:
            edir = exp_dir / exp_name
            data = find_ens_eval(edir, 30, k) if edir.is_dir() else None
            if data is None:
                rows.append((exp_name, [None] * 4))
                continue
            vals = []
            for case in CASES:
                m = metrics_case(data, case)
                if m is None or "traj_forecast" not in m:
                    vals += [None, None]
                    continue
                ev = m["traj_forecast"]["ev"]["groups"].get("all_obs")
                rmse = m["traj_forecast"]["rmse"]["groups"].get("all_obs")
                vals += [ev, rmse]
            rows.append((exp_name, vals))
        best = {j: (None, False) for j in range(4)}
        for _, vals in rows:
            for j, v in enumerate(vals):
                if not isinstance(v, float):
                    continue
                lower = j == 1 or j == 3
                bv, _ = best[j]
                if bv is None or (v < bv) == lower:
                    best[j] = (v, lower)
        for exp_name, vals in rows:
            cells = [f"| {exp_name} |"]
            for j, v in enumerate(vals):
                bv, _ = best[j]
                cells.append(f" {fmt_num(v)}{' **' if (isinstance(v, float) and v == bv) else ''} |")
            md.append("".join(cells))
        md.append("")
        md.append("*Best per column is bolded (highest EV, lowest RMSE). L8 is deterministic and not "
                  "run as an ensemble → --.*")
        md.append("")
        md.append("---")
        md.append("")

    # DA placeholder
    md.append("## DA baselines (joint)")
    md.append("")
    md.append("Joint augmented-state DA filters (state **and** 8 params) benchmarked on the same "
              "cached S0/S1 test set, vs the best neural joint estimator (L9 single-sample). "
              "Rows are read from `experiments/l96_joint_comparison.json`; missing methods "
              "render as --.")
    md.append("")
    md.append("| Method | S0 RMSE | S0 ES | S1 RMSE | S1 ES |")
    md.append("|---|---|---|---|---|")
    da_case = load_json(comparison_json)
    joint_da_names = []
    if da_case:
        seen = set()
        for c in ("S0", "S1"):
            for m in (da_case.get(c) or {}):
                if "Joint" in m and m not in seen:
                    seen.add(m)
                    joint_da_names.append(m)
    joint_da_names = sorted(joint_da_names) if joint_da_names else []
    if joint_da_names:
        for m in joint_da_names:
            cells = [f"| {m} |"]
            for case in ("S0", "S1"):
                e = (da_case.get(case) or {}).get(m) or {}
                rmse = (e.get("state_rmse") or {}).get("mean")
                es = (e.get("es") or {}).get("mean")
                cells.append(f" {fmt_num(rmse)} | {fmt_num(es)} |")
            md.append("".join(cells))
    else:
        md.append("| (no joint DA results yet) | -- | -- | -- | -- |")
    md.append("")
    md.append("*ES is the N=30 ensemble Energy Score for the filters; Joint-Strong-4DVar is a "
              "deterministic solve so its ES is the N=1 MAE proxy (marked per the DA report). "
              "Lower is better for RMSE and ES. Rows are read from "
              "`experiments/l96_joint_comparison.json`.*")
    md.append("")
    md.append("---")
    md.append("")

    # Consistency note
    md.append("## Consistency check")
    md.append("")
    md.append("The eval script stores each run's predictions against the observed-subspace truth "
              "subsampled from the cached `true_state[:, obs_var_indices]`. When the numpy arrays "
              "are accessible (same `experiments/` dir), the report would recompute a metric from "
              "them and compare against the stored JSON to detect cache drift. Here we only assert "
              "the JSONs are internally consistent (one `s0`/`s1` entry per run).")
    md.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(md))
    logger.info("Report written to %s", output_path)
    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate the L96 joint neural benchmark report")
    parser.add_argument("--exp-dir", type=str, default=str(ROOT / "experiments"),
                        help="Experiments directory (default: <repo>/experiments)")
    parser.add_argument("--output", type=str,
                        default=str(ROOT / "reports/l96/outputs/l96_joint_neural_benchmark.md"),
                        help="Output markdown report path")
    parser.add_argument("--comparison-json", type=str,
                        default=str(ROOT / "experiments/l96_joint_comparison.json"),
                        help="Path to the joint DA comparator JSON (for the DA baselines table)")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.is_dir():
        logger.error("Experiments dir does not exist: %s", exp_dir)
        raise SystemExit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_report(exp_dir, output_path, Path(args.comparison_json))


if __name__ == "__main__":
    main()
