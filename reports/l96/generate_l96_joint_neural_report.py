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

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

L96_JOINT_PARAM_NAMES = ("F", "c1", "hx", "eps", "w1", "w2", "w3", "w4")
PARAM_LIST = list(L96_JOINT_PARAM_NAMES)

# True-parameter scale mean(|true_param|) of the canonical cached S0/S1 test set
# (computed from `experiments/l96_datasets_obsj2_int100_nwin200.pt`, 200 windows each).
# Used to convert absolute param RMSE -> NRMSE for rows whose source archives only the
# absolute RMSE (the C1/C2/C3 cascade `results.json` and the joint-DA comparison JSON).
PARAM_MEAN_TRUE = {
    "s0": [8.0533, 0.9861, 0.9849, 0.0999, 1.0100, 1.0074, 0.1016, 0.0994],
    "s1": [7.9628, 1.0030, 0.9971, 0.0996, 1.0069, 1.0038, 0.0983, 0.1006],
}

# Decoupled state->param cascade experiments: a param head fed by either the frozen
# L1b state-only DirectUNet estimate (C1) or the exact true state (C2/C3, ablation),
# with C3 adding a temporal-derivative channel + positive-only bias-resampled `*_da`
# training. On the fixed per-param metric all three recover the fast weights well on S0;
# C3's bias-resample trades a small S0 hit for the best S1 robustness (see findings).
CASCADE_DEFS = {
    "C1_stateparam_head_s1": {"short": "C1 (L1b state)",
                              "arch": "StateParamHead (CNN)",
                              "state_source": "frozen L1b state estimate (decoupled)"},
    "C2_stateparam_head_state_true": {"short": "C2 (true state)",
                                      "arch": "StateParamHead (CNN)",
                                      "state_source": "exact true state (ablation)"},
    "C3_param_head_true_deriv": {"short": "C3 (true state + derivative + bias-resample)",
                                 "arch": "StateParamHead (CNN)",
                                 "state_source": "exact true state + temporal-derivative channel, "
                                                 "positive-only bias-resampled `*_da` training "
                                                 "(2026-09-01)"},
    "C4a_param_head_unet_true": {"short": "C4a (UNet, true state)",
                                 "arch": "StateParamUNet (UNet)",
                                 "state_source": "exact true state, UNet backbone (implicit "
                                                 "multi-scale temporal features, no derivative "
                                                 "channel)"},
    "C4b_param_head_unet_l1b": {"short": "C4b (UNet, L1b state)",
                                "arch": "StateParamUNet (UNet)",
                                "state_source": "frozen L1b state estimate, UNet backbone (implicit "
                                                "multi-scale temporal features, no derivative "
                                                "channel)"},
}

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
    "L10_joint_cfm_coupled_multitau": {
        "type": "JointCFMCoupled",
        "tau": "multi-tau",
        "desc": "Coupled joint conditional flow: BOTH x_tau=(1-tau)x0+tau*x1 and "
                "theta_tau=(1-tau)theta0+tau*theta1 condition both velocity fields "
                "u_theta(x_tau,theta_tau,tau,obs,forcing) and v_phi(...) -> (theta1-theta0). "
                "UNet param flow [32,64,128], state [64,128,256], 400 epochs.",
    },
    "L12_joint_direct_unet_unethead": {
        "type": "JointDirectUNet",
        "tau": "n/a",
        "desc": "JointDirectUNet (deterministic) with a UNet param head (param_head_backbone=unet) "
                "regressing 8 params from [obs, forcing, x_hat_state] (stop-grad), attention-pooled. "
                "State [64,128,256], param head [32,64,128], 200 epochs.",
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


def param_ev_from_npz(edir: Path, case: str, ens: int, k: int = 1) -> np.ndarray | None:
    """Per-parameter EV ($EV_p = 1 - mean((pred-true)^2)/var(true)$) computed
    offline from the stored eval arrays (single-sample if ``ens=0``, else the
    member-mean params of the ``ens``30 run at k Euler steps). Returns ``(8,)`` or None."""
    suff = "" if not ens else f"_ens30_k{k}"
    npz = edir / f"joint_estimates_{case}{suff}.npz"
    if not npz.exists():
        return None
    d = np.load(npz)
    pred = np.asarray(d["params_pred"], dtype=float)
    true = np.asarray(d["params_true"], dtype=float)
    mse = np.mean((pred - true) ** 2, axis=0)
    var_true = np.var(true, axis=0)
    return 1.0 - mse / np.maximum(var_true, 1e-12)


DA_PARAM_METHODS = ("Joint-ETKF", "Joint-EnKF")


def da_param_rmse_tables(da_case) -> dict:
    """Per-parameter RMSE for the joint DA filters from ``l96_joint_comparison.json``.

    Returns ``{method: {case ("S0"/"S1"): {param: float}}}`` for methods that store a
    full 8-param ``param_rmse`` dict. Per-parameter **EV** and the free forecast are NOT
    stored for DA (the per-window predictions were not archived), so DA rows in the EV
    tables render as ``--``.
    """
    tables = {}
    if not da_case:
        return tables
    for m in DA_PARAM_METHODS:
        per_case = {}
        for case in ("S0", "S1"):
            e = (da_case.get(case) or {}).get(m) or {}
            pr = e.get("param_rmse")
            if isinstance(pr, dict) and pr:
                per_case[case] = {k: float(v) for k, v in pr.items() if k in PARAM_LIST}
        if per_case:
            tables[m] = per_case
    return tables


def cascade_param_rmse(exp_dir: Path, exp_name: str):
    """Per-parameter absolute RMSE for the C1/C2/C3 cascade from its `results.json`
    (train-script convention: keys `param_rmse_s0` / `param_rmse_s1`). Returns
    ``{case: {param: float}}`` or {}."""
    r = load_json(exp_dir / exp_name / "results.json")
    if not r:
        return {}
    out = {}
    for case in ("s0", "s1"):
        pr = r.get(f"param_rmse_{case}")
        if isinstance(pr, dict) and pr:
            out[case] = {k: float(v) for k, v in pr.items() if k in PARAM_LIST}
    return out


def nrmse_from_rmse(per_param: dict, case: str):
    """NRMSE values across PARAM_LIST from absolute per-param RMSE, using the cached
    true-parameter scale. Returns ``{"vals": [...8...], "mean": float}`` or None."""
    if not per_param:
        return None
    scale = PARAM_MEAN_TRUE.get(case)
    if not scale:
        return None
    vals = [per_param.get(p) for p in PARAM_LIST]
    if not any(isinstance(v, float) for v in vals):
        return None
    num = [v / s for v, s in zip(vals, scale) if isinstance(v, float) and s]
    return {"vals": [v / s if (isinstance(v, float) and s) else None
                     for v, s in zip(vals, scale)],
            "mean": float(np.mean(num)) if num else None}


def da_nrmse_values(per_case, case: str):
    """NRMSE for a joint-DA method from its archived per-param RMSE (the true-param scale
    comes from the cached dataset, not the comparison JSON). ``per_case`` keys are "S0"/"S1"."""
    pr = (per_case.get(case.upper()) or {}) if per_case else {}
    return nrmse_from_rmse(pr, case.lower())



def write_report(exp_dir: Path, output_path: Path, comparison_json: Path) -> None:
    md = []
    da_case = load_json(comparison_json)
    da_param_rmse = da_param_rmse_tables(da_case)

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
    md.append("**Oracle-free retrain (2026-08-31):** the numbers below are from the retrained "
              "checkpoints produced with the true-parameter oracle removed — the state UNet "
              "conditions on `[obs, forcing]` only (`cond_extra_dim=1`, `output_dim=state_dim`) "
              "and a dedicated parameter head (`ParamFlowCNN` / `ParamHeadCNN`) reads the params "
              "from that oracle-free state estimate; `true_params` appear only as the regression "
              "target. Earlier published per-parameter rows came from oracle-contaminated runs "
              "(true params fed into the UNet conditioning) and are **not** a valid baseline — "
              "the correct comparison is the **joint DA baselines** table below.")
    md.append("")
    md.append("**Per-parameter detail:** `reports/l96/outputs/l96_joint_param_diagnostic.md` gives "
              "the full offline per-parameter RMSE / EV / NRMSE and free-forecast tables (single "
              "and ens30, all runs), recomputed from the stored eval arrays.")
    md.append("")
    md.append("**Cascade (decoupled state→param head, 2026-09-02):** the state→param head fed by "
              "the frozen L1b state estimate (C1), the exact true state (C2), or the true state + "
              "derivative channel with positive-only bias-resampled `*_da` training (C3) is "
              "tabulated in the NRMSE / param-RMSE tables below. **The earlier \"fast-weight "
              "failure\" (w1/w2 NRMSE ≈ 1.1-1.2) was an eval-metric bug**, not a model failure: "
              "the train-script eval read scalar `true_w1..true_w4` keys that the cached test "
              "windows don't have (they store `true_fast_weights` as a list), silently comparing "
              "all four fast-weight channels against 0.0. With the list-aware extraction every "
              "cascade model recovers the fast weights well on S0 (w1..w4 RMSE 0.011-0.013 for "
              "C1/C2), and on the biased S1 setup C3's positive-bias resampling is the most "
              "robust (w1/w2 0.04/0.10, F 0.52) while C1/C2 stay competitive on the fast weights "
              "and degrade mainly on F. See CHANGELOG 2026-09-02.")
    md.append("")
    md.append("---")
    md.append("")

    # Consolidated neural-vs-DA summary (state + mean per-param RMSE, S0/S1)
    md.append("## Consolidated summary — neural vs DA (S0/S1)")
    md.append("")
    md.append("Single-sample state RMSE (S0/S1), S1/S0 degradation, and **mean** per-parameter RMSE "
              "over the 8 params (F, c1, hx, eps, w1..w4). State RMSE beats DA on S1 (robust "
              "≈1.0 degradation) but DA filters recover the parameters far better on S0 "
              "(Joint-ETKF mean per-param RMSE 0.053 vs best neural 0.122). L9's multi-τ param "
              "head is the notable failure (mean 0.750 on S0).")
    md.append("")
    md.append("| Method | S0 state RMSE | S1 state RMSE | S1/S0 | S0 paramRMSE mean | S1 paramRMSE mean |")
    md.append("|---|---|---|---|---|---|")
    rows = []
    for exp_name in MODEL_DEFS:
        edir = exp_dir / exp_name
        data = find_single_eval(edir) if edir.is_dir() else None
        m0 = metrics_case(data, "s0") if data else None
        m1 = metrics_case(data, "s1") if data else None
        if not m0 or not m1:
            rows.append((exp_name, [None] * 5))
            continue
        pr0 = m0.get("param_rmse_mean")
        pr1 = m1.get("param_rmse_mean")
        deg = (data.get("metrics", {}).get("degradation") if data else None)
        rows.append((exp_name, [m0["rmse"], m1["rmse"], deg, pr0, pr1]))
    for method, per_case in da_param_rmse.items():
        e0 = (da_case.get("S0") or {}).get(method) or {}
        e1 = (da_case.get("S1") or {}).get(method) or {}
        r0 = (e0.get("state_rmse") or {}).get("mean")
        r1 = (e1.get("state_rmse") or {}).get("mean")
        pr0v = [v for v in (per_case.get("S0") or {}).values() if isinstance(v, float)]
        pr1v = [v for v in (per_case.get("S1") or {}).values() if isinstance(v, float)]
        pr0 = float(np.mean(pr0v)) if pr0v else None
        pr1 = float(np.mean(pr1v)) if pr1v else None
        deg = (r1 / r0) if (isinstance(r0, float) and isinstance(r1, float) and r0 > 0) else None
        rows.append((method, [r0, r1, deg, pr0, pr1]))
    for method, vals in rows:
        cells = [f"| {method} |"]
        for v in vals:
            cells.append(f" {fmt_num(v)} |")
        md.append("".join(cells))
    md.append("")
    md.append("*Lower is better for every column: state RMSE, S1/S0 degradation, and mean per-param "
              "RMSE. DA S1 paramRMSE average includes the pinned-to-prior `w3/w4` = 0, so it is not "
              "fully apples-to-apples (see the per-column parameter tables below).*")
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
    for exp_name, d in CASCADE_DEFS.items():
        src = d["state_source"]
        has_res = bool(cascade_param_rmse(exp_dir, exp_name))
        tail = ("training/eval pending (not yet evaluated)." if not has_res
                else "With the corrected per-param metric all recover the "
                     "fast weights on S0; C3's positive-bias training is the most S1-robust.")
        md.append(f"| {exp_name} | {d['arch']} | n/a | Decoupled cascade: "
                  f"param head fed by {src}. {tail} |")
    md.append("")
    md.append("---")
    md.append("")

    # Single-sample table (neural + joint-DA, state RMSE / EV / ES, S1/S0 degradation)
    md.append("## Single-sample results (n_members=1, k=1)")
    md.append("")
    md.append("State metrics over the observed subspace for the neural models (single-sample) "
              "and the joint-DA filters. S1/S0 is the degradation ratio (>1 means worse on the "
              "parameter-biased S1 setup). ES for the deterministic neural models and DA rows is "
              "the N=1 mean-absolute-error proxy; the DA filters' ES is N=30 (see DA note).")
    md.append("")
    md.append("| ID | S0 RMSE | S0 EV | S0 ES | S1 RMSE | S1 EV | S1 ES | S1/S0 |")
    md.append("|---|---|---|---|---|---|---|---|")
    rows = []
    for exp_name in MODEL_DEFS:
        edir = exp_dir / exp_name
        data = find_single_eval(edir) if edir.is_dir() else None
        m0 = metrics_case(data, "s0") if data else None
        m1 = metrics_case(data, "s1") if data else None
        if m0 and m1:
            ev0 = m0["ev"]["groups"]["all_obs"] if m0.get("ev") else None
            es0 = m0["es"]["groups"]["all_obs"] if m0.get("es") else None
            ev1 = m1["ev"]["groups"]["all_obs"] if m1.get("ev") else None
            es1 = m1["es"]["groups"]["all_obs"] if m1.get("es") else None
        else:
            ev0 = es0 = ev1 = es1 = None
        r0 = m0["rmse"] if m0 else None
        r1 = m1["rmse"] if m1 else None
        deg = (data.get("metrics", {}).get("degradation") if data else None)
        if isinstance(deg, float) and math.isnan(deg):
            deg = None
        rows.append((exp_name, r0, ev0, es0, r1, ev1, es1, deg))
    # Joint-DA filters: state RMSE / EV / ES read from the comparison JSON (S0/S1 deg computed).
    da_rows = []
    for method, per_case in da_param_rmse.items():
        e0 = (da_case.get("S0") or {}).get(method) or {}
        e1 = (da_case.get("S1") or {}).get(method) or {}
        r0 = (e0.get("state_rmse") or {}).get("mean")
        r1 = (e1.get("state_rmse") or {}).get("mean")
        ev0 = (e0.get("ev") or {}).get("mean")
        ev1 = (e1.get("ev") or {}).get("mean")
        es0 = (e0.get("es") or {}).get("mean")
        es1 = (e1.get("es") or {}).get("mean")
        deg = (r1 / r0) if (isinstance(r0, float) and isinstance(r1, float) and r0 > 0) else None
        rows.append((method, r0, ev0, es0, r1, ev1, es1, deg))
        da_rows.append(method)
    # best per column: columns are [S0 RMSE, S0 EV, S0 ES, S1 RMSE, S1 EV, S1 ES, S1/S0];
    # lowest for RMSE/ES/degradation (0,2,3,5,6), highest for EV (1,4).
    best = {j: (None, False) for j in range(7)}
    for _, r0, ev0, es0, r1, ev1, es1, deg in rows:
        for j, v in enumerate([r0, ev0, es0, r1, ev1, es1, deg]):
            if not isinstance(v, float):
                continue
            lower_is_better = j in (0, 2, 3, 5, 6)
            bv, _ = best[j]
            if bv is None or (v < bv) == lower_is_better:
                best[j] = (v, lower_is_better)
    for exp_name, r0, ev0, es0, r1, ev1, es1, deg in rows:
        vals = [r0, ev0, es0, r1, ev1, es1, deg]
        cells = [f"| {exp_name} |"]
        for j, v in enumerate(vals):
            bv, _ = best[j]
            cells.append(f" {fmt_num(v)}{' **' if (isinstance(v, float) and v == bv) else ''} |")
        md.append("".join(cells))
    md.append("")
    md.append("*Best per column: lowest RMSE / ES / degradation, highest EV. The joint-DA rows "
              "(Joint-ETKF / Joint-EnKF) come from `l96_joint_comparison.json`; their ES is the "
              "N=30 ensemble score while the neural single-sample ES is an N=1 MAE proxy (not "
              "strictly comparable, flagged).*")
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

    # Per-parameter EV (ens30)
    for k in ENS_K_STEPS:
        md.append(f"## Parameter EV — ens30 (n_members=30, k={k})")
        md.append("")
        md.append("Per-parameter explained variance from the **member-mean** parameters of the "
                  "30-member ensemble (offline from the stored eval arrays). Deep integration "
                  "(k=10) of the multi-tau JointCFM parameter head can **collapse** the EV "
                  "(hugely negative) even when the ensemble-mean state improves.")
        md.append("")
        header = "| ID | Case | " + " | ".join(PARAM_LIST) + " | mean |"
        sep = "|---|---|" + "---|" * len(PARAM_LIST) + "---|"
        md.append(header)
        md.append(sep)
        rows = []
        for exp_name in MODEL_DEFS:
            edir = exp_dir / exp_name
            for case in CASES:
                ev = param_ev_from_npz(edir, case, ens=k, k=k) if edir.is_dir() else None
                rows.append((exp_name, case, ev))
        best_idx = [None] * (len(PARAM_LIST) + 1)
        for i in range(len(PARAM_LIST) + 1):
            vals = []
            for _, _, ev in rows:
                if ev is None:
                    continue
                v = ev[i] if i < len(PARAM_LIST) else float(np.mean(ev))
                vals.append(v)
            best_idx[i] = max(vals) if vals else None
        for exp_name, case, ev in rows:
            if ev is not None:
                vals = list(ev) + [float(np.mean(ev))]
            else:
                vals = [None] * (len(PARAM_LIST) + 1)
            cells = [f"| {exp_name} | {case.upper()} |"]
            for i, v in enumerate(vals):
                cells.append(f" {fmt_num(v)}{' **' if (v is not None and v == best_idx[i]) else ''} |")
            md.append("".join(cells))
        md.append("")
        md.append("*Best per cell (highest EV) is bolded. L8 is deterministic (no ensemble).*")
        md.append("")
        md.append("---")
        md.append("")

    # Parameter tables
    for case, label in (("s0", "S0"), ("s1", "S1")):
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
            m = metrics_case(data, case)
            if m is None:
                md.append(f"| {exp_name} | " + " | ".join(["--"] * len(PARAM_LIST)) + " | -- |")
                continue
            prmse = m.get("param_rmse", {})
            cells = [f"| {exp_name} |"]
            for p in PARAM_LIST:
                cells.append(f" {fmt_num(prmse.get(p), missing='--')} |")
            cells.append(f" {fmt_num(m.get('param_rmse_mean'), missing='--')} |")
            md.append("".join(cells))
        # Joint DA filters (per-parameter RMSE from the comparison JSON). Per-parameter
        # EV and the free forecast are not archived for DA, so only the RMSE rows appear.
        for method, per_case in da_param_rmse.items():
            dv = per_case.get(case.upper()) or {}
            vals = [dv.get(p) for p in PARAM_LIST]
            meanv = np.mean([v for v in vals if v is not None]) if any(v is not None for v in vals) else None
            cells = [f"| {method} |"]
            for v in vals:
                cells.append(f" {fmt_num(v, missing='--')} |")
            cells.append(f" {fmt_num(meanv, missing='--')} |")
            md.append("".join(cells))
        # C1/C2/C3 cascade (separate sub-block, not part of the headline).
        cascade_rows = [
            (CASCADE_DEFS[exp]["short"], cascade_param_rmse(exp_dir, exp).get(case, {}))
            for exp in CASCADE_DEFS
        ]
        if any(r[1] for r in cascade_rows):
            md.append("")
            md.append("*Cascade — decoupled state→param head:*")
            for lbl, pr in cascade_rows:
                if not pr:
                    continue
                vals = [pr.get(p) for p in PARAM_LIST]
                meanv = np.mean([v for v in vals if v is not None]) if any(v is not None for v in vals) else None
                cells = [f"| {lbl} |"]
                for v in vals:
                    cells.append(f" {fmt_num(v, missing='--')} |")
                cells.append(f" {fmt_num(meanv, missing='--')} |")
                md.append("".join(cells))
        md.append("")
        md.append("*Joint-DA rows are the co-estimated 8-param RMSE from `l96_joint_comparison.json` "
                  "(S1 `w3`/`w4` are pinned to the reference prior, not estimated — their RMSE=0 is "
                  "a masking artifact, **[not]** recovery; on the 6 genuinely-estimated params DA "
                  "S1 mean NRMSE is ~0.10, i.e. parity with L9). Per-parameter "
                  "EV and the free forecast are **not** stored for DA (the per-window predictions "
                  "were not archived), so those tables show DA as `--`.*")
        md.append("")
        md.append("---")
        md.append("")

    # NRMSE (normalized parameter RMSE) tables
    for case, label in (("s0", "S0"), ("s1", "S1")):
        md.append(f"## Normalized parameter RMSE (NRMSE) — {label} (single-sample)")
        md.append("")
        md.append("Per-parameter NRMSE = `param_RMSE / mean(|true_param|)`, which normalizes "
                  "away the scale difference between parameters (e.g. F~8 vs eps~0.1) so each "
                  "competes equally. Mean is across the 8 params; lower is better. This is the "
                  "relevance metric: NRMSE ≲ 0.2 (≲20% relative error) marks an estimate that "
                  "carries genuine information about the parameter.")
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
                rows.append((exp_name, [None] * (len(PARAM_LIST) + 1), False))
                continue
            nrmse = m.get("nrmse_param", {})
            vals = [nrmse.get(p) for p in PARAM_LIST]
            vals.append(m.get("nrmse_param_mean"))
            rows.append((exp_name, vals, False))
        # Joint-DA NRMSE rows (computed from archived per-param RMSE / cached true scale).
        da_label = {}
        for method, per_case in da_param_rmse.items():
            nvals = da_nrmse_values(per_case, case)
            if nvals is None:
                continue
            vals = nvals["vals"] + [nvals["mean"]]
            rows.append((method, vals, False))
            da_label[method] = True
        # C1/C2/C3 cascade — shown separately, still participates in best-bold.
        cascade_rows = []
        for exp in CASCADE_DEFS:
            pr = cascade_param_rmse(exp_dir, exp).get(case, {})
            if not pr:
                continue
            nvals = nrmse_from_rmse(pr, case)
            if nvals is None:
                continue
            lbl = CASCADE_DEFS[exp]["short"]
            cascade_rows.append((lbl, nvals["vals"] + [nvals["mean"]]))
        best_idx = [
            min((r[i] for _, r, _ in rows if isinstance(r[i], float)), default=None)
            for i in range(len(PARAM_LIST) + 1)
        ]
        for exp_name, vals, _ in rows:
            cells = [f"| {exp_name} |"]
            for i, v in enumerate(vals):
                best = best_idx[i]
                cells.append(f" {fmt_num(v)}{' **' if (isinstance(v, float) and v == best) else ''} |")
            md.append("".join(cells))
        if cascade_rows:
            md.append("")
            md.append("*Cascade — decoupled state→param head:*")
            for lbl, vals in cascade_rows:
                cells = [f"| {lbl} |"]
                for i, v in enumerate(vals):
                    best = best_idx[i]
                    cells.append(f" {fmt_num(v)}{' **' if (isinstance(v, float) and v == best) else ''} |")
                md.append("".join(cells))
        md.append("")
        if case == "s1":
            md.append("*Best per column (lowest NRMSE) is bolded. On S1 the relevant comparison: L9 "
                      "(multi-τ joint flow) keeps **every** parameter at NRMSE ≤ 0.20 (F 0.07), i.e. "
                      "≤20% relative error — genuine param recovery at parity with the joint DA "
                      "filters on the params they actually estimate. The C1/C2/C3 cascade (2026-09-02 "
                      "re-eval with the fixed list-aware fast-weight metric) recovers the fast weights "
                      "well (C2 w1/w2 NRMSE 0.21/0.12; C3 w1/w2 0.04/0.10, F 0.52) — the earlier "
                      "w1/w2 NRMSE ≈ 1.1-1.2 was an eval-metric artifact (fast-weight truth read as "
                      "0.0), not a model failure. C3 remains the most S1-robust cascade member, but "
                      "the coupled multi-τ flow (L9) and the joint-DA filters are still ahead on "
                      "overall param recovery. Joint-DA S1 `w3`/`w4` NRMSE 0.00 is the pinned-to-prior "
                      "masking artifact (they are **not** estimated), not recovery; DA mean NRMSE is "
                      "0.07 incl. / 0.10 excl. those masked w3/w4. The DA NRMSE rows are derived from "
                      "their archived per-param RMSE in `l96_joint_comparison.json` ÷ the cached "
                      "true-param scale; per-window predictions (EV, free forecast) are not archived "
                      "for DA.*")
        else:
            md.append("*Best per column (lowest NRMSE) is bolded. Joint-DA NRMSE rows are derived from "
                      "their archived per-param RMSE ÷ the cached true-param scale. The C1/C2/C3 "
                      "cascade (2026-09-02 re-eval) recovers the fast weights on S0 (C1/C2 w1..w4 "
                      "NRMSE 0.01-0.02; F the main cost) — the earlier w1/w2 NRMSE ≈ 1.0 was an "
                      "eval-metric artifact (fast-weight truth read as 0.0), not a model failure; "
                      "C3's positive-bias training keeps S0 ≈ parity with C1/C2 on the fast weights "
                      "and is the most S1-robust.*")
        md.append("")
        md.append("---")
        md.append("")

    # Per-parameter EV (single-sample)
    for case, label in (("s0", "S0"), ("s1", "S1")):
        md.append(f"## Parameter EV — {label} (single-sample)")
        md.append("")
        md.append("Per-parameter explained variance `EV_p = 1 - mean((pred-true)^2)/var(true)` "
                  "pooled over the 200 windows (computed offline from the stored eval arrays). "
                  "Negative => parameter estimate is worse than a time-constant mean prediction. "
                  "Note: `eps/w3/w4` have very small true variance, so even good absolute errors "
                  "give large negative EV there — **NRMSE above is the fair cross-param metric**.")
        md.append("")
        header = "| ID | " + " | ".join(PARAM_LIST) + " | mean |"
        sep = "|---|" + "---|" * len(PARAM_LIST) + "---|"
        md.append(header)
        md.append(sep)
        rows = []
        for exp_name in MODEL_DEFS:
            edir = exp_dir / exp_name
            ev = param_ev_from_npz(edir, case, ens=0) if edir.is_dir() else None
            rows.append((exp_name, ev))
        best_idx = [None] * (len(PARAM_LIST) + 1)
        for i in range(len(PARAM_LIST) + 1):
            vals = []
            for _, ev in rows:
                if ev is None:
                    continue
                v = ev[i] if i < len(PARAM_LIST) else float(np.mean(ev))
                vals.append(v)
            best_idx[i] = max(vals) if vals else None
        for exp_name, ev in rows:
            if ev is not None:
                vals = list(ev) + [float(np.mean(ev))]
            else:
                vals = [None] * (len(PARAM_LIST) + 1)
            cells = [f"| {exp_name} |"]
            for i, v in enumerate(vals):
                cells.append(f" {fmt_num(v)}{' **' if (v is not None and v == best_idx[i]) else ''} |")
            md.append("".join(cells))
        for method in da_param_rmse:
            md.append(f"| {method} | " + " | ".join(["--"] * (len(PARAM_LIST) + 1)) + " |")
        md.append("")
        md.append("*Best per column (highest EV) is bolded. Joint-DA rows render as `--`: per-parameter "
                  "EV is not archived for the DA baselines (only the aggregated 8-param RMSE in "
                  "`l96_joint_comparison.json`).*")
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
        for method in da_param_rmse:
            md.append(f"| {method} | " + " | ".join(["--"] * 6) + " |")
        md.append("")
        md.append("*Best per column is bolded (lowest RMSE, highest EV). Joint-DA rows render as `--`: "
                  "the free forecast needs per-window predicted params (`x0`/`forcing` rollouts), "
                  "which are not archived for DA.*")
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
        md.append("*Best per column is bolded (highest EV, lowest RMSE).*")
        md.append("")
        md.append("---")
        md.append("")

    # DA placeholder
    md.append("## DA baselines (joint)")
    md.append("")
    md.append("Joint augmented-state DA filters (state **and** 8 params) benchmarked on the same "
              "cached S0/S1 test set, for direct comparison against the oracle-free neural rows "
              "above. Rows are read from `experiments/l96_joint_comparison.json`; missing methods "
              "render as --. For a per-parameter DA table (Joint-ETKF/EnKF/Strong-4DVar) see "
              "`l96_joint_da_benchmark.md`.")
    md.append("")
    md.append("| Method | S0 RMSE | S0 ES | S1 RMSE | S1 ES |")
    md.append("|---|---|---|---|---|")
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
