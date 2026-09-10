#!/usr/bin/env python3
"""Generate the QG case-study overview report.

JSON-only generator (no QG/neural code imports). Two sections:

1. **Benchmarked schemes** -- a description table covering all 5 schemes
   (the 4 DA baselines + Q1/DirectUNet): type, key hyperparameters, one-line
   description. Full DA-method detail (reference-case config, reconstruction
   figures) lives in ``qg_da_report.md`` (see ``generate_qg_da_report.py``);
   this section only summarizes enough to compare schemes side by side.
2. **Summary metrics** -- S0-only pooled EV/CRPS table combining the 4 DA
   baselines (``qg_repro_validation/*.json``, reference case lag=5.0d/
   noise=0.05) with Q1 (``qg_repro_validation/q1_direct_unet_lag5_noise05.json``,
   see ``eval_qg_q1_lag5_noise05.py``).

**Apples-to-apples caveat (2026-09-10):** Q1's row is NOT a fully fair
comparison -- the checkpoint was *trained* at ``train_qg_neural.py``'s old
defaults (lag=1.0d, noise_frac=0.01), and only *re-evaluated* (not retrained)
at the DA baselines' lag=5.0d/noise=0.05 reference case, so it's being tested
outside its training distribution. This report flags that explicitly in the
table rather than presenting it as a like-for-like result; a genuinely fair
comparison needs a matched retrain (see PLAN.md).

Run from the repository root::

    python reports/qg/generate_qg_neural_report.py
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DA_METHODS = [("EnKF", "enkf"), ("ETKF", "etkf"),
              ("Weak-4DVar", "weak4dvar"), ("Strong-4DVar", "strong4dvar")]

SCHEMES = [
    {
        "name": "ETKF",
        "type": "Ensemble DA (deterministic square-root)",
        "config": "N=80, inflation=1.0, loc_radius=6.0",
        "desc": "Ensemble Transform Kalman Filter -- deterministic ensemble-square-root "
                "analysis update, sequentially cycled over the assimilation window. "
                "No stochastic observation perturbation.",
    },
    {
        "name": "EnKF",
        "type": "Ensemble DA (stochastic, perturbed-obs)",
        "config": "N=80, inflation=1.0, loc_radius=6.0",
        "desc": "Perturbed-observation Ensemble Kalman Filter -- each ensemble member "
                "assimilates an independently perturbed observation. Same hyperparameters "
                "as ETKF for a controlled comparison.",
    },
    {
        "name": "Strong-4DVar",
        "type": "Variational DA (deterministic, perfect-model)",
        "config": "window=60 steps, LBFGS, max_iter=60, b_var_scale=1.0",
        "desc": "4D-Var assuming the DA dynamical model is exact over the assimilation "
                "window (strong constraint) -- optimizes only the initial condition.",
    },
    {
        "name": "Weak-4DVar",
        "type": "Variational DA (deterministic, weak-constraint)",
        "config": "window=60 steps, LBFGS, max_iter=60, b_var_scale=1.0, q_var_scale=0.1",
        "desc": "4D-Var with an added per-step model-error control term (weak constraint) "
                "-- can partially compensate for a biased/mismatched dynamical model, at "
                "the cost of a larger control space.",
    },
    {
        "name": "Q1 (DirectUNet)",
        "type": "Neural (deterministic, single-pass, supervised)",
        "config": "MONAI circular 2D U-Net, hidden=[64,128,256] (M tier), cosine LR, "
                  "200 epochs",
        "desc": "Direct single forward-pass estimator mapping observations to the full "
                "state (no iterative assimilation cycle, no dynamical model at inference "
                "time). Circular-padded Conv2d over the doubly-periodic (ny,nx) grid; "
                "trained via supervised regression on a combined psi + weighted PV-q loss.",
    },
]


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def fmt(x) -> str:
    return f"{x:.4f}" if x is not None else "--"


def fmt_sci(x, nd=2) -> str:
    return f"{x:.{nd}e}" if x is not None else "--"


def rank_marks(values: list, higher_is_better: bool = True) -> dict:
    """{index: 'bold'|'italic'} for the best/second-best of `values` (None skipped).

    Project-level report convention: bold the best score per metric column,
    italicize the second-best (ranked within that column only).
    """
    present = [(i, v) for i, v in enumerate(values) if v is not None]
    present.sort(key=lambda t: t[1], reverse=higher_is_better)
    marks = {}
    if len(present) >= 1:
        marks[present[0][0]] = "bold"
    if len(present) >= 2:
        marks[present[1][0]] = "italic"
    return marks


def mark(s: str, kind: str | None) -> str:
    if kind == "bold":
        return f"**{s}**"
    if kind == "italic":
        return f"*{s}*"
    return s


def load_da_summary(root: Path) -> list[dict]:
    out = []
    for label, fname in DA_METHODS:
        p = root / "qg_repro_validation" / f"{fname}.json"
        if not p.exists():
            out.append({"label": label, "data": None})
            continue
        d = load_json(p)
        s0 = d.get("scenarios", {}).get("test_s0", {})
        q = s0.get("metrics_per_field", {}).get("q", {})
        psi = s0.get("metrics_per_field", {}).get("psi", {})
        out.append({
            "label": label,
            "data": {
                "psi_ev": psi.get("full", {}).get("ev"),
                "q_ev": q.get("full", {}).get("ev"),
                "crps_normalized": s0.get("crps_normalized"),
                "crps_is_deterministic": s0.get("crps_is_deterministic"),
            },
        })
    return out


def load_q1_summary(root: Path) -> dict | None:
    p = root / "qg_repro_validation" / "q1_direct_unet_lag5_noise05.json"
    if not p.exists():
        return None
    d = load_json(p)
    return {
        "psi_ev": d["psi"]["pooled_ev"],
        "q_ev": d["q"]["pooled_ev"],
        "eval_config": d["eval_config"],
        "caveat": d["train_test_mismatch_caveat"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-root", default=str(ROOT / "reports/qg/outputs"))
    ap.add_argument("--out", default=str(ROOT / "reports/qg/outputs/qg_neural_report.md"))
    args = ap.parse_args()

    out_root = Path(args.json_root)
    da = load_da_summary(out_root)
    q1 = load_q1_summary(out_root)

    lines = []
    add = lines.append
    add("# QG Case Study: Benchmarked Schemes Overview")
    add("")
    add("Two-layer quasi-geostrophic (QG) Phillips-channel case study -- the four DA "
        "baselines (ETKF, EnKF, Strong-4DVar, Weak-4DVar) plus the Q1 neural "
        "estimator (DirectUNet), on the S0 (no model error) reference case.")
    add("")

    add("## 1. Benchmarked schemes")
    add("")
    add("| scheme | type | key config | description |")
    add("|---|---|---|---|")
    for s in SCHEMES:
        add(f"| {s['name']} | {s['type']} | {s['config']} | {s['desc']} |")
    add("")
    add("DA-method reference-case detail (full lag/noise sensitivity analysis, "
        "per-layer breakdown, reconstruction figures) is in `qg_da_report.md` "
        "(`generate_qg_da_report.py`) -- not repeated here.")
    add("")

    add("## 2. Summary metrics (S0 reference case, N=100)")
    add("")
    add("Pooled EV (higher is better) on ψ (streamfunction, both layers) and PV-q "
        "(both layers), plus normalized CRPS where available.")
    add("")
    add("| scheme | ψ EV | PV-q EV | CRPS (norm.) |")
    add("|---|---|---|---|")
    # Ranked (bold=best, italic=second-best) per column across all 5 rows,
    # including Q1 -- despite the apples-to-apples caveat below, the project
    # convention applies unconditionally to every report table.
    psi_evs = [row["data"]["psi_ev"] if row["data"] is not None else None for row in da]
    q_evs = [row["data"]["q_ev"] if row["data"] is not None else None for row in da]
    crps_norms = [row["data"]["crps_normalized"] if row["data"] is not None else None for row in da]
    if q1 is not None:
        psi_evs.append(q1["psi_ev"])
        q_evs.append(q1["q_ev"])
        crps_norms.append(None)
    psi_marks = rank_marks(psi_evs, higher_is_better=True)
    q_marks = rank_marks(q_evs, higher_is_better=True)
    crps_marks = rank_marks(crps_norms, higher_is_better=False)

    for i, row in enumerate(da):
        d = row["data"]
        if d is None:
            add(f"| {row['label']} | -- | -- | -- |")
            continue
        star = "*" if d["crps_is_deterministic"] else ""
        add(f"| {row['label']} | {mark(fmt(d['psi_ev']), psi_marks.get(i))} | "
            f"{mark(fmt(d['q_ev']), q_marks.get(i))} | "
            f"{mark(fmt(d['crps_normalized']) + star, crps_marks.get(i))} |")
    if q1 is not None:
        i = len(da)
        add(f"| Q1 (DirectUNet) † | {mark(fmt(q1['psi_ev']), psi_marks.get(i))} | "
            f"{mark(fmt(q1['q_ev']), q_marks.get(i))} | -- |")
    else:
        add("| Q1 (DirectUNet) † | -- | -- | -- | (not yet evaluated) |")
    add("")
    add("(Best per column **bolded**, second-best *italicized*.)")
    add("")
    add("\\* CRPS is deterministic (degenerates to MAE, no ensemble spread).")
    add("")
    if q1 is not None:
        add(f"† **Not apples-to-apples**: {q1['caveat']}")
    else:
        add("† Q1 row pending: run `eval_qg_q1_lag5_noise05.py` against the trained "
            "checkpoint first.")
    add("")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
