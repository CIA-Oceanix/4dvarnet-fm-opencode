#!/usr/bin/env python3
"""Generate the QG case-study overview report.

JSON-only generator (no QG/neural code imports). Two sections:

1. **Benchmarked schemes** -- a description table covering the 4 DA
   baselines (ETKF, EnKF, Strong-4DVar, Weak-4DVar) plus 4 neural DirectUNet
   variants on the T-merged-into-channels backbone (``models.
   monai_unet_qg2d.MonaiDirectUNetQGChannelTime``): Q1 obs-only, Q2 oracle
   forcing+param conditioning, Q3 noisy-trained forcing+param conditioning,
   Q4 (Q3's noisy conditioning plus a raw initial-condition input): type,
   key hyperparameters, one-line description. Full DA-method detail
   (reference-case config, sensitivity analysis, reconstruction figures)
   lives in ``qg_da_report.md`` (see ``generate_qg_da_report.py``); this
   section only summarizes enough to compare schemes side by side.
2. **Summary metrics** -- S0 (no model error) and S1 (combined model-error:
   param bias + corrupted wind + resolution mismatch) pooled EV table, all
   8 schemes at the identical lag=5.0d/noise_frac=0.05/s1_param_bias=
   s1_amp_bias=0.1 configuration (the DA baselines' own reference case).
   DA numbers from ``qg_repro_validation{,_s1}/*.json``; neural numbers from
   ``qg_neural_s0_s1_cross_scenario/results_lag5_noise0.05_bias0.1.json``
   (``eval_qg_neural_s0_s1.py``).

**Apples-to-apples (2026-09-14):** all four current neural schemes (Q1-Q4,
promoted from the T-channels bench refresh's Q7-noise05/Q8/Q9/Q10) were
*trained* directly at this exact lag=5.0d/noise=0.05/s1_param_bias=
s1_amp_bias=0.1 config -- train_qg_neural.py's own current fallback
default -- so every row is a genuinely fair, matched-distribution
comparison against the DA baselines; no per-row caveat marker is needed.
This table previously covered an older batch-folding-backbone family
(Q1/Q3/Q4/Q3-noise0.05/Q5, three of which were only re-evaluated, not
retrained, at this config) -- see PLAN.md's 2026-09-14 "T-channels bench
refresh" section for the full history; those older configs/checkpoints
still exist on disk as historical experiments, just retired from this
report.

Run from the repository root::

    python reports/qg/generate_qg_neural_report.py
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DA_METHODS = [("EnKF", "enkf"), ("ETKF", "etkf"),
              ("Weak-4DVar", "weak4dvar"), ("Strong-4DVar", "strong4dvar")]

# Neural schemes in cross-scenario results_lag5_noise0.05_bias0.1.json --
# the canonical T-channels DirectUNet family (2026-09-14). All four were
# trained directly at this exact config (see the apples-to-apples note
# above), so the third element (whether training distribution matches
# this eval config) is always True now -- kept for the marker mechanism's
# sake rather than dropping it outright.
NEURAL_SCHEMES = [
    ("Q1", "Q1 (DirectUNet, T-channels, obs-only)", True),
    ("Q2", "Q2 (oracle forcing+param cond.)", True),
    ("Q3", "Q3 (noisy-trained forcing+param cond.)", True),
    ("Q4", "Q4 (noisy cond. + IC)", True),
]

SCHEMES = [
    {
        "name": "ETKF",
        "type": "Ensemble DA (deterministic square-root)",
        "config": "N=80, inflation=1.0, loc_radius=2.0, etkf_ridge=0.1",
        "desc": "Ensemble Transform Kalman Filter -- deterministic ensemble-square-root "
                "analysis update, sequentially cycled over the assimilation window. "
                "No stochastic observation perturbation. `loc_radius=2.0`/"
                "`etkf_ridge=0.1` is the default as of 2026-09-15 (was "
                "loc_radius=6.0/etkf_ridge=1.0) -- see `qg_da_report.md`'s "
                "sensitivity-analysis section.",
    },
    {
        "name": "EnKF",
        "type": "Ensemble DA (stochastic, perturbed-obs)",
        "config": "N=80, inflation=1.0, loc_radius=2.0",
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
        "name": "Q1 (DirectUNet, T-channels)",
        "type": "Neural (deterministic, single-pass, supervised)",
        "config": "MonaiDirectUNetQGChannelTime, hidden=[64,128,256] (M tier), obs-only, "
                  "lag=5.0d/noise=0.05 training, gradient_clip_val=1.0, 200 epochs",
        "desc": "Direct single forward-pass estimator mapping observations to the full "
                "state (no iterative assimilation cycle, no dynamical model at inference "
                "time). Merges the T (days) axis into the *channel* dimension (`models."
                "monai_unet_qg2d.MonaiUNet2DQGSolver`) so the 2D circular-conv backbone's "
                "ordinary channel mixing can relate one day to another, unlike the older "
                "batch-folding backbone (every day processed fully independently). "
                "Trained via supervised regression on a combined psi + weighted PV-q loss.",
    },
    {
        "name": "Q2 (oracle cond.)",
        "type": "Neural (deterministic, single-pass, oracle-conditioned)",
        "config": "Q1 arch + true wind_curl field + true [U1,rd,rek] params "
                  "(cond_mode=\"true\"), lag=5.0d/noise=0.05 training",
        "desc": "Q1 plus exact (unjittered) forcing/parameter conditioning, mirroring the "
                "L96 SDA CFM oracle-conditioning study.",
    },
    {
        "name": "Q3 (noisy cond.)",
        "type": "Neural (deterministic, single-pass, robustly-conditioned)",
        "config": "Q1 arch + resampled-severity corrupted forcing/params "
                  "(cond_mode=\"noisy\", noisy_max=1.5), lag=5.0d/noise=0.05 training",
        "desc": "Q1 plus forcing/parameter conditioning where every training draw samples "
                "a fresh random corruption severity in [0, noisy_max] (mirroring the L96 "
                "SDA3 CFM study), instead of Q2's always-exact conditioning -- intended to "
                "be more robust to conditioning-input error at inference time.",
    },
    {
        "name": "Q4 (noisy cond. + IC)",
        "type": "Neural (deterministic, single-pass, robustly-conditioned + IC)",
        "config": "Q3-style noisy cond. (noisy_max=2.0, s1_param_bias=s1_amp_bias=0.1) "
                  "plus the raw initial-condition snapshot as a 4th input; lag=5.0d/"
                  "noise=0.05 training, gradient_clip_val=1.0",
        "desc": "Adds the same background information a DA method's own init_state "
                "gives it for free -- the raw IC (inverted to psi, broadcast identically "
                "across the window, a third conditioning class distinct from the "
                "per-day forcing field and the per-window param vector).",
    },
]


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def fmt(x) -> str:
    return f"{x:.4f}" if x is not None else "--"


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


def load_da_summary(root: Path, scenario_dir: str, scenario_key: str) -> list[dict]:
    out = []
    for label, fname in DA_METHODS:
        p = root / scenario_dir / f"{fname}.json"
        if not p.exists():
            out.append({"label": label, "data": None})
            continue
        d = load_json(p)
        sc = d.get("scenarios", {}).get(scenario_key, {})
        q = sc.get("metrics_per_field", {}).get("q", {})
        psi = sc.get("metrics_per_field", {}).get("psi", {})
        out.append({
            "label": label,
            "data": {
                "psi_ev": psi.get("full", {}).get("ev"),
                "q_ev": q.get("full", {}).get("ev"),
            },
        })
    return out


def load_neural_summary(root: Path) -> dict[str, dict] | None:
    p = root / "qg_neural_s0_s1_cross_scenario" / "results_lag5_noise0.05_bias0.1.json"
    if not p.exists():
        return None
    d = load_json(p)
    return d["results"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-root", default=str(ROOT / "reports/qg/outputs"))
    ap.add_argument("--out", default=str(ROOT / "reports/qg/outputs/qg_neural_report.md"))
    args = ap.parse_args()

    out_root = Path(args.json_root)
    da_s0 = load_da_summary(out_root, "qg_repro_validation", "test_s0")
    da_s1 = load_da_summary(out_root, "qg_repro_validation_s1", "test_s1")
    neural = load_neural_summary(out_root)

    lines = []
    add = lines.append
    add("# QG Case Study: Benchmarked Schemes Overview")
    add("")
    add("Two-layer quasi-geostrophic (QG) Phillips-channel case study -- the four DA "
        "baselines (ETKF, EnKF, Strong-4DVar, Weak-4DVar) plus four neural DirectUNet "
        "variants on the T-merged-into-channels backbone (Q1 obs-only, Q2 oracle "
        "forcing+param conditioning, Q3 noisy-trained conditioning, Q4 noisy "
        "conditioning + IC), on both the S0 (no model error) reference case and the S1 "
        "(combined model-error) case.")
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

    add("## 2. Summary metrics (S0 + S1, N=100, lag=5.0d/noise_frac=0.05/"
        "s1_param_bias=s1_amp_bias=0.1)")
    add("")
    add("Pooled EV (higher is better) on ψ (streamfunction, both layers) and PV-q "
        "(both layers). All 8 rows evaluated at the identical config above.")
    add("")
    add("| scheme | S0 ψ EV | S0 PV-q EV | S1 ψ EV | S1 PV-q EV |")
    add("|---|---|---|---|---|")

    labels = [row["label"] for row in da_s0]
    s0_psi = [row["data"]["psi_ev"] if row["data"] else None for row in da_s0]
    s0_q = [row["data"]["q_ev"] if row["data"] else None for row in da_s0]
    s1_psi = [row["data"]["psi_ev"] if row["data"] else None for row in da_s1]
    s1_q = [row["data"]["q_ev"] if row["data"] else None for row in da_s1]

    neural_rows = []
    if neural is not None:
        for key, label, matched in NEURAL_SCHEMES:
            r = neural.get(key)
            if r is None:
                continue
            labels.append(label)
            s0_psi.append(r["S0"]["psi"]["pooled_ev"])
            s0_q.append(r["S0"]["q"]["pooled_ev"])
            s1_psi.append(r["S1"]["psi"]["pooled_ev"])
            s1_q.append(r["S1"]["q"]["pooled_ev"])
            neural_rows.append((label, matched))

    psi0_marks = rank_marks(s0_psi)
    q0_marks = rank_marks(s0_q)
    psi1_marks = rank_marks(s1_psi)
    q1_marks = rank_marks(s1_q)

    n_da = len(da_s0)
    for i, label in enumerate(labels):
        suffix = ""
        if i >= n_da:
            _, matched = neural_rows[i - n_da]
            suffix = "" if matched else " †"
        add(f"| {label}{suffix} | {mark(fmt(s0_psi[i]), psi0_marks.get(i))} | "
            f"{mark(fmt(s0_q[i]), q0_marks.get(i))} | "
            f"{mark(fmt(s1_psi[i]), psi1_marks.get(i))} | "
            f"{mark(fmt(s1_q[i]), q1_marks.get(i))} |")
    add("")
    add("(Best per column **bolded**, second-best *italicized*, ranked across all "
        "8 rows.)")
    add("")
    add("All four neural rows (Q1-Q4) were trained directly at this exact "
        "lag=5.0d/noise=0.05/s1_param_bias=s1_amp_bias=0.1 config -- a genuinely "
        "fair, matched-distribution comparison against the DA baselines for every "
        "row, no per-row caveat needed. (An earlier version of this table covered "
        "an older batch-folding-backbone DirectUNet family where three of five "
        "rows were only re-evaluated, not retrained, at this config and needed a "
        "† marker -- see PLAN.md's 2026-09-14 \"T-channels bench refresh\" "
        "section for the full history.)")
    add("")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
