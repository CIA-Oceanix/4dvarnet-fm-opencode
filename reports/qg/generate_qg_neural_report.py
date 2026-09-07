#!/usr/bin/env python3
"""Generate the QG neural baseline (Q1/Q2) report.

JSON-only generator (no QG/neural code imports). It reads the Q1 (DirectUNet)
and Q2 (VanillaCFM tau=0) `results.json` from the standard experiment dirs
(rendering `--` when a run is not yet available) and compares their
streamfunction (psi) and PV-q (q) metrics against the four DA baselines in
``reports/qg/outputs/qg_repro_validation/`` on the S0 scenario.

The neural and DA PV-q metrics are directly comparable: both are physical-unit
pooled RMSE / explained-variance on the daily-mean full 2-layer PV q field. The
neural streamfunction metrics have no DA analogue here (the DA baselines report
psi-EV via `metrics_per_field`), so they appear only in the neural section.

Run from the repository root::

    python reports/qg/generate_qg_neural_report.py
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def fmt(x) -> str:
    return f"{x:.4f}" if x is not None else "--"


def fmt_sci(x, nd=2) -> str:
    return f"{x:.{nd}e}" if x is not None else "--"


def neural_row(results_path: Path) -> dict | None:
    """Extract the S0 psi/q pooled + per-layer metrics from a QG neural results.json."""
    if not results_path.exists():
        return None
    r = load_json(results_path)
    s0 = r.get("s0", {})
    return {
        "config": r.get("config", {}),
        "model_type": r.get("model_type"),
        "psi": s0.get("psi", {}),
        "q": s0.get("q", {}),
        "train_time_seconds": r.get("train_time_seconds"),
    }


def load_da_baselines(root: Path) -> list[dict]:
    methods = [("EnKF", "enkf"), ("ETKF", "etkf"),
               ("Weak-4DVar", "weak4dvar"), ("Strong-4DVar", "strong4dvar")]
    out = []
    for label, fname in methods:
        p = root / "qg_repro_validation" / f"{fname}.json"
        if not p.exists():
            out.append({"label": label, "data": None})
            continue
        d = load_json(p)
        s0 = d.get("scenarios", {}).get("test_s0", {})
        q = s0.get("metrics_per_field", {}).get("q", {})
        psy = s0.get("metrics_per_field", {}).get("psi", {})
        out.append({
            "label": label,
            "data": {
                "rmse": s0.get("rmse_mean"),
                "free_rmse": s0.get("forecast_rmse_mean"),
                "improv": s0.get("forecast_improvement"),
                "q_ev": q.get("full", {}).get("ev"),
                "q1_ev": q.get("layer1", {}).get("ev"),
                "q2_ev": q.get("layer2", {}).get("ev"),
                "psi_ev": psy.get("full", {}).get("ev"),
            },
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", default=str(ROOT / "experiments"))
    ap.add_argument("--json-root", default=str(ROOT / "reports/qg/outputs"))
    ap.add_argument("--out", default=str(ROOT / "reports/qg/outputs/qg_neural_report.md"))
    args = ap.parse_args()

    exp_root = Path(args.exp_root)
    out_root = Path(args.json_root)

    q1 = neural_row(exp_root / "Q1_direct_unet_s0" / "results.json")
    q2 = neural_row(exp_root / "Q2_vanilla_cfm_s0" / "results.json")
    da = load_da_baselines(out_root)

    lines = []
    add = lines.append
    add("# QG Neural Baseline (S0) — Q1 DirectUNet / Q2 VanillaCFM(τ=0)")
    add("")
    add("Neural estimators are trained on the daily-mean full 2-layer "
        "**streamfunction** (ψ) with an auxiliary PV-q loss, and scored on the "
        "same daily-mean fields (ψ and PV q). The PV-q RMSE/EV are directly "
        "comparable to the QG DA baselines (`qg_repro_validation`), which are "
        "also scored on the full PV q field.")
    add("")

    # DA baseline table
    add("## DA baselines (S0 reference, PV-q)")
    add("")
    add("| method | PV RMSE | free RMSE | improv | PV EV | PV q1 EV | PV q2 EV | ψ EV |")
    add("|---|---|---|---|---|---|---|---|")
    free_ref = None
    for row in da:
        d = row["data"]
        if d is None:
            add(f"| {row['label']} | -- | -- | -- | -- | -- | -- | -- |")
            continue
        free_ref = free_ref or d["free_rmse"]
        add(f"| {row['label']} | {fmt_sci(d['rmse'])} | {fmt_sci(d['free_rmse'])} | "
            f"{fmt(d['improv'])} | {fmt(d['q_ev'])} | {fmt(d['q1_ev'])} | "
            f"{fmt(d['q2_ev'])} | {fmt(d['psi_ev'])} |")
    add(f"| _free forecast_ | {fmt_sci(free_ref)} | — | 1.0 | {fmt(None)} | "
        f"{fmt(None)} | {fmt(None)} | — |")
    add("")

    # Neural table
    models = [("Q1 DirectUNet", q1), ("Q2 VanillaCFM(τ=0)", q2)]
    add("## QG neural estimators (S0)")
    add("")
    add("Metrics on the daily-mean full 2-layer ψ and PV q. `--` = run not yet "
        "available (training/eval pending).")
    add("")
    add("| model | nx | ψ RMSE | ψ EV | q RMSE | q EV | q1 EV | q2 EV |")
    add("|---|---|---|---|---|---|---|---|")
    for label, r in models:
        if r is None:
            add(f"| {label} | -- | -- | -- | -- | -- | -- |")
            continue
        cfg = r["config"]
        p, q = r["psi"], r["q"]
        add(f"| {label} | {cfg.get('nx', '--')} | {fmt_sci(p['pooled_rmse'])} | "
            f"{fmt(p['pooled_ev'])} | {fmt_sci(q['pooled_rmse'])} | "
            f"{fmt(q['pooled_ev'])} | {fmt(q['layer1']['ev'])} | "
            f"{fmt(q['layer2']['ev'])} |")
    add("")

    # Config note
    add("## Experiment settings")
    add("")
    for label, r in models:
        if r is None:
            add(f"- **{label}** — `--` (not run yet).")
            continue
        c = r["config"]
        add(f"- **{label}** — nx={c.get('nx')}, state_dim={c.get('state_dim')}, "
            f"epochs={c.get('epochs')}, train={c.get('num_train_windows')} "
            f"windows, test={c.get('num_test_windows')} windows, "
            f"q_loss_weight={c.get('q_loss_weight')}, "
            f"n_members={c.get('n_members')}.")
    ms = [m for _, m in models if m is not None and m.get("train_time_seconds") is not None]
    if ms:
        add("")
        add("Training time (both runs): "
            + ", ".join(f"{m['model_type']} {m['train_time_seconds']:.1f}s"
                        for m in ms) + ".")
    add("")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
