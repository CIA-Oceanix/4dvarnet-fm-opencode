#!/usr/bin/env python3
"""Generate the QG q-state vs psi-state DA comparison report.

JSON-only generator (no QG/neural code imports). It consumes the curated
result JSONs under ``reports/qg/outputs/`` for the q-state (default) and
psi-state DA ETKF runs on the S0 and S1 scenarios, and renders a Markdown
report summarising the per-field metrics (PV q1/q2/qall and streamfunction
psi1/psi2 explained variance), the two DA representations' relative skill,
and the decision that **q-state remains the default DA configuration**.

Metrics are pooled explained variance (EV) from ``metrics_per_field``, with
the aggregate headline from ``expvar_full`` (the full PV q field).

Run from the repository root::

    python reports/qg/generate_qg_psi_state_report.py
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def fmt_ev(x) -> str:
    return f"{x:+.3f}" if isinstance(x, (int, float)) else str(x)


def find_json(root: Path, subdir: str) -> dict | None:
    d = root / subdir
    if not d.is_dir():
        return None
    jsons = sorted(d.glob("*.json"))
    return load_json(jsons[0]) if jsons else None


def ev_cell(mpf: dict, field: str) -> str:
    f = mpf.get(field, {})
    l1 = f.get("layer1", {}).get("ev")
    l2 = f.get("layer2", {}).get("ev")
    full = f.get("full", {}).get("ev")
    return f"{fmt_ev(l1)} / {fmt_ev(l2)} / {fmt_ev(full)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-root", default=str(ROOT / "reports/qg/outputs"))
    ap.add_argument("--out", default=str(ROOT / "reports/qg/outputs/qg_psi_state_report.md"))
    args = ap.parse_args()

    root = Path(args.json_root)
    missing = []

    cases = [
        ("S0 (error-free)", "qg_matrix_c4_psi", "qg_s0_psi_state_nz0p01_lag1p0", "test_s0"),
        ("S1 same-res (da_nx=64)", "qg_s1_nores_lag1p0", "qg_s1_nores_psi_state_lag1p0", "test_s1"),
        ("S1 cross-res (da_nx=32)", "qg_s1_da32_lag1p0", "qg_s1_psi_state_da32_lag1p0", "test_s1"),
    ]

    # Load all result dicts once.
    results = {}
    for label, q_sub, psi_sub, scen in cases:
        qd = find_json(root, q_sub)
        pd = find_json(root, psi_sub)
        if qd is None:
            missing.append(q_sub)
        if pd is None:
            missing.append(psi_sub)
        results[label] = (qd, pd, q_sub, psi_sub, scen)

    md = []
    add = md.append

    add("# QG DA — q-state vs psi-state comparison")
    add("")
    add("**Date:** 2026-09-03")
    add("**Scope:** ETKF (N=80, inflation 1.0, Gaspari–Cohn radius 6), psi-obs "
        "(upper-layer streamfunction, 4 random meridional columns/day), lag 1.0 d. "
        "S0 (error-free) and S1 (param bias + corrupted wind); S1 at same-res "
        "da_nx=64 and cross-res da_nx=32 (vs 64×64 truth).")
    add("")
    add("## Decision: q-state is the default DA configuration")
    add("")
    add("The **q-state** DA formulation (PV `q` as the forecast/analysis state, "
        "with the streamfunction computed on demand via spectral inversion for "
        "psi-obs) is the **production default** for QG data assimilation. The "
        "**psi-state** variant (`obs_var=\"psi_state\"`), which carries the "
        "streamfunction as the state and converts q↔ψ each integration step, is "
        "a research alternative retained for comparison, not the default.")
    add("")
    add("- Code default: `--obs-var` resolves to `\"q\"` (q-state) in both "
        "`evaluation/run_qg_baselines.py` and `evaluation/sweep_qg_baselines.py`.")
    add("- The q-state DA scenario (psi-obs, 4 cols/day) is the configuration "
        "used for the production S0/S1 QG DA-baseline runs and for the "
        "`reports/qg/generate_qg_s0s1_report.py` consolidated report.")
    add("")

    add("## 1. Representations")
    add("")
    add("- **q-state (default):** the DA state is potential vorticity `q`. The "
        "psi-obs operator inverts `q → ψ` spectrally each observation step. The "
        "metrics measure the PV q field directly (well-conditioned).")
    add("- **psi-state:** the DA state is the streamfunction `ψ`. The psi-obs "
        "operator is an H-mode read of the psi-state (`_PsiMixin.streamfunctions` "
        "is the identity reshape) spectrally upsampled to the obs grid. The "
        "metrics convert the psi analysis back to PV via `forward_pv` "
        "(q ≈ ∇²ψ) before computing q-field skill.")
    add("")

    add("## 2. Per-case results")
    add("")

    for label, (qd, pd, q_sub, psi_sub, scen) in results.items():
        add(f"### {label}")
        add("")
        if qd is not None:
            qmpf = qd["scenarios"].get(scen, {}).get("metrics_per_field", {})
            pmpf = pd["scenarios"].get(scen, {}).get("metrics_per_field", {}) \
                if pd else {}
            if qmpf or pmpf:
                add("| field | metric (layer1 / layer2 / full) |")
                add("|---|---|")
                if qmpf:
                    add(f"| q-state (default) q | {ev_cell(qmpf, 'q')} |")
                    add(f"| q-state (default) ψ | {ev_cell(qmpf, 'psi')} |")
                if pmpf:
                    add(f"| psi-state q | {ev_cell(pmpf, 'q')} |")
                    add(f"| psi-state ψ | {ev_cell(pmpf, 'psi')} |")
            else:
                qs = qd["scenarios"][scen]
                ps = pd["scenarios"][scen] if pd else {}
                add("| representation | qall (expvar_full) |")
                add("|---|---|")
                add(f"| q-state (default) | {fmt_ev(qs.get('expvar_full'))} |")
                if pd:
                    add(f"| psi-state | {fmt_ev(ps.get('expvar_full'))} |")
            add("")
            add(f"- Headline `expvar_full` (qall): q-state "
                f"{fmt_ev(qd['scenarios'][scen].get('expvar_full'))} vs psi-state "
                f"{fmt_ev(pd['scenarios'][scen].get('expvar_full')) if pd else '—'}.")
            add("")

    add("## 3. Summary table")
    add("")
    add("| case | q-state qall | psi-state qall | q-state ψ1/ψ2 | psi-state ψ1/ψ2 |")
    add("|---|---|---|---|---|")
    for label, (qd, pd, q_sub, psi_sub, scen) in results.items():
        qs = qd["scenarios"][scen]
        ps = pd["scenarios"][scen] if pd else {}
        qall = qs.get("expvar_full")
        pall = ps.get("expvar_full")
        qsp = (ev_cell(qs["metrics_per_field"], "psi")
               if "metrics_per_field" in qs else "—")
        psp = (ev_cell(ps["metrics_per_field"], "psi")
               if "metrics_per_field" in ps else "—")
        add(f"| {label} | {fmt_ev(qall)} | {fmt_ev(pall)} | {qsp} | {psp} |")
    add("")

    add("## 4. Interpretation")
    add("")
    add("- **The q-state (default) wins on the PV q field in every case.** On S0 "
        "at noise 0.01 qall is 0.752 vs psi-state 0.583 (gap dominated by q2, "
        "0.688 vs 0.403); on S1 same-res it is 0.428 vs −2.93 and on da_nx=32 "
        "0.340 vs −3.22, where the psi-state q field collapses to strongly "
        "negative EV.")
    add("- **psi-state is competitive on the streamfunction field.** On S0 its "
        "ψ1/ψ2 (0.976/0.978) is the best per-field result, slightly above q-state "
        "(0.966/0.971); on S1 same-res its ψ1 (0.814) clearly beats q-state "
        "(0.435). But on S1 cross-res da_nx=32 q-state ψ2 (0.806) exceeds "
        "psi-state ψ2 (0.392).")
    add("- **Why the psi-state q field is degenerate:** the PV diagnostic "
        "q ≈ ∇²ψ (via `forward_pv`) amplifies high-wavenumber psi-analysis error "
        "by K², so a psi analysis that is skilful in bulk (large-scale dominated, "
        "positive ψ EV) still has small-scale error that explodes under the PV "
        "conversion. This is most severe on the corrupted S1 case and grows with "
        "the psi→q round-trip; it is a physical ψ↔q representation limitation, "
        "not a code bug (free forecasts are bit-identical).")
    add("- **Conclusion:** the q-state representation is the robust default: it "
        "keeps the PV q field — the physical variable the benchmark scores — "
        "well-conditioned across S0 and the corrupted cross-resolution S1 alike. "
        "The psi-state variant remains useful for studying streamfunction-direct "
        "observations but is not the production DA configuration.")
    add("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    print(f"wrote {out}")
    if missing:
        print("WARNING: missing JSONs:", missing)


if __name__ == "__main__":
    main()