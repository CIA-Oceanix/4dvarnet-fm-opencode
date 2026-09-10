#!/usr/bin/env python3
"""Generate the QG S0 DA-baselines report.

JSON-only generator (no QG/neural code imports). Reads the four DA baselines
(ETKF/EnKF/Strong-4DVar/Weak-4DVar) from ``reports/qg/outputs/
qg_repro_validation/`` and renders a comparison table (RMSE, CRPS, EV,
free-forecast EV) on the S0 scenario, plus pointers to the reconstruction
figures (see ``generate_qg_reconstruction_figs.py``).

Reference case (2026-09-08): lag=5.0d, noise_frac=0.05, N=100 test windows --
supersedes the earlier lag=1.0d/noise=0.01 case, which was found to be
unrealistically favorable (the background/free forecast alone already
explained most of the DA skill; see PLAN.md's "DA reference-case realism"
section for the full sensitivity analysis).

Does NOT cover the Q1/Q2 neural estimators -- see ``generate_qg_neural_report.py``
for the case-study overview report (scheme descriptions + a combined DA/neural
summary table), which links back to this report for DA method detail.

Renamed 2026-09-10 from ``generate_qg_neural_report.py``/``qg_neural_report.md``
(DA-only content, same as now) to free up that name for the new overview
report above -- git history before the rename still applies.

Run from the repository root::

    python reports/qg/generate_qg_da_report.py
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

METHODS = [("EnKF", "enkf"), ("ETKF", "etkf"),
           ("Weak-4DVar", "weak4dvar"), ("Strong-4DVar", "strong4dvar")]


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


def load_da_baselines(root: Path) -> list[dict]:
    out = []
    for label, fname in METHODS:
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
                "crps": s0.get("crps_mean"),
                "crps_normalized": s0.get("crps_normalized"),
                "crps_is_deterministic": s0.get("crps_is_deterministic"),
                "q_ev": q.get("full", {}).get("ev"),
                "q_ev_free": q.get("full", {}).get("ev_free"),
                "q1_ev": q.get("layer1", {}).get("ev"),
                "q1_ev_free": q.get("layer1", {}).get("ev_free"),
                "q2_ev": q.get("layer2", {}).get("ev"),
                "q2_ev_free": q.get("layer2", {}).get("ev_free"),
                "psi_ev": psy.get("full", {}).get("ev"),
                "psi_ev_free": psy.get("full", {}).get("ev_free"),
                "psi1_ev": psy.get("layer1", {}).get("ev"),
                "psi2_ev": psy.get("layer2", {}).get("ev"),
                "lag_days": d.get("init_lag_days"),
                "noise_frac": d.get("obs_noise_std_frac"),
                "num_windows": len(s0.get("rmse_list", [])) or None,
            },
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-root", default=str(ROOT / "reports/qg/outputs"))
    ap.add_argument("--out", default=str(ROOT / "reports/qg/outputs/qg_da_report.md"))
    args = ap.parse_args()

    out_root = Path(args.json_root)
    da = load_da_baselines(out_root)

    lines = []
    add = lines.append
    add("# QG DA Baselines (S0 reference case)")
    add("")
    add("Reference case (2026-09-08): **lag=5.0d, noise_frac=0.05, N=100 test "
        "windows**. Supersedes the earlier lag=1.0d/noise=0.01 case, which was "
        "found to be unrealistically favorable -- at that setting the "
        "background (free forecast, no assimilation) alone already reached "
        "psi full EV≈0.98, so DA's high score there was mostly inherited from "
        "the background rather than earned from the observational update. "
        "See `PLAN.md`'s \"DA reference-case realism\" section for the full "
        "lag/noise sensitivity analysis behind this choice.")
    add("")

    add("## DA baselines (S0, PV-q and ψ)")
    add("")
    add("CRPS is computed per-window on the q-state: ensemble methods "
        "(ETKF/EnKF) score their real per-member spread; deterministic "
        "methods (4DVar) have no ensemble, so CRPS degenerates exactly to "
        "the mean absolute error (marked `*`) -- lower is better for both. "
        "CRPS (norm.) divides by the pooled truth PV std over the whole "
        "test set (not per-window -- avoids the distortion a low-variance "
        "window would introduce), giving a dimensionless, cross-method-"
        "comparable score.")
    add("")
    add("| method | PV RMSE | improv | CRPS | CRPS (norm.) | PV EV | PV q1 EV | "
        "PV q2 EV | ψ EV |")
    add("|---|---|---|---|---|---|---|---|---|")
    first_valid = next((row["data"] for row in da if row["data"] is not None), None)
    if first_valid is not None:
        add(f"| _free forecast_ | {fmt_sci(first_valid['free_rmse'])} | 1.0 | -- | -- | "
            f"{fmt(first_valid['q_ev_free'])} | {fmt(first_valid['q1_ev_free'])} | "
            f"{fmt(first_valid['q2_ev_free'])} | {fmt(first_valid['psi_ev_free'])} |")
    else:
        add("| _free forecast_ | -- | -- | -- | -- | -- | -- | -- | -- |")
    # Ranked (bold=best, italic=second-best) among the 4 DA methods only --
    # the free-forecast row above is a reference baseline, not a competing
    # method, so it's excluded from ranking.
    cols = ["rmse", "improv", "crps", "crps_normalized", "q_ev", "q1_ev", "q2_ev", "psi_ev"]
    higher_better = {"rmse": False, "improv": True, "crps": False, "crps_normalized": False,
                     "q_ev": True, "q1_ev": True, "q2_ev": True, "psi_ev": True}
    col_marks = {}
    for c in cols:
        vals = [row["data"][c] if row["data"] is not None else None for row in da]
        col_marks[c] = rank_marks(vals, higher_is_better=higher_better[c])

    for i, row in enumerate(da):
        d = row["data"]
        if d is None:
            add(f"| {row['label']} | -- | -- | -- | -- | -- | -- | -- | -- |")
            continue
        star = "*" if d["crps_is_deterministic"] else ""
        crps_str = mark(fmt_sci(d["crps"]) + star, col_marks["crps"].get(i))
        crps_norm_str = mark(fmt(d["crps_normalized"]) + star, col_marks["crps_normalized"].get(i))
        rmse_str = mark(fmt_sci(d["rmse"]), col_marks["rmse"].get(i))
        improv_str = mark(fmt(d["improv"]), col_marks["improv"].get(i))
        q_ev_str = mark(fmt(d["q_ev"]), col_marks["q_ev"].get(i))
        q1_ev_str = mark(fmt(d["q1_ev"]), col_marks["q1_ev"].get(i))
        q2_ev_str = mark(fmt(d["q2_ev"]), col_marks["q2_ev"].get(i))
        psi_ev_str = mark(fmt(d["psi_ev"]), col_marks["psi_ev"].get(i))
        add(f"| {row['label']} | {rmse_str} | {improv_str} | "
            f"{crps_str} | {crps_norm_str} | {q_ev_str} | {q1_ev_str} | "
            f"{q2_ev_str} | {psi_ev_str} |")
    add("")
    add("(Best per column **bolded**, second-best *italicized*, ranked among "
        "the 4 DA methods -- the free-forecast reference row above is excluded.)")
    add("")

    have_data = [row["data"] for row in da if row["data"] is not None]
    q2_collapse = [row["label"] for row in da if row["data"] is not None
                   and row["data"]["q2_ev"] is not None and row["data"]["q2_ev"] < 0]
    if q2_collapse:
        add(f"> **Caveat:** {', '.join(q2_collapse)} collapse on PV q layer2 "
            "(the unobserved lower layer) at this reference case -- their "
            "PV EV is negative there despite psi EV being the best of all "
            "4 methods. This is consistent with PV being a Laplacian-like "
            "operator on psi (q ≈ ∇²ψ): small high-wavenumber errors in an "
            "otherwise excellent psi analysis get amplified when inverted "
            "to PV, especially in the layer with no direct observations. "
            "EnKF is the only method strongly positive on **both** psi and "
            "PV q at this setting.")
        add("")

    if have_data:
        n = have_data[0].get("num_windows")
        lag = have_data[0].get("lag_days")
        noise = have_data[0].get("noise_frac")
        add(f"Computed on N={n} test windows, lag={lag}, noise_frac={noise} "
            "(should match the reference case above -- if not, these JSONs "
            "are stale, regenerate them).")
        add("")

    add("## Reconstruction examples")
    add("")
    add("3 example test windows (best/median/worst by ETKF's per-window "
        "pooled PV-q RMSE) x all 4 methods, showing truth | free-forecast | "
        "analysis for streamfunction (ψ, both layers) and PV (q, both "
        "layers), plus an animated DA cycle (raw obs | wind-stress curl "
        "forcing | truth | EnKF analysis, PV q1) over the 30-day window. "
        "Generated by `generate_qg_reconstruction_figs.py`.")
    add("")
    fig_dir = out_root / "figs"
    report_dir = Path(args.out).resolve().parent

    def _rel(fig_path):
        # Relative to the report file's own directory (not out_root/json-root
        # in general -- Markdown image paths resolve relative to the file
        # containing them), so the link works regardless of where --out or
        # --json-root point.
        return fig_path.resolve().relative_to(report_dir) if fig_path.exists() else None

    for label in ("best", "median", "worst"):
        add(f"### {label.capitalize()} window")
        add("")
        static_rel = _rel(fig_dir / f"qg_s0_reconstruction_{label}.png")
        if static_rel is not None:
            add(f"![{label} window reconstruction]({static_rel.as_posix()})")
        else:
            add("`--` (figure not yet generated).")
        add("")
        anim_rel = _rel(fig_dir / f"qg_s0_dacycle_{label}.gif")
        if anim_rel is not None:
            add(f"![{label} window DA-cycle animation (EnKF)]({anim_rel.as_posix()})")
        else:
            add("`--` (animation not yet generated).")
        add("")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
