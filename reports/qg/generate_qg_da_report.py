#!/usr/bin/env python3
"""Generate the QG DA-baselines report (S0 reference case + S1 revised
model-error case).

JSON-only generator (no QG/neural code imports). Reads the four DA baselines
(ETKF/EnKF/Strong-4DVar/Weak-4DVar) from ``reports/qg/outputs/
qg_repro_validation/`` (S0) and ``reports/qg/outputs/qg_repro_validation_s1/``
(S1), renders a comparison table (RMSE, CRPS, EV, free-forecast EV) for each
scenario, a condensed hyperparameter-sensitivity section, and a closing
synthesis table (best known config per method, S0 vs S1), plus pointers to
the reconstruction figures (see ``generate_qg_reconstruction_figs.py``).

Reference case (2026-09-08): lag=5.0d, noise_frac=0.05, N=100 test windows --
supersedes the earlier lag=1.0d/noise=0.01 case, which was found to be
unrealistically favorable (the background/free forecast alone already
explained most of the DA skill; see PLAN.md's "DA reference-case realism"
section for the full sensitivity analysis). S1 (2026-09-10) adds param bias +
corrupted wind + da_nx=32 structural mismatch on top of S0's setup.

ETKF default changed 2026-09-12: `etkf_ridge=1.0` (was the implicit ~1e-4
floor at `etkf_ridge<=0`) -- see the "Hyperparameter sensitivity analysis"
section below and `reports/qg/outputs/da_sensitivity_s0_s1_report.md` for
the full sweep this is based on.

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

# Best known config per method, from the 2026-09-11/12 sensitivity study.
# All four are the CURRENT shipped defaults (etkf_ridge=1.0 was promoted
# specifically because the sweep found it best -- nothing else changed).
BEST_CONFIG = {
    "EnKF": "N=80, inflation=1.0, loc_radius=6.0 (default, unchanged -- own "
            "inflation sweep found no improvement over this)",
    "ETKF": "N=80, inflation=1.0, loc_radius=6.0, **etkf_ridge=1.0** "
            "(default since 2026-09-12; was the implicit ~1e-4 floor)",
    "Strong-4DVar": "b_var_scale=1.0 (default, unchanged -- sweep over "
                    "0.3-3.0 found no improvement, roughly flat)",
    "Weak-4DVar": "b_var_scale=1.0, q_var_scale=0.1 (default, unchanged -- "
                  "sweep over 0.1-3.0 found 0.1 already best, higher values "
                  "monotonically worse)",
}


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


def load_da_baselines(root: Path, scen_dir: str, scen_key: str) -> list[dict]:
    out = []
    for label, fname in METHODS:
        p = root / scen_dir / f"{fname}.json"
        if not p.exists():
            out.append({"label": label, "data": None})
            continue
        d = load_json(p)
        s = d.get("scenarios", {}).get(scen_key, {})
        q = s.get("metrics_per_field", {}).get("q", {})
        psy = s.get("metrics_per_field", {}).get("psi", {})
        out.append({
            "label": label,
            "data": {
                "rmse": s.get("rmse_mean"),
                "free_rmse": s.get("forecast_rmse_mean"),
                "improv": s.get("forecast_improvement"),
                "crps": s.get("crps_mean"),
                "crps_normalized": s.get("crps_normalized"),
                "crps_is_deterministic": s.get("crps_is_deterministic"),
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
                "num_windows": len(s.get("rmse_list", [])) or None,
            },
        })
    return out


def render_baseline_table(add, da: list[dict], scen_label: str) -> None:
    add(f"## DA baselines ({scen_label}, PV-q and ψ)")
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
            "PV EV is negative there despite psi EV being competitive or "
            "the best of all 4 methods. This is consistent with PV being a "
            "Laplacian-like operator on psi (q ≈ ∇²ψ): small "
            "high-wavenumber errors in an otherwise excellent psi analysis "
            "get amplified when inverted to PV, especially in the layer "
            "with no direct observations.")
        add("")

    if have_data:
        n = have_data[0].get("num_windows")
        lag = have_data[0].get("lag_days")
        noise = have_data[0].get("noise_frac")
        add(f"Computed on N={n} test windows, lag={lag}, noise_frac={noise} "
            "(should match the reference case above -- if not, these JSONs "
            "are stale, regenerate them).")
        add("")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-root", default=str(ROOT / "reports/qg/outputs"))
    ap.add_argument("--out", default=str(ROOT / "reports/qg/outputs/qg_da_report.md"))
    args = ap.parse_args()

    out_root = Path(args.json_root)
    da_s0 = load_da_baselines(out_root, "qg_repro_validation", "test_s0")
    da_s1 = load_da_baselines(out_root, "qg_repro_validation_s1", "test_s1")

    lines = []
    add = lines.append
    add("# QG DA Baselines (S0 reference case + S1 model-error case)")
    add("")
    add("Reference case (2026-09-08): **lag=5.0d, noise_frac=0.05, N=100 test "
        "windows** (S0). Supersedes the earlier lag=1.0d/noise=0.01 case, "
        "which was found to be unrealistically favorable -- at that setting "
        "the background (free forecast, no assimilation) alone already "
        "reached psi full EV≈0.98, so DA's high score there was mostly "
        "inherited from the background rather than earned from the "
        "observational update. See `PLAN.md`'s \"DA reference-case realism\" "
        "section for the full lag/noise sensitivity analysis behind this "
        "choice.")
    add("")
    add("**S1** (revised 2026-09-10): shares S0's initial-uncertainty setup "
        "exactly (lag=5.0d, noise_frac=0.05) and adds three independent "
        "model-error sources on top -- param bias (`rd`/`rek` scaled by "
        "`1-0.1`), corrupted wind (amplitude bias + storm-location jitter), "
        "and structural resolution mismatch (`da_nx=32`, half the truth "
        "grid). Obs are always drawn from the true, unbiased, full-res "
        "trajectory -- only the DA method's own model sees the corruption.")
    add("")
    add("**ETKF default changed 2026-09-12**: `etkf_ridge=1.0` (was the "
        "implicit ~1e-4 floor at `etkf_ridge<=0`) -- see the \"Hyperparameter "
        "sensitivity analysis\" section below. All ETKF numbers in this "
        "report use the new default.")
    add("")

    render_baseline_table(add, da_s0, "S0")
    render_baseline_table(add, da_s1, "S1")

    # ---- Hyperparameter sensitivity analysis ----
    add("## Hyperparameter sensitivity analysis")
    add("")
    add("Condensed summary of the 2026-09-11/12 sensitivity study -- full "
        "sweep tables (finer inflation grids, additive inflation, the full "
        "ridge grid, EnKF's own inflation sensitivity) are in the dedicated "
        "`reports/qg/outputs/da_sensitivity_s0_s1_report.md` "
        "(`reports/qg/generate_da_sensitivity_report.py`); `PLAN.md` has the "
        "full narrative trail.")
    add("")
    add("**Motivation**: EnKF consistently beat ETKF on S1 with no known "
        "cause (N=100: ETKF q EV 0.307 vs EnKF 0.331, at ETKF's *old* "
        "default). The natural first hypothesis -- ETKF is under/"
        "over-inflated -- turned out not to explain it.")
    add("")
    add("**ETKF/EnKF inflation (multiplicative)**: both methods collapse in "
        "near-lockstep under any inflation above 1.0 -- e.g. S0 q EV at "
        "inflation={1.00,...,1.05}: ETKF {0.402,0.402,0.296,0.147,-40.4,"
        "-123.5} vs EnKF {0.463,0.435,0.313,0.156,-43.0,-124.3} (N=10), "
        "virtually the same curve, same catastrophic threshold. This rules "
        "out \"ETKF's deterministic transform is uniquely fragile to "
        "over-inflation\" as an explanation -- both ensemble methods share "
        "the fragility equally, so it cannot explain the EnKF>ETKF gap. "
        "**Additive inflation** (never exercised before this study) also "
        "only ever hurts, though gracefully (no catastrophic divergence).")
    add("")
    add("**`etkf_ridge` (Kalman-gain transform-matrix regularization) is the "
        "real lever** -- a mechanism EnKF has no equivalent of. "
        "Monotonically helps up to a peak (S1 peaks around ridge=2.0, S0 "
        "plateaus 0.1-1.0), then mildly declines. `ridge=1.0` was picked as "
        "near-optimal on both scenarios. **Confirmed at full N=100** (not "
        "just the N=10 sweep):")
    add("")
    add("| | ETKF (old default) | EnKF | **ETKF + ridge=1.0 (new default)** |")
    add("|---|---|---|---|")
    add("| S0 psi EV | 0.921 | 0.947 | **0.957** |")
    add("| S0 q EV | 0.405 | **0.481** | 0.476 |")
    add("| S1 psi EV | 0.874 | 0.896 | **0.926** |")
    add("| S1 q EV | 0.307 | 0.331 | **0.357** |")
    add("")
    add("ETKF+ridge=1.0 beats EnKF outright on **both** fields on S1, and "
        "ties/beats it on S0 -- no trade-off on the unobserved PV layer "
        "either. This is why `etkf_ridge=1.0` was promoted to the default "
        "(2026-09-12): the previously \"unexplained\" EnKF>ETKF gap was "
        "largely an artifact of ETKF running with an under-regularized "
        "transform-matrix inversion, not a fundamental method limitation.")
    add("")
    add("**4DVar (Strong/Weak) covariance-weighting sweep -- negative "
        "result**: motivated by the same logic (both 4DVar variants "
        "collapse on PV q layer2 under S1: Strong -0.857, Weak -0.501 vs "
        "ETKF/EnKF staying positive), swept Strong-4DVar's `b_var_scale` "
        "(background-covariance whitening scale) and Weak-4DVar's "
        "`q_var_scale` (per-step model-error weight) on S1, N=5 (4DVar is "
        "~15-20x more expensive per window than ETKF/EnKF). Unlike ETKF's "
        "ridge, **neither lever helps**: `b_var_scale` is essentially flat "
        "across 0.3-3.0 (q EV -1.06/-1.07/-1.11); `q_var_scale` is "
        "monotonically *worse* the higher it's pushed above the existing "
        "default 0.1 (q EV -0.91/-0.99/-1.09/-1.11 at 0.1/0.3/1.0/3.0) -- "
        "giving the model-error controls more freedom actively hurts rather "
        "than helping. Both methods' existing defaults (`b_var_scale=1.0`, "
        "`q_var_scale=0.1`) were already at or near the best point found. "
        "This is consistent with the 4DVar q-layer2 collapse being a more "
        "structural limitation (a single optimized trajectory has no "
        "ensemble spread to exploit on the unobserved layer) rather than a "
        "fixable covariance-tuning gap, unlike ETKF's case.")
    add("")

    # ---- Synthesis ----
    add("## Synthesis: best configuration per method (S0 vs S1)")
    add("")
    add("Following the sensitivity study above, every method's *best known* "
        "config is now also its *shipped default* -- ETKF's default was the "
        "one that changed (`etkf_ridge=1.0`); EnKF, Strong-4DVar, and "
        "Weak-4DVar were already at their best tested configuration.")
    add("")
    add("| method | best config | S0 ψ EV | S0 PV EV | S1 ψ EV | S1 PV EV |")
    add("|---|---|---|---|---|---|")
    by_label_s0 = {row["label"]: row["data"] for row in da_s0}
    by_label_s1 = {row["label"]: row["data"] for row in da_s1}
    for label, _ in METHODS:
        d0 = by_label_s0.get(label)
        d1 = by_label_s1.get(label)
        cfg = BEST_CONFIG.get(label, "--")
        add(f"| {label} | {cfg} | {fmt(d0['psi_ev']) if d0 else '--'} | "
            f"{fmt(d0['q_ev']) if d0 else '--'} | "
            f"{fmt(d1['psi_ev']) if d1 else '--'} | "
            f"{fmt(d1['q_ev']) if d1 else '--'} |")
    add("")
    add("(PV-q and ψ full-field EV, N=100 both scenarios. Not "
        "rank-marked here -- see the per-scenario tables above for "
        "bold/italic ranking; this table's purpose is the config-per-method "
        "mapping, not re-ranking.)")
    add("")

    add("## Reconstruction examples")
    add("")
    add("3 example test windows (best/median/worst by ETKF's per-window "
        "pooled PV-q RMSE) x all 4 methods, showing truth | free-forecast | "
        "analysis for streamfunction (ψ, both layers) and PV (q, both "
        "layers), plus an animated DA cycle (raw obs | wind-stress curl "
        "forcing | truth | EnKF analysis, PV q1) over the 30-day window. "
        "Generated by `generate_qg_reconstruction_figs.py`. S0 only.")
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
