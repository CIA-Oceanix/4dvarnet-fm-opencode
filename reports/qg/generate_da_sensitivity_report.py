#!/usr/bin/env python3
"""Generate the QG DA (ETKF/EnKF) hyperparameter sensitivity report.

JSON-only generator (no QG/neural code imports). Covers multiplicative
inflation, additive inflation, and ridge regularization sensitivity, on both
the S0 reference case and the S1 revised combined-error case, for ETKF and
EnKF. Motivated by the open question of why EnKF consistently beats ETKF on
S1 (see PLAN.md's "ETKF inflation sweep" and "ETKF inflation sensitivity,
part 2" sections for the full narrative) -- this report consolidates that
sensitivity analysis into one place instead of scattered scratch-driver
output.

All sweeps use N=10 test windows (the first 10 of the production 100-window
S0 test cache; S1 windows are the same 10 truth trajectories re-wrapped with
the combined param-bias + wind-corruption + da_nx=32 model error) at the
current reference case (lag=5.0d, noise_frac=0.05). N=10 keeps each sweep
point cheap; absolute EV values will differ somewhat from the N=100
headline numbers in `qg_da_report.md`, but the *sensitivity pattern* (what
direction/magnitude each hyperparameter moves EV) is what this report is
for, not a replacement headline number.

Run from the repository root::

    python reports/qg/generate_da_sensitivity_report.py
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S1_DIR = "qg_repro_validation_s1"
SWEEP_DIR = "qg_da_sensitivity_sweep"


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def fmt(x) -> str:
    return f"{x:.3f}" if x is not None else "--"


def rank_marks(values: list, higher_is_better: bool = True) -> dict:
    """{index: 'bold'|'italic'} for the best/second-best of `values` (None skipped).

    Project convention: bold the best score per column, italicize the
    second-best.
    """
    present = [(i, v) for i, v in enumerate(values) if v is not None]
    present.sort(key=lambda t: t[1], reverse=higher_is_better)
    marks = {}
    if len(present) >= 1:
        marks[present[0][0]] = "bold"
    if len(present) >= 2:
        marks[present[1][0]] = "italic"
    return marks


def mark(s: str, kind) -> str:
    if kind == "bold":
        return f"**{s}**"
    if kind == "italic":
        return f"*{s}*"
    return s


def metrics(d, scenario: str):
    if d is None:
        return None
    s = d.get("scenarios", {}).get(scenario)
    if s is None:
        return None
    q = s["metrics_per_field"]["q"]["full"]
    psi = s["metrics_per_field"]["psi"]["full"]
    return {"q_ev": q["ev"], "q_ev_free": q["ev_free"],
            "psi_ev": psi["ev"], "psi_ev_free": psi["ev_free"],
            "improv": s["forecast_improvement"]}


def table_rows(root: Path, grid, scenario: str):
    """grid: list of (label, {"s0": relpath_or_None, "s1": relpath_or_None})."""
    rows = []
    for label, paths in grid:
        row = {"label": label}
        for scen_key, rel in paths.items():
            if rel is None:
                row[scen_key] = None
                continue
            d = load_json(root / rel)
            row[scen_key] = metrics(d, "test_s0" if scen_key == "s0" else "test_s1")
        rows.append(row)
    return rows


def render_ev_table(add, rows, col_label, scen_keys, free_row=None):
    header = f"| {col_label} | " + " | ".join(
        f"{sk.upper()} q EV" for sk in scen_keys) + " |"
    sep = "|---|" + "---|" * len(scen_keys)
    add(header)
    add(sep)
    if free_row is not None:
        cells = " | ".join(fmt(free_row.get(sk)) for sk in scen_keys)
        add(f"| _free forecast_ | {cells} |")
    col_marks = {}
    for sk in scen_keys:
        vals = [r[sk]["q_ev"] if r.get(sk) is not None else None for r in rows]
        col_marks[sk] = rank_marks(vals, higher_is_better=True)
    for i, r in enumerate(rows):
        cells = []
        for sk in scen_keys:
            m = r.get(sk)
            if m is None:
                cells.append("--")
            else:
                cells.append(mark(fmt(m["q_ev"]), col_marks[sk].get(i)))
        add(f"| {r['label']} | " + " | ".join(cells) + " |")
    add("")
    add("(PV-q full-field EV; best per column **bolded**, second-best "
        "*italicized*.)")
    add("")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-root", default=str(ROOT / "reports/qg/outputs"))
    ap.add_argument("--out", default=str(ROOT / "reports/qg/outputs/da_sensitivity_s0_s1_report.md"))
    args = ap.parse_args()
    root = Path(args.json_root)

    lines = []
    add = lines.append
    add("# QG DA Baseline Hyperparameter Sensitivity (S0 / S1)")
    add("")
    add("ETKF and EnKF sensitivity to their own tunable hyperparameters "
        "(inflation, additive inflation, ridge regularization), on both "
        "the S0 reference case (lag=5.0d, noise_frac=0.05) and the revised "
        "S1 combined-error case (S0's initial-uncertainty setup + param "
        "bias 0.1 + corrupted wind + `da_nx=32` structural resolution "
        "mismatch). All sweeps use **N=10 test windows** (the first 10 of "
        "the production 100-window test cache) -- cheap enough to sweep "
        "densely, but absolute EV values differ somewhat from the N=100 "
        "headline numbers in `qg_da_report.md`/`PLAN.md`'s S1 table. This "
        "report is about the *sensitivity pattern* per hyperparameter, not "
        "a replacement headline comparison.")
    add("")
    add("Motivation: EnKF consistently beats ETKF on S1 "
        "(N=100: ETKF q EV 0.307 vs EnKF 0.331) and the reason was an open "
        "question. The natural first hypothesis -- ETKF is under/over-"
        "inflated -- is what this report's inflation/additive/ridge sweeps "
        "test.")
    add("")

    # ---- ETKF multiplicative inflation ----
    add("## ETKF: multiplicative inflation")
    add("")
    etkf_infl_grid = [
        ("1.00", {"s0": f"{SWEEP_DIR}/s0_etkf_inflation1.json",
                  "s1": f"{S1_DIR}/etkf_n10.json"}),
        ("1.01", {"s0": f"{SWEEP_DIR}/s0_etkf_inflation1.01.json",
                  "s1": f"{S1_DIR}/etkf_n10_inflation1.01.json"}),
        ("1.02", {"s0": f"{SWEEP_DIR}/s0_etkf_inflation1.02.json",
                  "s1": f"{S1_DIR}/etkf_n10_inflation1.02.json"}),
        ("1.03", {"s0": f"{SWEEP_DIR}/s0_etkf_inflation1.03.json",
                  "s1": f"{S1_DIR}/etkf_n10_inflation1.03.json"}),
        ("1.04", {"s0": f"{SWEEP_DIR}/s0_etkf_inflation1.04.json",
                  "s1": f"{S1_DIR}/etkf_n10_inflation1.04.json"}),
        ("1.05", {"s0": f"{SWEEP_DIR}/s0_etkf_inflation1.05.json",
                  "s1": f"{S1_DIR}/etkf_n10_infl1.05.json"}),
        ("1.10", {"s0": None, "s1": f"{S1_DIR}/etkf_n10_infl1.1.json"}),
        ("1.20", {"s0": None, "s1": f"{S1_DIR}/etkf_n10_infl1.2.json"}),
        ("1.30", {"s0": None, "s1": f"{S1_DIR}/etkf_n10_infl1.3.json"}),
    ]
    rows = table_rows(root, etkf_infl_grid, "test_s0")
    free = rows[0].get("s0") if rows else None
    free_row = {"s0": free["q_ev_free"] if free else None,
                "s1": rows[0]["s1"]["q_ev_free"] if rows[0].get("s1") else None}
    render_ev_table(add, rows, "inflation", ["s0", "s1"], free_row=free_row)
    add("S0 (no model error) tolerates inflation somewhat better than S1, "
        "but both decline steeply well before 1.05; S1's decline becomes "
        "catastrophic divergence (note the scale change) from 1.05 on. "
        "`inflation=1.0` (effectively off) is the safe choice on both "
        "scenarios at this N.")
    add("")

    # ---- ETKF additive inflation ----
    add("## ETKF: additive inflation")
    add("")
    add("`etkf_additive`: Gaussian noise added directly to the ensemble "
        "each step (raw q-field units), expressed here as a fraction of "
        "the truth q field's own std (~2.06e-5 for both S0 and S1 -- same "
        "truth trajectories, only the DA model differs). Never exercised "
        "before this sensitivity study.")
    add("")
    etkf_add_grid = [
        ("0%", {"s0": f"{SWEEP_DIR}/s0_etkf_inflation1.json",
                "s1": f"{S1_DIR}/etkf_n10.json"}),
        ("~1%", {"s0": f"{SWEEP_DIR}/s0_etkf_additive0.01.json",
                 "s1": f"{S1_DIR}/etkf_n10_additive2e-7.json"}),
        ("~5%", {"s0": f"{SWEEP_DIR}/s0_etkf_additive0.05.json",
                 "s1": f"{S1_DIR}/etkf_n10_additive1e-6.json"}),
        ("~10%", {"s0": f"{SWEEP_DIR}/s0_etkf_additive0.1.json",
                  "s1": f"{S1_DIR}/etkf_n10_additive2e-6.json"}),
        ("~24%", {"s0": f"{SWEEP_DIR}/s0_etkf_additive0.24.json",
                  "s1": f"{S1_DIR}/etkf_n10_additive5e-6.json"}),
    ]
    rows = table_rows(root, etkf_add_grid, "test_s0")
    render_ev_table(add, rows, "additive (% of q std)", ["s0", "s1"], free_row=free_row)
    add("Also monotonically degrades EV on both scenarios, but **gracefully** "
        "-- no catastrophic divergence at any tested magnitude, unlike "
        "multiplicative inflation's blow-up past 1.04-1.05. Neither "
        "inflation flavor *helps*; both are pure downside once past zero.")
    add("")

    # ---- ETKF ridge ----
    add("## ETKF: ridge regularization")
    add("")
    add("`etkf_ridge`: multiplier on the Kalman-gain transform-matrix "
        "regularization (targets the deterministic square-root transform's "
        "matrix inversion directly, unlike inflation which targets ensemble "
        "spread). `etkf_ridge<=0` (the default used everywhere else in this "
        "report) internally floors to an implicit 1e-4-equivalent, so the "
        "\"default\" row below reuses the inflation=1.0 baseline files. "
        "Never exercised before this study.")
    add("")
    etkf_ridge_grid = [
        ("default (~1e-4)", {"s0": f"{SWEEP_DIR}/s0_etkf_inflation1.json",
                              "s1": f"{S1_DIR}/etkf_n10.json"}),
        ("1e-3", {"s0": f"{SWEEP_DIR}/s0_etkf_ridge0.001.json",
                  "s1": f"{SWEEP_DIR}/s1_etkf_ridge0.001.json"}),
        ("1e-2", {"s0": f"{SWEEP_DIR}/s0_etkf_ridge0.01.json",
                  "s1": f"{SWEEP_DIR}/s1_etkf_ridge0.01.json"}),
        ("1e-1", {"s0": f"{SWEEP_DIR}/s0_etkf_ridge0.1.json",
                  "s1": f"{SWEEP_DIR}/s1_etkf_ridge0.1.json"}),
        ("1.0", {"s0": f"{SWEEP_DIR}/s0_etkf_ridge1.json",
                 "s1": f"{SWEEP_DIR}/s1_etkf_ridge1.json"}),
        ("2.0", {"s0": None, "s1": f"{SWEEP_DIR}/s1_etkf_ridge2.json"}),
        ("5.0", {"s0": None, "s1": f"{SWEEP_DIR}/s1_etkf_ridge5.json"}),
    ]
    rows = table_rows(root, etkf_ridge_grid, "test_s0")
    render_ev_table(add, rows, "ridge", ["s0", "s1"], free_row=free_row)
    add("Unlike either inflation flavor, ridge **monotonically helps** up to "
        "a peak (S1 peaks at ridge=2.0, S0 plateaus around 0.1-1.0), then "
        "mildly declines (S1 ridge=5.0 < ridge=2.0). `ridge=1.0` is picked "
        "as one value near-optimal on both scenarios rather than tuning "
        "per-scenario.")
    add("")

    # ---- EnKF multiplicative inflation ----
    add("## EnKF: multiplicative inflation (for comparison against ETKF above)")
    add("")
    add("Same grid as ETKF's inflation sweep, run on EnKF, to test whether "
        "EnKF is robust because of its stochastic perturbed-observations "
        "update (which has built-in randomization against ensemble "
        "collapse that ETKF's deterministic square-root transform lacks) "
        "or just because `inflation=1.0`/`loc_radius=6.0` happen to suit "
        "it too. Never exercised before this study -- all prior EnKF runs "
        "used the fixed `inflation=1.0` default.")
    add("")
    enkf_infl_grid = [
        ("1.00", {"s0": f"{SWEEP_DIR}/s0_enkf_inflation1.json",
                  "s1": f"{SWEEP_DIR}/s1_enkf_inflation1.json"}),
        ("1.01", {"s0": f"{SWEEP_DIR}/s0_enkf_inflation1.01.json",
                  "s1": f"{SWEEP_DIR}/s1_enkf_inflation1.01.json"}),
        ("1.02", {"s0": f"{SWEEP_DIR}/s0_enkf_inflation1.02.json",
                  "s1": f"{SWEEP_DIR}/s1_enkf_inflation1.02.json"}),
        ("1.03", {"s0": f"{SWEEP_DIR}/s0_enkf_inflation1.03.json",
                  "s1": f"{SWEEP_DIR}/s1_enkf_inflation1.03.json"}),
        ("1.04", {"s0": f"{SWEEP_DIR}/s0_enkf_inflation1.04.json",
                  "s1": f"{SWEEP_DIR}/s1_enkf_inflation1.04.json"}),
        ("1.05", {"s0": f"{SWEEP_DIR}/s0_enkf_inflation1.05.json",
                  "s1": f"{SWEEP_DIR}/s1_enkf_inflation1.05.json"}),
        ("1.10", {"s0": f"{SWEEP_DIR}/s0_enkf_inflation1.1.json",
                  "s1": f"{SWEEP_DIR}/s1_enkf_inflation1.1.json"}),
    ]
    rows = table_rows(root, enkf_infl_grid, "test_s0")
    render_ev_table(add, rows, "inflation", ["s0", "s1"], free_row=free_row)
    add("")

    # ---- N=100 confirmation ----
    add("## N=100 confirmation")
    add("")
    add("Before committing to the expensive N=100 run, a quick N=10 check of "
        "whether ridge keeps helping past 1.0 found a **peak, not unbounded "
        "improvement** (S1 q EV: 0.253 default -> 0.332 @ ridge=1.0 -> "
        "**0.335 @ ridge=2.0, best** -> 0.323 @ ridge=5.0), and S0 (untested "
        "before) shows the same qualitative pattern. `ridge=1.0` was picked "
        "as one value near-optimal on both scenarios. Full N=100 ETKF run "
        "with `etkf_ridge=1.0`, against the canonical N=100 ETKF/EnKF "
        "baselines (note: the first interactive attempt at this silently "
        "OOM'd under this session's cgroup cap -- re-run via a real SLURM "
        "`sbatch` job, same fix as `run_qg_s1_full100.sbatch`'s prior note):")
    add("")
    n100_dir_s0 = "qg_repro_validation"
    n100_rows = []
    for label, s0_path, s1_path in [
        ("ETKF (default)", f"{n100_dir_s0}/etkf.json", f"{S1_DIR}/etkf.json"),
        ("EnKF", f"{n100_dir_s0}/enkf.json", f"{S1_DIR}/enkf.json"),
        ("ETKF + ridge=1.0", f"{n100_dir_s0}/etkf_ridge1.json", f"{S1_DIR}/etkf_ridge1.json"),
    ]:
        d_s0 = load_json(root / s0_path)
        d_s1 = load_json(root / s1_path)
        n100_rows.append({"label": label,
                           "s0": metrics(d_s0, "test_s0"),
                           "s1": metrics(d_s1, "test_s1")})
    add("| method | S0 psi EV | S0 q EV | S1 psi EV | S1 q EV |")
    add("|---|---|---|---|---|")
    psi_s0 = [r["s0"]["psi_ev"] if r["s0"] else None for r in n100_rows]
    q_s0 = [r["s0"]["q_ev"] if r["s0"] else None for r in n100_rows]
    psi_s1 = [r["s1"]["psi_ev"] if r["s1"] else None for r in n100_rows]
    q_s1 = [r["s1"]["q_ev"] if r["s1"] else None for r in n100_rows]
    m_psi_s0 = rank_marks(psi_s0)
    m_q_s0 = rank_marks(q_s0)
    m_psi_s1 = rank_marks(psi_s1)
    m_q_s1 = rank_marks(q_s1)
    for i, r in enumerate(n100_rows):
        add(f"| {r['label']} | "
            f"{mark(fmt(psi_s0[i]), m_psi_s0.get(i))} | "
            f"{mark(fmt(q_s0[i]), m_q_s0.get(i))} | "
            f"{mark(fmt(psi_s1[i]), m_psi_s1.get(i))} | "
            f"{mark(fmt(q_s1[i]), m_q_s1.get(i))} |")
    add("")
    add("(N=100, PV-q and psi full-field EV; best per column **bolded**, "
        "second-best *italicized*.)")
    add("")
    add("**Confirmed at full scale**: ETKF+ridge=1.0 beats EnKF outright on "
        "**both** fields on S1, and beats it on psi (ties within noise on "
        "q, -0.005) on S0. No trade-off on the unobserved PV layer either "
        "(S0 q layer2 improved over plain ETKF's ~0.34 to 0.41, matching "
        "EnKF's ~0.40, not sacrificed). `etkf_ridge=1.0` is a strictly "
        "better ETKF config than the implicit default on this reference "
        "case -- kept here as distinctly-tagged `etkf_ridge1.json` files, "
        "not yet promoted to overwrite the canonical `etkf.json` reference "
        "numbers in the main benchmark table pending that decision.")
    add("")

    add("## Synthesis")
    add("")
    add("- **Inflation-driven divergence is shared, not ETKF-specific**: "
        "ETKF and EnKF collapse in near-lockstep under multiplicative "
        "inflation, on both S0 and S1 -- e.g. S0 q EV at "
        "inflation={1.00,...,1.05}: ETKF {0.402,0.402,0.296,0.147,-40.4,"
        "-123.5} vs EnKF {0.463,0.435,0.313,0.156,-43.0,-124.3}, virtually "
        "the same curve, same catastrophic threshold. This **rules out** "
        "\"ETKF's deterministic transform makes it uniquely fragile to "
        "over-inflation\" -- both ensemble methods share this fragility "
        "equally, so it cannot explain the EnKF>ETKF gap.")
    add("- **Additive inflation**: also monotonically neutral-to-harmful "
        "(never helpful) on ETKF, but fails gracefully with no "
        "catastrophic divergence at any tested magnitude, unlike "
        "multiplicative inflation. Neither inflation flavor *helps* --  "
        "both are pure downside once past zero, confirming \"ETKF just "
        "needs the right inflation\" is not the fix.")
    add("- **`etkf_ridge` is the real, actionable lever**: unlike "
        "inflation, increasing the Kalman-gain transform-matrix "
        "regularization **monotonically helps** -- S1 q EV rises from "
        "0.253 (default, ~1e-4-equivalent) to **0.332 at ridge=1.0**, "
        "which *exceeds* both EnKF's own N=10 baseline here (0.279) and "
        "EnKF's N=100 headline number (0.331, from `qg_da_report.md`'s S1 "
        "table). This is a mechanism EnKF has no equivalent of (it has no "
        "deterministic transform-matrix inversion to regularize), so it's "
        "specific to ETKF's own weakness, not a general ensemble-DA fix.")
    add("- **Bottom line on the original open question -- CONFIRMED at "
        "N=100** (see the N=100 confirmation section above): the "
        "previously \"unexplained\" EnKF>ETKF gap on S1 was largely an "
        "artifact of running ETKF with an **under-regularized** "
        "transform-matrix inversion (the implicit ridge~1e-4 default), not "
        "a fundamental method limitation or an inflation-tuning problem. "
        "`etkf_ridge=1.0` at full N=100 beats EnKF outright on S1 (both "
        "fields) and ties/beats it on S0, with no trade-off on the "
        "unobserved PV layer. **Not yet decided**: whether to promote "
        "`etkf_ridge=1.0` to the default ETKF config in the main benchmark "
        "table (`qg_da_report.md`/PLAN.md's S0/S1 tables) -- kept as "
        "distinctly-tagged files pending that call.")
    add("- **Still open**: `N_ensemble` has never been varied (hardcoded 80 "
        "everywhere).")
    add("")
    add("Data: `reports/qg/outputs/qg_repro_validation_s1/etkf_n10_*.json` "
        "(original inflation/additive S1 sweep) + "
        f"`reports/qg/outputs/{SWEEP_DIR}/*.json` (consolidated S0 + EnKF + "
        "ridge sweep, N=10) + `reports/qg/outputs/qg_repro_validation"
        "{,_s1}/etkf_ridge1.json` (N=100 confirmation). Scratch drivers "
        "(not committed): `qg_da_s1_scratch.py`, "
        "`qg_da_sensitivity_sweep_scratch.py`, "
        "`qg_n100_ridge_confirm_scratch.py`.")
    add("")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
