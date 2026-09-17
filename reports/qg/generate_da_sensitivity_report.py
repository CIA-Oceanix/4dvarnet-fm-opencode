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
    q_l2 = s["metrics_per_field"]["q"]["layer2"]
    return {"q_ev": q["ev"], "q_ev_free": q["ev_free"],
            "psi_ev": psi["ev"], "psi_ev_free": psi["ev_free"],
            "q_ev_layer2": q_l2["ev"],
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


def render_metric_table(add, label_col, rows, cols):
    """rows: list of (label, metrics_dict_or_None). cols: list of (header, key)."""
    header = f"| {label_col} | " + " | ".join(h for h, _ in cols) + " |"
    add(header)
    add("|---|" + "---|" * len(cols))
    col_marks = []
    for _, key in cols:
        vals = [m[key] if m is not None else None for _, m in rows]
        col_marks.append(rank_marks(vals, higher_is_better=True))
    for i, (label, m) in enumerate(rows):
        cells = []
        for j, (_, key) in enumerate(cols):
            if m is None:
                cells.append("--")
            else:
                cells.append(mark(fmt(m[key]), col_marks[j].get(i)))
        add(f"| {label} | " + " | ".join(cells) + " |")
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
    add("Consolidates three related sensitivity studies into one place: "
        "ETKF/EnKF sensitivity to their own tunable hyperparameters "
        "(inflation, additive inflation, ridge regularization); a 4DVar "
        "(Strong/Weak) covariance-scale sensitivity negative result; and "
        "an ETKF/EnKF observation-density/configuration sensitivity study "
        "that went through a real mid-study correction after user pushback "
        "(both of its original headline findings were retracted once "
        "`loc_radius` was properly tuned per density -- see that section's "
        "own \"before correction\"/\"correction\" subsections for the full "
        "trail rather than only the corrected conclusion, per this "
        "project's convention of keeping the narrative honest). That "
        "correction then led to re-checking the reference density "
        "(cols=4) itself, finding it was *also* untuned -- `loc_radius=2.0`"
        "/`etkf_ridge=0.1` are now the project-wide default (2026-09-15/16, "
        "see the \"Reference density (cols=4) re-check\" subsection), and "
        "the density curve was extended to cols=1-64 (dedicated "
        "`qg_obs_density_report.md`). All three studies share the same "
        "underlying motivation: PV on the "
        "unobserved lower layer (q layer2) collapses under S1's model "
        "error, and each asks whether a different lever (inflation/ridge, "
        "4DVar covariance scale, or observation configuration/density) can "
        "fix or explain it.")
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
        "case -- **promoted to the actual default 2026-09-12** (was pending "
        "at first writing): `run_qg_baselines.py`'s/`sweep_qg_baselines.py`'s "
        "own default changed, and the canonical `etkf.json` reference "
        "numbers now hold these ridge=1.0 results (the old ~1e-4-floor "
        "numbers are archived as `etkf_ridge_default.json` in both "
        "directories, not deleted).")
    add("")

    # ---- 4DVar b_var_scale / q_var_scale sensitivity ----
    add("## 4DVar (Strong/Weak): covariance-scale sensitivity -- negative result")
    add("")
    add("Same motivation as the ETKF-ridge work above: both 4DVar variants "
        "collapse on the unobserved PV q layer2 under S1 (N=100: Strong "
        "-0.857, Weak -0.501, vs ETKF/EnKF staying positive -- see "
        "`qg_da_report.md`'s S1 table) -- checked whether a covariance-"
        "weighting sweep could fix or explain it the way `etkf_ridge` did "
        "for ETKF. Swept Strong-4DVar's `b_var_scale` (background-"
        "covariance whitening scale) and Weak-4DVar's `q_var_scale` "
        "(per-step model-error weight) on S1, **N=5** (4DVar is ~15-20x "
        "more expensive per window than ETKF/EnKF -- a single strong4dvar "
        "window took 364s in a timing probe vs ETKF's ~20-47s; N=5 was a "
        "cost/noise tradeoff, one order of magnitude below the ETKF sweeps' "
        "N=10). Each window already gets its own `QG4DVar`/LBFGS instance "
        "inside `run()`'s loop.")
    add("")
    fourdvar_dir = "qg_4dvar_sensitivity_sweep"
    strong_grid = [
        ("0.3", f"{fourdvar_dir}/s1_strong4dvar_bvar0.3.json"),
        ("1.0 (default)", f"{fourdvar_dir}/s1_strong4dvar_bvar1.json"),
        ("3.0", f"{fourdvar_dir}/s1_strong4dvar_bvar3.json"),
    ]
    strong_rows = [(label, metrics(load_json(root / rel), "test_s1"))
                   for label, rel in strong_grid]
    render_metric_table(add, "Strong-4DVar `b_var_scale`", strong_rows,
                         [("S1 q EV", "q_ev"), ("S1 psi EV", "psi_ev")])
    weak_grid = [
        ("0.1 (default)", f"{fourdvar_dir}/s1_weak4dvar_qvar0.1.json"),
        ("0.3", f"{fourdvar_dir}/s1_weak4dvar_qvar0.3.json"),
        ("1.0", f"{fourdvar_dir}/s1_weak4dvar_qvar1.json"),
        ("3.0", f"{fourdvar_dir}/s1_weak4dvar_qvar3.json"),
    ]
    weak_rows = [(label, metrics(load_json(root / rel), "test_s1"))
                 for label, rel in weak_grid]
    render_metric_table(add, "Weak-4DVar `q_var_scale`", weak_rows,
                         [("S1 q EV", "q_ev"), ("S1 psi EV", "psi_ev")])
    add("`b_var_scale` is essentially flat across an order of magnitude "
        "(mild decline, not a peak like ridge showed). `q_var_scale` is "
        "monotonically *worse* the higher it's pushed above the existing "
        "default 0.1 -- giving the per-step model-error controls more "
        "freedom to correct actively **hurts** rather than helping, the "
        "opposite of the initial hypothesis (that S1's real model error "
        "would need *more* absorption capacity). Both methods' existing "
        "defaults were already at or near the best point in the explored "
        "direction; **no config change made** (unlike ETKF). Caveats: N=5 "
        "is noisy (absolute EVs differ from the N=100 canonical numbers, "
        "though the ranking direction is consistent), and only "
        "`q_var_scale` values *above* 0.1 were tested -- a small chance a "
        "lower value (0.01-0.03) does better still, not checked given the "
        "cost. Consistent interpretation: the 4DVar q-layer2 collapse "
        "looks like a **structural** limitation (a single optimized "
        "trajectory has no ensemble spread to exploit on the unobserved "
        "layer), not a fixable covariance-tuning gap the way ETKF's "
        "transform-regularization gap was.")
    add("")
    add("Data: `reports/qg/outputs/qg_4dvar_sensitivity_sweep/*.json` "
        "(N=5, 7 files). Scratch driver (not committed): "
        "`qg_4dvar_sensitivity_sweep_scratch.py`.")
    add("")

    # ---- Obs-density sensitivity: psi2 obs, cols_sampling="random",
    # and the loc_radius correction ----
    add("## ETKF/EnKF obs-density sensitivity: psi2 observations, "
        "`cols_sampling=\"random\"`, and a `loc_radius` correction")
    add("")
    add("Motivated by the same q-layer2/unobserved-deep-layer weakness the "
        "ETKF-ridge and 4DVar sensitivity work above kept running into: "
        "does giving the DA method *any* direct information about the "
        "lower layer help, versus just adding more of the same "
        "(upper-layer-only) observation? Two new opt-in `QGConfig` "
        "capabilities were added to test this: `psi2_points_per_day` "
        "(independent lower-layer random-point observations, never "
        "directly observed before this work) and `cols_sampling=\"random\"` "
        "(upper-layer column observations as random `(t, x)` draws across "
        "the whole day, removing the default `\"sequential\"` sampler's "
        "`steps_per_day` ceiling on `cols_per_day` -- which turned out to "
        "hang forever above that ceiling, a pre-existing bug now fixed "
        "with a clear `ValueError`). All sweeps below: N=10, ETKF default "
        "`etkf_ridge=1.0`, `N_ensemble=80`.")
    add("")
    add("### Initial N=10 sweep (before correction)")
    add("")
    obsd_dir = "qg_obs_density_sweep"
    density_grid_s0 = [
        ("baseline (cols=4, sequential)", f"{obsd_dir}/s0_etkf_baseline_c4.json"),
        ("cols=8 (sequential)", f"{obsd_dir}/s0_etkf_cols8.json"),
        ("cols=4, random (sanity check)", f"{obsd_dir}/s0_etkf_colpts4.json"),
        ("cols=16, random", f"{obsd_dir}/s0_etkf_colpts16.json"),
        ("psi1(4)+psi2(10)", f"{obsd_dir}/s0_etkf_c4_psi2pts10.json"),
    ]
    density_grid_s1 = [
        ("baseline (cols=4, sequential)", f"{obsd_dir}/s1_etkf_baseline_c4.json"),
        ("cols=8 (sequential)", f"{obsd_dir}/s1_etkf_cols8.json"),
        ("cols=4, random (sanity check)", f"{obsd_dir}/s1_etkf_colpts4.json"),
        ("cols=16, random", f"{obsd_dir}/s1_etkf_colpts16.json"),
        ("psi1(4)+psi2(10)", f"{obsd_dir}/s1_etkf_c4_psi2pts10.json"),
    ]
    combined_rows = []
    for (label, rel_s0), (_, rel_s1) in zip(density_grid_s0, density_grid_s1):
        m0 = metrics(load_json(root / rel_s0), "test_s0")
        m1 = metrics(load_json(root / rel_s1), "test_s1")
        combined_rows.append((label, m0, m1))
    add("| config (`loc_radius=6.0` throughout) | S0 q full | S0 q layer2 | "
        "S0 psi full | S1 q full | S1 q layer2 | S1 psi full |")
    add("|---|---|---|---|---|---|---|")
    cols6 = [m["q_ev"] if m else None for _, m, _ in combined_rows]
    cols6b = [m["q_ev_layer2"] if m else None for _, m, _ in combined_rows]
    cols6c = [m["psi_ev"] if m else None for _, m, _ in combined_rows]
    cols6d = [m["q_ev"] if m else None for _, _, m in combined_rows]
    cols6e = [m["q_ev_layer2"] if m else None for _, _, m in combined_rows]
    cols6f = [m["psi_ev"] if m else None for _, _, m in combined_rows]
    marks6 = [rank_marks(c) for c in (cols6, cols6b, cols6c, cols6d, cols6e, cols6f)]
    for i, (label, m0, m1) in enumerate(combined_rows):
        vals = [cols6[i], cols6b[i], cols6c[i], cols6d[i], cols6e[i], cols6f[i]]
        cells = [mark(fmt(v), marks6[j].get(i)) for j, v in enumerate(vals)]
        add(f"| {label} | " + " | ".join(cells) + " |")
    add("")
    add("**Initially interpreted as two headline findings -- both later "
        "retracted, see the correction below**: (1) \"more upper-layer "
        "density destabilizes S1\" (cols=8 and especially cols=16-random "
        "degrade S1 psi, the latter catastrophically to -1.19); (2) "
        "\"psi2 observations are uniquely valuable\" (psi1+psi2 was the "
        "best or tied-best config on both scenarios). An EnKF cross-check "
        "at the same three configs found the identical graded "
        "destabilization pattern on S1, if anything more severe at the "
        "extreme (cols=16-random: EnKF psi -4.95 vs ETKF's -1.19) -- "
        "confirming the collapse (whatever its cause) is shared across "
        "both ensemble methods, not ETKF-specific.")
    add("")
    add("### Correction, step 1: inflation ruled out")
    add("")
    add("The user pushed back on finding (1) -- correctly -- and asked "
        "whether a too-large/too-weak inflation could explain it, and "
        "requested a `cols=64` run (expecting *more* data to help, not "
        "hurt). Sweeping inflation at `cols=16` (random, S1, `loc_radius="
        "6.0` unchanged):")
    add("")
    infl_grid = [
        ("0.90", f"{obsd_dir}/s1_etkf_colpts16_infl0.9.json"),
        ("0.95", f"{obsd_dir}/s1_etkf_colpts16_infl0.95.json"),
        ("1.00 (default)", f"{obsd_dir}/s1_etkf_colpts16_infl1.json"),
        ("1.02", f"{obsd_dir}/s1_etkf_colpts16_infl1.02.json"),
        ("1.05", f"{obsd_dir}/s1_etkf_colpts16_infl1.05.json"),
    ]
    infl_rows = [(label, metrics(load_json(root / rel), "test_s1"))
                 for label, rel in infl_grid]
    render_metric_table(add, "inflation", infl_rows, [("S1 q full EV", "q_ev")])
    add("`inflation=1.0` was already the tested optimum -- ruling out "
        "inflation mismatch as the explanation.")
    add("")
    add("### Correction, step 2: `cols=64` degrades S0 too")
    add("")
    m_s0_c64 = metrics(load_json(root / f"{obsd_dir}/s0_etkf_colpts64.json"), "test_s0")
    m_s1_c64 = metrics(load_json(root / f"{obsd_dir}/s1_etkf_colpts64.json"), "test_s1")
    add(f"At `cols=64` (random, `loc_radius=6.0` still unchanged): S1 got "
        f"worse still (q full {fmt(m_s1_c64['q_ev']) if m_s1_c64 else '--'}, "
        f"psi full {fmt(m_s1_c64['psi_ev']) if m_s1_c64 else '--'}), as "
        f"expected if this were pure model-error reinforcement -- but "
        f"critically, **S0 (no model error) also degraded** (psi full: "
        f"{fmt(m_s0_c64['psi_ev']) if m_s0_c64 else '--'}, down from the "
        f"baseline's ~0.907). A pure-model-error story predicts *no* S0 "
        f"effect -- this was the tell that something else was going on.")
    add("")
    add("### Correction, step 3: `loc_radius` sweep recovers and exceeds baseline")
    add("")
    add("Localization exists specifically to counter ensemble sampling "
        "error given a *fixed, small* ensemble (`N_ensemble=80` "
        "throughout, deliberately not scaled up, matching the operational "
        "constraint the user named: ensemble size stays small in practice). "
        "Higher observation density means more *simultaneous* columns per "
        "assimilation step, so a `loc_radius` tuned for the sparse 4-8/day "
        "regime becomes under-resourced at higher density. Shrinking it "
        "(N_ensemble fixed at 80) fully recovers and then exceeds the "
        "original baseline, at both `cols=64` and `cols=16`:")
    add("")
    loc64_grid_s0 = [
        ("6.0 (unchanged)", f"{obsd_dir}/s0_etkf_colpts64.json"),
        ("4.0", f"{obsd_dir}/s0_etkf_colpts64_loc4.json"),
        ("3.0", f"{obsd_dir}/s0_etkf_colpts64_loc3.json"),
        ("2.0", f"{obsd_dir}/s0_etkf_colpts64_loc2.json"),
        ("1.0", f"{obsd_dir}/s0_etkf_colpts64_loc1.json"),
    ]
    loc64_grid_s1 = [
        ("6.0 (unchanged)", f"{obsd_dir}/s1_etkf_colpts64.json"),
        ("4.0", f"{obsd_dir}/s1_etkf_colpts64_loc4.json"),
        ("3.0", f"{obsd_dir}/s1_etkf_colpts64_loc3.json"),
        ("2.0", f"{obsd_dir}/s1_etkf_colpts64_loc2.json"),
        ("1.0", f"{obsd_dir}/s1_etkf_colpts64_loc1.json"),
    ]
    loc64_rows = []
    for (label, rel_s0), (_, rel_s1) in zip(loc64_grid_s0, loc64_grid_s1):
        m0 = metrics(load_json(root / rel_s0), "test_s0")
        m1 = metrics(load_json(root / rel_s1), "test_s1")
        loc64_rows.append((label, {"s0_psi": m0["psi_ev"] if m0 else None,
                                    "s1_psi": m1["psi_ev"] if m1 else None,
                                    "s1_q": m1["q_ev"] if m1 else None}))
    render_metric_table(add, "cols=64, loc_radius", loc64_rows,
                         [("S0 psi EV", "s0_psi"), ("S1 psi EV", "s1_psi"),
                          ("S1 q full EV", "s1_q")])
    loc16_grid_s0 = [
        ("6.0 (unchanged)", f"{obsd_dir}/s0_etkf_colpts16.json"),
        ("4.0", f"{obsd_dir}/s0_etkf_colpts16_loc4.json"),
        ("3.0", f"{obsd_dir}/s0_etkf_colpts16_loc3.json"),
        ("2.0", f"{obsd_dir}/s0_etkf_colpts16_loc2.json"),
    ]
    loc16_grid_s1 = [
        ("6.0 (unchanged)", f"{obsd_dir}/s1_etkf_colpts16.json"),
        ("4.0", f"{obsd_dir}/s1_etkf_colpts16_loc4.json"),
        ("3.0", f"{obsd_dir}/s1_etkf_colpts16_loc3.json"),
        ("2.0", f"{obsd_dir}/s1_etkf_colpts16_loc2.json"),
    ]
    loc16_rows = []
    for (label, rel_s0), (_, rel_s1) in zip(loc16_grid_s0, loc16_grid_s1):
        m0 = metrics(load_json(root / rel_s0), "test_s0")
        m1 = metrics(load_json(root / rel_s1), "test_s1")
        loc16_rows.append((label, {"s0_psi": m0["psi_ev"] if m0 else None,
                                    "s1_psi": m1["psi_ev"] if m1 else None,
                                    "s1_q": m1["q_ev"] if m1 else None}))
    render_metric_table(add, "cols=16, loc_radius", loc16_rows,
                         [("S0 psi EV", "s0_psi"), ("S1 psi EV", "s1_psi"),
                          ("S1 q full EV", "s1_q")])
    add("At the best `loc_radius` (2.0 for cols=64, 3.0-4.0 for cols=16), "
        "both configs **beat** the original `cols=4`/`loc=6.0` baseline "
        "(S1 psi 0.829 baseline vs 0.96+ tuned). **Root cause**: classic "
        "ensemble sampling-error/rank-deficiency, not a physical/"
        "model-error effect -- `loc_radius` must be swept per obs-density "
        "config, not held at one project-wide default. **Finding (1) above "
        "is retracted**: properly localized, more upper-layer density is "
        "unambiguously better on both S0 and S1, as conventional DA wisdom "
        "predicts. (The other standard fix for the same problem, not "
        "tried here: serial/sequential per-observation processing instead "
        "of one large combined observation vector.)")
    add("")
    add("### Correction, step 4: psi1-only vs. psi1+psi2 at a fair `loc_radius`")
    add("")
    loc_psi2_grid_s1 = [
        ("6.0 (unchanged)", f"{obsd_dir}/s1_etkf_c4_psi2pts10.json"),
        ("5.0", f"{obsd_dir}/s1_etkf_c4_psi2pts10_loc5.json"),
        ("4.0", f"{obsd_dir}/s1_etkf_c4_psi2pts10_loc4.json"),
        ("3.0", f"{obsd_dir}/s1_etkf_c4_psi2pts10_loc3.json"),
        ("2.0", f"{obsd_dir}/s1_etkf_c4_psi2pts10_loc2.json"),
    ]
    psi2loc_rows = [(label, metrics(load_json(root / rel), "test_s1"))
                    for label, rel in loc_psi2_grid_s1]
    render_metric_table(add, "psi1(4)+psi2(10), loc_radius", psi2loc_rows,
                         [("S1 q full EV", "q_ev"), ("S1 psi EV", "psi_ev")])
    add("The psi1+psi2 combined config was only ever tested at "
        "`loc_radius=6.0`; it turns out to already be near its own "
        "optimum there (its total density is modest). But once "
        "**psi1-only** configs are given their own fair, density-matched "
        "`loc_radius` (from step 3 above), they clearly outperform the "
        "psi1+psi2 mix at its much lower density:")
    add("")
    final_cmp_rows = [
        ("psi1(4)+psi2(10), loc=5 (its own optimum)",
         metrics(load_json(root / f"{obsd_dir}/s1_etkf_c4_psi2pts10_loc5.json"), "test_s1")),
        ("psi1-only, cols=16, loc=3 (tuned)",
         metrics(load_json(root / f"{obsd_dir}/s1_etkf_colpts16_loc3.json"), "test_s1")),
        ("psi1-only, cols=64, loc=2 (tuned)",
         metrics(load_json(root / f"{obsd_dir}/s1_etkf_colpts64_loc2.json"), "test_s1")),
    ]
    render_metric_table(add, "config (S1)", final_cmp_rows,
                         [("q full EV", "q_ev"), ("psi EV", "psi_ev")])
    add("**Finding (2) above is therefore also not supported**: once the "
        "comparison is fair, this only shows \"more well-localized data "
        "beats less well-localized-but-mixed data,\" not \"psi1 alone "
        "beats psi1+psi2 at equal density.\" **Genuinely open question, "
        "not yet answered**: does psi2 information add anything *at "
        "matched total density* against a properly-tuned pure-psi1 config "
        "(e.g. cols~54+psi2=10 vs cols=64, both individually "
        "`loc_radius`-tuned)? Not yet tested.")
    add("")
    add("### N=100 confirmation (2026-09-14)")
    add("")
    add("Both loc_radius-tuned high-density configs (cols=16 loc=2.0, "
        "cols=64 loc=1.0), for **both** ETKF and EnKF, run at full N=100 -- "
        "closing the \"N=100 confirmation is the natural next step\" "
        "caveat from the correction above. EnKF's own `loc_radius` optimum "
        "was checked separately at N=10 first (`qg_enkf_loc_check_scratch.py`) "
        "rather than assumed to match ETKF's, since the untuned (loc=6.0) "
        "EnKF cross-check collapsed *more* severely than ETKF at the same "
        "density -- it turned out to match ETKF's exactly at both "
        "densities.")
    add("")
    obsd_n100_dir = "qg_obs_density_sweep_n100"
    n100_density_rows = []
    for method in ("etkf", "enkf"):
        base_s0 = metrics(load_json(root / f"qg_repro_validation/{method}.json"), "test_s0")
        base_s1 = metrics(load_json(root / f"qg_repro_validation_s1/{method}.json"), "test_s1")
        method_label = "ETKF" if method == "etkf" else "EnKF"
        n100_density_rows.append((f"{method_label} baseline (cols=4, loc=6.0)",
                                   {"s0_psi": base_s0["psi_ev"] if base_s0 else None,
                                    "s0_q": base_s0["q_ev"] if base_s0 else None,
                                    "s0_ql2": base_s0["q_ev_layer2"] if base_s0 else None,
                                    "s1_psi": base_s1["psi_ev"] if base_s1 else None,
                                    "s1_q": base_s1["q_ev"] if base_s1 else None,
                                    "s1_ql2": base_s1["q_ev_layer2"] if base_s1 else None}))
        for density, loc in (("colpts16", "2.0"), ("colpts64", "1.0")):
            m0 = metrics(load_json(root / f"{obsd_n100_dir}/s0_{method}_{density}_n100.json"), "test_s0")
            m1 = metrics(load_json(root / f"{obsd_n100_dir}/s1_{method}_{density}_n100.json"), "test_s1")
            label = f"{method_label} {density.replace('colpts', 'cols=')} (loc={loc})"
            n100_density_rows.append((label,
                                       {"s0_psi": m0["psi_ev"] if m0 else None,
                                        "s0_q": m0["q_ev"] if m0 else None,
                                        "s0_ql2": m0["q_ev_layer2"] if m0 else None,
                                        "s1_psi": m1["psi_ev"] if m1 else None,
                                        "s1_q": m1["q_ev"] if m1 else None,
                                        "s1_ql2": m1["q_ev_layer2"] if m1 else None}))
    render_metric_table(
        add, "config (N=100)", n100_density_rows,
        [("S0 psi", "s0_psi"), ("S0 q", "s0_q"), ("S0 q-layer2", "s0_ql2"),
         ("S1 psi", "s1_psi"), ("S1 q", "s1_q"), ("S1 q-layer2", "s1_ql2")])
    add("**Fully confirmed at N=100**: both tuned high-density configs "
        "decisively beat the canonical cols=4/loc=6.0 baseline, for both "
        "methods, on every field and both scenarios -- including the "
        "previously-collapsing unobserved deep layer (q layer2), which "
        "roughly **doubles** at cols=64 despite psi1 columns never "
        "directly observing it (S1: ETKF 0.266->0.480, EnKF 0.221->0.489). "
        "cols=64 beats cols=16 throughout, consistent with \"more density, "
        "properly localized, is unambiguously better\" holding at full "
        "scale too, not just N=10.")
    add("")
    add("**One new wrinkle**: at these higher densities, **EnKF edges out "
        "ETKF+ridge=1.0** on both fields (cols=64 S1: EnKF q=0.600 vs ETKF "
        "q=0.585; cols=16 S1: EnKF q=0.539 vs ETKF q=0.526) -- a partial "
        "reversal of ETKF's advantage at the cols=4 baseline (where "
        "`etkf_ridge=1.0` was specifically tuned and promoted). The margin "
        "is modest (~0.01-0.02) so not necessarily decisive, but it means "
        "\"ETKF+ridge=1.0 beats EnKF\" is a cols=4-specific finding, not a "
        "universal one -- whether ETKF's own ridge/inflation should be "
        "re-tuned at higher density, rather than reusing the cols=4-tuned "
        "value, is untested.")
    add("")
    add("**Resolved (2026-09-15/16) -- see the reference-density re-check "
        "below**: rather than picking a high-density config as a special "
        "override, the reference density (cols=4) itself was re-checked "
        "and its own `loc_radius`/`etkf_ridge` promoted instead, so the "
        "\"benchmark-design decision\" this callout flagged didn't end up "
        "needing to be made -- the actual fix applies uniformly.")
    add("")
    add(f"Data: `reports/qg/outputs/{obsd_n100_dir}/*.json` (N=100, 8 "
        "files: ETKF/EnKF x cols={16,64} x S0/S1). Scratch drivers (not "
        "committed): `qg_obs_density_n100_scratch.py`, "
        "`qg_enkf_loc_check_scratch.py` (the EnKF loc_radius check). "
        "Batch: `batch/run_qg_obs_density_n100.sbatch` (jobs 53537 [ETKF, "
        "1h50m], 53538 [EnKF, 2h36m]).")
    add("")

    add("### Reference density (cols=4) re-check, ridge re-tune, and "
        "final promotion (2026-09-15/16)")
    add("")
    add("The N=100 confirmation above raised an obvious follow-up: is "
        "`loc_radius=6.0` -- the project's own reference-case default, "
        "never itself swept -- actually tuned for cols=4? An N=10 sweep "
        "({1,2,3,4,6}, both methods) found **no**: `loc_radius=6.0` was "
        "the *worst* point in the grid for both ETKF and EnKF, and "
        "`loc_radius=2.0` (the same value that won at every other tested "
        "density) won again.")
    add("")
    add("Confirmed at N=100 (cols=4, both methods):")
    add("")
    cols4_rows = [
        ("ETKF, loc=6.0 (old default)",
         metrics(load_json(root / "qg_repro_validation/etkf_loc6_default.json"), "test_s0"),
         metrics(load_json(root / "qg_repro_validation_s1/etkf_loc6_default.json"), "test_s1")),
        ("ETKF, loc=2.0 (new)",
         metrics(load_json(root / "qg_obs_density_sweep_n100/s0_etkf_loc2_ridge0p1_n100.json"), "test_s0"),
         metrics(load_json(root / "qg_obs_density_sweep_n100/s1_etkf_loc2_ridge0p1_n100.json"), "test_s1")),
        ("EnKF, loc=6.0 (old default)",
         metrics(load_json(root / "qg_repro_validation/enkf_loc6_default.json"), "test_s0"),
         metrics(load_json(root / "qg_repro_validation_s1/enkf_loc6_default.json"), "test_s1")),
        ("EnKF, loc=2.0 (new)",
         metrics(load_json(root / "qg_obs_density_sweep_n100/s0_enkf_colpts4_n100.json"), "test_s0"),
         metrics(load_json(root / "qg_obs_density_sweep_n100/s1_enkf_colpts4_n100.json"), "test_s1")),
    ]
    add("| config | S0 psi | S0 q | S1 psi | S1 q |")
    add("|---|---|---|---|---|")
    for label, m0, m1 in cols4_rows:
        add(f"| {label} | {fmt(m0['psi_ev']) if m0 else '--'} | "
            f"{fmt(m0['q_ev']) if m0 else '--'} | "
            f"{fmt(m1['psi_ev']) if m1 else '--'} | "
            f"{fmt(m1['q_ev']) if m1 else '--'} |")
    add("")
    add("With `loc_radius` now 2.0, `etkf_ridge=1.0` (tuned specifically "
        "for the old `loc_radius=6.0`) needed re-checking too -- a fresh "
        "N=10 sweep ({0.0,0.1,0.5,1.0,2.0,3.0,5.0}) found its benefit had "
        "**inverted**: psi is flat across the whole range (noise-level "
        "differences), while q and q-layer2 decline *monotonically* from "
        "`ridge=0.0` (the implicit floor) upward -- confirmed at N=100 "
        "down to `ridge=0.25`, still declining, so `ridge=0.1` was taken "
        "as the new value (near the floor, not exactly at it, as a small "
        "safety margin). A matching ridge re-check at `loc_radius=1.0` "
        "(cols=64) found `ridge=0.1` works well there too -- one ridge "
        "value transfers across both `loc_radius` regimes, so no "
        "density-specific ridge tuning was needed. `inflation=1.0` was "
        "re-checked at the new config (both methods) and remains clearly "
        "optimal -- unaffected by the `loc_radius`/`ridge` change.")
    add("")
    add("**Promoted to the new project-wide reference default "
        "(2026-09-15/16)**: `loc_radius=2.0` for cols<=32, "
        "`loc_radius=1.0` for cols=64 (both methods); `etkf_ridge=0.1` "
        "(was 1.0) for ETKF at every density; `inflation=1.0` unchanged. "
        "The canonical `qg_repro_validation{,_s1}/{etkf,enkf}.json` now "
        "hold these numbers (old `loc_radius=6.0` results archived as "
        "`{etkf,enkf}_loc6_default.json` in both directories, not "
        "deleted). The density curve was also extended to cols={8,32} "
        "(previously only 1,2,4,16,64 had been tested) and cols={1,2,16,"
        "64}'s ETKF numbers re-run at `ridge=0.1` for full internal "
        "consistency -- see the dedicated "
        "`reports/qg/outputs/qg_obs_density_report.md` "
        "(`reports/qg/generate_qg_obs_density_report.py`) for the "
        "complete cols=1-64 curve. Notably, ETKF and EnKF are now nearly "
        "indistinguishable at every density once both are properly "
        "localized -- both the original EnKF>ETKF gap (2026-09-11) and "
        "the later ETKF>EnKF gap (after `etkf_ridge=1.0`'s promotion) "
        "were largely artifacts of the untuned `loc_radius=6.0`.")
    add("")
    add("Data: `reports/qg/outputs/qg_obs_density_sweep/s{0,1}_{etkf,enkf}"
        "_colpts4_loc*.json` (cols=4 loc_radius screen, N=10) + "
        "`s{0,1}_etkf_loc{1,2}_ridge*.json` (ridge re-checks at both "
        "`loc_radius` values, N=10) + `s{0,1}_{etkf,enkf}_loc2_infl*.json` "
        "(inflation re-check, N=10) + `s{0,1}_{etkf,enkf}_colpts{8,32}"
        "_loc*.json` (new density points, N=10) + "
        "`reports/qg/outputs/qg_obs_density_sweep_n100/*.json` (all N=100 "
        "confirmations). Scratch drivers (not committed): "
        "`qg_cols4_loc_scratch.py`, `qg_ridge_at_loc2_scratch.py`, "
        "`qg_ridge_at_loc1_cols64_scratch.py`, "
        "`qg_inflation_at_loc2_scratch.py`, `qg_cols8_loc_scratch.py`, "
        "`qg_cols32_loc_scratch.py`, `qg_density_n100_scratch.py` "
        "(generic N=100 driver).")
    add("")
    add("Data: `reports/qg/outputs/qg_obs_density_sweep/*.json` (N=10, "
        "231 files as of 2026-09-16: the original ETKF/EnKF density "
        "sweep, the cols=4/random sanity check, the EnKF cross-check, the "
        "ETKF inflation and `loc_radius` sweeps at cols=16/64, the "
        "psi1+psi2 `loc_radius` sweep, EnKF's own `loc_radius` checks "
        "(cols=16/64 and, later, cols=1/2/4/8/32), the cols=4 reference-"
        "density `loc_radius` re-check, the ridge re-checks at both "
        "`loc_radius` values, the inflation re-check, and the new "
        "cols=8/32 density screens -- see the \"Reference density "
        "(cols=4) re-check\" subsection above for the later additions). "
        "Tests: `tests/test_qg_psi2_points.py` "
        "(15), `tests/test_qg_cols_sampling.py` (10) -- mechanical "
        "correctness only (shapes, determinism, no-hang, H-function/"
        "localization correctness), no scientific claim baked in. Scratch "
        "drivers (not committed): `qg_obs_density_sweep_scratch.py`, "
        "`qg_enkf_loc_check_scratch.py`.")
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
        "unobserved PV layer. **Promoted to the default ETKF config** "
        "2026-09-12 (the canonical `etkf.json` reference numbers in "
        "`qg_da_report.md`/PLAN.md's S0/S1 tables now hold these ridge=1.0 "
        "results; the old ~1e-4-floor numbers are archived, not deleted).")
    add("- **4DVar's covariance-scale sweep found no equivalent lever**: "
        "unlike ETKF's ridge, neither Strong-4DVar's `b_var_scale` nor "
        "Weak-4DVar's `q_var_scale` improves on the existing default in "
        "the tested range -- the q-layer2 collapse under S1 looks "
        "structural (no ensemble spread to exploit on the unobserved "
        "layer), not a covariance-tuning gap. No config change made.")
    add("- **Obs-density/configuration sensitivity: a `loc_radius`-tuning "
        "lesson, not a new physical effect**: an initial N=10 sweep looked "
        "like it found \"more upper-layer density destabilizes S1\" and "
        "\"psi2 (lower-layer) observations are uniquely valuable\" -- "
        "**both retracted** after the user's pushback led to an inflation "
        "check, a `cols=64` run, and a `loc_radius` sweep. The real "
        "story: `loc_radius=6.0` was tuned for the sparse 4-8/day regime "
        "and becomes an ensemble-conditioning bottleneck at higher "
        "density with a fixed small ensemble (N=80) -- shrinking it fully "
        "recovers and then exceeds the original baseline at both cols=16 "
        "and cols=64, on both S0 and S1. **Fully confirmed at N=100** for "
        "both ETKF and EnKF (EnKF's own `loc_radius` optimum checked "
        "separately, matched ETKF's exactly): both tuned high-density "
        "configs decisively beat the cols=4 baseline on every field, "
        "including the previously-collapsing q layer2 (roughly doubles at "
        "cols=64). One new wrinkle: EnKF edges out ETKF+ridge=1.0 at these "
        "higher densities (modest margin, ~0.01-0.02) -- \"ETKF+ridge=1.0 "
        "beats EnKF\" turned out to be a cols=4-specific artifact of "
        "`loc_radius=6.0` also being untuned at the reference density "
        "itself. **Resolved (2026-09-15/16)**: re-checked `loc_radius` at "
        "cols=4 -- also found 6.0 was the worst point in the grid there "
        "too -- and re-tuned `etkf_ridge` (1.0->0.1) at the new "
        "`loc_radius`. Both promoted to the new project-wide default "
        "(see the \"Reference density (cols=4) re-check\" subsection "
        "above); the density curve was extended to cols={8,32} and every "
        "existing point re-confirmed at the final config. Once properly "
        "localized, ETKF and EnKF are nearly indistinguishable at every "
        "density from 1-64 cols/day -- see the dedicated "
        "`qg_obs_density_report.md`. Whether psi2 information adds value "
        "*at matched total density* against a properly-tuned pure-psi1 "
        "config remains genuinely open (untouched by this promotion).")
    add("- **Still open**: `N_ensemble` has never been varied (hardcoded 80 "
        "everywhere) in any of the three studies above.")
    add("")
    add("Data: `reports/qg/outputs/qg_repro_validation_s1/etkf_n10_*.json` "
        "(original inflation/additive S1 sweep) + "
        f"`reports/qg/outputs/{SWEEP_DIR}/*.json` (consolidated S0 + EnKF + "
        "ridge sweep, N=10) + `reports/qg/outputs/qg_repro_validation"
        "{,_s1}/etkf_ridge1.json` (N=100 confirmation) + "
        "`reports/qg/outputs/qg_4dvar_sensitivity_sweep/*.json` (4DVar "
        "covariance-scale sweep, N=5) + "
        "`reports/qg/outputs/qg_obs_density_sweep/*.json` (obs-density/"
        "configuration sweep with correction, N=10) + "
        "`reports/qg/outputs/qg_obs_density_sweep_n100/*.json` (N=100 "
        "confirmation). Scratch drivers "
        "(not committed): `qg_da_s1_scratch.py`, "
        "`qg_da_sensitivity_sweep_scratch.py`, "
        "`qg_n100_ridge_confirm_scratch.py`, "
        "`qg_4dvar_sensitivity_sweep_scratch.py`, "
        "`qg_obs_density_sweep_scratch.py`, "
        "`qg_obs_density_n100_scratch.py`, `qg_enkf_loc_check_scratch.py`, "
        "`qg_cols4_loc_scratch.py`, `qg_ridge_at_loc2_scratch.py`, "
        "`qg_ridge_at_loc1_cols64_scratch.py`, "
        "`qg_inflation_at_loc2_scratch.py`, `qg_cols8_loc_scratch.py`, "
        "`qg_cols32_loc_scratch.py`, `qg_density_n100_scratch.py`. See "
        "also the dedicated `reports/qg/outputs/qg_obs_density_report.md` "
        "for the complete cols=1-64 curve at the final promoted config.")
    add("")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
