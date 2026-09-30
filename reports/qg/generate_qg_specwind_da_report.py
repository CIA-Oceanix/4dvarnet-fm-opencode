"""Report: QG gyrostat case study -- DA baselines (S0 + realistic S1) on the spectral-wind test sets.

Layout follows the legacy QG DA report (`reports/qg/outputs/qg_da_report.md`):
reference case, per-scenario DA tables (best bold, second-best italic, among
the DA methods; the free forecast is a reference row), configuration
synthesis with the S1 error budget, observation density, forced vs coupled,
factor strata, and best/median/worst reconstruction examples with DA-cycle
animations (`generate_qg_specwind_da_figs.py`).

Reads the per-shard `*_meta.json` written by `evaluation.run_qg_specwind_da`
(per-window EVs computed with each window's own parameters) under
`<root>/<spec>/test/<config>/`, the dataset manifests for the window factors,
and the S1 attribution runs of `evaluation.qg_specwind_s1_sweep`.

    python reports/qg/generate_qg_specwind_da_report.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

FIELDS = ("psi1", "psi2", "q1", "q2")
LABEL = {"psi1": "ψ₁", "psi2": "ψ₂", "q1": "q₁", "q2": "q₂", "score": "score"}
SPECS = {"qg_specwind_gyrostat_v1": "forced", "qg_coupled_gyrostat_v1": "coupled"}
METHOD_NAME = {"etkf": "ETKF", "enkf": "EnKF", "enks": "EnKS (ETKF smoother)"}
METHODS = ("etkf", "enkf", "enks")
S0 = "S0"
ETKF_MODE = "ensrf"
S1_TUNED_R = 6.0
ENKS_LAG = 12
ENKS_TAPER = None


def load_runs(root: str) -> list[dict]:
    groups = defaultdict(list)
    for f in glob.glob(os.path.join(root, "*", "test", "*", "shard_*_meta.json")):
        groups[os.path.dirname(f)].append(f)
    runs = []
    for d, files in sorted(groups.items()):
        metas = [json.load(open(f)) for f in sorted(files)]
        if any("per_window" not in m for m in metas):
            continue
        m0 = metas[0]
        n_shards = int(os.path.basename(files[0]).split("_of_")[1].split("_")[0])
        pw = sorted((w for m in metas for w in m["per_window"]), key=lambda w: w["index"])
        runs.append({"dir": d, "spec": m0["spec"], "method": m0["method"],
                     "cols": m0["cols_per_day"], "kappa": m0.get("s1_kappa", 0.0) or 0.0,
                     "s1": (f"realistic {m0['s1_variant']}" if m0.get("s1_variant")
                            else (f"κ = {m0['s1_kappa']:g}" if m0.get("s1_kappa") else S0)),
                     "loc": m0["loc_radius"], "ridge": m0.get("etkf_ridge"),
                     "mode": m0.get("etkf_loc_mode", "square_root") if m0["method"] == "etkf" else "-",
                     "lag": m0.get("enks_lag"), "r_scale": float(m0.get("r_scale", 1.0) or 1.0),
                     "taper": m0.get("enks_taper_days"),
                     "complete": len(metas) == n_shards, "n_shards": n_shards,
                     "shards": len(metas), "per_window": pw,
                     "da_seconds": sum(m["da_seconds"] for m in metas)})
    return runs


def manifest_factors(ds_root: str) -> dict:
    out = {}
    for spec in SPECS:
        path = os.path.join(ds_root, spec, "test", "manifest.json")
        if os.path.exists(path):
            m = json.load(open(path))
            out[spec] = {w["index"]: w for w in m["windows"]}
    return out


def _boot_mean(x: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _values(run: dict, tag: str, key: str) -> np.ndarray:
    if key == "score":
        return np.array([np.mean([w[f"ev_{tag}_{f}"] for f in FIELDS]) for w in run["per_window"]])
    return np.array([w[f"ev_{tag}_{key}"] for w in run["per_window"]])


def _find(runs: list[dict], spec: str, method: str, cols: int, s1: str = S0,
          r_scale: float = 1.0) -> dict | None:
    hits = [r for r in runs if r["spec"] == spec and r["method"] == method and r["cols"] == cols
            and r["s1"] == s1 and r["complete"] and (method != "etkf" or r["mode"] == ETKF_MODE)
            and abs(r["r_scale"] - r_scale) < 1e-9
            and (method != "enks" or ((r["lag"] or None) == ENKS_LAG and r["taper"] == ENKS_TAPER))]
    return hits[0] if hits else None


def _rank_marks(cells: list[list[float | None]], ranked: list[bool]) -> list[list[str]]:
    """Bold the best and italicize the second-best value per column among the ranked rows."""
    out = [["" for _ in row] for row in cells]
    for j in range(len(cells[0])):
        col = sorted({round(r[j], 3) for r, rk in zip(cells, ranked) if rk and r[j] is not None},
                     reverse=True)
        for i, (r, rk) in enumerate(zip(cells, ranked)):
            if not rk or r[j] is None:
                continue
            v = round(r[j], 3)
            out[i][j] = "**" if col and v == col[0] else ("*" if len(col) > 1 and v == col[1] else "")
    return out


def _table(header: list[str], rows: list[tuple]) -> list[str]:
    """rows: (name, values, extra cells, ranked?)."""
    marks = _rank_marks([r[1] for r in rows], [r[3] for r in rows])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for (name, vals, extras, _), mk in zip(rows, marks):
        cells = [f"{mk[j]}{v:.3f}{mk[j]}" if v is not None else "—" for j, v in enumerate(vals)]
        lines.append("| " + " | ".join([name] + cells + extras) + " |")
    return lines


def scenario_table(runs: list[dict], spec: str, cols: int, s1: str, r_scale: float = 1.0) -> list[str]:
    rows, base = [], None
    for method in METHODS:
        r = _find(runs, spec, method, cols, s1, r_scale)
        if r is None:
            continue
        base = base or r
        vals = [float(_values(r, "da", k).mean()) for k in FIELDS + ("score",)]
        ci = _boot_mean(_values(r, "da", "score"))
        lag = ""
        if method == "enks":
            lag = f", lag {r['lag'] if r['lag'] is not None else 'whole window'}"
            lag += f", taper {r['taper']:g} d" if r["taper"] is not None else ""
        crps = float(np.mean([w["crps"] for w in r["per_window"]])) * 1e6
        spread = [w["spread_ratio_q1"] for w in r["per_window"] if "spread_ratio_q1" in w]
        rows.append((f"{METHOD_NAME[method]} (radius {r['loc']:g}{lag})", vals,
                     [f"[{ci[0]:.3f}, {ci[1]:.3f}]", crps,
                      f"{np.median(spread):.2f}" if spread else "—"], True))
    if base is None:
        return ["_No complete runs yet._"]
    ranked = sorted({round(x[2][1], 3) for x in rows})
    for i, (name, vals, extras, rk) in enumerate(rows):
        v = round(extras[1], 3)
        mk = "**" if v == ranked[0] else ("*" if len(ranked) > 1 and v == ranked[1] else "")
        rows[i] = (name, vals, [extras[0], f"{mk}{extras[1]:.3f}{mk}", extras[2]], rk)
    vals = [float(_values(base, "free", k).mean()) for k in FIELDS + ("score",)]
    rows.insert(0, ("_free forecast_", vals, ["", "", ""], False))
    return _table(["method"] + [LABEL[k] for k in FIELDS]
                  + ["score", "score 95% CI", "CRPS q (×10⁻⁶, lower is better)", "spread/RMSE q₁"], rows)


def d2_table(runs: list[dict], cols: int, s1: str) -> list[str]:
    lines = ["| method | field | forced | coupled | forced − coupled [95% CI] |", "|---|---|---|---|---|"]
    rng = np.random.default_rng(1)
    any_row = False
    for method in METHODS:
        a = _find(runs, "qg_specwind_gyrostat_v1", method, cols, s1)
        b = _find(runs, "qg_coupled_gyrostat_v1", method, cols, s1)
        if a is None or b is None:
            continue
        for k in FIELDS + ("score",):
            x, y = _values(a, "da", k), _values(b, "da", k)
            d = (x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
                 - y[rng.integers(0, len(y), (10000, len(y)))].mean(1))
            lines.append(f"| {METHOD_NAME[method]} | {LABEL[k]} | {x.mean():.3f} | {y.mean():.3f} | "
                         f"{x.mean() - y.mean():+.3f} [{np.percentile(d, 2.5):+.3f}, "
                         f"{np.percentile(d, 97.5):+.3f}] |")
            any_row = True
    return lines if any_row else ["_Needs complete forced and coupled runs._"]


def density_table(runs: list[dict]) -> list[str]:
    rows = []
    for cols in sorted({r["cols"] for r in runs}):
        r = _find(runs, "qg_specwind_gyrostat_v1", "etkf", cols, S0)
        if r is None:
            continue
        vals = [float(_values(r, "da", k).mean()) for k in FIELDS + ("score",)]
        rows.append((f"{cols} ({100 * cols / 64:.1f}% of columns)", vals, [], True))
    if not rows:
        return ["_No complete runs yet._"]
    return _table(["columns per day"] + [LABEL[k] for k in FIELDS] + ["score"], rows)


def strata_table(runs: list[dict], factors: dict, cols: int, s1: str) -> list[str]:
    r = _find(runs, "qg_specwind_gyrostat_v1", "etkf", cols, s1)
    if r is None or "qg_specwind_gyrostat_v1" not in factors:
        return ["_Needs the complete forced ETKF run._"]
    man = factors["qg_specwind_gyrostat_v1"]
    pw = r["per_window"]
    level = np.array([man[w["index"]]["factors"]["level"] for w in pw])
    unit = np.array([man[w["index"]]["factors"]["time_unit_days"] for w in pw])
    regime = np.array([man[w["index"]]["regime"] for w in pw])
    windy = level > 0
    lt = np.percentile(level[windy], [100 / 3, 200 / 3]) if windy.sum() >= 3 else [np.inf, np.inf]
    ut = np.percentile(unit, [100 / 3, 200 / 3])
    strata = [("calm (level 0)", ~windy), ("windy", windy),
              ("windy, level tercile 1", windy & (level <= lt[0])),
              ("windy, level tercile 2", windy & (level > lt[0]) & (level <= lt[1])),
              ("windy, level tercile 3", windy & (level > lt[1])),
              ("time unit tercile 1 (fast)", unit <= ut[0]),
              ("time unit tercile 2", (unit > ut[0]) & (unit <= ut[1])),
              ("time unit tercile 3 (slow)", unit > ut[1]),
              ("regime 0 or 15", np.isin(regime, (0, 15))), ("other regimes", ~np.isin(regime, (0, 15)))]
    lines = ["| stratum | n | " + " | ".join(f"{LABEL[k]} DA (free)" for k in FIELDS) + " |",
             "|---|---|" + "---|" * len(FIELDS)]
    for name, mask in strata:
        if mask.sum() == 0:
            continue
        cells = []
        for k in FIELDS:
            da = np.array([w[f"ev_da_{k}"] for w in pw])[mask]
            fr = np.array([w[f"ev_free_{k}"] for w in pw])[mask]
            cells.append(f"{da.mean():.3f} ({fr.mean():.3f})")
        lines.append(f"| {name} | {int(mask.sum())} | " + " | ".join(cells) + " |")
    return lines


def synthesis_table(runs: list[dict], s1: str) -> list[str]:
    cfg = {"etkf": "N = 80, inflation 1.0, **exact localized EnSRF update, radius 8, ridge 0.1**, "
                   "cross-layer weight 1, bred init",
           "enkf": "N = 80, inflation 1.0, **radius 6**, cross-layer weight 1, bred init"}
    lines = ["| method | tuned configuration (val, DA-2) | S0 ψ₁ / q₁ | S0 score | S1 ψ₁ / q₁ | S1 score |",
             "|---|---|---|---|---|---|"]
    for method in ("etkf", "enkf"):
        a = _find(runs, "qg_specwind_gyrostat_v1", method, 3, S0)
        b = _find(runs, "qg_specwind_gyrostat_v1", method, 3, s1)
        if a is None:
            continue
        s1_cells = (f"{_values(b, 'da', 'psi1').mean():.3f} / {_values(b, 'da', 'q1').mean():.3f} | "
                    f"{_values(b, 'da', 'score').mean():.3f}") if b is not None else "— | —"
        lines.append(f"| {METHOD_NAME[method]} | {cfg[method]} | {_values(a, 'da', 'psi1').mean():.3f} / "
                     f"{_values(a, 'da', 'q1').mean():.3f} | {_values(a, 'da', 'score').mean():.3f} | "
                     f"{s1_cells} |")
    return lines


def budget_table(s1_root: str, variant: str) -> list[str]:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from evaluation.qg_specwind_s1 import REALISTIC_GROUPS
    from evaluation.qg_specwind_s1_sweep import _load, realistic_name, shapley_over
    if not os.path.isdir(s1_root):
        return ["_No S1 attribution runs found._"]
    sh = shapley_over(_load(s1_root), REALISTIC_GROUPS, lambda c: realistic_name(variant, c))
    if sh is None:
        return [f"_Incomplete realistic-{variant} attribution runs._"]
    lines = ["| metric (S0 → S1) | " + " | ".join(REALISTIC_GROUPS) + " | interaction |",
             "|---|" + "---|" * (len(REALISTIC_GROUPS) + 1)]
    for m in ("score",) + FIELDS:
        d = sh[m]
        parts = []
        top = max(REALISTIC_GROUPS, key=lambda g: d["components"][g]["shapley"])
        for g in REALISTIC_GROUPS:
            share = f"{100 * d['components'][g]['shapley'] / d['loss']:.0f}%"
            parts.append(f"**{share}**" if g == top else share)
        lines.append(f"| {LABEL[m]} ({d['s0']:.3f} → {d['s1']:.3f}) | " + " | ".join(parts)
                     + f" | {d['interaction']:+.3f} |")
    return lines


def _fig(fig_dir: str, name: str, alt: str) -> list[str]:
    return [f"![{alt}](figs/{name})", ""] if os.path.exists(os.path.join(fig_dir, name)) else []


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="experiments/qg_specwind_da")
    p.add_argument("--s1-root", default="experiments/qg_specwind_s1_ensrf")
    p.add_argument("--datasets", default="experiments/qg_datasets")
    p.add_argument("--s1-variant", default="base", help="realistic S1 variant reported")
    p.add_argument("--out", default="reports/qg/outputs/qg_specwind_da_report.md")
    args = p.parse_args()
    runs = load_runs(args.root)
    factors = manifest_factors(args.datasets)
    s1 = f"realistic {args.s1_variant}"
    fig_dir = os.path.join(os.path.dirname(args.out), "figs")
    incomplete = [f"`{os.path.relpath(r['dir'], args.root)}` ({r['shards']}/{r['n_shards']})"
                  for r in runs if not r["complete"] and r["mode"] in (ETKF_MODE, "-")]
    md = ["# QG gyrostat case study: DA baselines (S0 + realistic S1)", "",
          "Two-layer QG ocean (64×64, 1000 km, dt 2 h) forced by the wind-stress curl of a gyrostat "
          "atmosphere (`l63ring4`, 12 Fourier modes |k| ≤ 2) plus relative-wind eddy drag. Test sets: "
          "100 windows of 30 days of the **forced** dataset `qg_specwind_gyrostat_v1` and of the paired "
          "two-way **coupled** dataset `qg_coupled_gyrostat_v1` (κ_fb = 1). Design: "
          "`docs/plans/analysis/qg_specwind_da_s0.md`, `docs/plans/analysis/qg_specwind_da_s1.md`; written "
          "findings: `docs/results/qg_specwind_da_s0_s1_test.md`.", "",
          "**Reference case (S0):** upper-layer ψ₁ observed on 3 random meridional columns per day (4.7% of "
          "the grid), each once per day, 5% noise; initial state = truth lagged about 5 days with a bred "
          "80-member ensemble; the DA model is exact and gets the true wind.", "",
          "**Filters:** ETKF with the exact localized EnSRF update (radius 8, ridge 0.1; since 2026-09-28, "
          "`docs/results/qg_specwind_etkf_loc_update.md` — earlier renders used a localized update with "
          "unnormalized covariances and a full-gain anomaly update, radius 8 / ridge 1); EnKF radius 6; both "
          "with cross-layer localization weight 1 and N = 80, inflation 1.0.", "",
          f"**S1 ({s1}, model error):** wind amplitude +15%, random wind error 25% of each mode's RMS, "
          "wind position error 50 km; rd −10%; bottom drag −50%; altimetry error 15% white + 15% "
          "correlated per pass (the filter's R = total variance, white); DA model on a 32×32 grid "
          "(rd unresolved). Calibrated on val: `docs/results/qg_specwind_da_s1_calibration.md`.", "",
          "**Metrics:** EV = 1 − MSE / Var(truth) per window over its 30 days, averaged over windows; ψ "
          "inverted with each window's own parameters; score = mean of the four EVs; 95% intervals by "
          "bootstrap over windows. Generated by `reports/qg/generate_qg_specwind_da_report.py` (figures: "
          "`reports/qg/generate_qg_specwind_da_figs.py`).", ""]
    if incomplete:
        md += ["**Incomplete runs (excluded):** " + ", ".join(incomplete), ""]
    md += ["## 1. Case study", ""] + _fig(fig_dir, "qg_specwind_gyrostat_animation.gif",
                                            "QG ocean forced by the gyrostat wind") + [
        "The gyrostat drives domain-scale wind-stress-curl patterns with regime changes; the ocean "
        "responds with a forced large-scale circulation on top of its own baroclinic eddies. Windows "
        "differ in ocean parameters (rd, U1, drag), wind level (21% of the test windows are calm, level "
        "0) and gyrostat time unit.", ""]
    for i, (s, title) in enumerate(((S0, "S0 (perfect model, true wind)"),
                                    (s1, f"S1 ({s1})")), start=2):
        md += [f"## {i}. DA baselines — {title}, 3 columns per day", ""]
        for spec, lab in SPECS.items():
            md += [f"**{lab.capitalize()} dataset**", ""] + scenario_table(runs, spec, 3, s) + [""]
        md += ["(Best per column **bolded**, second-best *italicized*, among the DA methods; the free "
               "forecast is a reference row; CRPS is ranked lowest-best; spread/RMSE q₁ (median over windows) is "
               "not ranked: 1 is calibrated. The EnKS is the localized ensemble Kalman smoother on the "
               "ETKF: it also uses observations after each time, so it is a reanalysis-type estimate, not "
               "a filter.)", ""]
    tuned = [r for r in runs if r["s1"] == s1 and abs(r["r_scale"] - S1_TUNED_R) < 1e-9 and r["complete"]]
    if tuned:
        md += [f"### S1-tuned filters (observation-error variance × {S1_TUNED_R:g}), 3 columns per day", "",
               "Same S1, with the filters' R scaled on val to absorb the model error "
               "(`docs/results/qg_specwind_s1_tuning.md`); inflation > 1 diverges here. The rows above use "
               "the S0-tuned filters.", ""]
        for spec, lab in SPECS.items():
            md += [f"**{lab.capitalize()} dataset**", ""] + scenario_table(runs, spec, 3, s1, S1_TUNED_R) + [""]
    md += _fig(fig_dir, "qg_specwind_da_s0_s1.png", "S0 vs S1 per field")
    md += ["## 4. Configuration synthesis and S1 error budget", "",
           "Both filters were tuned on val (`docs/results/qg_specwind_da2_val_tuning.md`): vertical "
           "localization (cross-layer weight 1, so the unobserved layer is updated) and the bred initial "
           "ensemble are the two gains; with both, a much weaker localization is optimal.", ""]
    md += synthesis_table(runs, s1) + [""]
    md += ["**S1 error budget** — Shapley share of the S0 → S1 EV loss per error group (ETKF, val, 20 "
           "windows, all 32 on/off combinations; largest share in bold):", ""]
    md += budget_table(args.s1_root, args.s1_variant) + [""]
    md += _fig(fig_dir, "qg_specwind_da_s1_budget.png", "S1 error budget")
    md += ["## 5. Observation density (S0, forced, ETKF)", ""] + density_table(runs) + [""]
    md += _fig(fig_dir, "qg_specwind_da_density.png", "observation density")
    md += ["**Bridge density (4 columns per day), both datasets:**", ""]
    for spec, lab in SPECS.items():
        md += [f"*{lab}*", ""] + scenario_table(runs, spec, 4, S0) + [""]
    md += ["## 6. Forced vs coupled", "",
           "Independent samples (paired windows share seeds but not trajectories), two-sample bootstrap.", ""]
    md += ["**S0**", ""] + d2_table(runs, 3, S0) + ["", f"**S1 ({s1})**", ""] + d2_table(runs, 3, s1) + [""]
    md += ["## 7. Factor strata (forced, ETKF, 3 columns per day)", "", "**S0**", ""]
    md += strata_table(runs, factors, 3, S0) + ["", f"**S1 ({s1})**", ""]
    md += strata_table(runs, factors, 3, s1) + [""]
    md += ["## 8. Reconstruction examples", "",
           "3 forced test windows (best / median / worst by the ETKF's S0 per-window score), re-run "
           "with the same settings: truth | free forecast | ETKF | EnKF for ψ and q in both layers at "
           "day 27 (whole-window EV in each panel), and the ETKF DA cycle (observed columns, true and "
           "DA-model wind-stress curl, truth / analysis; one frame per day).", ""]
    for scen, title in (("s0", "S0"), ("s1", f"S1 ({s1})")):
        md += [f"### {title}", ""]
        for label in ("best", "median", "worst"):
            block = (_fig(fig_dir, f"qg_specwind_da_recon_{scen}_{label}.png",
                          f"{title} {label} window reconstruction")
                     + _fig(fig_dir, f"qg_specwind_da_cycle_{scen}_{label}.gif",
                            f"{title} {label} window DA cycle"))
            if block:
                md += [f"**{label.capitalize()} window**", ""] + block
    md += ["## 9. Caveats", "",
           "- 100 test windows per cell; strata have 21–79 windows.",
           "- S1 uses the S0-tuned filters; the ETKF–EnKF ranking under S1 may change with S1-specific "
           "tuning (inflation, R, radius).",
           "- The initial state is the lagged truth in both scenarios (optimistic against an "
           "analysis-cycled first guess).",
           "- Only S0 has the density curve; S1 was run at 3 columns per day.", ""]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
