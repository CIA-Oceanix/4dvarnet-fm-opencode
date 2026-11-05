"""Report: QG gyrostat case study -- DA baselines and neural schemes (S0 + realistic S1) on the spectral-wind test sets.

Layout follows the legacy QG DA report (`reports/qg/outputs/qg_da_report.md`):
reference case, per-scenario DA tables (best bold, second-best italic, among
the DA methods; the free forecast is a reference row), configuration
synthesis with the S1 error budget, observation density, forced vs coupled,
factor strata, and best/median/worst reconstruction examples with DA-cycle
animations (`generate_qg_specwind_da_figs.py`).

Reads the per-shard `*_meta.json` written by `evaluation.run_qg_specwind_da`
(per-window EVs computed with each window's own parameters) under
`<root>/<spec>/test/<config>/`, the dataset manifests for the window factors,
and the S1 attribution runs of `evaluation.qg_specwind_s1_sweep`. Neural rows
(sections 9-10) are the `neural_*` configs written in the same schema by
`evaluation.run_qg_specwind_neural` (forced test set only).

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
METHOD_NAME = {"etkf": "ETKF", "enkf": "EnKF", "enks": "EnKS (ETKF smoother)",
               "strong4dvar": "strong 4D-Var", "weak4dvar": "weak 4D-Var"}
METHODS = ("etkf", "enkf", "enks")
VAR_METHODS = ("strong4dvar", "weak4dvar")
VAR_SETTINGS = {"b": "spectral", "b_var_scale": 1.0, "da_window_steps": 120, "max_iter": 60}
WEAK_Q = 0.1
S0 = "S0"
ETKF_MODE = "ensrf"
S1_TUNED_R = 6.0
ENKS_LAG = None
ENKS_TAPER = 8.0


def load_runs(root: str) -> list[dict]:
    groups = defaultdict(list)
    for f in glob.glob(os.path.join(root, "*", "test", "*", "shard_*_meta.json")):
        groups[os.path.dirname(f)].append(f)
    runs = []
    for d, files in sorted(groups.items()):
        metas = [json.load(open(f)) for f in sorted(files)]
        if any("per_window" not in m for m in metas) or metas[0]["method"].startswith("neural_"):
            continue
        m0 = metas[0]
        n_shards = int(os.path.basename(files[0]).split("_of_")[1].split("_")[0])
        pw = sorted((w for m in metas for w in m["per_window"]), key=lambda w: w["index"])
        runs.append({"dir": d, "spec": m0["spec"], "method": m0["method"],
                     "cols": m0["cols_per_day"], "noise": float(m0.get("obs_noise_frac", 0.05)), "kappa": m0.get("s1_kappa", 0.0) or 0.0,
                     "s1": (f"realistic {m0['s1_variant']}" if m0.get("s1_variant")
                            else (f"κ = {m0['s1_kappa']:g}" if m0.get("s1_kappa") else S0)),
                     "loc": m0["loc_radius"], "ridge": m0.get("etkf_ridge"),
                     "mode": m0.get("etkf_loc_mode", "square_root") if m0["method"] == "etkf" else "-",
                     "lag": m0.get("enks_lag"), "r_scale": float(m0.get("r_scale", 1.0) or 1.0),
                     "taper": m0.get("enks_taper_days"), "N": int(m0.get("N", 80)),
                     "fourdvar": m0.get("fourdvar"),
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


def _var_ok(r: dict) -> bool:
    fv = r.get("fourdvar") or {}
    return (all(fv.get(k) == v for k, v in VAR_SETTINGS.items())
            and (r["method"] != "weak4dvar" or abs(fv.get("q_var_scale", 0.0) - WEAK_Q) < 1e-9))


def _find(runs: list[dict], spec: str, method: str, cols: int, s1: str = S0,
          r_scale: float = 1.0, noise: float = 0.05) -> dict | None:
    runs = [r for r in runs if abs(r["noise"] - noise) < 1e-9]
    if method in VAR_METHODS:
        hits = [r for r in runs if r["spec"] == spec and r["method"] == method and r["cols"] == cols
                and r["s1"] == s1 and r["complete"] and abs(r["r_scale"] - r_scale) < 1e-9 and _var_ok(r)]
        return hits[0] if hits else None
    hits = [r for r in runs if r["spec"] == spec and r["method"] == method and r["cols"] == cols
            and r["s1"] == s1 and r["complete"] and (method != "etkf" or r["mode"] == ETKF_MODE)
            and abs(r["r_scale"] - r_scale) < 1e-9 and r["N"] == 80
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


def _var_desc(r: dict) -> str:
    fv = r["fourdvar"]
    desc = f"spectral B, {fv['da_window_steps'] / 12:g}-day sub-windows"
    if r["method"] == "weak4dvar":
        desc += f", model-error scale {fv['q_var_scale']:g}"
    n_fb = sum(w.get("n_fallback", 0) for w in r["per_window"])
    return desc + (f"; {n_fb} sub-window fallbacks" if n_fb else "")


def scenario_table(runs: list[dict], spec: str, cols: int, s1: str, r_scale: float = 1.0,
                   var_methods: tuple[tuple[str, float], ...] = ()) -> list[str]:
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
            lag = f", lag {r['lag']}" if r["lag"] else (", no lag cutoff" if r["taper"] else ", whole window")
            lag += f", time taper {r['taper']:g} d" if r["taper"] is not None else ""
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
    for method, var_r in var_methods:
        r = _find(runs, spec, method, cols, s1, var_r)
        if r is None:
            continue
        vals = [float(_values(r, "da", k).mean()) for k in FIELDS + ("score",)]
        ci = _boot_mean(_values(r, "da", "score"))
        mae = float(np.mean([w["crps"] for w in r["per_window"]])) * 1e6
        rows.append((f"{METHOD_NAME[method]} ({_var_desc(r)})", vals,
                     [f"[{ci[0]:.3f}, {ci[1]:.3f}]", f"({mae:.3f}, MAE)", "—"], True))
    vals = [float(_values(base, "free", k).mean()) for k in FIELDS + ("score",)]
    rows.insert(0, ("_free forecast_", vals, ["", "", ""], False))
    return _table(["method"] + [LABEL[k] for k in FIELDS]
                  + ["score", "score 95% CI", "CRPS q (×10⁻⁶, lower is better)", "spread/RMSE q₁"], rows)


OVERVIEW = (("S0", S0, (("etkf", 1.0), ("enkf", 1.0), ("enks", 1.0), ("strong4dvar", 1.0))),
            ("S1, S0-tuned", None, (("etkf", 1.0), ("enkf", 1.0), ("enks", 1.0), ("strong4dvar", 1.0))),
            ("S1-tuned", None, (("etkf", S1_TUNED_R), ("enkf", S1_TUNED_R), ("enks", S1_TUNED_R),
                                ("weak4dvar", 1.0))))


def overview_table(runs: list[dict], s1: str) -> list[str]:
    """Score per method and scenario, forced / coupled; best bold, second italic per scenario column."""
    cols = [(lab, scen or s1, cells) for lab, scen, cells in OVERVIEW]
    names = {"etkf": "ETKF", "enkf": "EnKF", "enks": "EnKS", "strong4dvar": "4D-Var (strong / weak)",
             "weak4dvar": "4D-Var (strong / weak)"}
    order = ["ETKF", "EnKF", "EnKS", "4D-Var (strong / weak)"]
    table = {n: {} for n in order}
    for j, (_, scen, cells) in enumerate(cols):
        for method, r_scale in cells:
            for spec in SPECS:
                r = _find(runs, spec, method, 3, scen, r_scale)
                table[names[method]][(j, spec)] = None if r is None else float(_values(r, "da", "score").mean())
    header = ["method"] + [f"{lab} {SPECS[spec]}" for lab, _, _ in cols for spec in SPECS]
    rows = [(n, [table[n].get((j, spec)) for j in range(len(cols)) for spec in SPECS], [], True) for n in order]
    return _table(header, rows)


LARGE_N = 320
LARGE_N_SETTINGS = {
    S0: {"etkf": {"loc": 12.0}, "enks": {"loc": 12.0, "taper": 16.0}},
    "S1": {"etkf": {"loc": 8.0}, "enks": {"loc": 8.0, "taper": 8.0}},
}


def _find_large(runs: list[dict], spec: str, method: str, s1: str, r_scale: float) -> dict | None:
    want = LARGE_N_SETTINGS[S0 if s1 == S0 else "S1"][method]
    hits = [r for r in runs if abs(r["noise"] - 0.05) < 1e-9 and r["spec"] == spec and r["method"] == method and r["cols"] == 3
            and r["s1"] == s1 and r["complete"] and r["N"] == LARGE_N
            and abs(r["r_scale"] - r_scale) < 1e-9 and abs(r["loc"] - want["loc"]) < 1e-9
            and (method != "enks" or r["taper"] == want["taper"])]
    return hits[0] if hits else None


def large_ensemble_table(runs: list[dict], s1: str) -> list[str]:
    lines = ["| scenario | dataset | method (N = 320 settings) | score N = 80 | score N = 320 | N 320 − 80 [95% CI] | "
             "CRPS q (×10⁻⁶) N = 80 → 320 |", "|---|---|---|---|---|---|---|"]
    rng = np.random.default_rng(2)
    any_row = False
    for scen, label, r_scale in ((S0, "S0", 1.0), (s1, f"S1-tuned ({s1}, R × {S1_TUNED_R:g})", S1_TUNED_R)):
        for spec, lab in SPECS.items():
            for method in ("etkf", "enks"):
                big = _find_large(runs, spec, method, scen, r_scale)
                small = _find(runs, spec, method, 3, scen, r_scale)
                if big is None or small is None:
                    continue
                d = _values(big, "da", "score") - _values(small, "da", "score")
                m = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
                want = LARGE_N_SETTINGS[S0 if scen == S0 else "S1"][method]
                desc = f"radius {want['loc']:g}" + (f", taper {want['taper']:g} d" if method == "enks" else "")
                crps = [float(np.mean([w["crps"] for w in r["per_window"]])) * 1e6 for r in (small, big)]
                lines.append(f"| {label} | {lab} | {METHOD_NAME[method]} ({desc}) | "
                             f"{_values(small, 'da', 'score').mean():.3f} | **{_values(big, 'da', 'score').mean():.3f}** | "
                             f"{d.mean():+.3f} [{np.percentile(m, 2.5):+.3f}, {np.percentile(m, 97.5):+.3f}] | "
                             f"{crps[0]:.3f} → {crps[1]:.3f} |")
                any_row = True
    return lines if any_row else ["_No complete N = 320 runs yet._"]


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


BUDGETS = (("ETKF, S0-tuned (R × 1)", ""), ("ETKF, S1-tuned (R × 6)", "etkf_R6"),
           ("EnKS, S1-tuned (R × 6, taper 8 d)", "enks_R6"))


def budget_compare_table(s1_root: str, variant: str) -> list[str]:
    """Absolute Shapley score loss per error group for each budget, with paired contrasts."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from evaluation.qg_specwind_s1 import REALISTIC_GROUPS
    from evaluation.qg_specwind_s1_sweep import _load, realistic_name, shapley_over
    shs = {}
    for label, sub in BUDGETS:
        d = os.path.join(s1_root, sub) if sub else s1_root
        if os.path.isdir(d):
            sh = shapley_over(_load(d), REALISTIC_GROUPS, lambda c: realistic_name(variant, c))
            if sh is not None:
                shs[label] = sh
    if len(shs) < 2:
        return ["_Needs at least two complete attribution sweeps._"]
    lines = ["| filter (score S0 corner → S1) | " + " | ".join(REALISTIC_GROUPS) + " | interaction | total loss |",
             "|---|" + "---|" * (len(REALISTIC_GROUPS) + 2)]
    for label, sh in shs.items():
        d = sh["score"]
        top = max(REALISTIC_GROUPS, key=lambda g: d["components"][g]["shapley"])
        cells = []
        for g in REALISTIC_GROUPS:
            c = d["components"][g]
            v = f"{c['shapley']:.3f} [{c['shapley_ci'][0]:.3f}, {c['shapley_ci'][1]:.3f}]"
            cells.append(f"**{v}**" if g == top else v)
        lines.append(f"| {label} ({d['s0']:.3f} → {d['s1']:.3f}) | " + " | ".join(cells)
                     + f" | {d['interaction']:+.3f} | {d['loss']:.3f} |")
    rng = np.random.default_rng(3)
    labels = list(shs)
    contrasts = [(labels[1], labels[0], "S1 tuning (ETKF, R × 6 − R × 1)")] if len(labels) > 1 else []
    if len(labels) > 2:
        contrasts.append((labels[2], labels[1], "smoothing (EnKS − ETKF, both R × 6)"))
    lines += ["", "| paired contrast (score loss, 95% CI) | " + " | ".join(REALISTIC_GROUPS) + " |",
              "|---|" + "---|" * len(REALISTIC_GROUPS)]
    for a, b, name in contrasts:
        cells = []
        for g in REALISTIC_GROUPS:
            x = (np.array(shs[a]["score"]["components"][g]["phi"])
                 - np.array(shs[b]["score"]["components"][g]["phi"]))
            m = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
            cells.append(f"{x.mean():+.3f} [{np.percentile(m, 2.5):+.3f}, {np.percentile(m, 97.5):+.3f}]")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


NEURAL_SPEC = "qg_specwind_gyrostat_v1"
OBS_CONFIGS = (("c3", "S0, 3 col/day, 5%", 3, 0.05, False),
               ("s1", "S1 (realistic base), 3 col/day", 3, 0.05, True),
               ("c6", "S0, 6 col/day, 5%", 6, 0.05, False),
               ("c12", "S0, 12 col/day, 5%", 12, 0.05, False),
               ("n15", "S0, 3 col/day, 15%", 3, 0.15, False),
               ("n02", "S0, 3 col/day, 2%", 3, 0.02, False))
NOISE_SUFFIX = {0.05: "", 0.15: "_n15", 0.02: "_n02"}
# (label, tag per config key ("*" = all), members, ensemble?)
NEURAL_SCHEMES = (
    ("G1-M-3D DirectUNet, stitched", {"*": "G1M3D_b16_w15_kpow_s0"}, 1, False),
    ("G1-M-3D DirectUNet, full window", {"*": "G1M3D_b16_w15_kpow_s0_full"}, 1, False),
    ("G1-L-3D DirectUNet, stitched", {"*": "G1L3D_b16_w15_kpow_s0"}, 1, False),
    ("G1-L-3D DirectUNet, full window", {"*": "G1L3D_b16_w15_kpow_s0_full"}, 1, False),
    ("G2-3D CFM, stitched", {"*": "G2_3d_w15_kpow_s0"}, 30, True),
    ("G3-3D SDA prior, pure-noise SDA", {"*": "G3_3d_w15_pub_g0.01_st100_c1",
                                         "s1": "G3M_3d_w15_pub_g0.01_st100_c1"}, 30, True),
    ("hybrid, G2-3D warm start (weight 5; S1: 2)", {"*": "G3_3d_w15_warmG2_t0.5_gw5_st10",
                                                     "s1": "G3_3d_w15_warmG2_t0.5_gw2_st10"}, 30, True),
    ("hybrid, G1-L-3D warm start (weight 2)", {"*": "G3_3d_w15_warmG1L3Dfull_t0.5_gw2_st10"}, 30, True),
)
HYBRID_LABEL = NEURAL_SCHEMES[-1][0]
END_TO_END = tuple(s[0] for s in NEURAL_SCHEMES if s[0].startswith(("G1", "G2")))
# DA reference rows per config key: (label, method, r_scale, N, deterministic?)
_S0_STRESS = {"c6": 1.0, "c12": 1.0, "n15": 1.0, "n02": 1.0}
DA_REFS = (("ETKF (N = 80)", "etkf", {"c3": 1.0, "s1": S1_TUNED_R, **_S0_STRESS}, 80, False),
           ("EnKS (N = 80)", "enks", {"c3": 1.0, "s1": S1_TUNED_R, **_S0_STRESS}, 80, False),
           ("EnKS (N = 320)", "enks", {"c3": 1.0, "s1": S1_TUNED_R}, LARGE_N, False),
           ("4D-Var (strong; S1: weak)", "4dvar", {"c3": 1.0, "s1": 1.0, **_S0_STRESS}, 0, True))


def load_neural(root: str) -> dict[str, dict]:
    out = {}
    for d in sorted(glob.glob(os.path.join(root, NEURAL_SPEC, "test", "neural_*"))):
        files = sorted(glob.glob(os.path.join(d, "shard_*_meta.json")))
        if not files:
            continue
        metas = [json.load(open(f)) for f in files]
        n_shards = int(os.path.basename(files[0]).split("_of_")[1].split("_")[0])
        if len(metas) != n_shards or any("per_window" not in m for m in metas):
            continue
        pw = sorted((w for m in metas for w in m["per_window"]), key=lambda w: w["index"])
        out[os.path.basename(d)] = {"meta": metas[0], "per_window": pw}
    return out


def _neural_dir(tag: str, cfg: tuple, members: int) -> str:
    key, _, cols, noise, s1 = cfg
    return f"neural_{tag}{NOISE_SUFFIX[noise]}_c{cols}_cubic_N{members}" + ("_k1_rbase" if s1 else "")


def _neural_run(neural: dict, scheme: tuple, cfg: tuple) -> dict | None:
    _, tags, members, _ = scheme
    tag = tags.get(cfg[0], tags["*"])
    r = neural.get(_neural_dir(tag, cfg, members))
    if r is None:
        return None
    m = r["meta"]
    if (m["cols_per_day"] != cfg[2] or abs(m["obs_noise_frac"] - cfg[3]) > 1e-9
            or bool(m.get("s1_variant")) != cfg[4] or len(r["per_window"]) != 100):
        raise ValueError(f"{_neural_dir(tag, cfg, members)}: meta does not match config {cfg[0]}")
    return r


def _da_ref_run(runs: list[dict], ref: tuple, cfg: tuple, s1: str) -> dict | None:
    _, method, r_scales, n, _ = ref
    if cfg[0] not in r_scales:
        return None
    scen = s1 if cfg[4] else S0
    if method == "4dvar":
        return _find(runs, NEURAL_SPEC, "weak4dvar" if cfg[4] else "strong4dvar", cfg[2], scen,
                     noise=cfg[3])
    if n == LARGE_N:
        return _find_large(runs, NEURAL_SPEC, method, scen, r_scales[cfg[0]])
    return _find(runs, NEURAL_SPEC, method, cfg[2], scen, r_scales[cfg[0]], cfg[3])


def _score(r: dict) -> np.ndarray:
    return np.array([np.mean([w[f"ev_da_{f}"] for f in FIELDS]) for w in r["per_window"]])


def _rows_for(runs: list[dict], neural: dict, cfg: tuple, s1: str) -> list[tuple[str, dict, bool, bool]]:
    """(label, run, ensemble?, neural?) for every scheme scored on this config."""
    rows = []
    for ref in DA_REFS:
        r = _da_ref_run(runs, ref, cfg, s1)
        if r is not None:
            rows.append((ref[0], r, not ref[4], False))
    for scheme in NEURAL_SCHEMES:
        r = _neural_run(neural, scheme, cfg)
        if r is not None:
            rows.append((scheme[0], r, scheme[3], True))
    return rows


def _mark(values: list[float | None], lower_better: bool = False) -> list[str]:
    ranked = sorted({round(v, 3) for v in values if v is not None}, reverse=not lower_better)
    out = []
    for v in values:
        if v is None:
            out.append("—")
            continue
        mk = "**" if round(v, 3) == ranked[0] else ("*" if len(ranked) > 1 and round(v, 3) == ranked[1] else "")
        out.append(f"{mk}{v:.3f}{mk}")
    return out


def neural_field_table(runs: list[dict], neural: dict, cfg: tuple, s1: str) -> list[str]:
    rows = _rows_for(runs, neural, cfg, s1)
    if not rows:
        return ["_No complete runs yet._"]
    vals = [[float(np.mean([w[f"ev_da_{f}"] for w in r["per_window"]])) for f in FIELDS]
            + [float(_score(r).mean())] for _, r, _, _ in rows]
    cols = [_mark([v[j] for v in vals]) for j in range(len(FIELDS) + 1)]
    crps = [float(np.mean([w["crps"] for w in r["per_window"]])) * 1e6 for _, r, _, _ in rows]
    crps_marked = _mark([c if ens else None for c, (_, _, ens, _) in zip(crps, rows)], lower_better=True)
    lines = ["| scheme | " + " | ".join(LABEL[k] for k in FIELDS) + " | score | score 95% CI | "
             "CRPS q (×10⁻⁶) | spread/RMSE q₁ / q₂ |", "|" + "---|" * 9]
    for i, (name, r, ens, is_neural) in enumerate(rows):
        lo, hi = _boot_mean(_score(r))
        crps_cell = crps_marked[i] if ens else f"({crps[i]:.3f}, MAE)"
        sr = [np.median([w[k] for w in r["per_window"]]) if ens and k in r["per_window"][0] else None
              for k in ("spread_ratio_q1", "spread_ratio_q2")]
        sr_cell = " / ".join(f"{x:.2f}" if x is not None else "—" for x in sr)
        label = name if is_neural else f"_{name}_ (DA)"
        lines.append(f"| {label} | " + " | ".join(c[i] for c in cols)
                     + f" | [{lo:.3f}, {hi:.3f}] | {crps_cell} | {sr_cell} |")
    return lines


def obs_config_tables(runs: list[dict], neural: dict, s1: str) -> list[str]:
    labels = [r[0] for r in DA_REFS] + [s[0] for s in NEURAL_SCHEMES]
    score = {lab: {} for lab in labels}
    crps = {lab: {} for lab in labels}
    for cfg in OBS_CONFIGS:
        for name, r, ens, _ in _rows_for(runs, neural, cfg, s1):
            score[name][cfg[0]] = float(_score(r).mean())
            if ens:
                crps[name][cfg[0]] = float(np.mean([w["crps"] for w in r["per_window"]])) * 1e6
    header = "| scheme | " + " | ".join(c[1] for c in OBS_CONFIGS) + " |"
    sep = "|" + "---|" * (len(OBS_CONFIGS) + 1)
    out = ["**Score (mean EV)**", "", header, sep]
    marks = {c[0]: _mark([score[lab].get(c[0]) for lab in labels]) for c in OBS_CONFIGS}
    for i, lab in enumerate(labels):
        if score[lab]:
            name = lab if i >= len(DA_REFS) else f"_{lab}_ (DA)"
            out.append(f"| {name} | " + " | ".join(marks[c[0]][i] for c in OBS_CONFIGS) + " |")
    out += ["", "**CRPS of q (×10⁻⁶, lower is better; ensembles only)**", "", header, sep]
    marks = {c[0]: _mark([crps[lab].get(c[0]) for lab in labels], lower_better=True) for c in OBS_CONFIGS}
    for i, lab in enumerate(labels):
        if crps[lab]:
            name = lab if i >= len(DA_REFS) else f"_{lab}_ (DA)"
            out.append(f"| {name} | " + " | ".join(marks[c[0]][i] for c in OBS_CONFIGS) + " |")
    return out


def _paired(a: dict, b: dict, seed: int = 3) -> tuple[float, float, float]:
    ia = {w["index"]: s for w, s in zip(a["per_window"], _score(a))}
    ib = {w["index"]: s for w, s in zip(b["per_window"], _score(b))}
    common = sorted(set(ia) & set(ib))
    d = np.array([ia[i] - ib[i] for i in common])
    m = d[np.random.default_rng(seed).integers(0, len(d), (10000, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_delta_table(runs: list[dict], neural: dict, s1: str) -> list[str]:
    """Paired score difference of the hybrid and of pure-noise SDA vs the best end-to-end scheme per config."""
    lines = ["| config | best end-to-end (score) | hybrid − best end-to-end [95% CI] | "
             "pure-noise SDA − best end-to-end [95% CI] | best DA reference (score) | "
             "hybrid − best DA [95% CI] |", "|---|---|---|---|---|---|"]
    by_label = {s[0]: s for s in NEURAL_SCHEMES}
    for cfg in OBS_CONFIGS:
        rows = _rows_for(runs, neural, cfg, s1)
        e2e = [(n, r) for n, r, _, nn in rows if nn and n in END_TO_END]
        hyb = _neural_run(neural, by_label[HYBRID_LABEL], cfg)
        sda = _neural_run(neural, by_label["G3-3D SDA prior, pure-noise SDA"], cfg)
        if not e2e or hyb is None:
            continue
        best_n, best = max(e2e, key=lambda x: _score(x[1]).mean())
        cells = [f"{cfg[1]}", f"{best_n} ({_score(best).mean():.3f})"]
        for other in (hyb, sda):
            if other is None:
                cells.append("—")
                continue
            m, lo, hi = _paired(other, best)
            cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        da = [(n, r) for n, r, _, nn in rows if not nn]
        if da:
            dn, dr = max(da, key=lambda x: _score(x[1]).mean())
            m, lo, hi = _paired(hyb, dr)
            cells += [f"{dn} ({_score(dr).mean():.3f})", f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"]
        else:
            cells += ["—", "—"]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _fig(fig_dir: str, name: str, alt: str) -> list[str]:
    return [f"![{alt}](figs/{name})", ""] if os.path.exists(os.path.join(fig_dir, name)) else []


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="experiments/qg_specwind_da")
    p.add_argument("--neural-root", default=None, help="neural rows (default: --root)")
    p.add_argument("--s1-root", default="experiments/qg_specwind_s1_ensrf")
    p.add_argument("--datasets", default="experiments/qg_datasets")
    p.add_argument("--s1-variant", default="base", help="realistic S1 variant reported")
    p.add_argument("--out", default="reports/qg/outputs/qg_specwind_da_report.md")
    args = p.parse_args()
    runs = load_runs(args.root)
    neural = load_neural(args.neural_root or args.root)
    factors = manifest_factors(args.datasets)
    s1 = f"realistic {args.s1_variant}"
    fig_dir = os.path.join(os.path.dirname(args.out), "figs")
    incomplete = [f"`{os.path.relpath(r['dir'], args.root)}` ({r['shards']}/{r['n_shards']})"
                  for r in runs if not r["complete"] and r["mode"] in (ETKF_MODE, "-")]
    md = ["# QG gyrostat case study: DA baselines and neural schemes (S0 + realistic S1)", "",
          "Two-layer QG ocean (64×64, 1000 km, dt 2 h) forced by the wind-stress curl of a gyrostat "
          "atmosphere (`l63ring4`, 12 Fourier modes |k| ≤ 2) plus relative-wind eddy drag. Test sets: "
          "100 windows of 30 days of the **forced** dataset `qg_specwind_gyrostat_v1` and of the paired "
          "two-way **coupled** dataset `qg_coupled_gyrostat_v1` (κ_fb = 1). Design: "
          "`docs/plans/analysis/qg_specwind_da_s0.md`, `docs/plans/analysis/qg_specwind_da_s1.md`; written "
          "findings: `docs/results/qg_specwind_da_s0_s1_test.md`. Neural schemes (DirectUNet, CFM, SDA "
          "prior and SDA hybrids) on the forced test set, and their observation-configuration stress "
          "test: sections 9–10.", "",
          "**Reference case (S0):** upper-layer ψ₁ observed on 3 random meridional columns per day (4.7% of "
          "the grid), each once per day, 5% noise; initial state = truth lagged about 5 days with a bred "
          "80-member ensemble; the DA model is exact and gets the true wind.", "",
          "**Filters:** ETKF with the exact localized EnSRF update (radius 8, ridge 0.1; since 2026-09-28, "
          "`docs/results/qg_specwind_etkf_loc_update.md` — earlier renders used a localized update with "
          "unnormalized covariances and a full-gain anomaly update, radius 8 / ridge 1); EnKF radius 6; both "
          "with cross-layer localization weight 1 and N = 80, inflation 1.0.", "",
          "**4D-Var:** deterministic, from the lagged-truth background; climatological B from a "
          "per-wavenumber-shell 2×2 inter-layer spectral covariance (`spectral_b_sqrt`), whitened control, "
          "L-BFGS (60 iterations), cycled over 10-day sub-windows; a sub-window whose optimization is "
          "non-finite or raises the cost falls back to the background forecast. Strong constraint in S0; "
          "in S1 also the weak constraint (additive model-error control, scale 0.1 of B), tuned on val "
          "(`docs/results/qg_specwind_4dvar.md`). Its CRPS column is the MAE of q (a deterministic "
          "CRPS) and is not ranked against the ensembles.", "",
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
    md += ["## Benchmark overview (score, 3 columns per day, N = 80)", "",
           "Mean of the four EVs over 100 test windows; best per column **bolded**, second-best "
           "*italicized*. S1-tuned = ensembles with R × 6, 4D-Var with the weak constraint (strong in the "
           "other two columns). Per-field tables and intervals in sections 2–3. The neural schemes and the "
           "DA reference rows side by side, across observation configurations: section 10.", ""]
    md += overview_table(runs, s1) + [""]
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
            md += [f"**{lab.capitalize()} dataset**", ""] + scenario_table(
                runs, spec, 3, s, var_methods=(("strong4dvar", 1.0),)) + [""]
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
               "the S0-tuned filters. The 4D-Var row is the weak constraint at R × 1 (inflating R hurts "
               "4D-Var; `docs/results/qg_specwind_4dvar.md`). Relaxation inflation (RTPS/RTPP) was "
               "tested on val as an alternative: it is stable and calibrates the spread, but trails "
               "R × 6 by about 0.03 and adds at most +0.004 on top of it, so the S1-tuned rows keep "
               "R × 6 without relaxation (`docs/results/qg_specwind_relax_inflation.md`).", ""]
        for spec, lab in SPECS.items():
            md += [f"**{lab.capitalize()} dataset**", ""] + scenario_table(
                runs, spec, 3, s1, S1_TUNED_R, var_methods=(("weak4dvar", 1.0),)) + [""]
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
    md += ["**S1-tuned error budgets** — the same attribution with each filter fixed at its S1-tuned setting "
           "(R × 6) in all 32 combinations (the S0 corner then runs R × 6 on perfect-model data: it costs "
           "nothing, ETKF 0.768 vs 0.764, EnKS 0.809 vs 0.806). Absolute Shapley score loss per group "
           "(bootstrap 95% over the 20 windows; largest in bold) and paired contrasts "
           "(`docs/results/qg_specwind_da_s1_calibration.md` §3.1d):", ""]
    md += budget_compare_table(args.s1_root, args.s1_variant) + [""]
    md += ["Share of the EV loss per field, S1-tuned EnKS:", ""]
    md += budget_table(os.path.join(args.s1_root, "enks_R6"), args.s1_variant) + [""]
    md += ["### Large-ensemble reference (N = 320)", "",
           "The benchmark uses N = 80. At N = 320, S0 is still not converged (val: +0.022 for the EnKS "
           "from 80 to 320), while S1 saturates by N = 160. In S0 the N = 320 rows use settings re-tuned "
           "at that size (wider localization). Use these rows to quote converged skill, for instance "
           "against learned methods (`docs/results/qg_specwind_ensemble_size.md`).", ""]
    md += large_ensemble_table(runs, s1) + [""]
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
    md += ["## 9. Neural schemes (forced test set)", "",
           "Learned reconstructions on the same 100 forced test windows, observation draws and metrics as "
           "the DA rows (design: `docs/plans/analysis/qg_specwind_neural.md`; findings, val tuning and "
           "ablations: `docs/results/qg_specwind_neural_obs_configs.md`). The networks see only the "
           "observations: no initial state, no wind, no ocean parameters. They estimate daily means, "
           "cubic-interpolated to the 2-hour DA grid. All are 3D U-Nets (2 space + time) trained on "
           "15-day crops of regenerated forced training windows (the true model), with a random "
           "observing system per draw: 3–30 observed (step, column) slots per day drawn with a "
           "K^-1.5 tilt towards sparse, and an observation error from the S0 model to the S1 one "
           "(white 5–15% plus a per-pass correlated 0–15% of the ψ₁ std; the end-to-end nets and the "
           "observation-conditioned flow only, the SDA prior never sees observations). They are "
           "applied to the 30-day windows either **stitched** (overlapping 15-day passes) or in one "
           "**full-window** pass. One training seed each.", "",
           "- **G1 DirectUNet** — deterministic end-to-end estimator; M tier (16.8 M parameters) and "
           "L tier.",
           "- **G2 CFM** — observation-conditioned flow (M tier, matched to G1-M-3D), 30 members.",
           "- **G3 SDA prior** — unconditional score-based prior (10.0 M parameters) with SDA guided "
           "posterior sampling from pure noise, as published (Rozet & Louppe: γ 0.01, 100 steps, 1 "
           "corrector, observation σ = the noise level), 30 members. The tier-matched 16.8 M retrain "
           "scores the same (within 0.002 at 3 and 12 columns and 15% noise); the S1 row uses it, "
           "the old prior was not scored at S1.",
           "- **Hybrid** — the G3 prior's SDA sampler started at τ₀ = 0.5 from an end-to-end estimate "
           "(SDEdit), 10 steps, fixed-length normalized guidance step (weight). The G2 warm start uses "
           "one CFM sample per member; the G1-L-3D warm start uses its full-window estimate. The "
           "G1-L-3D hybrid uses one setting everywhere (τ₀ 0.5, weight 2, selected on val).", "",
           "DA reference rows (italic, marked DA) are the benchmark ones from sections 2–4 (S1: the "
           "S1-tuned ensembles and weak 4D-Var). Unlike the networks, DA starts from the truth lagged "
           "about 5 days and runs the exact model (S0) with the true wind. Ranking (bold best, italic "
           "second-best) is over all rows; CRPS is ranked over the ensembles only (deterministic rows "
           "show the MAE of q); spread/RMSE (median over windows) is not ranked, 1 is calibrated.", ""]
    for cfg in OBS_CONFIGS[:2]:
        md += [f"**{cfg[1]}**", ""] + neural_field_table(runs, neural, cfg, s1) + [""]
    md += ["## 10. Observation configurations (neural stress test)", "",
           "The same networks, not retrained, on denser observations (6 and 12 columns per day) and on "
           "noisier or cleaner ones (15% and 2% of the ψ₁ std) at 3 columns per day. 6 and 12 columns "
           "and 15% noise are inside the training observing system, 2% is below its error range. The "
           "SDA observation σ follows the noise level; the hybrids keep their single setting, except "
           "the G2-3D one (weight 5 everywhere but S1). The DA rows (ETKF, EnKS, strong 4D-Var) keep their "
           "reference-case settings too (radius 8, 4D-Var tuned at 3 columns per day); their R follows "
           "the noise level. The N = 320 EnKS was run at the reference case and S1 only.", ""]
    md += obs_config_tables(runs, neural, s1) + [""]
    md += ["**Paired differences** (per-window score differences over the same 100 windows; bootstrap "
           "95% CI). Best end-to-end = the best G1/G2 row in that column.", ""]
    md += paired_delta_table(runs, neural, s1) + [""]
    md += ["## 11. Caveats", "",
           "- 100 test windows per cell; strata have 21–79 windows.",
           "- S1 tuning is a single global R scale for the ensembles and a single model-error scale for "
           "weak 4D-Var; relaxation inflation does not beat R × 6, and scale-dependent model error was "
           "not tried.",
           "- 4D-Var is deterministic (no spread); its background error is climatological, not cycled.",
           "- The initial state is the lagged truth in both scenarios (optimistic against an "
           "analysis-cycled first guess).",
           "- Only S0 has the density curve; S1 was run at 3 columns per day.",
           "- The stress-test DA rows are not re-tuned: the ETKF's q₂ falls from 6 to 12 columns per "
           "day (0.592 → 0.566), a sign that radius 8 is too wide at that density, and 4D-Var's "
           "sub-window fallbacks rise at 15% and 2% noise (6 each).",
           "- Neural vs DA is not like for like in information: DA gets the lagged truth as its initial "
           "state and, in S0, the exact model and wind; the networks get only the observations but "
           "were trained on simulations of the true model. In S1 the networks therefore carry no "
           "model error (their S1 is the corrupted observations only), so most of the S1 gap to DA "
           "is the DA model error, not a better assimilation.",
           "- Neural rows: one training seed each; the hybrid with the G1-L-3D warm start mixes tiers "
           "(L-tier warm start, M-tier-size prior), and G2/G3 exist at M tier only.",
           "- The hybrid's ensembles are under-dispersed (spread/RMSE 0.47–0.74); pure-noise SDA is the "
           "best calibrated scheme but the weakest in score off the reference case.", ""]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
