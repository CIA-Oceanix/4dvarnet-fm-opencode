"""Sections 1-4 and the cross-family reading of ``p1_l96_benchmark.md``, under the
benchmark-default evaluation framework.

Every row is scored by the extended report's own code
(``generate_l96_benchmark_extended_report.da_main`` / ``learned_main``, on the
``benchmark_extended`` input bundle), so the P1 tables and ``l96_benchmark_extended.md``
cannot disagree: per-window RMSE on the 24 observed channels of the 200 P1 test windows,
on the regular 30-obs set and on the canonical random observing system, S0 and S1, seeds
pooled per window. The P1-protocol checkpoints (fixed observing system, 400 epochs, one
seed, S+/M/L tiers) are the report's annex, rendered by the P1 generator itself.
"""
from __future__ import annotations

import json

import numpy as np

import _inputs

TAU0 = _inputs.root("p1_benchmark", "here") / "tau0_1200_2026-10-02"
TAU0_SETS = (("reg", "regular"), ("can", "rlayout"))
CASES = ("s0", "s1")
B1200 = "Benchmark default (1200 ep)"
B400 = "Benchmark recipe, 400 ep"

# (section key, label shown, extended-report (group, label), parameters, flag). An empty flag is a
# benchmark row; a flagged row is a sensitivity row, kept out of the bests.
DA_ROWS = (("ETKF", "ETKF", "λ 1.15 / 2.5", ""),
           ("EnKF", "EnKF", "λ 1.2 / 3.0", ""),
           ("ETKS", "ETKS", "λ 1.15 / 2.5", ""),
           ("Strong-4DVar", "Strong-4DVar", "—", ""),
           ("ETKS, 100 members, inflation S0 1.05 / S1 2.5 (ensemble-size sensitivity)", "ETKS, 100 members",
            "λ 1.05 / 2.5", "ensemble size: 100 members (the benchmark and the learned ensembles use 30), "
            "inflation re-selected on the validation windows at N=100"))
LEARNED = {
    "det": ((B1200, "DirectUNet-M", "DirectUNet-M", "5.89 M, 1200 ep", ""),
            (B400, "DirectUNet-M", "DirectUNet-M, 400 ep", "5.89 M, 400 ep", "training budget: 400 epochs"),
            ("Benchmark recipe, 400 ep, 3000 windows", "DirectUNet-M", "DirectUNet-M, 400 ep, 3000 windows",
             "5.89 M, 400 ep", "training data: 3000 windows (benchmark: 1000), same gradient steps as 1200 ep")),
    "fm": ((B1200, "PredictStateCFM-M", "PredictStateCFM-M", "5.89 M, 1200 ep", ""),
           (B1200, "VanillaCFM-M", "VanillaCFM-M", "5.89 M, 1200 ep", ""),
           (B400, "PredictStateCFM-M", "PredictStateCFM-M, 400 ep", "5.89 M, 400 ep", "training budget: 400 epochs"),
           (B400, "VanillaCFM-M", "VanillaCFM-M, 400 ep", "5.89 M, 400 ep", "training budget: 400 epochs"),
           ("Flow ensemble, 1200 ep (3 networks)", "PredictStateCFM-M x3", "PredictStateCFM-M x3", "3 x 5.89 M",
            "velocity average of the three seeds: 3x the parameters and sampling cost")),
    "sda": (("SDA, 1200 ep, gw 25", "SDA1-M", "SDA1-M", "5.89 M, gw 25", ""),
            ("SDA, 1200 ep, gw 25", "SDA2-M", "SDA2-M", "5.89 M, gw 25", ""),
            ("SDA, 1200 ep, gw 25", "SDA3-fix-M", "SDA3-fix-M", "5.89 M, gw 25", ""),
            ("Hybrid, 1200-ep SDA prior (tau0 0.1, gw 2)", "DirectUNet-M(1200 ep) -> SDA3-fix-M",
             "Hybrid DirectUNet-M -> SDA3-fix-M", "2 x 5.89 M, tau0 0.1, gw 2",
             "two families: SDA3-fix-M guided sampling started from the DirectUNet-M estimate (reference)")),
}
TITLES = {"da": "1. DA baselines", "det": "2. Deterministic point estimators", "fm": "3. Flow matching",
          "sda": "4. SDA (score-based prior + guidance)"}


def collect(cache: dict) -> tuple[dict, list[str]]:
    """Rows of sections 1-4 under the benchmark framework, and the consistency errors."""
    import generate_l96_benchmark_extended_report as ext
    da = {r["label"]: r for r in ext.da_main()}
    learned, errors = ext.learned_main(cache)
    errors += ext.check_etks100()
    by = {(r["group"], r["label"]): r for r in learned}
    out = {"da": []}
    for key, label, params, flag in DA_ROWS:
        r = da[key]
        out["da"].append({"label": label, "params": params, "flag": flag, "seeds": "—", "reg": r["reg"],
                          "can": r["can"], "vr": None})
    for sec, specs in LEARNED.items():
        out[sec] = []
        for group, key, label, params, flag in specs:
            r = by[(group, key)]
            vr = None
            if sec == "det" and r["reg"] is not None and r["can"] is not None:
                vr = {t: float(np.mean([ext.var_ratio(d, cache) for d in r["dirs"][t]])) for t in ("reg", "can")}
            out[sec].append({"label": label, "params": params, "flag": flag,
                             "seeds": f"{r.get('n_reg', 0)}/{r['n_total']}", "reg": r["reg"], "can": r["can"], "vr": vr})
    return out, errors


def tau0_rows() -> list[dict] | None:
    """tau=0 means of the 1200-epoch flows: per family, seed-mean of the per-run window-mean RMSE."""
    files = {(t, c): TAU0 / f"tau0_{name}_{c}.json" for t, name in TAU0_SETS for c in CASES}
    if not all(f.exists() for f in files.values()):
        return None
    out = {}
    for (t, c), f in files.items():
        for r in json.loads(f.read_text())["rows"]:
            fam = r["label"].rsplit(" seed", 1)[0]
            rec = out.setdefault(fam, {})
            rec.setdefault((t, c), []).append(r)
    rows = []
    for fam, cells in out.items():
        rows.append({"label": f"{fam.replace(' M', '-M')} (tau=0)",
                     "rmse": {k: [r["rmse"]["mean"] for r in v] for k, v in cells.items()},
                     "mae": float(np.mean([r["crps_mae"]["mean"] for r in cells[("reg", "s0")]])),
                     "draws": float(np.mean([r["draw_dispersion"] for r in cells[("reg", "s0")]]))})
    return rows


def _ms(a) -> str:
    return f"{np.mean(a):.3f} ± {np.std(a, ddof=1):.3f}"


def _f(x, d=3) -> str:
    return "—" if x is None else f"{x:.{d}f}"


def mean(r: dict, t: str, c: str) -> float | None:
    return None if r[t] is None else float(np.mean(r[t][c]["rmse"]))


def table(sec: str, rows: list[dict]) -> list[str]:
    det = sec == "det"
    extra = "var ratio regular / random (S0)" if det else "spread/RMSE regular / random (S0)"
    A = [f"\n## {TITLES[sec]}\n",
         f"| scheme | params | seeds | regular S0 | regular S1 | random S0 | random S1 | S1/S0 (regular) | "
         f"CRPS regular S0 (S1) | CRPS random S0 (S1) | {extra} | note |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ok = [r for r in rows if not r["flag"] and r["reg"] is not None and r["can"] is not None]
    best = {(t, c): min((mean(r, t, c) for r in ok), default=None) for t in ("reg", "can") for c in CASES}
    for r in rows:
        if r["reg"] is None or r["can"] is None:
            A.append(f"| {r['label']} | {r['params']} | {r['seeds']} | pending | pending | pending | pending | | | | | {r['flag']} |")
            continue
        cell = {}
        for t in ("reg", "can"):
            for c in CASES:
                b = "**" if not r["flag"] and best[(t, c)] is not None and abs(mean(r, t, c) - best[(t, c)]) < 1e-12 else ""
                cell[(t, c)] = f"{b}{_ms(r[t][c]['rmse'])}{b}"
        crps = {t: f"{_f(r[t]['s0']['crps'])} ({_f(r[t]['s1']['crps'])})" for t in ("reg", "can")}
        if det:
            ex = f"{r['vr']['reg']:.3f} / {r['vr']['can']:.3f}" if r["vr"] else "—"
            crps = {t: "—" for t in crps}
        else:
            ex = f"{_f(r['reg']['s0']['sp'], 2)} / {_f(r['can']['s0']['sp'], 2)}"
        lab = f"*{r['label']}*" if r["flag"] else r["label"]
        A.append(f"| {lab} | {r['params']} | {r['seeds']} | {cell[('reg', 's0')]} | {cell[('reg', 's1')]} | "
                 f"{cell[('can', 's0')]} | {cell[('can', 's1')]} | {mean(r, 'reg', 's1') / mean(r, 'reg', 's0'):.2f} | "
                 f"{crps['reg']} | {crps['can']} | {ex} | {r['flag']} |")
    return A


def tau0_table(rows: list[dict] | None, flows: list[dict]) -> list[str]:
    A = ["\n### The flows' tau=0 mean components, as deterministic estimators\n"]
    if rows is None:
        return A + [f"_pending: `{TAU0}`_\n"]
    full = {r["label"]: r for r in flows if not r["flag"]}
    A += ["`mu(x0, tau=0, y)` averaged over 30 draws of the 1200-epoch flows, seeds 1-3 (each cell: mean over "
          "seeds of the window-mean RMSE ± seed sd). At tau=0, `x_tau = x0` is independent of `x1`, so this is the "
          "flow's own posterior mean `E[x1|y]` used as a point estimator; it costs 30 model calls against "
          "DirectUNet's 1. `full` = the flow's ensemble-mean RMSE (section 3), regular S0.\n",
          "| scheme | regular S0 | regular S1 | random S0 | random S1 | MAE regular S0 | draw dispersion | full (regular S0) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: np.mean(x["rmse"][("reg", "s0")])):
        fam = r["label"].replace(" (tau=0)", "")
        cells = " | ".join(f"{np.mean(r['rmse'][(t, c)]):.3f} ± {np.std(r['rmse'][(t, c)], ddof=1):.3f}"
                           for t in ("reg", "can") for c in CASES)
        f = f"{mean(full[fam], 'reg', 's0'):.3f}" if fam in full else "—"
        A.append(f"| {r['label']} | {cells} | {r['mae']:.3f} | {r['draws']:.3f} | {f} |")
    A.append("\n`draw dispersion` is the across-draw scatter of `m` itself (regular S0); 0 would mean fully "
             "deterministic in `y`.\n")
    return A


def reading(rows: dict, tau0: list[dict] | None, annex_best: dict) -> list[str]:
    def best(sec, t="reg", c="s0"):
        ok = [r for r in rows[sec] if not r["flag"] and r[t] is not None]
        return min(ok, key=lambda r: mean(r, t, c)) if ok else None
    b = {s: best(s) for s in ("da", "det", "fm", "sda")}
    bc = {s: best(s, "can") for s in ("da", "det", "fm", "sda")}
    if any(v is None for v in list(b.values()) + list(bc.values())):
        return ["\n## Cross-family reading\n", "_pending: rows missing_\n"]
    hyb = next(r for r in rows["sda"] if r["label"].startswith("Hybrid"))
    det, fm, sda, da = (mean(b[s], "reg", "s0") for s in ("det", "fm", "sda", "da"))
    A = ["\n## Cross-family reading\n",
         f"Best regular / random S0 of each family: deterministic {b['det']['label']} {det:.3f} / "
         f"{mean(bc['det'], 'can', 's0'):.3f}, flow matching {b['fm']['label']} {fm:.3f} / {mean(bc['fm'], 'can', 's0'):.3f}, "
         f"SDA {b['sda']['label']} {sda:.3f} / {mean(bc['sda'], 'can', 's0'):.3f}, DA {b['da']['label']} {da:.3f} / "
         f"{mean(bc['da'], 'can', 's0'):.3f}.\n"]
    gap = 100 * (fm - det) / det
    sds = [float(np.std(b[s]["reg"]["s0"]["seed_means"], ddof=1)) for s in ("det", "fm")]
    within = "within" if abs(fm - det) <= max(sds) else "outside"
    sp = b["fm"]["reg"]["s0"]["sp"]
    A.append(f"- **Deterministic and flow matching are level under the benchmark recipe**: the best flow is "
             f"{abs(gap):.1f}% {'behind' if gap > 0 else 'ahead of'} DirectUNet on the regular set ({fm:.3f} vs "
             f"{det:.3f}), {within} the seed sd ({max(sds):.3f}). What the flows add is an ensemble (CRPS "
             f"{b['fm']['reg']['s0']['crps']:.3f}), under-dispersed (spread/RMSE {sp:.2f}), not a better mean. "
             "Under the P1 protocol (annex A: fixed observing "
             f"system, frozen per-window noise, 400 epochs) the best flow led DirectUNet by "
             f"{100 * (annex_best['det'] - annex_best['fm']) / annex_best['det']:.0f}% ({annex_best['fm']:.3f} vs "
             f"{annex_best['det']:.3f}): DirectUNet overfit the frozen noise, so that gap is a training-protocol "
             "artefact, not a property of flow matching.")
    A.append(f"- **SDA trails by {100 * (sda - min(det, fm)) / min(det, fm):.0f}%** at S0 regular as a stand-alone "
             f"scheme, but its guided sampler is the best refinement of a DirectUNet estimate: the hybrid reaches "
             f"{mean(hyb, 'reg', 's0'):.3f} / {mean(hyb, 'can', 's0'):.3f} (regular / random S0), the best of every row.")
    da_r = [mean(r, "reg", "s1") / mean(r, "reg", "s0") for r in rows["da"] if not r["flag"]]
    le_r = [mean(r, "reg", "s1") / mean(r, "reg", "s0") for s in ("det", "fm", "sda") for r in rows[s]
            if not r["flag"] and r["reg"] is not None]
    e100 = next(r for r in rows["da"] if r["flag"])
    lb, lb1 = min(det, fm), min(mean(b[s], "reg", "s1") for s in ("det", "fm"))
    A.append(f"- **The S1/S0 ratio separates the two worlds.** Learned schemes stay flat under model error "
             f"({min(le_r):.2f}-{max(le_r):.2f}) because they never use a forward model; the DA baselines degrade "
             f"{min(da_r):.1f}-{max(da_r):.1f}x, since their forward operator carries the bias. At S0 the best DA "
             f"(ETKS {da:.3f}) is {100 * (da - lb) / lb:.0f}% behind the best learned scheme with 30 members, "
             f"{100 * (mean(e100, 'reg', 's0') - lb) / lb:.0f}% with 100 members (flagged row); at S1 its RMSE is "
             f"{mean(e100, 'reg', 's1') / lb1:.1f}x the best learned scheme's at either size.")
    if tau0:
        t = min(tau0, key=lambda r: np.mean(r["rmse"][("reg", "s0")]))
        tv = float(np.mean(t["rmse"][("reg", "s0")]))
        fam = t["label"].replace(" (tau=0)", "")
        full = next((mean(r, "reg", "s0") for r in rows["fm"] if r["label"] == fam), None)
        A.append(f"- **The flows' tau=0 mean as a point estimator**: the best ({t['label']}) scores {tv:.3f} at "
                 f"regular S0, against {_f(full)} for the same flow's full sampler and {det:.3f} for DirectUNet "
                 f"({100 * (tv - det) / det:+.0f}%), at 30 model calls against 1. "
                 + ("Under the benchmark recipe the accuracy is carried by the sampler, not by the tau=0 head "
                    "(the P1-protocol flows of annex A.1 had the opposite ordering against DirectUNet)."
                    if full is not None and tv > full else ""))
    A.append("- Tier (S+ / M / L), CFM parameterization and SDA conditioning comparisons exist only under the P1 "
             "protocol: annex A.\n")
    return A
