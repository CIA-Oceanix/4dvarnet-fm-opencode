"""Sections 1-4 and the cross-family reading of ``p1_l96_benchmark.md``, under the
benchmark-default evaluation framework.

Every row is scored by the extended report's own code
(``generate_l96_benchmark_extended_report.da_main`` / ``learned_main``, on the
``benchmark_extended`` input bundle), so the P1 tables and ``l96_benchmark_extended.md``
cannot disagree: per-window RMSE on the 24 observed channels of the 200 P1 test windows,
on the regular 30-obs set and on the canonical random observing system, S0 and S1, seeds
pooled per window. Every U-Net row is the 2D (time x sector) circular backbone (2026-10-21);
rows whose 2D checkpoint is not trained yet render as pending, single-seed rows carry a note.
The annex is the 2D S / M / L tiers.
"""
from __future__ import annotations

import numpy as np

CASES = ("s0", "s1")

# (section key, label shown, extended-report (group, label), parameters, flag). An empty flag is a
# benchmark row; a flagged row is a sensitivity row, kept out of the bests.
DA_ROWS = (("ETKF", "ETKF", "λ 1.15 / 2.5", ""),
           ("EnKF", "EnKF", "λ 1.2 / 3.0", ""),
           ("ETKS", "ETKS", "λ 1.15 / 2.5", ""),
           ("Strong-4DVar", "Strong-4DVar", "—", ""),
           ("Weak-4DVar (model-error scale q S0 0.03 / S1 0.3)", "Weak-4DVar", "q 0.03 / 0.3, 40 it", ""),
           ("ETKS, 100 members, inflation S0 1.05 / S1 2.5 (ensemble-size sensitivity)", "ETKS, 100 members",
            "λ 1.05 / 2.5", "ensemble size: 100 members (the benchmark and the learned ensembles use 30), "
            "inflation re-selected on the validation windows at N=100"))
G2D, G2D_SDA, G2D_HYB, G2D_TIER = ("2D backbone, 1200 ep", "2D SDA, 1200 ep, gw 25", "2D hybrid (tau0 0.1, gw 1)",
                                   "2D tiers, 1200 ep (seed 1)")
PENDING = "not trained yet"
SEED1 = "seed 1 only: seeds 2-3 not trained yet"
HYB_REF = "two families: SDA guided sampling started from the DirectUNet-2D-M estimate (reference)"
# Every U-Net row is the 2D (time x sector) circular backbone; the 1D rows stay in l96_benchmark_extended.md.
LEARNED = {
    "det": ((G2D, "DirectUNet-2D-M", "DirectUNet-2D-M", "5.63 M, 1200 ep", ""),),
    "fm": ((G2D, "PredictStateCFM-2D-M", "PredictStateCFM-2D-M", "1200 ep", ""),
           (G2D, "VanillaCFM-2D-M", "VanillaCFM-2D-M", "5.64 M, 1200 ep", "")),
    "sda": ((G2D_SDA, "SDA1-2D-M", "SDA1-2D-M", "5.63 M, gw 25", ""),
            (G2D_SDA, "SDA2-2D-M", "SDA2-2D-M", "gw 25", ""),
            (G2D_SDA, "SDA3-fix-2D-M", "SDA3-fix-2D-M", "gw 25", ""),
            (G2D_HYB, "DirectUNet-2D-M -> SDA3-fix-2D-M", "Hybrid DirectUNet-2D-M -> SDA3-fix-2D-M", "tau0 0.1, gw 1", HYB_REF),
            (G2D_HYB, "DirectUNet-2D-M -> SDA1-2D-M", "Hybrid DirectUNet-2D-M -> SDA1-2D-M", "2 x 5.63 M, tau0 0.1, gw 1",
             HYB_REF + "; SDA1 prior until SDA3-fix-2D-M is trained"),
            (G2D_HYB, "DirectUNet-2D-M -> SDA3-fix-M (1D prior)", "Hybrid DirectUNet-2D-M -> SDA3-fix-M (1D prior)",
             "5.63 M + 5.89 M, tau0 0.1, gw 1",
             "interim: the 1D SDA3-fix-M prior started from the 2D DirectUNet, until SDA3-fix-2D-M is trained")),
}
TIERS = (("DirectUNet-2D", {"S": (G2D_TIER, "DirectUNet-2D-S", "2.03 M"), "M": (G2D, "DirectUNet-2D-M", "5.63 M"),
                             "L": (G2D_TIER, "DirectUNet-2D-L", "22.5 M")}),
         ("VanillaCFM-2D", {"S": (G2D_TIER, "VanillaCFM-2D-S", "2.03 M"), "M": (G2D, "VanillaCFM-2D-M", "5.64 M"),
                            "L": (G2D_TIER, "VanillaCFM-2D-L", "22.5 M")}),
         ("SDA1-2D (gw 25)", {"S": (G2D_TIER, "SDA1-2D-S", "2.03 M"), "M": (G2D_SDA, "SDA1-2D-M", "5.63 M"),
                              "L": (G2D_TIER, "SDA1-2D-L", "22.5 M")}),
         ("Hybrid DirectUNet-2D-M -> SDA1-2D (gw 1)", {"M": (G2D_HYB, "DirectUNet-2D-M -> SDA1-2D-M", "2 x 5.63 M"),
                                                       "L": (G2D_TIER, "DirectUNet-2D-M -> SDA1-2D-L", "5.63 M + 22.5 M")}))
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
    if "Weak-4DVar (model-error scale q S0 0.03 / S1 0.3)" != ext.WEAK4DVAR_LABEL:
        errors.append("p1_benchmark_sections.DA_ROWS: Weak-4DVar label differs from the extended report's")
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
            n = r.get("n_reg", 0)
            note = "; ".join(x for x in (flag, PENDING if n == 0 else SEED1 if n < r["n_total"] else "") if x)
            out[sec].append({"label": label, "params": params, "flag": flag, "note": note,
                             "seeds": f"{n}/{r['n_total']}", "reg": r["reg"], "can": r["can"], "vr": vr})
    out["tiers"] = tier_rows(by, cache)
    return out, errors


def tier_rows(by: dict, cache: dict) -> list[dict]:
    """S / M / L of each 2D family, seed 1 (the M cell is seed 1 of the 3-seed row)."""
    import generate_l96_benchmark_extended_report as ext
    rows = []
    for fam, tiers in TIERS:
        for t, (group, label, params) in tiers.items():
            r = by[(group, label)]
            first = {tg: next((d for d in r["dirs"][tg] if "seed1" in str(d) or "Ms1_" in str(d)), None) for tg in ("reg", "can")}
            if first["reg"] is None or first["can"] is None:
                continue
            sc = {tg: ext.score_dir(first[tg], cache) for tg in ("reg", "can")}
            rows.append({"family": fam, "tier": t, "params": params,
                         "rmse": {(tg, c): float(np.mean(sc[tg][c]["rmse"])) for tg in ("reg", "can") for c in CASES},
                         "crps": {(tg, c): sc[tg][c]["crps"] for tg in ("reg", "can") for c in CASES}})
    return rows


def tier_table(rows: list[dict]) -> list[str]:
    A = ["\n## Annex A. 2D tiers (S / M / L, seed 1)\n",
         "S / M / L widths of the 2D backbone (the widths of the 1D S / M / L tiers), benchmark "
         "recipe, seed 1 of each (the M row is seed 1 of the 3-seed row above, so it can differ from the section tables). "
         "Per-window RMSE, window mean; CRPS for the ensembles. The 1D P1-protocol annex (tiers, CFM parameterization, SDA "
         "conditioning under a fixed observing system) is superseded and no longer rendered; its 1D rows remain in git "
         "history and the 1D benchmark rows in `l96_benchmark_extended.md`.\n",
         "| family | tier | params | regular S0 | regular S1 | random S0 | random S1 | CRPS regular S0 | CRPS random S0 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        best = min(x["rmse"][("reg", "s0")] for x in rows if x["family"] == r["family"])
        b = "**" if r["rmse"][("reg", "s0")] == best else ""
        cells = " | ".join(f"{r['rmse'][(t, c)]:.3f}" for t in ("reg", "can") for c in CASES)
        A.append(f"| {r['family']} | {b}{r['tier']}{b} | {r['params']} | {cells} | {_f(r['crps'][('reg', 's0')])} | "
                 f"{_f(r['crps'][('can', 's0')])} |")
    return A


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
    ok = [r for r in rows if not r["flag"]]
    best = {(t, c): min((mean(r, t, c) for r in ok if r[t] is not None), default=None) for t in ("reg", "can") for c in CASES}
    for r in rows:
        if r["reg"] is None or (r["can"] is None and sec != "da"):
            A.append(f"| {r['label']} | {r['params']} | {r['seeds']} | pending | pending | pending | pending | | | | | {r.get('note', r['flag'])} |")
            continue
        cell = {}
        for t in ("reg", "can"):
            for c in CASES:
                if r[t] is None:
                    cell[(t, c)] = "—"
                    continue
                b = "**" if not r["flag"] and best[(t, c)] is not None and abs(mean(r, t, c) - best[(t, c)]) < 1e-12 else ""
                cell[(t, c)] = f"{b}{_ms(r[t][c]['rmse'])}{b}"
        crps = {t: f"{_f(r[t]['s0']['crps'])} ({_f(r[t]['s1']['crps'])})" if r[t] is not None else "—" for t in ("reg", "can")}
        if det:
            ex = f"{r['vr']['reg']:.3f} / {r['vr']['can']:.3f}" if r["vr"] else "—"
            crps = {t: "—" for t in crps}
        else:
            ex = f"{_f(r['reg']['s0']['sp'], 2)} / {_f(r['can']['s0']['sp'], 2) if r['can'] is not None else '—'}"
        lab = f"*{r['label']}*" if r["flag"] else r["label"]
        A.append(f"| {lab} | {r['params']} | {r['seeds']} | {cell[('reg', 's0')]} | {cell[('reg', 's1')]} | "
                 f"{cell[('can', 's0')]} | {cell[('can', 's1')]} | {mean(r, 'reg', 's1') / mean(r, 'reg', 's0'):.2f} | "
                 f"{crps['reg']} | {crps['can']} | {ex} | {r.get('note', r['flag'])} |")
    return A


def tau0_table(rows: list[dict] | None, flows: list[dict]) -> list[str]:
    """The tau=0 mean components were measured on the 1D flows only; the 2D probe is not run yet."""
    del rows, flows
    return ["\n### The flows' tau=0 mean components, as deterministic estimators\n",
            "_pending for the 2D flows: `mu(x0, tau=0, y)` has only been measured on the 1D flows "
            "(`reports/l96/eval_mean_component.py`, git history of this report)._\n"]


def reading(rows: dict) -> list[str]:
    def best(sec, t="reg", c="s0"):
        ok = [r for r in rows[sec] if not r["flag"] and r[t] is not None]
        return min(ok, key=lambda r: mean(r, t, c)) if ok else None
    b = {s: best(s) for s in ("da", "det", "fm", "sda")}
    bc = {s: best(s, "can") for s in ("da", "det", "fm", "sda")}
    if any(v is None for v in list(b.values()) + list(bc.values())):
        return ["\n## Cross-family reading\n", "_pending: rows missing_\n"]
    det, fm, sda, da = (mean(b[s], "reg", "s0") for s in ("det", "fm", "sda", "da"))
    alone = [r for r in rows["sda"] if not r["label"].startswith("Hybrid") and r["reg"] is not None]
    sa = min(alone, key=lambda r: mean(r, "reg", "s0"))
    hyb = [r for r in rows["sda"] if r["label"].startswith("Hybrid") and r["reg"] is not None and r["can"] is not None]
    seeds = lambda r: f"{r['seeds'].split('/')[0]} seed{'s' if r['seeds'][0] != '1' else ''}"  # noqa: E731
    A = ["\n## Cross-family reading\n",
         "Every learned row is the 2D (time x sector) circular backbone. "
         f"Best regular / random S0 of each family: deterministic {b['det']['label']} {det:.3f} / "
         f"{mean(bc['det'], 'can', 's0'):.3f} ({seeds(b['det'])}), flow matching {b['fm']['label']} {fm:.3f} / "
         f"{mean(bc['fm'], 'can', 's0'):.3f} ({seeds(b['fm'])}), SDA {b['sda']['label']} {sda:.3f} / "
         f"{mean(bc['sda'], 'can', 's0'):.3f} ({seeds(b['sda'])}), DA {b['da']['label']} {da:.3f} / {mean(bc['da'], 'can', 's0'):.3f}.\n"]
    gap = 100 * (fm - det) / det
    A.append(f"- **Flow matching vs the deterministic U-Net**: the best 2D flow is {abs(gap):.1f}% "
             f"{'behind' if gap > 0 else 'ahead of'} DirectUNet-2D-M on the regular set ({fm:.3f} vs {det:.3f}; the flow "
             f"is {seeds(b['fm'])}), and on the random set {mean(bc['fm'], 'can', 's0'):.3f} vs {mean(bc['det'], 'can', 's0'):.3f}. "
             f"It also brings an ensemble (CRPS {b['fm']['reg']['s0']['crps']:.3f}, per-window spread/RMSE "
             f"{b['fm']['reg']['s0']['sp']:.2f}). With the 1D backbone the two were level (0.351 / 0.340, "
             "`l96_benchmark_extended.md`).")
    A.append(f"- **SDA alone trails** ({sa['label']} {mean(sa, 'reg', 's0'):.3f} regular S0, "
             f"{100 * (mean(sa, 'reg', 's0') - min(det, fm)) / min(det, fm):.0f}% behind the best amortised scheme), "
             "but the guided sampler refines a DirectUNet-2D estimate: "
             + "; ".join(f"{r['label']} {mean(r, 'reg', 's0'):.3f} / {mean(r, 'can', 's0'):.3f} ({seeds(r)})" for r in hyb)
             + " (regular / random S0). SDA2-2D and SDA3-fix-2D, and hence the paper's DirectUNet -> SDA3-fix hybrid, "
             "are not trained yet.")
    da_r = [mean(r, "reg", "s1") / mean(r, "reg", "s0") for r in rows["da"] if not r["flag"]]
    le_r = [mean(r, "reg", "s1") / mean(r, "reg", "s0") for s in ("det", "fm", "sda") for r in rows[s]
            if not r["flag"] and r["reg"] is not None]
    e100 = next(r for r in rows["da"] if r["flag"])
    allbest = min(mean(r, "reg", "s0") for s in ("det", "fm", "sda") for r in rows[s] if r["reg"] is not None and not r["flag"])
    allbest1 = min(mean(r, "reg", "s1") for s in ("det", "fm", "sda") for r in rows[s] if r["reg"] is not None and not r["flag"])
    A.append(f"- **The S1/S0 ratio separates the two worlds.** Learned schemes stay flat under model error "
             f"({min(le_r):.2f}-{max(le_r):.2f}); the DA baselines degrade {min(da_r):.1f}-{max(da_r):.1f}x. At S0 the "
             f"best DA (ETKS {da:.3f}) is {100 * (da - allbest) / allbest:.0f}% behind the best learned row with 30 "
             f"members, {100 * (mean(e100, 'reg', 's0') - allbest) / allbest:.0f}% with 100 members (flagged row); at "
             f"S1 its RMSE is {mean(e100, 'reg', 's1') / allbest1:.1f}x the best learned row's.")
    A.append("- Tiers (S / M / L) of the 2D families: annex A. PredictStateCFM, SDA2 and SDA3-fix have no 2D "
             "checkpoint yet; their rows stay pending.\n")
    return A
