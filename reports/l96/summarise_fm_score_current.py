"""FM-operator score tables for the current L96 benchmark rows, with window-bootstrap intervals.

Reads the per-window ``*.npz`` of ``probe_fm_score_eval.py`` (learned rows),
``probe_fm_score_stored.py`` (DirectUNet, Strong-4DVar, benchmark-file
ETKF/EnKF/ETKS, Gaussian) and ``probe_fm_score_da.py`` (ETKF/ETKS ensembles
re-run with members). Seeds are averaged per window, then every statistic gets a
95% percentile bootstrap over the 200 test windows; comparisons are paired.

FMS is reported in physical units (sigma_c^2-weighted mean over channels); RMSE
keeps the benchmark convention (mean over windows x channels of the per-window
per-channel RMSE); spread/skill is the pooled ratio in physical units.

usage: python reports/l96/summarise_fm_score_current.py IN_DIR OUT_MD [OUT_JSON]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.bootstrap import bootstrap_ci, mean_stat, paired_ci, ratio_stat  # noqa: E402
from reports.l96.probe_fm_score import SCALE_TAUS, SCALES, TAUS, norm_stats  # noqa: E402

ROWS = [
    ("DA", "ETKF (λ 1.15 / 2.5)", r"dastored_{lay}_ETKF_{case}"),
    ("DA", "EnKF (λ 1.2 / 3.0)", r"dastored_{lay}_EnKF_{case}"),
    ("DA", "ETKS (λ 1.15 / 2.5)", r"dastored_{lay}_ETKS_{case}"),
    ("DA", "Strong-4DVar", r"dastored_{lay}_Strong_4DVar_{case}"),
    ("DirectUNet (1200 ep)", "DirectUNet-M", r"directunet_ep1200_seed\d_{lay}_{case}"),
    ("CFM (1200 ep)", "PredictStateCFM-M", r"predictstatecfm_ep1200_seed\d_{lay}_{case}"),
    ("CFM (1200 ep)", "VanillaCFM-M", r"vanillacfm_ep1200_seed\d_{lay}_{case}"),
    ("SDA (1200 ep, gw 25)", "SDA1-M", r"B4_sda1_monaiM_ep1200_l96_seed\d_{lay}_{case}"),
    ("SDA (1200 ep, gw 25)", "SDA2-M", r"A3_sda2_monaiM_ep1200_l96_seed\d_{lay}_{case}"),
    ("SDA (1200 ep, gw 25)", "SDA3-fix-M", r"A3_sda3fix_monaiM_ep1200_l96_seed\d_{lay}_{case}"),
    ("*reference*", "Hybrid DirectUNet-M → SDA3-fix-M", r"hybrid_DU1200s\d_sda3fix1200s1_{lay}_{case}"),
]
MEMBER_ROWS = [
    ("ETKF (members)", r"da_{lay}_.*_{case}_ETKF_members"),
    ("ETKS (members)", r"da_{lay}_.*_{case}_ETKS_members"),
]
REFERENCE = "PredictStateCFM-M"
COLUMNS = [("reg", "s0"), ("reg", "s1"), ("rand", "s0"), ("rand", "s1")]
COL_TITLES = ["regular S0", "regular S1", "random S0", "random S1"]
ALL = 2


def load(in_dir: Path, pattern: str) -> tuple[dict | None, int]:
    rx = re.compile("^" + pattern + "$")
    files = sorted(f for f in in_dir.glob("*.npz") if rx.match(f.stem))
    if not files:
        return None, 0
    arrs = [dict(np.load(f)) for f in files]
    return {k: np.mean([a[k].astype(np.float64) for a in arrs], axis=0) for k in arrs[0]}, len(files)


def stats_for(a: dict, w2: np.ndarray) -> dict:
    """Bootstrap statistics of one row (seeds already averaged per window)."""
    W = a["se"].shape[0]
    out = {"rmse": bootstrap_ci(mean_stat(a["rmse_phys"]), W)}
    for i, tau in enumerate(TAUS):
        out[f"fms{tau}"] = bootstrap_ci(mean_stat(a["fms_gauss"][:, :, i] * w2), W)
    out["spread_skill"] = bootstrap_ci(ratio_stat(a["var"] * w2, a["se"] * w2, sqrt=True), W)
    sc = a["scale_phys"][:, ALL]
    for i, tau in enumerate(SCALE_TAUS):
        if tau not in (0.5, 0.75):
            continue
        def loss(idx: np.ndarray, i: int = i) -> float:
            curve = sc[idx, i].mean(axis=0)
            return float(100 * (curve[20] - curve[:-1].min()) / curve[20])
        out[f"closs{tau}"] = bootstrap_ci(loss, W)
        out[f"cstar{tau}"] = float(SCALES[int(sc[:, i].mean(axis=0)[:-1].argmin())])
        if "fms_kde" in a:
            j = TAUS.index(tau)
            kde = (a["fms_kde"][:, :, j] * w2).mean(axis=1)
            gau = sc[:, i, -1]
            out[f"nongauss{tau}"] = bootstrap_ci(lambda idx, k=kde, g=gau: float(100 * (k[idx].sum() / g[idx].sum() - 1)), W)
    return out


def fmt(t: tuple[float, float, float], f: str = ".4f") -> str:
    return f"{format(t[0], f)} [{format(t[1], f)}, {format(t[2], f)}]"


def ranked(cells: list[tuple[float, str] | None]) -> list[str]:
    vals = sorted({c[0] for c in cells if c is not None})
    out = []
    for c in cells:
        if c is None:
            out.append("—")
        elif c[0] == vals[0]:
            out.append(f"**{c[1]}**")
        elif len(vals) > 1 and c[0] == vals[1]:
            out.append(f"*{c[1]}*")
        else:
            out.append(c[1])
    return out


def build(in_dir: Path, heading: str = "##") -> tuple[list[str], dict]:
    """The markdown section and the {(scheme, layout, case): statistics} table."""
    _, sd = norm_stats()
    w2 = sd ** 2
    table, raw, seeds = {}, {}, {}
    for fam, name, pat in ROWS + [("DA (members)", n, p) for n, p in MEMBER_ROWS]:
        for lay, case in COLUMNS:
            a, n = load(in_dir, pat.format(lay=lay, case=case))
            if a is None:
                continue
            raw[(name, lay, case)] = a
            table[(name, lay, case)] = stats_for(a, w2)
            seeds[name] = n
    L = [f"{heading} Distributional score FMS_τ (current benchmark)", "",
         "FMS_τ(q) = E‖E_q[x₁ | x_τ] − x₁*‖² on the path x_τ = τ x₁ + (1−τ) x₀ (x₀ ~ N(0, I) in the z-scored "
         "space), reported in **physical units²** (σ_c²-weighted over the 24 observed channels). τ = 0 is the MSE "
         "of the ensemble mean; each τ ∈ (0, 1) is strictly proper; for a Gaussian forecast it is minimised at "
         "spread² = MSE at every τ, so τ > 0 rewards calibration; a point estimate scores its MSE at every τ. "
         "All rows use the Gaussian form (per-element N(mean, var)), so DA, flows and SDA are scored alike. "
         "Seeds are averaged per window; every cell is the value on the 200 test windows with its **95% window-"
         "bootstrap interval** (2000 replicates). Per column, best **bold**, second *italic*. τ = 0.95 is omitted "
         "(dominated by the universal 1/snr floor). Code: `reports/l96/probe_fm_score*.py`, "
         "`reports/l96/summarise_fm_score_current.py`; notes: `docs/results/l96_p1_fm_score.md`.", ""]
    for i, tau in enumerate(TAUS[:4]):
        L += [f"**FMS at τ = {tau}** (lower is better{'; = MSE' if tau == 0 else ''})", "",
              "| family | scheme | seeds | " + " | ".join(COL_TITLES) + " |", "|---|---|---|" + "---|" * 4]
        cols = []
        for lay, case in COLUMNS:
            cols.append([(table[(n, lay, case)][f"fms{tau}"][0], fmt(table[(n, lay, case)][f"fms{tau}"]))
                         if (n, lay, case) in table else None for _, n, _ in ROWS])
        cells = list(zip(*[ranked(c) for c in cols]))
        for (fam, name, _), row in zip(ROWS, cells):
            if all(c == "—" for c in row):
                continue
            L.append(f"| {fam} | {name} | {seeds.get(name, '—') if fam != 'DA' else '—'} | " + " | ".join(row) + " |")
        L.append("")
    L += ["**Calibration**: pooled spread/skill (physical units; calibrated √(N/(N+1)) = 0.984 for N = 30) and "
          "calibration loss = share of FMS removed by the best scalar rescaling c·var (c* in brackets; refitted "
          "in every bootstrap replicate). Point estimates have no spread.", "",
          "| scheme | metric | " + " | ".join(COL_TITLES) + " |", "|---|---|" + "---|" * 4]
    for _, name, _ in ROWS:
        if not any((name, lay, case) in table and table[(name, lay, case)]["spread_skill"][0] > 0
                   for lay, case in COLUMNS):
            continue
        for key, label, f in (("spread_skill", "spread/skill", ".2f"), ("closs0.5", "cal. loss τ=0.5 (%)", ".0f"),
                              ("closs0.75", "cal. loss τ=0.75 (%)", ".0f")):
            row = []
            for lay, case in COLUMNS:
                t = table.get((name, lay, case))
                if t is None:
                    row.append("—")
                    continue
                cs = f" (c* {t['cstar' + key[5:]]:.1f})" if key.startswith("closs") else ""
                row.append(fmt(t[key], f) + cs)
            L.append(f"| {name} | {label} | " + " | ".join(row) + " |")
    L.append("")
    L += [f"**Paired differences vs {REFERENCE}** (Δ = scheme − reference, physical units²; negative = better "
          "than the reference; wins = windows out of 200 where the scheme scores lower).", "",
          "| scheme | τ | " + " | ".join(COL_TITLES) + " |", "|---|---|" + "---|" * 4]
    for _, name, _ in ROWS:
        if name == REFERENCE:
            continue
        for tau in (0.0, 0.75):
            j = TAUS.index(tau)
            row = []
            for lay, case in COLUMNS:
                a, r = raw.get((name, lay, case)), raw.get((REFERENCE, lay, case))
                if a is None or r is None:
                    row.append("—")
                    continue
                va = (a["fms_gauss"][:, :, j] * w2).mean(axis=1)
                vr = (r["fms_gauss"][:, :, j] * w2).mean(axis=1)
                d = paired_ci(mean_stat(va), mean_stat(vr), len(va))
                row.append(f"{fmt(d)}; {int((va < vr).sum())}/{len(va)}")
            L.append(f"| {name} | {tau} | " + " | ".join(row) + " |")
    L.append("")
    L += ["**Non-Gaussian check** (ensemble rows): KDE-smoothed ensemble FMS relative to the Gaussian at the KDE's "
          "inflated variance, in %; negative = the ensemble's shape beats a Gaussian of the same variance. DA rows "
          "are the re-run with stored members (a fresh ensemble realisation).", "",
          "| scheme | τ | " + " | ".join(COL_TITLES) + " |", "|---|---|" + "---|" * 4]
    for name in [n for _, n, _ in ROWS] + [n for n, _ in MEMBER_ROWS]:
        for tau in (0.5, 0.75):
            row = [fmt(table[(name, lay, case)][f"nongauss{tau}"], "+.1f")
                   if (name, lay, case) in table and f"nongauss{tau}" in table[(name, lay, case)] else "—"
                   for lay, case in COLUMNS]
            if all(c == "—" for c in row):
                continue
            L.append(f"| {name} | {tau} | " + " | ".join(row) + " |")
    L.append("")
    return L, table


def main(in_dir: Path, out_md: Path, out_json: Path | None) -> None:
    L, table = build(in_dir)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L) + "\n")
    if out_json is not None:
        out_json.write_text(json.dumps({f"{n}|{lay}|{case}": v for (n, lay, case), v in table.items()}, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None)
