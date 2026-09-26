"""Extended L96 benchmark: everything measured after the benchmark-default
report (``l96_benchmark_default.md``) on the same inputs.

- Main table on the regular 30-obs test set and the canonical random test set,
  S0/S1: DA with analysis-ensemble CRPS, benchmark-default learned models at
  400 and 1200 epochs, the 3000-window run, SDA at the validation-tuned
  guidance weight, the DirectUNet -> SDA hybrid, P1 fixed-obs references.
- Training budget (1000 x 400 vs 1000 x 1200 vs 3000 x 400 epochs).
- RMSE vs number of observation times and vs number of observed fast channels
  (#243 factorial grid + out-of-range probes, and the canonical set binned per
  window).
- Out-of-range probes, validation tuning (SDA weight, hybrid, flow
  calibration), the marginal-value-of-observations DA rerun.

Every metric is recomputed from stored per-window scores / estimates /
ensembles and cached (``experiments/l96_benchmark_extended_metrics.json``,
keyed by path + mtime). Every learned result is checked against its test set
first (``scripts/check_l96_testset_consistency.py``); rows whose runs have not
finished render as "pending", so re-running the script finalizes the report.

  python reports/l96/generate_l96_benchmark_extended_report.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "reports" / "l96"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluation.estimate_metrics import _groups_from_per_window  # noqa: E402
from evaluation.run_l96 import make_obs_j_indices  # noqa: E402
from generate_p1_l96_benchmark import generative  # noqa: E402
from scripts.check_l96_testset_consistency import check_da, check_learned  # noqa: E402
import _inputs  # noqa: E402

REPORT = "benchmark_extended"
SHARED = _inputs.shared()
HERE = _inputs.root(REPORT, "here")
BENCH = _inputs.root(REPORT, "bench")
P1 = _inputs.root(REPORT, "p1")
N20 = _inputs.root(REPORT, "n20") / "eval_rlayout_n10-100_k4-16_w200"
RANDOM_OBS_TIMES = _inputs.root(REPORT, "random_obs_times")
FLOW_SUB = "ens30_no20"
FLOW_SAMPLING = {"n_members": 30, "n_outer": 20, "step_power": 0.5}
REG_TEST = SHARED / "l96_datasets_obsj2_int100_nwin200.pt"
CAN_TEST = SHARED / "l96_testset_rlayout_n10-100_k4-16_w200_d1.pt"
CAN = HERE / "eval_rlayout_n10-100_k4-16_w200"
REGE = HERE / "eval_regular"
FACT = HERE / "eval_factorial"
OOD = HERE / "eval_ood"
DA243 = _inputs.root(REPORT, "da_random_layout") / "l96_da_random_layout"
DAHERE = HERE / "l96_da_random_layout"
OUT = ROOT / "reports/l96/outputs"
CACHE = ROOT / "experiments" / "l96_benchmark_extended_metrics.json"
CASES = ("s0", "s1")
IDX = np.array(make_obs_j_indices(8, 4, 2))
G = _groups_from_per_window


def seeds(fmt: str, base: Path, sub: str, n=(1, 2, 3)) -> list[Path]:
    return [base / fmt.format(s=s) / sub for s in n]


# (group, label, [regular result dirs], [random-set result dirs])
def rows() -> list[tuple]:
    D, FL = "ens1_no1", FLOW_SUB
    R = []
    for lab, fam, base in (("DirectUNet-M", "directunet", BENCH), ("PredictStateCFM-M", "predictstatecfm", HERE),
                           ("VanillaCFM-M", "vanillacfm", BENCH)):
        sub = D if fam == "directunet" else FL
        can_root = CAN if fam == "directunet" else N20
        R.append(("Benchmark default, 400 ep", lab, seeds(f"L96B_{fam}_monaiM_seed{{s}}", base, sub),
                  seeds(f"L96B_{fam}_monaiM_seed{{s}}", can_root, sub)))
    for lab, fam in (("DirectUNet-M", "directunet"), ("PredictStateCFM-M", "predictstatecfm"), ("VanillaCFM-M", "vanillacfm")):
        sub = D if fam == "directunet" else FL
        R.append(("Benchmark default, 1200 ep", lab, seeds(f"L96B_{fam}_monaiM_ep1200_seed{{s}}", HERE, sub),
                  seeds(f"L96B_{fam}_monaiM_ep1200_seed{{s}}", CAN, sub)))
    R.append(("Benchmark default, 3000 windows", "DirectUNet-M", [HERE / "L96B_directunet_monaiM_ntrain3000_seed1" / D],
              [CAN / "L96B_directunet_monaiM_ntrain3000_seed1" / D]))
    R.append(("Flow ensemble, 1200 ep (3 networks)", "PredictStateCFM-M x3", [REGE / "ens3_psc123" / FL],
              [CAN / "ens3_psc123" / FL]))
    for lab, name in (("SDA1-M", "B4_sda1_monaiM_l96"), ("SDA2-M", "A3_sda2_monaiM_l96"), ("SDA3-fix-M", "A3_sda3fix_monaiM_l96")):
        R.append(("SDA, gw 25", lab, seeds(f"{name}_seed{{s}}", REGE, "ens30_gw25"), seeds(f"{name}_seed{{s}}", CAN, "ens30_gw25")))
    for lab, name in (("SDA1-S+", "B4_sda1_monaiSplus_l96"), ("SDA1-L", "B4_sda1_monaiL_l96")):
        R.append(("SDA, gw 20", lab, [P1 / name / "ens30_gw20"], [CAN / name / "ens30_gw20"]))
    for lab, name in (("DirectUNet-M(400 ep) -> SDA2-M", "hybrid_DU{s}_A3_sda2_monaiM_l96"),
                      ("DirectUNet-M(400 ep) -> SDA1-M", "hybrid_DU{s}_B4_sda1_monaiM_l96"),
                      ("DirectUNet-M(1200 ep) -> SDA2-M", "hybrid_DU1200s{s}_A3_sda2_monaiM_l96"),
                      ("DirectUNet-M(1200 ep) -> SDA3-fix-M", "hybrid_DU1200s{s}_A3_sda3fix_monaiM_l96")):
        R.append(("Hybrid (tau0 0.1, gw 2)", lab, seeds(name, REGE, "tau0.1_gw2"), seeds(name, CAN, "tau0.1_gw2")))
    for lab, name, sub in (("DirectUNet-M", "P1_directunet_monaiM_noaug_l96", D), ("PredictStateCFM-M", "A2_predictstatecfm_monaiM_l96", FL),
                           ("VanillaCFM-M", "A1_vanillacfm_monaiM_l96", FL)):
        R.append(("P1 fixed obs (reference)", lab, [P1 / name / sub], [(CAN if sub == D else N20) / name / sub]))
    return R


def done(d: Path) -> bool:
    return any((d / f).exists() for f in ("neural_eval.json", "sda_eval.json", "hybrid_eval.json")) and \
        ((d / "scores_s1.npz").exists() or (d / "estimates_s1.npz").exists())


def sampling_errors(d: Path) -> list[str]:
    """Flow results must be scored with the benchmark sampler (#257)."""
    j = d / "neural_eval.json"
    if d.name != FLOW_SUB or not j.exists():
        return []
    got = json.load(open(j)).get("sampling", {})
    bad = {k: got.get(k) for k, v in FLOW_SAMPLING.items() if got.get(k) != v}
    return [f"{d}: sampling {bad} != benchmark {FLOW_SAMPLING}"] if bad else []


def score_dir(d: Path, cache: dict) -> dict:
    """Per-window RMSE (all_obs) per case, plus CRPS / spread-over-RMSE means."""
    files = [d / f"{k}_{c}.npz" for k in ("scores", "estimates", "members") for c in CASES]
    key = str(d.resolve()) + "|" + "|".join(str(os.path.getmtime(f)) for f in files if f.exists())
    if key in cache:
        return cache[key]
    out = {}
    for c in CASES:
        if (d / f"scores_{c}.npz").exists():
            z = np.load(d / f"scores_{c}.npz")
            r = G(z["rmse"])["all_obs"]
            out[c] = {"rmse": r.tolist(), "crps": float(G(z["crps"])["all_obs"].mean()),
                      "sp": float(G(z["spread"])["all_obs"].mean() / r.mean())}
        elif (d / f"members_{c}.npz").exists():
            g = generative(d / f"members_{c}.npz")
            out[c] = {"rmse": g["rmse"].tolist(), "crps": float(g["crps"].mean()), "sp": float(g["spread"].mean() / g["rmse"].mean())}
        else:
            z = np.load(d / f"estimates_{c}.npz")
            out[c] = {"rmse": G(np.sqrt(((z["trajectories"] - z["truth"]) ** 2).mean(1)))["all_obs"].tolist(),
                      "crps": None, "sp": None}
    cache[key] = out
    CACHE.write_text(json.dumps(cache))
    return out


def da_main() -> list[dict]:
    """Regular: ETKF/EnKF from the analysis-CRPS rerun (inflation S0 1.5 / S1 2.0), Strong-4DVar from the
    corrected inflation-2.0 run. Random set: the random-layout DA runs on the canonical layouts."""
    truth = torch.load(REG_TEST, weights_only=False)
    tr = {c: np.stack([w["true_state"].numpy()[:, IDX] for w in truth[f"test_{c}"]]) for c in CASES}
    zc = np.load(HERE / "l96_baselines_trajectories_dws500_s0c_crps_infs0-1.5_s1-2.0_etkf_infs0-1.5_s1-2.0_obsj2_int100_fw_dafw.npz")
    z2 = np.load(RANDOM_OBS_TIMES / "l96_baselines_trajectories_dws500_s0c_inf2.0_etkf_inf2.0_obsj2_int100_fw_dafw.npz")
    pc = np.load(DAHERE / "per_window_rlayout_n10-100_k4-16_w200_d1_infs0-1.5_s1-2.0.npz")
    p2 = np.load(DA243 / "per_window_rlayout_n10-100_k4-16_w200_d1_inf2.0.npz")
    sel = lambda a: a[..., IDX] if a.shape[-1] > len(IDX) else a  # noqa: E731
    out = []
    for m in ("ETKF", "EnKF", "Strong-4DVar"):
        k = m.replace("-", "_")
        rec = {"label": m, "reg": {}, "can": {}}
        for c in CASES:
            z = z2 if m == "Strong-4DVar" else zc
            t = sel(z[f"{c}_{k}_trajectories"])
            r = G(np.sqrt(((t - tr[c]) ** 2).mean(1)))["all_obs"]
            rec["reg"][c] = {"rmse": r, "crps": None, "sp": None}
            if m != "Strong-4DVar":
                rec["reg"][c]["crps"] = float(G(sel(z[f"{c}_{k}_crps"]))["all_obs"].mean())
                rec["reg"][c]["sp"] = float(G(np.sqrt(np.clip(sel(z[f"{c}_{k}_ensemble_variance"]), 0, None)).mean(1))["all_obs"].mean() / r.mean())
            p = p2 if m == "Strong-4DVar" else pc
            rr = p[f"{c}_{k}_rmse_all_obs"]
            rec["can"][c] = {"rmse": rr, "crps": float(p[f"{c}_{k}_crps_all_obs"].mean()) if m != "Strong-4DVar" else None,
                             "sp": float(p[f"{c}_{k}_spread_all_obs"].mean() / rr.mean()) if m != "Strong-4DVar" else None}
        out.append(rec)
    return out


def learned_main(cache: dict) -> tuple[list[dict], list[str]]:
    truths = {}
    for name, path in (("reg", REG_TEST), ("can", CAN_TEST)):
        d = torch.load(path, weights_only=False)
        truths[name] = {c: np.stack([w["true_state"].numpy()[:, IDX] for w in d[f"test_{c}"]]) for c in CASES}
    errors, out = [], []
    for group, label, regs, cans in rows():
        rec = {"group": group, "label": label, "n_total": len(regs), "reg": None, "can": None}
        for tag, dirs, path in (("reg", regs, REG_TEST), ("can", cans, CAN_TEST)):
            ok = [d for d in dirs if done(d)]
            for d in ok:
                errors += check_learned(str(d), str(path), truths[tag]) + sampling_errors(d)
            if ok:
                per = [score_dir(d, cache) for d in ok]
                rec[tag] = {c: {"rmse": np.mean([p[c]["rmse"] for p in per], axis=0),
                                "seed_means": [float(np.mean(p[c]["rmse"])) for p in per],
                                "crps": None if per[0][c]["crps"] is None else float(np.mean([p[c]["crps"] for p in per])),
                                "sp": None if per[0][c]["sp"] is None else float(np.mean([p[c]["sp"] for p in per]))}
                            for c in CASES}
                rec[f"n_{tag}"] = len(ok)
        out.append(rec)
    return out, errors


def ms(r: np.ndarray) -> str:
    return f"{r.mean():.3f} ± {r.std(ddof=1):.3f}"


def fmt(x, d=3) -> str:
    return "—" if x is None else f"{x:.{d}f}"


def main_tables(da: list[dict], learned: list[dict]) -> list[str]:
    A = ["## 1. Main table\n",
         "Per-window RMSE on the 24D observed space, **mean ± sd across the 200 windows** (seeds pooled "
         "per window). `seeds` = finished seeds / planned. Deterministic schemes have no CRPS.\n",
         "| group | scheme | seeds | regular S0 | regular S1 | random S0 | random S1 | random/regular S0 | seed sd (reg S0) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in da:
        A.append(f"| DA | {r['label']} | — | {ms(r['reg']['s0']['rmse'])} | {ms(r['reg']['s1']['rmse'])} | {ms(r['can']['s0']['rmse'])} | "
                 f"{ms(r['can']['s1']['rmse'])} | {r['can']['s0']['rmse'].mean() / r['reg']['s0']['rmse'].mean():.2f} | — |")
    for r in learned:
        if r["reg"] is None or r["can"] is None:
            A.append(f"| {r['group']} | {r['label']} | 0/{r['n_total']} | pending | pending | pending | pending | — | — |")
            continue
        sm = r["reg"]["s0"]["seed_means"]
        sd = f"{np.std(sm, ddof=1):.3f}" if len(sm) > 1 else "—"
        A.append(f"| {r['group']} | {r['label']} | {r['n_reg']}/{r['n_total']} | {ms(r['reg']['s0']['rmse'])} | {ms(r['reg']['s1']['rmse'])} | "
                 f"{ms(r['can']['s0']['rmse'])} | {ms(r['can']['s1']['rmse'])} | "
                 f"{r['can']['s0']['rmse'].mean() / r['reg']['s0']['rmse'].mean():.2f} | {sd} |")
    A += ["\n### CRPS / spread-over-RMSE (S0; S1 in parentheses)\n",
          "| group | scheme | CRPS regular | CRPS random | spread/RMSE regular | spread/RMSE random |", "|---|---|---|---|---|---|"]
    for r in da:
        if r["reg"]["s0"]["crps"] is None:
            continue
        A.append(f"| DA | {r['label']} | {fmt(r['reg']['s0']['crps'])} ({fmt(r['reg']['s1']['crps'])}) | {fmt(r['can']['s0']['crps'])} ({fmt(r['can']['s1']['crps'])}) | "
                 f"{fmt(r['reg']['s0']['sp'], 2)} ({fmt(r['reg']['s1']['sp'], 2)}) | {fmt(r['can']['s0']['sp'], 2)} ({fmt(r['can']['s1']['sp'], 2)}) |")
    for r in learned:
        if r["reg"] is None or r["reg"]["s0"]["crps"] is None or r["can"] is None:
            continue
        A.append(f"| {r['group']} | {r['label']} | {fmt(r['reg']['s0']['crps'])} ({fmt(r['reg']['s1']['crps'])}) | {fmt(r['can']['s0']['crps'])} ({fmt(r['can']['s1']['crps'])}) | "
                 f"{fmt(r['reg']['s0']['sp'], 2)} ({fmt(r['reg']['s1']['sp'], 2)}) | {fmt(r['can']['s0']['sp'], 2)} ({fmt(r['can']['s1']['sp'], 2)}) |")
    return A


def budget_section(cache: dict) -> list[str]:
    import glob

    import pandas as pd
    runs = [("1000 windows x 400 ep", BENCH / "L96B_directunet_monaiM_seed1", 1000, 400),
            ("1000 windows x 1200 ep", HERE / "L96B_directunet_monaiM_ep1200_seed1", 1000, 1200),
            ("3000 windows x 400 ep", HERE / "L96B_directunet_monaiM_ntrain3000_seed1", 3000, 400)]
    A = ["## 2. Training budget (DirectUNet-M, seed 1)\n",
         "Steps per epoch = windows / 16. At equal gradient steps the 1000 x 1200 and 3000 x 400 runs are at the "
         "same point of their cosine schedules, so their curves compare step for step.\n",
         "| run | gradient steps | regular S0 | random S0 | final val_loss | val_loss @ 25k / 50k steps |", "|---|---|---|---|---|---|"]
    for lab, d, nw, ep in runs:
        csv = sorted(glob.glob(str(d / "outputs/stage1/version_*/metrics.csv")))
        v = pd.read_csv(csv[0]).dropna(subset=["val_loss"]).groupby("epoch")["val_loss"].last() if csv else None
        at = []
        for st in (25000, 50000):
            e = int(round(st / (nw / 16)))
            at.append(f"{v.loc[e]:.4f}" if v is not None and e in v.index else "—")
        reg = np.mean(score_dir(d / "ens1_no1", cache)["s0"]["rmse"]) if done(d / "ens1_no1") else None
        can_d = CAN / d.name / "ens1_no1"
        can = np.mean(score_dir(can_d, cache)["s0"]["rmse"]) if done(can_d) else None
        A.append(f"| {lab} | {int(nw / 16 * ep):,} | {fmt(reg)} | {fmt(can)} | {v.iloc[-1]:.4f} | {' / '.join(at)} |" if v is not None
                 else f"| {lab} | — | {fmt(reg)} | {fmt(can)} | — | — |")
    A.append("")
    return A


def pick(d: Path) -> Path | None:
    """The benchmark-protocol result sub-dir of a run: ens1_no1 (deterministic), ens30_no20 (flows,
    #257 sampler) or ens30_gw25 (SDA). Never falls back to the old flow protocol (ens30_no10)."""
    for sub in ("ens1_no1", FLOW_SUB, "ens30_gw25"):
        if done(d / sub):
            return d / sub
    return None


def _cell_scores(root: Path, cell: str, name: str, cache: dict, case: str):
    sub = pick(root / cell / name)
    return None if sub is None else float(np.mean(score_dir(sub, cache)[case]["rmse"]))


def _da_cell(tag: str, method: str, case: str) -> float | None:
    for d in (DA243, DAHERE):
        s0 = d / f"summary_{tag}_s0_inf1.5.json"
        both = d / f"summary_{tag}_inf2.0.json"
        pc = sorted(d.glob(f"summary_{tag}_infs0*.json"))
        if case == "s0" and method != "Strong-4DVar" and s0.exists():
            f = s0
        elif both.exists():
            f = both
        elif pc:
            f = pc[0]
        else:
            continue
        return json.load(open(f))["cases"][case][method]["rmse"]["all_obs"]["mean"]
    return None


LEARNED_CURVES = (("DirectUNet-M", "L96B_directunet_monaiM_seed{s}"), ("PredictStateCFM-M", "L96B_predictstatecfm_monaiM_seed{s}"),
                  ("VanillaCFM-M", "L96B_vanillacfm_monaiM_seed{s}"))


def curve(cells: list[tuple[str, str]], cache: dict, case: str) -> dict:
    """cells = [(tag, 'fact'|'ood')]; learned = 3-seed mean where available (factorial), seed 1 (probes)."""
    res = {k: [] for k in [lab for lab, _ in LEARNED_CURVES] + ["SDA1-M", "ETKF", "EnKF", "Strong-4DVar"]}
    for tag, kind in cells:
        root = FACT if kind == "fact" else OOD
        for lab, fmt_ in LEARNED_CURVES:
            vals = [v for v in (_cell_scores(root, tag, fmt_.format(s=s), cache, case) for s in ((1, 2, 3) if kind == "fact" else (1,)))
                    if v is not None]
            res[lab].append(float(np.mean(vals)) if vals else None)
        res["SDA1-M"].append(_cell_scores(root, tag, "B4_sda1_monaiM_l96_seed1", cache, case))
        for m in ("ETKF", "EnKF", "Strong-4DVar"):
            res[m].append(_da_cell(tag, m, case))
    return res


def curve_table(title: str, xs: list, labels: list[str], res: dict) -> list[str]:
    A = [f"\n{title}\n", "| " + " | ".join(["x"] + list(res)) + " |", "|" + "---|" * (len(res) + 1)]
    for i, x in enumerate(labels):
        A.append(f"| {x} | " + " | ".join(fmt(res[k][i]) for k in res) + " |")
    return A


COLORS = {"DirectUNet-M": "#2a78d6", "PredictStateCFM-M": "#eb6834", "VanillaCFM-M": "#1baf7a", "SDA1-M": "#eda100",
          "ETKF": "#4a3aa7", "EnKF": "#e87ba4", "Strong-4DVar": "#008300"}


def curve_figure(path: Path, xs: list, curves: dict, xlabel: str, logx: bool, shade=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, case in zip(axes, CASES):
        for lab, ys in curves[case].items():
            pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if pts:
                ax.plot(*zip(*pts), marker="o", markersize=5, linewidth=2, color=COLORS[lab], label=lab,
                        linestyle="--" if lab in ("ETKF", "EnKF", "Strong-4DVar") else "-")
        if shade:
            for lo, hi in shade:
                ax.axvspan(lo, hi, color="#f0efec", zorder=0)
        if logx:
            ax.set_xscale("log")
        ax.set_xticks(xs, [f"{x:g}" for x in xs])
        ax.minorticks_off()
        ax.set_yscale("log")
        ax.set_yticks([0.2, 0.3, 0.5, 0.7, 1.0, 1.5], ["0.2", "0.3", "0.5", "0.7", "1.0", "1.5"])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(case.upper())
        ax.set_xlabel(xlabel)
        ax.grid(True, color="#e6e5e0", linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("RMSE, 24D observed (log)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, loc="lower center", ncol=len(lab))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def density_sections(cache: dict) -> list[str]:
    A = ["## 3. Performance vs number of observation times per window\n",
         "Identical obs for every scheme: the #243 factorial layouts (20 windows x 3 draws; learned = 3-seed mean, "
         "SDA1-M seed 1 at gw 25) and the out-of-range probes (`*`: 6, 300, 1000 obs; learned seed 1). "
         "All 16 observed fast channels. Shaded: outside the training range (10-300).\n"]
    ns = [6, 10, 20, 30, 50, 75, 100, 300, 1000]
    cells = [(f"rlayout_n{n}-{n}_k16-16_w20_d3", "ood" if n in (6, 300, 1000) else "fact") for n in ns]
    curves = {c: curve(cells, cache, c) for c in CASES}
    png = OUT / "l96_benchmark_extended_nobs.png"
    curve_figure(png, ns, curves, "observation times per window (log)", True, shade=[(5, 10), (300, 1100)])
    A.append(f"![RMSE vs n_obs]({png.name})")
    for c in CASES:
        A += curve_table(f"**{c.upper()}** (k = 16)", ns, [f"{n}{'*' if n in (6, 300, 1000) else ''}" for n in ns], curves[c])
    A.append("\nAveraged over k in {4, 8, 12, 16} (factorial only):")
    for c in CASES:
        avg = {}
        for n in (10, 20, 30, 50, 75, 100):
            cs = curve([(f"rlayout_n{n}-{n}_k{k}-{k}_w20_d3", "fact") for k in (4, 8, 12, 16)], cache, c)
            for lab, v in cs.items():
                vals = [x for x in v if x is not None]
                avg.setdefault(lab, []).append(float(np.mean(vals)) if len(vals) == 4 else None)
        A += curve_table(f"**{c.upper()}**", [10, 20, 30, 50, 75, 100], ["10", "20", "30", "50", "75", "100"], avg)
    A += ["\n## 4. Performance vs number of observed fast channels\n",
          "30 obs per window; k = 0 and 2 are probes (`*`, below the training minimum of 4, learned seed 1).\n"]
    ks = [0, 2, 4, 8, 12, 16]
    cells = [(f"rlayout_n30-30_k{k}-{k}_w20_d3", "ood" if k in (0, 2) else "fact") for k in ks]
    curves = {c: curve(cells, cache, c) for c in CASES}
    png = OUT / "l96_benchmark_extended_kfast.png"
    curve_figure(png, ks, curves, "observed fast channels (of 16)", False, shade=[(-0.5, 3.5)])
    A.append(f"![RMSE vs k]({png.name})")
    for c in CASES:
        A += curve_table(f"**{c.upper()}** (30 obs)", ks, [f"{k}{'*' if k in (0, 2) else ''}" for k in ks], curves[c])
    A += binned_canonical(cache)
    return A


def binned_canonical(cache: dict) -> list[str]:
    d = torch.load(CAN_TEST, weights_only=False)
    nobs = {c: np.array([int(w["obs_mask"].sum()) for w in d[f"test_{c}"]]) for c in CASES}
    kf = {c: np.array([int((~torch.isnan(w["obs"][w["obs_mask"], 8:])).sum(1)[0]) for w in d[f"test_{c}"]]) for c in CASES}
    pc = np.load(DAHERE / "per_window_rlayout_n10-100_k4-16_w200_d1_infs0-1.5_s1-2.0.npz")
    p2 = np.load(DA243 / "per_window_rlayout_n10-100_k4-16_w200_d1_inf2.0.npz")
    A = ["\n### Canonical random test set, windows binned by their own obs count / fast channels\n",
         "Learned: benchmark default 400 ep, 3-seed mean; SDA1-M gw 25, 3 seeds; hybrid DirectUNet-M(400) -> SDA2-M.\n"]
    for c in CASES:
        s = {}
        for lab, f in LEARNED_CURVES:
            dirs = [pick(CAN / f.format(s=x)) or pick(N20 / f.format(s=x)) for x in (1, 2, 3)]
            s[lab] = np.mean([score_dir(p, cache)[c]["rmse"] for p in dirs], axis=0)
        s["SDA1-M"] = np.mean([score_dir(CAN / f"B4_sda1_monaiM_l96_seed{x}" / "ens30_gw25", cache)[c]["rmse"] for x in (1, 2, 3)], axis=0)
        hyb = [CAN / f"hybrid_DU{x}_A3_sda2_monaiM_l96" / "tau0.1_gw2" for x in (1, 2, 3)]
        if all(done(h) for h in hyb):
            s["Hybrid DU->SDA2"] = np.mean([score_dir(h, cache)[c]["rmse"] for h in hyb], axis=0)
        s["ETKF"], s["EnKF"], s["Strong-4DVar"] = pc[f"{c}_ETKF_rmse_all_obs"], pc[f"{c}_EnKF_rmse_all_obs"], p2[f"{c}_Strong_4DVar_rmse_all_obs"]
        for by, bins, lab in ((nobs[c], [(10, 25), (25, 40), (40, 55), (55, 70), (70, 85), (85, 101)], "n_obs"),
                              (kf[c], [(4, 7), (7, 10), (10, 13), (13, 17)], "k")):
            A += [f"\n**{c.upper()}, by {lab}**\n", f"| {lab} | windows | " + " | ".join(s) + " |", "|" + "---|" * (len(s) + 2)]
            for lo, hi in bins:
                m = (by >= lo) & (by < hi)
                A.append(f"| {lo}-{hi - 1} | {m.sum()} | " + " | ".join(f"{v[m].mean():.3f}" for v in s.values()) + " |")
    return A


def probes_section(cache: dict) -> list[str]:
    tags = [("n6-6_k16-16", "6 obs"), ("n30-30_k0-0", "slow-only (k 0)"), ("n30-30_k2-2", "k 2"), ("n300-300_k16-16", "300 obs"),
            ("n1000-1000_k16-16", "1000 obs"), ("n30-30_k16-16_r0.25", "noise R 0.25"), ("n30-30_k16-16_r1", "noise R 1.0")]
    A = ["\n## 5. Out-of-range probes\n",
         "Training range: n_obs 10-300, k 4-16, R 0.5. 20 windows x 3 draws; learned seed 1, SDA1-M gw 25; DA told the "
         "true R. Reference in-range cell: 30 obs, k 16.\n",
         "| probe | " + " | ".join(["DirectUNet-M", "PredictStateCFM-M", "VanillaCFM-M", "SDA1-M", "ETKF", "Strong-4DVar"]) + " |",
         "|" + "---|" * 7]
    for tag, lab in tags:
        cell = f"rlayout_{tag.split('_r')[0] if '_r' in tag else tag}_w20_d3" + (f"_r{tag.split('_r')[1]}" if "_r" in tag else "")
        vals = []
        for c in CASES:
            vals.append([_cell_scores(OOD, cell, f"{f.format(s=1)}", cache, c) for _, f in LEARNED_CURVES]
                        + [_cell_scores(OOD, cell, "B4_sda1_monaiM_l96_seed1", cache, c), _da_cell(cell, "ETKF", c), _da_cell(cell, "Strong-4DVar", c)])
        A.append(f"| {lab} | " + " | ".join(f"{fmt(a, 2)} / {fmt(b, 2)}" for a, b in zip(*vals)) + " |")
    A.append("\nCells are S0 / S1 RMSE.")
    return A


def tuning_section(cache: dict) -> list[str]:
    V = HERE / "eval_val"
    A = ["\n## 6. Evaluation-time settings tuned on validation windows\n",
         "50 validation windows per case with seeds disjoint from train/val/test "
         "(`scripts/make_l96_validation_sets.py`), regular and random-layout versions. Mean of S0 and S1 RMSE.\n"]
    mean2 = lambda d: float(np.mean([np.mean(score_dir(d, cache)[c]["rmse"]) for c in CASES]))  # noqa: E731
    A += ["**SDA guidance weight** (P1 used 20)\n", "| regime | model | " + " | ".join(f"gw {g}" for g in (10, 15, 20, 25, 30, 40)) + " |", "|" + "---|" * 8]
    for rg in ("regular", "rlayout"):
        for m in ("B4_sda1_monaiM_l96", "A3_sda2_monaiM_l96", "A3_sda3fix_monaiM_l96_seed1"):
            A.append(f"| {rg} | {m} | " + " | ".join(fmt(mean2(V / rg / 'sda_gw' / m / f'gw{g}')) if done(V / rg / 'sda_gw' / m / f'gw{g}') else '—'
                                                    for g in (10, 15, 20, 25, 30, 40)) + " |")
    A += ["\n**DirectUNet-M -> SDA hybrid** (rows tau0, columns guidance weight); `DU alone` for reference\n"]
    for rg in ("regular", "rlayout"):
        du = V / rg / "du_alone/L96B_directunet_monaiM_seed1"
        A.append(f"\n{rg}: DirectUNet-M(400 ep) alone {fmt(mean2(du)) if done(du) else '—'}\n")
        gws = ["0.5", "1", "2", "5", "10", "20"]
        A += ["| prior | tau0 | " + " | ".join(f"gw {g}" for g in gws) + " |", "|" + "---|" * 8]
        for m in ("B4_sda1_monaiM_l96", "A3_sda2_monaiM_l96"):
            for t in ("0.1", "0.2", "0.3", "0.5", "0.7"):
                A.append(f"| {m} | {t} | " + " | ".join(fmt(mean2(V / rg / 'hybrid' / m / f'tau{t}_gw{g}')) if done(V / rg / 'hybrid' / m / f'tau{t}_gw{g}')
                                                         else '—' for g in gws) + " |")
        du12 = V / rg / "du_alone/L96B_directunet_monaiM_ep1200_seed1"
        if done(du12):
            A.append(f"\nWith the 1200-epoch DirectUNet-M (alone {fmt(mean2(du12))}), SDA2-M / SDA3-fix-M priors. **tau0 0.05 is not a "
                     "warm start**: the sampler snaps tau0 to its 10-step grid with `round(tau0 * N_outer)`, and 0.5 "
                     "rounds to 0, so that row is plain SDA from noise at a far-too-low guidance weight (the tuned SDA "
                     "weight is 25).\n")
            gws2 = ["1", "2", "5"]
            A += ["| prior | tau0 | " + " | ".join(f"gw {g}" for g in gws2) + " |", "|" + "---|" * 5]
            for m in ("A3_sda2_monaiM_l96", "A3_sda3fix_monaiM_l96_seed1"):
                for t in ("0.05", "0.1", "0.2"):
                    d = V / rg / "hybrid_du1200" / m
                    A.append(f"| {m} | {t} | " + " | ".join(fmt(mean2(d / f'tau{t}_gw{g}')) if done(d / f'tau{t}_gw{g}') else '—'
                                                             for g in gws2) + " |")
    A += ["\n**Flow calibration** (benchmark default seed 1; S0 RMSE / CRPS / spread-over-RMSE)\n",
          "| regime | model | 10 steps | 20 steps | 50 steps | sigma 0.75 | sigma 1.0 |", "|---|---|---|---|---|---|---|"]
    for rg in ("regular", "rlayout"):
        for m in ("L96B_vanillacfm_monaiM_seed1", "L96B_predictstatecfm_monaiM_seed1"):
            cells = []
            for s in ("no10_signone", "no20_signone", "no50_signone", "no10_sig0.75", "no10_sig1.0"):
                d = V / rg / "calib" / m / s
                x = score_dir(d, cache)["s0"] if done(d) else None
                cells.append("—" if x is None else f"{np.mean(x['rmse']):.3f} / {x['crps']:.3f} / {x['sp']:.2f}")
            A.append(f"| {rg} | {m} | " + " | ".join(cells) + " |")
    return A


def marginal_section() -> list[str]:
    M = ("Strong-4DVar", "ETKF", "EnKF")
    js = lambda p: json.load(open(p))  # noqa: E731
    try:
        leg30 = js(HERE / "l96_baselines_dws500_legacy_inf2.0_etkf_inf2.0_obsj2_int100_dafw.json")
        leg15 = js(HERE / "l96_baselines_dws500_legacy_inf2.0_etkf_inf2.0_obsj2_int200_dafw.json")
        b15 = js(HERE / "l96_baselines_dws500_s0c_infs0-1.5_s1-2.0_etkf_infs0-1.5_s1-2.0_obsj2_int200_fw_dafw.json")
        r15 = js(HERE / "l96_baselines_dws500_s0c_inf1.5_etkf_inf1.5_obsj2_int100_fw_dafw.json")
        r20 = js(RANDOM_OBS_TIMES / "l96_baselines_dws500_s0c_inf2.0_etkf_inf2.0_obsj2_int100_fw_dafw.json")
    except FileNotFoundError:
        return ["\n## 7. Marginal value of observations (DA)\n", "pending"]
    b30 = {"s0": {"ETKF": r15["s0"]["ETKF"], "EnKF": r15["s0"]["EnKF"], "Strong-4DVar": r20["s0"]["Strong-4DVar"]},
           "s1": {k: r20["s1"][k] for k in M}}
    paper = {"Strong-4DVar": (0.9701, 0.7788, 1.4751, 1.4276), "ETKF": (1.0973, 0.8815, 1.6367, 1.4680), "EnKF": (1.0927, 0.9046, 1.6503, 1.5022)}
    A = ["\n## 7. Marginal value of observations for DA (15 -> 30 regular obs per window)\n",
         "The P1 paper's headline (6.2x for Strong-4D-Var vs 1.9x for filters) came from a run whose DA model never "
         "received the per-window `fast_weights`. Rerun with them: on the original data configuration at the original "
         "inflation (only change), and on the benchmark data at the per-case inflation. Ratio = S0 gain / S1 gain "
         "(recomputed from the RMSEs; the paper rounds its Strong-4D-Var ratio to 6.2x from the rounded percentages).\n",
         "| scheme | setting | S0 15 -> 30 | S1 15 -> 30 | ratio |", "|---|---|---|---|---|"]
    for m in M:
        for lab, a, b, c, d in (("paper (no fast_weights)",) + paper[m],
                                ("original data + fast_weights, inflation 2.0", leg15["s0"][m]["mean"], leg30["s0"][m]["mean"], leg15["s1"][m]["mean"], leg30["s1"][m]["mean"]),
                                ("benchmark data + fast_weights, inflation S0 1.5 / S1 2.0", b15["s0"][m]["mean"], b30["s0"][m]["mean"], b15["s1"][m]["mean"], b30["s1"][m]["mean"])):
            g0, g1 = (a - b) / a, (c - d) / c
            A.append(f"| {m} | {lab} | {a:.3f} -> {b:.3f} ({100 * g0:.1f}%) | {c:.3f} -> {d:.3f} ({100 * g1:.1f}%) | {g0 / g1:.1f}x |")
    return A


def findings(da, learned) -> list[str]:
    row = {(r["group"], r["label"]): r for r in learned}
    m = lambda g, lab, t, c="s0": None if row[(g, lab)][t] is None else float(row[(g, lab)][t][c]["rmse"].mean())  # noqa: E731
    best_da = min(da, key=lambda r: r["reg"]["s0"]["rmse"].mean())
    hg = "Hybrid (tau0 0.1, gw 2)"
    h2, h3 = (hg, "DirectUNet-M(1200 ep) -> SDA2-M"), (hg, "DirectUNet-M(1200 ep) -> SDA3-fix-M")
    A = ["## Findings\n"]
    A.append(f"1. **Best scheme: the DirectUNet-M(1200 ep) -> SDA3-fix-M hybrid** (tau0 0.1, gw 2; the SDA3-fix prior is "
             f"also the better one on the validation windows): regular S0 / S1 {fmt(m(*h3, 'reg'))} / {fmt(m(*h3, 'reg', 's1'))}, "
             f"random S0 / S1 {fmt(m(*h3, 'can'))} / {fmt(m(*h3, 'can', 's1'))} -- flat under model error. The SDA2-M prior "
             f"is marginally better at S0 ({fmt(m(*h2, 'reg'))} / {fmt(m(*h2, 'can'))}) but degrades at S1 "
             f"({fmt(m(*h2, 'reg', 's1'))} / {fmt(m(*h2, 'can', 's1'))}): it was trained with DA params equal to the true ones, "
             f"so the +10% S1 parameter bias leaks into the posterior. Best DA on regular S0: {best_da['label']} "
             f"{best_da['reg']['s0']['rmse'].mean():.3f}.")
    A.append("2. **The 400-epoch budget of the benchmark default is too short.** At 1200 epochs every family gains 9-19% "
             f"(regular S0: DirectUNet {fmt(m('Benchmark default, 400 ep', 'DirectUNet-M', 'reg'))} -> {fmt(m('Benchmark default, 1200 ep', 'DirectUNet-M', 'reg'))}, "
             f"PredictStateCFM {fmt(m('Benchmark default, 400 ep', 'PredictStateCFM-M', 'reg'))} -> {fmt(m('Benchmark default, 1200 ep', 'PredictStateCFM-M', 'reg'))}, "
             f"VanillaCFM {fmt(m('Benchmark default, 400 ep', 'VanillaCFM-M', 'reg'))} -> {fmt(m('Benchmark default, 1200 ep', 'VanillaCFM-M', 'reg'))}), "
             "and the family gaps largely close; ~85% of the 3000-window DirectUNet gain is training length, not data.")
    A.append("3. **Observation-count crossover at S0**: DA (ETKF) is best at <= 10 obs per window; from ~20 obs every learned "
             "scheme beats every DA baseline, and the gap grows with density. Under model error (S1) the learned schemes win "
             "at every density. PredictStateCFM / SDA are best when sparse, DirectUNet when dense; SDA and DirectUNet are "
             "complementary, which is why the hybrid works.")
    A.append("4. **Fast channels**: no crossover -- learned beat DA at every k, including slow-only (k 0, below training range). "
             "The learned slow-variable error is flat (~0.2-0.3); all the k-dependence is in the fast variables. Under model "
             "error, more fast obs make DA's *slow* variables worse (biased slow-fast coupling).")
    A.append("5. **Beyond the training range**: learned models are weak at 6 obs and degrade with 1000 obs (DirectUNet "
             "0.17 -> 0.34 from 300 to 1000); VanillaCFM degrades least. Noise shifts are handled gracefully.")
    sd2, sd3 = ("SDA, gw 25", "SDA2-M"), ("SDA, gw 25", "SDA3-fix-M")
    A.append("6. **Params conditioning matters once it is tested.** P1's SDA3 was inert by construction (training DA params "
             "equalled the true ones), and every S1 evaluation before 2026-09-25 fed the conditioned priors the TRUE params. "
             f"With the biased DA params at S1, SDA2-M degrades ({fmt(m(*sd2, 'reg'))} -> {fmt(m(*sd2, 'reg', 's1'))} regular) "
             f"while SDA3-fix-M, trained on noisy DA params, does not ({fmt(m(*sd3, 'reg'))} -> {fmt(m(*sd3, 'reg', 's1'))}). "
             "Alone the gap is within seed noise; as the hybrid prior it decides robustness to model error (finding 1).")
    A.append("7. **Marginal value of observations**: the Strong-4D-Var collapse under model error survives the fast_weights "
             "fix (6.4-7.0x); the filters' 1.9x does not (1.1x at the original setting, 1.6-1.7x at the benchmark inflation).")
    e3, p1 = ("Flow ensemble, 1200 ep (3 networks)", "PredictStateCFM-M x3"), ("Benchmark default, 1200 ep", "PredictStateCFM-M")
    A.append(f"8. **Flow ensembles**: averaging the velocities of the three 1200-epoch PredictStateCFM-M seeds (equal weights, "
             f"one shared trajectory) gives regular / random S0 {fmt(m(*e3, 'reg'))} / {fmt(m(*e3, 'can'))} vs "
             f"{fmt(m(*p1, 'reg'))} / {fmt(m(*p1, 'can'))} for a single network, with unchanged calibration -- at 3x the "
             "parameters and sampling cost. tau-varying weights (a PredictStateCFM -> VanillaCFM hand-over, or random "
             "schedules) add nothing over equal weights (`docs/results/l96_cfm_velocity_ensembles.md`).\n")
    return A


def main() -> None:
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    da = da_main()
    learned, errors = learned_main(cache)
    manifest = json.load(open(str(CAN_TEST) + ".manifest.json"))
    for tag in ("inf2.0", "s0_inf1.5"):
        errors += check_da(str(DA243 / f"per_window_rlayout_n10-100_k4-16_w200_d1_{tag}.npz"), manifest)
    errors += check_da(str(DAHERE / "per_window_rlayout_n10-100_k4-16_w200_d1_infs0-1.5_s1-2.0.npz"), manifest)
    for sub, root in (("l96_testsets_factorial", FACT), ("l96_testsets_ood", OOD)):
        for ts in sorted((SHARED / sub).glob("l96_testset_*.pt")):
            cell = ts.stem.replace("l96_testset_", "")
            mf = json.load(open(str(ts) + ".manifest.json"))
            d = torch.load(ts, weights_only=False)
            tr = {c: np.stack([w["true_state"].numpy()[:, IDX] for w in d[f"test_{c}"]]) for c in CASES}
            for res in sorted((root / cell).glob("*/*")):
                if done(res):
                    errors += check_learned(str(res), str(ts), tr)
            del mf
    if errors:
        for e in errors:
            print("FAIL", e)
        sys.exit("consistency check failed; report not written")
    OUT.mkdir(parents=True, exist_ok=True)
    A = ["# L96 benchmark -- extended results (training budget, observing-system dependence, SDA, hybrid)\n",
         "Follow-up to `l96_benchmark_default.md`, same inputs: the 200 P1 test windows on the regular 30-obs set and "
         "on the canonical random observing system (the exact obs the DA baselines assimilated). S0 = true parameters, "
         "S1 = biased DA model / corrupted forcing. Every learned result passed the test-set consistency check "
         "(dataset + truth window for window); DA runs matched the canonical layouts.\n",
         "**Protocol changes since the benchmark-default report**: DA CRPS on the *analysis* ensemble (the old "
         "`_ESAccumulator` scored the forecast ensemble); SDA guidance weight 25 (validation-tuned; P1 used 20); SDA3 "
         "retrained with its bias conditioning actually active (SDA3-fix); a DirectUNet -> SDA hybrid tuned on "
         "validation windows; 1200-epoch arms of the benchmark default.\n",
         "**Flow sampler**: every VanillaCFM / PredictStateCFM row, curve and probe is scored with the benchmark "
         "protocol of #257 -- 30 members x 20 early-fine Euler steps (`tau_k = 1 - (1 - k/20)^0.5`, `ens30_no20`); "
         "the report refuses to render a flow result recorded with any other sampling. Only the calibration study "
         "(section 6) varies the sampler, on the uniform grid, by design. SDA and the hybrid use the SDA sampler "
         "(10 guided steps). The `PredictStateCFM-M x3` row averages the velocities of three trained networks "
         "(`models/cfm_blend.py`), so it costs 3x a single-model row.\n",
         "**SDA conditioning at S1**: the params-conditioned priors (SDA2, SDA3-fix, alone and as hybrid priors) are "
         "conditioned on the *biased DA-model* params (`*_da`, +10%) and the corrupted forcing -- the same model the DA "
         "baselines assimilate with. Results before 2026-09-25 fed them the TRUE params at S1 (the eval collate read "
         "the plain keys, which hold the truth in S1 windows); every affected run was re-evaluated.\n"]
    A += findings(da, learned)
    A += main_tables(da, learned)
    A.append("")
    A += budget_section(cache)
    A += density_sections(cache)
    A += probes_section(cache)
    A += tuning_section(cache)
    A += marginal_section()
    A += ["\n## Caveats\n",
          "- Single-seed rows: P1 references, SDA1-S+/L, the 3000-window run, all probes and the factorial SDA columns.",
          "- The hybrid's 400-epoch mean was tuned and evaluated with the 400-epoch DirectUNet; the 1200-epoch-mean "
          "hybrid uses the same validation-selected setting (re-checked on validation, section 6).",
          "- Probes and factorial cells use 20 windows x 3 draws, not the 200-window test sets.",
          "- Flow calibration was only probed (sampling-time settings); the benchmark keeps 10 integration steps."]
    (OUT / "l96_benchmark_extended.md").write_text("\n".join(A) + "\n")
    print(f"wrote {OUT / 'l96_benchmark_extended.md'}")


if __name__ == "__main__":
    main()
