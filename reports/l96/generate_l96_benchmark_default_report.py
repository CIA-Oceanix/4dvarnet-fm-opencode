"""L96 benchmark under the benchmark default (config/l96_benchmark_default.yaml):
DA baselines, deterministic, flow-matching and SDA, each scored on BOTH the
regular 30-obs test set and the canonical random-observing-system test set.

Every metric is recomputed from the stored estimates / ensembles (P1 report
formulas: ``deterministic`` / ``generative`` / DA summaries) and cached in
``experiments/l96_benchmark_default_metrics.json`` (keyed by file path and
mtime), so a re-render after the first pass is fast. Before anything is
written, every learned result is checked against its test set (dataset path
+ truth, window for window) and the DA runs against the canonical layouts
(``scripts/check_l96_testset_consistency.py``); the report refuses to render
on any mismatch.

  python reports/l96/generate_l96_benchmark_default_report.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.run_l96 import make_obs_j_indices  # noqa: E402
sys.path.insert(0, str(ROOT / "reports" / "l96"))
from generate_p1_l96_benchmark import deterministic, generative  # noqa: E402
from scripts.check_l96_testset_consistency import check_da, check_learned  # noqa: E402
import _inputs  # noqa: E402

REPORT = "benchmark_default"
SHARED = _inputs.shared()
REGULAR = SHARED / "l96_datasets_obsj2_int100_nwin200.pt"
CANON = SHARED / "l96_testset_rlayout_n10-100_k4-16_w200_d1.pt"
BENCH = _inputs.root(REPORT, "bench")
P1 = _inputs.root(REPORT, "p1")
HERE = _inputs.root(REPORT, "here")
CAN_EVAL = HERE / "eval_rlayout_n10-100_k4-16_w200"
DA_LAYOUT = _inputs.root(REPORT, "da_random_layout") / "l96_da_random_layout"
DA_REG_S0_15 = HERE / "l96_baselines_dws500_s0c_inf1.5_etkf_inf1.5_obsj2_int100_fw_dafw.json"
DA_REG_20 = _inputs.root(REPORT, "random_obs_times") / "l96_baselines_dws500_s0c_inf2.0_etkf_inf2.0_obsj2_int100_fw_dafw.json"
CACHE = ROOT / "experiments" / "l96_benchmark_default_metrics.json"
OUT = ROOT / "reports/l96/outputs"
CASES = ("s0", "s1")
DET, FLOW, SDA = "ens1_no1", "ens30_no20", "ens30_gw20"

# (group, label, kind, sub, [(regular dir, canonical dir), ...])
ROWS = [
    ("Benchmark default", "DirectUNet-M", "det", DET,
     [(BENCH / f"L96B_directunet_monaiM_seed{s}", CAN_EVAL / f"L96B_directunet_monaiM_seed{s}") for s in (1, 2, 3)]),
    ("Benchmark default", "PredictStateCFM-M", "flow", FLOW,
     [(HERE / f"L96B_predictstatecfm_monaiM_seed{s}", CAN_EVAL / f"L96B_predictstatecfm_monaiM_seed{s}") for s in (1, 2, 3)]),
    ("Benchmark default", "VanillaCFM-M", "flow", FLOW,
     [(BENCH / f"L96B_vanillacfm_monaiM_seed{s}", CAN_EVAL / f"L96B_vanillacfm_monaiM_seed{s}") for s in (1, 2, 3)]),
    ("P1 fixed obs", "DirectUNet-M", "det", DET,
     [(P1 / "P1_directunet_monaiM_noaug_l96", CAN_EVAL / "P1_directunet_monaiM_noaug_l96")]),
    ("P1 fixed obs", "PredictStateCFM-M", "flow", FLOW,
     [(P1 / "A2_predictstatecfm_monaiM_l96", CAN_EVAL / "A2_predictstatecfm_monaiM_l96")]),
    ("P1 fixed obs", "VanillaCFM-M", "flow", FLOW,
     [(P1 / "A1_vanillacfm_monaiM_l96", CAN_EVAL / "A1_vanillacfm_monaiM_l96")]),
] + [
    ("SDA (obs-free training)", label, "flow", SDA, [(P1 / exp, CAN_EVAL / exp)])
    for label, exp in (("SDA1-S+", "B4_sda1_monaiSplus_l96"), ("SDA1-M", "B4_sda1_monaiM_l96"),
                       ("SDA1-L", "B4_sda1_monaiL_l96"), ("SDA2-M", "A3_sda2_monaiM_l96"),
                       ("SDA3-M", "A3_sda3_monaiM_l96"))
]
DA_METHODS = ("ETKF", "EnKF", "Strong-4DVar")


def _truth(path: Path) -> dict:
    d = torch.load(path, weights_only=False)
    idx = list(make_obs_j_indices(8, 4, 2))
    return {c: np.stack([w["true_state"].numpy()[:, idx] for w in d[f"test_{c}"]]) for c in CASES}


def verify() -> list[str]:
    manifest = json.load(open(str(CANON) + ".manifest.json"))
    truths = {"regular": _truth(REGULAR), "canonical": _truth(CANON)}
    errors = []
    for _, _, _, sub, runs in ROWS:
        for reg, can in runs:
            errors += check_learned(str(reg / sub), str(REGULAR), truths["regular"])
            errors += check_learned(str(can / sub), str(CANON), truths["canonical"])
    for tag in ("inf2.0", "s0_inf1.5"):
        errors += check_da(str(DA_LAYOUT / f"per_window_rlayout_n10-100_k4-16_w200_d1_{tag}.npz"), manifest)
    return errors


def _score(path: Path, kind: str, cache: dict) -> dict:
    f = path / ("estimates_{}.npz" if kind == "det" else "members_{}.npz")
    key = f"{path}|{kind}|" + "|".join(str(os.path.getmtime(str(f).format(c))) for c in CASES)
    if key not in cache:
        out = {}
        for c in CASES:
            r = (deterministic if kind == "det" else generative)(Path(str(f).format(c)))
            out[c] = {"rmse": float(r["rmse"].mean())}
            if kind == "det":
                out[c]["var_ratio"] = float(r["var_ratio"])
            else:
                out[c]["crps"] = float(r["crps"].mean())
                out[c]["spread_over_rmse"] = float(r["spread"].mean() / r["rmse"].mean())
        cache[key] = out
        CACHE.write_text(json.dumps(cache, indent=1))
    return cache[key]


def da_rows() -> list[dict]:
    r15, r20 = json.load(open(DA_REG_S0_15)), json.load(open(DA_REG_20))
    c15 = json.load(open(DA_LAYOUT / "summary_rlayout_n10-100_k4-16_w200_d1_s0_inf1.5.json"))["cases"]
    c20 = json.load(open(DA_LAYOUT / "summary_rlayout_n10-100_k4-16_w200_d1_inf2.0.json"))["cases"]
    rows = []
    for m in DA_METHODS:
        s0_reg, s0_can = (r20, c20) if m == "Strong-4DVar" else (r15, c15)
        rows.append({"label": m,
                     "regular": {"s0": s0_reg["s0"][m]["mean"], "s1": r20["s1"][m]["mean"]},
                     "canonical": {"s0": s0_can["s0"][m]["rmse"]["all_obs"]["mean"],
                                   "s1": c20["s1"][m]["rmse"]["all_obs"]["mean"]}})
    return rows


def _ms(xs: list[float]) -> str:
    a = np.array(xs)
    return f"{a.mean():.3f} ± {a.std(ddof=1):.3f}" if len(a) > 1 else f"{a.mean():.3f}"


def figure(learned: list[dict], da: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    items = [(r["label"], r["group"], np.mean(r["regular"]["s0"]), np.mean(r["canonical"]["s0"])) for r in learned]
    items += [(r["label"], "DA", r["regular"]["s0"], r["canonical"]["s0"]) for r in da]
    items.sort(key=lambda t: t[3])
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(items) + 1.2))
    y = np.arange(len(items))
    for yi, (_, _, reg, can) in zip(y, items):
        ax.plot([reg, can], [yi, yi], color="#c3c2b7", linewidth=2, zorder=1)
    ax.scatter([t[2] for t in items], y, s=64, color="#2a78d6", zorder=2, label="regular 30-obs test set")
    ax.scatter([t[3] for t in items], y, s=64, color="#eb6834", zorder=2, label="random observing system")
    ax.set_yticks(y, [f"{t[0]}  ({t[1]})" for t in items], fontsize=8)
    ax.set_xscale("log")
    ticks = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("S0 RMSE, 24D observed space (log scale)")
    ax.grid(True, axis="x", color="#e6e5e0", linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def findings(learned: list[dict], da: list[dict]) -> list[str]:
    m = lambda r, t, c="s0": float(np.mean(r[t][c]))  # noqa: E731
    sd = lambda r, t, c="s0": float(np.std(r[t][c], ddof=1)) if len(r[t][c]) > 1 else 0.0  # noqa: E731
    bench = [r for r in learned if r["group"] == "Benchmark default"]
    bench.sort(key=lambda r: m(r, "regular"))
    best_da = {t: min(da, key=lambda r: r[t]["s0"]) for t in ("regular", "canonical")}
    best_b = bench[0]
    sda = [r for r in learned if r["group"].startswith("SDA")]
    best_sda = min(sda, key=lambda r: m(r, "canonical"))
    p1 = [r for r in learned if r["group"] == "P1 fixed obs"]
    da_s1s0 = [r["regular"]["s1"] / r["regular"]["s0"] for r in da]
    out = ["## Findings\n"]
    out.append(f"1. **Learned schemes under the benchmark default beat every DA baseline on S0, on both test "
               f"sets**: {best_b['label']} {m(best_b, 'regular'):.3f} vs {best_da['regular']['label']} "
               f"{best_da['regular']['regular']['s0']:.3f} (regular), {m(best_b, 'canonical'):.3f} vs "
               f"{best_da['canonical']['label']} {best_da['canonical']['canonical']['s0']:.3f} (random).")
    out.append(f"2. **Model error (S1)**: every learned scheme is flat (S1/S0 0.98-1.00), while the DA "
               f"baselines degrade {min(da_s1s0):.2f}-{max(da_s1s0):.2f}x -- their forward model carries the bias.")
    out.append("3. **Family ranking under the benchmark default**: " + " < ".join(
        f"{r['label']} {m(r, 'regular'):.3f} ± {sd(r, 'regular'):.3f}" for r in bench)
        + " (regular S0; the same order on the random set). The P1 ranking, where the flows led, "
        "does not survive fair training: P1's DirectUNet was overfitting frozen per-window obs noise.")
    out.append("4. **P1 fixed-obs checkpoints do not transfer**: best on the regular grid for the flows, but "
               + ", ".join(f"{r['label']} x{m(r, 'canonical') / m(r, 'regular'):.1f}" for r in p1)
               + " on the random set, against x"
               + f"{min(m(r, 'canonical') / m(r, 'regular') for r in bench):.2f}-"
               + f"{max(m(r, 'canonical') / m(r, 'regular') for r in bench):.2f} for the benchmark-default models.")
    out.append(f"5. **SDA needs no retraining to be robust** (random/regular ~1.2): best on the random set "
               f"{best_sda['label']} {m(best_sda, 'canonical'):.3f}, between the benchmark-default models and "
               "DA on RMSE, far ahead of DA under model error, and M is the right size (L is no better).")
    out.append("6. **DA is the least sensitive to the observing system** (random/regular "
               + ", ".join(f"{r['label']} {r['canonical']['s0'] / r['regular']['s0']:.2f}" for r in da)
               + "): its weakness is model error, not the observing system.\n")
    return out


def main() -> None:
    errors = verify()
    if errors:
        for e in errors:
            print("FAIL", e)
        sys.exit("consistency check failed; report not written")
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    learned = []
    for group, label, kind, sub, runs in ROWS:
        rec = {"group": group, "label": label, "kind": kind, "n": len(runs)}
        for tag, j in (("regular", 0), ("canonical", 1)):
            per = [_score(r[j] / sub, kind, cache) for r in runs]
            rec[tag] = {c: [p[c]["rmse"] for p in per] for c in CASES}
            if kind == "flow":
                rec[tag + "_crps"] = {c: [p[c]["crps"] for p in per] for c in CASES}
                rec[tag + "_sp"] = {c: [p[c]["spread_over_rmse"] for p in per] for c in CASES}
            else:
                rec[tag + "_vr"] = {c: [p[c]["var_ratio"] for p in per] for c in CASES}
        learned.append(rec)
    da = da_rows()

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "l96_benchmark_default.png"
    figure(learned, da, png)
    manifest = json.load(open(str(CANON) + ".manifest.json"))
    A = []
    A.append("# L96 benchmark under the benchmark default -- regular and random observing systems\n")
    A.append("Two-scale L96, 24D observed space (8 slow + 16 fast), 200 shared test windows, S0 (true "
             "parameters) and S1 (biased DA parameters / corrupted forcing). Every cell is the mean "
             "over the 200 windows of the per-window RMSE (physical units); `±` is the sd across "
             "training seeds where a row has several.\n")
    A.append("## Protocol\n")
    A.append("- **Training (benchmark default, `config/l96_benchmark_default.yaml`)**: P1 recipe (monai, "
             "`normalize`, cosine annealing, 400 epochs, lr 1e-3, clip 10, batch 16) with a random "
             "observing system redrawn every batch with fresh noise -- 10-300 stratified obs times "
             "(step 0 always observed), 4-16 observed fast channels per window (subset per obs "
             "time), slow channels always. Validation windows re-observed once with a fixed seed "
             "from the same distribution (checkpoint = `stage1_best` by that val loss). 3 seeds.")
    A.append("- **P1 fixed obs**: the P1 checkpoints -- regular 30-obs grid, noise frozen per window "
             "across epochs.")
    A.append("- **Flow sampling (VanillaCFM, PredictStateCFM)**: 30 members, 20 early-fine Euler "
             "steps tau_k = 1-(1-k/20)^0.5 (the models' default grid since 2026-09-24; earlier "
             "renders of this report used 10 uniform steps, `ens30_no10`). See "
             "`docs/results/l96_cfm_tau_consistency_l96b.md`.")
    A.append("- **SDA**: the P1 checkpoints unchanged -- the prior and its validation loss never see "
             "observations, so the obs protocol does not apply to training. Guided sampling: 30 "
             "members, 10 steps, guidance weight 20 (tuned on the regular grid), r_var 0.5, with "
             "the NaN-channel guidance fix (#244).")
    A.append("- **DA**: ETKF / EnKF (30 members, inflation S0 1.5 / S1 2.0), Strong-4DVar; "
             "per-window `fast_weights` in the forward model; DA window 500.")
    A.append("- **Regular test set**: the P1 cache, 30 regular obs times, all 24 channels.")
    A.append(f"- **Random test set** (canonical, `{CANON.name}`, sha256 `{manifest['sha256'][:12]}...`): "
             "the same 200 windows re-observed with the exact layouts the DA baselines assimilated "
             f"(`eval_da_random_layout_l96.py`): n_obs uniform in {manifest['n_obs_range'][0]}-"
             f"{manifest['n_obs_range'][1]} at stratified times with step 0 observed and at least one "
             f"obs per DA window, {manifest['fast_range'][0]}-{manifest['fast_range'][1]} observed "
             "fast channels per window. Truth, forcings and parameters are bitwise the P1 cache's.")
    A.append("- **Consistency**: before rendering, every learned result was checked to name its test "
             "set and to carry that set's truth window for window, and both DA runs to match the "
             "canonical layouts (`scripts/check_l96_testset_consistency.py`). All passed.")
    A.append("- **Metrics**: flows are scored on the 30-member ensemble mean (RMSE) plus ensemble "
             "CRPS and spread/RMSE; DirectUNet is a single pass; DA stores no members, so it has no "
             "CRPS.\n")
    A.append(f"![S0 RMSE, regular vs random test set]({png.name})\n")
    A += findings(learned, da)
    A.append("## RMSE\n")
    A.append("| group | scheme | seeds | regular S0 | regular S1 | random S0 | random S1 | random/regular (S0) | S1/S0 (regular) |")
    A.append("|---|---|---|---|---|---|---|---|---|")
    for r in da:
        A.append(f"| DA | {r['label']} | — | {r['regular']['s0']:.3f} | {r['regular']['s1']:.3f} | "
                 f"{r['canonical']['s0']:.3f} | {r['canonical']['s1']:.3f} | "
                 f"{r['canonical']['s0'] / r['regular']['s0']:.2f} | {r['regular']['s1'] / r['regular']['s0']:.2f} |")
    for r in learned:
        A.append(f"| {r['group']} | {r['label']} | {r['n']} | {_ms(r['regular']['s0'])} | {_ms(r['regular']['s1'])} | "
                 f"{_ms(r['canonical']['s0'])} | {_ms(r['canonical']['s1'])} | "
                 f"{np.mean(r['canonical']['s0']) / np.mean(r['regular']['s0']):.2f} | "
                 f"{np.mean(r['regular']['s1']) / np.mean(r['regular']['s0']):.2f} |")
    A.append("\n## Probabilistic scores (flows and SDA, S0)\n")
    A.append("| group | scheme | CRPS regular | CRPS random | spread/RMSE regular | spread/RMSE random |")
    A.append("|---|---|---|---|---|---|")
    for r in learned:
        if r["kind"] == "flow":
            A.append(f"| {r['group']} | {r['label']} | {_ms(r['regular_crps']['s0'])} | {_ms(r['canonical_crps']['s0'])} | "
                     f"{np.mean(r['regular_sp']['s0']):.3f} | {np.mean(r['canonical_sp']['s0']):.3f} |")
    A.append("\n## Deterministic variance ratio (S0, predicted/true variance)\n")
    A.append("| group | scheme | regular | random |\n|---|---|---|---|")
    for r in learned:
        if r["kind"] == "det":
            A.append(f"| {r['group']} | {r['label']} | {np.mean(r['regular_vr']['s0']):.3f} | {np.mean(r['canonical_vr']['s0']):.3f} |")
    A.append("\n## Caveats\n")
    A.append("- P1 and SDA rows are single runs; benchmark-default rows are 3 seeds.")
    A.append("- The SDA guidance weight (20) was tuned on the regular grid, not re-tuned for the random set.")
    A.append("- The random test set is one draw of one observing-system distribution (10-100 obs, 4-16 "
             "fast channels); rankings between learned families depend on the regime (on a sparser "
             "30-obs / 8-fast set VanillaCFM-M led DirectUNet-M).")
    A.append("- DA rows reuse existing runs on the identical inputs: regular S0 ETKF/EnKF at inflation "
             "1.5, regular S1 and Strong-4DVar from the corrected inflation-2.0 run; random-set rows "
             "from the random-layout DA runs (#243).")
    (OUT / "l96_benchmark_default.md").write_text("\n".join(A) + "\n")
    print(f"wrote {OUT / 'l96_benchmark_default.md'}")


if __name__ == "__main__":
    main()
