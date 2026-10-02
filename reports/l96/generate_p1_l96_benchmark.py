#!/usr/bin/env python3
"""P1 L96 benchmark: DA, deterministic, flow-matching and SDA schemes.

Sections 1-4 and the cross-family reading use the benchmark-default evaluation
framework (``p1_benchmark_sections.py``): models trained on a random observing
system with fresh noise every batch, 1200 epochs, 3 seeds, scored on the regular
and the canonical random test sets with the extended report's own code.

Annex A keeps the P1 protocol for the sensitivity analyses the benchmark framework
does not cover (S+ / M / L tiers, CFM parameterization, SDA conditioning): every
annex row is trained under a single declared recipe -- monai backbone,
`data.normalize: true`, cosine annealing, 400 epochs, batch 16, fixed regular
observing system, NO obs-density augmentation -- so that architecture and
capacity are the only axes that move.
Unrolled/4DVarNet schemes are deliberately OUT of scope for P1; the two FDV1CFM
rows that appear are diagnostics, marked as such, and excluded from the
cross-family verdict.

Structure follows `generate_l96_consolidated_report.py`: a scheme table that
carries the setup of each row, then one metric table per family, then the
cross-family reading. Metrics are recomputed here from the stored ensembles
(`evaluation/estimate_metrics.py` conventions) rather than read from cached
JSON, so the table cannot drift from the artifacts.

Three protocol facts that make or break comparability, all enforced below:

* **Generative rows** are scored on `ens30_no20` -- 30 members, 20 early-fine
  Euler steps tau_k = 1-(1-k/20)^0.5 (the benchmark flow sampler since
  2026-09-24; earlier renders used `ens30_no10`, 10 uniform steps) -- as
  ensemble-mean RMSE plus a proper ensemble CRPS. **Deterministic rows** are
  a single pass. A one-draw score of a generative model is a strictly worse
  estimator and is never mixed into the same column.
* **SDA rows** must come from `eval_sda_l96.py` with a guidance weight: SDA1 is
  an UNCONDITIONAL prior whose observations enter only as an inference-time
  guidance cost, so the unguided sampler returns climatological spread
  (RMSE ~1.64), not a DA result. `gw=20` is tuned (swept 10-80 on SDA1-M; flat
  basin 20-30, inherited `gw=40` was 8% worse).
* **tau=0 rows** are `mu(x0, tau=0, y)` averaged over 30 draws. At tau=0,
  `x_tau = x0` is independent of `x1`, so `E[x1|x_tau,y] = E[x1|y]` exactly --
  these are the flow's own implied posterior mean used as a point estimator.
  They cost 30 model calls against DirectUNet's 1, which the table states.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.estimate_metrics import _groups_from_per_window  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _inputs  # noqa: E402
import per_window_summary  # noqa: E402
import fm_score_summary  # noqa: E402
import p1_benchmark_sections as p1s  # noqa: E402

HERE = _inputs.root("p1_benchmark", "here")
DA_BASES = (HERE, _inputs.shared())

DA_CURRENT = HERE / "da_current_2026-09-29"
_DA_NEW = "l96_baselines_trajectories_dws500_s0c_crps_infs0-1.2_s1-3.0_etkf_infs0-1.15_s1-2.5_obsj2_int100_fw_dafw.npz"
DA_SOURCES = {
    "ETKF": (_DA_NEW, "ETKF", "λ 1.15 / 2.5"),
    "EnKF": (_DA_NEW, "EnKF", "λ 1.2 / 3.0"),
    "ETKS": ("l96_baselines_trajectories_dws500_s0c_test_etks-correct-Lfull-infs0-1.15_s1-2.5_infs0-1.2_s1-3.0"
             "_etkf_infs0-1.15_s1-2.5_obsj2_int100_fw_dafw.npz", "ETKS", "λ 1.15 / 2.5"),
    "Strong-4DVar": ("l96_baselines_trajectories_dws500_s0c_inf2.0_etkf_inf2.0_obsj2_int100_fw_dafw.npz",
                     "Strong_4DVar", "—"),
}
DA_METHODS = list(DA_SOURCES)
CASES = ("s0", "s1")


def make_obs_j_indices(no: int = 8, j_truth: int = 4, j_obs: int = 2) -> np.ndarray:
    """The 24D observed subspace inside the full 40D state (8 slow + 8x2 fast).

    Must match `generate_l96_consolidated_report.py` exactly: the DA caches store
    S0 in the full 40D state and S1 already reduced to 24D, so only S0 is indexed.
    """
    x_idx = list(range(no))
    y_idx = [no + k * j_truth + j for k in range(no) for j in range(j_obs)]
    return np.array(x_idx + y_idx)


DET = [
    ("DirectUNet-S+", "P1_directunet_monaiSplus_noaug_l96", "ens1_no1", "1.48 M", ""),
    ("DirectUNet-M", "P1_directunet_monaiM_noaug_l96", "ens1_no1", "5.89 M", ""),
    ("DirectUNet-L", "P1_directunet_monaiL_noaug_l96", "ens1_no1", "23.5 M", "unstable"),
]
FM = [
    ("VanillaCFM-S+", "A1_vanillacfm_monaiSplus_l96", "ens30_no20", "1.48 M", ""),
    ("VanillaCFM-M", "A1_vanillacfm_monaiM_l96", "ens30_no20", "5.89 M", ""),
    ("VanillaCFM-L", "A1_vanillacfm_monaiL_l96", "ens30_no20", "23.5 M", ""),
    ("PredictStateCFM-S+", "A2_predictstatecfm_monaiSplus_l96", "ens30_no20", "1.48 M", ""),
    ("PredictStateCFM-M", "A2_predictstatecfm_monaiM_l96", "ens30_no20", "5.89 M", ""),
    ("PredictStateCFM-L(lr3e-4)", "A2_predictstatecfm_monaiL_lr3e4_l96", "ens30_no20", "23.5 M", "lr 3e-4"),
    ("PredictStateCFM-L(lr5e-4)", "A2_predictstatecfm_monaiL_lr5e4_l96", "ens30_no20", "23.5 M", "lr 5e-4"),
]
SDA = [
    ("SDA1-S+", "B4_sda1_monaiSplus_l96", "ens30_gw20", "1.48 M", ""),
    ("SDA1-M", "B4_sda1_monaiM_l96", "ens30_gw20", "5.89 M", ""),
    ("SDA1-L", "B4_sda1_monaiL_l96", "ens30_gw20", "23.5 M", ""),
    ("SDA2-M", "A3_sda2_monaiM_l96", "ens30_gw20", "5.89 M", ""),
    ("SDA3-M", "A3_sda3_monaiM_l96", "ens30_gw20", "5.89 M", "bias never active in training (DA params = truth): identical to SDA2 by construction"),
    ("SDA3-fix-M", "A3_sda3fix_monaiM_l96_seed1", "ens30_gw20", "5.89 M", "SDA3 with the noisy-DA-bias conditioning active (added 2026-09-26)"),
]
# tau=0 mean components, read from eval_mean_component.py's JSON.
TAU0_LABELS = {
    "VanillaCFM S+": "VanillaCFM-S+ (tau=0)",
    "VanillaCFM M": "VanillaCFM-M (tau=0)",
    "VanillaCFM L": "VanillaCFM-L (tau=0)",
    "PredictStateCFM S+": "PredictStateCFM-S+ (tau=0)",
    "PredictStateCFM M": "PredictStateCFM-M (tau=0)",
    "PredictStateCFM L (lr3e-4)": "PredictStateCFM-L (tau=0)",
}


def _crps(members: np.ndarray, truth: np.ndarray) -> np.ndarray:
    M = members.shape[-1]
    mae = np.abs(members - truth[:, :, :, None]).mean(axis=(1, 3))
    coeff = 2 * np.arange(M) - M + 1
    pairwise = (2 * np.einsum("k,wtdk->wtd", coeff, np.sort(members, axis=-1))
                / (M * M)).mean(axis=1)
    return _groups_from_per_window(mae - 0.5 * pairwise)["all_obs"]


def generative(path: Path, group: str = "all_obs"):
    if not path.exists():
        z = np.load(path.with_name(path.name.replace("members_", "scores_")))
        return {k: _groups_from_per_window(z[k].astype(np.float64))[group] for k in ("rmse", "crps", "spread")}
    z = np.load(path)
    m = z["members"].astype(np.float64)
    t = z["truth"].astype(np.float64)
    rmse = _groups_from_per_window(np.sqrt(((m.mean(-1) - t) ** 2).mean(axis=1)))[group]
    crps = _crps(m, t)
    spread = _groups_from_per_window(m.std(axis=-1, ddof=1).mean(axis=1))[group]
    del z, m, t
    return dict(rmse=rmse, crps=crps, spread=spread)


def deterministic(path: Path, group: str = "all_obs"):
    z = np.load(path)
    tr = z["trajectories"].astype(np.float64)
    t = z["truth"].astype(np.float64)
    rmse = _groups_from_per_window(np.sqrt(((tr - t) ** 2).mean(axis=1)))[group]
    mae = _groups_from_per_window(np.abs(tr - t).mean(axis=1))[group]
    var_ratio = float(tr.var(axis=1).mean() / t.var(axis=1).mean())
    del z, tr, t
    return dict(rmse=rmse, mae=mae, var_ratio=var_ratio)


def ms(a: np.ndarray) -> str:
    return f"{a.mean():.4f} ± {a.std(ddof=1):.4f}"


def best_idx(rows, key):
    vals = [r["m"][key].mean() for r in rows if r.get("m") and not r["flag"]]
    if not vals:
        return None
    return min(vals)


# Best learned scheme per subcategory, plus two DA references. Keys are the
# DISPLAY names used as figure row labels: `plot_hovmoller` passes each through
# the consolidated report's `short_name`, which is a dict lookup with a
# "split on the first underscore" fallback -- so underscore-free display names
# pass through unchanged, while raw experiment dir names would render as "A1".
EXT_HERE = _inputs.root("benchmark_extended", "here")
FIGURE_SCHEMES = [
    ("ETKS", ("da", "ETKS")),
    ("Strong-4DVar", ("da", "Strong-4DVar")),
    ("DirectUNet-M", ("est", EXT_HERE / "L96B_directunet_monaiM_ep1200_seed1" / "ens1_no1")),
    ("PredictStateCFM-M", ("est", EXT_HERE / "L96B_predictstatecfm_monaiM_ep1200_seed1" / "ens30_no20")),
    ("SDA1-M", ("est", EXT_HERE / "eval_regular" / "B4_sda1_monaiM_ep1200_l96_seed1" / "ens30_gw25")),
    ("Hybrid", ("est", EXT_HERE / "eval_regular" / "hybrid_DU1200s1_A3_sda3fix_monaiM_ep1200_l96" / "tau0.1_gw2")),
]


OBS_BAND_HALF_WIDTH = 5


def widen_obs(obs: np.ndarray, obs_times: np.ndarray, half_width: int = OBS_BAND_HALF_WIDTH):
    """Render each observation as a band of columns instead of a single column.

    The Obs row is the panel a reader uses to understand the observing system,
    and at 30 observations over a 3000-step window each one is 1/3000 of the
    axis -- roughly 0.15 px once rasterized. Most vanish, and *which* ones
    survive differs between the slow and fast panels because their axes round
    differently, which reads as two different observation patterns. There is
    only one: `obs_mask` is a single 1-D mask over time shared by every
    dimension. What actually differs between slow and fast is SPATIAL (all 8
    slow X are observed; only 16 of 32 fast Y, `obs_j=2` of `J=4`).

    Widening is display-only and slightly inflates the Obs row's |error| column,
    which `plot_hovmoller` computes as |obs - truth| and labels "obs noise":
    the band reuses one observation against a truth that has moved. Measured on
    window 187: half-width 5 costs +1.7% (0.7216 -> 0.7338), half-width 10 costs
    +6.2%. The caption carries the undistorted value.
    """
    wide = np.full_like(obs, np.nan)
    for t in obs_times:
        lo, hi = max(0, t - half_width), min(obs.shape[0], t + half_width + 1)
        wide[lo:hi] = obs[t]
    return wide


def true_obs_noise(obs: np.ndarray, obs_times: np.ndarray, truth: np.ndarray) -> float:
    """RMS |obs - truth| at the observed timesteps only, before any widening."""
    return float(np.sqrt(np.nanmean((obs[obs_times] - truth[obs_times]) ** 2)))


def figure_trajectories(case, truth_shape):
    """Per-window point estimates for the figure rows, or None if unavailable.

    Generative rows are reduced to their ENSEMBLE MEAN, which is the estimator
    their RMSE column scores -- plotting a single member would show a different
    (worse) object than the table reports.
    """
    est = {}
    idx = make_obs_j_indices()
    for label, spec in FIGURE_SCHEMES:
        if spec[0] == "da":
            fn, key_method, _ = DA_SOURCES[spec[1]]
            if not (DA_CURRENT / fn).exists():
                continue
            z = np.load(DA_CURRENT / fn)
            key = f"{case}_{key_method}_trajectories"
            if key not in z.files:
                continue
            traj = z[key].astype(np.float64)
            est[label] = traj[..., idx] if traj.shape[-1] > len(idx) else traj
        elif spec[0] == "est":
            path = spec[1] / f"estimates_{case}.npz"
            if not path.exists():
                continue
            est[label] = np.load(path)["trajectories"].astype(np.float64)
        else:
            kind, exp, sub = spec
            fn = "members" if kind == "gen" else "estimates"
            path = HERE / exp / sub / f"{fn}_{case}.npz"
            if not path.exists():
                continue
            z = np.load(path)
            est[label] = (z["members"].astype(np.float64).mean(-1) if kind == "gen"
                          else z["trajectories"].astype(np.float64))
        if est[label].shape != truth_shape:
            del est[label]
    return est


def _ratio(m, key):
    a, b = m["s1"][key].mean(), m["s0"][key].mean()
    return a / b if b else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="all_obs", choices=["slow", "obs_fast", "all_obs"])
    p.add_argument("--mean-json", default="reports/l96/outputs/p1_mean_component.json")
    p.add_argument("--output", default="reports/l96/outputs/p1_l96_benchmark.md")
    p.add_argument("--json-output", default="reports/l96/outputs/p1_l96_benchmark.json")
    args = p.parse_args()

    out, missing, truth = {}, [], {}
    for name, runs, kind in (("det", DET, "det"), ("fm", FM, "gen"), ("sda", SDA, "gen")):
        rows = []
        for label, exp, sub, params, flag in runs:
            fn = "members" if kind == "gen" else "estimates"
            m, ok = {}, True
            for case in CASES:
                path = HERE / exp / sub / f"{fn}_{case}.npz"
                scores = path.with_name(f"scores_{case}.npz")
                if not path.exists() and not (kind == "gen" and scores.exists()):
                    missing.append(f"{label} [{case}]: {path}")
                    ok = False
                    continue
                m[case] = generative(path, args.group) if kind == "gen" else deterministic(path, args.group)
                if case not in truth and path.exists():
                    z = np.load(path)
                    truth[case] = z["truth"].astype(np.float64)
                    del z
            rows.append(dict(label=label, params=params, flag=flag,
                             m=m if ok and len(m) == len(CASES) else None, exp=exp))
        out[name] = rows

    tau0 = []
    mj = ROOT / args.mean_json
    if mj.exists():
        for r in json.load(open(mj))["rows"]:
            if r["label"] in TAU0_LABELS:
                tau0.append(dict(label=TAU0_LABELS[r["label"]], rmse=r["rmse"]["mean"],
                                 rmse_std=r["rmse"]["std"], mae=r["crps_mae"]["mean"],
                                 draws=r["draw_dispersion"]))
    else:
        missing.append(f"tau=0 mean components (annex): {mj}")

    import generate_l96_benchmark_extended_report as ext
    cache = json.loads(ext.CACHE.read_text()) if ext.CACHE.exists() else {}
    bench, errors = p1s.collect(cache)
    if errors:
        for e in errors:
            print("FAIL", e)
        sys.exit("consistency check failed; report not written")
    tau0_bench = p1s.tau0_rows()
    if tau0_bench is None:
        missing.append(f"tau=0 mean components of the 1200-epoch flows: {p1s.TAU0}")
    annex_best = {k: min(r["m"]["s0"]["rmse"].mean() for r in out[k] if r["m"] and not r["flag"]) for k in ("det", "fm")}

    L = []
    A = L.append
    A("# P1 L96 benchmark — DA baselines, deterministic, flow-matching and SDA\n")
    A("Two-scale L96, `obs_j=2` (24D observed space: 8 slow + 16 fast), the 200 shared P1 test windows. "
      "**S0** = true parameters; **S1** = ±20% parameter perturbation with ±10% bias (the DA forward model "
      "uses the biased `*_da` values), so `S1/S0` is a robustness-to-model-error ratio: 1.0 means untouched. "
      "Every RMSE cell is the **per-window RMSE on the 24 observed channels, mean ± sd across the 200 windows**, "
      "in **physical units** (seeds pooled per window).\n")
    A("**Evaluation framework (2026-10-02).** Sections 1-4, the cross-family reading, the per-window summary, "
      "FMS_τ and the reconstruction examples all use the **benchmark-default** framework of "
      "`l96_benchmark_extended.md` and are scored by that report's code: learned models trained on a random "
      "observing system with fresh noise every batch (`config/l96_benchmark_default.yaml`), 1200 epochs, 3 seeds, "
      "M tier; DA at the per-method inflation of #295 plus the ETKS of #299; every scheme on **both** the regular "
      "30-obs test set and the canonical random observing system. Flagged rows (italic, with a note) are "
      "sensitivity rows and stay out of the bests. **Annex A** keeps the earlier P1 protocol (fixed regular "
      "observing system, frozen per-window noise, 400 epochs, one seed) for the analyses only it covers: "
      "S+ / M / L tiers, the CFM parameterization and SDA conditioning.\n")
    A("## Protocol\n")
    A("| family | protocol | columns |\n|---|---|---|")
    A("| DA baselines | per-window assimilation (dws=500, DA fast weights), 30 members; ETKS = the ETKF plus a "
      "full-window smoother | RMSE; analysis-ensemble CRPS and spread for ETKF/EnKF/ETKS; none for Strong-4DVar |")
    A("| Deterministic | DirectUNet, single forward pass | RMSE; `var ratio` = predicted / true temporal variance "
      "(1.0 = no smoothing) |")
    A("| Flow matching | `ens30_no20` (30 members, 20 early-fine Euler steps) | ensemble-mean RMSE; proper "
      "ensemble CRPS; `spread/RMSE` (1.0 calibrated) |")
    A("| SDA | guided `ens30`, 10 steps, `gw=25` (validation-tuned), `r_var=0.5`; the params-conditioned priors "
      "(SDA2, SDA3-fix) get the biased DA params at S1 | as above |")
    A("| tau=0 mean | `mu(x0, 0, y)` over 30 draws of each flow | the flow's implied posterior mean as a point "
      "estimator |")
    A(f"| Test sets | regular: the P1 cache, 30 regular obs times, all 24 channels; random: "
      f"`{ext.CAN_TEST.name}`, 10-100 obs at stratified times, 4-16 observed fast channels per window, the exact "
      "obs the DA baselines assimilated | every learned result is checked against its test set window for "
      "window before rendering |\n")
    A("A single draw from a generative model is a strictly worse estimator than its ensemble "
      "mean, so the two never share a column.\n")
    for sec in ("da", "det"):
        for line in p1s.table(sec, bench[sec]):
            A(line)
    for line in p1s.tau0_table(tau0_bench, bench["fm"]):
        A(line)
    for sec in ("fm", "sda"):
        for line in p1s.table(sec, bench[sec]):
            A(line)
    A("\nRows and their sources: `l96_benchmark_extended.md` (section 1 and its input bundle).\n")
    for line in p1s.reading(bench, tau0_bench, annex_best):
        A(line)
    for line in per_window_summary.section():
        A(line)
    for line in fm_score_summary.section():
        A(line)
    A("## Caveats\n")
    A("- DA baselines receive the same per-window parameters as truth generation (S0) or their "
      "biased `*_da` counterparts (S1); the learned schemes see observations only. This is what "
      "makes the comparison apples-to-apples, and also why their S1 behaviour differs so much.")
    A("- DA CRPS is the per-window analysis-ensemble CRPS stored by the runs (30 members), comparable "
      "to the generative families' ensemble CRPS.")
    A("- SDA is the only learned family with a tuned inference hyper-parameter (`gw`); the hybrid also tunes "
      "its start time and guidance (tau0 0.1, gw 2), on the validation windows.")
    A("- The benchmark framework has M-tier models only; tier, parameterization and conditioning comparisons are "
      "in annex A, under the P1 protocol.\n")
    # --- reconstruction figures -------------------------------------------
    fig_rows = []
    try:
        import importlib.util as _u

        import torch
        _spec = _u.spec_from_file_location(
            "_consol", str(Path(__file__).parent / "generate_l96_consolidated_report.py"))
        _c = _u.module_from_spec(_spec)
        _spec.loader.exec_module(_c)
        ds_path = next(b / "l96_datasets_obsj2_int100_nwin200.pt" for b in DA_BASES
                       if (b / "l96_datasets_obsj2_int100_nwin200.pt").exists())
        figs = ROOT / "reports" / "l96" / "outputs" / "figures"
        figs.mkdir(parents=True, exist_ok=True)
        for case in CASES:
            est = figure_trajectories(case, truth[case].shape)
            if not est:
                continue
            names = [lbl for lbl, _ in FIGURE_SCHEMES if lbl in est]
            ref = "Strong-4DVar" if "Strong-4DVar" in est else names[0]
            sel = _c.select_windows(est[ref], truth[case])
            ds = torch.load(ds_path, map_location="cpu", weights_only=False)[f"test_{case}"]
            for rank in ("best", "median", "worst"):
                wi, sr = sel[rank]
                w = ds[wi]
                fp = figs / f"p1_l96_hovm_{case}_{rank}.png"
                obs_raw = w["obs"].numpy().astype(np.float64)
                otimes = np.where(w["obs_mask"].numpy())[0]
                noise = true_obs_noise(obs_raw, otimes, truth[case][wi])
                _c.plot_hovmoller(fp, case, rank, wi, sr, names,
                                  {n: est[n][wi] for n in names}, truth[case][wi],
                                  widen_obs(obs_raw, otimes), otimes, 0.001)
                cells = [f"{np.sqrt(((est[n][wi] - truth[case][wi]) ** 2).mean()):.3f}"
                         for n in names]
                fig_rows.append((case, rank, wi, sr, names, cells,
                                 fp.relative_to(ROOT / "reports" / "l96" / "outputs"),
                                 len(otimes), noise))
    except Exception as exc:  # figures are a nice-to-have; never fail the tables
        missing.append(f"reconstruction figures: {type(exc).__name__}: {exc}")

    if fig_rows:
        names = fig_rows[0][4]
        A("\n## Reconstruction examples\n")
        A("Best / median / worst windows, ranked by Strong-4DVar per-window RMSE (the same "
          "convention as the consolidated benchmark, so window choices are comparable across "
          "reports). Rows are Truth / Obs / one scheme per family under the benchmark framework (seed 1 of each "
          "1200-epoch model; the hybrid as reference); columns are the "
          "state and |error| maps for the slow X (8D) and fast Y (16D) blocks. Generative rows "
          "are plotted as their **ensemble mean** — the estimator their RMSE column scores.\n")
        A("| case | rank | window | 4DVar win-RMSE | " + " | ".join(names) + " |")
        A("|---|---|---|---|" + "---|" * len(names))
        for case, rank, wi, sr, _n, cells, _fp, _nt, _no in fig_rows:
            A(f"| {case.upper()} | {rank} | {wi} | {sr:.3f} | " + " | ".join(cells) + " |")
        A("")
        nobs = fig_rows[0][7]
        A(f"**Observing system.** `obs_mask` is a single 1-D mask over time shared by every "
          f"dimension, so slow and fast are observed at exactly the same instants: **{nobs} "
          "observed timesteps per window** (`obs_interval=100` over T=3000). What differs "
          "between them is spatial, not temporal — all 8 slow X are observed, but only 16 of "
          "the 32 fast Y (`obs_j=2` of `J=4`), which is why the fast block is 16 rows.\n")
        A(f"Each observation is drawn as a {2 * OBS_BAND_HALF_WIDTH + 1}-column band rather "
          "than a single 1/3000 column, which would be ~0.15 px and mostly vanish under "
          "rasterization. That is display-only and inflates the Obs row's |error| column "
          "(labelled 'obs noise') by ~1.7%; the undistorted values are "
          + ", ".join(f"{c.upper()}/{r} {no:.4f}" for c, r, _w, _s, _n, _c2, _f, _nt, no
                      in fig_rows) + ".\n")
        for case, rank, _wi, _sr, _n, _c2, fp, _nt, _no in fig_rows:
            A(f"![{case.upper()} {rank}]({fp})")
        A("")

    A("\n## Annex A. P1 protocol: tier, parameterization and conditioning sensitivity\n")
    A("**Not the benchmark framework.** Every row of this annex is a P1-protocol checkpoint: fixed regular "
      "30-obs observing system with the observation noise frozen per window across epochs, 400 epochs, one "
      "seed, scored on the regular test set only (the same 200 windows as sections 1-4). It is kept for the "
      "comparisons the benchmark framework does not have: S+ / M / L tiers, VanillaCFM vs PredictStateCFM, "
      "and SDA conditioning (SDA1 / SDA2 / SDA3). Absolute numbers are **not** comparable with sections 1-4: "
      "under this protocol DirectUNet overfits the frozen noise (DirectUNet-M 0.470 here vs 0.340 in section 2), "
      "while the flows barely move. The SDA rows use guidance weight 20 (the benchmark: 25).\n")
    def emit(title, rows, mode):
        A(f"\n### {title}\n")
        if mode == "gen":
            A("| scheme | params | S0 RMSE | S1 RMSE | S1/S0 | S0 CRPS | S1 CRPS | sp/RMSE (S0) | note |")
            A("|---|---|---|---|---|---|---|---|---|")
        else:
            extra = "var ratio" if mode == "det" else "sp/RMSE (S0)"
            err = "CRPS (MAE for 4D-Var)" if mode == "da" else "MAE"
            A(f"| scheme | params | S0 RMSE | S1 RMSE | S1/S0 | S0 {err} | S1 {err} | {extra} | note |")
            A("|---|---|---|---|---|---|---|---|---|")
        avail = [r for r in rows if r["m"] and not r["flag"]]
        best = min((r["m"]["s0"]["rmse"].mean() for r in avail), default=None)
        for r in rows:
            if not r["m"]:
                A(f"| {r['label']} | {r['params']} | _missing_ | | | | | | |")
                continue
            b = "**" if best is not None and abs(r["m"]["s0"]["rmse"].mean() - best) < 1e-12 else ""
            note = ""
            if r["flag"] == "unstable":
                note = "single unreliable run — an identical config gave 0.4705 on another seed"
            elif r["flag"].startswith("lr "):
                note = f"{r['flag']} — the standard 1e-3 diverged at this tier"
            elif r["flag"]:
                note = r["flag"]
            if mode == "gen":
                c0, c1 = ms(r["m"]["s0"]["crps"]), ms(r["m"]["s1"]["crps"])
                sp = f"{r['m']['s0']['spread'].mean() / r['m']['s0']['rmse'].mean():.3f}"
            elif mode == "da" and "crps" in r["m"]["s0"]:
                c0, c1 = f"{r['m']['s0']['crps'].mean():.4f}", f"{r['m']['s1']['crps'].mean():.4f}"
                sp = (f"{r['m']['s0']['spread'].mean() / r['m']['s0']['rmse'].mean():.3f}"
                      if "spread" in r["m"]["s0"] else "—")
            else:
                c0 = f"{r['m']['s0']['mae'].mean():.4f}"
                c1 = f"{r['m']['s1']['mae'].mean():.4f}"
                if mode == "det":
                    sp = f"{r['m']['s0']['var_ratio']:.3f}"
                else:
                    sp = (f"{r['m']['s0']['spread'].mean() / r['m']['s0']['rmse'].mean():.3f}"
                          if "spread" in r["m"]["s0"] else "—")
            A(f"| {b}{r['label']}{b} | {r['params']} | {b}{ms(r['m']['s0']['rmse'])}{b} | "
              f"{ms(r['m']['s1']['rmse'])} | {_ratio(r['m'], 'rmse'):.3f} | {c0} | {c1} | {sp} | {note} |")

    emit("A.1 Deterministic point estimators (P1 protocol)", out["det"], "det")
    if tau0:
        A("\n#### The P1 flows' tau=0 mean components (S0)\n")
        A("| scheme | RMSE | MAE | draw dispersion |\n|---|---|---|---|")
        for t in sorted(tau0, key=lambda x: x["rmse"]):
            A(f"| {t['label']} | {t['rmse']:.4f} ± {t['rmse_std']:.4f} | {t['mae']:.4f} | "
              f"{t['draws']:.4f} |")
        A("")
    emit("A.2 Flow matching (P1 protocol)", out["fm"], "gen")
    emit("A.3 SDA (P1 protocol, gw 20)", out["sda"], "gen")
    A("\n### A.4 Reading under the P1 protocol\n")
    A("- **M is the right tier for every learned family.** S+ -> M is a large gain everywhere; "
      "M -> L gains nothing and is actively unreliable (2 of 5 L-tier runs failed to train).")
    A("- **The CFM parameterization is irrelevant.** VanillaCFM (velocity target) and "
      "PredictStateCFM (endpoint target) are statistically identical at S+ (paired t = 0.7, p = 0.48) "
      "despite a 3x gap in training val_loss — val_loss is not comparable across objectives.")
    A("- **SDA's params conditioning buys nothing here**: SDA2/SDA3 never beat SDA1-M, and with the "
      "biased DA params at S1 (fixed 2026-09-25; S1 previously fed them the true params) both lose "
      "1-2%, since both were trained with DA params equal to the true ones. The guidance weight is "
      "worth far more (8% from tuning alone); an SDA3 trained on noisy DA params is robust at S1 "
      "at the tuned gw 25 (`l96_benchmark_extended.md`) -- at this report's gw 20 its SDA3-fix-M row "
      "still loses 1.6%, and on the random observing system it degrades about as much as the "
      "unconditional SDA1 (1.5% vs 2.0% at gw 25). At 1200 epochs (section 4) every SDA prior gains "
      "7-9% at S0 and the pattern holds: SDA2 still loses at S1, SDA3-fix does not.")
    A("- **Every flow's tau=0 mean beats DirectUNet as a point estimator** under this protocol only (with the "
      "benchmark recipe the tau=0 means trail DirectUNet, section 2), so here the advantage is "
      "not only about sampling.\n")
    A("- **Caveats of this annex.** The P1-protocol gap between the flows and DirectUNet (A.1 vs A.2) is a "
      "training-protocol artefact (section 2 vs 3 under the benchmark framework); read the annex for the "
      "within-family comparisons only.")
    A("- The two PredictStateCFM-L rows use a non-standard lr, forced by an optimization failure "
      "at 1e-3 (val_loss jumped 6x at epoch 10 and never recovered).")
    A("- L-tier numbers are single runs with large measured seed sensitivity (DirectUNet-L: "
      "0.4705 vs 0.8579 on two seeds of one config). Treat any single L cell as indicative.")
    A("- Checkpoint-selection noise on this family was measured at 15-21%; smaller differences "
      "need the paired within-window tests, not these point estimates.")

    if missing:
        A("\n## Missing artifacts\n")
        for m in missing:
            A(f"- {m}")

    text = "\n".join(L) + "\n"
    outp = ROOT / args.output
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(text)
    payload = {fam: [{"label": r["label"], "params": r["params"], "flag": r["flag"],
                      "metrics": ({c: {k: float(v.mean()) for k, v in r["m"][c].items()
                                       if hasattr(v, "mean")} | {
                                       k: float(v) for k, v in r["m"][c].items()
                                       if not hasattr(v, "mean")}
                                   for c in CASES} if r["m"] else None)}
                     for r in rows] for fam, rows in out.items()}
    payload = {"annex_p1_protocol": payload, "annex_tau0": tau0,
               "benchmark": {sec: [{"label": r["label"], "params": r["params"], "flag": r["flag"], "seeds": r["seeds"],
                                    "rmse": {t: ({c: p1s.mean(r, t, c) for c in CASES} if r[t] is not None else None)
                                             for t in ("reg", "can")}}
                                   for r in rows] for sec, rows in bench.items()}}
    (ROOT / args.json_output).write_text(json.dumps(payload, indent=2))
    print(text)
    print(f"wrote {outp} and {ROOT / args.json_output}")


if __name__ == "__main__":
    main()
