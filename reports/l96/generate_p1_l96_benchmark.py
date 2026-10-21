#!/usr/bin/env python3
"""P1 L96 benchmark: DA, deterministic, flow-matching and SDA schemes, 2D backbone.

Sections 1-4 and the cross-family reading use the benchmark-default evaluation
framework (``p1_benchmark_sections.py``): models trained on a random observing
system with fresh noise every batch, 1200 epochs, scored on the regular and the
canonical random test sets with the extended report's own code. Every learned row
is the 2D (time x sector) circular U-Net (2026-10-21); rows whose 2D checkpoint is
not trained yet render as pending, single-seed rows carry a note. Then the
per-window (``per_window_summary.py``) and pooled (``p1_skill.py``, the paper's
``tab:s0`` / ``tab:s1``) summaries, the reconstruction examples, and annex A, the
2D S / M / L tiers. The 1D rows remain in ``l96_benchmark_extended.md``.

Protocol facts that make or break comparability, all enforced by the extended
report's checks:

* **Generative rows** are scored on `ens30_no20` -- 30 members, 20 early-fine
  Euler steps -- as ensemble-mean RMSE plus a proper ensemble CRPS.
  **Deterministic rows** are a single pass. A one-draw score of a generative
  model is a strictly worse estimator and is never mixed into the same column.
* **SDA rows** come from `eval_sda_l96.py` with guidance weight 25: SDA1 is an
  UNCONDITIONAL prior whose observations enter only as an inference-time
  guidance cost, so the unguided sampler returns climatological spread.
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
import p1_skill  # noqa: E402
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


# Best learned scheme per subcategory, plus two DA references. Keys are the
# DISPLAY names used as figure row labels: `plot_hovmoller` passes each through
# the consolidated report's `short_name`, which is a dict lookup with a
# "split on the first underscore" fallback -- so underscore-free display names
# pass through unchanged, while raw experiment dir names would render as "A1".
EXT_HERE = _inputs.root("benchmark_extended", "here")
FIGURE_SCHEMES = [
    ("ETKS", ("da", "ETKS")),
    ("Strong-4DVar", ("da", "Strong-4DVar")),
    ("DirectUNet-2D-M", ("est", EXT_HERE / "L96B_directunet2d_monaiM_seed1" / "ens1_no1")),
    ("VanillaCFM-2D-M", ("est", EXT_HERE / "L96B_vanillacfm_monai2dM_ep1200_seed1" / "ens30_no20")),
    ("SDA1-2D-M", ("est", EXT_HERE / "eval_regular" / "L96B_sda1_monai2dM_ep1200_seed1" / "ens30_gw25")),
    ("Hybrid-2D", ("est", EXT_HERE / "eval_regular" / "hybrid_DU2dMs1_L96B_sda1_monai2dM_ep1200_seed1" / "tau0.1_gw1")),
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="reports/l96/outputs/p1_l96_benchmark.md")
    p.add_argument("--json-output", default="reports/l96/outputs/p1_l96_benchmark.json")
    args = p.parse_args()

    import generate_l96_benchmark_extended_report as ext
    cache = json.loads(ext.CACHE.read_text()) if ext.CACHE.exists() else {}
    bench, errors = p1s.collect(cache)
    if errors:
        for e in errors:
            print("FAIL", e)
        sys.exit("consistency check failed; report not written")
    missing = []
    truth = {c: np.load(EXT_HERE / "L96B_directunet2d_monaiM_seed1" / "ens1_no1" / f"estimates_{c}.npz")["truth"].astype(np.float64)
             for c in CASES}

    L = []
    A = L.append
    A("# P1 L96 benchmark — DA baselines, deterministic, flow-matching and SDA (2D backbone)\n")
    A("Two-scale L96, `obs_j=2` (24D observed space: 8 slow + 16 fast), the 200 shared P1 test windows. "
      "**S0** = true parameters; **S1** = ±20% parameter perturbation with ±10% bias (the DA forward model "
      "uses the biased `*_da` values), so `S1/S0` is a robustness-to-model-error ratio: 1.0 means untouched. "
      "Every RMSE cell is the **per-window RMSE on the 24 observed channels, mean ± sd across the 200 windows**, "
      "in **physical units** (seeds pooled per window).\n")
    A("**2D backbone (2026-10-21).** Every learned row is the 2D (time x sector) circular U-Net of "
      "`models/monai_unet_l96_2d.py`: the observed state on a (T, 8) grid, sector k carrying X_k and its observed "
      "Y_k,j as channels, circular padding over the sectors (never downsampled), zero padding and 2x resampling in "
      "time, mid-level attention along time, the widths of the 1D tiers. The 1D U-Net rows this report used to show "
      "remain in `l96_benchmark_extended.md`. **Seeds:** the benchmark plans 3 seeds per row; `seeds` = finished / "
      "planned, rows with fewer than 3 carry a note, and rows whose 2D checkpoint is not trained yet (PredictStateCFM, "
      "SDA2, SDA3-fix and the DirectUNet -> SDA3-fix hybrid) render as pending.\n")
    A("**Evaluation framework (2026-10-02).** Sections 1-4, the cross-family reading, the per-window and pooled "
      "summaries and the reconstruction examples use the **benchmark-default** framework of `l96_benchmark_extended.md` "
      "and are scored by that report's code: learned models trained on a random observing system with fresh noise "
      "every batch (`config/l96_benchmark_default.yaml`), 1200 epochs, M tier; DA at the per-method inflation of "
      "#295 plus the ETKS of #299; every scheme on **both** the regular 30-obs test set and the canonical random "
      "observing system. Flagged rows (italic, with a note) are sensitivity or interim rows and stay out of the "
      "bests. **Annex A** compares the 2D S / M / L tiers (seed 1).\n")
    A("## Protocol\n")
    A("| family | protocol | columns |\n|---|---|---|")
    A("| DA baselines | per-window assimilation (dws=500, DA fast weights), 30 members; ETKS = the ETKF plus a "
      "full-window smoother | RMSE; analysis-ensemble CRPS and spread for ETKF/EnKF/ETKS; none for Strong-4DVar |")
    A("| Deterministic | DirectUNet-2D, single forward pass | RMSE; `var ratio` = predicted / true temporal variance "
      "(1.0 = no smoothing) |")
    A("| Flow matching | `ens30_no20` (30 members, 20 early-fine Euler steps) | ensemble-mean RMSE; proper "
      "ensemble CRPS; `spread/RMSE` = window-mean spread / window-mean RMSE (biased low against the pooled "
      "spread/skill, `docs/results/l96_p1_fm_score.md`) |")
    A("| SDA | guided `ens30`, 10 steps, `gw=25`, `r_var=0.5`; the params-conditioned priors (SDA2, SDA3-fix) get "
      "the biased DA params at S1 | as above |")
    A("| Hybrid | SDA guided sampling started at tau0 0.1 from the DirectUNet-2D-M estimate, gw 1 (validation-selected "
      "for this mean; the 1D hybrid used gw 2) | as above |")
    A(f"| Test sets | regular: the P1 cache, 30 regular obs times, all 24 channels; random: "
      f"`{ext.CAN_TEST.name}`, 10-100 obs at stratified times, 4-16 observed fast channels per window, the exact "
      "obs the DA baselines assimilated | every learned result is checked against its test set window for "
      "window before rendering |\n")
    A("A single draw from a generative model is a strictly worse estimator than its ensemble "
      "mean, so the two never share a column.\n")
    for sec in ("da", "det"):
        for line in p1s.table(sec, bench[sec]):
            A(line)
    for line in p1s.tau0_table(None, bench["fm"]):
        A(line)
    for sec in ("fm", "sda"):
        for line in p1s.table(sec, bench[sec]):
            A(line)
    A("\nRows and their sources: `l96_benchmark_extended.md` (section 1, groups `2D ...`, and its input bundle).\n")
    for line in p1s.reading(bench):
        A(line)
    for line in per_window_summary.section(backbone="2d"):
        A(line)
    for line in p1_skill.section():
        A(line)
    A("## Distributional score FMS_τ\n")
    A("_pending for the 2D rows: the FMS_τ probes (`reports/l96/probe_fm_score*.py`) need the 2D ensembles' "
      "per-element mean and variance, which the benchmark runs did not store; they will be re-run with the 2D "
      "training of step 2. The 1D section is in this report's git history and `docs/results/l96_p1_fm_score.md`._\n")
    A("## Caveats\n")
    A("- DA baselines receive the same per-window parameters as truth generation (S0) or their "
      "biased `*_da` counterparts (S1); the learned schemes see observations only. This is what "
      "makes the comparison apples-to-apples, and also why their S1 behaviour differs so much.")
    A("- DA CRPS is the per-window analysis-ensemble CRPS stored by the runs (30 members), comparable "
      "to the generative families' ensemble CRPS.")
    A("- SDA is the only learned family with a tuned inference hyper-parameter (`gw`, 25 as for 1D, not re-tuned for "
      "2D); the hybrid tunes its start time and guidance on the validation windows (tau0 0.1, gw 1 for the "
      "DirectUNet-2D mean).")
    A("- Single-seed 2D rows: VanillaCFM-2D-M, SDA1-2D-M, the DirectUNet-2D-M -> SDA1-2D-M hybrid and the annex "
      "tiers. Their seed spread is unknown; the 1D seed sd was 0.002-0.005 at regular S0.\n")
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

    for line in p1s.tier_table(bench["tiers"]):
        A(line)
    A("")

    if missing:
        A("\n## Missing artifacts\n")
        for m in missing:
            A(f"- {m}")

    text = "\n".join(L) + "\n"
    outp = ROOT / args.output
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(text)
    payload = {"backbone": "2d",
               "benchmark": {sec: [{"label": r["label"], "params": r["params"], "flag": r["flag"], "seeds": r["seeds"],
                                    "rmse": {t: ({c: p1s.mean(r, t, c) for c in CASES} if r[t] is not None else None)
                                             for t in ("reg", "can")}}
                                   for r in rows] for sec, rows in bench.items() if sec != "tiers"},
               "tiers": [{"family": r["family"], "tier": r["tier"], "params": r["params"],
                          "rmse": {f"{t}_{c}": v for (t, c), v in r["rmse"].items()}} for r in bench["tiers"]]}
    (ROOT / args.json_output).write_text(json.dumps(payload, indent=2))
    print(text)
    print(f"wrote {outp} and {ROOT / args.json_output}")


if __name__ == "__main__":
    main()
