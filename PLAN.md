# 4DVarNet-FM: Implementation Plan

## Overview

Three model families + CS3/CS4 randomized-parameter tests + Experiment G ablation.

- **DirectUNet**: Single UNet pass `obs → state` via MSE (no flow matching) — implemented
- **VanillaCFM**: Standard conditional flow matching, no Tweedie decomposition — implemented
- **TweedieSolver**: Original two-stage solver — legacy, maintained
- **RandomParamDataset**: Per-window randomized `σ,ρ,β` ±20% for robust training — implemented
- **CS3/CS4**: Evaluation on unseen random parameter draws — implemented
- **Exp G (τ=0 CFM)**: Ablate multi-τ training to isolate CFM's source of advantage — implemented (L63; τ=0 runs superseded by the S-series, also L63)

**System naming convention:** the E/F/G/S experiment series are all **Lorenz-63**
(`train_mix: cs1+cs2`, `state_dim=3` checkpoints). The **L-series** is the two-scale
**Lorenz-96** benchmark (`system: lorenz96`, `state_dim=24` observed subspace).

## QG (two-layer quasi-geostrophic) — integrated to master 2026-09-02

DA-baseline case study on a two-layer Phillips channel (pyqg-compatible), now with
the **full executable codebase integrated to master** (previously only the JSON-only
report + generator were on master).

- **Dynamics**: `models/qg_dynamics.py` (`QGDynamics`, 2-layer) + `models/qg1l_dynamics.py`
  (`QG1LDynamics`, reduced-gravity 1-layer structural-error model) + `models/qg_interp.py`
  (spectral resize). Native torch port of pyqg v0.4.0 (RK4, flux-form advection, pyqg
  exponential filter, masked PV inversion), moving-storm wind-stress curl forcing.
- **Data**: `data/qg.py` (`QGConfig`, `QGDataset`, `make_qg_s0_s1_datasets`) — S0/S1
  along-track + random-column obs, corrupted wind/param bias, qg1l structural-error scenario.
  **Random-column geometry (2026-09-05):** `cols_per_day` distinct meridional columns are now
  each observed exactly once per day at its **own** randomly-sampled intra-day step (no two
  columns of a day share a step) — `obs (T,ny)`, `obs_columns (T,)`, `OBS_GEOMETRY_VERSION=2`
  folded into the truth-cache key so a geometry change auto-invalidates old obs caches.
  DA re-run vs the old single-simultaneous-event-per-day constellation: S0 cols=8/lag 1.0
  clearly improves (RMSE −7.7%, EV +0.777→+0.812), cols=4 essentially unchanged; S1 da_nx
  32/64 slightly degrades (RMSE +6–7%, EV −9 to −12 pts) since dispersing the simultaneous
  multi-column updates reduces each update's spatial info under model error.
- **DA baselines**: `evaluation/run_qg_baselines.py` (+ `eval_qg1l_rscale_probe.py`,
  `sweep_qg_baselines.py`) drive EnKF/ETKF on ψ/q. `evaluation/baselines.py` carries the
  shared `ObsOperator` H-mode, `_build_qg_loc_matrices`/`_build_qg_col_loc_matrices`,
  and per-time `loc_Lx_t`/`loc_Ly_t` localization + `init_ensemble` in ETKF/EnKF, merged
  with the L96/joint/ES work.
- **Psi-state variant**: `models/qg_psi_dynamics.py` (`QGPsiDynamics`/`QG1LPsiDynamics`,
  `wrap_psi`) integrates with the **streamfunction as the state variable**, so the psi
  observation operator reduces to a trivial index lookup (`obs_var="psi_state"`). The
  q-space physics is bit-identical to the q-state model (windows around the same
  `_rk4_step`/filter/clip), so free forecasts match to ~1e-6 relative and ETKF skill is
  comparable to the legacy H-function psi-obs. Works for same-resolution S0 + full-res
  qg1l. **Cross-resolution S1 (da_nx 32/16) is now supported** (2026-09-03) via an H-mode
  psi obs operator (`_psi_h` spectrally upsamples the DA-model psi-state to the obs grid
  before column selection); same-res behavior is bit-identical (re-verified EV −2.925).
  **Caveat:** psi_state DA gives a skilful **streamfunction** analysis (psi full EV +0.59
  at da_nx=32) but a degenerate **PV (q) field** (q full EV −3.2 vs the q-state da_nx=32
  ref +0.34): `forward_pv` (q ≈ ∇²ψ) amplifies high-wavenumber psi-analysis error by K²,
  so the q-field skill score (which `expvar_full` reports) collapses while the psi field
  itself is well-estimated. This is a physical psi↔q representation limitation, not a code
  bug (free forecasts still match exactly). On S0 at the q-state-matching noise 0.01 the
  psi-state q-field EV is 0.583 (q1 0.76 / q2 0.40) vs the q-state+psi-obs 0.752 (0.82/0.69),
  while psi-state's streamfunction fields are the best per-field (psi1/psi2 ≈ 0.98); at
  default 0.05 noise the S0 psi-state qall is 0.487 vs the q-state 0.660.
  **Decision (2026-09-03): q-state is the default DA config** — recorded in the dedicated
  report (`reports/qg/generate_qg_psi_state_report.py` → `qg_psi_state_report.md`) and in
  the `--obs-var` default (`'q'`) / help text of `run_qg_baselines.py` and
  `sweep_qg_baselines.py`. psi_state/psi remain research alternatives.
- **Reference-case DA benchmark (2026-09):** `QG4DVar` (strong- and weak-constraint; psi-obs H
  with absolute time index, whitened control, Adam/LBFGS) added to `run_qg_baselines.py`, plus
  dedicated EnKF/Strong/Weak-4DVar reference runs. At the reference case (S0, c4, psi-obs,
  nx=64, 1% noise, N=80, loc 6.0, lags 1.0/2.0) the report's §4.1 now lists **all four** DA
  methods: ETKF/EnKF (≈1.16/0.75 lag1, ≈1.61/0.65 lag2), Strong-4DVar (LBFGS w=60: 1.33/0.725
  lag1, 1.43/0.322 lag2), and **Weak-4DVar** (LBFGS w60 q=0.1: **lag1 1.43/0.788 — best lag-1.0
  row**, lag2 1.44/0.370). Weak uses per-step model-error controls
  `q_t = dyn(q_{t-1}) + Lq·u_t`, `Jq = 0.5Σu[1:]²`; LBFGS over 5-day windows + `q_var_scale=0.1`
  is the robust config (Adam diverges, q<0.1 under-fits, q=1.0 drops below free forecast).
  **S1 (2026-09-05, job 52087):** Weak-4DVar at da_nx=64 (nores) beats the ETKF reference on the
  S1 model-error case — §5.6: improv **1.92**/EV 0.398 (lag1) and 1.64/EV 0.023 (lag2) vs ETKF
  1.55/0.428 (lag1) — the per-step model-error controls absorb the S1 param/wind bias.
  **S1 extension, cross-res + QG1L (2026-09-06, `feature/qg-da-integration`):**
  - *S1-cross-res* (`da_nx=32` vs truth `nx=64`, param+wind bias on top of the resolution
    mismatch): ETKF/EnKF 1.47x-1.48x/0.34; Strong-4DVar 1.25x/-0.94; Weak-4DVar 1.39x/**-0.23**.
    Same pattern as S1-no-res, more pronounced: Weak-4DVar clearly beats Strong-4DVar but
    neither yet matches ETKF/EnKF here — cross-resolution adds its own difficulty on top of
    the bias effect. Open follow-up: tune Weak-4DVar's `b_var_scale`/`q_var_scale`
    specifically for the cross-res case (only the S1-no-res values have been tried).
  - *S1-QG1L* (structural error, full-res 1-layer DA model) — **the scenario itself is
    broken for every method at these settings, not a 4DVar-specific bug**: ETKF gets
    EV=-11.2, EnKF gets EV=-13.3, and even the **free forecast** gets EV=-0.496 (already
    worse than climatology before any DA). Weak-4DVar gets finite, roughly ETKF/EnKF-scale
    per-window RMSE on 4/5 windows (NaN on the 5th) — not uniquely broken, same boat as the
    ensemble methods. Points to the pre-existing `qg_s1_qg1l_rscale_probe.py`/
    `generate_qg1l_report.py` (r-scale sweep) as the right lever, not further DA-side
    hyperparameter tuning.
  **Obs-protocol reproducibility validation (2026-09-06, `feature/qg-da-integration`,
  job 52174) — DONE, results hold:** the archived reference-case JSONs above were
  produced with the **pre-#156 constellation-style random-columns obs** (`(T, C·ny)`,
  simultaneous multi-column events); `data/qg.py` (PR #156) switched to per-column
  independent intra-day timing (`(T, ny)`). Re-ran all four methods at the exact
  reference settings (S0, c4, psi-obs, lag=1.0) under current `origin/master`:

  | method | archived (pre-#156) | current master (PR #156 geometry) |
  |---|---|---|
  | ETKF | 1.16 / 0.752 | 1.14 / 0.747 |
  | EnKF | 1.17 / 0.754 | 1.16 / 0.749 |
  | Strong-4DVar | 1.33 / 0.725 | **1.44 / 0.773** |
  | Weak-4DVar | 1.43 / 0.788 | **1.50 / 0.805** |

  Small shifts throughout, no qualitative change — if anything the conclusion
  strengthens: both 4DVar methods improved under the new geometry (Strong-4DVar now
  also beats ETKF/EnKF, not just Weak-4DVar), while ETKF/EnKF softened marginally.
  Archived JSONs are superseded by `reports/qg/outputs/qg_repro_validation/*.json`
  as the current reference numbers; the archived ones are kept for provenance only.
- **Report**: `reports/qg/generate_qg_s0s1_report.py` (JSON-only) renders from the result
  JSONs under `reports/qg/outputs/` → `reports/qg/outputs/qg_s0s1_report.md` (revised:
  governing equations, case-study table, S0 / S1-QG2L da_nx 16/32/64 / S1-QG1L sections,
  psi-obs focus). Dedicated `reports/qg/generate_qg1l_report.py` → `qg1l_report.md` for
  the reduced-gravity structural-error case (r-scale sweep).
- **Illustrations (S0 / S1-QG2L da_nx=32)**: `reports/qg/generate_qg_s0s1_figs.py` (no
  DA-cache dependency) runs a single-window production ETKF (nx=64, N=80, psi-obs,
  cols=4, 1% noise, lag 1.0) and writes per-scenario obs-days 2×2 panel + full-window
  obs Hovmöller, forcing, truth psi/q, analysis-vs-free-vs-truth, and a DA-cycle GIF to
  `reports/qg/outputs/figs/`. `generate_qg_s0s1_report.py` §8 embeds them
  (`![](figs/...)`, still JSON-only). The generator now (2026-09-03 fix) selects the
  **first window with a non-zero wind amplitude** (the S1 wind-levels list starts at
  0.0, so window 0 previously rendered a flat all-zero wind-curl forcing), draws the
  **corrupted** wind-curl (so the S1 figure shows the actual corrupted moving storm),
  fixes the DA-cycle GIF's per-panel `ax` handling (truth q₁ and DA-analysis q₁ panels
  were blank), and rebuilds the obs Hovmöller as a time×column storm-track field with
  cross-time interpolation (previously ~96% blank). Regenerated production figures
  committed alongside the fix. A follow-up DA-cycle obs-panel fix (2026-09-04) renders
  the **raw observations** (not a per-day aggregate): observed meridional columns are
  drawn **vertical** (removed the `img.T` transpose that had rotated them horizontal),
  at a single window-wide **fixed color scale** (previously re-normalized per frame),
  and each frame shows the nearest preceding raw obs event so the panel is never blank.
- **Tests**: 7 QG test files (`test_qg_dynamics`, `test_qg_data`, `test_qg_baselines`,
  `test_qg_s0s1`, `test_qg_random_columns`, `test_qg1l_dynamics`,
  `test_qg_psi_state`) — all in the master CI gate.
- **sbatch**: 32 `batch/run_qg_*.sbatch` for the S0/S1 matrix + S1-resolution + qg1l sweeps
  (+ `run_qg_figs.sbatch`, the illustration/DA-cycle regeneration job).
- QG is DA-baseline-only (no QG neural estimator; not wired into `train.py`/`get_dynamics()`).

### QG neural baseline (Q1/Q2) — infrastructure on `feature/qg-neural-baseline` (2026-09-06)

Self-contained DirectUNet / VanillaCFM(τ=0) neural estimators on the **S0**
case study, comparing against the QG DA baselines. Not wired into `train.py`
(dedicated entry point `train_qg_neural.py` + `data/qg_neural.py` wrapper).

- **Supervision**: daily mean of the full 2-layer **streamfunction** ψ
  (`psi_daily`), 30 days/window, with an auxiliary PV-q loss
  (`--q-loss-weight`, default 0.1) that inverts the ψ estimate to PV per-window
  (`norm_q_from_psi`) and MSEs it against the PV target. Tracked as both a ψ
  primary loss and a q auxiliary term per window.
- **Observations**: upper-layer ψ random-column obs, grid-expanded + daily
  aggregated + NaN-masked, padded to the full state width (zeros in the lower
  layer) so the shared `DirectUNet`/`VanillaCFM` (obs_dim==state_dim) work
  unchanged. `QGBatch.params` added (None) for the shared model forward paths.
- **Normalization (superseded 2026-09-08, see below):** the original design
  used `window_scales` (each window's own per-layer ψ/q std, `WindowScale`) so
  samples whose streamfunction energy spans a wide dynamic range (empirically
  ~30→10⁴ at nx=8 across windows) are each mapped to O(1), avoiding a single
  global scalar being dominated by the highest-energy windows. Q1 now uses
  **global** per-layer z-score normalization for ψ instead (see the
  "Global psi normalization" entry below) — `window_scales`/`WindowScale` are
  retained as a diagnostic only, no longer used to build training targets.
  **Critical (unchanged):** the ψ<->q inverter is cached per device
  (`_INVERTER_CACHE[..., device]`) so the GPU q-loss never moves the shared CPU
  inverter used by `psi_daily`/scales — which would otherwise break CPU-side
  eval after a GPU training step.
- **Model/lightning**: `QGNeuralLightning(pl.LightningModule)` computes
  DIRECTUNET `MSE(est, ψ_norm)` or τ=0 CFM `MSE(x0+v, (ψ_norm−x0))` plus the
  per-window q-loss; `create_trainer` for the Lightning loop.
- **Eval**: `estimate_windows` produces per-window physical ψ; `psi_to_q`
  maps to physical PV; both ψ and PV-q RMSE/EV per layer + pooled are written
  to `estimates_s0.npz` + `results.json` (`s0.psi`, `s0.q`).
- **Configs**: `config/experiment/Q1_direct_unet_s0.yaml`,
  `Q2_vanilla_cfm_s0.yaml` (documentation specs → CLI flags; not Hydra).
- **Report**: `reports/qg/generate_qg_da_report.py` → `qg_da_report.md` renders
  the 4 DA baselines (`qg_repro_validation`) on PV-q + ψ; `reports/qg/
  generate_qg_neural_report.py` → `qg_neural_report.md` is the case-study
  overview (scheme descriptions + a combined DA/Q1 summary table, linking
  back to `qg_da_report.md` for DA detail). Renamed 2026-09-10 (previously
  one script/file covered both).
- **Tests**: `tests/test_qg_neural.py` (12 fast tests: psi-day matches the
  window's own upper-psi target, per-window scale O(1), dataset/collate shapes,
  denorm round-trip, `norm_q_from_psi` round-trip, per-device inverter-cache
  isolation, QGNeuralLightning fwd/bwd for both models, estimate_windows
  shapes) — added to the master CI gate.
- **HPC note**: per-window truth generation (`QGS01Dataset` spinup) is
  expensive (~85 s/window at nx=8 single-thread; nx=64 far more). The
  `num_train/val/test` defaults in the configs are a production target; see
  the "1000/100/100 train/val/test dataset generation" section below for the
  now-merged array-parallel/GPU-side generation infra that makes this
  tractable, and "Truth-only cache + on-the-fly obs" further below for how
  `train_qg_neural.py` consumes it.

### QG neural baseline — global psi normalization (2026-09-08)

Replaced the per-window ψ/q normalization (`WindowScale`) with **classic
global mean/std z-score normalization**, applied **only to ψ** (PV/q stays in
raw physical units), matching the L96 track's `data/normalization.py`
(`compute_channel_stats`/`normalize`/`denormalize`, added in #166) instead of
a QG-specific abstraction.

- **Precompute**: `precompute_qg_norm_stats.py` loads the 1000-window S0
  train truth cache and computes (a) global per-layer ψ (mean, std) via
  `data.normalization.compute_channel_stats`, (b) the per-window ψ std range
  (diagnostic, reuses `window_scales`) to check the dynamic-range concern
  documented above at production resolution, (c) the global pooled Var(q) and
  a derived `q_loss_weight = 1/Var(q)`. Run via
  `batch/run_qg_precompute_norm_stats.sbatch` (CPU-only, `--account=mee
  --qos=mee_short` on `Mee_Global_CPU` — the `odyssey` account's `low` QOS
  isn't permitted there) since loading the ~33GB train truth cache OOMs an
  interactive/unreserved shell. Output: `experiments/qg_psi_norm_stats.pt`
  (mean/std) + `..._extra.pt` (q_var/q_loss_weight/per-window std diagnostics).
- **Measured at nx=64, train-seed=42, 1000 windows (2026-09-07)**:
  `psi1 mean≈3.5e-6 std≈1.85e4`, `psi2 mean≈1.7e-6 std≈1.62e4`,
  `q_var≈3.93e-10` → `q_loss_weight≈2.5415e9`. Per-window ψ std range:
  layer1 min=3.9e3/max=9.19e4 (23.6x), layer2 min=1.0e3/max=8.6e4 (82.8x) —
  real but less extreme than the ~300x measured at nx=8; accepted trade-off
  (low-energy windows are under-weighted in the loss vs. the old per-window
  scheme, by up to ~16x at p10 energy) after review.
- **Why q stays raw + why q_loss_weight is ~1e9**: ψ std (~1e4) and q std
  (~1e-5) differ by ~9 orders of magnitude. Z-scoring only ψ makes
  `loss_psi` ~O(1) (unit target variance); `loss_q` in raw units would then be
  ~O(1e-10) and, at the old default `q_loss_weight=0.1`, contribute nothing
  (its gradient would be ~9 orders of magnitude too small to matter) — so the
  weight itself must absorb the scale gap. `q_loss_weight=1/Var(q)` is the
  balancing choice: it makes q's raw-unit MSE comparable in scale to ψ's
  (unit-variance) normalized MSE, and is derived/auditable rather than a
  hand-picked literal.
- **Config**: `training.q_loss_weight`/`data.normalize`/`data.norm_stats_path`
  in `config/experiment/Q{1,2}_..._s0.yaml` are now the actual source of truth
  (loaded via `OmegaConf.load` in `train_qg_neural.py`'s `main()`, CLI flags
  `--q-loss-weight`/`--normalize`/`--norm-stats-path` override); previously
  these YAMLs were documentation-only and `--q-loss-weight` had its own
  independently-hardcoded argparse default that could silently drift from
  them.
- **API**: `data/qg_neural.py`'s `QGNorm`/`compute_norm` (global, std-only,
  informational-only) are removed; `denorm_state`/`norm_q_from_psi` are
  renamed `denorm_psi`/`q_from_psi_norm` and take the `{"mean","std"}` dict
  from `data.normalization` instead of a per-window `WindowScale`. `QGBatch`
  drops its `scale` field (no longer needed — ψ normalization is a training-
  run constant, not a per-sample value, and q is unnormalized).

### 1000/100/100 train/val/test dataset generation (2026-09-07)

Scaling the S0/S1 truth-window generation from the ~5-200 window exploratory scale to a
**1000 train / 100 val / 100 test** window dataset exposed a critical perf bug and required
array-job parallelization to make the scale tractable.

- **Device-placement fix**: `QGS01Dataset._generate_truth` never moved its per-window
  `QGDynamics` to the requested device, so the ~2-year spin-up (the dominant cost, ~4.5
  min/window) ran on CPU regardless of GPU allocation. Fixed: `dyn`/`wind_full`/`traj_full`
  now roll out on the given `device`, then move back to CPU before the (CPU-only)
  obs/field-extraction post-processing. **7.4x speedup** measured on a dedicated A40:
  36.65s ± 0.16s/window (n=20) vs. the documented ~270s/window CPU baseline.
  `QGS01Dataset.__init__`/`make_qg_s0_s1_datasets` now accept an optional `device` param.
- **Per-window RNG independence**: `_generate_truth`'s param/geometry draws (`u/r/k/x0/y0/
  cx/cy`) used one `RandomState` shared sequentially across all `n` windows, so window `i`
  could only be generated after replaying windows `0..i-1`. Switched to a per-window
  `RandomState(cfg.seed + i)` (matching the already-index-keyed wind/traj/obs seed
  formulas) and added an `indices: list[int] | None` param so any subset of window indices
  can be generated independently, by any worker, in any order — required for array-job
  parallelization. Verified bit-identical to a full serial run over the same indices.
- **Array-parallel generation**: `reports/qg/generate_qg_window_chunk.py` (one chunk of
  window indices -> one `.pt` file per window) + `batch/run_qg_window_chunk_array.sbatch`
  (SLURM array task, `--gres=gpu:a40:1` — this cluster rejects untyped `gpu:1` GRES
  requests) + `reports/qg/assemble_qg_windows.py` (glob + sort by index -> single combined
  file at the exact `_truth_cache_path` the config/n/obs-geometry key expects, so
  `make_qg_s0_s1_datasets(..., cache_dir=...)` hits the cache afterward with no code
  changes downstream). Launched 3 array jobs (40+5+5 = 50 tasks, 25/20/20-window chunks)
  across the cluster's 3 idle A40 GPUs (this partition requires a specific GPU model, no
  generic request): **1200 windows generated in ~3h27min wall-clock** (23:43->03:10),
  150/150 tasks exit 0, no errors. Train-split assembly (1000 windows, ~32GB in memory)
  OOM'd under the interactive session's 16GB job-allocation cgroup cap; reran as its own
  sbatch job with `--mem=96G` (succeeded in 3m48s). Final cache: 32GB (train) + 3.2GB
  (val) + 3.2GB (test) = 38GB total, `reports/qg/outputs/qg_windows_1000_100_100/cache/`.
  End-to-end verified: `make_qg_s0_s1_datasets` cache-hit load (val, 100 windows) in 9.1s,
  correct S0/S1/S1-QG1L scenario shapes/metadata.
- Full QG suite (115 tests incl. 2 new: indices-subset reproducibility, device roundtrip)
  green.

### Test-split obs/IC correction + truth/obs/IC separation (2026-09-07)

The above generation used `QGConfig` defaults for `obs_geometry`/`cols_per_day`/
`obs_noise_std_frac`/`init_lag_days`, not the S0 reference-case settings
(`random_columns`/4/0.01/1.0) -- caught before running DA baselines on the test split.
Since truth generation doesn't depend on those fields, split `_generate_truth` into
`_generate_truth_only` (expensive rollout) + `_generate_obs_ic` (cheap: rebuilds a
`QGDynamics` from cached `true_params`, no rollout, redraws obs/init-state) --
`_generate_truth` now composes both, unchanged for existing callers.
`reports/qg/fix_qg_test_obs_ic.py` applies this to the test split in seconds, writing
`truth_only/test.pt`, `obs_ic/test_reference.pt`, and a corrected combined cache at the
hash `make_qg_s0_s1_datasets` expects. Train/val untouched (obs/IC for those are meant to
be generated on the fly downstream, not read from this cache). Added `--seed`/
`--cache-dir` to `run_qg_baselines.py` (seed was hardcoded to 7; no CLI path existed to
reach the pre-generated cache) so `run()`'s existing `ds=` bypass is reachable from the
command line.

**Bug caught by the test suite** during the `_generate_obs_ic` split: `init_lead_truth`
(cached, indices `[0, lead)`) doesn't include index `lead` itself (`traj[0]`), needed for
a zero-day lag draw -- fixed by concatenating `traj[:1]` before indexing. 115 tests green
after the fix.

**Separately found:** `run_qg_baselines.py` for a 100-window scenario OOM'd in this
interactive session's 16GB cgroup -- `run()` accumulates full per-window `(360, 8192)`
arrays across all windows before computing summary metrics (fine at ~5-window
exploratory scale, needs real memory at 100). Moved to `batch/
run_qg_test100_reference.sbatch` (`--mem=64G`, `--time=12:00:00`, ETKF/EnKF/
Strong-4DVar/Weak-4DVar at the exact S0 reference settings); results pending.

### Neural dataloader: on-the-fly obs for train/val (2026-09-07, `feature/qg-neural-baseline`)

Consumes the `_generate_truth_only`/`_generate_obs_ic` split above (unchanged
here) from the neural training side: `data.qg_neural.QGNeuralDataset(on_the_fly_obs=True)`
redraws obs/init-state fresh (random seed) from a truth-only window on every
`__getitem__` call, so train/val see a different obs realization each epoch
from the same cached truth instead of one fixed draw baked in — increasing
training obs diversity without re-paying the rollout. `data.qg.ensure_truth_only_cache`
(a lightweight single-process hash-keyed cache, distinct from the array-parallel
`generate_qg_window_chunk.py` pipeline above — useful for smaller-scale/smoke
runs; the production 1000-window cache from that pipeline
(`reports/qg/outputs/qg_windows_1000_100_100/cache/`) can also be pointed to
directly via `--cache-dir` once `train_qg_neural.py` is scaled up) wraps
`QGS01Dataset._generate_truth_only`, keyed only by the rollout-relevant
`QGConfig` fields so obs-geometry/noise tuning reuses the same cache.
`train_qg_neural.py` builds train/val from this + `on_the_fly_obs=True` by
default (`--fixed-split-obs` reverts to the old fixed-cache behavior); the
**test split is unchanged** (`ensure_truth_cache`, fixed reproducible obs).
`--num-test` default 200→100 (train/val defaults unchanged at 1000/100),
matching the 1000/100/100 split above. Note: `forcing` in `QGBatch` remains a
zero placeholder (Q1/Q2 are `cond_extra_dim=0`, obs-only) — on-the-fly
*forcing* conditioning was not wired, only obs; see CHANGELOG 2026-09-07 for
the full rationale/verification.

### Q1 launch: reusing the production 1000/100/100 nx=64 cache (2026-09-07, `feature/qg-neural-q1-launch`)

Wired `train_qg_neural.py` to reuse the already-generated production truth
cache (`reports/qg/outputs/qg_windows_1000_100_100/cache/`, 41GB, built by the
`generate_qg_window_chunk.py` array-job pipeline above) instead of paying the
~2-year-spinup rollout again from this branch:
- New `--train-seed`/`--val-seed`/`--test-seed` (default 42/10042/20042,
  matching `SPLIT_SEED_BASE`) and `--obs-geometry`/`--cols-per-day`/
  `--obs-noise-std-frac`/`--init-lag-days` (default `random_columns`/4/0.01/1.0,
  the S0 DA-baseline reference case) CLI flags, applied to `test_cfg` (which
  also seeds train/val's on-the-fly obs draws via the shared cfg object).
- Train/val now load via `ensure_truth_cache` (not `ensure_truth_only_cache`):
  the production cache's train/val entries already carry (QGConfig-default,
  i.e. wrong-for-S0) baked-in obs, but `on_the_fly_obs=True` overwrites
  `obs`/`obs_mask`/`init_state` on every `__getitem__` regardless, so reusing
  the full cache is correct and avoids a second, redundant cache format.
  `ensure_truth_only_cache` remains available/tested (`data/qg.py`) for
  contexts with no pre-existing full cache to reuse.
- **Bug found and fixed while validating this**: `build_cfg(...)` never set
  `num_windows`, leaving it at the `QGConfig` dataclass default (200) instead
  of the split size. Since `_truth_cache_path` hashes the *entire* `asdict(cfg)`
  (`num_windows` included) plus a separate `n_windows` key, this silently
  missed the production cache (whose entries were written with
  `num_windows` equal to the split size, matching
  `generate_qg_window_chunk.py`) and would have fallen back to a full
  from-scratch nx=64 rollout (~90h extrapolated) instead of a cache hit.
  Fixed by passing `num_windows=args.num_{train,val,test}` explicitly to each
  `build_cfg(...)` call. Caught interactively (stopped the hung process after
  ~5 min, well before real damage) by benchmarking `ensure_truth_cache`
  in isolation before trusting it inside a long GPU job — verified the fix by
  recomputing `_truth_cache_path` and diffing against the exact on-disk
  filenames (`qg_truth_a3c24de…`/`e7f84d…`/`ab5c09…` for train/val/test),
  then a live load (100-window test cache: 3.2s, was hanging past 90s before
  the fix).
- **On-the-fly obs cost at nx=64, benchmarked**: `QGS01Dataset._generate_obs_ic`
  (single window, no rollout) ≈32ms/draw → ≈1.8h aggregate over
  200 epochs × 1000 train windows, well hidden by GPU prefetch at
  `num_workers=1` (not a bottleneck).
- `batch/run_qg_q1_train.sbatch` (new): `Odyssey_GPU` partition, 1×A40,
  `--mem=96G` (must comfortably hold the 33GB train cache in RAM),
  `--time=24:00:00`, points `--cache-dir` at the production cache's absolute
  path (on the sibling `4dvarnet-fm-qg-100samples` worktree's filesystem
  location — the cache itself is untracked/gitignored, not portable via git).
- **Incidental environment fixes** (shared `fdv` conda env, unrelated to this
  branch's code): mid-session, a concurrent process modified the shared env
  twice — `setuptools` drifted to 84.0.0 (dropped the `pkg_resources` shim
  `pytorch_lightning` 2.3.3 needs; fixed with `pip install "setuptools<81"`),
  and separately `torch`'s `libtorch_global_deps.so` went briefly missing
  mid-reinstall (self-resolved after waiting; confirmed `torch 2.4.1+cu121`
  healthy after). Neither is caused by or specific to this branch's changes.

**Q1 status:** launched via `sbatch batch/run_qg_q1_train.sbatch` (job id
recorded in CHANGELOG 2026-09-07); monitor for completion before starting Q2.

### DA reference-case realism: lag/noise sensitivity (2026-09-08)

The S0 DA-baseline reference case (`lag=1.0`, `noise_frac=0.01`, i.e.
`qg_repro_validation`'s EnKF/ETKF/4DVar numbers: psi full EV 0.97-0.99,
q full EV 0.75-0.80) was flagged as unrealistically favorable: at `lag=1.0`
the DA background/init state is the **true state from only ~1 day earlier**,
rolled forward with the (near-)true model (`evaluation/run_qg_baselines.py`'s
`_sample_init_state`) — not an independent forecast. The **free forecast
alone** (no assimilation) already has psi full EV=0.979 at that lag, higher
than 2 of the 4 DA baselines post-assimilation; DA's high psi score there is
almost entirely inherited from the background, not the observational update.

Ran an ETKF-only sensitivity sweep (`--geometry random_columns --cols-per-day
4 --obs-var psi`, N=80 ensemble, inflation=1.0, loc=6.0 — matching
`qg_repro_validation`'s ETKF settings) varying `init_lag_days` (1/3/5/7/10)
and `obs_noise_std_frac` (0.01/0.05/0.10/0.20) independently, first on a
10-window subset of the production S0 test cache (fast iteration; real but
noisy at that sample size -- e.g. its psi full EV at lag=5/noise=0.05 was
0.843 vs. 0.922 on the full set, see below), then a combined
lag=5/noise=0.05 joint sweep (1-5 days), then consolidated on the **full
100-window test set**:

- **Lag alone** (noise fixed at 0.01): psi full EV falls monotonically
  0.962(lag1) -> 0.910(3) -> 0.858(5) -> 0.783(7) -> 0.715(10); free-forecast
  q EV crosses zero around lag=5 (the point past which the background is no
  longer informative on its own).
- **Noise alone** (lag fixed at 1.0): psi full EV falls 0.962(0.01) ->
  0.931(0.05) -> 0.866(0.10) -> 0.755(0.20).
- **Joint lag x noise=0.05** (10-window subset): at `lag<=2d` DA is
  neutral-to-**negative** on psi relative to the free forecast (obs too
  noisy to help a still-fresh background); the sign flips positive around
  `lag=3d` and grows through `lag=5d`, while q's DA contribution is large
  and positive from `lag>=2d` on (free-forecast q EV is already negative by
  `lag=5d`). Also confirmed **`init_lag_days` is physical days**, not model
  steps (`_sample_init_state`'s `lag_steps = lag_days * steps_per_day`,
  `steps_per_day = round(86400/cfg.dt)`).
- **Consolidated reference case — `lag=5.0d, noise_frac=0.05`, full 100-window
  S0 test set** (chosen as the new candidate realistic reference; upper=layer1
  is the only observed layer):

  | field.layer | EV (DA) | EV (free) | delta |
  |---|---|---|---|
  | psi layer1 | 0.909 | 0.836 | +0.073 |
  | psi layer2 | 0.935 | 0.935 | +0.000 |
  | **psi full** | **0.922** | 0.885 | **+0.036** |
  | q layer1 | 0.470 | -0.044 | +0.514 |
  | q layer2 | 0.339 | -0.018 | +0.357 |
  | **q full** | **0.405** | -0.031 | **+0.436** |

  (psi std over these 100 windows: mean=14494, range=[4058, 59030] -- the
  10-window dev subset's mean of 10306 understated this, hence the psi-EV gap
  above.) psi full EV=0.92 is just above the 80-90% target band; q full is a
  clear, well-earned DA contribution (free-forecast q EV is negative here).
  If a tighter 85-90% psi band is wanted, nudge lag to ~6-7d or noise to
  ~0.07-0.08 from this point.

**All 4 methods confirmed at lag=5.0d/noise=0.05, N=100** (2026-09-08,
follow-up to the ETKF-only numbers above): the psi-vs-q story is genuinely
different per method, not just a matter of degree —

| method | psi full EV | q full EV |
|---|---|---|
| ETKF | 0.922 | 0.405 |
| EnKF | 0.948 | 0.484 |
| Strong-4DVar | **0.971** (best) | **-0.126** (worse than free forecast) |
| Weak-4DVar | 0.966 | -0.035 |

Both 4DVar variants have the best psi analysis of all 4 methods but
**collapse on q layer2** (the unobserved lower layer: ETKF/EnKF stay
positive there, +0.34/+0.40, while Strong/Weak-4DVar go to -0.73/-0.54) —
q layer1 (the observed layer) stays positive and consistent across all 4
methods (0.47-0.57), so this isn't a general 4DVar weakness, it's specific to
inferring the unobserved layer. Consistent with PV being a Laplacian-like
operator on psi (`q ≈ ∇²ψ`): small high-wavenumber psi errors get amplified
when inverted to PV. Weak's per-step model-error controls soften this
(Δq_full -0.003 vs Strong's -0.095) without reversing it. **EnKF is the only
method strongly positive on both fields** — worth keeping in mind if a
single "headline" DA number is ever needed.

**This lag=5.0d/noise=0.05 case is now the actual reference case**: full
N=100 reruns for all 4 methods (this time also computing per-window CRPS —
see `evaluation/metrics.py`'s `crps()` fix and `run_qg_baselines.py`'s
per-window CRPS wiring, same PR) are committed to
`reports/qg/outputs/qg_repro_validation/*.json` for the first time — that
directory never actually existed on master before (see `reports/qg/
generate_qg_da_report.py`, DA-baselines-only — renamed 2026-09-10 from
`generate_qg_neural_report.py`, see above — and `reports/qg/
generate_qg_reconstruction_figs.py` for 3 example-window reconstruction
figures, best/median/worst by ETKF's per-window q RMSE). Scratch driver
(not committed): `qg_da_sensitivity_scratch.py` in the repo root.

### Revised S1 (model-error) reference case (2026-09-10)

Redefined S1 to share S0's initial-uncertainty setup exactly (lag=5.0d,
noise_frac=0.05) and add three independent model-error sources on top:
**param bias** (`rd`/`rek` scaled by `1 - s1_param_bias`), **corrupted wind**
(amplitude bias + always-on storm-location OU jitter, std=62.5km,
correlation=10d, plus amplitude OU jitter — see `_make_corrupted_wind_state`),
and **structural resolution mismatch** (`da_nx=32`, half the truth grid,
`qg2l_lores`). Obs are always drawn from the true (unbiased, full-res)
trajectory regardless of scenario — only the DA method's own model sees the
corruption.

**Bias level chosen (0.1/0.1, not the 0.15/0.15 project default)**: an
isolated per-factor sensitivity sweep (N=10, ETKF, `da_nx=64` — no structural
mismatch, so each factor is tested alone) found the two error sources behave
very differently. Wind-forcing bias is nearly harmless to the DA analysis
even at 0.30 (ψ EV 0.82→0.78, q EV 0.40→0.39) — frequent obs updates correct
it before it accumulates, even though the *free forecast* degrades a lot
(ψ EV(free) 0.75→0.36). Param bias is the dangerous one: mild (0.05) costs
little, but by the project default 0.15 the analysis is already badly
degraded (q EV 0.40→0.11), and at 0.30 ETKF **diverges outright**
(EV=-9.2, worse than climatology) despite the free forecast barely
changing — a filter-divergence signature, not gradual degradation. Chose
0.1/0.1 for the combined (param+wind+da_nx=32) scenario as meaningfully hard
without being degenerate — confirmed via an N=10 check with all three
factors combined: ψ EV 0.82→0.70, q EV 0.40→0.25, no divergence. Sensitivity
sweep data: `reports/qg/outputs/qg_repro_validation_s1_sensitivity/*.json`.

**ETKF inflation sweep (N=10, revised S1 combined config) — open question**:
tested whether under-inflation explains EnKF's consistent edge over ETKF
(see results below). Found the opposite of the hypothesis: ANY inflation
above 1.0 (1.05/1.1/1.2/1.3) causes immediate catastrophic ETKF divergence
(q EV -39.5/-250.8/-345.7/-375.9 respectively, vs 0.253 at inflation=1.0) —
not a gradual improve-then-degrade curve. `inflation=1.0` (off) appears
necessary for stability in this nonlinear, already-biased/coarse setting,
not an undertuned default. EnKF's edge over ETKF here remains unexplained;
data at `reports/qg/outputs/qg_repro_validation_s1/etkf_n10_infl*.json`.

**Full N=100 results (revised S1 vs S0, all via `qg_da_s1_scratch.py`,
scratch driver, not committed)**:

| method | S0 ψ EV | S0 q EV | S1 ψ EV | S1 q EV |
|---|---|---|---|---|
| ETKF | 0.921 | 0.405 | 0.874 | 0.307 |
| EnKF | 0.947 | 0.481 | 0.896 | 0.331 |
| Strong-4DVar | 0.971 | -0.126 | 0.931 | -0.857 |
| Weak-4DVar | 0.966 | -0.035 | 0.947 | -0.501 |

EnKF is the clear S1 winner overall — the only method staying positive and
reasonably stable on q, while both 4DVar methods go from mildly negative
(S0) to badly negative (S1) once real model error is added. ψ stays
reasonably robust for everyone; the PV/q story is where S1 separates the
methods, consistent with 4DVar's strong reliance on a (nearly) correct
dynamical model, which S1 deliberately violates.

Committed data: `reports/qg/outputs/qg_repro_validation_s1/{etkf,enkf,
strong4dvar,weak4dvar}.json` (N=100, all 4 methods) + `etkf_n10*.json` (the
N=10 sanity checks/inflation sweep above). Strong-4DVar/Weak-4DVar's first
N=100 runs (SLURM jobs 52911/52912) completed successfully but their output
JSONs were lost to a `git stash -u` accident before being committed — see
`feedback_stash_u_deletes_untracked_scratch_files` memory; re-run (jobs
52994/52995, after also fixing a stale-`.pyc`-bytecode-cache issue that
broke two earlier re-run attempts) reproduced bit-for-bit identical numbers.
Scratch drivers (not committed): `qg_da_s1_scratch.py`,
`qg_da_s1_factor_sensitivity_scratch.py` in the repo root.

Not yet done: folding these S1 numbers into `qg_da_report.md`/
`qg_neural_report.md` (currently S0-only) — separate follow-up.

### QG neural schemes beyond Q1: Q3 (oracle) / Q4 (noisy) forcing+param
### conditioning (2026-09-10, `feature/qg-q3q4-forcing-param-cond`)

New DirectUNet variants that additionally condition on the wind-forcing
field and physical params (`U1, rd, rek`), alongside observations --
answers whether exogenous conditioning helps a QG estimator, mirroring the
L96 SDA CFM forcing/param-conditioning study (`models/sda.py`'s
`ConditionalPriorCFM`, `SDA2_cond_nominal_l96.yaml`/`SDA2_cond_mixed_l96.yaml`/
`SDA3_cond_noisy_l96.yaml`) rather than L96's earlier, weaker `L6_vanilla_
cfm_s0s1_forcing_cond.yaml` (obs+corrupted-forcing on a 1D UNet, found
neutral-to-slightly-negative — see the L96 Q3 note above).

- **Bug fixed as a prerequisite**: `train_qg_neural.py`'s `build_model()`
  hardcoded `param_dim=0, cond_extra_dim=0` directly in the
  `MonaiDirectUNetQG`/`VanillaCFM` constructor calls, silently ignoring
  `config/experiment/Q{1,2}_..._s0.yaml`'s `model.param_dim`/
  `model.cond_extra_dim` keys entirely (dead YAML fields since those
  configs were introduced). No behavior change for Q1/Q2 (their YAMLs
  already say 0/0) but blocked any new conditioned scheme until fixed.
  `build_model()` now takes explicit `param_dim`/`cond_extra_dim` args,
  read from the experiment YAML in `main()`.
- **Forcing representation**: the real spatial `wind_curl` field
  (`(T, ny, nx)`, from the true or corrupted trajectory), daily-mean binned
  to `(days, ny, nx)`, NOT a scalar broadcast to a spatially-uniform
  channel -- the storm's location is exactly the physically meaningful
  part of the forcing, and `MonaiDirectUNetQG` already treats a channel as
  a spatial field, so broadcasting a scalar would have thrown that away.
  `QGBatch.forcing`'s shape changed project-wide from `(B, days)` to
  `(B, days, ny, nx)` (Q1/Q2 unaffected: their `cond_extra_dim=0` means the
  model never reads it regardless of shape).
- **Params**: `[U1, rd, rek]` (`U2` and `beta` excluded -- `U2` is always 0
  and `beta` is never jittered by `QGS01Dataset._generate_truth_only`
  (only `U1`/`rd`/`rek` get a per-window random draw), so both are exact
  constants across the train split; z-scoring a zero-variance channel
  divides by std=0 -- **caught by a training smoke test** (job 53016)
  producing NaN loss within the first epoch when `beta` was originally
  included, before this fix -- see `data/qg_neural.py`'s `PARAM_KEYS`
  comment) broadcast as constant spatial channels (matches
  `MonaiDirectUNetQG`'s existing, previously-dead broadcast code), z-scored
  via a new `precompute_qg_norm_stats.py --output-params` companion stats
  file (`experiments/qg_param_norm_stats.pt`) -- required whenever
  `cond_mode != "none"` since the 3 params span ~9 orders of magnitude raw
  (`rd≈1.5e4` vs `rek≈5.8e-7`); a raw constant-channel broadcast would
  make one or two params numerically dominate or vanish next to the
  z-scored psi/obs channels.
- **Two variants, controlled by `data.qg_neural.QGNeuralDataset`'s new
  `cond_mode`**:
  - **Q3 (`cond_mode="true"`, oracle)** — the window's exact true
    `wind_curl` field + exact `true_params`, no jitter, deterministic per
    window. Config: `config/experiment/Q3_direct_unet_s0_oracle_cond.yaml`.
  - **Q4 (`cond_mode="noisy"`)** — mirrors the L96 SDA3 CFM study's
    per-step **resampled** corruption severity (not one fixed S1 bias):
    every training draw samples a fresh random fraction in
    `[0, noisy_max]` (default 1.5, matching L96's `noisy_da_max`) of the
    full S1-style wind corruption (`s1_amp_bias`/`s1_loc_sigma_frac`/
    `s1_sigma_eta_frac`, via `data.qg._make_corrupted_wind_state` at scaled
    severity) and the `rd`/`rek` param bias (`s1_param_bias`), reusing the
    existing S1 corruption machinery directly rather than a separate S1
    dataset/scenario wrapper (the corruption is synthesized straight from
    `wind_state_true`/`true_params`, so it works on the same base S0-scenario
    windows Q1/Q2/Q3 already use — no new train/val window-loading path
    needed). Config: `config/experiment/Q4_direct_unet_s1_noisy_cond.yaml`.
- **Both configs share Q1's model_type=`direct_unet`** (MONAI circular 2D
  U-Net), so `train_qg_neural.py` gained a new `--exp-id` flag to pick the
  YAML explicitly (Q1/Q2's auto-derived `Q{1,2}_{model_type}_s0` naming
  can't disambiguate Q1 vs Q3 vs Q4, all `direct_unet`) -- same split
  seeds/obs protocol/psi normalization/cosine-LR default as Q1 (see Q1's
  own config comments), only the forcing+param conditioning differs.
- **Training smoke tests + perf fix (2026-09-10)**: 2-epoch smoke tests on
  an A40 (matching Q1's own recorded ~225s/epoch) validated both configs
  train cleanly end-to-end: Q3 ~220s/epoch (essentially free -- its
  conditioning just daily-bins the already-computed true `wind_curl`), Q4
  ~504-533s/epoch (~2.4x slower). Root-caused: Q4's `cond_mode="noisy"`
  path called `data.qg._make_qg_dynamics(cfg)` -- which rebuilds the *full*
  spectral PV-inversion machinery (wavenumber grids, filters) from scratch,
  none of which `wind_curl_field` actually needs -- on **every training
  draw** (~1000/epoch at batch_size=2), serialized with GPU training since
  `train_qg_neural.py` hardcoded `num_workers=1` (the exact bottleneck
  PyTorch Lightning's own "may be a bottleneck" warning flagged in the
  logs). Fixed at the source with two changes, not a GPU-side rewrite of
  the curl computation itself (moving that math to GPU wouldn't have
  addressed either the redundant object rebuild or the lack of CPU/GPU
  overlap, and doing GPU work inside forked DataLoader workers is awkward
  regardless): (1) `data/qg_neural.py`'s `_cached_qg_dynamics(cfg)` caches
  the `QGDynamics` object per-cfg (deterministic given `cfg`, so no result
  change -- confirmed bitwise-identical `wind_curl_field` output vs. an
  uncached rebuild); (2) `train_qg_neural.py` gained `--num-workers`
  (default 4, was hardcoded 1) + `persistent_workers=True` so multiple
  draws' CPU-bound prep happens in parallel across worker processes while
  the GPU trains on the previous batch, instead of serially blocking it --
  `batch/run_qg_q{3,4}_train.sbatch` bumped to `--cpus-per-task=6` to back
  this. First full-run launch was also switched from A40 to an idle A100
  node (`sl-mee-br-206`) for additional speed: Q3 (GPU-bound) got a real
  ~2.7x speedup there (~82s/epoch steady-state); Q4 (CPU-bound before this
  fix) only got ~1.28x, confirming the diagnosis that Q4's bottleneck
  wasn't GPU throughput. Both jobs were restarted with the worker/caching
  fix applied before any significant training progress was lost.
- **Training collapse + forcing-normalization fix (2026-09-10)**: the
  relaunched full 200-epoch Q3 run (A100, with the perf fix) collapsed at
  epoch 28 -- train/val loss froze at an exact constant (train_loss≈2.0,
  val_loss≈1.732) for 13+ consecutive epochs after declining healthily
  through epoch 27, confirmed via the raw PyTorch Lightning CSV logger (not
  a tqdm-rendering artifact). The frozen value is diagnostic: it exactly
  matches the loss a model gets from predicting the (normalized) zero mean
  for both targets -- `loss_psi≈1` (MSE of 0 vs. a unit-variance z-scored
  target) `+ q_loss_weight·loss_q≈1` (same zero-prediction baseline, scaled
  by the derived `q_loss_weight=1/Var(q)`) `≈2.0` -- i.e. **the network
  died** (collapsed to a constant output) after some destabilizing event.
  Root cause: the forcing field was left unnormalized on an *unverified*
  design assumption ("its own per-grid-cell scale is already comparable to
  the z-scored channels it's concatenated with") -- empirically checked
  after the collapse and found false by ~12-13 orders of magnitude: raw
  `wind_curl` is `O(1e-13)-O(1e-12)` (`wind_amp` ranges 0-3e-11 over a
  Witch-of-Agnesi profile peaking at ~1), vs. the unit-variance psi/obs/
  z-scored-param channels it's concatenated with. **Fixed**: forcing is now
  z-scored too, with a single **global scalar** mean/std (not per-grid-cell,
  to preserve the spatially-meaningful storm-location pattern, just rescale
  its overall magnitude) computed by `precompute_qg_norm_stats.py`'s new
  `--output-forcing` -> `experiments/qg_forcing_norm_stats.pt`, required
  (like `param_norm_stats`) whenever `cond_mode != "none"`. Both jobs were
  stopped before completing (Q4 hadn't reached a comparable epoch count
  yet, at ~epoch 21, so it's unknown whether it would have hit the same
  failure, but the same latent bug applied to it too) -- relaunch pending
  a longer validation run (a 2-epoch smoke test cannot catch a divergence
  that only manifests dozens of epochs in; this failure was invisible to
  every smoke test run so far).
- **Tests**: `tests/test_qg_neural.py` (cond_mode none/true/noisy shapes +
  determinism/diversity contracts, param-norm-stats requirement, collate
  params stacking, `build_model` YAML-wiring regression) and
  `tests/test_monai_unet_qg2d.py` (forward-pass conditioning sanity check:
  zeroing forcing/params must change the output).

### Q3/Q4 full training + cross-scenario S0/S1 evaluation outcome (2026-09-11)

Both trained their full 200 epochs cleanly on an A100 (`sl-mee-br-206`) after
the forcing-normalization fix above: **Q3** 5.02h, final S0 PSI EV=0.913 /
PV-q EV=0.205 (slightly ahead of Q1's 0.909/0.197); **Q4** 9.71h, S0
PSI EV=0.913 / PV-q EV=0.205 (statistically indistinguishable from Q3 on
S0 -- matches the L96 SDA study's own finding that conditioning-regime
differences vanish on the easy/matched case).

**Cross-scenario S0/S1 eval — new `cond_mode="scenario"`** (added to
`data/qg_neural.py`/`QGNeuralDataset`, distinct from `"true"`/`"noisy"`):
deterministic, reads whatever the window's *own* scenario wrapper
(`QGS01Dataset._scenario_window`) designates as believed --
`wind_state_corrupted`/`da_params` (equal to the true values exactly on an
S0-scenario window) -- giving a genuine apples-to-apples test of the same
model-error sensitivity DA baselines face, unlike `"true"`/`"noisy"` which
always ignore the scenario label (fine for training diversity, useless for
an S1 eval). New `eval_qg_neural_s0_s1.py` runs Q1 (`cond_mode="none"`,
scenario-agnostic by construction) + Q3/Q4 (`cond_mode="scenario"`) on both
test_s0/test_s1.

**Two apples-to-apples config bugs caught before trusting the first
result** (see `feedback_apples_to_apples_benchmarks` memory): (1) the
initial run used `train_qg_neural.py`'s own lag=1.0d/noise=0.01 training
distribution, not comparable in absolute terms to the DA baselines'
lag=5.0d/noise=0.05 reference case -- `eval_qg_neural_s0_s1.py` gained
`--lag-days`/`--noise-frac` (mirroring `eval_qg_q1_lag5_noise05.py`'s
existing cache-reuse trick: the cached truth is keyed by the *whole*
`QGConfig` including obs/IC fields, so naively rebuilding `QGConfig` with
different lag/noise and calling `make_qg_s0_s1_datasets` would silently
miss the cache and trigger an hours-long from-scratch rollout -- instead
load the cached truth at its original key, then cheaply redraw obs/init-
state at the new lag/noise via `QGS01Dataset._generate_obs_ic`). (2) Even
after matching lag/noise, the S1 severity itself was still wrong --
`_scenario_forcing_and_params`'s `da_params`/`wind_state_corrupted` used
`QGConfig`'s **class default** `s1_param_bias=s1_amp_bias=0.15`, but the
actual DA S1 reference campaign (`qg_da_s1_scratch.py`) uses an explicit
**0.1/0.1** override (deliberately milder than the 0.15 default, chosen
because 0.15 sits right at ETKF's divergence edge in isolation -- see the
2026-09-10 QG revised-S1 section above). Added `--s1-param-bias`/
`--s1-amp-bias` overrides to fix this.

**Final result, fully matched (lag=5.0d/noise=0.05/bias=0.1, identical to
the DA campaign)**:

| scheme | S0 ψ EV | S0 q EV | S1 ψ EV | S1 q EV | Δψ | Δq |
|---|---|---|---|---|---|---|
| ETKF | 0.921 | 0.405 | 0.874 | 0.307 | −0.047 | −0.098 |
| EnKF | 0.947 | 0.481 | 0.896 | 0.331 | −0.051 | −0.150 |
| Strong-4DVar | 0.971 | −0.126 | 0.931 | −0.857 | −0.040 | −0.731 |
| Weak-4DVar | 0.966 | −0.035 | 0.947 | −0.501 | −0.019 | −0.466 |
| Q1 (obs-only) | 0.897 | 0.161 | 0.897 | 0.161 | 0.000 | 0.000 |
| Q3 (oracle) | 0.905 | 0.177 | 0.891 | 0.141 | −0.014 | −0.036 |
| Q4 (noisy-trained) | 0.903 | 0.171 | 0.901 | 0.164 | −0.002 | −0.007 |

**Q4 is essentially immune to S1 model error** (Δq=−0.007, ~14x smaller
than even ETKF's own degradation) -- the noisy-conditioning training
design works as intended. **Q3, at the correctly-matched bias level, is
actually *less* sensitive than ETKF/EnKF** both in absolute Δq and
relative-to-own-baseline terms (20% vs 24%/31%) -- an earlier, wrong-bias
(0.15) run had suggested Q3 was more fragile than DA, which doesn't hold
once the severity genuinely matches. Both neural schemes are dramatically
more robust than either 4DVar variant's collapse. Still an out-of-
training-distribution eval for all three neural models (trained at
lag=1.0d/noise=0.01, evaluated here at lag=5.0d/noise=0.05) -- same caveat
`eval_qg_q1_lag5_noise05.py` already carried for Q1 alone. Results:
`reports/qg/outputs/qg_neural_s0_s1_cross_scenario/results_lag5_noise0.05_bias0.1.json`.
Not yet folded into `qg_neural_report.md` -- open follow-up.

### Q5: Q4 + initial-condition input, trained at the DA reference case's own
### lag/noise (2026-09-11, planned, not yet implemented)

Motivated by the above: DA baselines get real skill from their background
(`init_state`, rolled forward from a sampled lag) that Q1/Q3/Q4 get zero
equivalent of -- confirmed neither `init_state` nor any IC-derived quantity
is referenced anywhere in `data/qg_neural.py`/`models/monai_unet_qg2d.py`/
`train_qg_neural.py`. Q5 = Q4's noisy forcing+param conditioning **plus**
the raw IC as a 4th input, trained at `init_lag_days=5.0`/
`obs_noise_std_frac=0.05` (the DA reference case's own values, not Q1/Q3/Q4's
1.0/0.01 defaults) -- both because that's what makes the IC actually
informative (a lag=1.0-old snapshot is barely decorrelated; lag=5.0 makes it
a genuine partial-information input like DA's own background) and because it
subsumes the "should Q1/Q3/Q4 be retrained at the DA reference case?"
question into something more purposeful than just retraining the same
recipe at a harder setting.

- **IC representation**: the exact `init_state` `_generate_obs_ic` samples
  (raw PV/q, always from the *true* trajectory -- like obs, never
  scenario-corrupted, since physically it represents a recent
  analysis/observation, not the DA model's own internal forecast). Inverted
  to ψ via the existing per-window cached spectral inverter and normalized
  with the **existing global `psi_norm_stats`** (no new stats file).
- **A third conditioning class, architecturally**: unlike `forcing` (varies
  per day) or `params` (a scalar vector), the IC is one static field for the
  whole window, broadcast identically across all `days` -- needs a new
  `ic_dim` (2, matching `nlayers`) on `MonaiDirectUNetQG`, alongside
  `cond_extra_dim`/`param_dim`, with its own per-window (not per-day)
  broadcast.
- **Noisy-conditioning severity, revised for Q5**: `s1_param_bias=0.1`,
  `s1_amp_bias=0.1` (kept at the actual DA-reference value, unlike Q4's
  0.15-default-derived range) with `noisy_max=2.0` -- samples the effective
  bias fraction uniformly over `[0, 0.2]` (`frac~U(0,2)` × 0.1), i.e. up to
  2x the DA reference severity, not Q4's indirect `1.5×0.15=0.225` ceiling.
  Requires new `--s1-param-bias`/`--s1-amp-bias` CLI overrides on
  `train_qg_neural.py` itself (previously only `eval_qg_neural_s0_s1.py`
  had them; training always silently used the 0.15 class default before
  this) -- no behavior change for Q1/Q3/Q4 (they never set these flags).
- New config `Q5_direct_unet_s1_noisy_ic_cond.yaml`.

### Q5 full training + Q3-noise0.05 collapse/gradient-clip ablations (2026-09-12/13)

**Q5 trained cleanly, 200 epochs, A100**: 9.08h, S0 psi EV=0.935 / q EV=0.372
-- meaningfully ahead of Q1/Q3/Q4 (psi~0.90-0.91 / q~0.16-0.21) on both
metrics, and epoch-matched loss curves (checked at epoch 25 across runs)
confirm this is real faster/lower convergence, not an artifact of different
training lengths -- consistent with the design intent: the raw IC gives a
much more direct trajectory signal than forcing/param conditioning alone.

**Also retrained Q3 at the same updated obs config as Q5** (`noise=0.05`,
named **Q3-noise0.05**, not "Q3-lag5" -- `init_lag_days` has zero effect on
Q3, it never reads `include_ic`; see
`config/experiment/Q3_direct_unet_s0_oracle_cond_noise05.yaml`'s full
naming-history comment). This run **collapsed at epoch ~19-20** -- same
frozen-constant-loss signature as the epoch-28 forcing-normalization
collapse above, but confirmed a *different* root cause (that fix is
unchanged/active). The only functionally relevant change vs. the original
(clean) Q3 is `obs_noise_std_frac` 0.01→0.05.

**Two ablations launched to disambiguate** ("was it bad luck, or the noise
level"): (1) same config, `--gradient-clip-val 1.0` instead of the default
10.0; (2) same config, `--seed 123` (default clip) -- the project had no
seeding infrastructure before this (`pl.seed_everything()` newly added to
`train_qg_neural.py`), so every prior run used whatever untracked randomness
the process started with.

**Result: gradient clipping, not seed, is the fix.** `seed123` collapsed too
-- just later (epoch 22-24), then stayed frozen at the exact same dead
value through epoch 66+ (confirmed permanent, not transient). `gradclip1`
ran cleanly through all 200 epochs, clearing both collapse windows: final
S0 psi EV=0.911 / q EV=0.189, 4.74h. Working hypothesis (documented in the
config): the 5x higher obs noise occasionally produces an outlier
daily-aggregated obs value (sparse `random_columns` sampling) whose
gradient the looser default clip doesn't fully contain.

Re-ran Q5 at `--gradient-clip-val 1.0` too, for consistency (Q5 shares
`obs_noise_std_frac=0.05` with Q3-noise0.05) even though Q5 never collapsed
at the default clip -- confirmed no downside: psi EV=0.934 / q EV=0.385
(both within noise of the original 0.935/0.372, q EV if anything slightly
better), 12.13h. **Adopted `gradient_clip_val: 1.0` as the standing default**
in both `Q3_direct_unet_s0_oracle_cond_noise05.yaml` and
`Q5_direct_unet_s1_noisy_ic_cond.yaml` (`train_qg_neural.py`'s
`--gradient-clip-val` now defaults to `None` and falls back to the
experiment YAML's `training.gradient_clip_val`, then 10.0 -- same
YAML-fallback pattern already used for `obs_noise_std_frac`/`init_lag_days`,
so the config alone is the source of truth, no CLI flag required at launch
time). The validated checkpoints live at
`experiments/Q3_direct_unet_s0_oracle_cond_noise05_gradclip1/` and
`experiments/Q5_direct_unet_s1_noisy_ic_cond_gradclip1/`.

**Q5 added to the DA-matched cross-scenario eval** (`eval_qg_neural_s0_s1.py`,
using the gradclip1 checkpoints for both Q3-noise0.05 and Q5; full config
match: lag=5.0d/noise=0.05/`s1_param_bias=s1_amp_bias=0.1`, identical to the
DA campaign and to the Q1/Q3/Q4 table above):

| scheme | S0 ψ EV | S0 q EV |
|---|---|---|
| Q1 (obs-only) | 0.897 | 0.161 |
| Q3 (oracle) | 0.905 | 0.177 |
| Q4 (noisy-trained) | 0.903 | 0.171 |
| **Q5 (noisy + IC)** | **0.936** | **0.376** |

vs. the DA baselines' S0 reference (`qg_da_report.md`): EnKF 0.947/**0.481**,
ETKF 0.921/*0.405*, Strong-4DVar **0.971**/-0.126, Weak-4DVar *0.966*/-0.035.
**Q5 substantially closes the PV-q gap** left by Q1/Q3/Q4 (0.376 vs their
~0.16-0.18) -- within reach of ETKF (0.405) and clearly ahead of both
4DVar variants, though still behind ETKF/EnKF; on ψ, Q5 (0.936) beats
Q1/Q3/Q4 (~0.90) but still trails all 4 DA methods (0.92-0.97). S1 numbers
for Q5 (with the `"scenario"` cond_mode + true IC, mirroring Q3/Q4's S1
eval) not yet pulled into this table -- open follow-up, same as folding all
of this into `qg_neural_report.md`.

### ETKF inflation sensitivity, part 2: finer grid + additive inflation (2026-09-11)

Follow-up to the ETKF inflation sweep above, still N=10, revised S1
combined config, `qg_da_s1_scratch.py --etkf-additive/--etkf-ridge` (new CLI
overrides added to the scratch driver). Two questions: (a) is the 1.0→1.05
gap a hard cliff or a graded decline, and (b) does **additive** inflation
(`ETKF.etkf_additive`, Gaussian noise added directly to the ensemble each
step, raw q-field units — never exercised before; q std ≈2.06e-5 at this
scenario/resolution) behave differently from multiplicative inflation.

**Multiplicative inflation, finer grid**:

| inflation | q EV | psi improv |
|---|---|---|
| 1.00 | 0.253 | — |
| 1.01 | 0.220 | 1.45x |
| 1.02 | 0.112 | 1.36x |
| 1.03 | −0.029 | 1.26x |
| 1.04 | −0.559 | 1.15x |
| 1.05 | −39.5 | (prior data — catastrophic) |

Not a cliff — a steep but continuous monotonic decline from 1.00 to 1.04,
crossing zero between 1.02/1.03, **then** a genuine discontinuity (2 orders
of magnitude worse) between 1.04 and 1.05. So the divergence is real filter
blow-up, not just "the metric happens to cross zero here."

**Additive inflation** (`etkf_additive`, multiplicative inflation fixed at
1.0, values as a q-std fraction):

| etkf_additive | ~% of q std | q EV |
|---|---|---|
| 0 (baseline) | 0% | 0.253 |
| 2e-7 | ~1% | 0.251 |
| 1e-6 | ~5% | 0.224 |
| 2e-6 | ~10% | 0.138 |
| 5e-6 | ~24% | −0.067 |

Also monotonically degrades EV, but **gracefully** — no catastrophic
divergence at any tested magnitude (up to ~24% of the field's own std),
unlike multiplicative inflation's blow-up past 1.04. Data:
`reports/qg/outputs/qg_repro_validation_s1/etkf_n10_{inflation,additive}*.json`.

**Refined interpretation of the open question**: neither inflation flavor
*helps* — both are monotonically neutral-to-harmful across their entire
tested range in this stacked-error (param+wind+da_nx=32) S1 regime, they
just fail at very different rates. This weakens "ETKF just needs the right
inflation" as an explanation for EnKF's edge, and points more toward a
**structural** difference: ETKF's deterministic square-root ensemble
transform vs. EnKF's stochastic perturbed-observations update — the former
has no built-in randomization to counteract ensemble collapse/skew under
strong nonlinearity, the latter does. Still untested directly. Next steps
(not yet run): (1) `etkf_ridge` sweep (Kalman-gain regularization, also never
exercised — targets the transform-matrix inversion specifically, unlike
inflation which targets ensemble spread); (2) repeat the same
inflation/additive/ridge sweeps on **EnKF itself** (never stress-tested — if
EnKF also degrades under any of its own inflation, the story is
ensemble-collapse-general rather than ETKF-transform-specific); (3)
ensemble-size sweep (`N_ensemble`, hardcoded 80 everywhere so far).

Also discovered running many (7) of these short GPU jobs concurrently in the
background silently kills most of them (5/7 died mid-run, no traceback, exit
code 0) — not a numerical issue (both stable and unstable configs died
identically) but resource contention. Re-running strictly one-at-a-time
fixed it. Worth remembering for any future batch of short interactive GPU
sensitivity runs on this node.

### ETKF/EnKF sensitivity, part 3: ridge sweep + EnKF cross-check — resolves the open question (2026-09-11)

Consolidated 28-config sweep (`qg_da_sensitivity_sweep_scratch.py`, one
process, cache loaded once) filling the remaining gaps from parts 1-2: S0
inflation + additive grids (never run on S0 before), `etkf_ridge` grid on
S1 (never exercised), and EnKF's own inflation grid on **both** S0 and S1
(EnKF had never been hyperparameter-swept at all — every prior EnKF result
used the fixed `inflation=1.0` default). Full report generator:
`reports/qg/generate_da_sensitivity_report.py` →
`reports/qg/outputs/da_sensitivity_s0_s1_report.md`.

**Bug caught before wasting GPU time**: the first attempt's S0 window
builder skipped `QGS01Dataset`'s scenario-wrapping step (`_scenario_window`,
which populates `da_params`/`da_model`/`da_nx` — required unconditionally by
`_build_dyn`, not just for S1's biased case), so all 10 S0 configs failed
instantly with `KeyError: 'da_params'`. Fixed by wrapping S0 windows through
`QGS01Dataset(cfg, "test_s0", base_windows=...)` exactly like the S1 builder
already did, mirroring `data.qg._scenario_window`'s `"test_s0"` branch
(unbiased `da_params`, `da_model="qg2l"`, `da_nx=cfg.nx`).

**Key result 1 — inflation-driven divergence is shared, not ETKF-specific**:
ETKF and EnKF collapse in near-lockstep under multiplicative inflation, on
**both** S0 and S1. S0 q EV at inflation={1.00...1.05}: ETKF
{0.402,0.402,0.296,0.147,−40.4,−123.5} vs EnKF
{0.463,0.435,0.313,0.156,−43.0,−124.3} — virtually the same curve, same
catastrophic threshold (between 1.03 and 1.04). S1 shows the identical
pattern (ETKF {0.253,...,−39.5} vs EnKF {0.279,...,−39.5} at
inflation={1.00,1.05}, both also diverging by inflation=1.1). This **rules
out** "ETKF's deterministic square-root transform makes it uniquely fragile
to over-inflation" as the explanation for EnKF's edge — both ensemble
methods share this fragility equally.

**Key result 2 — `etkf_ridge` is the real, actionable lever**: unlike
either inflation flavor (both purely harmful, see part 2), increasing the
Kalman-gain transform-matrix ridge regularization **monotonically helps**
ETKF on S1: q EV rises 0.253 (default, ~1e-4-equivalent) → 0.258 (1e-3) →
0.275 (1e-2) → 0.303 (1e-1) → **0.332 (ridge=1.0)** — which *exceeds* both
EnKF's own N=10 baseline here (0.279) and EnKF's N=100 headline number
(0.331, from the main S1 table above). EnKF has no equivalent knob (no
deterministic transform-matrix inversion to regularize), so this is
specific to fixing ETKF's own weakness.

**Refined conclusion**: the previously "unexplained" EnKF>ETKF gap on S1
looks like it was largely an artifact of running ETKF with an
**under-regularized transform-matrix inversion** (the implicit
`etkf_ridge~1e-4` default used everywhere in the existing benchmark), not a
fundamental method limitation, and not an ensemble-collapse/inflation
story (that part is shared equally by both methods).

**N=100 confirmation (2026-09-11, same day) — CONFIRMED, gap closed**:
before committing to the expensive N=100 run, a quick N=10 check of whether
ridge keeps helping past 1.0 found a peak, not unbounded improvement: S1 q EV
0.253(default)→0.332(ridge=1.0)→**0.335(ridge=2.0, best)**→0.323(ridge=5.0);
S0 (untested before, no model error) shows the same qualitative pattern,
0.402(default)→0.463(ridge=0.1..1.0 plateau). Picked `ridge=1.0` as one
value that's near-optimal on both scenarios rather than tuning per-scenario.
Ran the full N=100 confirmation via `qg_n100_ridge_confirm_scratch.py` +
`batch/run_qg_n100_ridge_confirm.sbatch` (jobs 53159/53160 — the first
interactive attempt at this silently OOM'd under this session's cgroup cap,
the same symptom `run_qg_s1_full100.sbatch`'s 2026-09-10 note already
documented; a real sbatch job fixed it, same as that prior fix):

| | ETKF (default) | EnKF | **ETKF + ridge=1.0** |
|---|---|---|---|
| S0 psi EV | 0.921 | 0.947 | **0.957** |
| S0 q EV | 0.405 | 0.481 | 0.476 |
| S0 q layer2 EV | ~0.34 | ~0.40 | **0.410** |
| S1 psi EV | 0.874 | 0.896 | **0.926** |
| S1 q EV | 0.307 | 0.331 | **0.357** |
| S1 q layer2 EV | — | — | 0.266 |

ETKF+ridge=1.0 beats EnKF outright on **both** fields on S1, and beats it on
psi (ties within noise on q, −0.005) on S0 — no trade-off on the unobserved
layer either (S0 q layer2 improved over plain ETKF's ~0.34, not sacrificed).
This confirms the N=10 finding holds at full scale: **`etkf_ridge=1.0` is a
strictly better ETKF config than the implicit default** on this reference
case. Data: `reports/qg/outputs/qg_repro_validation/etkf_ridge1.json` (S0,
N=100), `reports/qg/outputs/qg_repro_validation_s1/etkf_ridge1.json` (S1,
N=100) — kept as distinctly-tagged files, NOT overwriting the canonical
`etkf.json` reference numbers pending a decision on promoting
`etkf_ridge=1.0` to the default ETKF config in the main benchmark table
(`qg_da_report.md`/PLAN.md's S0/S1 tables above) and updating
`run_qg_baselines.py`'s/`sweep_qg_baselines.py`'s CLI default. `N_ensemble`
remains unswept (still hardcoded 80 everywhere).

### `etkf_ridge=1.0` promoted to the default ETKF config (2026-09-12)

Following the N=100 confirmation above, `etkf_ridge=1.0` is now the actual
default, not just a documented-but-unused finding:

- `evaluation/run_qg_baselines.py`'s `run()` default changed `etkf_ridge=0.0`
  → `1.0`; its CLI gained `--etkf-ridge`/`--etkf-additive` flags (previously
  only reachable by calling `run()` programmatically, e.g. from scratch
  drivers). `evaluation/sweep_qg_baselines.py`'s `--etkf-ridge-list` fallback
  changed `[0.0]` → `[1.0]`.
- **Deliberately NOT touched**: `evaluation/baselines.py`'s shared `ETKF`
  class constructor default (`etkf_ridge: float = 0.0`) — that class is
  used by both L96 and QG (per
  [[project_qg_da_baselines_plan_2026-09-05]]'s "merged with the L96/joint/
  ES work" note), and the sensitivity study was QG-specific; changing the
  shared class default would have silently changed L96 ETKF behavior too,
  unvalidated. The QG-specific driver-level default is the right place for
  this.
- Canonical committed N=100 JSONs updated: `reports/qg/outputs/
  qg_repro_validation{,_s1}/etkf.json` now hold the `ridge=1.0` results
  (previously in `etkf_ridge1.json`, which is kept as-is, now identical to
  `etkf.json`); the old ~1e-4-floor results are archived as
  `etkf_ridge_default.json` in both directories for provenance, not deleted.
- `reports/qg/generate_qg_da_report.py` significantly extended: now covers
  **both** S0 and S1 (closing the "Not yet done: folding these S1 numbers
  into `qg_da_report.md`" note above), adds a condensed "Hyperparameter
  sensitivity analysis" section (inflation/ridge/4DVar findings, linking to
  the dedicated `da_sensitivity_s0_s1_report.md` for full sweep tables), and
  a closing "Synthesis: best configuration per method (S0 vs S1)" table.
  `reports/qg/generate_qg_neural_report.py`'s ETKF description/summary-table
  row updated to match (was still citing the old 0.921/0.405 numbers).

### 4DVar (Strong/Weak) hyperparameter sensitivity — negative result (2026-09-12)

Same motivation as the ETKF work above: both 4DVar variants collapse on PV
q layer2 under S1 (Strong -0.857, Weak -0.501 at N=100, vs ETKF/EnKF
staying positive) — checked whether a covariance-weighting sweep could fix
or explain it the way `etkf_ridge` did for ETKF, before concluding it's a
fundamental limitation.

Swept Strong-4DVar's `b_var_scale` (background-covariance whitening scale)
and Weak-4DVar's `q_var_scale` (per-step model-error weight) on S1, N=5
(4DVar is ~15-20x more expensive per window than ETKF/EnKF — a single
strong4dvar window took 364s in a timing probe, vs ETKF's ~20-47s; N=5 was
chosen as a cost/noise tradeoff, one order of magnitude below the ETKF
sweeps' N=10). Each window already gets its own `QG4DVar`/LBFGS instance
inside `run()`'s loop (no risk of the "batching independent LBFGS problems"
pitfall from the original 2026-09-05 QG DA baselines plan).

**Result: neither lever helps, unlike ETKF's ridge**:

| Strong-4DVar | b_var_scale | q EV | psi EV |
|---|---|---|---|
| | 0.3 | -1.06 | 0.706 |
| | 1.0 (default) | -1.07 | 0.706 |
| | 3.0 | -1.11 | 0.698 |

| Weak-4DVar | q_var_scale | q EV | psi EV |
|---|---|---|---|
| | **0.1 (default)** | **-0.91** | 0.687 |
| | 0.3 | -0.99 | 0.606 |
| | 1.0 | -1.09 | 0.545 |
| | 3.0 | -1.11 | 0.544 |

`b_var_scale` is essentially flat across an order of magnitude (mild
decline, not a peak like ridge showed). `q_var_scale` is monotonically
*worse* the higher it's pushed above the existing default 0.1 — giving the
per-step model-error controls more freedom to correct actively hurts
rather than helping, the opposite of the initial hypothesis (that S1's
real model error would need *more* absorption capacity). Both methods'
existing defaults were already at or near the best point in the explored
direction; **no config change made** (unlike ETKF). Caveats: N=5 is noisy
(absolute EVs differ from the N=100 canonical numbers, though the ranking
direction is consistent), and only q_var_scale values *above* 0.1 were
tested — a small chance a lower value (0.01-0.03) does better still, not
checked given the cost. Consistent interpretation: the 4DVar q-layer2
collapse looks like a structural limitation (a single optimized trajectory
has no ensemble spread to exploit on the unobserved layer), not a fixable
covariance-tuning gap the way ETKF's transform-regularization gap was.

Data: `reports/qg/outputs/qg_4dvar_sensitivity_sweep/*.json` (N=5, 7
files). Scratch driver (not committed): `qg_4dvar_sensitivity_sweep_scratch.py`.

### ETKF obs-configuration sensitivity: cols_per_day=8/16 + new psi2 (lower-layer) point obs + new cols_sampling="random" mode (2026-09-13/14)

New obs-density experiment series: `cols_per_day=8/16` (vs. the reference
4/day) plus a genuinely new capability -- observing the *lower* layer
(psi2), never directly observed before this -- at 10 random points/day
alongside the existing psi1 columns. Motivated by the same q-layer2/unobserved-
deep-layer weakness the ETKF-ridge and 4DVar sensitivity work above kept
running into: does giving the DA method *any* direct information about the
deep layer help, versus just adding more of the same (biased, upper-layer-
only) observation?

**Two new capabilities implemented** (both opt-in, zero behavior change at
their default-disabled value):

- **`psi2_points_per_day`** (`QGConfig`): independent lower-layer (psi2)
  random-point obs, `P`/day, drawn like the existing column sampler (own
  randomly-sampled intra-day step, no two points of the same day collide)
  but generalized to an arbitrary layer and to single points instead of
  full columns (`data.qg._layer_field`/`_generate_random_point_observations`).
- **`cols_sampling="random"`** (`QGConfig`, default `"sequential"`):
  upper-layer (psi1) obs as `cols_per_day` random `(t, x)` column-point
  draws across the **whole day**, allowing **multiple columns at the same
  timestep** -- an alternate sampling *mode* for the same `cols_per_day`
  density knob, not a separate count (an earlier version of this work
  briefly introduced a second `col_points_per_day` field before this
  naming/design was cleaned up to keep `cols_per_day` the single density
  parameter). Motivated by discovering `cols_per_day=16` is infeasible
  under the default `"sequential"` mode: at the reference `dt=7200s`,
  `steps_per_day=12`, and that sampler enforces one column per step with
  **no** two columns of the same day sharing a step -- so more than
  `steps_per_day` columns/day can never be scheduled. This is a
  **pre-existing bug** (nobody had tried `cols_per_day > steps_per_day`
  before): the collision-avoidance loop just spins forever instead of
  failing cleanly -- confirmed via an isolated 15s-timeout call that hung
  (exit 124). Added a clear `ValueError` guard to both the column and
  (analogous) point samplers instead of leaving the silent hang, and added
  `cols_sampling="random"` as the actual fix for wanting a higher density
  (removes the per-step ceiling entirely, no code changes needed elsewhere
  since the assimilation side already treats "list of columns per time" as
  its native contract, never actually restricted to length 1).

**Machinery reused, not rebuilt**: both new streams route through the
combined-observation machinery built for this exact purpose during the
`etkf_ridge` sensitivity work (`_psi_h_combined`, `_combined_index_at`,
`_build_qg_col_point_loc_matrices`, and `ETKF._per_time`'s `idx.numel()`-
based per-time-varying observation width) -- that machinery was already
general enough to handle multiple columns per step and an optional extra
point per step; only the *generation* functions and a `_make_obs_system`
routing check were new. `evaluation/baselines.py`'s shared `ETKF` class
itself needed no further changes.

**N=10 results, ETKF (etkf_ridge=1.0 default), S0 (reference) + S1
(revised model-error case)**:

| config | S0 q full | S0 q layer2 | S0 psi full | S1 q full | S1 q layer2 | S1 psi full |
|---|---|---|---|---|---|---|
| baseline (cols=4, sequential) | 0.467 | 0.399 | 0.907 | 0.328 | 0.231 | 0.829 |
| cols=8 (sequential) | 0.531 | 0.433 | 0.922 | 0.326 | 0.191 | 0.732 |
| cols=4, random (sanity check) | 0.466 | 0.397 | 0.908 | 0.339 | 0.242 | 0.835 |
| cols=16, random | 0.524 | 0.403 | 0.895 | **0.103** | **-0.100** | **-1.189** |
| **psi1(4)+psi2(10)** | **0.502** | **0.447** | **0.924** | **0.389** | **0.316** | **0.878** |

**Sanity check requested and run (2026-09-14)**: does the new `"random"`
sampling mode reproduce `"sequential"`'s performance at the *same* density
(cols=4), where the two mechanisms should agree? Yes -- all four fields
match within ~0.001-0.010 EV on both S0 and S1 (see the "cols=4, random"
row above vs. the baseline row), well inside N=10 noise, and if anything
marginally *better* on S1, not worse. This rules out the new sampler
introducing some systematic artifact of its own at low density.

**Key finding, directly confirms the original motivation**: on S1 (real
model error), simply observing *more* of the same (biased) upper layer is
not just unhelpful but actively **destabilizing** -- cols=8 already
degrades psi (0.829→0.732) despite double the column density, and
cols=16 (random) **collapses catastrophically** (psi=-1.19, worse than
climatology; q layer2 goes negative). This is graded (baseline→cols8→
cols16-random monotonically worse on S1), not a fluke at one setting,
consistent with more-frequent updates from an increasingly model-
inconsistent obs stream reinforcing rather than correcting the S1 bias.
**Adding independent psi2 observations, by contrast, is the best or
tied-best config on both S0 and S1** -- and by a wide margin on S1
specifically, the only config that doesn't degrade relative to baseline.
Genuinely new information about the previously-unobserved deep layer helps
where more of the same upper-layer information hurts.

**Collapse investigated (2026-09-14), not a bug**: the cols=16 (random)
S1 collapse looked dramatic enough to double-check. Per-window PV-q RMSE
values are all finite and unremarkable (6e-6 to 2.4e-5, no NaN/blow-up),
and the result reproduces across reruns -- so it isn't a numerical/coding
fault. The mechanism: `psi rmse=13973` is *worse* than the free-forecast
psi rmse (8064, improv=0.58x -- the DA analysis is worse than doing
nothing in psi-space), while `q rmse=1.56e-5` still *beats* free forecast
(improv=1.38x) -- the DA update genuinely helps q but the resulting
analysis, once inverted to psi, is worse than no assimilation at all. PV
inversion (`psi = ∇⁻²q`) amplifies *large-scale* (low-wavenumber) error
(1/k² blows up as k→0) while smoothing small-scale error -- so a
large-scale/low-wavenumber bias in the q-analysis (plausibly reinforced by
frequent updates from an obs stream that's increasingly inconsistent with
S1's biased model) can produce a large psi RMSE even while q's own score,
dominated by better-constrained smaller scales, looks only mildly
degraded. Confirmed this isn't specific to the new sampler: `cols=8`
(the *old*, `"sequential"` mechanism) already shows the identical-direction
degradation (psi 0.829→0.732), just less extreme -- cols=16(random) is a
more extreme point on the same real trend, not an isolated artifact.

Caveats: N=10 (screening scale, same convention as the ETKF/4DVar
sensitivity work above); an N=100 confirmation of both the psi1+psi2 win
and the cols=16 collapse magnitude is still the natural next step (the
qualitative direction of both findings is now well-supported, but exact
N=100 numbers aren't in yet).

Data: `reports/qg/outputs/qg_obs_density_sweep/*.json` (N=10, 10 files,
including the cols=4/random sanity-check pair). New tests:
`tests/test_qg_psi2_points.py` (15), `tests/test_qg_cols_sampling.py` (10,
renamed from `test_qg_col_points.py` in the naming cleanup) -- both new obs
streams, the combined H-function/localization machinery, the
`cols_per_day` hang guard, and end-to-end ETKF smoke runs. Scratch driver
(not committed): `qg_obs_density_sweep_scratch.py`. sbatch:
`batch/run_qg_obs_density_sweep.sbatch` (interactive runs in this session
repeatedly died silently around large `torch.load` calls -- same
established fix as the ETKF-ridge N=100 confirmation and the 4DVar
sensitivity work: real sbatch job instead).

**Naming cleanup (2026-09-14)**: the first version of this work introduced
a separate `col_points_per_day` field, but "points" was misleading (it
counts *columns*, same physical quantity as `cols_per_day`, just sampled
differently) and having two competing "columns per day" knobs was
confusing. Refactored to a single density parameter (`cols_per_day`,
unchanged) plus a `cols_sampling` mode selector (`"sequential"` default /
`"random"` new) -- one meaningfully-named field per concept, not two
overlapping ones.

**Tooling gotcha**: adding a new `QGConfig` field (even at a neutral
default) changes `_truth_cache_path`'s hash (it hashes the *entire*
`asdict(cfg)`), silently invalidating every existing hash-keyed truth
cache, including the production 1000/100/100 cache used throughout this
whole session's history. Scratch scripts written *after* a `QGConfig`
field addition must reference the pre-existing cache file by its
already-known name directly rather than recomputing the (now different)
hash. Worth keeping in mind for any future `QGConfig` field addition.

## L96 (two-scale Lorenz-96) — merged to master 2026-08-18

- **Dynamics/DA baselines** (`feat/weighted-fast-coupling` merged into master, SW/MAOOAM excluded):
  - `models/lorenz96_dynamics.py`, `data/lorenz96.py` — two-scale L96 (NO=8, J=4, state_dim=40, weighted fast coupling)
  - `evaluation/run_l96*.py` + `reports/outputs/l96_baseline_report.md` — Waves 1-4 DA sweeps + ETKF ablation, pooled-EV metric
  - `models/dynamics.py` `get_dynamics()` supports only `lorenz63`/`lorenz96` (SW/MAOOAM deferred)
- **Training infrastructure** (built, no runs launched):
  - `conf/schema.py` → `DataConfig.to_lorenz96_config()`
  - `train.py` → `data.system` dispatch (`lorenz96`), `param_names`-generalized eval
  - configs: `config/experiment/L1_direct_unet_s0s1.yaml`, `L2_vanilla_cfm_s0s1.yaml`
  - test: `tests/test_lorenz96_training.py`

## L96 Neural Training (`feat/l96-neural-training`, from master @ 0687e07)

Implements all-5-param randomization + neural S0/S1 training (commit `3a1c8d5`).
See `L96_NEURAL_TRAINING_PROGRESS.md` for the per-WP tracker and handoff.

- **All 5 params randomized ±20%**: `models/lorenz96_dynamics.py` accepts `c1,h,hx,eps`
  + `F` kwargs; `data/lorenz96.py` `RandomParam`/`RandomBias` + `make_l96_s0_s1_trainval`.
- **Neural `param_dim=0` + `cond_extra_dim=0`** (DirectUNet, VanillaCFM-τ=0): obs-only
  input (no forcing/params conditioning; 24D in, 24D out).
- **DA parity**: `evaluation/run_l96.py` + `evaluate_all_l96.py` pass per-window all-5
  params to DA; S1 uses biased `*_da` params.
- Configs: `config/lorenz96_default.yaml`, `L1_direct_unet_s0s1.yaml`,
  `L2_vanilla_cfm_s0s1.yaml` (both `param_dim=0`; L2 is τ=0).
- Tests: `tests/test_lorenz96_training.py` (11 tests).
- sbatch: `run_one_epoch_tests_l96`, `run_l96_da_consistency`,
  `run_l96_neural_training`, `run_l96_evaluate_all`.

### Multi-agent review workflow (git/PR)

Code changes on this branch go through an implementer → reviewer → verifier loop.
Two execution paths:

- **Option A — GitHub PR**: `.github/workflows/ci.yml` runs ruff + pytest on PRs
  to `feat/l96-*`. Agents use `gh pr create` / `gh pr review` / `gh pr merge`.
  Blocked until `gh auth login` is run interactively (W3).
- **Option B — Local**: `scripts/agent_review_loop.sh <STEP> "<desc>" [--review]`
  provides the same loop with local git (works immediately).

**Run-to-completion policy:** the general rules (branch naming, drive-to-merge, reviewer
identity, CI gate) live in `AGENTS.md` under **Git / PR Workflow** and apply to every
session — follow those. This branch additionally uses `feat/l96-*` as its integration
namespaces subject to the repo ruleset.

**REMINDER:** run `gh auth login` and enable branch protection on `feat/l96-*`
(require 1 PR approval + status checks) to unlock the GitHub PR path.

**Status (2026-08-24)**: Q1–Q3 closed; the canonical L96 benchmark artifact is
`reports/l96/outputs/l96_consolidated_benchmark.md` (all-metric tables RMSE/EV/ES ×
{all/slow/fast} + consistency checks + Hovmöller reconstruction examples, regenerated by
`reports/l96/generate_l96_consolidated_report.py`). Headline: neural beats the best DA
baseline on both S0 and S1 (RMSE 0.62 vs 0.74), degradation ≈1.00 vs DA ≈1.9×.
Note: tables use the **pooled** RMSE convention for every method (the DA cache stores
mean-of-per-window RMSE). ES note (**DA performance pending update**): a normalization
bug in `_ESAccumulator` (accuracy term divided by N twice) deflated cached EnKF/ETKF ES
to spread-dominated values, and separately `Strong4DVar.assimilate_batch` never computed
ES in-run (historical Strong values came from a since-deleted offline backfill). Fixes
merged to `feat/l96-neural-eval-fix`: PR #67 (accumulator formula + esfix infra),
#68 (batch-path Strong ES wiring + relative gate tolerance), #70 (dim-compatible truth
for reduced-dynamics S1 methods; #69 was an accidental empty merge).
**All DA ES values in the canonical s0c cache have been updated** (bug-fixed
accumulator, swapped 2026-08-24); RMSE/EV rankings are unaffected (fresh runs agree
within ~1%) and neural numbers are unaffected. The consolidated report now shows
proper ensemble ES for DA EnKF/ETKF (N=30) and L3 (ens30×10, N=30); deterministic
methods (Strong-4DVar, L1b/L2b/L4/L5/L6) show N=1 MAE ES (marked `*`). The remaining
legacy caches (s0c int200 fw, fw6 int100/int200, legacy int100/int200, dws50) were
**not** swapped: the original esfix array (49383) **failed** (ran against the pre-fix
`rerun_l96_esfix.py` without missing-`es` handling / strict 5e-3 tolerance). Their
regeneration is deferred — the `_ESAccumulator` fix needed to produce correct textbook
ES is now on master (PR #74), so resubmitting `batch/run_l96_esfix.sbatch` against
master and swapping the passing caches is open follow-up (see
"Deferred future work (Phases B & C)" → Phase C-adjacent DA regen note).

**Open questions (L96)** — all answered (standalone DA-parity eval, cached test set):
- **Q1 (REVISED 2026-08-24, L3 ens30 study — see below)**: the original answer
  (multi-τ worse than τ=0, +8.6%) was an artifact of single-sample × 1-step evaluation.
  With N=30 members and proper integration, multi-τ CFM **beats** conditional-mean
  estimation: L3 0.5643 vs τ=0 L2b 0.6290 (−10.4%), also beating DirectUNet L4
  (0.6189) — the new overall best on S0. Decomposition of the published L3 0.688:
  −5.5% from 30-member averaging (sampling variance), −13.2% further from 10 Euler
  steps (integration coarseness). τ=0 control is exactly invariant to n_outer
  (single-Euler-step shortcut), confirming the effect is specific to multi-τ.
- **Q1-original (superseded)**: multi-τ CFM does NOT beat conditional-mean estimation —
  L3 0.688/0.690 vs τ=0 L2b 0.633/0.633 (+8.6%); mirrors the L63 G-series finding
  **at 1 member × 1 step only**; see the ens30 revision above before quoting this.
- **Q2 (answered, L4/L5)**: size sensitivity is model-dependent — small DirectUNet
  (L4) slightly beats default L1b (0.619 vs 0.622); small τ=0 CFM (L5) is worse than
  default L2b (+4.3%). CFM benefits from capacity; DirectUNet does not. (Note: Q2's
  "best overall" ranking is superseded for S0 by L3 ens30×10 = 0.5643.)
- **Q3 (answered, L6)**: corrupted-forcing conditioning is neutral-to-slightly-negative
  (L6 0.639/0.638 vs obs-only L2b 0.633/0.633); neural degradation was already ≈1.00,
  so there was no robustness gap for conditioning to close.

### L3 ensemble study (`ens30`, S0 + S1, job arrays 49350 / 49447)

N=30 members to match DA EnKF/ETKF `N_ensemble=30`; outputs in
`experiments/{L3,L2b}_vanilla_cfm_s0s1/ens30_no{1,10}/` (`estimates_s0.npz` member mean,
`members_s0.npz` (200,3000,24,30) f32, `neural_eval.json` with a `sampling` block).
ES note: this S0 study ran while the DA `_ESAccumulator` still had its normalization bug,
so the S0 JSONs store both conventions ("cache" = legacy buggy `mae/M − 0.5·pairwise`,
"textbook" = proper scoring rule); after the 2026-08-24 fix there is a single ES
(the textbook formula) everywhere — the table below keeps both columns as record.
The S1 ens30 study (`experiments/L3_vanilla_cfm_s0s1/ens30_s1_no{1,10}/`, job 49447,
2026-08-24) ran against the bug-fixed code, so its JSONs store the single textbook
ensemble ES (matched by the DA EnKF/ETKF N=30 caches).

| Model | members × steps | RMSE | EV | ESens(cache)* | ESens(textbook) | spread |
|---|---|---|---|---|---|---|
| L3 multi-τ | 30 × 1 | 0.6503 | 0.845 | −0.094 | 0.336 | 0.194 |
| L3 multi-τ | 30 × 10 | **0.5643** | 0.879 | −0.140 | 0.265 | 0.278 |
| L2b τ=0 | 30 × 1 | 0.6290 | 0.854 | −0.021 | 0.371 | 0.062 |
| L2b τ=0 | 30 × 10 | 0.6290 (≡ no1 bitwise) | 0.854 | −0.021 | 0.371 | 0.062 |

(*legacy buggy convention, kept for provenance only.)

Reference points (single-sample): L3 0.688, L2b 0.633, L4 DirectUNet 0.6189;
best DA Strong-4DVar 0.742 (S1 1.432). Multi-τ spread (~0.28 at 10 steps) is ~4.5×
the τ=0 spread — the τ-sampled velocity field yields genuinely diverse members whose
mean beats every deterministic scheme; whether that diversity helps probabilistic
scores (CRPS vs the ES conventions here) is open follow-up work. The S1 ens30 study
is complete (2026-08-24, see below); other models' ensemble runs remain open follow-up.

### 5-seed reproducibility (S0, job array 49419)

Five independent 30-member ensembles (seeds 1–5) confirm the multi-τ advantage is
not a seed artifact. Report: `reports/l96/outputs/ens30_seed_report.md`.

| scheme | seeds 1-5 mean±std | seed0 (orig) | range (6 runs) |
|---|---|---|---|
| 1-step (n_outer=1) | 0.6502 ± 0.0002 | 0.6503 | [0.6500, 0.6506] |
| 10-step (n_outer=10) | 0.5642 ± 0.0005 | 0.5643 | [0.5637, 0.5650] |

10-step/1-step ratio = 0.868 (−13.2%), cross-seed std < 0.001 for both schemes.
At inference τ is a deterministic schedule (k/N_outer), not random; all member
diversity comes from fresh x₀ noise. The improvement comes from proper ODE
integration of the multi-τ-trained field across τ∈(0,1], not from τ=0 evaluations
(the 1-step result at 0.650 is worse than the τ=0-trained L2b control at 0.629).

### L3 ens30 on S1 (job array 49447, 2026-08-24)

S1 counterpart of the S0 ens30 study, run against the bug-fixed single-ES code
(`--cases s1 --n-members 30 --seed 0`, n_outer ∈ {1,10}). Outputs in
`experiments/L3_vanilla_cfm_s0s1/ens30_s1_no{1,10}/` (`members_s1.npz` (200,3000,24,30) f32,
`estimates_s1.npz`, `neural_eval.json` with a single textbook `ensemble.es`).

| Model | members × steps | RMSE | EV | ES (ens, N=30) | spread |
|---|---|---|---|---|---|
| L3 multi-τ | 30 × 1 | 0.6528 | 0.843 | 0.338 | 0.194 |
| L3 multi-τ | 30 × 10 | **0.5667** | 0.877 | 0.267 | 0.278 |

The 10-step/1-step ratio is 0.868 (−13.2%) — the integration-coarseness effect is
statistically identical to S0. S1/S0 degradation at 30×10 is ≈ **1.004** (S1 0.5667 vs
S0 0.5643): the multi-τ ensemble is essentially as good on S1 as on S0, consistent
with the neural models' known robustness to the parameter-biased S1 test setup.

## Deferred future work (Phases B & C)

These were proposed alongside the S1 ens30 study (Phase A, done 2026-08-24) but are
**not committed for execution** — recorded so a future session can pick them up.

### Phase B — CFM architecture variants (low priority; **requires a design doc first**)

Investigate whether a Tweedie-style two-stage decomposition or a diffusion-style
variant improves on VanillaCFM for L96. **Design doc drafted 2026-08-25**
(`docs/phase_B_l96_cfm_variants.md`) — defines V1 (L96 TweedieSolver port,
obs-only, `use_energy=false`, 2-stage) and V2 (CFM + Tweedie residual hybrid)
precisely with reference bars vs L2b/L3/L4 and the open design questions to
resolve (cond_extra_dim plumbing, energy flag, stage-1 budget, 24D cell
validity, multi-member sampling, single-sample vs ens30×10 bar). **V3 diffusion
  deferred** (no scaffolding exists). **No Phase B code implements these yet** — a
  future session must resolve the doc's open questions before implementing.

**V2/V3 worktree state (checked 2026-08-28) — for the Phase B session:**
The V2/V3 topic worktree (`4dvarnet-fm-cfm-v2v3`, branch `feature/l96-v2v3-pure`)
is separate from master and **not yet trainable**:
- Its `train.py` still has **unresolved git merge-conflict markers** (B1 blocker;
  `py_compile` fails with `SyntaxError: unmatched ')'`) from commit `f7749a9`,
  so no training/eval can run from that branch as-is.
- It does **not** carry the vectorized data-gen fix merged to master as PR #105
  (`data/lorenz96.py` there has no `fast_generation` path). When the Phase B
  session resumes it should re-sync master first (which includes both #105 and the
  joint-DA work).
- The **parallel-merge CHANGELOG conflict** on master (joint-DA 2026-08-28 entry vs
  the V2/V3-sourced #105 2026-08-27 data-gen entry) is already **resolved cleanly**
  on `origin/master` — no markers remain. The V2/V3 tree did NOT contribute the
  fix to master (it merged independently); only its own branch needs the sync.


### Phase C — L96 joint state-parameter neural estimation (infra done 2026-08-25; training + eval done 2026-08-26)

Extend the existing **L63 joint infrastructure** to L96. Currently only L96 **Joint DA
baselines** exist; the L96 joint **neural** models were missing.
- **Existing (real) pieces:** `JointCFM` (`models/vanilla_cfm.py:75`, L63-shaped,
  `output_dim = state_dim + param_dim`); `JointCFMConfig` (`conf/schema.py:162`);
  L63 configs `H1_joint_cfm_default.yaml`, `H2_joint_cfm_tau0.yaml`, `S5/S6`; L96
  Joint DA baselines `JointEnKFL96`/`JointETKFL96`/`JointStrong4DVarL96`
  (`evaluation/baselines.py`) + `eval_joint_comparison_l96.py` (ready DA comparator).
- **Done (2026-08-25):** design doc `docs/phase_C_l96_joint_neural.md`; L96 joint
  neural models `JointCFM` (port) + `JointDirectUNet` (new); 3 configs L7/L8/L9;
  `data/lorenz96.py` flattens `fast_weights` to per-index `w1..w4`/`true_w1..`/`_da`
  scalar keys; `train.py`/`lightning_module.py` dispatch; `eval_joint_neural_l96.py`
  (extended `evaluation/neural_inference.py` for joint types); 8 joint-neural tests
  + WP1 dataset-key tests; 2 sbatch (training 3-task array, eval). `param_dim=8`,
  h fixed. Training completed; standalone eval + ens30 completed (2026-08-26, below).
- **Eval bugs fixed (2026-08-26):** PR #81 (`state_dim` inference, `obs_var_indices`,
  member-param collection), #83 (ens30 sbatch IFS, ensemble evaluator, `N_outer`
  passthrough), #85 (`collate_joint_eval` legacy-`fast_weights`-list support for the
  pre-flattening cached dataset), #89 (ens30 `params_pred` member-mean shape fix),
  #90 (report-generator table column-order/separator/best-marking fixes).
- **Results (2026-08-26, cached S0/S1 test set, Obs30, 200 windows):** see the L96
  joint neural benchmark at `reports/l96/outputs/l96_joint_neural_benchmark.md`.
  Single-sample: L7 0.606/0.662, L8 0.610/0.661, L9 0.626/0.631 (S0/S1 state RMSE).
  L9 ens30×10 is the best joint estimator (S0 **0.525**/S1 **0.531**), matching L3's
  multi-τ integration advantage. **L9 recovers the 8 params (paramRMSE 0.058) while L7
  τ=0 fails (1.21) despite matching state RMSE**; L8 deterministic recovers them well
  (0.061). **Joint ETKF DA baseline run (2026-08-28, Job 50577)** — see
  `l96_joint_da_benchmark.md`: S0 Joint-ETKF 0.633/param 0.053 (≈ L9 parity); S1 1.497/
  param 0.128 after the inflation/stability fix (see "L96 joint state-parameter DA
  baseline" under Experiments). **Joint-EnKF added 2026-08-28 (job 50655)** — the
  state-only-inflation fix was ported to `JointEnKFL96` (RC: it inflated the whole
  augmented state like the old ETKF, risking the same S1 divergence) and a joint
  `assimilate_batch` was added (the inherited parent batch silently dropped params);
  S0 0.726/param 0.057, S1 **1.459**/param 0.148 (S1 is the best DA row, stable after
  the fix). **Joint-Strong-4DVar run (2026-08-31, Job 51000)** — batched pure-gradient
  Adam solve (no LBFGS) over all 200 windows: S0 0.712/param 0.226, S1 **1.200**/param
  0.299. Joint-Strong-4DVar is now the **best DA row on S1** (beats Joint-EnKF 1.459 /
  Joint-ETKF 1.497 / vanilla Strong-4DVar 1.432) and close to Joint-ETKF on S0; its
  param recovery (esp. F 0.85) is weaker than the filters, but its S1 state skill is
  the joint-DA best.

### Phase C-adjacent (blocked, unblocks DA-parity ES): L96 DA cache ES regeneration

The original esfix array (job 49383 + resubmissions) failed before regenerating the
non-canonical baseline caches. A 2026-08-24 resubmission (job 49488, `--array=1-7`,
against fixed master) **still cannot complete**: it *resumed from the stale partial
`*_esfix*` caches* written by 49383 rather than doing a clean regeneration, and the
validation gate then **failed on 6 of 7 caches** (RMSE mismatch vs originals: e.g. dws50
EnKF S0 1.009→1.102, Strong-4DVar and S1 EnKF/ETKF drifting). Only the **legacy int100**
cache passes cleanly and was swapped (`.bak` + promoted esfix). Rather than force-swap
gate-failing caches (would corrupt RMSE consistency), the rest stay stale.

**Important scoping note:** the consolidated report's DA candidates
(`DA_JSON_CANDIDATES` in `reports/l96/generate_l96_consolidated_report.py`) point at the
**canonical s0c int100 fw** cache first, which was already swapped bug-fixed and is
correct (ETKF/EnKF proper textbook ES) — so **the report is already correct on the DA
side** and none of the 6 non-report caches affect it. The 6 remaining caches (s0c int200
fw, fw6 int100/int200, legacy int200, dws50, dws50 fw) serve other lineages and are
**not** referenced by the consolidated report. To finish them correctly, delete the stale
`*_esfix*` files for those caches and do a clean full regeneration (hours GPU each) —
outcome uncertain given the CHANGELOG's note that some L96 caches "are not reproducible
under current code semantics."

### Decoupled cascade (C1/C2) — documented negative (2026-09-01)

A **decoupled state→param cascade** (`model_type=param_head`, `models/param_head.py`) was
tested as an alternative to the coupled joint flow for recovering the 8 L96 params: a
`StateParamHead` reads obs + biased `*_da` params + forcing + a state estimate, trained
under two state sources — **C1** = frozen L1b state-only DirectUNet estimate, **C2** =
exact true state (ablation). **Both fail the fast weights `w1/w2` on S1 (NRMSE ≈ 1.1-1.2,**
i.e. relative error > 100%), even with the exact true state (C2) — an
**information/architecture bottleneck**, not a state-quality issue (F is partly state-quality:
true state halves it 1.67→0.86). Only the coupled **multi-τ flow (L9)** recovers all 8 params
(S1 per-param NRMSE all ≤ 0.20, F 0.07) at parity with the joint DA filters on the params they
actually estimate. **Recorded as a documented negative, not a benchmark win.** Details + NRMSE
table: CHANGELOG 2026-09-01 and `reports/l96/outputs/l96_joint_neural_benchmark.md` (which now
also carries computed DA NRMSE rows with a `w3/w4`=pinned-prior masking footnote).



All E/F/G/S rows are **Lorenz-63** (`cs1+cs2` mixes); results live under `experiments/`.
The **L-series** (Lorenz-96) is listed separately below.

### Lorenz-63

| ID | Model | Hidden | Epochs | Train mix | Status |
|---|---|---|---|---|---|
| E1_direct_unet_default | DirectUNet | [64,128,256] | 200 | cs1+cs2 | done |
| E2_direct_unet_small | DirectUNet | [32,64,128] | 200 | cs1+cs2 | done |
| E3_direct_unet_rand | DirectUNet | [32,64,128] | 200 | cs1_rand+cs2_rand | done |
| F1_vanilla_cfm_default | VanillaCFM | [64,128,256] | 400 | cs1+cs2 | done |
| F2_vanilla_cfm_small | VanillaCFM | [32,64,128] | 400 | cs1+cs2 | done |
| F3_vanilla_cfm_rand | VanillaCFM | [32,64,128] | 400 | cs1_rand+cs2_rand | done |
| G1_vanilla_cfm_t0_default | VanillaCFM (τ=0) | [64,128,256] | 400 | cs1+cs2 | done |
| G2_vanilla_cfm_t0_small | VanillaCFM (τ=0) | [32,64,128] | 400 | cs1+cs2 | done |
| G3_vanilla_cfm_t0_rand | VanillaCFM (τ=0) | [32,64,128] | 400 | cs1_rand+cs2_rand | done |
| S1–S10 (incl. τ=0 + joint-CFM variants) | various | various | — | s0_s1 | done |

### Lorenz-96 (DA-parity: all-5 params ±20%, obs_j=2 → 24D, Obs30)

| ID | Model | Hidden | Epochs | τ mode | Status |
|---|---|---|---|---|---|
| L1b_direct_unet_s0s1 | DirectUNet | [64,128,256] | 200 | n/a | done (beats DA on S0+S1) |
| L2b_vanilla_cfm_s0s1 | VanillaCFM | [64,128,256] | 400 | τ=0 | done (≈ L1b) |
| L3_vanilla_cfm_s0s1 | VanillaCFM | [64,128,256] | 400 | multi-τ | done (ens30×10 best on S0 0.564 + S1 0.567, job 49447) |
| L4_direct_unet_s0s1_small | DirectUNet | [32,64,128] | 200 | n/a | done (Q2: best overall, 0.619/0.621) |
| L5_vanilla_cfm_s0s1_small_tau0 | VanillaCFM | [32,64,128] | 400 | τ=0 | done (Q2: small hurts CFM, +4.3%) |
| L6_vanilla_cfm_s0s1_forcing_cond | VanillaCFM | [64,128,256] | 400 | τ=0 + forcing cond | done (Q3: neutral vs obs-only) |
| L7_joint_cfm_s0s1 | JointCFM | [64,128,256] | 400 | τ=0, joint 8-param | done (state 0.606/0.662; paramRMSE 1.21 — τ=0 fails params) |
| L8_joint_direct_unet_s0s1 | JointDirectUNet | [64,128,256] | 200 | joint 8-param | done (state 0.610/0.661; paramRMSE 0.061) |
| L9_joint_cfm_s0s1_multitau | JointCFM | [64,128,256] | 400 | multi-τ, joint 8-param | done (ens30×10 best joint: state 0.525/0.531; paramRMSE 0.058) |

**Standalone DA-parity results (cached test set, Obs30, 200 windows)** — S0/S1 RMSE
(single-sample convention; L3's S0 ranking is superseded by the ens30 study above):
L4 **0.619**/0.621 < L1b 0.622/0.625 < L2b 0.633/0.633 ≈ L6 0.639/0.638 < L5 0.660/0.660 < L3 0.688/0.690.
All neural degradation ≈ 1.00; best DA (Strong-4DVar): 0.742/1.432.

**L96 joint state-parameter estimation (Phase C, 2026-08-26)** — cached S0/S1, Obs30,
200 windows; 24D state + 8 params (F, c1, hx, eps, w1..w4). Single-sample S0/S1 state
RMSE: L7 0.606/0.662, L8 0.610/0.661, L9 0.626/0.631. Best joint = **L9 multi-τ at
ens30×10: S0 0.525 / S1 0.531** (matches L3's multi-τ integration advantage). Param
recovery is model-dependent: L9 recovers the 8 params (paramRMSE 0.058) as does L8
JointDirectUNet (0.061), but **L7 τ=0 fails to recover params (1.21)** despite matching
state RMSE. Full tables: `reports/l96/outputs/l96_joint_neural_benchmark.md`.

**L96 joint state-parameter DA baseline (2026-08-28, ETKF + EnKF)** — see
`reports/l96/outputs/l96_joint_da_benchmark.md`. Joint-ETKF vs vanilla ETKF, and
Joint-EnKF vs vanilla EnKF, on the cached S0/S1 set (200 windows). **S0** Joint-ETKF
state RMSE **0.633** (EV 0.82, ES 0.30) vs vanilla 0.878; paramRMSE mean **0.053** —
**at parity with the L9 neural model** (state 0.626 / param 0.059 single-sample).
Joint-EnKF S0 0.726 (EV 0.77, ES 0.37) beats vanilla EnKF 0.891 but is worse than
Joint-ETKF. **S1** Joint-EnKF **1.459** (EV 0.23, ES 0.84) is the best DA row, ahead of
Joint-ETKF 1.497 & vanilla EnKF 1.505 / ETKF 1.554 — both joint filters stabilized by
the state-only-inflation fix (RC: the filters were inflating the unobserved param
block, growing spread into the reduced J=2 forecast). Neural (L9) still clearly ahead
on S1 (0.631) via its ≈1.00 bias robustness. **Joint-Strong-4DVar run 2026-08-31**
(job 51000, batched Adam over 200 windows): S0 state RMSE **0.712** / S1 **1.200** —
beats vanilla Strong-4DVar (0.750/1.432) on both cases and is the **best DA row on S1**
(ahead of Joint-EnKF 1.459 / Joint-ETKF 1.497); param RMSE mean 0.226 (S0) / 0.299
(S1), weaker than the filters (F 0.85 dominates).

**Multi-method reconstruction artifacts (merged 2026-09-01, PR #134)** —
`eval_joint_comparison_l96.py` persists per-window reconstruction `.npz` arrays
(`trajectories`, per-member `ensemble_variance`, `params`, `es`) for every benchmarked
method, merged into `experiments/l96_joint_baselines_trajectories.npz`. Re-ran the 3
joint methods (jobs 51098/51131) + vanilla Strong-4DVar (51294, so it appears in the
comparator schema) on the canonical cached S0/S1 test set. `experiments/l96_joint_comparison.json`
is now the full **6-method** comparison (vanilla ETKF/EnKF/Strong-4DVar + Joint-ETKF/EnKF/
Strong-4DVar; fresh re-run values: Joint-ETKF S0 0.6348, Joint-EnKF 0.7244, Joint-Strong-4DVar
0.7054/1.1999). Both joint reports regenerate against it, so the neural report's DA-baselines
table now lists all 6 DA methods (incl. Joint-Strong-4DVar). Note the master committed DA
report retains oracle-free neural values; a stale local regeneration against oracle-era JSONs
on the master worktree was discarded in favor of the merged oracle-free content.

### S1 DA corrupted-forcing fix (2026-09-02) + slow-only obsj0 DA baselines

**S1 forcing bug fixed:** `cfg_s1` in both DA eval paths (`run_and_cache_baselines` and
`eval_joint_comparison_l96.py`) was built without `case=2`, so `Lorenz96Config.use_corrupted_forcing`
returned False and `evaluate_baseline` fed the DA the **true** forcing (`forcing_true`) on S1 —
the `forcing_state_bias=0.1` corruption designed into S1 was silently never applied to the DA
(the cached S1 windows *do* hold a genuine `forcing_corrupted`; it was just never read). Fixed by
setting `case=2` in `cfg_s1` (S0 unchanged). Re-ran the canonical obsj2 S1 DA (state-only + joint)
with the fix; **S1 changes only mildly** (filters <1%: Joint-ETKF 1.4976→1.5125, EnKF 1.5044→1.5123,
Strong-4DVar 1.4319→1.4369) — the DA is robust to the forcing corruption. **S0 reproduced within
noise** (S0 gate <2%), confirming the fix did not disturb S0. Canonical caches swapped (`.bak`
backups: `l96_baselines_dws500_s0c_*_obsj2_int100_fw.json*` + `l96_joint_comparison.json`); the
consolidated / joint-DA / joint-neural reports regenerated with the corrected S1 rows.

**New slow-only obsj0 configuration (2026-09-02):** decoupled the observation count from the S1
reduced-dynamics J and the eval metric group. Now `run_and_cache_baselines` / the comparator take
`obs_j` (fast vars observed; 0 = slow-only), `s1_j` (S1 reduced dynamics J, kept=2 for the obsj0
study), and `eval_j` (eval metric group, kept=2 → the same 24D slow+first-2-fast group) — all three
independent. This lets a **slow-only** observation (only the 8 slow X) be scored on the identical
24D eval subspace as the canonical obsj2 config, apples-to-apples. Ran state-only + joint DA
baselines (obsj0, Obs30, 200 windows): S0 EnKF 1.27 / ETKF 1.25 / Strong-4DVar 1.44 / Joint-ETKF
1.19; S1 EnKF 1.70 / ETKF 1.71 / Strong-4DVar 1.62 / Joint-ETKF 1.60. vs obsj2 the degradation is
dominated by the **unobserved** obs_fast group (S0 obs_fast ≈ 1.6–1.95 vs 0.88–1.10 obsj2) while the
**slow subgroup is preserved** (S0 slow ≈ 0.41–0.46). Joint-DA param recovery: S1 Joint-ETKF
0.130→0.158 (hx/F degrade), S0 slightly improves (0.045 vs 0.054, F-driven). Full tables:
`reports/l96/outputs/l96_da_baselines_obsj2_vs_obsj0.md`.

### Fast-Y observation-density generalization study (2026-09-09, `feature/l96-obs-density-generalization`)

Inference-time-only generalization test (no retraining) for the 4 best-of-subcategory
monai-backbone schemes from the consolidated benchmark (`l96_consolidated_benchmark.md`) --
**DirectUNet-L(monai,cos)**, **CFM-M(monai,flat)**, **SDA3(monai)**, **DirectUNet+SDA3** -- on
the same 200-window S0/S1 cached test set, with the fast-Y observation density randomly
reduced: only `keep_k` of the 16 canonical fast-Y channels (2 per slow node) are kept,
**redrawn independently at every observation time** within a window (the harder OOD test,
chosen over a simpler fixed-per-window mask); the 8 slow-X channels always stay fully
observed. `keep_k ∈ {16 (sanity check), 8, 4, 0}`.

Distinct from the pre-existing `l96_da_baselines_obsj2_vs_obsj0.md` above: that study is DA
baselines only, a single fixed slow-only (obsj0) config; this one is the 4 best neural/SDA
schemes, a per-obs-time randomly redrawn density sweep.

- **New utility `evaluation/obs_density.py`**: `fast_channel_keep_mask` draws an exact-count
  random keep-mask per (window, obs-time) via per-slice `topk` scores (not a Bernoulli
  approximation); `apply_density_mask_to_obs` NaNs out the dropped channels directly in `obs`.
- **Architecture-driven design split** (verified against the code, not assumed): DirectUNet/
  VanillaCFM/FourDVarNet/Joint* consume `obs` only via `torch.nan_to_num(obs, nan=0.0)`, no
  separate mask channel -- trained only on whole-timestep NaN blocks, never partial-channel NaN
  within an observed timestep, so a dropped fast-Y channel is genuinely indistinguishable from a
  real near-zero observation for them (flagged as an explicit, unfixable-without-retraining
  caveat in the report). SDA's prior network never conditions on raw obs at all (only the
  guided-sampling cost, `evaluation/sda_sampler.py::guided_obs_cost`, reads it) -- architecturally
  clean, no zero-imputation ambiguity there.
- **`evaluation/sda_sampler.py`**: `guided_obs_cost`/`sda_guided_sample` generalized with a new
  `obs_channel_mask` param (boolean, may vary per timestep/batch element -- unlike the
  pre-existing `obs_indices`, which applies one fixed subset across the whole trajectory;
  mutually exclusive with it), combined multiplicatively with the existing temporal `obs_mask`.
- **`evaluation/neural_inference.py`**: `_run_case_inference`/`run_inference` gained
  `obs_density_keep_k` (mutually exclusive with `obs_indices`), applied uniformly across every
  model type dispatched there -- NaNs `batch["obs"]` for direct-obs-consuming models, forwards
  `obs_channel_mask` to `sda_guided_sample` for the SDA priors. `None` (default) is a true no-op.
- **New `eval_obs_density_l96.py`**: orchestrates all 4 methods x {S0,S1} x keep_k x
  `--n-repeats` (independent seed reruns; default 3 -- each run already averages over ~30
  obs-times x 200 windows of independent random draws, so a small repeat count suffices to
  sanity-check aggregate RNG sensitivity) x reusing each scheme's existing checkpoint at its
  canonical benchmark hyperparameters (n_members/n_outer/guidance_weight/tau0 all match
  `l96_consolidated_benchmark.md` exactly, so `keep_k=16` is a reproduction sanity check, not
  just a nominal baseline). The hybrid's DirectUNet-M warm start gets the density-masked obs
  directly (same ambiguity as plain DirectUNet); its SDA3 guidance stage gets `obs_channel_mask`
  (clean).
- **New `reports/l96/generate_l96_obs_density_generalization_report.py`**: RMSE/EV(all_obs)
  table (rows=keep_k, columns=method x case, mean±std across repeats) + a degradation table
  (RMSE relative to the keep_k=16 baseline per method/case).
- **New `batch/run_l96_obs_density_generalization.sbatch`** launches the sweep + report on the
  cluster (checkpoints/cached dataset live only in `experiments/` on the HPC filesystem, not in
  this git worktree).
- Tests: `tests/test_obs_density.py` (mask exact-count/shape/no-op invariants),
  `tests/test_sda_sampler.py` (obs_channel_mask restricts/varies-per-timestep/mutual-exclusion/
  zero-guidance-equivalence), `tests/test_neural_inference.py` (obs_density_keep_k NaNs the
  right channels for direct-obs models, no-ops at keep_k=16, forwards correctly to the SDA path).

**Sweep results (job 52845, run 2026-09-09 via an ad-hoc launcher pointing at the
`4dvarnet-fm-l96-eval-config-persist` worktree's checkpoints, `--n-repeats 3`):** `keep_k=16`
reproduces the canonical benchmark exactly for all 3 completed methods (DirectUNet-L 0.4863/0.4865,
CFM-M 0.4809/0.4783, SDA3 0.5369/0.5363, S0/S1). DirectUNet-L and CFM-M degrade steeply and near-
identically: **~1.9x RMSE at keep_k=8, ~2.3x at keep_k=4, ~2.7x at keep_k=0** -- confirming the
architecture-driven OOD caveat above is real, not just theoretical. SDA3 degrades far more
gracefully (**~1.45x / ~1.72x / ~2.0x** at the same keep_k values) and at `keep_k=0` its RMSE
(1.08) is actually *better* than DirectUNet/CFM's (1.29/1.32) despite starting from a worse
full-density baseline -- confirming the "obs excluded cleanly from the guidance cost" design is
architecturally robust with zero retraining, exactly as predicted. DirectUNet+SDA3 hybrid
(warm-started) inherits much of SDA3's graceful-degradation benefit (e.g. S0 keep_k=8: 0.803 vs.
plain DirectUNet's 0.906) while keeping its better full-density baseline (0.420 vs. SDA3's 0.537).
Repeat-to-repeat std stays tiny throughout (≤0.006 RMSE), confirming `n_repeats=3` was sufficient.
**This result is the direct motivation for the follow-on training-augmentation work below**; its
full tables were superseded by (and are reproduced inside) the consolidated summary table in
`reports/l96/outputs/l96_obs_density_augmented_training.md` (2026-09-10 cleanup: this report's
own standalone output, `l96_obs_density_generalization.md`, was retired as a redundant subset --
see that report's §2 for the equivalent non-augmented rows).

### Fast-Y observation-density-augmented TRAINING (2026-09-09, `feature/l96-obs-density-training`)

Follow-on to the generalization study above: instead of only measuring the OOD failure at eval
time, train DirectUNet-L(cos)/CFM-M(flat) with the same fast-Y density reduction applied as a
TRAINING-time augmentation, so a dropped channel stops being an untrained-for input pattern.
SDA3/the hybrid are excluded from this work -- the sweep above already showed they need no
retraining (architecturally robust via the guidance-cost exclusion, not the obs-consuming path).

- **Relocated `evaluation/obs_density.py` -> `data/obs_density.py`** (all references updated: 
  `evaluation/neural_inference.py`, `evaluation/sda_sampler.py` docstring, `eval_obs_density_l96.py`,
  `tests/test_obs_density.py`, `tests/test_neural_inference.py`) -- the masking primitives are
  consumed by both `evaluation/` (eval-time) and now `data/`/`train.py` (train-time), so `data/`
  is the correct home; importing "up" from a training consumer into `evaluation/` would have been
  backwards.
- **New `data/obs_density.py::random_variable_keep_mask`**: generalizes the eval-time
  `random_keep_mask`'s single scalar `keep_k` to a per-(window,timestep)-varying tensor, via
  per-slice random-score ranks (topk only supports one `k` for a whole tensor) -- needed because
  training mixes full-density and randomly-reduced-density events within the same batch, unlike
  eval's one-keep_k-per-sweep-cell design.
- **New `data/obs_density.py::sample_training_density_mask`**: independently at every
  (window, obs-time), with probability `full_prob` (default 0.4) keeps full density (so the model
  doesn't lose sharpness on the still-common canonical case); otherwise draws `keep_k` uniformly
  from `{min_keep,...,15}` (default `min_keep=0`) -- broad coverage of the whole degradation
  spectrum, not just the eval sweep's 4 discrete points, so the model learns a smooth
  interpolation. No existing precedent in this codebase for a "sometimes augment" mixing scheme
  (checked: only prior precedent, `noisy_da_bias`, always randomizes, never skips) -- this
  mixing probability is the one genuinely new design choice here.
- **`data/dataloader.py::make_collate_fm`** gained an `obs_density_cfg` param (dict with
  `full_prob`/`min_keep`, or `None` -- true no-op, the default): draws a fresh mask every batch
  and NaNs the dropped fast-Y channels of `obs` before normalization. Requires the canonical
  24D (8 slow + 16 fast) obsj2 subspace, raises otherwise.
- **`train.py::make_l96_dataloaders`** now builds train and val with *different* collate fns --
  `obs_density_cfg` only ever applies to `"train"`; `"val"` always stays at full canonical density
  so its loss/metrics remain comparable across epochs and against the eval protocol. Wired from
  new `DataConfig` fields `obs_density_augment: bool = False` / `obs_density_full_prob: float = 0.4`
  / `obs_density_min_keep: int = 0` (`conf/schema.py`) -- default `False` leaves every existing
  config byte-for-byte unaffected (verified: `make_collate_fm(norm_stats, obs_density_cfg=None)`
  reproduces plain `collate_fm`/the pre-existing normalize-only path exactly).
- **New experiment configs** `L1b_monai_unet_s0s1_norm_l_cosine_obsdensity.yaml` /
  `L2b_monai_vanilla_cfm_s0s1_norm_obsdensity.yaml`: identical architecture/hyperparameters to
  the current best-of-subcategory checkpoints, changing only `data.obs_density_augment=true` --
  isolates the augmentation's effect cleanly. Not yet trained at full scale (200/400 epochs) --
  1-epoch smoke tests on both passed (real checkpoints, tiny window counts, confirmed the
  augmented collate path runs end-to-end with no crashes); full training is a follow-up once
  this PR merges, then re-run through `eval_obs_density_l96.py` (unchanged) for a direct
  before/after comparison against today's baseline numbers.
- Tests: `tests/test_obs_density.py` (new mask functions' exact-count/shape/mixture/range-
  validation invariants), `tests/test_l96_normalization.py` (`make_collate_fm`'s
  `obs_density_cfg` -- NaN pattern, full_prob=1.0 no-op, wrong-dim raises, composes with
  normalization), `tests/test_joint_estimation_l96_neural.py`
  (`make_l96_dataloaders` augments train only, val stays clean).

### Fast-Y observation-density-augmented training -- results (2026-09-10)

Follow-up to the plan above, run interactively (not yet a separate PR-tracked branch at the
time of writing -- code lives on `feature/l96-obs-density-augmented-report`).

- **L-tier collapsed, M-tier fixed it.** DirectUNet-**L**(cosine, augmented) converged on
  `val_loss` but collapsed toward the fast-Y conditional mean at eval time (variance ratio
  ~35%, correlation ~0.55-0.64) -- root-caused via a side-by-side diagnostic against CFM-M's
  *identical* augmentation pipeline (variance ratio ~96%, correlation ~0.94-0.95 there),
  ruling out a data/eval bug and pointing at an L-tier-specific pathology. Retrained
  DirectUNet-**M** with cosine (`L1b_monai_unet_s0s1_norm_obsdensity.yaml`, new): variance
  ratio 99%, correlation 0.94 -- fully healthy, and RMSE (0.473/0.476) improved on the
  non-augmented M-tier baseline (0.501-0.507). Cosine annealing adopted as the default for
  all obs-density-augmented training going forward, not just L.
- **Reduced-density payoff confirmed.** `eval_obs_density_l96.py` against both healthy
  augmented checkpoints: degradation ratio at `keep_k=8` dropped from ~1.9x to **1.49x** for
  both CFM-M and DirectUNet-M, without sacrificing (in fact slightly improving) the
  full-density baseline.
- **Best scheme of the whole investigation:** SDA3 guidance warm-started from the *augmented*
  DirectUNet-M mean estimate (instead of the original non-augmented one) -- full-density RMSE
  **0.389** (best of every scheme tested this session, non-augmented or augmented) and the
  best absolute worst-case RMSE at `keep_k=0` (**1.065**, ahead of even SDA3's own 1.082).
  SDA3 alone still has the flattest *relative* degradation curve (2.02x vs the hybrid's
  2.74x) -- the hybrid's much lower starting point means even a larger relative drop still
  lands ahead in absolute terms, not a contradiction, just two different ways to read
  "robustness."
- **New dedicated report** `reports/l96/generate_l96_obs_density_augmented_report.py` ->
  `l96_obs_density_augmented_training.md`: experiment description, a combined summary table
  across all 7 method variants (2 non-augmented baselines, 2 augmented alone, SDA3, the
  non-augmented hybrid, the augmented hybrid), and a best/median/worst-window analysis
  (mirroring `generate_l96_consolidated_report.py`'s `select_windows` convention, but with
  `keep_k` as the varying axis for the best method instead of comparing methods at fixed
  density) with both a per-keep_k RMSE table and a Hovmöller-style figure per rank.
- **Consolidated benchmark updated**: added `DirectUNet-M(monai,cos,obsdensity)` /
  `CFM-M(monai,flat,obsdensity)` rows to every table in `l96_consolidated_benchmark.md`
  (scheme description, RMSE/EV/ES pooled, RMSE/EV/CRPS per-window) computed directly from
  their `estimates_{s0,s1}.npz` via `evaluation/estimate_metrics.py` -- not a full script
  regeneration, which needs DA-baseline/joint-comparison cache files living only in a
  different worktree; both rows explicitly noted as N=1 single-pass evaluations (marked `*`
  in ES/CRPS), unlike the ensemble (N=30) convention the original CFM-M rows use.

## Phases

### Phase 0: Plan
- [x] Initial PLAN.md created
- [x] CS3/CS4 experiment plan in `docs/case_studies.tex`
- [x] Exp G (τ=0) experiment plan in `docs/experiment_G_tau0_cfm.md`

### Phase 1: Implementation (complete)
- [x] `models/direct_unet.py` — DirectUNet nn.Module
- [x] `models/vanilla_cfm.py` — VanillaCFM nn.Module with CFM loss + sampling
- [x] `data/random_param_dataset.py` — Randomized Lorenz-63 parameters per window
- [x] `conf/schema.py` — `DirectUNetConfig`, `VanillaCFMConfig`, `DataConfig` with CS3/CS4 fields
- [x] `training/lightning_module.py` — `LitModel` dispatches all 3 model types
- [x] `training/pipeline.py` — `create_trainer`, `train_stage`, `run_2stage_pipeline`
- [x] `train.py` — `model_factory`, `evaluate_model`, CS3/CS4 evaluation
- [x] 6 experiment YAML configs (E1-E3, F1-F3)
- [x] CS3/CS4 test cases in data generation, evaluation, and report
- [x] `evaluate_all.py` — Unified baseline + CFM comparison script
- [x] `reports/generate_unet_cfm_report.py` — CS3/CS4 report

### Phase 2: sbatch Infrastructure (this session)
- [x] `batch/run_lint.sbatch` — ruff + mypy in batch
- [x] `batch/run_test_suite.sbatch` — pytest (fast) in batch
- [x] `batch/run_config_validation.sbatch` — Hydra config + model factory validation
- [x] Deprecated duplicate `run_vanilla_experiments.sbatch` and interactive `run_tests.sh`

### Phase 3: τ=0 CFM Ablation (Exp G) — complete (2026-07-01)
- [x] `conf/schema.py` — `train_tau_0_only: bool = False` on `VanillaCFMConfig`
- [x] `models/vanilla_cfm.py` — τ=0 logic in `compute_cfm_loss` and `sample`
- [x] `train.py` — `train_tau_0_only` wired through `model_factory`
- [x] 3 config YAMLs: G1_vanilla_cfm_t0_default, G2_vanilla_cfm_t0_small, G3_vanilla_cfm_t0_rand
- [x] `batch/run_one_epoch_tests.sbatch` + `batch/run_new_experiments.sbatch` updated with G1-G3
- [x] Tests for τ=0 mode

### Phase 4: Verify (all via sbatch) — complete (2026-07-01)
- [x] `sbatch batch/run_config_validation.sbatch` — all configs load
- [x] `sbatch batch/run_lint.sbatch` — ruff + mypy pass
- [x] `sbatch batch/run_test_suite.sbatch` — fast tests pass
- [x] `sbatch batch/run_one_epoch_tests.sbatch` — GPU smoke test (E1-F3 + G1-G3, 1 epoch)

### Phase 5: Launch — complete (2026-07)
- [x] `sbatch batch/run_new_experiments.sbatch` — full E1-F3 + G1-G3
- [x] Results collected under `experiments/` (see Experiments tables above)
- [x] CHANGELOG.md entries per change

## Interfaces

### Model forward signatures (for LightningModule dispatch):
```
TweedieSolver:
  training_step(stage=1): model.estimate_mean(obs) → (B,T,D)
  training_step(stage=2): model(obs) → (B,T,D)
  config_optim(stage=1): model.mean_estimator.parameters()
  config_optim(stage=2): model.non_gaussian.parameters()

DirectUNet:
  training_step: model(obs) → (B,T,D)
  loss: StateMSELoss(pred, batch.states)
  config_optim: model.parameters()

VanillaCFM:
  training_step: compute_cfm_loss(batch) → scalar
  config_optim: model.parameters()
  sampling: model.sample(obs, N_outer) → (B,T,D)
```

### Dataset output format:
```python
{
    "true_state": Tensor(T, 3),
    "obs": Tensor(T, 3),
    "obs_mask": Tensor(T,),
    "forcing_true": Tensor(T,),
    "forcing_corrupted": Tensor(T,),
}
```

### results.json format (per experiment):
```json
{
  "experiment_id": "...",
  "config": {...},
  "epochs_trained": ...,
  "total_time_seconds": ...,
  "train_time_seconds": ...,
  "eval_time_seconds": ...,
  "fm_cs1": {"X": {"mean": ..., "std": ...}, "Y": ..., "Z": ..., "mean": ...},
  "fm_cs2": {...},
  "fm_cs3": {...},
  "fm_cs4": {...},
  "fm_degradation": ...,
  "fm_degradation_cs3cs4": ...
}
```
