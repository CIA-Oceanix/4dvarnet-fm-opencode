# ETKF/EnKF DA on the spectral-wind QG datasets, S0 (ocean only, true wind known) — experiment design

**Status:** DRAFT v2 (2026-09-26): decisions settled (§9); DA-1 in progress. Nothing run. It designs the first DA
baselines on the new QG datasets:
- the forced dataset `qg_specwind_gyrostat_v1` (Option B + eddy drag;
  `docs/results/qg_specwind_demo_dataset.md`);
- the two-way coupled dataset `qg_coupled_gyrostat_v1` (Option C, κ_fb = 1;
  `docs/results/qg_coupled_dataset.md`).

The scope is S0 and the **ocean component only**: the DA state is the
two-layer QG PV, and the wind is prescribed from the truth. The DA methods
are the project's ETKF and EnKF (`evaluation/baselines.py`) at their current
QG defaults.

---

## 1. Questions

- **D1: reconstruction skill.** How well do ETKF and EnKF reconstruct the
  ocean state (ψ₁, ψ₂, q₁, q₂) from sparse, nadir-like ψ₁ observations when
  the wind is known exactly? This is the reference row for the neural
  methods trained with train-time regeneration (G5).
- **D2: forced vs coupled truth.** Given the true wind, the ocean equations
  are the same in both datasets; the coupling only changes the atmosphere.
  **S0 is therefore a perfect-model setting for both.** D2 checks this:
  skill on the two test sets should agree within confidence intervals. A
  significant gap would point to a data or forcing mismatch, not to physics.
- **D3: what the new datasets add.** Skill stratified by the factors the
  datasets vary: wind level (calm vs windy), gyrostat time unit, regime
  code, `rd`, `U1`. The legacy QG benchmark cannot do this.
- **D4: bridge to the legacy benchmark.** The same methods at the legacy
  observation density (4 columns per day). This places the new numbers next
  to `reports/qg/outputs/qg_da_report.md`, with the caveat that the forcing
  differs.

## 2. Setting

| element | choice |
|---|---|
| truth | the full-resolution **test** splits (500 windows each; stored every step) |
| DA state | two-layer QG PV, 64 × 64 × 2 = 8192 |
| DA model | `QGDynamics` with the window's **true** parameters (`rd`, `U1`, `rek`, `beta`, `U2`) and **the truth's wind amplitudes** (12 per step, stored at full resolution in both datasets), through the spectral forcing |
| model error | none. The eddy drag `r_cf` is in both truth and model. For the coupled truth the wind is the coupled atmosphere's actual series, so the ocean model is exact given the wind |
| window | 30 days (360 steps of 2 h) after the 10-day lead |
| initial condition | the QG reference case: truth lagged by **5 days** (`init_lag_days = 5.0`), ensemble dispersion as in `run_qg_baselines._ensemble_from_init` (`disp_frac = 1`). One shared initial state per window for the ensemble and the free forecast |

### 2.1 Observations (nadir-like, ~5% of the grid, one observation per day per location)

- **Field:** upper-layer streamfunction ψ₁, the SSH proxy (`obs_var = "psi"`).
- **Geometry:** the existing `random_columns` geometry (v2) in
  `data/qg.py`. Each day, `cols_per_day` distinct meridional columns are
  observed, each **once**, at its own random intra-day step. A column is
  64 points, a full meridional pass, so:

  | `cols_per_day` | share of grid points per day | role |
  |---|---|---|
  | **3** | **4.7%** | **primary** (nadir-like, ≈ 5%) |
  | 4 | 6.25% | legacy-bridge (D4) |
  | 1, 2, 6 | 1.6%, 3.1%, 9.4% | sensitivity (§4.3) |

- **Noise:** 5% of the ψ₁ standard deviation (`obs_noise_std_frac = 0.05`,
  the QG reference case).
- **Caveat:** real nadir tracks are inclined, not meridional. Meridional
  columns are the approximation the QG case study already uses. An inclined
  track geometry is a follow-up, noted in §8.

### 2.2 Methods and hyperparameters

| method | settings (current QG defaults, `qg_da_report.md`) |
|---|---|
| ETKF | N = 80, inflation 1.0, `loc_radius` 2.0, `etkf_ridge` 0.1 |
| EnKF | N = 80, inflation 1.0, `loc_radius` 2.0 |
| free forecast | the shared initial state rolled forward with the true wind, no observations (already returned by `run()`) |

Localization defaults were tuned at 4 columns per day on the storm-forced
QG. The forcing here is domain-scale spectral, and the density slightly
lower, so §4.1 re-checks them **on val**.

## 3. Implementation

The DA loop, observation system, localization and metrics are reused as
they are. Only the window source and the DA model's wind change.

| piece | change |
|---|---|
| spectral DA model | `QGDynamics` gains the spectral wind hook of the Option B plan's PR-2 (`wind_driver`, `wind_state_dim`, and a `_wind_curl_spectral` dispatch). The legacy path stays bit-identical. **Minimal version for this study:** only the `QGDynamics` part; the one-layer model and S1 corruption stay in PR-2 proper |
| `evaluation/run_qg_baselines.py` | `_build_dyn` returns the spectral-forced `QGDynamics` when the window carries `specwind` metadata. `WindStateAdapter` already forwards a wind row of any width |
| window source | a thin driver `evaluation/run_qg_specwind_da.py`: `materialize_split(spec, split)` → `with_fixed_obs(windows, cfg)` → adds the S0 scenario fields (`da_model = "qg2l"`, `da_nx = 64`, `da_params = true_params`, `wind_state_corrupted = wind_state_true`) and the `target_state_psi` / `target_state_q` fields `run()` reads → `run(method, cfg, ds={"test_s0": windows}, ...)`. Everything reused comes from G5 (`data/qg_specwind_neural.py`) |
| scoring | `evaluation/estimate_metrics.py`, per layer and field; spread/RMSE; per-day error curves; factor-stratified summaries from the manifest's factors |
| report | `reports/qg/generate_qg_specwind_da_report.py` → `reports/qg/outputs/qg_specwind_da_s0_report.md`, with best and second-best marked per metric column |

**Tests:**
- the spectral `QGDynamics` step equals `BatchedQGDynamics` with spectral
  forcing (float64);
- the legacy path is bit-identical;
- an S0 window built by the driver runs one ETKF analysis end to end;
- driver windows use the test split only with `purpose='test'`, and the
  tuning runs use val.

## 4. Experiments

### 4.1 Tuning on val (never on test)

- **Data:** val windows re-materialized at full resolution (G5's
  `materialize_split`, bit-exact to the stored frames), 20 windows, forced
  dataset only. Given the wind, the DA problem is identical for both
  datasets (D2), so tuning once suffices.
- **ETKF:** `loc_radius` ∈ {1, 2, 3} × `etkf_ridge` ∈ {0, 0.1, 1}.
- **EnKF:** `loc_radius` ∈ {1, 2, 3}.
- **Inflation** stays 1.0; the QG study found any value > 1.0 harmful.
- **Selection:** best mean q EV over the 20 windows, tie-break on ψ EV.
  Keep the defaults unless a config beats them by more than the val
  bootstrap interval.

### 4.2 Main runs (test)

- **Methods:** ETKF and EnKF at the tuned settings, plus the free forecast.
- **Data:** both test sets, at `cols_per_day` = 3 (primary) and 4 (bridge).
- **Window counts:** 100 windows per test set for the tables, as the legacy
  QG reference uses N = 100. All 500 windows for the final D2 comparison and
  the stratified D3 analysis, where the per-stratum counts need them.

### 4.3 Density sensitivity (ETKF only, forced test, 100 windows)

`cols_per_day` ∈ {1, 2, 3, 4, 6}, re-using the val-tuned localization. The
QG obs-density study showed `loc_radius` must shrink with density, so if
the curve is non-monotonic, re-tune on val at that density before reading
it.

### 4.4 Metrics and statistics

- **Per window, averaged over the 30 days:** RMSE and EV of ψ₁, ψ₂, q₁
  and q₂ (the legacy QG metrics); ensemble spread/RMSE; free-forecast
  RMSE.
- **Per day:** the error curve over the window (spin-up of the filter).
- **Confidence:** 95% intervals from a bootstrap over windows.
- **D2:** a paired comparison is **not** possible, since test windows of
  the two datasets share seeds but not trajectories (the chaos). So the
  forced and coupled test sets are compared as independent samples, with a
  two-sample bootstrap on the difference of means.
- **D3:** strata of the manifest factors: calm vs windy; level terciles;
  time-unit terciles; regime groups {0, 15} vs others.

## 5. Expected outcomes and how to read them

- **D1:** skill below the legacy storm-forced QG at matched density is
  plausible. The spectral forcing is steadier and domain-scale, which may
  give the unobserved layer a different predictability. There is no prior
  number to beat; this study creates it.
- **D2:** forced ≈ coupled within intervals. **If not**, first check the
  wind series fed to the DA model against the truth's, and the materialized
  full-resolution truth against storage, before any physical
  interpretation.
- **D3:** calm windows are expected to be easier for the lower layer (less
  forced variability) and harder for the upper layer in relative terms
  (smaller signal). Regime and time-unit effects are open.
- **D4:** the gap between 3 and 4 columns per day shows how steep the
  density curve is around the nadir-like point.

## 6. Compute

- **Cost per window:** the legacy QG ETKF takes about 20–47 s per window
  (`da_sensitivity_s0_s1_report.md` timing probe); EnKF is similar. Taking
  35 s:

  | runs | windows × configs | time |
  |---|---|---|
  | tuning on val (ETKF 9 + EnKF 3 configs) | 20 × 12 | ≈ 2.3 h |
  | main, primary density: 2 methods × 2 test sets | 100 × 4 | ≈ 3.9 h |
  | main, bridge density | 100 × 4 | ≈ 3.9 h |
  | D2/D3 extension to 500 windows (ETKF only, both test sets, primary density) | 400 × 2 more | ≈ 7.8 h |
  | density sensitivity (3 new densities) | 100 × 3 | ≈ 2.9 h |
  | **total** | | **≈ 21 GPU-hours** |

- **Materializing val at full resolution:** about 4 min per 500 windows.
- **Parallelism:** splitting windows over processes is independent per
  window. Batching members across windows would be a later optimization.
- **Location constraint.** Both datasets live on `/SCRATCH` of
  `sl-mee-br-202`, which other cluster nodes cannot read. Two options:
  1. run on that server's single RTX 8000 (about 21 h, sequential);
  2. stage the test splits and the full-resolution val subset to `/Odyssey`
     (test: 7.4 GB each; val 20 windows: < 0.5 GB) and run as a SLURM
     array.

  Option 2 is recommended; decision 1.

## 7. Work breakdown

| step | content |
|---|---|
| DA-1 | Spectral hook in `QGDynamics` (the minimal part of PR-2), the `_build_dyn` dispatch, the driver `evaluation/run_qg_specwind_da.py`, and tests. One PR. |
| DA-2 | Tuning on val (§4.1); record in `docs/results/`. |
| DA-3 | Main runs, bridge runs and density sensitivity (§4.2–4.3); SLURM array if the data are staged. |
| DA-4 | Report generator and report; results doc with D1–D4. Archive the run config and estimates per the archive convention. |

## 8. Out of scope, and follow-ups

- **S1 (model error):** needs PR-2's spectral S1 corruption (wind
  amplitude bias and a coherent phase shift, parameter bias) and, for a
  structural error, the one-layer DA model. Natural next step once S0
  stands.
- **Coupled DA (ocean + atmosphere state):** a strongly coupled ETKF with
  the gyrostat modes in the state and localization between the global modes
  and the local ocean. That is the reanalysis-like setting the coupled
  dataset was built for, and a separate design.
- **Inclined nadir tracks** instead of meridional columns, and a realistic
  along-track noise model.
- **4D-Var:** about 15–20× the cost of the ensemble methods per window, and
  it collapses on q layer 2 in QG S1. Deferred.

## 9. Decisions (settled 2026-09-26)

| # | outcome |
|---|---|
| 1 | **Stage to `/Odyssey`** (`experiments/qg_datasets/`: both test splits, the val splits and the reports) and run as a SLURM array |
| 2 | **3 columns per day (4.7%) headlines the tables**; 4 columns per day is the bridge |
| 3 | **100 test windows per dataset** for the tables; extended to 500 only for D2/D3 |

The original options are below, for the record.


1. **Where to run:** stage the test and val subsets to `/Odyssey` for a
   SLURM array (recommended; about 15 GB), or run sequentially on
   `sl-mee-br-202` (about 21 h)?
2. **Primary density:** 3 columns per day (4.7%, proposed as the nadir-like
   ≈ 5%) or 4 (6.25%, the legacy reference)? Both are run; this decides
   which one headlines the tables.
3. **Window counts:** 100 per test set for the tables (proposed, matching
   the legacy reference), extended to 500 only where D2/D3 need it; or 500
   everywhere (+≈ 11 h)?
