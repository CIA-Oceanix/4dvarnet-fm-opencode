# Gyrostat wind forcing for the QG case study — implementation and response-analysis plan

**Status:** DRAFT v1 (2026-09-26). Nothing implemented. This plan executes
**WP1 and WP2** of `docs/plans/case_study/qg_gyrostat_synoptic_forcing.md`:
Option A, a gyrostat driver for the existing storm parameters, plus the
response analysis built around a phase-randomized surrogate. It is written
so that the driver is reusable by
`docs/plans/case_study/qg_hurricane_coupled.md` (the storm environment of
QG-TC) and later by `docs/plans/case_study/cgoa_coupled_gyrostats.md`.
The DA scenarios (scoping WP6) and Options B–F are out of scope, apart from
the seams listed in §8.

---

## 1. Goal and decision

**Goal.** Add a second wind driver to the QG case study. It must:

1. produce the same `(A, x_c, y_c)` storm state as today, so that nothing
   downstream changes;
2. leave the default (OU) path and every existing cache **bit-identical**;
3. support a phase-randomized surrogate, so that forcing *structure* can be
   separated from forcing *energy*.

The runs then answer Q1 of the scoping note: **at a fixed forcing spectrum,
does the upper-ocean response depend on the higher-order structure of the
forcing?**

**Decision the plan produces (§6.4).** Either:
- structure matters, measurably above intrinsic variability: proceed to
  Option B and the DA scenarios; or
- it does not: record a negative result, and move the gyrostat work to QG-TC,
  where the coupling is two-way and structure is expected to matter more.

## 2. What the code does today (verified on master @ c92b9b3)

**Storm generation.**
- `QGDynamics.generate_wind_state` (`models/qg_dynamics.py:156`) integrates
  three independent Ornstein–Uhlenbeck (OU) processes step by step: the
  amplitude `A` (memory `wind_tau_days = 15`, stationary std = `wind_amp`)
  and position jitters `wx, wy` (10 days, 50 km).
- The centre drifts at `(wind_cx, wind_cy)`.
- Output: a `(T, 3)` tensor `(A, x_c, y_c)`.
- `wind_amp = 0` returns zeros, and a test pins that as bit-identical to the
  unforced model (`tests/test_qg_dynamics.py`).

**Truth generation** (`QGS01Dataset._generate_truth_only`, `data/qg.py:446`).
Per window `i`:
- **Random draws:** `rd, U1, rek` within ±15%, start position `(x0, y0)`, and
  drift `cx ∈ [0.25, 0.75]`, `cy ∈ [−0.06, 0.06]` m/s.
- **Amplitude level:** cycles over `_S1_WIND_LEVELS = (0, 3e-12, 1e-11,
  2e-11, 3e-11)` (`data/qg.py:159`), so **one window in five has no wind**.
- **Seeds:** the storm seed is `cfg.seed + 2000 + 101 i`.
- **Spinup:** **2 years unforced**, then a 10-day lead and the 30-day window,
  both forced (`generate_full_trajectory`, `models/qg_dynamics.py:322`, runs
  its spinup loop without wind).

**Consumers of the storm state.** All of them read the `(A, x_c, y_c)` tensor
or the curl field built from it; none regenerates the storm:
- `wind_curl_field` (the PV source);
- the S1 corruption `_make_corrupted_wind_state` (`data/qg.py:392`), which is
  relative to `A.std()`;
- the DA models, through `WindStateAdapter` (`evaluation/run_qg_baselines.py:35`);
- the psi-state wrappers (`models/qg_psi_dynamics.py`);
- the neural forcing channel: the daily-mean `wind_curl`, normalized by
  stats from `precompute_qg_norm_stats.py`;
- `forcing_true` / `forcing_corrupted` (the amplitude only).

`QG1LDynamics` has its own copy of `generate_wind_state`. It is used only to
generate data in its own tests, never for the benchmark truth.

**Caches.**
- `_truth_cache_path` (`data/qg.py:666`) hashes `asdict(cfg)`.
- `_truth_only_cache_path` hashes the fields listed in
  `_TRUTH_ONLY_CFG_FIELDS` (`data/qg.py:687`).
- **Adding a field to `QGConfig` changes the first hash for every existing
  config, even at its default value.** That would silently invalidate every
  cached truth, at about 4.5 min per window to regenerate. §4.3 avoids it.

## 3. Lorenz-63 time scales (measured, not assumed)

Standard parameters (σ, ρ, β) = (10, 28, 8/3), RK4 with `dt = 0.005`, a
2000-unit trajectory after burn-in:

| mode | mean | std | ACF e-folding (units) | kurtosis | note |
|---|---|---|---|---|---|
| x | 0.09 | 7.92 | 0.305 | 2.31 | bimodal (two lobes), platykurtic |
| y | 0.09 | 9.02 | 0.220 | 2.86 | |
| z | 23.53 | 8.64 | 0.140 | 2.12 | oscillatory ACF (integral ≈ 0) |

- **Lobe residence** (time between sign changes of x): mean 1.72 units,
  median 1.53, 90th percentile 3.57.
- **Oscillation** around each fixed point: about 0.7 units.

**Consequences for the mapping:**

- **Matching today's amplitude memory** (15-day e-folding) on `x` gives
  `τ_g = 15 / 0.305 ≈ 49 days per L63 unit`. Then:
  - storm amplitude oscillates with a period of about a month;
  - regimes flip every ~3 months (median 75 days);
  - one 30-day window usually sits inside a single regime.
- **Matching a synoptic time scale** (a ~5-day oscillation) gives
  `τ_g ≈ 7 d/unit`: e-folding ~2 days, regimes ~12 days, several regime
  changes per window.
- **The "structure" L63 offers is bimodality, oscillation and regime
  switching, not heavy tails.** `x` is platykurtic (kurtosis 2.3 < 3). If
  intermittent extremes are the target, the mapping has to create them, e.g.
  `A ∝ x² − ⟨x²⟩`, or a product of modes. Decision 2 (§9).

## 4. Design

### 4.1 Driver module — `models/gyrostat_driver.py` (new)

Pure torch, float64 internally, deterministic per seed.

- **`GyrostatSystem`** — a gyrostat right-hand side in the form of the
  scoping notes, `ẏ = N(y, y) + L y + F − κ y`. The first preset is
  `"l63"` (the gyrostat form of Lorenz-63: `(p, q, r) = (0, −1, 1)`,
  `c = −σ`, damping `(σ, 1, β)`, forcing `−β(ρ + σ)` on the shifted z).
  A `"chain2"` preset (5 modes) is added only if decision 3 asks for it. The
  class exposes `energy(y)` for the conservation test.
- **`integrate(n_steps, dt_days, time_unit_days, seed, burnin_units=50)`**:
  - seeded initial condition, then a burn-in onto the attractor;
  - RK4 with internal substeps chosen so the L63 step is ≤ 0.005 units (at
    `τ_g = 49 d/unit` the 2-hour model step is 0.0017 units, so one substep;
    at 7 d/unit, 0.012 units, so three substeps);
  - returns `(n_steps, n_modes)`.
- **`map_to_storm(y, amp, cx, cy, x0, y0, L, W, dt, mapping)`** returns the
  `(n_steps, 3)` storm state:
  - **amplitude:** `A = amp · ỹ_A`, where `ỹ` is the mode standardized by its
    **attractor** mean and std (constants measured once and stored with the
    preset, not per window). The per-window level cycle therefore still sets
    the energy, and `amp = 0` gives an exactly zero `A`;
  - **zonal position:** integrated from a modulated speed,
    `x_c ← x_c + cx · (1 + b · ỹ_U) · dt`, modulo L;
  - **meridional position:** `y_c = y0 + cy · t + d · ỹ_Y`, modulo W;
  - **default mapping for L63:** `A ← x`, `U ← z`, `Y ← y`. Keep in mind that
    `x` and `y` are strongly correlated (same lobe), which ties storm-track
    latitude to the sign of the forcing, as in NAO-like regimes.
  - Mode-to-role assignments are a named `mapping` dict, so R2's "two or three
    mappings" is a config change, not a code change.
- **`phase_randomized_surrogate(series, seed)`** — the multivariate surrogate
  of Prichard & Theiler (1994):
  - FFT each channel;
  - add the **same** random phases to all channels, which preserves every
    auto- and cross-spectrum;
  - inverse FFT.
  - It acts on the standardized modes **before** mapping, over a series about
    4× longer than needed; cutting the middle avoids periodic-edge artefacts.
  - Surrogate marginals are Gaussian by construction, which is the point of
    the control.

### 4.2 Hook in `QGDynamics`

- **New constructor arguments:** `wind_driver: str = "ou"` and a
  `gyro: dict | None = None` bundle (preset, `time_unit_days`, `mapping`,
  `b`, `d`, `burnin_units`, `surrogate: bool`).
- **Dispatch:** `generate_wind_state` dispatches on `wind_driver`:
  - `"ou"`: the **current body, unchanged**;
  - `"gyrostat"`: integrate, then map;
  - `"gyrostat_surrogate"`: integrate, randomize phases, then map.
- **Same signature** `(num_steps, seed, x0, y0)`, so `data/qg.py` needs no
  call-site change beyond passing the constructor arguments.
- **Not touched:** `QG1LDynamics`, `wind_curl_field`, `_tendency`, and every
  consumer listed in §2.

### 4.3 Config and caches — `data/qg.py`

- **`QGConfig` gains:**
  - `wind_driver: str = "ou"`;
  - `gyro_preset: str = "l63"`;
  - `gyro_time_unit_days: float`, the value from decision 1;
  - `gyro_mapping: str = "x_z_y"`;
  - `gyro_speed_mod: float` and `gyro_lat_mod_m: float`;
  - `gyro_burnin_units: float = 50.0`.
- **`_generate_truth_only`** passes them to `QGDynamics`. The per-window
  draws of `x0, y0, cx, cy, amp` and the seeds are **unchanged**, so an OU
  window and a gyrostat window with the same index share the same ocean
  parameters, start position, drift and energy level. Only the storm's
  time series differs.
- **Cache keys.**
  - Both hash functions drop every `wind_driver`/`gyro_*` field when
    `wind_driver == "ou"`, so every existing key is unchanged.
  - Otherwise they add those fields plus a `GYRO_DRIVER_VERSION` constant,
    in the same way as `OBS_GEOMETRY_VERSION`.
  - A test pins the current key strings for the default config (§5).
- **The S1 scenarios apply unchanged.** `_make_corrupted_wind_state` scales
  by `A.std()` and jitters the position, which is still meaningful. Under a
  gyrostat truth, though, this is still *unstructured* corruption. The
  structured S1-gyro scenarios are scoping WP6, not this plan.

### 4.4 Long forced runs for the response analysis

**The benchmark truth is not suitable for Q1–Q3.** The ocean is spun up
without wind and then forced for only 40 days, so it is never in statistical
equilibrium with the forcing. The response analysis therefore uses separate
long runs, in a new script `reports/qg/run_wind_response.py`:

- **Setup:** nominal ocean parameters (no ±15% draws), a fixed amplitude level
  (`1e-11`, the middle of the level cycle; decision 4), unforced spinup
  (2 years).
- **Forced run:** 1 year of forced adjustment, discarded, then **5 years
  analysed**. That is about 26k steps. The script loops on `step`, keeping
  daily snapshots of `ψ₁, ψ₂, q₁` plus an hourly forcing series and the
  per-step domain integrals.
- **Ensembles** (Eₖ) batched on GPU, starting from 10 ocean states (spinup
  seeds) under **identical** forcing.
- **Storage:** snapshots and ensemble members go to **node-local `/tmp`**
  (the standing rule since the 2026-09-24 disk-full incident). Only reduced
  diagnostics (spectra, statistics, composites; a few MB) are written to
  `reports/qg/outputs/`.

## 5. Tests (`tests/test_qg_gyrostat_driver.py`, fast unless marked)

| test | asserts |
|---|---|
| default unchanged | `wind_driver="ou"` returns the current `generate_wind_state` output bit for bit (fixed seed); a 50-step QG rollout is bit-identical |
| cache keys unchanged | `_truth_cache_path` and `_truth_only_cache_path` of the default `QGConfig` equal their current strings (pinned literals) |
| cache keys separate | a gyrostat config and its surrogate hash to different keys, both different from the OU key |
| L63 = gyrostat | the `"l63"` preset's right-hand side equals the textbook Lorenz-63 right-hand side after the `z` shift (random states, float64) |
| energy | the unforced, undamped core conserves `energy(y)` to RK4 accuracy |
| attractor constants | the stored mean and std per mode match a fresh long run within 5% (`slow`) |
| zero amplitude | `amp = 0` gives `A ≡ 0`, and a rollout bit-identical to the unforced model |
| determinism | same seed gives an identical storm state; different window seeds give different series |
| surrogate spectrum | the surrogate preserves each channel's power spectrum and the cross-spectra (to FFT precision) and has Gaussian marginals (kurtosis within 3 ± 0.3 on a long series) |
| format | output shape `(T, 3)`, dtype and device follow the dynamics object, `x_c ∈ [0, L)`, `y_c ∈ [0, W)` |
| end to end | a 2-window `QGS01Dataset` with `wind_driver="gyrostat"` builds, and S0/S1 windows carry `wind_state_true`/`wind_state_corrupted` with the usual shapes (`slow` if spinup dominates; `spinup_years` can be cut for the test) |

Per the repo rules: `ruff check` on the touched files, and the fast suite
before any PR.

## 6. Runs and analysis (WP2)

### 6.1 Experiments

| id | driver | purpose |
|---|---|---|
| E0 | OU (current) | reference |
| E1 | gyrostat, `τ_g` from decision 1 | the new forcing |
| E2 | surrogate of E1 | same spectrum as E1, Gaussian: **the control** |
| E3 | gyrostat, the other `τ_g` of decision 1 | time-scale sensitivity |
| E1-m2 | E1 with a second mapping | mapping sensitivity (R2) |
| E1-k, E2-k | E1 and E2, 10 ocean initial states each, identical forcing | forced response vs intrinsic variability |

All runs have the same ocean, the same amplitude level and 5 analysed
years.

### 6.2 Diagnostics (script `reports/qg/analyze_wind_response.py`)

1. **Frequency spectra** of upper-layer KE, `ψ₁` and `ψ₂` (domain-mean and
   point-wise), integrated in three bands: synoptic (< 10 d), intraseasonal
   (10–90 d) and low-frequency (> 90 d).
2. **Transfer function and coherence** from `A(t)` to domain-integrated
   upper-layer KE and to `ψ₁` at the storm centre.
3. **Energy budget:** wind work `⟨ψ₁ · curl τ⟩`, EKE per layer, and the
   dissipation by bottom drag.
4. **One-point statistics:** PDFs, skewness and kurtosis of `ψ₁` and `q₁`.
5. **Regime composites** (E1 only): ocean anomalies averaged around lobe
   switches of the gyrostat, from −30 to +60 days.
6. **Forced vs intrinsic:** in E1-k/E2-k, the variance of the ensemble mean
   (forced) against the mean ensemble variance (intrinsic), per band.

### 6.3 Statistics

- Moving-block bootstrap over the 5-year series (block length ≥ 3× the
  longest band's time scale) for single runs.
- Across-member spread for the ensembles.
- A difference is called **significant** when it exceeds the 95% bootstrap
  interval **and** the E1-k/E2-k member spread.

### 6.4 Decision criteria

- **Structure matters** if E1 and E2 differ significantly in at least one of:
  band-integrated KE (by more than 10%), `ψ₁`/`q₁` kurtosis, or the
  forced-variance fraction, **and** the difference keeps its sign under
  E1-m2. Next: Option B, then scoping WP6 (DA scenarios).
- **Structure does not matter at this level** if no such difference
  survives. Next: write the negative result in `docs/results/`, keep the
  driver (QG-TC uses it for its environment), and do not build Option B for
  the one-way case.
- **Always reported:** E0 vs E2 (the effect of the spectrum) and E1 vs E3
  (the effect of the time scale), whatever the E1/E2 outcome.

## 7. Work breakdown and PRs

| step | content | PR |
|---|---|---|
| 1 | `models/gyrostat_driver.py` (L63 preset, integrate, map, surrogate) + its unit tests | PR-1 |
| 2 | `QGDynamics` hook, `QGConfig` fields, conditional cache keys, the "unchanged" tests | PR-1 (same PR: step 1 is unusable alone) |
| 3 | attractor constants (measured once, stored in the preset) + the L63 time-scale numbers of §3, recorded in `docs/results/` | PR-1 |
| 4 | `reports/qg/run_wind_response.py` + batch script (default env `fdv-monai-proto`, outputs to node-local `/tmp`) | PR-2 |
| 5 | E0–E3 + ensembles on the cluster | — (runs) |
| 6 | `reports/qg/analyze_wind_response.py`, report under `reports/qg/outputs/`, results doc in `docs/results/` stating the §6.4 decision | PR-3 |

**Cost, to be confirmed in step 4.** Truth generation today runs at about
4.5 min per window on CPU for ~9.2k steps, i.e. ~35 steps/s. A 6-year run
(~26k steps) is about 13 min on CPU, less on GPU, and the 10-member
ensembles batch through `generate_batch_trajectories`. The whole of WP2 is
on the order of a few GPU-hours.

## 8. Seams kept for later work (not built here)

- **QG-TC:** the driver returns the modes as well as the storm state, so
  QG-TC can map them to translation speed, heading and shear
  (`qg_hurricane_coupled.md` §5.1) without a second implementation.
- **CGOA:** `GyrostatSystem` takes an explicit gyrostat specification, so
  the CGOA builder can later produce the presets instead of hand-written
  right-hand sides.
- **Option B:** a spectral mapping is another `map_to_*` function returning a
  wider wind state. It is kept out of `forcing_true` as scoping R5 requires.
- **Neural rows under the new driver** (scoping WP6): the forcing-channel
  normalization stats must be recomputed per driver
  (`precompute_qg_norm_stats.py`). Every method in a comparison must use the
  same driver, per the apples-to-apples rule.

## 9. Open decisions

1. **Time unit `τ_g`.** Match today's 15-day amplitude memory
   (`τ_g ≈ 49 d/unit`: monthly oscillation, ~3-month regimes), or a synoptic
   time scale (`τ_g ≈ 7 d/unit`: 2-day memory, ~12-day regimes)? The first
   keeps E0 and E1 comparable in memory; the second is closer to real storms.
   **Proposal:** the first for E1, the second as E3.
2. **Which structure to test.** L63 gives bimodality and regimes, not heavy
   tails (§3). Keep the linear mapping (`A ∝ x`, testing regimes), or add a
   heavy-tailed variant (`A ∝ x² − ⟨x²⟩`, or a product of modes)?
   **Proposal:** linear for E1, the quadratic variant as E1-m2, which also
   serves as the mapping-sensitivity run.
3. **One gyrostat or a chain of two.** L63's modes are strongly correlated.
   A two-gyrostat chain (5 modes) gives more independent roles for amplitude,
   speed and latitude. **Proposal:** start with L63; add `chain2` only if
   the correlated mapping turns out to confound the composites.
4. **Amplitude level for the long runs.** One level (`1e-11`, proposed), or
   the full level cycle as separate runs? The level cycle matters for the DA
   benchmark, less for Q1.
5. **Position jitter.** The OU path has a 50 km position jitter; the
   gyrostat path replaces it with deterministic modulation. Keep a small OU
   jitter on top, for comparability of track roughness, or not?
   **Proposal:** no. The surrogate already controls for the spectrum, and
   mixing in OU noise blurs the structure test.

## References

- Lorenz (1963), deterministic nonperiodic flow. *J. Atmos. Sci.* 20, 130–141.
- Prichard & Theiler (1994), generating surrogate data for time series with several simultaneously measured variables. *Phys. Rev. Lett.* 73, 951–954.
- Theiler, Eubank, Longtin, Galdrikian & Farmer (1992), testing for nonlinearity in time series: the method of surrogate data. *Physica D* 58, 77–94.
- Künsch (1989), the jackknife and the bootstrap for general stationary observations. *Ann. Statist.* 17, 1217–1241.
- Scoping context: `docs/plans/case_study/qg_gyrostat_synoptic_forcing.md`, `docs/plans/case_study/qg_hurricane_coupled.md`, `docs/plans/case_study/cgoa_coupled_gyrostats.md`.
