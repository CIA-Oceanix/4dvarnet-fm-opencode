# Coupled hurricane–ocean configuration for the QG case study (QG-TC)

**Status:** SCOPING v1 (2026-09-25). Nothing implemented. Written to be read
without the rest of the repo. It extends the QG case study with a
tropical-cyclone (TC) event and an upper-ocean thermal response, and it builds
on two companion notes:
`docs/plans/case_study/qg_gyrostat_synoptic_forcing.md` (gyrostat atmosphere
driving the QG ocean, Options A–E) and
`docs/plans/case_study/cgoa_coupled_gyrostats.md` (coupled gyrostat engine and
the shared DA pieces). In the numbering of the first note, this is an
**Option F**: a localized, two-way coupled vortex. It needs a light version of
the deferred Option E (an SST tracer), but **not** Option B, because the
feedback acts through a scalar, the SST under the storm core.

---

## 1. Summary

A hurricane is the sharpest case of coupled ocean–atmosphere dynamics that a
two-layer QG ocean can still say something useful about:

- the forcing is intense, localized and fast (hours to days);
- the ocean response is slow (days to weeks) and partly geostrophic;
- the feedback is strong, local and negative: the storm cools the sea surface
  under itself and loses intensity, more so when it moves slowly.

The proposal adds three low-order components to the existing QG ocean:

1. a **parametric vortex** (Holland wind profile) whose intensity obeys a
   quadratic, Emanuel-type equation driven by the SST under its core;
2. a **slab mixed layer** carrying SST and mixed-layer depth, cooled by
   wind-driven entrainment and by thermocline uplift read from the two-layer
   interface;
3. a **gyrostat environment** (the Option A driver) that sets the storm's
   translation speed, heading and vertical wind shear.

A ladder of configurations (§6) goes from a linear check against Geisler
(1970) to the fully coupled system. §8 discusses how the setting could be
carried to real wind and ocean observations for reconstruction, forecasting
and calibration. §9 gives the implementation plan.

The QG benchmark default is unchanged. Every configuration here is a new
setting, and the current OU storm stays the default.

## 2. Starting point in the repo

**Ocean.** `models/qg_dynamics.py`, a torch port of pyqg v0.4.0: two layers,
doubly periodic 1000 km × 1000 km, `nx = 64` (Δx ≈ 15.6 km), `rd = 15 km`,
`δ = 0.25`, `U1 = 5 cm/s`, RK4 at `dt = 2 h`. Defaults sit in `QGConfig`
(`data/qg.py:15`). The upper-layer thickness `H1 = 500 m` is hard-coded in the
energy diagnostic (`models/qg_dynamics.py:389`).

**Wind path, as it is today.**

- `generate_wind_state` (`models/qg_dynamics.py:156`) precomputes a `(T, 3)`
  tensor `(A, x_c, y_c)` before the rollout.
- `wind_curl_field` (`:185`) turns it into a Mexican-hat PV source of width
  `wind_sigma = 250 km`.
- `_tendency` (`:222`) adds that source to `dq₁/dt` through
  `_wind_curl_spectral` (`:205`).
- `rollout_steps` reads `wind_state[k]` at step `k`.

Two consequences for this note:

- The forcing is **one-way by construction**: it is computed before the ocean
  runs. Two-way coupling needs a dynamics class that co-steps the atmosphere
  and the ocean (§9.1).
- `forcing_true` carries the amplitude only. The QG gyrostat note (its risk
  R5) already recommends keeping it that way and storing a wider wind state
  separately; this note follows that recommendation.

**Truth cache.** `_truth_cache_path` (`data/qg.py:666`) keys the cache on the
full `QGConfig` plus `OBS_GEOMETRY_VERSION` (`:679`). Every new field below
must enter that key.

## 3. Physics, and what a QG ocean can represent

### 3.1 The linear two-layer response: Geisler (1970)

Geisler solved the linear, hydrostatic response of a two-layer ocean to a
hurricane translating at speed `U`. The control parameter is the Froude number
`U/c`, with `c` the baroclinic long-wave speed:

- **`U > c` (the usual case).** A wake of near-inertial internal waves trails
  the storm, with wavelength ≈ `2πU/f`, on top of a ridge of upwelled
  interface along the track.
- **`U < c`.** A local upwelling under the storm, with no wave wake.
- **Barotropic mode.** `c ~ 200 m/s ≫ U`: a weak, quasi-steady response.

For the repo's ocean, `c = f·rd ≈ 1.5 m/s` if `f ≈ 10⁻⁴ s⁻¹` (pyqg fixes
`rd`, not `f₀`, so this is an assumption to record in the config). A 5 m/s
storm sits at `U/c ≈ 3`.

**What QG keeps.** QG filters inertia–gravity waves. It keeps the balanced
part of Geisler's solution: the upwelled ridge (a cyclonic PV anomaly in the
upper layer), its geostrophic currents, and its slow evolution by β-drift,
Rossby-wave radiation and interaction with the eddy field. It loses the
inertial wake entirely.

### 3.2 Sea-surface cooling

The observed cold wake is mostly due to **entrainment mixing** at the base of
the mixed layer, driven by shear of the near-inertial currents (Price 1981;
Price, Sanford & Forristall 1994). Upwelling contributes, more so for slow
storms. The two-layer QG ocean has neither a mixed layer nor near-inertial
currents, so:

- upwelling can be read from the interface displacement
  `η ∝ (f₀/g′)(ψ₁ − ψ₂)`, which QG does carry;
- mixing must be parameterized from the wind itself, through a friction-velocity
  closure (`w_e ∝ u*³`, Kraus–Turner / Niiler type) rather than from resolved
  ocean shear.

### 3.3 Feedback on the storm

A hurricane draws enthalpy from the ocean. Its potential intensity rises with
SST under the core; its dissipation rises with the cube of wind speed (Emanuel
1986, 1988). Cooling the SST under the core lowers the storm's intensity. Coupled
models show that this negative feedback is strongest for slow storms and for
shallow mixed layers (Schade & Emanuel 1999). Deep warm water, e.g. a warm-core
eddy, weakens the feedback and favours intensification (Shay, Goni & Black
2000; Mainelli et al. 2008).

The QG ocean has an active mesoscale eddy field. Its interface depth varies
from eddy to eddy, which gives a natural, dynamically consistent **ocean heat
content** field for the storm to cross. This is the main scientific reason to
use the QG ocean rather than a 1D ocean.

### 3.4 Summary table

| process | real ocean | QG-TC |
|---|---|---|
| Ekman upwelling under the core | yes | yes (PV source) |
| balanced ridge and geostrophic wake | yes | yes |
| near-inertial wake (Geisler, `U > c`) | yes | **no** (filtered) |
| entrainment cooling | dominant | parameterized, `u*³` closure |
| rightward bias of the cold wake | yes (inertial resonance) | partial: only through wind asymmetry from translation |
| eddy modulation of cooling | yes | yes (interface depth) |
| SST → intensity feedback | yes | yes (scalar SST under the core) |
| intensity → track feedback | weak | not modelled |

## 4. Scientific questions

- **Q1 — Geisler in a turbulent ocean.** How does the balanced wake of §3.1
  evolve in a baroclinically unstable eddy field? How long does it stay
  detectable in ψ₁ (i.e. in SSH) and in the interface?
- **Q2 — Eddy control of intensity.** For fixed storm parameters, how much of
  the spread in peak intensity comes from the ocean state the storm crosses?
  This is the QG-TC version of the warm-eddy effect.
- **Q3 — Translation speed.** How do cooling and the intensity feedback vary
  with `U/c` and with the residence time `R_max/U` when the translation speed
  comes from a chaotic environment (the gyrostat) rather than being fixed?
- **Q4 — Coupled DA.** With the storm observed densely and the ocean sparsely:
  - do ocean observations (SSH along-track or swath, sparse profiles) improve
    intensity estimates and forecasts?
  - do storm observations constrain the ocean wake?
  - what is the gain of strongly coupled over weakly coupled DA (CGOA §1, Q2)?
- **Q5 — Parameter identifiability.** Can exchange and mixing parameters (the
  ratio `C_k/C_D`, the entrainment coefficient `m₀`) be recovered by joint
  state–parameter estimation, and from which observations?

## 5. The configuration

### 5.1 Environment (gyrostat driver)

The Option A driver of the QG gyrostat note: a sparse chain of two gyrostats
(5 modes), time unit in days. Three modes are mapped to storm-scale
quantities:

- translation speed `U(t) = U₀ (1 + b · y_U)`;
- heading `θ(t) = θ₀ + d · y_θ`;
- vertical wind shear `S(t) = s₀ · y_S²` (non-negative, and still quadratic).

As in the QG note, the mapping is a modelling choice (its risk R2). The
phase-randomized surrogate of that note applies unchanged, as a control for
forcing structure versus forcing energy.

### 5.2 Vortex

State `z_v = (x_c, y_c, V)`, optionally with `R_max`.

- **Track.** `ẋ_c = U cos θ`, `ẏ_c = U sin θ`.
- **Intensity.** A schematic, quadratic Emanuel-type equation:

  ```
  dV/dt = (C_D / 2h_b) · (V_p² − V²) − κ_s S V
  V_p²  = V_p0² + λ (T_core − T_ref)
  ```

  where `T_core` is the mixed-layer temperature averaged over `r < 2 R_max`.
  `V_p0`, `λ`, `h_b` and `κ_s` are calibrated so that intensification and
  decay rates, peak intensity and the SST sensitivity of `V_p` are in the
  observed ranges. The form is quadratic in `(V, T_core, S)`, so it also fits
  the CGOA `QuadraticTensorDynamics` engine for a purely low-order variant
  (§6, H-lo).
- **Life cycle.** A genesis and decay envelope on `V` guarantees **one
  passage per window**. Without it the storm re-enters the periodic domain.

### 5.3 Wind stress and ocean forcing

- **Wind.** Holland (1980) axisymmetric profile with parameters
  `(V, R_max, B)`, plus a fraction of the translation vector. The latter gives
  the right–left asymmetry of the stress.
- **Stress.** `τ = ρ_a C_D(|u|) |u| u`, with `C_D` saturating at high wind
  speeds (Powell, Vickery & Reinhold 2003).
- **PV source.** `dq₁/dt += curl τ / (ρ₀ H₁)`. This replaces the Mexican-hat
  profile behind a `wind_profile` switch; the default stays `"mexhat"`.

**Order of magnitude.** `V = 40 m/s`, `C_D = 2×10⁻³`: `|τ| ≈ 4 N m⁻²`,
`curl τ ~ τ/R ≈ 8×10⁻⁵ N m⁻³` for `R = 50 km`. That gives:

- a PV source ≈ `1.5×10⁻¹⁰ s⁻²` over about a day, 15 times the current
  `wind_amp = 1e-11` (see risk R2);
- Ekman pumping `curl τ/(ρ₀ f) ≈ 8×10⁻⁴ m/s ≈ 70 m/day`, a realistic value.

### 5.4 Slab mixed layer

Two new 2-D fields: SST `T` and mixed-layer depth `h`.

```
∂T/∂t = −J(ψ₁, T) − (w_e / h)(T − T_b) − (T − T*) / τ_r
∂h/∂t = −J(ψ₁, h) + w_e − (h − h*) / τ_h
w_e   = m₀ u*³ / (g α h (T − T_b)),   u*² = |τ| / ρ₀
T_b   = T_ref − Γ (h + D₀ − η),       η ∝ (f₀/g′)(ψ₁ − ψ₂)
```

- `T_b` is the temperature just below the mixed layer. Thermocline uplift
  (`η > 0`, from Ekman pumping or a cold eddy) makes it colder, which couples
  the mixed layer to the QG interface.
- The restoring terms stand for air–sea fluxes and slow recovery of the wake
  (weeks).
- The mixed layer is **passive** for the QG PV: it does not change ψ. This
  keeps the QG core unchanged.

The closure is crude on purpose. The sign convention of `η` and the exact
closure are WP2 design choices (§9.4).

### 5.5 Coupling loop and state vector

```
environment (gyrostat) ──► U, θ, S ──► vortex (x_c, y_c, V)
                                            │  τ(x, y, t)
                                            ▼
                               QG ocean (q₁, q₂) ──► η
                                            │  u*³, η
                                            ▼
                               mixed layer (T, h)
                                            │  T_core
                                            └──────► back to V
```

State: `[q (2·ny·nx) | T, h (2·ny·nx) | z_v (3–4) | z_env (5)]`, with
`component_slices = {"ocean_q", "ml", "vortex", "env"}`, the same metadata
CGOA §5.1 uses for scoring, localization and whitening.

### 5.6 Proposed sizes (to be calibrated)

| quantity | proposal | note |
|---|---|---|
| grid | `nx = 128` over 1000 km (Δx ≈ 7.8 km) | 64 points leave only 2–3 across the core |
| time step | `dt = 1 h` | intensity changes on hours; stronger local velocities |
| `R_max` | 40–60 km | deliberately large, for resolution |
| translation `U₀` | 2–8 m/s | spans `U/c` from ≈ 1 to ≈ 5 |
| window | 5 d pre-storm, ~3 d passage, 15–20 d wake | close to the 30-day QG window |
| mixed layer | `h* = 30–60 m`, `τ_r = 10–30 d` | |

## 6. Configuration ladder

| id | components | coupling | purpose |
|---|---|---|---|
| **H0** | QG ocean, prescribed vortex, weak forcing | one-way, linear regime | check the balanced part of Geisler (1970) exactly (§9.3) |
| **H1** | QG ocean, prescribed vortex (fixed track and `V(t)`) | one-way | balanced wake in an eddy field (Q1) |
| **H2** | H1 + passive mixed layer | one-way | cold wake, eddy modulation of cooling; calibrate the closure |
| **H3** | H2 + intensity equation + gyrostat environment | **two-way** | the coupled case: Q2–Q5 |
| H-lo | vortex + 1D mixed layer + gyrostat, no QG | two-way | cheap low-order variant on the CGOA engine; close to Schade & Emanuel (1999) |
| H-sw | H3 with a two-layer shallow-water ocean | two-way | recovers Geisler's inertial wake; optional, large (§11, decision 2) |

## 7. Synthetic experiments and DA scenarios

### 7.1 Observation scenarios

| scenario | storm | ocean | analogue |
|---|---|---|---|
| **O-track** | `(x_c, y_c, V)` every 6 h, with noise | none | best-track data |
| **O-alt** | O-track | ψ₁ along-track (existing `alongtrack` geometry) | nadir altimetry |
| **O-swath** | O-track | ψ₁ in wide columns (existing `random_column` geometry) | SWOT |
| **O-sst-ir** | O-track | `T` on the grid, masked within ~300 km of the storm | infrared SST under clouds |
| **O-sst-mw** | O-track | `T` coarsened to ~25 km, masked in the core | microwave SST |
| **O-prof** | O-track | a few `(T, h, η)` profiles along the track | Argo, AXBT, gliders |
| **O-full** | all | all | reference |

### 7.2 Model-error scenarios

- **S0:** perfect model.
- **S1-tc-exchange:** biased `C_k/C_D`, hence a biased `V_p`.
- **S1-tc-mix:** biased entrainment coefficient `m₀`.
- **S1-tc-uncoupled:** the DA model runs the vortex at fixed SST, i.e. no
  feedback. This is the structural error of an uncoupled forecast system.
- **S1-tc-noml:** the DA model has no mixed layer; cooling comes from
  upwelling only.
- **S1-gyro-struct:** from the QG gyrostat note, applied to the environment.

The standing caveat of `docs/README.md` applies: these scenarios price model
error only for methods that run a forward model at inference.

### 7.3 Joint estimation

Estimate `(q, T, h, z_v, z_env)` and, for Q5, `(C_k/C_D, m₀)`. The repo
already runs joint state–parameter DA on L96. The SCDA/WCDA comparison uses
`block_localization` on `component_slices` (CGOA §5.2).

## 8. Towards real wind and ocean observations

### 8.1 What exists

| side | observation | resolution / coverage | maps to |
|---|---|---|---|
| storm | best track (IBTrACS; Knapp et al. 2010) | 6 h; position, `V_max`, radii | `z_v` |
| storm | reconnaissance: flight level, SFMR, dropsondes | during flights | `V`, `R_max`, wind profile |
| storm | scatterometers (ASCAT, HY-2), L-band radiometers (SMAP, SMOS), SAR (Sentinel-1, RCM), GNSS-R (CYGNSS) | swaths; SAR ~1 km; L-band for high winds | τ field, Holland parameters |
| storm | reanalysis winds (ERA5) | ~30 km, hourly | environment; biased low in the core |
| ocean | nadir altimetry, SWOT KaRIn swath | along-track; 2 km swath data since 2023 | ψ₁ (SSH) |
| ocean | microwave SST (AMSR2, GMI) | ~25 km, through clouds, not in rain | `T` |
| ocean | infrared SST, L4 analyses (GHRSST) | ~1–5 km, cloud-masked | `T` |
| ocean | Argo, air-launched floats, AXBT, gliders, uncrewed surface vehicles | sparse profiles | `T`, `h`, `η` |
| ocean | reanalyses (GLORYS, HYCOM) and heat-content products | daily, eddy-permitting | prior for `(q, η, h)` |

**Variable mapping.**

- ψ₁ ↔ SSH through `η_ssh = f₀ ψ₁ / g`.
- Interface displacement ↔ thermocline depth, e.g. the depth of the 20 °C or
  26 °C isotherm, and thus ocean heat content.
- `T`, `h` ↔ SST and mixed-layer depth.

### 8.2 Three problems

**(a) Reconstruction.** The closest to 4DVarNet's origins, which are SSH and
SST mapping from satellite data (Fablet et al. 2021).

- *Cold-wake mapping under clouds.* Fill SST gaps near the storm from
  microwave SST, SSH, profiles and the wind history. The wind history is the
  key extra input: the wake is largely determined by it.
- *Surface-wind reconstruction.* Fuse scatterometer, SAR, L-band and
  reconnaissance data with a parametric vortex prior. The Holland profile acts
  as a physics-based prior, or as a parameterized observation operator.
- *Post-storm ocean state.* Map the geostrophic wake and the mixed layer from
  SSH (SWOT swaths in particular) and profiles.

**(b) Forecasting.**

- *Intensity, conditioned on the ocean.* Does an estimated ocean state
  (interface depth, mixed layer) improve intensity forecasts at 1–5 days? The
  synthetic Q2/Q4 results say how much skill to expect and from which
  observations.
- *Wake evolution.* Forecast recovery of SST and the balanced wake over
  weeks.
- *Probabilistic output.* CFM and SDA give ensembles directly. That matters
  for rapid intensification, where the distribution, not the mean, is the
  product. Score with CRPS and reliability.

**(c) Calibration.**

- *Exchange coefficients.* `C_D` and `C_k` at high wind speed, and their
  ratio, which sets `V_p`, are poorly constrained (Bell, Montgomery & Emanuel
  2012).
- *Mixing.* The entrainment coefficient `m₀` and the closure itself.
- *Wind profile.* Holland `B`, and the translation fraction in the
  asymmetry.
- *Method.* Joint state–parameter estimation per storm, then pooling across
  storms. Emergent relations serve as calibration targets: cooling versus
  intensity, translation speed and pre-storm mixed-layer depth, as documented
  for real storms (e.g. Lloyd & Vecchi 2011).

### 8.3 Transfer protocol

The QG-TC ocean is idealized: doubly periodic, no coasts, no seasonal cycle,
no resolved mixed-layer dynamics. **Training on QG-TC and applying the model
to real data directly is not credible.** Three steps instead:

1. **OSSE on QG-TC.** Method development and ablations: window length across
   scales, SCDA vs WCDA, observation value, model error (§7).
2. **Realistic OSSE.** A reanalysis or coupled-model output is the nature run
   (ocean reanalysis plus reanalysis winds with a parametric vortex inserted,
   or a coupled hurricane model where available). Real observation
   geometries are sampled from it. The QG-TC components move into the method
   as priors:
   - the parametric vortex in the forcing or observation operator;
   - the mixed-layer closure as a physical constraint or a residual target;
   - QG-TC pretraining, then fine-tuning.
3. **OSE on real events.** Leave-one-sensor-out validation: withhold SWOT, or
   floats and gliders, or reconnaissance, and score against it.

**Candidate events** (data availability to be checked): Atlantic and Gulf of
Mexico storms that crossed documented warm or cold eddies, or with dense
in-situ sampling, e.g. Opal (1995), Katrina and Rita (2005), Ida (2021), Sam
(2021), Ian (2022), Idalia (2023) and Milton (2024). SWOT swath data are
available from 2023 onwards.

**Two regimes of data size.** Few storms have rich ocean observations. Many
have best track, scatterometer and altimetry. The protocol should use the
second group for training and the first for validation.

### 8.4 What does not transfer

- The inertial wake and inertial-shear mixing. They are absent from QG-TC; a
  real-data method must learn or parameterize them (or use H-sw).
- Salinity barrier layers, which modulate cooling in some basins.
- Coastal and shelf dynamics at landfall.

## 9. Implementation plan

**Principles** (as in CGOA §5):

- no change to QG defaults, gated by a bitwise test;
- one new dynamics class, no copy of the QG stack;
- DA and scoring go through the shared pieces that CGOA WP3 extracts;
- the new config enters the truth-cache key.

### 9.1 Files

| file | content |
|---|---|
| `models/qg_dynamics.py` | Minimal hook: `_tendency` accepts an optional precomputed upper-layer PV source in spectral space. `None` is the current path, bitwise. `wind_profile: "mexhat" \| "holland"` in the constructor, default `"mexhat"`. |
| `models/tc_vortex.py` (new) | Holland wind, stress with saturating `C_D`, curl on the grid (analytic, summed over periodic images like `wind_curl_field`), intensity equation, track, life-cycle envelope. Pure torch, differentiable. |
| `models/qg_mixed_layer.py` (new) | Slab mixed-layer tendencies from `(ψ₁, ψ₂, τ)`: advection by ψ₁, entrainment, restoring. Reuses the QG spectral operators for the Jacobian. |
| `models/gyrostat_driver.py` (new, shared) | The Option A driver of the QG gyrostat note. Written once, used by both notes. |
| `models/qg_tc_dynamics.py` (new) | `QGTCDynamics(DynamicsBase)`: augmented state (§5.5), RK4 co-stepping of all components, `coupling ∈ {"oneway", "twoway"}`, `component_slices`, field accessors (`ssh`, `sst`, `mld`, `interface`). In `"oneway"` mode, the vortex trajectory is precomputed, which reproduces the H1/H2 setting. |
| `data/qg_tc.py` (new) | `QGTCConfig` (extends `QGConfig`), event-window generation, §7.1 observation operators (reusing the existing along-track and column geometries), §7.2 scenarios. Emits the window-dict contract of `data/qg.py`; `forcing_true` stays the amplitude, the full vortex and environment state goes in a separate key. A `TC_VERSION` constant enters the cache key, like `OBS_GEOMETRY_VERSION`. |
| `evaluation/run_qg_tc_baselines.py` (new, thin) | Builds windows and methods only; the loop, per-component whitening, grouped scoring and block localization come from CGOA §5.2. If CGOA WP3 has not landed, this WP waits rather than copying `run_qg_baselines.py`. |
| `scripts/prepare_tc_obs.py` (later) | Real-data preparation (§8): IBTrACS, satellite and in-situ data onto a regional grid, in the same window-dict format. |
| `reports/qg_tc/` | Response diagnostics (§9.2) and DA reports, read through `evaluation.archive`. |

QG entry points are still argparse (`docs/plans/tech/qg_hydra_migration.md`).
New flags follow the existing pattern until that migration lands.

### 9.2 Diagnostics

- Interface ridge amplitude and width along the track, and its decay time.
- Cold-wake amplitude, width, rightward offset and recovery time.
- Cooling under the core versus `U/c`, `R_max/U` and pre-storm `h` and `η`.
- Intensity versus the ocean state crossed (Q2), with an ensemble of ocean
  initial conditions under identical storms (the Eₖ design of the QG
  gyrostat note).
- Wind work `⟨ψ₁ · curl τ⟩` and EKE per layer.

### 9.3 Tests

- **Default unchanged:** `wind_profile="mexhat"` and no extra PV source
  reproduce the current wind state and a short QG trajectory bitwise.
- **Geisler check (H0):** with β = 0, no mean shear, weak forcing and a
  steady translating vortex, the steady linear QG response in the storm frame
  is solvable by FFT. The model's interface displacement must match it after
  spin-up. This checks sign conventions, the stress-to-PV scaling and the
  periodic images at once.
- **Ekman pumping:** the diagnosed pumping under a reference vortex matches
  `curl τ/(ρ₀ f)`.
- **Mixed layer:** no wind → `T` and `h` relax to `T*`, `h*`; entrainment
  only cools and only deepens.
- **Feedback sign:** everything else fixed, a slower storm cools more and
  peaks lower.
- **Low-order variant:** H-lo written in `QuadraticTensorDynamics` matches
  the direct implementation.

### 9.4 Work packages

| WP | content | depends on |
|---|---|---|
| WP0 | This note. | — |
| WP1 | QG hook, `tc_vortex.py`, `QGTCDynamics` in one-way mode; H0 Geisler test; H1 runs. | — |
| WP2 | Mixed layer; H2; closure calibration against published cooling statistics. Results in `docs/results/`. | WP1 |
| WP3 | Intensity equation, gyrostat environment, two-way mode; H3; Q1–Q3 diagnostics. | WP2; QG gyrostat note WP1 |
| WP4 | `data/qg_tc.py` scenarios; DA baselines (ETKF/EnKF, weak 4D-Var) with SCDA/WCDA. | WP3; CGOA WP3 |
| WP5 | Neural rows (4DVarNet, CFM, SDA) through the QG neural path. | WP4 |
| WP6 | Real data, step 2: preparation script, realistic OSSE on a reanalysis nature run. | WP4 |
| WP7 | Real data, step 3: OSE on selected events, leave-one-sensor-out. | WP6 |
| WP8 | Optional: H-sw, a two-layer shallow-water ocean. | decision 2 |

WP1–WP2 can start now. WP6's data preparation can run in parallel with WP3–WP5.

## 10. Risks

| # | risk | mitigation |
|---|---|---|
| R1 | Resolution: the core is barely resolved at `nx = 64`. | `nx = 128` and a large `R_max`; check convergence of wake statistics at `nx = 256` on a few events. |
| R2 | The storm forcing is ~15× the current one; local velocities grow, and the filter or `clip_range` may act on the wake. | `dt = 1 h`; monitor filter dissipation; H0 test before H1. |
| R3 | QG validity: the local Rossby number under the core is not small. | Interpret only the balanced wake; H-sw as the check (WP8). |
| R4 | The mixed-layer closure drives the cooling results. | Calibrate against observed statistics (WP2); report conclusions that survive two closures. |
| R5 | Periodic re-entry of the storm or its wake. | Life-cycle envelope; one passage per window; wake decay checked against the window length. |
| R6 | Cache reuse across configurations. | `QGTCConfig` and `TC_VERSION` in the key. |
| R7 | Sim-to-real gap (§8.4). | The three-step protocol of §8.3; no direct toy-to-real claims. |
| R8 | Few storms with rich ocean data. | Train on the large sparse set, validate on the small rich set; pool across basins if needed. |

## 11. Open decisions

1. **Priority.** Physics of the coupled response (Q1–Q3), the synthetic DA
   benchmark (Q4–Q5), or the real-data track (§8)? They share WP1–WP3.
2. **Ocean model.** Stay with QG and accept the missing inertial wake, or
   plan H-sw (a two-layer shallow-water ocean, Geisler's own model) as a
   second ocean?
3. **Grid.** A fixed basin grid (current) or a storm-following grid? The
   second suits real data and resolution but changes the QG code more.
4. **First real-data problem.** Reconstruction is recommended: it is closest
   to 4DVarNet's SSH/SST heritage, and its observations are the most
   abundant. Calibration second, intensity forecasting third.
5. **Paper fit.** Q4–Q5 and the S1-tc scenarios fit P1 (model error) and the
   CGOA coupled-DA questions. The real-data reconstruction could be a
   separate application paper.

## References

- Bell, Montgomery & Emanuel (2012), air–sea enthalpy and momentum exchange at major hurricane wind speeds (CBLAST). *J. Atmos. Sci.* 69, 3197–3222.
- Emanuel (1986), an air–sea interaction theory for tropical cyclones. *J. Atmos. Sci.* 43, 585–605.
- Emanuel (1988), the maximum intensity of hurricanes. *J. Atmos. Sci.* 45, 1143–1155.
- Fablet et al. (2021), learning variational data assimilation models and solvers. *J. Adv. Model. Earth Syst.* 13, e2021MS002572.
- Geisler (1970), linear theory of the response of a two layer ocean to a moving hurricane. *Geophys. Astrophys. Fluid Dyn.* 1, 249–272.
- Gill (1984), on the behavior of internal waves in the wakes of storms. *J. Phys. Oceanogr.* 14, 1129–1151.
- Holland (1980), an analytic model of the wind and pressure profiles in hurricanes. *Mon. Wea. Rev.* 108, 1212–1218.
- Knapp et al. (2010), the International Best Track Archive for Climate Stewardship (IBTrACS). *Bull. Amer. Meteor. Soc.* 91, 363–376.
- Kraus & Turner (1967), a one-dimensional model of the seasonal thermocline II. *Tellus* 19, 98–106.
- Lloyd & Vecchi (2011), observational evidence for oceanic controls on hurricane intensity. *J. Climate* 24, 1138–1153.
- Mainelli, DeMaria, Shay & Goni (2008), application of oceanic heat content estimation to operational forecasting of recent Atlantic category 5 hurricanes. *Wea. Forecasting* 23, 3–16.
- Powell, Vickery & Reinhold (2003), reduced drag coefficient for high wind speeds in tropical cyclones. *Nature* 422, 279–283.
- Price (1981), upper ocean response to a hurricane. *J. Phys. Oceanogr.* 11, 153–175.
- Price, Sanford & Forristall (1994), forced stage response to a moving hurricane. *J. Phys. Oceanogr.* 24, 233–260.
- Schade & Emanuel (1999), the ocean's effect on the intensity of tropical cyclones: results from a simple coupled atmosphere–ocean model. *J. Atmos. Sci.* 56, 642–651.
- Shay, Goni & Black (2000), effects of a warm oceanic feature on Hurricane Opal. *Mon. Wea. Rev.* 128, 1366–1383.
- Companion notes: `docs/plans/case_study/qg_gyrostat_synoptic_forcing.md`, `docs/plans/case_study/cgoa_coupled_gyrostats.md`.
