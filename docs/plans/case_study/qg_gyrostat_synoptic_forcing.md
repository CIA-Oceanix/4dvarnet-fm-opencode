# Upper-ocean response to synoptic atmospheric variability: gyrostat-driven forcing for the QG case study

**Status:** SCOPING v1 (2026-09-25). Nothing implemented. Written to be read
without the rest of the repo. It extends the existing QG case study; it is
**not** a new testbed. Companion to
`docs/plans/case_study/cgoa_coupled_gyrostats.md`, which proposes a fully
coupled gyrostat ocean–atmosphere case study.

---

## 1. Summary

The QG case study is a two-layer quasi-geostrophic ocean driven by wind. The
wind is a **prescribed, kinematic storm**: its amplitude and position follow
simple Gaussian random processes, and the ocean never acts back on it. This
note proposes replacing that storm generator with a **low-order chaotic
atmosphere built from coupled gyrostats**, then using the pair to study how
the upper ocean responds to synoptic atmospheric variability.

Five options are laid out, from a drop-in change to a two-way coupled system
(§4). **Recommendation:** start with Option A, a one-way gyrostat driver that
feeds the existing storm parameters. Pair it with a
**phase-randomized surrogate** of the same forcing (§5). That control has the
same power spectrum but Gaussian statistics. The ocean's response to the two
then isolates the effect of forcing *structure* (regimes, intermittency,
non-Gaussianity) from the effect of forcing *energy*.

The existing QG benchmark stays unchanged: every option is a new setting, and
the current storm stays the default.

## 2. The QG case study today

**Ocean.** A native torch port of pyqg v0.4.0 (`models/qg_dynamics.py`):

| quantity | value |
|---|---|
| domain | doubly periodic, 1000 km × 1000 km, `nx = 64` (Δx ≈ 15.6 km) |
| layers | 2, thickness ratio `δ = 0.25` |
| deformation radius | `rd = 15 km` |
| mean shear | `U1 = 5 cm/s`, `U2 = 0` (baroclinically unstable) |
| other physics | `β = 1.5e-11 m⁻¹s⁻¹`, linear bottom drag, pyqg exponential filter |
| time step | 2 h, RK4 |

**Wind forcing** (`models/qg_dynamics.py:29-39`, `generate_wind_state`,
`wind_curl_field`). A single localized storm enters as an upper-layer PV
source, `dq₁/dt += curl τ(x, y, t)`:

- **Shape:** a Mexican-hat profile of width `σ = 250 km`, summed over periodic
  images.
- **Amplitude:** an Ornstein–Uhlenbeck (OU) process, i.e. a Gaussian random
  process relaxing to zero. Its scale is `1e-11 s⁻²` and its memory is 15 days.
- **Position:** a mean drift of (0.5, 0.03) m/s, which gives a ~23-day zonal
  crossing of the basin. On top of it, an OU jitter of 50 km with a 10-day
  memory.

**Data assimilation setup** (`data/qg.py`):

- 30-day windows.
- Upper-layer observations every 12 h, in along-track or random-column
  geometry.
- Scenarios:
  - **S0:** perfect model.
  - **S1:** biased physical parameters (15%) plus a corrupted storm: location
    jitter of 0.25σ, amplitude bias of 15% plus an OU error.
  - **S1-qg1l:** a reduced-gravity one-layer DA model, a structural error.

**Limitations of the forcing** for the questions in §3:

1. **Gaussian and linear.** An OU process has no regimes, no intermittency and
   no skewness. A real storm track has all three.
2. **Slow.** Memory times of 10–15 days and a 0.5 m/s translation are slow
   compared with real synoptic systems: 2–7 days, and roughly 10 m/s.
   This was a deliberate choice, made so storms act on ocean time scales.
3. **One-way.** The ocean never acts back on the atmosphere.
4. **Unstructured forcing error.** S1 corrupts the forcing with independent
   noise. Real forcing error comes from an atmospheric model with the wrong
   dynamics.

## 3. Scientific questions

- **Q1 — Structure vs energy.** At a fixed forcing spectrum, does the
  upper-ocean response depend on higher-order forcing statistics (regimes,
  intermittency)? Where does it show: in ψ₁, in q₁, in the lower layer?
- **Q2 — Frequency response.** Which forcing time scales reach the upper
  layer, and which are filtered out? This is the ocean acting as an
  integrator, as in Hasselmann's stochastic climate model (1976) and
  Frankignoul & Hasselmann (1977).
- **Q3 — Rectification.** How does synoptic forcing turn into mesoscale-eddy
  and low-frequency variability? Diagnostics: wind work, eddy kinetic energy
  (EKE), energy transfer across scales.
- **Q4 — Feedback.** If the wind stress is computed from the wind *relative to
  the surface current*, the ocean damps its own eddies ("current-feedback eddy
  killing", Renault et al. 2016). How large is that effect in this setup, and
  does it feed back on the atmosphere's variability?
- **Q5 — Data assimilation.** With a dynamical atmosphere:
  - forcing error becomes *structured*;
  - joint estimation of ocean state and atmospheric state becomes meaningful;
  - one can ask whether ocean observations constrain the atmosphere at all.

## 4. Options

All options keep the two-layer QG ocean as is.

### Option A — One-way gyrostat driver, parametric mapping (recommended first)

The gyrostat atmosphere drives the **existing** storm parameters
`(A, x_c, y_c)`. `wind_curl_field` and everything downstream stay untouched.

- **Atmosphere.** A small system of coupled gyrostats, chaotic and bounded
  because the gyrostat core conserves energy. Candidates:
  - a single Lorenz-63 gyrostat (3 modes);
  - a sparse chain of 2–4 gyrostats (5–9 modes), which allows several
    regimes with different residence times.

  The time unit is set in days: this is the synoptic time-scale control (§9,
  decision 3).
- **Mapping** (one natural choice among several; see R2):
  - **Amplitude:** `A(t) = a · y_A(t)`, a signed, zero-mean mode, like the
    current OU amplitude.
  - **Zonal translation:** `ẋ_c = c_x (1 + b · y_U(t))`, so storm speed
    depends on the regime.
  - **Storm-track latitude:** `y_c = ȳ + d · y_Y(t)`. This mimics shifts of the
    North Atlantic Oscillation (NAO) type.
- **Calibration.** Scale `a` so that `Var(A)` and the amplitude's memory time
  match the OU default. Then the **energy** input is comparable and only the
  **structure** differs.
- **Code change.** Small; §7 has the sketch.

### Option B — One-way gyrostat driver, spectral mapping

Gyrostat modes set the amplitudes of `K` fixed wind-curl patterns. The
natural choice is low-order doubly periodic Fourier modes: zonal wavenumbers
1–4, meridional 1–2, as in MAOOAM's atmospheric basis.

- A **ring** of gyrostats gives a travelling storm track with several storms
  at once, and a spatial spectrum of forcing rather than a single blob.
- This is closer to the physics, but the wind state grows from 3 numbers to
  `K`. Code that reads `forcing_true` (currently the storm amplitude alone)
  and the joint-estimation paths would need to follow.

### Option C — Two-way mechanical coupling

Add ocean-to-atmosphere and ocean-to-ocean feedback through the relative wind:

- **In the ocean:** stress computed from `u_a − u_o`. To leading order this
  adds a damping of upper-layer relative vorticity, `−r_cf ζ₁`, to the PV
  equation. This is the eddy-killing term of Q4.
- **In the atmosphere:** project the ocean's surface streamfunction onto the
  atmospheric patterns and add the momentum-conserving drag of
  `docs/plans/case_study/cgoa_coupled_gyrostats.md` §3.3 to the gyrostat
  modes. The two-way exchange is then energy-consistent: it dissipates
  `γ I_a (y_a − y_o)²` and creates no energy.
- **Requires Option B.** Feedback needs a spatial projection, which Option A's
  three storm parameters don't provide.

### Option D — Gyrostat reduced model of the ocean itself

Truncate the two-layer QG equations onto a few Fourier modes, in a way that
keeps energy conserved, and write the result as coupled gyrostats
(Gluhovsky's construction). Two uses:

1. **Analysis.** Identify which mode triads carry energy from the forced
   modes to large scales. This gives Q3 an analytic counterpart.
2. **DA model.** Use the reduced model as the DA model while the full QG
   system is the truth. This is a new, extreme structural-error scenario, in
   the same family as the existing `qg1l` scenario.

**Caveat.** At `rd = 15 km` and `Δx ≈ 15.6 km` the ocean has an active
mesoscale-eddy field that a few modes cannot represent. As a forecast model
it would need a stochastic closure.

### Option E — Thermal coupling (deferred)

The QG ocean carries PV only; it has **no sea-surface temperature (SST)**.
Heat-flux coupling would need an added mixed-layer SST tracer, advected by the
upper-layer flow and relaxed towards an atmospheric temperature mode. That is
a real model extension, and it moves the setup towards MAOOAM (CGOA Tier B).
Listed for completeness; not proposed now.

### Comparison

| option | code change | answers | benchmark impact | main risk |
|---|---|---|---|---|
| **A** parametric driver | small: generator swap | Q1, Q2, Q5 | none (new setting) | mapping is a modelling choice (R2) |
| **B** spectral driver | moderate: new curl builder, wider wind state | Q1–Q3, Q5 | none (new setting); touches `forcing_true` | more parameters to calibrate |
| **C** two-way mechanical | moderate on top of B | Q4 | new scenario | size of feedback coefficient |
| **D** ocean reduced model | large: derivation + new DA model | Q3, P1 model error | new S1 scenario | the reduced model misses the eddies |
| **E** thermal | large: new tracer | heat-flux feedback | new scenario | scope creep toward MAOOAM |

## 5. Response-analysis protocol (Options A/B)

**Experiments.** Same ocean, same spinup and truth-cache machinery, long runs
of several model years after spinup.

| id | forcing | role |
|---|---|---|
| E0 | current OU storm | reference; existing benchmark |
| E1 | gyrostat driver | the new forcing |
| **E2** | **phase-randomized surrogate of E1's forcing** | **key control:** same power spectrum as E1, Gaussian statistics (Theiler et al. 1992) |
| E3 | E1 with a faster synoptic time unit (2–7 d) | sensitivity to the time-scale choice |
| Eₖ | E1 with an ensemble of ocean initial conditions, identical forcing | separates **forced** response (ensemble mean) from **intrinsic** variability (spread) |

**Diagnostics.**

- Frequency spectra of ψ₁, q₁ and ψ₂, at points and averaged over the domain.
- Coherence and transfer function from forcing to response.
- Wind work, `⟨ψ₁ · curl τ⟩`, and EKE per layer.
- PDFs and kurtosis of ψ₁ and q₁.
- Composites of the ocean state around atmospheric regime transitions.
- Impulse responses from the tangent-linear model. We already differentiate
  the QG model with autograd, so this comes for free.

**Reading the results.**

- **E1 vs E2** answers Q1. A difference means forcing structure matters.
- **E0 vs E2** measures what moving from the OU spectrum to the gyrostat
  spectrum does.
- **Eₖ** says how much of any difference is signal and how much is intrinsic
  ocean variability (R3).

## 6. Data-assimilation angle (Q5)

- **Default unchanged.** The current OU storm stays the QG benchmark default,
  and the published S0/S1 numbers stay valid. Every new setting is a new
  scenario, and all methods in a comparison use the same driver.
- **New model-error scenarios** on the P1 axis:
  - **S1-gyro-param:** truth and DA model both use the gyrostat driver, but
    the DA model's gyrostat parameters are biased.
  - **S1-gyro-struct:** the truth uses the gyrostat driver; the DA model uses
    an OU storm with matched moments. Its forcing model is right in energy and
    wrong in structure.
- **Joint estimation.** Estimate the gyrostat atmosphere's state together with
  the ocean state. This asks directly whether ocean observations constrain the
  atmosphere; the expected answer is weakly, and the size of that effect is a
  result in itself. The repo already has joint state-and-parameter DA on L96.
- **Standing caveat** (from the P1 scoping): model-error scenarios price the
  error only for methods that run a forward model at inference. For
  model-free neural schemes the S1/S0 ratio is definitional.

## 7. Implementation sketch (Option A)

| file | change |
|---|---|
| `models/qg_dynamics.py` | `wind_driver: "ou" \| "gyrostat"`; `generate_wind_state` dispatches on it and keeps its `(A, x_c, y_c)` output. |
| `models/gyrostat_driver.py` (new) | Integrates the gyrostat atmosphere in days and maps modes to `(A, x_c, y_c)`. Moment-matching helper. Phase-randomized surrogate generator. Reuses the CGOA gyrostat builder once that exists; Option A does not wait for it, since a small gyrostat is a few lines of RK4. |
| `data/qg.py` | `QGConfig` fields for the driver. **Fold the driver and its parameters into the truth-cache key**, as `OBS_GEOMETRY_VERSION` is, so a driver change cannot silently reuse the OU trajectories. |
| QG entry points | New CLI flags. QG is still argparse; see `docs/plans/tech/qg_hydra_migration.md`. |
| `reports/qg/` | Response-analysis script for §5. |

**Tests:**

- **Default unchanged:** `wind_driver="ou"` reproduces the current wind state
  and a short QG trajectory **bitwise**.
- **Gyrostat core:** energy conserved without forcing and damping.
- **Moment matching:** the driver's amplitude variance and memory time match
  OU within tolerance.
- **Surrogate:** the phase-randomized surrogate preserves the power spectrum.

## 8. Work packages

| WP | content |
|---|---|
| WP1 | Option A + surrogate generator + tests. No change to defaults. |
| WP2 | E0–E3 and Eₖ runs; response diagnostics; a results doc in `docs/results/`. |
| WP3 | Option B, spectral driver; repeat E1/E2. |
| WP4 | Option C, relative-wind feedback; Q4. |
| WP5 | Option D, ocean gyrostat reduced model: analysis first, DA scenario second. |
| WP6 | DA scenarios S1-gyro-param and S1-gyro-struct, plus joint estimation. Can start right after WP1. |

Option E is deferred.

## 9. Risks

| # | risk | mitigation |
|---|---|---|
| R1 | Changing the forcing's time scale changes the ocean regime, and results get confounded with it. | Keep E1 at the current storm time scales. Treat the realistic synoptic time scale as a separate sensitivity (E3). |
| R2 | The mapping from gyrostat modes to storm parameters is a modelling choice; a result could be an artefact of it. | The E2 surrogate controls for spectrum. Test two or three mappings; report only conclusions that survive all of them. |
| R3 | The ocean is baroclinically unstable, so intrinsic eddy variability can swamp the forced response. | The Eₖ ensembles separate forced from intrinsic variability. Measure signal-to-noise before interpreting. |
| R4 | Cached truth trajectories are silently reused across drivers. | Driver in the cache key (§7). |
| R5 | Option B widens `forcing_true` and breaks consumers that assume the amplitude alone. | Audit consumers in WP3. Keep `forcing_true` as the amplitude and store the full wind state separately. |

## 10. Open decisions (for discussion)

1. **Priority.** Physics of the response (Q1–Q4), or the DA benchmark (Q5)?
   The two share WP1 and then diverge.
2. **Option order.** A → B → C is proposed. Is D, the ocean reduced model,
   of interest early, for its analytic value?
3. **Synoptic time scale.** Keep the current slow storm for comparability, or
   move to realistic 2–7-day synoptic time scales as the main setting?
4. **Paper fit.** The new S1-gyro scenarios would feed P1 (model error). The
   response physics could be a separate, shorter oceanographic note.
5. **Link to CGOA.** Implement the gyrostat atmosphere once, in the CGOA
   engine, and have both efforts share it? Or keep Option A standalone so it
   is not blocked?

## References

- Hasselmann (1976), Stochastic climate models, Part I. *Tellus* 28, 473–485.
- Frankignoul & Hasselmann (1977), Stochastic climate models, Part II. *Tellus* 29, 289–305.
- Renault et al. (2016), Modulation of wind work by oceanic current interaction with the atmosphere. *J. Phys. Oceanogr.* 46, 1685–1704.
- Theiler et al. (1992), Testing for nonlinearity in time series: the method of surrogate data. *Physica D* 58, 77–94.
- Lorenz (1963), Deterministic nonperiodic flow. *J. Atmos. Sci.* 20, 130–141.
- Gluhovsky & Tong (1999), *Phys. Fluids* 11, 334; Gluhovsky (2007), *J. Nonlinear Sci.*; Gluhovsky (2017), *Complexity* 6176045.
- Seshadri & Lakshmivarahan (2023), *Physica D* 133948; arXiv 2503.10782 (2026).
- De Cruz, Demaeyer & Vannitsem (2016), MAOOAM v1.0, *Geosci. Model Dev.* 9, 2793–2808.
- pyqg: https://github.com/pyqg/pyqg
