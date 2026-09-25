# Gyrostat spectral wind forcing for the QG case study — implementation and response-analysis plan

**Status:** DRAFT v2 (2026-09-26). Nothing implemented. This plan implements
**Option B** of `docs/plans/case_study/qg_gyrostat_synoptic_forcing.md`
directly: gyrostat modes drive the amplitudes of fixed 2D wind-curl
patterns (low-order Fourier modes). It skips Option A. It also covers the
response analysis built around a phase-randomized surrogate (scoping WP2).
The driver is written so that QG-TC
(`docs/plans/case_study/qg_hurricane_coupled.md`) and CGOA
(`docs/plans/case_study/cgoa_coupled_gyrostats.md`) can reuse it. Out of
scope, apart from the seams in §9: Option C (two-way feedback) and the DA
scenarios (scoping WP6).

> **v2 vs v1.** v1 planned Option A (the gyrostat drives one storm's
> amplitude and position). v2 goes straight to Option B, which is also the
> prerequisite for Option C's feedback. A measurement made for v2 (§3)
> shows that today's storm forcing already lives almost entirely in 12
> Fourier modes, so Option B **contains** the current forcing rather than
> replacing it. The Lorenz-63 time-scale measurements of v1 are kept (§4).

---

## 1. Goal and decision

**Goal.** Add spectral wind drivers to the QG case study. The PV source
becomes

```
curl τ(x, y, t) = Σₖ aₖ(t) φₖ(x, y)
```

a sum of fixed patterns `φₖ` with time-varying amplitudes `aₖ(t)`. The
patterns are the real parts (cos and sin) of the Fourier modes with
`|k| ≤ K`. The drivers differ only in how `a(t)` is generated:

| driver | amplitudes `a(t)` |
|---|---|
| `"ou"` (default, unchanged) | the current Mexican-hat storm, `(A, x_c, y_c)` |
| `"spectral_ou"` | independent OU processes per mode, same spatial spectrum as the storm |
| `"gyrostat"` | a gyrostat system, mapped to the modes, same spatial spectrum |
| `"gyrostat_surrogate"` | multivariate phase-randomized surrogate of the gyrostat series |

All three new drivers share **the same spatial spectrum** (variance per
mode), calibrated on today's storm (§5.3). They differ only in **temporal
and cross-mode structure**.

**Requirements.**
- The default path and every existing cache key stay **bit-identical**.
- DA models, the S1 corruption, the neural forcing channel and the
  psi-state wrappers work with the wider wind state.

**Decision the plan produces (§7.4).** Does the structure of the forcing
(its regimes and the coherence between modes) change the upper-ocean
response measurably beyond intrinsic variability?
- **Yes:** proceed to Option C (feedback) and the DA scenarios.
- **No:** record a negative result, keep the drivers, and move the gyrostat
  work to QG-TC.

## 2. What the code does today (verified on master @ c92b9b3)

**Storm generation.**
- `QGDynamics.generate_wind_state` (`models/qg_dynamics.py:156`) integrates
  three OU processes step by step: the amplitude `A` (memory 15 days,
  stationary std `wind_amp`) and position jitters (10 days, 50 km). The
  centre drifts at `(wind_cx, wind_cy)`.
- Output: a `(T, 3)` tensor `(A, x_c, y_c)`.
- `wind_curl_field` (`:185`) evaluates a Mexican hat of width
  `wind_sigma = 250 km`, summed over periodic images.
- `_wind_curl_spectral` (`:205`) FFTs it inside `_tendency` (`:222`).

**Code that assumes a 3-component wind state** (must be generalized):

| place | assumption |
|---|---|
| `models/qg_dynamics.py:160, 287, 312` | `torch.zeros(steps, 3)` default wind state |
| `models/qg_dynamics.py:187` | `wind_state[..., 0]` is the amplitude |
| `models/qg_dynamics.py` `_wind_curl_spectral` | skips the forcing when `wind_state_t[0] == 0`, which would wrongly drop a spectral state whose first mode is zero |
| `models/qg1l_dynamics.py:145, 172, 271, 289` | same, in the one-layer model: **the S1-qg1l DA model consumes the truth's wind state** |
| `models/qg_psi_dynamics.py:145, 218` | `torch.zeros(steps, 3)` default |
| `data/qg.py:392` `_make_corrupted_wind_state` | columns are `(A, x_c, y_c)` |
| `data/qg.py:661–662` | `forcing_true = ws[:, 0]` |

**Consumers that are already shape-agnostic.**
- `WindStateAdapter.step` (`evaluation/run_qg_baselines.py:35`) forwards a
  row of the wind state.
- The neural forcing channel is the daily-mean `wind_curl` field
  (`data/qg_neural.py`), so it follows `wind_curl_field`.

To verify in PR-1: `WindStateAdapter.rollout` indexes `forcing[..., k − 1]`,
i.e. the last axis. That suggests it is not on the QG path (QG uses
`rollout_trajectory`), but it should be confirmed before relying on it.

**Truth generation** (`QGS01Dataset._generate_truth_only`, `data/qg.py:446`).
Per window:
- random `rd, U1, rek` (±15%), start position `(x0, y0)`, drift
  `cx ∈ [0.25, 0.75]`, `cy ∈ [−0.06, 0.06]` m/s;
- an amplitude level cycling over `_S1_WIND_LEVELS = (0, 3e-12, 1e-11,
  2e-11, 3e-11)` (`data/qg.py:159`), so one window in five has no wind;
- **2 years of unforced spinup**, then a forced 10-day lead and 30-day window.

**Caches.**
- `_truth_cache_path` (`data/qg.py:666`) hashes `asdict(cfg)`.
- `_truth_only_cache_path` hashes `_TRUTH_ONLY_CFG_FIELDS` (`data/qg.py:687`).
- **A new `QGConfig` field changes the first hash for every existing config,
  even at its default value.** That would silently invalidate all cached
  truth, at about 4.5 min per window to regenerate. §6.4 avoids it.

## 3. Today's storm in Fourier space (measured)

The measurement used `QGDynamics.wind_curl_field` on the 64 × 64, 1000 km
grid, at 64 random storm positions, averaging the power spectrum of the
curl (the mean is exactly zero):

| basis | independent real amplitudes | share of curl variance |
|---|---|---|
| `|k| ≤ 1` | 4 | 74.0% |
| `|kx|, |ky| ≤ 1` | 8 | 99.1% |
| **`|k| ≤ 2`** | **12** | **99.8%** |
| `|k| ≤ 3` | 28 | 100.0% |

The radial spectrum peaks at domain wavenumber 1.

**Reading.** On a 1000 km periodic domain, the "storm" is not a localized
synoptic system. The positive core of a σ = 250 km Mexican hat reaches
r = √2 σ ≈ 350 km, and its negative ring extends further, so the forcing is
effectively a **domain-scale pattern that translates**. Consequences:

1. A 12-amplitude basis (`|k| ≤ 2`) reproduces today's forcing almost
   exactly. The current storm is one particular trajectory in that space, in
   which the modes are phase-locked into a coherent blob.
2. Every driver can therefore share one spatial spectrum. Comparisons then
   isolate temporal and cross-mode structure.
3. **Synoptic-scale structure (250–500 km) is absent today.** Adding it
   needs `|k| ≤ 4` (48 amplitudes) and a target spectrum that has power
   there (decision 1).

The scoping note describes the forcing as "a single localized storm"; this
measurement qualifies that description.

## 4. Lorenz-63 time scales (measured in v1, kept)

Standard parameters (10, 28, 8/3), RK4 with `dt = 0.005`, 2000 units after
burn-in:

| mode | std | ACF e-folding (units) | kurtosis |
|---|---|---|---|
| x | 7.92 | 0.305 | 2.31 (bimodal) |
| y | 9.02 | 0.220 | 2.86 |
| z | 8.64 | 0.140 | 2.12 (oscillatory ACF) |

- **Lobe residence:** mean 1.72 units, median 1.53.
- **Implication for Option B:** these numbers transfer only approximately to
  the gyrostat chain proposed in §5.2. Its time scales must be measured the
  same way once the chain is fixed (WP-B1). The time-unit decision
  (decision 4) is then made on the chain's numbers, not on Lorenz-63's.

## 5. Design

### 5.1 Basis — `models/qg_wind_modes.py` (new)

`FourierWindBasis(nx, ny, L, W, kmax, shape)`, where `shape` is `"disk"`
(`|k| ≤ kmax`) or `"square"` (`|kx|, |ky| ≤ kmax`).

- **Modes.** One representative of each ± pair of wavevectors (half-plane),
  giving `n_amp = 2 × n_wavevectors` real amplitudes (cos and sin).
  `|k| ≤ 2` gives 6 wavevectors and 12 amplitudes.
- **`curl_spectral(a)`** writes the amplitudes straight into the `rfft2`
  coefficients (with the conjugate symmetry the half-spectrum needs). This
  is **exact and cheap**: no grid evaluation and no FFT per step. It is the
  hot path inside `_tendency`.
- **`curl_field(a)`** gives the grid field (`irfft2` of the above), for
  stored `wind_curl` and the neural forcing channel.
- **`project(field)`** does the reverse, grid to amplitudes. It is used in
  the tests and later by Option C to project the ocean's surface ψ₁.
- **`translate(a, dx, dy)`** is the spatial shift expressed as a rotation of
  each (cos, sin) pair by `k·dx + l·dy`. It is used for the storm-track
  drift (§5.3) and for the S1 location error (§6.3).
- **`storm_mode_std(sigma, amp)`** gives the per-mode std of today's storm at
  level `amp`, averaged over positions. It is computed once from
  `wind_curl_field` and cached as a constant. This is the target spatial
  spectrum.

### 5.2 Gyrostat system — `models/gyrostat_driver.py` (new)

- **`GyrostatSystem(spec)`** is a gyrostat right-hand side,
  `ẏ = N(y, y) + L y + F − κ y`, with `energy(y)` for the conservation test.
  Presets:
  - `"l63"`: Lorenz-63 in gyrostat form (`(p, q, r) = (0, −1, 1)`,
    `c = −σ`, damping `(σ, 1, β)`, forcing `−β(ρ + σ)` on the shifted z).
    Used for tests and as a small-system check.
  - `"chain6"`: a sparse nested chain of 6 gyrostats, `M = 13` modes (the
    Seshadri–Lakshmivarahan hierarchy). Each gyrostat has L63-like
    coefficients, and neighbours share one mode. **13 modes cover 12
    amplitudes plus one spare** (§5.3). Chaos and time scales are measured
    in WP-B1, not assumed (CGOA risk R1).
- **`integrate(n_steps, dt_days, time_unit_days, seed, burnin_units=50)`**:
  seeded initial condition, burn-in onto the attractor, then RK4 with
  internal substeps so the gyrostat step stays ≤ 0.005 units.
- **`phase_randomized_surrogate(y, seed)`** is the multivariate surrogate of
  Prichard & Theiler (1994):
  - apply the **same** random phases to every channel, which keeps all auto-
    and cross-spectra;
  - make the series about 4× longer than needed and keep the middle, to avoid
    periodic-edge effects;
  - the marginals are Gaussian by construction.

### 5.3 From gyrostat modes to pattern amplitudes

For window `i`, with its existing draws `(x0, y0, cx, cy, amp)`:

1. **Standardize** each gyrostat mode by its attractor mean and std. These
   are constants measured once per preset and stored with it.
2. **Assign modes to amplitudes.** The default is the identity: gyrostat
   modes 1–12 go to the 12 amplitudes, ordered so that each cos/sin pair is
   fed by **neighbouring** chain modes. Neighbours are correlated through
   their shared mode, which gives the pairs a coherent but not locked phase
   relation. An alternative is a fixed orthogonal map `W` (seeded), selected
   by name through `mapping`, so that R2's mapping sensitivity is a config
   change.
3. **Set the spatial spectrum.** `aₖ = storm_mode_std(σ, amp)ₖ · ỹₖ`. At
   `amp = 0`, `a ≡ 0` exactly.
4. **Drift.** Rotate each pair by `k·(x0 + cx·t) + l·(y0 + cy·t)`. The
   per-window drift of today's storm is kept, so the pattern translates as
   the storm does.
5. **Optional speed modulation:** the 13th mode modulates the drift,
   `cx·(1 + b·ỹ₁₃)`. Off by default.

`"spectral_ou"` uses the same steps 3–4, with step 2 replaced by independent
unit-variance OU processes of memory `wind_tau_days`.

### 5.4 Hooks in the dynamics

- **`QGDynamics`:**
  - new constructor arguments `wind_driver: str = "ou"` and
    `wind_modes: dict | None` (basis and gyrostat settings);
  - a `wind_state_dim` attribute: 3 for `"ou"`, `n_amp` otherwise;
  - `generate_wind_state`, `wind_curl_field` and `_wind_curl_spectral`
    dispatch on the driver. The `"ou"` bodies are untouched;
  - the zero-state shortcut becomes "all components zero" for the spectral
    drivers;
  - the default `zeros(steps, 3)` becomes `zeros(steps, self.wind_state_dim)`,
    which is still 3 by default, hence bit-identical.
- **`QG1LDynamics`:** the same three dispatches and the same attribute, via
  the shared `FourierWindBasis`. This is needed because S1-qg1l hands it the
  truth's wind state. Its OU storm code is not refactored (scoped diff).
- **`QG1LPsiDynamics` / `QGPsiDynamics`:** default zeros of width
  `inner.wind_state_dim`.

## 6. Data, S1 and caches — `data/qg.py`

### 6.1 Config

`QGConfig` gains:
- `wind_driver: str = "ou"`;
- `wind_kmax: int = 2` and `wind_basis_shape: str = "disk"`;
- `gyro_preset: str = "chain6"`, `gyro_time_unit_days: float` and
  `gyro_mapping: str = "identity"`;
- `gyro_speed_mod: float = 0.0` and `gyro_burnin_units: float = 50.0`.

### 6.2 Truth generation

`_generate_truth_only` passes the new settings to `QGDynamics`. The
per-window draws and seeds are **unchanged**, so windows with the same index
share ocean parameters, start phase, drift and energy level across drivers.
`wind_state_true` becomes `(T, n_amp)` for the spectral drivers.

### 6.3 S1 corruption for spectral states

`_make_corrupted_wind_state` dispatches on `cfg.wind_driver` and keeps
**the same S1 parameters**, so the S1 semantics stay comparable:

- **Amplitude:** `aₖ ← aₖ (1 + s1_amp_bias) + ηₖ`, where `ηₖ` is an OU process
  with memory `s1_tau_days` and std `s1_sigma_eta_frac × std(aₖ)`.
- **Location:** a coherent shift of the whole field by the same OU series
  `(e_x, e_y)` as today (std `s1_loc_sigma_frac × wind_sigma = 62.5 km`),
  applied with `translate`.

`data/qg_neural.py`'s "jitter" forcing mode calls this function, so it
follows automatically.

`forcing_true` / `forcing_corrupted` are scalars per step today (the storm
amplitude). For spectral drivers they become the **domain-rms curl**. It is
unsigned, and documented as such. The full state stays in
`wind_state_true` / `wind_state_corrupted`. No QG consumer reads these
scalars (checked: `data/qg_neural.py`, `train_qg_neural.py`,
`eval_qg_neural_s0_s1.py`, `evaluation/qg_runs.py`).

### 6.4 Cache keys

- Both hashes **drop every new field when `wind_driver == "ou"`**, so every
  existing key is unchanged.
- Otherwise they add the fields plus a `WIND_DRIVER_VERSION` constant, in the
  same way as `OBS_GEOMETRY_VERSION`.
- Tests pin the current key strings for the default config.

## 7. Response analysis

### 7.1 Long forced runs

The benchmark truth is forced for only 40 days after an unforced spinup, so
it never reaches equilibrium with the forcing. The analysis therefore uses
separate long runs, in `reports/qg/run_wind_response.py`:

- nominal ocean, amplitude level `1e-11`;
- 2 years unforced spinup, 1 year forced and discarded, **5 years analysed**
  (~26k steps);
- daily snapshots of ψ₁, ψ₂ and q₁; hourly amplitudes; per-step domain
  integrals;
- ensembles of 10 ocean initial states under **identical** forcing, batched
  on GPU;
- snapshots and members go to **node-local `/tmp`** (the standing rule since
  2026-09-24). Only reduced diagnostics are kept, in `reports/qg/outputs/`.

### 7.2 Experiments

| id | driver | question |
|---|---|---|
| E0 | `"ou"` (current storm) | reference |
| E0b | `"spectral_ou"` | same spatial spectrum and memory as E0, modes independent: **does the storm's phase coherence matter?** (E0 vs E0b) |
| E1 | `"gyrostat"` | the new forcing |
| E2 | `"gyrostat_surrogate"` of E1 | same auto- and cross-spectra as E1, Gaussian: **does structure beyond spectra matter?** (E1 vs E2) |
| E3 | E1 with the other time unit of decision 4 | time-scale sensitivity |
| E1-W | E1 with the orthogonal mapping `W` | mapping sensitivity (R2) |
| E1-k, E2-k | 10 ocean initial states each | forced response vs intrinsic variability |
| E4 (optional) | E1 on `|k| ≤ 4` with a synoptic target spectrum | only if decision 1 adds synoptic scales |

E0b vs E2 then isolates the effect of the gyrostat's time spectra against
OU's.

### 7.3 Diagnostics and statistics

The diagnostics are those of the scoping note, computed per run in
`reports/qg/analyze_wind_response.py`:
- frequency spectra of upper-layer KE, ψ₁ and ψ₂, in synoptic (< 10 d),
  intraseasonal (10–90 d) and low-frequency (> 90 d) bands;
- transfer function and coherence from each mode amplitude to the
  projection of ψ₁ on the same mode (`project`), which is natural in this
  basis;
- wind work `⟨ψ₁ · curl τ⟩`, EKE per layer, and bottom-drag dissipation;
- skewness and kurtosis of ψ₁ and q₁;
- composites around gyrostat regime changes;
- forced vs intrinsic variance per band, from the ensembles.

**Significance:** a difference must exceed a moving-block bootstrap 95%
interval **and** the ensemble spread.

### 7.4 Decision criteria

- **Structure matters** if E1 and E2 differ significantly in at least one of:
  band-integrated KE (by more than 10%), ψ₁/q₁ kurtosis, the
  mode-to-response transfer, or the forced-variance fraction, **and** the
  sign holds under E1-W. Next: Option C, then scoping WP6.
- **Otherwise:** write the negative result in `docs/results/`, keep the
  drivers (QG-TC uses the gyrostat for its environment), and do not build
  Option C for the one-way case.
- **Always reported:** E0 vs E0b (coherence) and E0b vs E2 (time spectra).

## 8. Tests (`tests/test_qg_wind_modes.py`, `tests/test_gyrostat_driver.py`)

| test | asserts |
|---|---|
| default unchanged | `wind_driver="ou"`: `generate_wind_state`, `wind_curl_field` and a 50-step rollout bit-identical to today (fixed seeds), for `QGDynamics` and `QG1LDynamics` |
| cache keys unchanged | the default config's two cache paths equal pinned literals |
| cache keys separate | each spectral driver hashes to its own key |
| basis exactness | `curl_spectral(a)` equals `rfft2(curl_field(a))`; `project(curl_field(a)) == a` (float64) |
| storm in basis | today's storm at random positions, projected on `|k| ≤ 2`, keeps ≥ 99.5% of its curl variance (pins the §3 measurement) |
| translation | `curl_field(translate(a, dx, dy))` equals the field rolled by `(dx, dy)` for grid-multiple shifts |
| spectrum match | long `"spectral_ou"` and `"gyrostat"` series reproduce `storm_mode_std` within 5% per mode (`slow`) |
| zero amplitude | `amp = 0` gives `a ≡ 0` and a rollout bit-identical to the unforced model |
| L63 = gyrostat | the `"l63"` preset equals the textbook right-hand side after the z shift |
| energy | unforced, undamped cores (`"l63"`, `"chain6"`) conserve `energy(y)` to RK4 accuracy |
| surrogate | auto- and cross-spectra preserved to FFT precision; kurtosis 3 ± 0.3 on a long series |
| 1L/2L agree | `QG1LDynamics` and `QGDynamics` build the same upper-layer PV source from the same spectral state |
| S1 semantics | the corrupted spectral state has amplitude bias `s1_amp_bias` and an rms shift of ≈ 62.5 km (`slow`) |
| end to end | a 2-window `QGS01Dataset` with `wind_driver="gyrostat"` builds S0/S1/S1-qg1l windows with `(T, n_amp)` wind states, and the psi-state DA wrapper steps with it |

`ruff check` on the touched files and the fast suite locally before each PR.

## 9. Work breakdown and PRs

| step | content | PR |
|---|---|---|
| WP-B1 | `qg_wind_modes.py` (basis, projection, translation, storm spectrum) and `gyrostat_driver.py` (`"l63"`, `"chain6"`, integrate, surrogate). Measure `"chain6"`'s chaos and time scales; record them in `docs/results/` with §3 and §4. | PR-1 |
| WP-B2 | Dynamics hooks (2L, 1L, psi wrappers), `QGConfig` fields, conditional cache keys, S1 dispatch, `forcing_true` semantics; all "unchanged" and end-to-end tests. | PR-2 |
| WP-B3 | `run_wind_response.py` + batch script (env `fdv-monai-proto`, outputs to node-local `/tmp`); E0–E3 and ensembles on the cluster. | PR-3 + runs |
| WP-B4 | `analyze_wind_response.py`, report in `reports/qg/outputs/`, results doc stating the §7.4 decision. | PR-4 |

PR-1 is independent of the QG code. PR-2 depends on PR-1. Both base on master
(stacked PRs get no CI).

**Cost.** Truth generation today runs at ~35 steps/s on CPU. The spectral
forcing adds a K-term write into the spectrum per step, which is negligible,
and removes today's grid evaluation plus FFT per step. A 6-year run is about
13 min on CPU; the ensembles batch on GPU. WP-B3 is a few GPU-hours.

**Seams kept, not built:**
- **Option C:** `project` gives the ocean-to-atmosphere projection, and
  relative-wind drag acts mode by mode.
- **QG-TC:** it reuses `GyrostatSystem` for its environment.
- **CGOA:** its builder can later produce the presets.
- **Neural rows under a new driver** (WP6): recompute the forcing-channel
  normalization stats per driver (`precompute_qg_norm_stats.py`), and run
  every compared method under the same driver.

## 10. Risks

| # | risk | mitigation |
|---|---|---|
| R1 | `"chain6"` is regular or weakly chaotic. | Measure the Lyapunov exponent and ACFs in WP-B1. Fall back to independent L63 copies per pair, or tune the coefficients. |
| R2 | Results depend on the mode assignment. | E1-W with an orthogonal map; report only conclusions that survive both. |
| R3 | A 12-amplitude basis cannot represent synoptic scales. | Intended: §3 shows today's forcing has none. E4 and decision 1 add them if wanted. |
| R4 | Intrinsic eddy variability swamps the forced signal. | Ensembles and forced-variance diagnostics before interpreting. |
| R5 | The wider wind state breaks a consumer the audit missed. | End-to-end test through S0, S1 and S1-qg1l, and the psi-state DA wrapper. |
| R6 | Stale caches across drivers. | Conditional keys + `WIND_DRIVER_VERSION`; tests. |

## 11. Open decisions

1. **Basis size.** `|k| ≤ 2` (12 amplitudes: contains today's forcing,
   clean comparison with E0), or `|k| ≤ 4` (48 amplitudes: adds 250–500 km
   synoptic scales that the current case study does not have)?
   **Proposal:** `|k| ≤ 2` for E0b–E3; `|k| ≤ 4` as the optional E4.
2. **Target spatial spectrum.** Today's storm spectrum (proposed, for
   comparability), or a synoptic-peaked spectrum? The latter only makes
   sense with `|k| ≤ 4`.
3. **Gyrostat topology.** `"chain6"` (13 modes, identity mapping), or
   independent L63 copies (one per cos/sin pair, plus weak coupling)? The
   chain gives cross-mode structure through shared modes; the copies are
   better understood. **Proposal:** `"chain6"`, with L63 copies as the R1
   fallback.
4. **Time unit.** Set from `"chain6"`'s measured ACF so the amplitude memory
   matches today's 15 days, or a synoptic ~2-day memory? **Proposal:** match
   15 days for E1; the synoptic value as E3.
5. **Drift.** Keep today's imposed drift as a phase rotation (proposed; keeps
   E0 comparability), or let propagation come from the gyrostat dynamics
   only?
6. **`forcing_true` for spectral drivers.** The domain-rms curl (proposed,
   unsigned), or the amplitude of the leading mode (signed, but
   basis-dependent)?

## References

- Lorenz (1963), deterministic nonperiodic flow. *J. Atmos. Sci.* 20, 130–141.
- Prichard & Theiler (1994), generating surrogate data for time series with several simultaneously measured variables. *Phys. Rev. Lett.* 73, 951–954.
- Theiler, Eubank, Longtin, Galdrikian & Farmer (1992), testing for nonlinearity in time series: the method of surrogate data. *Physica D* 58, 77–94.
- Künsch (1989), the jackknife and the bootstrap for general stationary observations. *Ann. Statist.* 17, 1217–1241.
- Seshadri & Lakshmivarahan (2023), minimal chaotic models from the Volterra gyrostat. *Physica D* 133948.
- De Cruz, Demaeyer & Vannitsem (2016), MAOOAM v1.0. *Geosci. Model Dev.* 9, 2793–2808 (a low-order Fourier atmospheric basis of the same kind).
- Scoping context: `docs/plans/case_study/qg_gyrostat_synoptic_forcing.md`, `docs/plans/case_study/qg_hurricane_coupled.md`, `docs/plans/case_study/cgoa_coupled_gyrostats.md`.
