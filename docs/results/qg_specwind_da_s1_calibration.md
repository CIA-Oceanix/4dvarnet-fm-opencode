# S1 (model error) for the spectral-wind QG DA: calibration and attribution on val

**Status:** RESULTS (2026-09-27; base attribution re-run 2026-09-28 with the
exact localized ETKF). Design: `docs/plans/analysis/qg_specwind_da_s1.md`. Val
only, 20 windows of the forced dataset; test untouched. No S1 re-tuning; no
4D-Var.

**Which ETKF:**
- **§3.1, the current base attribution:** the exact localized EnSRF update
  (`loc_mode="ensrf"`, radius 8, ridge 0.1, cross-layer weight 1, bred init,
  N = 80; `docs/results/qg_specwind_etkf_loc_update.md`).
- **Everything else (the scenario scan in §2 and the other attributions in
  §3):** the legacy localized update (radius 8, ridge 1). It divides R by
  N − 1 and over-contracts the spread. Those numbers stay as a record of
  how the scenario was chosen; the choice itself is not affected.

**Numbers:** `evaluation/qg_specwind_s1_sweep.py` (`--summarize`, with `--kappa 2`,
`--variant base` or `--variant high` for the attributions) over
`experiments/qg_specwind_s1_ensrf/` (EnSRF ETKF, §3.1) and `experiments/qg_specwind_s1/`
(legacy ETKF). Run with `batch/run_qg_specwind_s1.sbatch` (SLURM 55959, 56000, 56034,
56039, 56172, 56181; EnSRF 56545).

**Target** (user request): per-window analysis EVs typically in upper-layer
q ∈ [0, 0.25] and upper-layer ψ ∈ [0.7, 0.9], read as the median over the 20
windows, with the fraction of windows inside reported.

**Selected S1: the realism-anchored scenario at its central levels
(`REALISTIC_VARIANTS["base"]`),** chosen on 2026-09-27 after the variant
scan. Every magnitude is the middle of what is defensible for an ocean
reanalysis. It puts the ETKF's ψ₁ where the user wanted it: with the EnSRF
ETKF, 0.862 mean on val and 0.848 on test
(`docs/results/qg_specwind_da_s0_s1_test.md`); the old ETKF gave 0.865 / 0.854.
The price is that q₁ stays above the original 0–0.25 band (0.389 on val, 0.433
on test).

History:
1. The first target (q₁ ∈ [0, 0.25] and ψ₁ ∈ [0.7, 0.9]) was met by the
   legacy-analogue κ = 2 scenario, but only through an rd bias about 3×
   real uncertainty.
2. Within realistic ranges, the target was met only by the upper-edge
   variant, "high".
3. The user then aimed at ψ₁ ≈ 0.85–0.9, which base meets.

"high" and the interpolated "mid" variants stay available for a harsher
S1 (§2).

## 1. Scenarios

| | legacy analogue, κ = 2 | **realistic, base (selected)** | realistic, "high" |
|---|---|---|---|
| wind amplitude bias | +30% | +15% | +20% |
| random wind error (per mode, × train RMS, τ = 10 d) | 0.60 | 0.25 | 0.35 |
| wind position error (OU, per axis) | 100 km | 50 km | 70 km |
| rd | −30% | −10% | −15% |
| bottom drag | −30% | −50% | −60% |
| observation error (fraction of ψ₁ std) | 5% white (S0) | 15% white + 15% correlated per pass | 20% + 20% |
| DA model grid | 64 (truth) | 32 (rd unresolved) | 32 |

The correlated per-pass error is an offset plus a tilt along each observed
column.

κ = 2 scales the legacy QG S1 levels (`REFERENCE`) by two. Its −30% rd bias
is about three times real stratification uncertainty, and it has no
observation or structural error.

## 2. Calibration

| scenario | ψ₁ median [IQR] | ψ₁ in [0.7, 0.9] | q₁ median [IQR] | q₁ in [0, 0.25] | ψ₂ | q₂ | score | free ψ₁ / q₁ |
|---|---|---|---|---|---|---|---|---|
| S0 | 0.955 [0.90, 0.96] | 25% | 0.572 [0.49, 0.64] | 0% | 0.956 | 0.554 | 0.754 | 0.746 / −0.185 |
| κ = 1.5 | 0.886 [0.78, 0.95] | 50% | 0.369 [0.32, 0.42] | 0% | 0.845 | 0.197 | 0.571 | −0.147 / −0.831 |
| κ = 2 | 0.828 [0.69, 0.93] | 40% | 0.142 [0.10, 0.24] | 80% | 0.752 | −0.271 | 0.359 | −0.908 / −1.370 |
| κ = 3 | 0.626 [0.26, 0.87] | 35% | −0.835 | 0% | 0.422 | −2.842 | −0.716 | −3.90 / −4.59 |
| realistic, low | 0.939 [0.88, 0.95] | 35% | 0.521 [0.40, 0.58] | 0% | 0.921 | 0.445 | 0.694 | 0.563 / −0.262 |
| realistic, base, DA grid 64 | 0.903 [0.85, 0.93] | 45% | 0.418 [0.29, 0.51] | 15% | 0.879 | 0.265 | 0.605 | 0.323 / −0.401 |
| realistic, base, DA grid 48 | 0.907 [0.84, 0.93] | 45% | 0.428 [0.30, 0.51] | 10% | 0.874 | 0.263 | 0.604 | 0.301 / −0.383 |
| realistic, base | 0.894 [0.81, 0.92] | 55% | 0.365 [0.25, 0.44] | 25% | 0.855 | 0.146 | 0.551 | 0.287 / −0.333 |
| **realistic, high** | **0.847** [0.73, 0.89] | **65%** | **0.174** [0.07, 0.33] | 45% | 0.788 | −0.194 | 0.393 | −0.070 / −0.479 |

**Between base and high** (linear interpolation of every level, DA grid 32;
`mid25` / `mid50` / `mid75` = 25 / 50 / 75% of the way to high):

| variant | ψ₁ mean / median | ψ₁ in [0.7, 0.9] | q₁ mean / median | q₁ in [0, 0.25] | ψ₂ | q₂ | score |
|---|---|---|---|---|---|---|---|
| **base (selected)** | 0.865 / 0.894 | 55% | 0.338 / 0.365 | 25% | 0.855 | 0.146 | 0.551 |
| mid25 | 0.854 / 0.883 | 55% | 0.301 / 0.326 | 35% | 0.841 | 0.076 | 0.518 |
| mid50 | 0.843 / 0.872 | 60% | 0.260 / 0.278 | 45% | 0.824 | −0.003 | 0.481 |
| mid75 | 0.831 / 0.861 | 65% | 0.213 / 0.224 | 40% | 0.807 | −0.094 | 0.439 |
| high | 0.817 / 0.847 | 65% | 0.163 / 0.174 | 45% | 0.788 | −0.194 | 0.393 |

Across these, ψ₁ moves little (means 0.865 → 0.817) and q₁ about 3× as
much (0.338 → 0.163). The variants mainly differ in PV skill.

- **Only the upper edge of every realistic range reaches the first target.** At
  base levels q₁ stays at 0.37. So the targeted degradation needs either
  errors at the pessimistic end of each range, or error sources this model
  cannot represent: mesoscale forcing errors, an analysis-based first guess
  instead of the lagged truth, missing physics.
- **"high" spreads the degradation more evenly** than κ = 2: 65% of windows
  have ψ₁ in range, against 40%. κ = 2 leaves some windows almost
  unaffected (IQR up to 0.93) and ruins others (0.69).
- **The DA beats the free forecast in every window in both scenarios.** The
  free forecast collapses under S1 (ψ₁ EV −0.07 in "high", −0.91 at κ = 2).
- **Resolution alone is a small error.** A 48×48 DA grid is
  indistinguishable from 64×64. Only 32×32, where rd (12.8–17.2 km) falls
  below the grid spacing (31 km), costs about 0.05 in q₁.
- **Intensity is steep in the legacy mix:** q₁ goes 0.37 → 0.14 → −0.84 for
  κ = 1.5 → 2 → 3, driven by the parameter bias. Its stand-alone dose
  response is catastrophic at κ = 4 (q₁ −3.9).

## 3. Attribution (Shapley over all on/off combinations, per-window bootstrap)

Each component's Shapley value is its average marginal EV loss over all
orders of switching components on. The values sum exactly to the S0 → S1
loss. "Interaction" is the full loss minus the sum of stand-alone losses.

### 3.1 Realistic base, selected — EnSRF ETKF (5 groups, 32 runs)

| metric (S0 → S1) | forcing | rd | drag | obs | res | interaction |
|---|---|---|---|---|---|---|
| score (0.764 → 0.580) | 9% | 18% | 16% | **31%** | 25% | −0.000 |
| ψ₁ (0.939 → 0.862) | 11% | 13% | 10% | **40%** | 26% | +0.004 |
| ψ₂ (0.950 → 0.838) | 18% | 7% | 24% | **27%** | 25% | +0.011 |
| q₁ (0.615 → 0.389) | 6% | 19% | 4% | **46%** | 26% | −0.026 |
| q₂ (0.552 → 0.229) | 7% | 23% | 24% | 21% | **25%** | +0.009 |

- **Observation error is still the largest source,** but by less: 31% of
  the score loss (0.058 [0.047, 0.069]) and 46% of the q₁ loss (0.104
  [0.080, 0.128]).
- **Resolution is a solid second** on every metric (25–26%).
- **The q₂ loss spreads evenly** over rd (23%), drag (24%), obs (21%) and
  resolution (25%).
- **With the exact update the errors no longer compound.** The interaction
  terms are about zero, against +0.042 (score) and +0.131 (q₂) with the
  legacy ETKF. The legacy filter's over-confidence made the errors amplify
  each other; the exact filter degrades additively.
- **Forcing stays the smallest group** (9%; 18% of ψ₂).

### 3.1c Realistic base — legacy ETKF (5 groups, 32 runs; superseded by §3.1)

| metric (S0 → S1) | forcing | rd | drag | obs | res | interaction |
|---|---|---|---|---|---|---|
| score (0.754 → 0.551) | 9% | 16% | 10% | **43%** | 22% | +0.042 |
| ψ₁ (0.934 → 0.865) | 16% | 13% | 6% | **38%** | 28% | +0.008 |
| ψ₂ (0.956 → 0.855) | **27%** | 6% | 18% | **27%** | 22% | +0.013 |
| q₁ (0.573 → 0.338) | 6% | 18% | 0% | **54%** | 21% | +0.017 |
| q₂ (0.554 → 0.146) | 5% | 18% | 15% | **41%** | 21% | +0.131 |

- **Observation error was the largest single source**: q₁ loss 0.127
  [0.082, 0.190] of 0.235, and 38% of the ψ₁ loss.
- **Resolution is second** on ψ₁ and q₁ (28% and 21%), and ahead of rd
  here.
- **rd still compounds** with the other errors: its Shapley value on q₁ is
  0.042, against 0.024 alone.
- **Forcing** matters mainly for ψ₂ (27%). **Drag** matters only below the
  upper layer.
- **The ranking of sources is the same as at "high"**, but interactions are
  smaller. The q₂ interaction is +0.131 of 0.407, against +0.313 of 0.748
  at "high".

### 3.1b Realistic "high" — legacy ETKF (5 groups, 32 runs)

| metric (S0 → S1) | forcing | rd | drag | obs | res | interaction |
|---|---|---|---|---|---|---|
| score (0.754 → 0.393) | 10% | 23% | 8% | **44%** | 16% | +0.108 |
| ψ₁ (0.934 → 0.817) | 18% | 17% | 6% | **42%** | 18% | +0.020 |
| ψ₂ (0.956 → 0.788) | **32%** | 8% | 15% | **31%** | 15% | +0.027 |
| q₁ (0.573 → 0.163) | 7% | 25% | −1% | **55%** | 14% | +0.071 |
| q₂ (0.554 → −0.194) | 6% | 26% | 11% | **41%** | 16% | +0.313 |

- **Observation error is the largest single source**: q₁ loss 0.227
  [0.151, 0.331] of 0.411, and 42% of the ψ₁ loss. Part of its effect comes
  from the filter treating correlated per-pass errors as white.
- **rd is second on PV, despite only −15%.** Its Shapley value on q₁
  (0.101) is twice its stand-alone loss (0.053): an rd error compounds with
  the other errors.
- **Forcing errors matter mostly for the unobserved lower layer** (32% of
  the ψ₂ loss, 6–7% of the PV losses).
- **Resolution** takes 14–18% on every metric. **Drag** matters only below
  the upper layer (15% of the ψ₂ loss, 11% of the q₂ loss).
- **The errors compound strongly in q₂**: interaction +0.313 of a 0.748
  loss.

### 3.2 Legacy analogue κ = 2 — legacy ETKF (4 components, 16 runs)

| metric (S0 → S1) | amp | noise | shift | param (rd + drag) | interaction |
|---|---|---|---|---|---|
| score (0.754 → 0.359) | 3% | 26% | 3% | **69%** | +0.048 |
| ψ₁ (0.934 → 0.800) | 2% | 41% | 5% | **51%** | +0.004 |
| ψ₂ (0.956 → 0.752) | 2% | **73%** | 3% | 22% | +0.006 |
| q₁ (0.573 → 0.154) | 3% | 20% | 4% | **73%** | +0.055 |
| q₂ (0.554 → −0.271) | 3% | 14% | 2% | **81%** | +0.124 |

This attribution is dominated by the inflated rd bias. **Systematic wind
errors** (amplitude +30%, position 100 km) are absorbed by daily
assimilation within 30-day windows, at about 2–5% each. The same holds in
the realistic scenario, where the whole forcing group costs 7–18% of the
upper-layer losses.

## 4. Caveats

- **20 val windows, one density** (3 columns per day), one DA
  configuration (S0-tuned ETKF, no inflation). Under S1, inflation or a
  larger R could recover part of the loss. That re-tuning is a follow-up.
- **Parameter errors have one sign each** (rd and drag underestimated).
  The opposite signs are not tested.
- **The observation-error correlation model is simple** (offset and linear
  tilt per pass). Real long-wavelength altimetry errors are more varied.
- **Structural error comes from resolution only.** The one-layer DA model
  and the missing coupled feedback are not included.
- **"Realistic" means defensible, not measured**, for this idealized box
  and this wind basis.
