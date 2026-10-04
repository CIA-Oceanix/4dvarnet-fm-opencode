# Strong and weak 4D-Var for the QG gyrostat DA (S0 and realistic S1)

**Status:** RESULTS (2026-10-04). This adds the variational baselines to the
ensemble benchmark (`docs/results/qg_specwind_da_s0_s1_test.md`). It covers
strong-constraint 4D-Var in S0 and S1, and weak-constraint 4D-Var in S1. Both
use the observations, initial background and DA model of the ensemble
methods.

**Numbers:** test rows in `reports/qg/outputs/qg_specwind_da_report.md`.
- Test: 100 windows, from
  `experiments/qg_specwind_da/<spec>/test/*4dvar*` (SLURM 58317–58349).
- Val: 10 windows of the forced dataset, under
  `experiments/qg_specwind_da/qg_specwind_gyrostat_v1/val/*4dvar*`
  (SLURM 57998–58310, rtx8000).

## Method (`QG4DVar`, `evaluation/run_qg_baselines.py`)

**Background and control:**
- Deterministic.
- Background = the lagged-truth initial state (the ensemble mean's
  counterpart).
- Whitened control x₀ = x_b + B^½ u.

**Background covariance B:**
- **Spectral climatological B** (`spectral_b_sqrt`): one 2×2 inter-layer
  covariance per wavenumber shell.
- It is estimated from the lead buffer of the dataset, like the bred
  ensemble.
- This makes B homogeneous and isotropic, with the true spectrum and the true
  ψ₁–ψ₂ coupling.

**Weak constraint:**
- An additive model-error control at each sub-window.
- Its covariance is Q = (scale) × B; the scale is set by `--q-var-scale`.

**Optimization:**
- L-BFGS with lr 1.0 and 60 iterations.
- It cycles over sub-windows of `--da-window-steps` (120 steps = 10 days).
- Each sub-window's analysis trajectory gives the next sub-window's
  background.

**Safeguards:**
- A sub-window whose optimization is non-finite, or whose final cost exceeds
  the background's, falls back to the background forecast.
- These are counted per window (`n_fallback`).
- They caught one NaN window on val and a finite blow-up on test (window 35,
  score −4269 before the guard).

## Val tuning (S0, strong)

| B | sub-window | score | ψ₁ | q₁ | q₂ |
|---|---|---|---|---|---|
| diagonal (legacy QG baseline) | 5 d | 0.215 | 0.887 | 0.300 | **−1.075** |
| spectral × 1 | 3 d (6 windows) | 0.741 | 0.942 | 0.595 | 0.469 |
| spectral × 1 | 5 d | 0.774 | 0.958 | 0.646 | 0.520 |
| **spectral × 1** | **10 d** | **0.824** | 0.975 | 0.722 | 0.613 |
| spectral × 0.5 | 10 d | NaN window (before the fallback) | | | |
| spectral × 2 | 10 d | 0.805 (2 fallbacks) | 0.971 | 0.685 | 0.582 |
| spectral × 1 | 15 d | 0.812 (1 fallback) | 0.971 | 0.690 | 0.602 |
| spectral × 1 | 30 d (one window) | 0.628 (1 fallback) | 0.898 | 0.318 | 0.342 |
| spectral × 1, 120 iterations | 10 d | 0.830 (1 fallback) | 0.971 | 0.740 | 0.628 |

**Rows that were not adopted:**
- At 5 days, the B scale does not matter (0.769 / 0.774 / 0.772 for × 0.5 /
  1 / 2).
- 120 iterations gains +0.006 but adds a fallback.

**Selected:** spectral B × 1, 10-day sub-windows, 60 iterations.

**Why the diagonal B collapses:**
- A diagonal B in grid space has no inter-layer covariance and no spatial
  correlation.
- The lower layer is then corrected only through the dynamics, and those
  corrections are noise-dominated.
- The spectral B is what makes 4D-Var competitive.

**Why 10-day sub-windows:**
- A longer window uses more observations per analysis.
- Past about 15 days the strong-constraint cost becomes too non-linear for
  L-BFGS from the background.

## Val tuning (S1, realistic base)

| variant | R scale | model-error scale | score | ψ₁ | q₁ | q₂ |
|---|---|---|---|---|---|---|
| strong | 1 | — | 0.519 | 0.855 | 0.346 | 0.070 |
| strong | 6 | — | 0.466 | 0.817 | 0.289 | 0.051 |
| weak, 10 d | 1 | 0.05 | 0.590 | 0.886 | 0.443 | 0.196 |
| **weak, 10 d** | **1** | **0.1** | **0.598** | 0.888 | 0.456 | 0.210 |
| weak, 10 d | 1 | 0.2 | 0.592 | 0.881 | 0.446 | 0.208 |
| weak, 10 d | 1 | 0.5 | 0.573 | 0.869 | 0.415 | 0.185 |
| weak, 10 d | 1 | 1.0 | 0.562 | 0.863 | 0.395 | 0.174 |
| weak, 5 d | 1 | 0.2 / 0.5 / 1.0 | 0.536 / 0.517 / 0.503 | | | |
| weak, 10 d | 6 | 0.01 / 0.05 / 0.2 | 0.470 / 0.496 / 0.551 | | | |

**Selected for S1:** weak 4D-Var, 10-day sub-windows, model-error scale 0.1,
R × 1. The strong constraint is kept as the S0-tuned S1 row.

**Unlike the ensembles, 4D-Var does not want a larger R:**
- Inflating R hurts both the strong and the weak constraint.
- The ensembles' R × 6 compensates for spread that is too small.
- 4D-Var has no spread: its B is fixed. Its model-error treatment is the
  weak constraint.

## Test (100 windows; forced / coupled)

**S0:**

| | ψ₁ | ψ₂ | q₁ | q₂ | score |
|---|---|---|---|---|---|
| ETKF | 0.937 | 0.933 | 0.647 | 0.578 | 0.774 / 0.779 |
| EnKS | 0.966 | 0.959 | 0.727 | 0.596 | 0.812 / 0.819 |
| **strong 4D-Var** | **0.978** | **0.983** | **0.752** | **0.648** | **0.840 / 0.847** |

Field EVs are for the forced dataset. 4D-Var had 3 / 2 fallbacks.

**S1 (realistic base):**

| | ψ₁ | ψ₂ | q₁ | q₂ | score |
|---|---|---|---|---|---|
| strong 4D-Var | 0.852 | 0.706 | 0.368 | 0.139 | 0.516 / 0.527 |
| weak 4D-Var (scale 0.1) | **0.897** | 0.797 | 0.480 | 0.273 | 0.612 / 0.619 |
| ETKF, R × 6 | 0.865 | 0.791 | 0.473 | 0.402 | 0.633 / 0.642 |
| EnKS, R × 6 | 0.894 | 0.783 | **0.547** | **0.437** | **0.665 / 0.673** |

Field EVs are for the forced dataset.

**Paired differences (forced, bootstrap 95%):**

| comparison | score difference |
|---|---|
| S0: strong 4D-Var − EnKS | +0.028 [+0.020, +0.036] (coupled +0.028) |
| S0: strong 4D-Var − ETKF | +0.067 |
| S1: weak − strong 4D-Var | +0.096 [+0.084, +0.109] |
| S1: weak 4D-Var − ETKF (R × 6) | −0.021 [−0.041, −0.005] |
| S1: weak 4D-Var − EnKS (R × 6) | −0.053 [−0.074, −0.036] |

## Findings

1. **In the perfect model, 4D-Var is the best estimate.** It beats the EnKS
   on every field, most on the lower layer (ψ₂ +0.024, q₂ +0.052). The
   EnKS's N = 320 reference (0.841 / 0.847) only matches it. A climatological
   but correctly structured B, combined with an exact model over 10 days, is
   worth more than the flow-dependent covariances of 80 localized members.
2. **Under model error, the ranking flips.** The strong constraint drops to
   0.516, below every ensemble method. The weak constraint recovers +0.10,
   but stays below the S1-tuned ETKF (−0.02) and the EnKS (−0.05).
3. **The S1 deficit is in the unobserved lower-layer PV.** Weak 4D-Var
   matches the best upper-layer ψ (0.897) and the ETKF's ψ₂ and q₁. Its q₂ is
   0.27, against 0.40 for the ETKF and 0.44 for the EnKS. A model-error
   covariance proportional to B corrects the observed layer well, but cannot
   attribute errors in rd and drag to the deep layer the way the ensemble's
   flow-dependent cross-covariances do.
4. **Model-error treatment is method-specific.** The ensembles need a larger
   observation error (R × 6). 4D-Var needs the weak constraint and is hurt by
   a larger R. Each family should be tuned on its own terms before it is
   compared.

## Caveats

- **10 val windows.** Several 3-day runs covered 6 windows only.
- **The background is the lagged truth**, with no cycled first guess. B is
  climatological and not updated between sub-windows (no hybrid B).
- **Q is proportional to B.** A structured model-error covariance (for
  example, the deep layer, or rd and drag tendencies) is the obvious next
  step for weak 4D-Var under S1.
- **4D-Var is deterministic.** Its "CRPS" column in the report is the MAE of
  q, not comparable to the ensemble CRPS. It has no spread.
- **Cost:** about 25 min (strong) and about 45 min (weak) per shard of 10
  windows, against a few minutes for the ETKF. Times are on a mix of GPU
  types.
