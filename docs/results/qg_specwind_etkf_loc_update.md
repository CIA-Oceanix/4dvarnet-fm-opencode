# The localized ETKF update: two defects, and the exact EnSRF alternative (QG gyrostat, val)

**Status:** RESULTS (2026-09-28). A spread check prompted by EnKF > ETKF under
S1 in `docs/results/qg_specwind_da_s0_s1_test.md`.

**Numbers:** `evaluation/qg_specwind_etkf_check.py` (`--summarize`) over
`experiments/qg_specwind_etkf_check/`. Run with
`batch/run_qg_specwind_etkf_check.sbatch` (SLURM 56406, rtx8000): 20 val
windows of the forced dataset, S0 and the realistic S1 (base).

## The two defects

The localized branch of `evaluation.baselines.ETKF` (`loc_mode="square_root"`,
the default) has two problems:

1. **Unnormalized covariances.** `Pf_Ht = A.T @ HA` and `H_Pf_Ht = HA.T @ HA`
   come from raw anomalies, while `R_obs = R_var·I` is not rescaled. The gain
   is therefore P̂Hᵀ(HP̂Hᵀ + R/(N−1))⁻¹, with P̂ the sample covariance: with
   N = 80, the observation error is treated as 79× smaller than it is. The
   ridge (`etkf_ridge` × the largest obs-space variance, tuned to 1.0) then
   partly plays the role of R.
2. **Full-gain anomaly update.** The anomalies are updated as (I − KH)A,
   whose covariance (I − KH)P(I − KH)ᵀ lacks the +KRKᵀ term of an exact
   analysis. The spread is over-contracted.

**Evidence:** in `tests/test_etkf_ensrf_localized.py`, one analysis step
with ridge 0 leaves a variance of about 1e-4 at the observed points under
the legacy update, against about 0.5 for the exact update. The EnKF normalizes by N − 1
and uses perturbed observations, so it is not affected. The unlocalized ETKF
(used for L96, fixed separately in #291) is also unaffected.

## The exact alternative: `loc_mode="ensrf"`

`evaluation.baselines._ensrf_localized_analysis`:
- the covariances are normalized by N − 1 and localized;
- the mean gets the Kalman gain K = PHᵀS⁻¹;
- the anomalies get the modified gain K̃ = PHᵀS^(−1/2)(S^(1/2) + R^(1/2))⁻¹
  (Andrews 1968; Whitaker & Hamill 2002). Without localization the analysis
  covariance is then exactly (I − KH)P; a unit test checks this;
- `etkf_ridge` × the largest obs-space variance is added to R.

The legacy mode stays the default.

## Results (val, 20 windows, N = 80, inflation 1.0, cross-layer weight 1, bred init)

| configuration | S0 score | S0 q₁ / q₂ | S0 spread/RMSE q₁ / q₂ | S1 score | S1 q₁ / q₂ | S1 spread/RMSE q₁ / q₂ |
|---|---|---|---|---|---|---|
| ETKF legacy (radius 8, ridge 1) | 0.754 | 0.573 / 0.554 | 0.75 / 0.92 | 0.551 | 0.338 / 0.146 | **0.58 / 0.76** |
| ETKF EnSRF, radius 8, ridge 0.1 | **0.764** | 0.615 / 0.552 | 0.74 / 0.90 | *0.580* | 0.389 / 0.229 | 0.68 / 0.89 |
| ETKF EnSRF, radius 6, ridge 0.1 | *0.759* | 0.605 / 0.558 | 0.81 / 0.96 | *0.580* | 0.393 / 0.265 | 0.74 / 0.97 |
| ETKF EnSRF, radius 6, ridge 0 | 0.754 | 0.606 / 0.540 | 0.73 / 0.90 | 0.568 | 0.383 / 0.240 | 0.72 / 0.94 |
| ETKF EnSRF, radius 8, ridge 0 | 0.748 | 0.602 / 0.510 | 0.66 / 0.82 | 0.566 | 0.377 / 0.200 | 0.67 / 0.85 |
| ETKF EnSRF, radius 4, ridge 0.1 | 0.732 | 0.570 / 0.535 | 0.87 / 1.01 | 0.562 | 0.392 / 0.303 | 0.82 / 1.07 |
| EnKF (radius 6) | 0.753 | 0.605 / 0.538 | 0.70 / 0.88 | **0.593** | 0.407 / 0.279 | 0.69 / 0.92 |

Paired differences in score (bootstrap 95% over windows):

| comparison | S0 | S1 |
|---|---|---|
| EnSRF (radius 8, ridge 0.1) − legacy | +0.010 [+0.004, +0.016] | +0.028 [−0.001, +0.072] |
| EnSRF (radius 8, ridge 0.1) − legacy, q₂ EV | −0.001 [−0.012, +0.010] | +0.083 [+0.010, +0.195] |
| EnSRF (radius 6, ridge 0.1) − legacy, q₂ EV | +0.004 [−0.004, +0.012] | +0.119 [+0.045, +0.235] |
| EnSRF (radius 8, ridge 0.1) − EnKF | +0.011 [+0.008, +0.014] | −0.013 [−0.019, −0.008] |
| EnKF − legacy | −0.001 [−0.006, +0.004] | +0.042 [+0.009, +0.089] |

## Findings

1. **Under model error the legacy ETKF is over-confident.** Its S1
   spread/RMSE is the lowest of all configurations (0.58 in q₁, 0.76 in
   q₂), and its PV skill the worst. The EnSRF restores the spread to the
   EnKF's level (0.68–0.74 / 0.89–0.97) and recovers q₂ significantly
   (+0.08 to +0.12).
2. **The ETKF–EnKF gap under S1 shrinks from 0.042 to 0.013.** The
   remainder is small but significant; the stochastic update or further
   S1 tuning may account for it.
3. **In S0 the tuned ridge masked the defects.** The legacy spread looks
   reasonable there (0.75 / 0.92, above the EnKF's). Even so, the EnSRF at
   radius 8 with ridge 0.1 is significantly best: +0.010 over the legacy
   ETKF and +0.011 over the EnKF.
4. **The EnSRF prefers a small ridge** (0.1 beats 0 at every radius) and
   radius 6–8. Radius 4 over-disperses slightly and loses ψ skill.
5. **All filters are somewhat under-dispersive in upper-layer q**
   (spread/RMSE 0.66–0.87). Since the EnKF is the most under-dispersive in
   S0 without being worse, spread calibration alone does not explain
   rankings. Inflation remains a tuning lever.

## Implications

- **QG gyrostat benchmark:** switching the ETKF to `loc_mode="ensrf"`
  (radius 8, ridge 0.1) would improve both scenarios and make the ETKF
  rows correct. The switch needs the ETKF test rows re-run; this is
  pending the user's decision.
- **Legacy QG reports** (storm-forced case, `qg_da_report.md` etc.) used the
  flawed localized update throughout. Their ETKF numbers are
  configuration-specific rather than filter-intrinsic, especially the
  promoted large ridge.

## Caveats

- 20 val windows, one density, S1 realistic base only.
- The EnSRF was not re-tuned for inflation.
