# S1 tuning of the QG gyrostat DA: inflation diverges, a larger R absorbs the model error

**Status:** RESULTS (2026-09-30). The benchmark filters were tuned in S0
(`docs/results/qg_specwind_da2_val_tuning.md`,
`docs/results/qg_specwind_etkf_loc_update.md`, `docs/results/qg_specwind_enks.md`).
This note tunes them under the realistic S1 (base,
`docs/results/qg_specwind_da_s1_calibration.md`). The report then shows both:
- the S0-tuned rows (how S0-tuned DA degrades);
- the S1-tuned rows (how well the filters can do under model error).

**Numbers:** `evaluation/qg_specwind_s1_tuning.py` (`--summarize`) over
`experiments/qg_specwind_s1_tuning/` (SLURM 57138 and 57183, rtx8000, 20 val
windows of the forced dataset). Test rows:
`reports/qg/outputs/qg_specwind_da_report.md` (S1-tuned section), from
`experiments/qg_specwind_da/` (`RSCALE=6`, SLURM 57213–57218).

## Grid

- **ETKF** (EnSRF, ridge 0.1) and **EnKF:**
  - inflation {1.0, 1.05, 1.1, 1.2} × radius {6, 8};
  - R scale {1.5, 2, 3} × radius {6, 8} at inflation 1;
  - R scale extended to {4, 6, 8} at each method's best radius, because the
    first grid peaked at its edge (R × 3).
- **EnKS** (localized smoother on the ETKF): it needs inflation 1, so it is
  tuned on R scale {1, 1.5, 2, 3, 4, 6, 8} × radius {6, 8} × lag {6, 12}.
  Radius 6 was dropped from the extension.

The R scale multiplies the filter's observation-error variance on top of the
S1 altimetry-error level the filter already assumes.

## Results (val, 20 windows)

**Inflation diverges.**

| inflation | ETKF radius 8 | EnKF radius 6 |
|---|---|---|
| 1.0 | 0.580 | 0.592 |
| 1.05 | 0.346 | 0.315 |
| 1.1 | −101.7 | −103.5 |
| 1.2 | −223.1 | −224.5 |

Inflation is applied at every analysis, about 90 per 30-day window. With
sparse localized observations, the poorly observed directions grow without
bound: spread/RMSE reaches 3 at 1.05 and 9 at 1.2. The legacy QG study
found inflation > 1 harmful too.

**A larger R helps monotonically, up to a plateau at 6–8.** Score by R scale:

| R scale | 1 | 1.5 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| ETKF, radius 8 | 0.580 | 0.601 | 0.613 | 0.627 | 0.634 | **0.641** | 0.643 |
| EnKF, radius 6 | 0.592 | 0.607 | 0.617 | 0.627 | 0.631 | **0.636** | 0.636 |
| EnKS, radius 8, lag 12 | 0.608 | 0.630 | 0.643 | 0.658 | 0.665 | **0.673** | 0.675 |
| EnKS, radius 8, lag 6 | 0.605 | 0.627 | 0.639 | 0.653 | 0.661 | 0.668 | 0.670 |

Spread/RMSE of q₁ (median) at R × 1 → R × 6: ETKF 0.68 → 0.90, EnKF 0.69 →
0.88, EnKS 0.63 → 0.87.

**Selected (S1-tuned): R × 6 for all three.** It is within 0.002 of R × 8
and avoids the extreme end. Methods keep their S0 radius (ETKF 8, EnKF 6,
EnKS 8) and the EnKS its lag 12; radius 6 vs 8 differs by ≤ 0.004 at every
R scale.

| val, S1 | S0-tuned | S1-tuned (R × 6) | Δ (bootstrap 95%) |
|---|---|---|---|
| ETKF | 0.580 | 0.641 | +0.061 [+0.055, +0.068] |
| EnKF | 0.592 | 0.636 | +0.045 [+0.038, +0.051] |
| EnKS | 0.608 | **0.673** | +0.065 [+0.056, +0.073] |

## Findings (val)

1. **Under S1, model error is best absorbed as observation error.** An R
   scale of 6 means an assumed observation-error std about 2.4× the true
   altimetry error. The filters then trust each observation less, keep more
   spread, and stop over-fitting observations to a wrong model. Every EV
   improves. The unobserved-layer PV q₂ improves by far the most (+0.145
   ETKF, +0.157 EnKS), then q₁ (+0.04–0.05), ψ₂ (+0.03–0.04) and ψ₁
   (+0.02).
2. **Multiplicative inflation is not usable in this configuration.** It
   compounds over about 90 analyses per window. Relaxation inflation
   (RTPS / RTPP) was tried later (`docs/results/qg_specwind_relax_inflation.md`).
   It is stable, but it trails R × 6 by about 0.03 and adds at most +0.004 on
   top of it.
3. **The ETKF–EnKF gap closes.** S0-tuned, the EnKF led the ETKF by 0.012
   (test: 0.017). S1-tuned, the ETKF leads by 0.005 on val, and the two tie
   exactly on test. The earlier gap was spread/trust calibration, which the
   R scale fixes for both.
4. **The smoother gains most** (+0.065). With a better-calibrated
   observation weight, later observations become more usable for earlier
   states. The EnKS is again clearly the best estimate (0.673 against 0.641).

## Test (100 windows)

S1 (realistic base), 3 columns per day; forced / coupled:

| | S0-tuned | S1-tuned (R × 6) | Δ (bootstrap 95%) | S1-tuned spread/RMSE q₁ |
|---|---|---|---|---|
| ETKF | 0.579 / 0.586 | 0.633 / 0.642 | +0.054 [+0.049, +0.059] / +0.055 [+0.051, +0.059] | 0.87 / 0.88 |
| EnKF | 0.596 / 0.604 | 0.634 / 0.642 | +0.037 [+0.033, +0.041] / +0.038 [+0.034, +0.041] | 0.86 / 0.86 |
| **EnKS** | 0.602 / 0.610 | **0.664 / 0.672** | +0.063 [+0.058, +0.068] / +0.062 [+0.057, +0.067] | 0.84 / 0.85 |

- **Test confirms val.** R × 6 gains +0.04 to +0.06 for every method on
  both datasets.
- **S1-tuned, the ETKF and the EnKF tie exactly:** ETKF − EnKF = −0.001
  [−0.004, +0.002] / −0.000 [−0.004, +0.003]. On val the ETKF led by 0.005.
- **The EnKS leads the S1-tuned filters** by +0.031 [+0.026, +0.036] /
  +0.030 [+0.025, +0.035], mostly in q₁ (0.537 / 0.545 against 0.473 /
  0.481 for the ETKF). These are lag-12 numbers. With the later τ = 8-day
  time taper (`docs/results/qg_specwind_enks.md`) the EnKS scores 0.665 /
  0.673.
- **Spread is near calibrated** for all three (spread/RMSE 0.84–0.88,
  against 0.64–0.70 S0-tuned).

## Caveats

- 20 val windows; one S1 variant (base) and one density.
- **R × 6 is S1-specific.** In S0 the true R is right, and the S0 rows keep
  R × 1.
- **The R scale is a single global factor.** A per-layer or scale-dependent
  model-error treatment (additive noise, weak-constraint 4D-Var) could do
  better.
