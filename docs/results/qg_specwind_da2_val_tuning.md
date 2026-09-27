# S0 ETKF/EnKF tuning on val for the spectral-wind QG datasets (DA-2)

**Status:** RESULTS (2026-09-27). DA-2 of
`docs/plans/analysis/qg_specwind_da_s0.md` (§4.1). Tuned on 20 val windows
of the forced dataset `qg_specwind_gyrostat_v1`; test untouched.

**Setup:**
- **Observations:** upper-layer ψ₁ on 3 random meridional columns per day
  (4.7% of the grid), each once per day, with 5% noise. The lower layer is
  never observed.
- **Model and initial state:** perfect model with the true wind and
  parameters. The initial state is the truth lagged by about 5 days, with
  the lag and the perturbations drawn per window. N = 80, inflation 1.0.

**Numbers:** `evaluation/qg_specwind_da2_sweep.py` (`--summarize`) over
`experiments/qg_specwind_da2/` (one JSON per configuration, with per-window
metrics). Run with `batch/run_qg_specwind_da2.sbatch` (SLURM 55853, 55855,
55918, 55937, 55947; rtx8000, about 17 min per configuration).

**Score:** the mean over the 20 windows of the average of the four
per-window EVs (ψ₁, ψ₂, q₁, q₂). EVs are computed per window with that
window's own parameters. Intervals are paired bootstrap 95% over windows,
against the defaults (radius 2, ridge 0.1, cross-layer weight 0, white).

## Selected settings

**ETKF, `loc_radius` 8, `etkf_ridge` 1.0, `loc_cross_layer` 1.0, bred
initial ensemble (`breed_days` 3).** These replace the DA-1 defaults for
DA-3. For the EnKF: radius 6, the same weight and initial ensemble (see
the EnKF radius extension).

| config | score | Δ vs defaults | ψ₁ | ψ₂ | q₁ | q₂ | ψ₂ worse than free |
|---|---|---|---|---|---|---|---|
| free forecast | — | — | 0.746 | 0.856 | −0.185 | −0.074 | — |
| defaults: radius 2, ridge 0.1, cross 0, white | 0.646 | 0 | 0.876 | 0.854 | 0.461 | 0.392 | 6/20 |
| **radius 8, ridge 1, cross 1, bred (selected)** | *0.754* | *+0.108 [+0.089, +0.130]* | *0.934* | *0.956* | *0.573* | **0.554** | **0/20** |
| radius 6, ridge 0.1, cross 1, bred | **0.755** | **+0.109 [+0.087, +0.133]** | *0.933* | 0.943 | **0.600** | 0.543 | 1/20 |
| radius 10, ridge 1, cross 1, bred | *0.754* | *+0.108 [+0.087, +0.131]* | **0.936** | **0.958** | *0.574* | *0.547* | **0/20** |

Bold marks the best value in each column and italics the second best;
values within 0.001 of each other share a mark.

**Why radius 8 with ridge 1 rather than the top score (radius 6, ridge
0.1):**
- **The scores are tied:** paired Δ −0.001 [−0.006, +0.005].
- **The trade-off:** ψ₂ is better (+0.013 [+0.007, +0.020]) and q₁ worse
  (−0.027 [−0.035, −0.018]).
- **Robustness:** with ridge 1 the score stays within 0.007 over radius
  8–12. With ridge 0.1 the peak is sharp: −0.008 at radius 8, −0.034 at 10,
  and a collapse at 16 (q₂ EV −0.98). DA-3 changes the observation density
  and the dataset, so a setting on a plateau is the safer default.
- **It fixes the DA-1 problem:** ψ₂ beats the free forecast in every
  window.

## Findings

1. **Vertical localization is the largest single fix for the unobserved
   layer.** The original column localization gave the lower layer weight
   0, so the ETKF never updated q₂ directly. At the defaults, weight 1
   raises ψ₂ from 0.854 to 0.900 and q₂ from 0.392 to 0.420. Weights 0.5
   and 1 beat 0 in every radius, ridge, initial ensemble and method block
   of the grid, and 1 beats 0.5 in all but two blocks, where they tie
   within 0.0013 (ETKF radius 3 with ridge 0 and white noise; radius 1
   with ridge 1 and bred init).
2. **The bred initial ensemble helps everywhere, and the gain grows with
   the radius.** Bred beats white in every matched pair: +0.021 at the
   defaults, +0.029 at radius 2 with weight 1. The white-noise controls at
   large radius show why:

   | radius (ridge 0.1, weight 1) | bred | white | bred − white |
   |---|---|---|---|
   | 2 | 0.704 | 0.675 | +0.029 |
   | 6 | 0.755 | 0.688 | +0.067 |
   | 10 | 0.721 | 0.598 | +0.123 |

   With white noise, q₂ EV falls to 0.135 at radius 10. With bred
   anomalies it stays at 0.450. The bred anomalies are large-scale and
   vertically coherent, so their long-range sample covariances are usable.
   The grid-scale white-noise anomalies have only spurious long-range
   covariance.
3. **With both fixes, much weaker localization is optimal.** Score against
   radius (weight 1, bred), in grid cells of 15.6 km; Gaspari–Cohn weights
   vanish beyond twice the radius:

   | ridge | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 16 |
   |---|---|---|---|---|---|---|---|---|---|---|
   | 0 | 0.689 | 0.701 | 0.712 | 0.719 | 0.711 | 0.681 | | | | |
   | 0.1 | 0.683 | 0.704 | 0.719 | 0.735 | 0.749 | **0.755** | 0.747 | 0.721 | 0.676 | 0.243 |
   | 1 | 0.639 | 0.687 | 0.708 | | | | *0.754* | *0.754* | 0.748 | 0.722 |

   The earlier QG obs-density study (PR #208) set radius 2 as the
   default for this density range. That study used the
   white-noise ensemble and no vertical localization, and at those settings
   this sweep agrees: radius 2 or 1 is best there. The larger radius is a
   consequence of the two fixes, not a contradiction. The ridge
   regularizes the ETKF transform, and larger radii need more of it.
4. **The residual ψ₂ failures were the calm windows.** At the defaults, ψ₂
   is worse than the free forecast in the 6 least-forced val windows
   (wind level ≤ 4.2 × 10⁻¹², two with zero wind). Without forcing, the
   lower layer evolves slowly and the free forecast already scores about
   0.80–0.85 on it. The selected settings turn all six around:

   | window (level) | defaults | selected | free |
   |---|---|---|---|
   | 0 (4.4e-14) | 0.45 | 0.87 | 0.80 |
   | 2 (0) | 0.43 | 0.88 | 0.81 |
   | 7 (2.5e-12) | 0.66 | 0.88 | 0.80 |
   | 11 (4.2e-12) | 0.74 | 0.93 | 0.85 |
   | 17 (3.8e-12) | 0.96 | 0.99 | 0.99 |
   | 19 (0) | 0.75 | 0.92 | 0.84 |

5. **ETKF and EnKF are equivalent at matched settings.** In the first grid
   (radius ≤ 3), the EnKF's best is 0.717 against 0.719 for the ETKF
   (radius 3, weight 1, bred). Its response to the two new axes is the
   same. With the radius extended, the EnKF peaks at radius 6 (0.753) and
   ties the selected ETKF; see below.
6. **Ridge:** 0 and 0.1 are nearly tied at radius ≤ 4. Ridge 1 is harmful
   at small radius (radius 1: −0.029 against the defaults with white
   noise), but it is what keeps large radii stable.

## EnKF radius extension

EnKF with weight 1 and bred init. The EnKF has no ridge. Paired Δ is
against the selected ETKF:

| radius | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|
| score | 0.717 | 0.736 | 0.748 | **0.753** | 0.744 |
| Δ vs selected ETKF | | −0.018 | −0.006 | −0.001 [−0.006, +0.004] | −0.010 |
| ψ₂ worse than free | 6/20 | 5/20 | 3/20 | 2/20 | 1/20 |

**EnKF for DA-3: radius 6, weight 1, bred init** (ψ₁ 0.932, ψ₂ 0.939, q₁
0.605, q₂ 0.538). It ties the selected ETKF on the score, with higher q₁
and lower ψ₂. Like the ETKF with ridge 0.1, its optimum is a peak rather
than a plateau. The driver defaults are the ETKF settings, so DA-3 passes
`LOC=6` for the EnKF.

## Caveats

- **20 val windows:** the intervals are over windows, and the ranking of
  near-ties (within about 0.005) is not resolved.
- **One density:** tuned at 3 columns per day. The plan's density
  sensitivity (§4.3) re-uses these settings and re-tunes if the curve is
  non-monotonic.
- **The cross-layer weight was not explored above 1.** A weight of 1 means
  the lower layer gets the same horizontal weight as the upper layer.
  Larger values would amplify the sample covariance and are not a
  localization.
- **Forced dataset only.** As planned (D2), the settings carry over to the
  coupled dataset, whose DA problem is identical given the wind.
- **`run()`'s built-in ψ metrics** invert every window with window 0's
  parameters, so they are approximate for these datasets. All numbers here
  come from the sweep's own per-window metrics.
