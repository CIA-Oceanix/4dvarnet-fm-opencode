# Localized EnKS on the QG gyrostat benchmark (smoother vs filter)

**Status:** RESULTS (2026-09-29). A localized ensemble Kalman smoother (EnKS)
on the benchmark ETKF, the counterpart for the localized QG setting of the
L96 ETKS (`docs/results/l96_etks_benchmark.md`). The L96 ETKS works in
ensemble space on the unlocalized ETKF, so it does not apply here.

**Numbers:** `reports/qg/outputs/qg_specwind_da_report.md` (test rows) and
`evaluation/qg_specwind_etkf_check.py` (lag selection):
- lag selection: `evaluation/qg_specwind_etkf_check.py` (`--summarize`, tasks
  16–25) over `experiments/qg_specwind_etkf_check/`, SLURM 56925, 20 val
  windows of the forced dataset;
- test rows: `reports/qg/outputs/qg_specwind_da_report.md`, from
  `experiments/qg_specwind_da/` (`METHOD=enks`, SLURM 56975–56978, rtx8000
  and l40s).

## Method

`evaluation.baselines.EnKS`: the localized ETKF with the exact EnSRF update
(`loc_mode="ensrf"`, radius 8, ridge 0.1, cross-layer weight 1, bred init,
N = 80, inflation 1) runs unchanged. Each analysis records its forecast obs
anomalies HA, S⁻¹dy and the modified-gain factor
M = S^(−1/2)(S^(1/2) + R^(1/2))⁻¹ (`_ensrf_localized_analysis(record=...)`).
`_enks_smooth_localized` then applies the analyses in time order. Analysis k
updates the stored states s in [t_{k−lag}, t_k), using the cross-covariance
C = L ∘ (A_sᵀ HA_k)/(N − 1), where L is that analysis's localization between
the state points and its observations:
- mean += C S⁻¹dy;
- anomalies −= HA_k (C M)ᵀ.

**Exactness:** without localization, with N − 1 ≥ D and a linear model, this
is the exact RTS smoother of the filter. `tests/test_enks_localized.py`
checks mean and covariance against a closed-form RTS.

The lag is counted in analyses. At 3 columns per day, 12 analyses = 4 days.
Inflation must be 1 and there must be no additive noise: both hold for the
QG benchmark.

## Lag selection (val, 20 windows)

| | S0 score | S0 ψ₁ / q₁ | S1 score | S1 ψ₁ / ψ₂ / q₁ | S1 spread/RMSE q₁ |
|---|---|---|---|---|---|
| ETKF (filter) | 0.764 | 0.939 / 0.615 | 0.580 | 0.862 / 0.838 / 0.389 | 0.68 |
| EnKF (filter) | 0.753 | 0.932 / 0.605 | 0.593 | 0.858 / 0.828 / 0.407 | 0.69 |
| EnKS, lag 3 (1 day) | 0.777 | 0.948 / 0.644 | 0.596 | 0.877 / 0.852 / 0.416 | 0.66 |
| EnKS, lag 6 (2 days) | 0.786 | 0.955 / 0.666 | *0.605* | 0.885 / 0.853 / 0.439 | 0.65 |
| **EnKS, lag 12 (4 days)** | *0.798* | 0.963 / 0.692 | **0.608** | 0.888 / 0.832 / 0.466 | 0.63 |
| EnKS, lag 24 (8 days) | **0.802** | 0.968 / 0.707 | 0.588 | 0.872 / 0.760 / 0.482 | 0.60 |
| EnKS, whole window | 0.791 | 0.966 / 0.700 | 0.550 | 0.850 / 0.681 / 0.472 | 0.56 |

Paired differences for lag 12 (score, bootstrap 95%):

| | S0 | S1 |
|---|---|---|
| − ETKF | +0.034 [+0.029, +0.038] | +0.028 [+0.020, +0.036] |
| − EnKF | +0.044 [+0.038, +0.051] | +0.015 [+0.005, +0.025] |
| − lag 24 | −0.004 [−0.007, −0.002] | +0.020 [+0.007, +0.037] |
| − lag 6 | +0.011 [+0.010, +0.013] | +0.003 [−0.003, +0.008] |

**Selected: lag 12.**

## Findings (val)

1. **The smoother helps in both scenarios.** At lag 12: +0.034 over the
   ETKF in S0 and +0.028 in S1. It is also the best estimate under S1,
   ahead of the EnKF (+0.015).
2. **Model error shortens the useful lag.**
   - In S0 the gain grows up to lag 24 (8 days) and falls slightly for the
     whole window.
   - In S1 the gain peaks at 6–12 analyses. Beyond that it turns negative:
     over the whole window the EnKS is **below the filter** (0.550), and the
     lower layer collapses (ψ₂ 0.68).
   - With a wrong model, later observations stop being informative about
     earlier states, and the sampled cross-covariances turn into noise. The
     L96 ETKS showed the same pattern: a smaller gain on S1, and an effective
     lag of 1–2 analyses there.
3. **The gain is mostly in PV**: S0 q₁ 0.615 → 0.692 at lag 12, ψ₁ 0.939 →
   0.963. Smoothing constrains the small scales that one day of columns
   cannot.
4. **The smoother is more under-dispersed than the filter**, and more so at
   longer lags: S1 spread/RMSE q₁ goes 0.68 → 0.63 at lag 12 and 0.56 over the
   whole window. The L96 ETKS showed the same, as expected when a smoother
   treats the forecast as exact between analyses. Probabilistic scores (CRPS)
   gain less than RMSE-type ones.

## Test (100 windows, lag 12)

| | S0 forced | S0 coupled | S1 forced | S1 coupled |
|---|---|---|---|---|
| ETKF (filter) | *0.774* | *0.779* | 0.579 | 0.586 |
| EnKF (filter) | 0.761 | 0.768 | *0.596* | *0.604* |
| **EnKS (lag 12)** | **0.804** | **0.813** | **0.602** | **0.610** |

Paired differences (bootstrap 95%), forced / coupled:

| | S0 | S1 |
|---|---|---|
| EnKS − ETKF, score | +0.030 [+0.027, +0.033] / +0.034 [+0.031, +0.037] | +0.023 [+0.015, +0.030] / +0.024 [+0.016, +0.030] |
| EnKS − EnKF, score | +0.043 [+0.039, +0.047] / +0.045 [+0.040, +0.049] | +0.005 [−0.005, +0.014] / +0.006 [−0.004, +0.015] |
| EnKS − ETKF, q₁ EV | +0.070 [+0.066, +0.074] / +0.075 [+0.071, +0.080] | +0.071 [+0.065, +0.076] / +0.075 [+0.070, +0.080] |
| EnKS − ETKF, ψ₂ EV | +0.015 [+0.013, +0.018] / +0.018 [+0.015, +0.021] | −0.016 [−0.040, +0.005] / −0.020 [−0.045, −0.000] |

- **Test confirms val.** The smoother gains about +0.03 over the filter in
  S0 and about +0.02 in S1, mostly in upper-layer PV (+0.07 in q₁).
- **In S1 it ties the EnKF on the score**, with better ψ₁ and q₁ (0.872 /
  0.504 against 0.847 / 0.450) and slightly lower ψ₂ and q₂.
- **In S1 the unobserved ψ₂ loses a little** against the filter (−0.016 /
  −0.020, significant for coupled). This is the long-lag failure mode of
  finding 2, in milder form.
- **Spread/RMSE in q₁ (median)** falls from 0.72 to 0.68 in S0 and from 0.70
  to 0.64 in S1: the smoother is more under-dispersed than the filter.

## Caveats

- **The EnKS is a smoother.** It uses observations after each time, so it is
  a reanalysis-type estimate. It is reported next to the filters, not
  instead of them.
- **Same localization as the analysis.** Each analysis's state–observation
  localization is reused for the earlier states, as is standard for a
  localized EnKS. A time-dependent localization, tapering with lag, could
  make longer lags safe under model error.
- **Lag tuned on 20 val windows at one density.**
