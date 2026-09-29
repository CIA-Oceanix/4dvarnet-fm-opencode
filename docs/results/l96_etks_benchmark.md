# L96 ETKS benchmark rows: validation-tuned inflation and test results

**Status:** RESULTS (2026-09-28). PR 2 of `docs/plans/tech/l96_etks_smoother.md`: the ETKS (`correct` retro-inflation, full window) on the 200 P1 test windows, regular grid and random layouts, S0 and S1. Two inflation settings: the pre-#295 benchmark λ (S0 1.5 / S1 2.0) and the λ selected here on the validation windows (S0 1.1 / S1 2.5). #295 independently retuned the benchmark ETKF default to S0 1.15 / S1 2.5 (`docs/results/l96_da_inflation_post291.md`), close to the selection here but not equal; no ETKS row was run at exactly 1.15 / 2.5. The ETKF was rerun in the same jobs for paired comparisons.

**Numbers:** `reports/l96/outputs/l96_benchmark_extended.md` (ETKS rows of the main and CRPS tables). The validation sweep comes from `batch/run_l96_etks_val_tuning.sbatch` + `evaluation/l96_etks_select_inflation.py`, and the test rows from `batch/run_l96_etks_test.sbatch`. The paired statistics below were computed from the same bundle files.

## Question

The prototype (`docs/results/l96_etks_prototype.md`) measured the smoother gain with a different RMSE aggregation and no tuning. Here:
- what are the ETKS benchmark rows under the benchmark metric?
- is the benchmark inflation right for the ETKF and the ETKS on validation windows?

## Validation sweep

50 validation windows per case (`scripts/make_l96_validation_sets.py`), regular grid and random layouts drawn from the same windows.
- **Grid:** λ ∈ {1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 2.5, 3.0}, the same λ for the ETKF and the ETKS.
- **Why that grid:** the first grid (1.1–2.0) put every selection on an edge, so it was extended.
- **Criterion:** per case and method, the mean over the two layouts of CRPS relative to that layout's best.

CRPS (regular / random), selected values in bold:

| λ | S0 ETKF | S0 ETKS | S1 ETKF | S1 ETKS |
|---|---|---|---|---|
| 1.05 | 0.311 / 0.288 | 0.293 / 0.292 | 1.552 / 1.457 | 1.496 / 1.384 |
| 1.1 | **0.288 / 0.291** | **0.254 / 0.254** | 1.494 / 1.331 | 1.405 / 1.223 |
| 1.2 | 0.274 / 0.309 | 0.239 / 0.272 | 1.373 / 1.151 | 1.241 / 1.051 |
| 1.5 (benchmark S0) | 0.352 / 0.419 | 0.297 / 0.360 | 1.108 / 0.869 | 0.980 / 0.828 |
| 2.0 (benchmark S1) | 0.467 / 0.622 | 0.415 / 0.562 | 0.882 / 0.758 | 0.876 / 0.720 |
| 2.5 | 0.561 / 0.955 | 0.511 / 0.875 | **0.786 / 0.775** | **0.799 / 0.734** |
| 3.0 | 0.641 / 1.357 | 0.604 / 1.245 | 0.761 / 0.844 | 0.770 / 0.786 |

- **Same selection for both methods:** S0 λ = 1.1, S1 λ = 2.5.
- **The two layouts disagree within a case.** S0: regular prefers 1.2, random 1.05–1.1. S1: regular prefers 3.0, random 2.0. The selected values are compromises.
- **On S0 the benchmark λ = 1.5 is far from optimal for both methods.**

## Test results

200 P1 test windows, benchmark metric: per-window mean over the 24 observed channels of the per-channel RMSE. Paired differences are against the ETKF of the same run, with 95% intervals.

| | λ | ETKF | ETKS | ΔRMSE | windows ETKS better | ΔCRPS |
|---|---|---|---|---|---|---|
| regular S0 | 1.5 | 0.707 | 0.582 | −17.7% ± 1.0 | 199/200 | −14.7% |
| regular S0 | 1.1 (val) | 0.625 | ***0.514*** | −17.8% ± 2.6 | 173/200 | −11.2% |
| regular S1 | 2.0 | 1.479 | 1.400 | −5.4% ± 0.2 | 200/200 | −0.5% |
| regular S1 | 2.5 (val) | 1.410 | ***1.338*** | −5.1% ± 0.2 | 200/200 | +1.6% |
| random S0 | 1.5 | 0.864 | 0.770 | −10.8% ± 1.0 | 186/200 | −12.7% |
| random S0 | 1.1 (val) | 0.662 | ***0.566*** | −14.4% ± 2.1 | 170/200 | −9.9% |
| random S1 | 2.0 | 1.376 | ***1.282*** | −6.8% ± 0.5 | 189/200 | −5.3% |
| random S1 | 2.5 (val) | 1.406 | 1.349 | −4.0% ± 0.7 | 170/200 | −5.2% |

Best ETKF/ETKS RMSE per case and layout in bold italics (Strong-4DVar and EnKF are not compared here).

Validation-tuned vs benchmark λ, ETKF only (paired):
- regular S0: −11.6%, 155/200 windows better;
- random S0: −23.3%, 185/200;
- regular S1: −4.7%, 200/200;
- random S1: +2.2%, 111/200 (worse).

Spread/RMSE:

| | ETKF (1.5 / 2.0) | ETKS (1.5 / 2.0) | ETKF (val λ) | ETKS (val λ) |
|---|---|---|---|---|
| regular S0 | 1.00 | 0.71 | 0.51 | 0.33 |
| regular S1 | 0.37 | 0.24 | 0.61 | 0.34 |
| random S0 | 1.13 | 0.81 | 0.62 | 0.44 |
| random S1 | 0.65 | 0.41 | 0.93 | 0.53 |

## Findings

1. **Filter vs smoother (the F3 term), at fixed inflation.**
   - The ETKS gains 11–18% RMSE on S0 and 4–7% on S1 over the ETKF of the same run.
   - It is better in 170–200 of 200 windows in every cell, and all intervals exclude zero.
   - The gain is smaller under model error, as predicted in the P1 plan. It is also smaller on the random layouts at S0 (−10.8% vs −17.7% at λ = 1.5).
   - The prototype's S0 regular figure (−17.7% under its own metric) is reproduced exactly under the benchmark metric.
2. **The old benchmark S0 inflation (1.5) is too high on the post-#291 ETKF** — the same finding as #295, which moved the default to S0 1.15 / S1 2.5 (`docs/results/l96_da_inflation_post291.md`).
   - Tuned on validation, λ = 1.1 improves the S0 ETKF by 12% (regular) and 23% (random) on test, and its CRPS by 17% / 34%.
   - Hypothesis, not tested here: the #291 transform fix keeps spread that the old thin-SVD transform dropped. That matters most on the random layouts (12–24 observed channels for 30 members), so the fix lowered the optimal inflation. This is consistent with the rerun ETKF at λ = 1.5 being 8% *worse* than the pre-#291 row on random S0 (0.864 vs 0.798), while regular S1 is unchanged (1.479 vs 1.478). #295 measured the same (`docs/results/l96_da_inflation_post291.md`).
   - This sweep selects the same S1 value as #295 (2.5) and S0 1.1 vs its 1.15. The criteria differ: CRPS relative to each layout's best here, versus the worse layout's RMSE loss in #295. The ETKS's optimum equals the ETKF's in both cases, so the smoother needs no inflation of its own.
3. **S1 has no single good inflation.** λ = 2.5 helps the regular grid (−4.7%) and hurts the random layouts (+2.2%). A per-layout inflation, or adaptive inflation, would be needed to do better. The tuned S1 rows are not uniformly better, and the best random-S1 ETKF/ETKS row is the ETKS at λ = 2.0.
4. **Best ETKF/ETKS rows.** ETKS at the validation-selected λ:
   - regular S0 **0.514**, S1 **1.338**; random S0 **0.566**, S1 1.349.
   - Against the current benchmark ETKF (#298: 1.15 / 2.5; regular S0 0.610, S1 1.409; random S0 0.679, S1 1.407): −16% / −5% regular, −17% / −4% random. These are not paired: a different run, and λ 1.1 vs 1.15 at S0.
   - Against the pre-#291 benchmark ETKF (0.687 regular S0, 0.798 random S0): −25% / −29%.

   For comparison, the learned schemes are at ≈0.34 (regular S0) and ≈0.44 (random S0). DA's regular-S0 deficit goes from 2.0× (pre-#291 ETKF, 0.687) to 1.8× (retuned benchmark ETKF, 0.610) to 1.5× (ETKS, 0.514).
   Roughly, of the original 0.35 gap, the inflation retune removes 0.08 and smoothing another 0.10 (at the two runs' slightly different S0 λ); about half remains.
   On random S0 the ETKS is at 1.3×.
5. **Calibration.**
   - At the tuned λ, both filter and smoother are under-dispersed (ETKS spread/RMSE 0.33–0.53), because the CRPS criterion trades spread for accuracy.
   - At S1 the tuned ETKS has a slightly *worse* CRPS than the tuned ETKF on the regular grid (+1.6%) despite 5% lower RMSE: smoothing shrinks spread further.
   - The S0 calibration-identity check of the plan (variance drop = MSE drop) therefore still has no calibrated filter to test on. Neither λ setting gives spread/RMSE ≈ 1 for both methods and layouts.

## Status and follow-ups

- The report shows the benchmark ETKF/EnKF rows (#298, at the #295 defaults) followed by two paired ETKF + ETKS groups from this PR: λ 1.5 / 2.0 and λ 1.1 / 2.5.
- **Not done here:**
  - an ETKS row at exactly the new benchmark default (1.15 / 2.5, #295);
  - the #243 obs-count sweep and RMSE(t);
  - speeding up the ETKS smoothing, which is currently about the filter's cost again, on CPU.
