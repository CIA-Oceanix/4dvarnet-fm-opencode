# L96 FDV2 × score-based flow matching: training-free posterior sampling vs the FDV2→SDA hybrid

**Status:** RESULTS (2026-10-14; S1 check added 2026-10-15). Phases 0–1 of `docs/plans/analysis/l96_fdv2_fm_posterior.md` (validation sweeps and held-out test), S0, both observing layouts, FDV2 seeds 1 and 2, plus an S1 test of the frozen S0 configurations (section 4). No training. Phase 2 (M2, M3) was not run.

**Numbers:** inline. Per-window JSON outputs (scores only; no member arrays kept) are under `experiments/l96_fdv2_fm_posterior/{val,test}/` of the main checkout (not checked in), produced by `eval_fdv_fm_posterior_l96.py` at commit 2fa1f10 (code identical to #344) with `batch/run_l96_fdv2_fm_posterior.sbatch`. The row lists and the job IDs are in that directory too (`rows_*.txt`, `submitted_jobs.txt`). The S1 rows (section 4) are under `s1/{val,test}/` of the same directory (row lists `rows_s1_{val,test}.txt`), run at commit aaada67 (the L96 harness code is unchanged since 2fa1f10).

## Question and answer

**Question.** Can one configuration that combines the FDV2 unrolled solver with the SDA1-M flow prior, used as a score in a posterior sampler, match the FDV2→SDA hybrid's best RMSE *and* best CRPS while being calibrated (plan section 0.2)?

**Answer: nearly.** **M1bI** has three ingredients:
- perturbed-observation FDV2 starts at 0.75× the observation noise;
- a DPS-guided stochastic FM sampler from τ0 = 0.7 (γ = 0.5);
- one inflation factor tuned on validation.

Measured on 200 held-out test windows per cell, paired against the hybrid on the same windows:
1. **CRPS is 4.2–7.3% better than the hybrid's best-CRPS setting, and 3.8–6.9% better than B5** (the hybrid inflated to SSR = 1) in every cell. Every 95% interval excludes 0.
2. **It is calibrated in every cell** (SSR 0.90–0.99, 90% coverage 0.89–0.92, rank-histogram reliability 0.03–0.09). On the random layout, post-hoc inflation fixes the hybrid's spread but not the shape of its rank histogram (B5 reliability 0.17–0.21).
3. **RMSE ties the hybrid's best-RMSE setting on the random layout (−0.4%, −0.0%) but misses it on the regular layout (+0.8%, +0.6%; intervals exclude 0).** So the pre-registered criterion S-acc fails on 2 of 4 cells, by under 1%.

The remaining gap is a trade-off inside the method, set by the perturbation strength. At 0.5× observation noise, RMSE and CRPS pass in all four cells, but the inflated ensemble is over-dispersed in the tails (reliability 0.15–0.27) and S-cal fails everywhere.

**On S1 (model error, section 4)**, the same frozen configuration keeps its CRPS lead (−4.7…−7.2% against the hybrid, −4.5…−7.0% against B5) and SSR ≈ 1.04. RMSE passes in 3 of 4 cells. Reliability passes on the regular layout but not on the random layout (0.12–0.17 against a target of about 0.10), although it stays below the hybrid's and B5's.

## Setup

- **Data:** S0, regular layout (`l96_datasets_obsj2_int100_nwin200.pt`) and random layout (`l96_testset_rlayout_n10-100_k4-16_w200_d1.pt`), 200 test windows each. Validation sets are disjoint, with 100 windows each (seeds 1057 / 1067).
- **Models:** FDV2 = the coupled deep-supervision fine-tune (seeds 1 and 2, `docs/results/l96_fdv_m_mask_prior_unroll.md`). Flow prior = SDA1-M (`B4_sda1_monaiM_ep1200_l96_seed1`).
- **Ensembles and scores:** 30 members, physical units. A cell is a layout × FDV2 seed.
- **Statistics:** paired window bootstrap (2000 resamples, percentile 95% intervals) of the difference against the reference row on the same windows.
- **Calibration terms:** "SSR" is the pooled spread-skill ratio √((M+1)/M)·√(Σvar / Σmse). "Reliability" is Σ_k |f_k − 1/(M+1)| over the rank histogram. "cov90" is the coverage of the 5–95% member interval.

**Methods.** Everything is training-free.

| row | what |
|---|---|
| B3 (the hybrid) | FDV2 mean → SDA warm start at τ0, deterministic guided ODE, 10 steps |
| B5 | B3 at its CRPS-best setting, members inflated about the mean to SSR = 1 (tuned on val) |
| B2 | FDV2 on perturbed observations (ε ~ N(0, p²R)) |
| M1 | FDV2 mean → `fm_posterior_sample` from τ0: stochastic Euler–Maruyama with noise γ, DPS or ΠGDM likelihood guidance |
| M1b | M1 started from the B2 ensemble instead of the mean |
| M1bI | M1b inflated to SSR = 1, the factor tuned on val and fixed on test |
| M1c | M1 / M1b with a per-window τ0 = a − b·rank(J_prior) |

**Selection.** Everything was chosen on validation.
- The hybrid's best settings: a 4 (τ0) × 3 (gw) grid on 100 windows.
- FM rows: 488 rows on windows 0–39, paired with the hybrid on the same windows.
- Test was run once, with two configurations frozen beforehand.

Validation inflation is tuned in-sample, for B5 and M1bI alike.

## 1. The hybrid on validation, and the B5 gate (G0)

- **Both of the hybrid's best settings use gw = 1, not the probe's 2.** The best-RMSE setting is τ0 = 0.5 and the best-CRPS setting is τ0 = 0.1, in all four cells. So the hybrid needs two different τ0 to reach its two optima.
- **gw = 1 makes the hybrid stronger than the 2026-10-12 probe** that motivated the plan. For example, regular test seed 2 at τ0 = 0.1 gives CRPS 0.1113 vs 0.1166 in the probe.
- **The best-CRPS setting is already moderately dispersed:** SSR is 0.82 on the regular layout and 0.61–0.67 on the random layout.
- **B5 calibrates the hybrid's spread at no CRPS cost.** Its inflation is 1.20–1.23 on the regular layout and 1.49–1.57 on the random layout.
  - On the regular layout, B5 has reliability 0.02–0.03, so the calibration bar is "beat B5" (plan section 0.2).
  - On the random layout it does not fix the rank histogram: reliability is 0.17–0.21, against 0.20 for B3.

## 2. What drives the FM posterior (validation)

1. **The sampler's noise γ adds little spread.** With FDV2-mean starts, γ from 0 to 1 moves SSR only from 0.42 to 0.51 at τ0 = 0.3, and γ = 2 costs 6% or more in CRPS. With likelihood guidance at gw = 1, the spread comes from the starts, as it does for the hybrid.
2. **DPS and ΠGDM give the same scores** to three digits at our settings, and DPS is about 3× cheaper. Predictor-corrector steps and 20 steps instead of 10 do not help.
3. **The normalized (SDA-style) guidance is worse at every τ0.**
4. **Perturbed-observation starts carry the gain.** M1b at τ0 = 0.5, γ = 0.5 cuts CRPS 3–5% against both the hybrid and B5 in every cell. It is still under-dispersed (SSR 0.71–0.81), and RMSE is +0.5…+2.3% against the hybrid's best-RMSE setting.
5. **Inflating M1b (M1bI) fixes the spread.** Its inflation is 1.24–1.41, against B5's 1.20–1.57. Reliability is 0.04–0.11, and CRPS is −4…−6% against both baselines.
6. **The perturbation scale p trades RMSE against calibration:**
   - p = 1.5: RMSE +6…+9%;
   - p = 0.75: RMSE −0.4…+1.1%, calibrated in 3 of 4 cells;
   - p = 0.5: RMSE −0.9…+0.4%, but inflation of 2.0–2.6 over-disperses the tails.
7. **A per-window τ0 from J_prior (M1c) does not beat a fixed τ0.** None of the 9 (a, b) pairs did better.

## 3. Test (200 windows per cell)

Reference rows:

| cell | B3 τ0 = 0.5 RMSE | B3 τ0 = 0.1 RMSE / CRPS / SSR / rel | B5 CRPS / SSR / cov90 / rel |
|---|---|---|---|
| regular, seed 1 | 0.2326 | 0.2412 / 0.1132 / 0.81 / 0.166 | 0.1127 / 0.99 / 0.911 / 0.032 |
| regular, seed 2 | 0.2236 | 0.2359 / 0.1113 / 0.82 / 0.168 | 0.1109 / 0.99 / 0.905 / 0.022 |
| random, seed 1 | 0.2755 | 0.2853 / 0.1335 / 0.61 / 0.204 | 0.1333 / 0.96 / 0.943 / 0.205 |
| random, seed 2 | 0.2664 | 0.2786 / 0.1310 / 0.62 / 0.199 | 0.1304 / 0.93 / 0.935 / 0.166 |

**M1bI, DPS, γ = 0.5, τ0 = 0.7, p = 0.75** (the calibration-first configuration). Inflation from val is 1.65–1.86.

| cell | RMSE | ΔRMSE vs B3 τ0 = 0.5 | CRPS | ΔCRPS vs B3 τ0 = 0.1 | ΔCRPS vs B5 | SSR | cov90 | rel (target) |
|---|---|---|---|---|---|---|---|---|
| regular, seed 1 | 0.2344 | +0.8% [+0.4, +1.2] ✗ | **0.1084** | **−4.2%** [−5.0, −3.3] | −3.8% | 0.99 | 0.902 | **0.028** (≤ 0.083) |
| regular, seed 2 | 0.2250 | +0.6% [+0.2, +1.0] ✗ | **0.1045** | **−6.1%** [−6.8, −5.4] | −5.7% | 0.99 | 0.894 | **0.031** (≤ 0.084) |
| random, seed 1 | **0.2744** | −0.4% [−1.1, +0.3] | **0.1245** | **−6.8%** [−8.4, −5.4] | −6.6% | 0.93 | 0.922 | **0.092** (≤ 0.102) |
| random, seed 2 | **0.2664** | −0.0% [−0.7, +0.7] | **0.1214** | **−7.3%** [−8.8, −6.0] | −6.9% | 0.90 | 0.898 | **0.032** (≤ 0.100) |

**M1bI, DPS, γ = 0.5, τ0 = 0.5, p = 0.5** (the accuracy-first configuration). Inflation from val is 2.04–2.34.

| cell | ΔRMSE vs B3 τ0 = 0.5 | ΔCRPS vs B3 τ0 = 0.1 | ΔCRPS vs B5 | SSR | cov90 | rel |
|---|---|---|---|---|---|---|
| regular, seed 1 | −0.2% [−0.5, +0.1] | −4.7% [−5.5, −4.0] | −4.3% | 0.99 | 0.941 | 0.166 ✗ |
| regular, seed 2 | −0.2% [−0.5, +0.2] | −6.5% [−7.2, −5.9] | −6.1% | 0.99 | 0.940 | 0.153 ✗ |
| random, seed 1 | −0.8% [−1.4, −0.3] | −5.7% [−7.1, −4.5] | −5.6% | 0.95 | 0.955 | 0.269 ✗ |
| random, seed 2 | −0.5% [−1.2, +0.1] | −6.8% [−8.0, −5.7] | −6.5% | 0.92 | 0.945 | 0.207 ✗ |

**Against the plan's criteria:**

| criterion | p = 0.75, τ0 = 0.7 | p = 0.5, τ0 = 0.5 |
|---|---|---|
| S-acc, CRPS | passes, 4 of 4 cells | passes, 4 of 4 cells |
| S-acc, RMSE | passes on the random layout only | passes, 4 of 4 cells |
| S-cal | passes, 4 of 4 cells | fails, 4 of 4 cells |
| beats B5 | yes: CRPS everywhere; reliability on the random layout | yes on CRPS; no on reliability |
| S-robust (both seeds) | consistent | consistent |

S1: section 4.

**Validation was a fair guide.** The test differences sit within about 1 point of the validation estimates on windows 0–39, and the frozen inflation factors give SSR 0.90–0.99 on test.

**Cost.** Each member takes 20 network evaluations: 10 FDV2 iterations and 10 DPS steps, each with a vector–Jacobian product. That is 2.8 s per window for 30 members on an A40, against 1.4 s for B3.

## 4. S1: the frozen configurations under model error

**Protocol.** S1 is the model-error case of the L96 benchmark: the `test_s1` split of the same validation and test files, whose windows come from a perturbed-parameter L96, while FDV2 and the SDA1-M prior were trained on S0 only.
- Nothing was re-selected on S1. The hybrid's τ0 = 0.5 / 0.1 and both M1bI configurations are the S0 choices.
- Only the inflation factors were re-tuned, on the S1 validation windows, the same way as on S0: B5 on 100 windows, M1bI on windows 0–39.
- Test: 200 windows per cell, same scores and paired bootstrap as section 3. The plan reports S1 without gating it.

Reference rows (S1):

| cell | B3 τ0 = 0.5 RMSE | B3 τ0 = 0.1 RMSE / CRPS / SSR / rel | B5 CRPS / SSR / cov90 / rel |
|---|---|---|---|
| regular, seed 1 | 0.2270 | 0.2365 / 0.1116 / 0.82 / 0.147 | 0.1113 / 1.02 / 0.915 / 0.056 |
| regular, seed 2 | 0.2184 | 0.2326 / 0.1102 / 0.83 / 0.148 | 0.1099 / 1.01 / 0.909 / 0.036 |
| random, seed 1 | 0.2724 | 0.2810 / 0.1319 / 0.63 / 0.205 | 0.1326 / 1.05 / 0.954 / 0.256 |
| random, seed 2 | 0.2617 | 0.2741 / 0.1287 / 0.64 / 0.194 | 0.1295 / 1.04 / 0.952 / 0.249 |

**M1bI, DPS, γ = 0.5, τ0 = 0.7, p = 0.75** (S1). Inflation from S1 val is 1.69–2.04 (S0: 1.65–1.86).

| cell | RMSE | ΔRMSE vs B3 τ0 = 0.5 | CRPS | ΔCRPS vs B3 τ0 = 0.1 | ΔCRPS vs B5 | SSR | cov90 | rel (target) |
|---|---|---|---|---|---|---|---|---|
| regular, seed 1 | 0.2284 | +0.6% [+0.2, +1.1] ✗ | **0.1063** | **−4.7%** [−5.4, −4.0] | −4.5% | 1.04 | 0.915 | **0.066** (≤ 0.073) |
| regular, seed 2 | 0.2186 | +0.1% [−0.3, +0.5] | **0.1023** | **−7.2%** [−7.9, −6.5] | −6.9% | 1.05 | 0.908 | **0.050** (≤ 0.074) |
| random, seed 1 | 0.2738 | +0.5% [−0.1, +1.1] | **0.1255** | **−4.8%** [−6.0, −3.8] | −5.4% | 1.04 | 0.938 | 0.165 (≤ 0.103) ✗ |
| random, seed 2 | 0.2622 | +0.2% [−0.5, +0.9] | **0.1204** | **−6.5%** [−7.6, −5.3] | −7.0% | 1.04 | 0.928 | 0.119 (≤ 0.097) ✗ |

**M1bI, DPS, γ = 0.5, τ0 = 0.5, p = 0.5** (S1). Inflation from S1 val is 2.10–2.47.

| cell | ΔRMSE vs B3 τ0 = 0.5 | ΔCRPS vs B3 τ0 = 0.1 | ΔCRPS vs B5 | SSR | cov90 | rel |
|---|---|---|---|---|---|---|
| regular, seed 1 | −0.3% [−0.7, +0.1] | −4.9% [−5.5, −4.3] | −4.6% | 1.05 | 0.949 | 0.218 ✗ |
| regular, seed 2 | −0.6% [−0.9, −0.2] | −7.2% [−7.8, −6.6] | −6.9% | 1.04 | 0.948 | 0.210 ✗ |
| random, seed 1 | +0.0% [−0.6, +0.7] | −3.9% [−4.8, −2.9] | −4.4% | 1.02 | 0.962 | 0.309 ✗ |
| random, seed 2 | −0.3% [−0.8, +0.3] | −5.5% [−6.4, −4.5] | −6.1% | 1.06 | 0.962 | 0.308 ✗ |

**Against the plan's criteria on S1:**

| criterion | p = 0.75, τ0 = 0.7 | p = 0.5, τ0 = 0.5 |
|---|---|---|
| S-acc, CRPS | passes, 4 of 4 cells | passes, 4 of 4 cells |
| S-acc, RMSE | passes in 3 of 4 cells (fails regular seed 1; random seed 1 is at the +0.5% margin) | passes, 4 of 4 cells |
| S-cal | passes on the regular layout; fails reliability on the random layout | fails, 4 of 4 cells |
| beats B5 | yes: CRPS everywhere; reliability on the random layout | yes on CRPS; no on reliability |
| S-robust (both seeds) | consistent | consistent |

**What changes from S0:**
- **The ordering holds.** M1bI beats the hybrid and B5 on CRPS by the same margins as on S0, in every cell and for both seeds.
- **RMSE is unchanged in size.** The gap to the hybrid's best-RMSE setting is −0.4…+0.8% on S0 and +0.1…+0.6% on S1; only the cells that fail differ.
- **The spread needs more inflation, and the frozen factor slightly over-disperses.** Inflation rises to 1.69–2.04, and test SSR is 1.04–1.05 (0.90–0.99 on S0).
- **The rank histogram on the random layout is no longer flat enough.** Reliability is 0.12–0.17, against 0.03–0.09 on S0. It is still below the hybrid's 0.19–0.21 and B5's 0.25–0.26; both M1bI and B5 lose 0.05–0.09 from S0. One global inflation factor does not fix the shape of the spread under model error, which strengthens the case for the inflation-free variant (section 5).

## 5. Reading

- **One configuration does the hybrid's two jobs.** M1bI (p = 0.75) is as accurate as the hybrid's best-RMSE setting to within 1% and beats its best-CRPS setting by 4–7%. The hybrid needs two τ0 values for those two optima, and neither of them is calibrated.
- **The gain is not just calibration.** At equal SSR ≈ 1, M1bI beats B5 on CRPS by 4–7%, and on the random layout it also beats B5 on rank-histogram reliability.
- **The remaining RMSE gap comes from the ensemble mean.** Perturbed starts move the mean slightly off the FDV2 iterate, and inflation does not change the mean. Phase 2 (M2, a score-prior FDV2 with an iterate-error-matched noise level) targets this directly. It is the natural next step if the 0.6–0.8% regular-layout gap matters.
- **S1 confirms the accuracy result but weakens the calibration result** (section 4): CRPS gains of 4–7% and SSR ≈ 1 carry over, but the rank histogram on the random layout fails the target.
- **Open:** an inflation-free variant (stochasticity calibrated by the sampler alone). γ did not achieve that here.
