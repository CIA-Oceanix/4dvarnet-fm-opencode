# L96 ETKS prototype on the benchmark ETKF (filter vs smoother, S0/S1)

**Status:** RESULTS (2026-09-28). First measurement of the ensemble transform Kalman smoother (ETKS) planned in `docs/plans/tech/l96_etks_smoother.md`, from a throwaway prototype on top of the benchmark ETKF (after the square-root transform fix, PR #291). All 200 P1 test windows, S0 and S1, regular grid. Not benchmark rows: see "Status of these numbers".

**Numbers:** inline (tables below). The prototype scripts, per-window scores (`etks_full.json`) and run log are archived, not checked in, under `experiments/l96/etks_prototype_2026-09-28/` in the main checkout. The two linear-Gaussian toy checks are in the same folder.

## Question

How much does smoothing gain over the filter at fixed model, ensemble, inflation and observations? That is the F3 (sequential inference) restriction term of P1. Secondary questions:
- which retrospective-inflation mode of the plan works;
- whether the ensemble RTS smoother is worth building;
- whether the S0 calibration identity (MSE_F − MSE_S = var_F − var_S) holds.

## Setup

- **Filter:** the benchmark `ETKF` class with the benchmark settings of the extended report's DA rows:
  - P1 test cache `experiments/l96_datasets_obsj2_int100_nwin200.pt` (sha256 `8eba1f87…`);
  - per-window `*_da` parameters and fast weights passed to the DA model (`--da-fast-weights`);
  - N = 30, R = 0.5, inflation λ = 1.5 (S0) / 2.0 (S1);
  - obs every 100 steps, so K = 29 analyses per 3000-step window.
- **Smoothers:** post-processing of the stored filter ensembles.
  - Each analysis map G_k = C + (1 w_kᵀ + T_k) P is rebuilt from the stored forecasts. It matches the filter to a relative 2–3·10⁻⁷ in every window. No NaN windows.
  - **ETKS `none`:** later pre-inflation maps G_j applied to the stored post-inflation ensembles, truncated at lag L (in analyses).
  - **ETKS `correct`:** later maps rescaled to G_j^(m) (mean increment × λ^(−m), covariance reduction × λ^(−2m); formula in the plan). At the analysis time itself it uses the pre-inflation ensemble.
  - **Ensemble RTS:** backward RTS with inflation read as forecast error. The cross-covariance uses the pre-inflation analysis propagated without inflation. P_f⁺ comes from a rank-truncated SVD, keeping singular values above 0.1 × the largest in S0 and 10⁻⁵ × the largest in S1 (see below).
- **Scores:**
  - RMSE: the per-window RMS over the evaluated variables and all 3000 steps, averaged over windows.
  - CRPS: per member ensemble, averaged the same way.
  - spread/RMSE: √(mean var) / √(mean MSE).
  - Paired differences against the filter with 95% intervals (1.96 σ/√n) and the share of windows where the smoother is better.
  - Evaluated variables: the 24 observed channels (S0: slice of the 40-dimensional state; S1: the whole reduced state).

## Results

200 windows per case. Paired 95% intervals on the RMSE differences are ±0.003–0.006 throughout.

| method | S0 RMSE | S0 CRPS | S0 spr/RMSE | S0 wins | S1 RMSE | S1 CRPS | S1 spr/RMSE | S1 wins |
|---|---|---|---|---|---|---|---|---|
| ETKF (filter) | 0.773 | 0.343 | 1.11 | — | 1.496 | 0.855 | 0.37 | — |
| ETKS `correct`, L = 1 | 0.639 (−17.3%) | 0.297 (−13.6%) | 0.86 | 200/200 | ***1.405 (−6.1%)*** | ***0.835 (−2.3%)*** | 0.25 | 200/200 |
| ETKS `correct`, L = 2 | ***0.636 (−17.7%)*** | 0.293 (−14.5%) | 0.80 | 200/200 | 1.416 (−5.4%) | 0.847 (−0.9%) | 0.24 | 200/200 |
| ETKS `correct`, L = 4 | 0.637 (−17.5%) | ***0.292 (−14.9%)*** | 0.77 | 200/200 | 1.411 (−5.7%) | 0.849 (−0.7%) | 0.24 | 200/200 |
| ETKS `correct`, full window | 0.639 (−17.3%) | 0.293 (−14.7%) | 0.76 | 200/200 | 1.412 (−5.6%) | 0.850 (−0.5%) | 0.24 | 200/200 |
| ETKS `none`, L = 1 | 0.640 (−17.2%) | 0.297 (−13.5%) | 0.86 | 200/200 | 1.407 (−6.0%) | 0.836 (−2.2%) | 0.26 | 200/200 |
| ETKS `none`, L = 2 | 0.659 (−14.7%) | 0.308 (−10.2%) | 0.69 | 200/200 | 1.456 (−2.7%) | 0.904 (+5.7%) | 0.19 | 196/200 |
| ensemble RTS | 0.654 (−15.4%) | 0.303 (−11.6%) | 0.73 | 200/200 | 1.411 (−5.7%) | 0.858 (+0.4%) | 0.21 | 200/200 |

Best per column in bold italics. Worst single window for ETKS `correct` L = 2: 0.93 × the filter RMSE (S0), 0.98 × (S1).

On the first 20 windows, `none` degrades steadily with lag: S0 RMSE 0.645 / 0.662 / 0.696 / 0.731 / 0.747 at L = 1 / 2 / 4 / 8 / full, with spread/RMSE falling to 0.34 at full lag. `correct` stays flat from L = 2 to the full window.

### Calibration identity

If filter and smoother were both calibrated, the variance drop would equal the MSE drop. The ratio (var_F − var_S) / (MSE_F − MSE_S):

| | `correct` L = 1 | `correct` L = 2 | `correct` full | `none` L = 2 | ensemble RTS |
|---|---|---|---|---|---|
| S0 | 2.30 | 2.49 | 2.68 | 3.29 | 3.09 |
| S1 | 0.68 | 0.81 | 0.79 | 1.90 | 0.87 |

## Findings

1. **Smoothing gains a lot on S0 and a little on S1, systematically.**
   - S0: RMSE −17.7%, CRPS −14.5% (`correct`, L = 2).
   - S1: RMSE −6.1%, CRPS −2.3% (`correct`, L = 1).
   - The smoother is better in every window of both cases.
   - The smaller S1 gain is consistent with the P1 prediction: with a biased model, later observations carry less usable information about earlier states.
2. **Almost all of the gain comes from the next analysis.** L = 1 reaches 97–100% of the best RMSE gain on both cases. The plan predicted an effective lag of 3–6 cycles from λ^(−m); the measured one is 1–2 analyses (100–200 steps). On L96 the decay of the ensemble cross-covariances, not the inflation correction, sets the lag.
3. **`correct` is the robust mode.** It is flat in L, so the lag needs no tuning. `none` is as good at L = 1 but over-corrects beyond: at L = 2 S1 CRPS is already 5.7% *worse* than the filter's, and spread collapses. `inflate` was not run.
4. **The ensemble RTS is not worth building.**
   - S1 (24 variables, fewer than N − 1 = 29): it matches `correct` on RMSE and is slightly worse than the filter on CRPS.
   - S0 (40 variables, more than N − 1): its gain needs a pseudo-inverse of a rank-deficient forecast covariance.
     - It diverges (20-window RMSE 6.1 ± 21) unless the SVD keeps only singular values above 3–10% of the largest (1e-4: 6.06, 1e-2: 0.96, 3e-2: 0.69, 0.1: 0.655 on 20 windows).
     - Even then it is worse than `correct`.
   - The ETKS works entirely in ensemble space and never inverts a state-space covariance.
5. **The calibration identity fails in both cases, in opposite directions.** At the benchmark inflation, S0 cannot serve as the clean check of the restriction reading that the plan intended.
   - S0: the variance drops 2.3–2.7× more than the MSE. The λ = 1.5 filter is over-dispersed (spread/RMSE 1.11).
   - S1: the variance drop is 0.7–0.8× the MSE drop, with a strongly under-dispersed filter (0.37).
   - On S1 the smoother shrinks the spread further (0.37 → 0.24). That is why its CRPS gain is small beyond L = 1 even though its RMSE gain holds.
6. **Toy checks** (3-variable linear system, 40 members, exact up to sampling):
   - The chained `correct` factors are 2–10% of the smoothing increment away from the exact linear smoother at λ = 1.5–2. They are identical at λ = 1.
   - Using the post-inflation ensemble at the analysis time itself is clearly wrong under the model-error reading.

## Status of these numbers

**Not a benchmark row, and no report table was changed.** Three reasons:
- **RMSE metric.** These RMSEs are per-window RMS over variables. The benchmark reports the mean of per-variable RMSE, which is smaller. On the benchmark's own stored ETKF trajectories, S0 is 0.753 here vs 0.687 in the benchmark, and S1 1.496 vs 1.478.
- **Only the relative gains transfer.** Applied to the benchmark ETKF, the S0 ETKS would land near 0.57, against ~0.34 for the learned schemes. So filtering-vs-smoothing is a minority of the DA–learned gap on S0. This is approximate until the ETKS is scored with the benchmark metric.
- **Tuning.** It is measured on test windows, with the filter's inflation, and no validation tuning of λ or L.
- **Code.** The prototype is scratch code. PR 1/PR 2 of the plan produce the committed ETKS and the benchmark rows.

The filter rerun is 2.6% above the published ETKF on S0 under the same metric (0.773 vs 0.753). Plausible causes are the #291 transform fix and a different initial-ensemble draw. The comparisons above are all paired within this rerun.

## Consequences for the plan

These are recorded in `docs/plans/tech/l96_etks_smoother.md`:
- `correct` is the default mode, `none` is kept for L ≤ 2 only, and the ensemble RTS (PR 1b) is dropped.
- PR 2 scores with the benchmark metric and tunes λ on validation windows: at λ = 1.5 the S0 filter is over-dispersed.
- The calibration-identity reading of S0 needs a calibrated filter first.
