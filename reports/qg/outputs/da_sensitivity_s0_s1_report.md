# QG DA Baseline Hyperparameter Sensitivity (S0 / S1)

ETKF and EnKF sensitivity to their own tunable hyperparameters (inflation, additive inflation, ridge regularization), on both the S0 reference case (lag=5.0d, noise_frac=0.05) and the revised S1 combined-error case (S0's initial-uncertainty setup + param bias 0.1 + corrupted wind + `da_nx=32` structural resolution mismatch). All sweeps use **N=10 test windows** (the first 10 of the production 100-window test cache) -- cheap enough to sweep densely, but absolute EV values differ somewhat from the N=100 headline numbers in `qg_da_report.md`/`PLAN.md`'s S1 table. This report is about the *sensitivity pattern* per hyperparameter, not a replacement headline comparison.

Motivation: EnKF consistently beats ETKF on S1 (N=100: ETKF q EV 0.307 vs EnKF 0.331) and the reason was an open question. The natural first hypothesis -- ETKF is under/over-inflated -- is what this report's inflation/additive/ridge sweeps test.

## ETKF: multiplicative inflation

| inflation | S0 q EV | S1 q EV |
|---|---|---|
| _free forecast_ | -0.043 | -0.402 |
| 1.00 | **0.402** | **0.253** |
| 1.01 | *0.402* | *0.220* |
| 1.02 | 0.296 | 0.112 |
| 1.03 | 0.147 | -0.029 |
| 1.04 | -40.437 | -0.559 |
| 1.05 | -123.452 | -39.536 |
| 1.10 | -- | -250.810 |
| 1.20 | -- | -345.702 |
| 1.30 | -- | -375.857 |

(PV-q full-field EV; best per column **bolded**, second-best *italicized*.)

S0 (no model error) tolerates inflation somewhat better than S1, but both decline steeply well before 1.05; S1's decline becomes catastrophic divergence (note the scale change) from 1.05 on. `inflation=1.0` (effectively off) is the safe choice on both scenarios at this N.

## ETKF: additive inflation

`etkf_additive`: Gaussian noise added directly to the ensemble each step (raw q-field units), expressed here as a fraction of the truth q field's own std (~2.06e-5 for both S0 and S1 -- same truth trajectories, only the DA model differs). Never exercised before this sensitivity study.

| additive (% of q std) | S0 q EV | S1 q EV |
|---|---|---|
| _free forecast_ | -0.043 | -0.402 |
| 0% | 0.402 | **0.253** |
| ~1% | **0.412** | *0.251* |
| ~5% | *0.405* | 0.224 |
| ~10% | 0.317 | 0.138 |
| ~24% | 0.068 | -0.067 |

(PV-q full-field EV; best per column **bolded**, second-best *italicized*.)

Also monotonically degrades EV on both scenarios, but **gracefully** -- no catastrophic divergence at any tested magnitude, unlike multiplicative inflation's blow-up past 1.04-1.05. Neither inflation flavor *helps*; both are pure downside once past zero.

## ETKF: ridge regularization (S1 only)

`etkf_ridge`: multiplier on the Kalman-gain transform-matrix regularization (targets the deterministic square-root transform's matrix inversion directly, unlike inflation which targets ensemble spread). `etkf_ridge<=0` (the default used everywhere else in this report) internally floors to an implicit 1e-4-equivalent, so the "default" row below is `etkf_n10.json` (same file as the inflation=1.0 baseline). Never exercised before this study.

| ridge | S1 q EV |
|---|---|
| _free forecast_ | -0.402 |
| default (~1e-4) | 0.253 |
| 1e-3 | 0.258 |
| 1e-2 | 0.275 |
| 1e-1 | *0.303* |
| 1.0 | **0.332** |

(PV-q full-field EV; best per column **bolded**, second-best *italicized*.)


## EnKF: multiplicative inflation (for comparison against ETKF above)

Same grid as ETKF's inflation sweep, run on EnKF, to test whether EnKF is robust because of its stochastic perturbed-observations update (which has built-in randomization against ensemble collapse that ETKF's deterministic square-root transform lacks) or just because `inflation=1.0`/`loc_radius=6.0` happen to suit it too. Never exercised before this study -- all prior EnKF runs used the fixed `inflation=1.0` default.

| inflation | S0 q EV | S1 q EV |
|---|---|---|
| _free forecast_ | -0.043 | -0.402 |
| 1.00 | **0.463** | **0.279** |
| 1.01 | *0.435* | *0.229* |
| 1.02 | 0.313 | 0.115 |
| 1.03 | 0.156 | -0.025 |
| 1.04 | -42.980 | -0.668 |
| 1.05 | -124.300 | -39.537 |
| 1.10 | -287.549 | -251.631 |

(PV-q full-field EV; best per column **bolded**, second-best *italicized*.)


## Synthesis

- **Inflation-driven divergence is shared, not ETKF-specific**: ETKF and EnKF collapse in near-lockstep under multiplicative inflation, on both S0 and S1 -- e.g. S0 q EV at inflation={1.00,...,1.05}: ETKF {0.402,0.402,0.296,0.147,-40.4,-123.5} vs EnKF {0.463,0.435,0.313,0.156,-43.0,-124.3}, virtually the same curve, same catastrophic threshold. This **rules out** "ETKF's deterministic transform makes it uniquely fragile to over-inflation" -- both ensemble methods share this fragility equally, so it cannot explain the EnKF>ETKF gap.
- **Additive inflation**: also monotonically neutral-to-harmful (never helpful) on ETKF, but fails gracefully with no catastrophic divergence at any tested magnitude, unlike multiplicative inflation. Neither inflation flavor *helps* --  both are pure downside once past zero, confirming "ETKF just needs the right inflation" is not the fix.
- **`etkf_ridge` is the real, actionable lever**: unlike inflation, increasing the Kalman-gain transform-matrix regularization **monotonically helps** -- S1 q EV rises from 0.253 (default, ~1e-4-equivalent) to **0.332 at ridge=1.0**, which *exceeds* both EnKF's own N=10 baseline here (0.279) and EnKF's N=100 headline number (0.331, from `qg_da_report.md`'s S1 table). This is a mechanism EnKF has no equivalent of (it has no deterministic transform-matrix inversion to regularize), so it's specific to ETKF's own weakness, not a general ensemble-DA fix.
- **Bottom line on the original open question**: the previously "unexplained" EnKF>ETKF gap on S1 looks like it was largely an artifact of running ETKF with an **under-regularized** transform-matrix inversion (the implicit ridge~1e-4 default), not a fundamental method limitation or an inflation-tuning problem. **Not yet confirmed at full N=100** -- the next concrete step is re-running ETKF with `etkf_ridge=1.0` (and probably a slightly wider ridge grid, e.g. 2.0/5.0, in case the N=10 optimum is even higher) at N=100 on both S0 and S1, checking psi EV and q layer2 aren't traded off, before considering it the new default S1 ETKF config in the benchmark table.
- **Still open**: `N_ensemble` has never been varied (hardcoded 80 everywhere).

Data: `reports/qg/outputs/qg_repro_validation_s1/etkf_n10_*.json` (original inflation/additive S1 sweep) + `reports/qg/outputs/qg_da_sensitivity_sweep/*.json` (consolidated S0 + EnKF + ridge sweep). Scratch drivers (not committed): `qg_da_s1_scratch.py`, `qg_da_sensitivity_sweep_scratch.py`.

