# QG DA Baselines (S0 reference case + S1 model-error case)

Reference case (2026-09-08): **lag=5.0d, noise_frac=0.05, N=100 test windows** (S0). Supersedes the earlier lag=1.0d/noise=0.01 case, which was found to be unrealistically favorable -- at that setting the background (free forecast, no assimilation) alone already reached psi full EV≈0.98, so DA's high score there was mostly inherited from the background rather than earned from the observational update. See `PLAN.md`'s "DA reference-case realism" section for the full lag/noise sensitivity analysis behind this choice.

**S1** (revised 2026-09-10): shares S0's initial-uncertainty setup exactly (lag=5.0d, noise_frac=0.05) and adds three independent model-error sources on top -- param bias (`rd`/`rek` scaled by `1-0.1`), corrupted wind (amplitude bias + storm-location jitter), and structural resolution mismatch (`da_nx=32`, half the truth grid). Obs are always drawn from the true, unbiased, full-res trajectory -- only the DA method's own model sees the corruption.

**ETKF default changed 2026-09-12**: `etkf_ridge=1.0` (was the implicit ~1e-4 floor at `etkf_ridge<=0`) -- see the "Hyperparameter sensitivity analysis" section below. All ETKF numbers in this report use the new default.

## DA baselines (S0, PV-q and ψ)

CRPS is computed per-window on the q-state: ensemble methods (ETKF/EnKF) score their real per-member spread; deterministic methods (4DVar) have no ensemble, so CRPS degenerates exactly to the mean absolute error (marked `*`) -- lower is better for both. CRPS (norm.) divides by the pooled truth PV std over the whole test set (not per-window -- avoids the distortion a low-variance window would introduce), giving a dimensionless, cross-method-comparable score.

| method | PV RMSE | improv | CRPS | CRPS (norm.) | PV EV | PV q1 EV | PV q2 EV | ψ EV |
|---|---|---|---|---|---|---|---|---|
| _free forecast_ | 1.93e-05 | 1.0 | -- | -- | -0.0312 | -0.0443 | -0.0180 | 0.8854 |
| EnKF | **1.27e-05** | **1.5153** | **5.73e-06** | **0.2818** | **0.5000** | **0.5566** | **0.4434** | 0.9584 |
| ETKF | *1.27e-05* | *1.5105* | *5.75e-06* | *0.2830* | *0.4982* | *0.5539* | *0.4424* | 0.9593 |
| Weak-4DVar | 1.41e-05 | 1.3691 | 9.19e-06* | 0.4524* | -0.0345 | 0.4677 | -0.5367 | *0.9660* |
| Strong-4DVar | 1.40e-05 | 1.3793 | 9.22e-06* | 0.4538* | -0.1257 | 0.4830 | -0.7344 | **0.9714** |

(Best per column **bolded**, second-best *italicized*, ranked among the 4 DA methods -- the free-forecast reference row above is excluded.)

> **Caveat:** Weak-4DVar, Strong-4DVar collapse on PV q layer2 (the unobserved lower layer) at this reference case -- their PV EV is negative there despite psi EV being competitive or the best of all 4 methods. This is consistent with PV being a Laplacian-like operator on psi (q ≈ ∇²ψ): small high-wavenumber errors in an otherwise excellent psi analysis get amplified when inverted to PV, especially in the layer with no direct observations.

Computed on N=100 test windows, lag=5.0, noise_frac=0.05 (should match the reference case above -- if not, these JSONs are stale, regenerate them).

## DA baselines (S1, PV-q and ψ)

CRPS is computed per-window on the q-state: ensemble methods (ETKF/EnKF) score their real per-member spread; deterministic methods (4DVar) have no ensemble, so CRPS degenerates exactly to the mean absolute error (marked `*`) -- lower is better for both. CRPS (norm.) divides by the pooled truth PV std over the whole test set (not per-window -- avoids the distortion a low-variance window would introduce), giving a dimensionless, cross-method-comparable score.

| method | PV RMSE | improv | CRPS | CRPS (norm.) | PV EV | PV q1 EV | PV q2 EV | ψ EV |
|---|---|---|---|---|---|---|---|---|
| _free forecast_ | 2.20e-05 | 1.0 | -- | -- | -0.3822 | -0.3091 | -0.4554 | 0.6760 |
| EnKF | **1.39e-05** | **1.5854** | **6.39e-06** | **0.3144** | **0.4126** | **0.4844** | **0.3408** | 0.9410 |
| ETKF | *1.39e-05* | *1.5829* | *6.40e-06* | *0.3150* | *0.4116* | *0.4828* | *0.3404* | *0.9424* |
| Weak-4DVar | 1.62e-05 | 1.3587 | 1.09e-05* | 0.5378* | -0.5008 | 0.3214 | -1.3229 | **0.9468** |
| Strong-4DVar | 1.68e-05 | 1.3097 | 1.15e-05* | 0.5648* | -0.8573 | 0.2819 | -1.9966 | 0.9314 |

(Best per column **bolded**, second-best *italicized*, ranked among the 4 DA methods -- the free-forecast reference row above is excluded.)

> **Caveat:** Weak-4DVar, Strong-4DVar collapse on PV q layer2 (the unobserved lower layer) at this reference case -- their PV EV is negative there despite psi EV being competitive or the best of all 4 methods. This is consistent with PV being a Laplacian-like operator on psi (q ≈ ∇²ψ): small high-wavenumber errors in an otherwise excellent psi analysis get amplified when inverted to PV, especially in the layer with no direct observations.

Computed on N=100 test windows, lag=5.0, noise_frac=0.05 (should match the reference case above -- if not, these JSONs are stale, regenerate them).

## Hyperparameter sensitivity analysis

Condensed summary of the 2026-09-11/12 sensitivity study -- full sweep tables (finer inflation grids, additive inflation, the full ridge grid, EnKF's own inflation sensitivity) are in the dedicated `reports/qg/outputs/da_sensitivity_s0_s1_report.md` (`reports/qg/generate_da_sensitivity_report.py`); `PLAN.md` has the full narrative trail.

**Motivation**: EnKF consistently beat ETKF on S1 with no known cause (N=100: ETKF q EV 0.307 vs EnKF 0.331, at ETKF's *old* default). The natural first hypothesis -- ETKF is under/over-inflated -- turned out not to explain it.

**ETKF/EnKF inflation (multiplicative)**: both methods collapse in near-lockstep under any inflation above 1.0 -- e.g. S0 q EV at inflation={1.00,...,1.05}: ETKF {0.402,0.402,0.296,0.147,-40.4,-123.5} vs EnKF {0.463,0.435,0.313,0.156,-43.0,-124.3} (N=10), virtually the same curve, same catastrophic threshold. This rules out "ETKF's deterministic transform is uniquely fragile to over-inflation" as an explanation -- both ensemble methods share the fragility equally, so it cannot explain the EnKF>ETKF gap. **Additive inflation** (never exercised before this study) also only ever hurts, though gracefully (no catastrophic divergence).

**`etkf_ridge` (Kalman-gain transform-matrix regularization) is the real lever** -- a mechanism EnKF has no equivalent of. Monotonically helps up to a peak (S1 peaks around ridge=2.0, S0 plateaus 0.1-1.0), then mildly declines. `ridge=1.0` was picked as near-optimal on both scenarios. **Confirmed at full N=100** (not just the N=10 sweep):

| | ETKF (pre-2026-09-12 default) | EnKF (at loc=6.0) | ETKF + ridge=1.0 (2026-09-12 default) |
|---|---|---|---|
| S0 psi EV | 0.921 | 0.947 | **0.957** |
| S0 q EV | 0.405 | **0.481** | 0.476 |
| S1 psi EV | 0.874 | 0.896 | **0.926** |
| S1 q EV | 0.307 | 0.331 | **0.357** |

ETKF+ridge=1.0 beat EnKF outright on **both** fields on S1, and tied/beat it on S0 -- no trade-off on the unobserved PV layer either. This is why `etkf_ridge=1.0` was promoted to the default on 2026-09-12: the previously "unexplained" EnKF>ETKF gap was largely an artifact of ETKF running with an under-regularized transform-matrix inversion, not a fundamental method limitation. **Superseded by the 2026-09-15 `loc_radius` re-tune below** -- all four numbers in this table used `loc_radius=6.0`, which turned out to be untuned even at this reference density.

**`loc_radius=6.0` itself was never tuned -- also superseded (2026-09-15)**: the obs-density sensitivity study (cols=1/2/16/64, `da_sensitivity_s0_s1_report.md`) found `loc_radius=2.0` near-universally beats `6.0`, and checking it at the reference density itself (cols=4) confirmed the same -- `loc_radius=6.0` was the *worst* point in a {1,2,3,4,6} grid for both methods. Once `loc_radius` moved to 2.0, `etkf_ridge` needed re-checking too (the two regularizers interact): a fresh ridge sweep found `ridge=1.0`'s benefit had inverted -- q/q-layer2 now decline monotonically from `ridge=0` (the implicit floor) upward, with psi flat throughout. `inflation=1.0` was re-checked at the new config and remains optimal for both methods (unaffected). **New current default (2026-09-15): `loc_radius=2.0`, `etkf_ridge=0.1` (ETKF only; EnKF has no ridge), `inflation=1.0` unchanged** -- confirmed at full N=100:

| | ETKF (loc=6.0, ridge=1.0) | EnKF (loc=6.0) | ETKF (loc=2.0, ridge=0.1) | EnKF (loc=2.0) |
|---|---|---|---|---|
| S0 psi EV | 0.957 | 0.947 | 0.959 | **0.958** |
| S0 q EV | 0.476 | 0.481 | *0.498* | **0.500** |
| S1 psi EV | 0.926 | 0.896 | **0.942** | *0.941* |
| S1 q EV | 0.357 | 0.331 | *0.412* | **0.413** |

ETKF and EnKF are now nearly indistinguishable at this reference density once both are properly localized -- the EnKF>ETKF gap (and later ETKF>EnKF gap after the ridge promotion) were both artifacts of running at an untuned `loc_radius=6.0`. (S0/S1 q-layer2 EV, the previously-collapsing unobserved deep layer, moves even more: ETKF 0.410->0.442, EnKF 0.394->0.443 at S0; ETKF 0.266->0.340, EnKF 0.221->0.341 at S1 -- see `da_sensitivity_s0_s1_report.md` for the full trail.) Full sweep + N=100 confirmation tables are in the dedicated report; `PLAN.md` has the complete narrative.

**4DVar (Strong/Weak) covariance-weighting sweep -- negative result**: motivated by the same logic (both 4DVar variants collapse on PV q layer2 under S1: Strong -0.857, Weak -0.501 vs ETKF/EnKF staying positive), swept Strong-4DVar's `b_var_scale` (background-covariance whitening scale) and Weak-4DVar's `q_var_scale` (per-step model-error weight) on S1, N=5 (4DVar is ~15-20x more expensive per window than ETKF/EnKF). Unlike ETKF's ridge, **neither lever helps**: `b_var_scale` is essentially flat across 0.3-3.0 (q EV -1.06/-1.07/-1.11); `q_var_scale` is monotonically *worse* the higher it's pushed above the existing default 0.1 (q EV -0.91/-0.99/-1.09/-1.11 at 0.1/0.3/1.0/3.0) -- giving the model-error controls more freedom actively hurts rather than helping. Both methods' existing defaults (`b_var_scale=1.0`, `q_var_scale=0.1`) were already at or near the best point found. This is consistent with the 4DVar q-layer2 collapse being a more structural limitation (a single optimized trajectory has no ensemble spread to exploit on the unobserved layer) rather than a fixable covariance-tuning gap, unlike ETKF's case.

**ETKF/EnKF obs-density/configuration sensitivity -- a `loc_radius`-tuning lesson, not a new physical effect (2026-09-13/14)**: an initial N=10 sweep varying `cols_per_day` and a new independent lower-layer (psi2) point-observation stream looked like it found "more upper-layer density destabilizes S1" and "psi2 observations are uniquely valuable" -- **both retracted** after user pushback prompted an inflation check, a `cols=64` run, and a `loc_radius` sweep. The real story: `loc_radius=6.0` was tuned for the sparse 4-8/day regime and becomes an ensemble-conditioning bottleneck at higher density with a fixed small ensemble (N=80) -- shrinking it fully recovers and then exceeds the original baseline at both cols=16 and cols=64, on both S0 and S1. Whether psi2 information adds value *at matched total density* against a properly-tuned pure-psi1 config remains genuinely open. Full correction trail in `da_sensitivity_s0_s1_report.md`'s dedicated section.

## Synthesis: best configuration per method (S0 vs S1)

Following the sensitivity study above, every method's *best known* config is now also its *shipped default* -- both ETKF's and EnKF's default changed on 2026-09-15 (`loc_radius` 6.0->2.0, plus `etkf_ridge` 1.0->0.1 for ETKF); Strong-4DVar and Weak-4DVar were already at their best tested configuration.

| method | best config | S0 ψ EV | S0 PV EV | S1 ψ EV | S1 PV EV |
|---|---|---|---|---|---|
| EnKF | N=80, inflation=1.0, **loc_radius=2.0** (default since 2026-09-15; was loc_radius=6.0, untuned) | 0.9584 | 0.5000 | 0.9410 | 0.4126 |
| ETKF | N=80, inflation=1.0, **loc_radius=2.0, etkf_ridge=0.1** (default since 2026-09-15; was loc_radius=6.0, etkf_ridge=1.0) | 0.9593 | 0.4982 | 0.9424 | 0.4116 |
| Weak-4DVar | b_var_scale=1.0, q_var_scale=0.1 (default, unchanged -- sweep over 0.1-3.0 found 0.1 already best, higher values monotonically worse) | 0.9660 | -0.0345 | 0.9468 | -0.5008 |
| Strong-4DVar | b_var_scale=1.0 (default, unchanged -- sweep over 0.3-3.0 found no improvement, roughly flat) | 0.9714 | -0.1257 | 0.9314 | -0.8573 |

(PV-q and ψ full-field EV, N=100 both scenarios. Not rank-marked here -- see the per-scenario tables above for bold/italic ranking; this table's purpose is the config-per-method mapping, not re-ranking.)

## Reconstruction examples

3 example test windows (best/median/worst by ETKF's per-window pooled PV-q RMSE) x all 4 methods, showing truth | free-forecast | analysis for streamfunction (ψ, both layers) and PV (q, both layers), plus an animated DA cycle (raw obs | wind-stress curl forcing | truth | EnKF analysis, PV q1) over the 30-day window. Generated by `generate_qg_reconstruction_figs.py`. S0 only.

### Best window

![best window reconstruction](figs/qg_s0_reconstruction_best.png)

![best window DA-cycle animation (EnKF)](figs/qg_s0_dacycle_best.gif)

### Median window

![median window reconstruction](figs/qg_s0_reconstruction_median.png)

![median window DA-cycle animation (EnKF)](figs/qg_s0_dacycle_median.gif)

### Worst window

![worst window reconstruction](figs/qg_s0_reconstruction_worst.png)

![worst window DA-cycle animation (EnKF)](figs/qg_s0_dacycle_worst.gif)

