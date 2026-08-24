# Phase B — CFM Architecture Variants for L96 (design doc)

**Status:** DESIGN ONLY — not implemented. Low priority. Per `PLAN.md`
"Deferred future work", any Phase B implementation requires this design doc
first and must define precise variants + success criteria before coding.

**Question:** does a Tweedie-style two-stage decomposition or a diffusion-style
variant improve on `VanillaCFM` for L96?

---

## Grounded anchors (existing code)

- **`TweedieSolver`** (`models/solver.py:7`, legacy two-stage solver):
  Stage 1 `estimate_mean` (`MeanEstimatorCell`, `models/residual.py:63`)
  computes the conditional mean E[x1|obs] via `K_inner` residual iterations.
  Stage 2 `forward` (`IterativeUpdateCell`, `models/residual.py:6`) refines a
  Kalman-blended state `(1−K)·mean + K·x` with weighted residuals and Euler
  integration (`N_outer` steps). Two-stage training: stage 1 trains
  `mean_estimator` (MSE on the mean), stage 2 freezes it and trains
  `non_gaussian` (MSE on the sampled trajectory). See `lightning_module.py:31-73`.
- **`VanillaCFM`** (`models/vanilla_cfm.py:21`): single UNet predicts the
  velocity field directly; loss `MSE(v_pred, x1 − x0)` over random τ (or τ=0 if
  `train_tau_0_only`); sampling = Euler integration over `N_outer` steps.
- **`LinearInterpolant`** (`models/interpolant.py`): linear path
  `x_tau = (1−τ)x0 + τx1`, `alpha=1−τ`, `beta=τ`. Shared by both. `gain_matrix`,
  `ng_prefactor`, `compute_drift` are Tweedie-specific.
- **`UNet1D`** (`models/unet.py:102`): shared backbone; already has a τ time
  embedding and `output_dim`/`cond_extra_dim` knobs.

## Reference bars (cached DA-parity test set, 24D, Obs30)

| Scheme | S0 RMSE | S1 RMSE | Notes |
|---|---|---|---|
| L4 DirectUNet small | 0.619 | 0.621 | deterministic frontier |
| L1b DirectUNet | 0.622 | 0.625 | |
| L2b VanillaCFM τ=0 | 0.633 | 0.633 | |
| L3 VanillaCFM multi-τ (ens30×10) | **0.5643** | 0.5667 | ensemble (sampling + integration) |
| Best DA Strong-4DVar | 0.742 | 1.432 | S1/S0 ≈ 1.9× |

Q1 finding: multi-τ CFM only beats conditional-mean estimation when trained
multi-τ **and** integrated with 10 Euler steps **and** averaged over 30 members.
At 1 member × 1 step it is worse than τ=0 (0.688 vs 0.633).

## Proposed variants

### V1 — L96 TweedieSolver (obs-only, `use_energy=false`)

Port the legacy two-stage solver to `state_dim=24`, obs-only (`cond_extra_dim=0`
to match L1b/L2b), train 2-stage (stage1 mean, stage2 residual). Direct test:
does the mean+residual decomposition beat single-stage conditional-mean
estimation (L1b/L4)?

- **Design questions to resolve before coding:** (a) `cond_extra_dim` must be
  plumbed through `MeanEstimatorCell`/`IterativeUpdateCell`/`TweedieSolver`
  (currently not accepted — `residual.py:7,64`, `solver.py:8`); set to 0 for
  obs-only. (b) `use_energy=true` is vacuous with no operators (adds 3·24 zero
  channels — `solver.py:64-83`); set `use_energy=false`. (c) stage-1 epoch
  budget (two-stage at L63 200+400 would triple the wall-clock vs L2b's 400);
  consider a shorter stage 1. (d) 24D `MeanEstimatorCell` has never been
  trained/tested (all tests are state_dim=3); validate receptive field /
  `K_inner`. (e) **evaluation bar:** single-sample (must beat 0.619/0.622), or
  ensemble (would need a multi-member path for tweedie — currently none).

### V2 — CFM + Tweedie residual hybrid

Stage 1 = `MeanEstimatorCell` (conditional mean, frozen after stage 1); stage 2
= a CFM velocity field trained on the **residual** `x1 − mean(obs)` with the
standard `compute_cfm_loss` (NOT Tweedie's MSE-on-trajectory). Tests whether the
Tweedie *decomposition* (mean + residual) helps when the residual is learned via
flow matching rather than direct MSE.

- **Design questions:** (a) does stage 2 CFM train multi-τ or τ=0? (b) success
  criterion: if V2 supports multi-member sampling, must it beat L3 ens30×10
  (0.5643)? If single-sample, must it beat L1b/L4 (0.619/0.622)? (c) how is the
  mean made conditioning-aware (concatenate `mean(obs)` to the UNet input)?

### V3 — Diffusion-style variant (DEFERRED / out of scope for this plan)

No diffusion scaffolding exists in the repo (no noise schedule, no ε/score
prediction). Recommending a separate design + effort. The only generative
scaffolding is `LinearInterpolant` (linear path). If V1/V2 show promise, revisit.

## Cross-cutting requirements (any variant)

- Multi-member sampling path (for ensemble comparison vs L3 ens30): Tweedie
  currently has no `--n-members`/`--n-outer` support in `eval_neural_l96.py`.
- S1/S0 degradation must remain ≈ 1.0.

## Decision

Draft only. Before any Phase B implementation, confirm V1/V2 scope, resolve the
design questions above, and record success criteria. Do not re-run the previous
session's under-defined "V2/V3" without this doc.
