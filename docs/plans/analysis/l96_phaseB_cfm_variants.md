# Phase B — CFM Architecture Variants for L96 (design doc)

**Status:** DESIGN + PARTIAL IMPLEMENTATION. V2/V3 code restored on branch
`feature/l96-v2v3-pure`; **training + eval blocked** (see Blockers). This doc
records the resolved design decisions and the gaps found during the
2026-08-27 review so the next implementation pass is well-defined.

**Question:** does a Tweedie-style two-stage decomposition or a predict-μ ODE
formulation improve on `VanillaCFM` for L96 state estimation (24D, obs-only)?

---

## Grounded anchors (existing code)

- **`TweedieSolver`** (`models/solver.py:7`, legacy two-stage solver):
  Stage 1 `estimate_mean` (`MeanEstimatorCell`, `models/residual.py:63`)
  computes the conditional mean E[x1|obs] via `K_inner` residual iterations.
  Stage 2 `forward` (`IterativeUpdateCell`, `models/residual.py:6`) refines a
  Kalman-blended state with weighted residuals and Euler integration. **Not
  used by V2/V3** — kept for reference.
- **`VanillaCFM`** (`models/vanilla_cfm.py:21`): single UNet predicts the
  velocity field `v = x1 − x0`; loss `MSE(v_pred, x1 − x0)` over random τ (or
  τ=0 if `train_tau_0_only`); sampling = Euler integration over `N_outer`
  steps. The L3 multi-τ baseline.
- **`LinearInterpolant`** (`models/interpolant.py`): linear path
  `x_τ = (1−τ)x0 + τx1`, `alpha=1−τ`, `beta=τ`. Shared by all variants.
- **`UNet1D`** (`models/unet.py:102`): shared backbone; `obs_dim`,
  `cond_extra_dim`, `output_dim` knobs. `forward(x, obs=, tau=)` expects
  `(B, C, L)` (channels-first); callers transpose `(B, T, D) → (B, D, T)`.

## Reference bars (cached DA-parity test set, 24D, Obs30, 200 windows)

| Scheme | S0 RMSE | S1 RMSE | Notes |
|---|---|---|---|
| L4 DirectUNet small | 0.619 | 0.621 | deterministic frontier |
| L1b DirectUNet | 0.622 | 0.625 | |
| L2b VanillaCFM τ=0 | 0.633 | 0.633 | single-sample CFM |
| L3 VanillaCFM multi-τ (1-sample) | 0.688 | 0.690 | single-sample, 1 step |
| L3 VanillaCFM multi-τ (ens30×10) | **0.5643** | 0.5667 | ensemble (sampling + integration) |
| Best DA Strong-4DVar | 0.742 | 1.432 | S1/S0 ≈ 1.9× |

Q1 finding: multi-τ CFM only beats conditional-mean estimation when trained
multi-τ **and** integrated with 10 Euler steps **and** averaged over 30
members. At 1 member × 1 step it is worse than τ=0 (0.688 vs 0.633).

## Implemented variants (on `feature/l96-v2v3-pure`)

The branch restored two variants (V1 "port TweedieSolver" was superseded by
the implemented V2 hybrid, which is the doc's intended comparison).

### V2 — `TweedieCFM` (mean-estimator + CFM-residual hybrid)

Stage 1 = `MeanEstimatorCell` (conditional mean, frozen after stage 1); stage
2 = a CFM velocity UNet trained on the **residual** `x1 − mean(obs)` with the
standard `compute_cfm_loss` (target `v = residual − x0`, NOT Tweedie's
MSE-on-trajectory). Sampling: `mean + CFM_sample(residual)`.

**Resolved design decisions:**
- (a) **Stage 2 trains multi-τ** (`train_tau_0_only: false`) to match L3,
  enabling the ensemble/integration comparison that is the whole point.
- (b) **Success criterion:** single-sample must beat L1b/L4 (0.619/0.622);
  ens30×10 must beat L3 ens30×10 (0.5643). Both bars.
- (c) **Mean conditioning:** `cond = cat([obs, mean], dim=-1)`, so the
  velocity UNet's `obs_dim = 2·state_dim = 48` (obs + mean concatenated). The
  `cond_extra_dim=0` knob is *not* used for the mean — it rides in `obs_dim`.
- (d) **Stage-1 epoch budget: 100** (not 200). Halves the two-stage wall-clock
  vs the original config; the mean estimator is a simpler target than the full
  state, so 100 epochs should suffice. Stage 2 = 400 epochs (matches L2b/L3).
- (e) **K_inner=5 at 24D** is unvalidated (all `MeanEstimatorCell` tests are
  state_dim=3). Risk: inadequate receptive field. Mitigation: a 1-epoch smoke
  test must run before full training; if stage-1 loss doesn't drop, revisit.

**Open implementation concern (must verify before training):** the
`MeanEstimatorCell.forward(x, obs, tau)` signature vs `TweedieCFM.estimate_mean`
calls `self.mean_estimator(x, obs.transpose(1,2), tau)`. `MeanEstimatorCell`
forwards `obs` straight to `UNet1D.forward(x, obs=...)` which expects
channels-first `(B, C, L)`. The transpose is correct *if* the rest of the
pipeline is channels-last. **A unit test must confirm the shapes line up**
before training — this is a likely silent bug.

### V3 — `PredictStateCFM` (predict-μ ODE formulation)

Single-stage. The network predicts `μ = E[x1 | x_τ, y]` directly; `compute_loss
= MSE(μ_pred, x1)` (NOT the standard CFM `v = x1 − x0` loss). Sampling
integrates `v = (μ − x)/(1 − τ)` forward over `N_outer` steps. A different ODE
formulation of the same linear interpolant — *not* the doc's deferred
diffusion V3 (no noise schedule / score prediction exists in the repo).

**Resolved design decisions:**
- **Scope: keep V3.** It's a cheap, clean single-stage comparison vs V2's
  two-stage and L2b's predict-v. 400 epochs, single-stage.
- **τ mode: multi-τ** (`train_tau_0_only: false`) — same rationale as V2.
- **Success criterion:** same bars as V2 (L1b/L4 single-sample, L3 ens30×10).
- **`output_dim=state_dim`** on the UNet (predicts the state, not a velocity).
- Note: `sample` clamps `1 − τ` at 0.999 on the last step (target τ=1 never
  quite reached) — consistent with `VanillaCFM.sample`, not a blocker.

### V1 — L96 TweedieSolver (NOT implemented; superseded by V2)

The doc's original "port the legacy two-stage solver" is superseded by V2,
which reuses `MeanEstimatorCell` for stage 1 and replaces the Tweedie stage-2
with a CFM velocity field. If V2 shows the mean+residual decomposition helps,
a full Tweedie port (Kalman-blended IterativeUpdateCell stage 2) could be
revisited as V1. Not in scope for this pass.

## Blockers (must fix before training/eval)

### B1 — `train.py` has unresolved git merge conflict markers (CRITICAL)

Commit `f7749a9` ("fix: add smoke_cached_data extraction from DataConfig")
introduced unresolved `<<<<<<<`/`=======`/`>>>>>>>` markers at lines 358-396
and 410-464 of `train.py`. The file does not compile (`SyntaxError: unmatched
')'`). **No training has ever run on this branch** — the CHANGELOG's
"dataset generation hang" diagnosis is incorrect; the actual failure is a
broken source file.

**Fix:** resolve the two conflict regions. Both sides add a
`smoke_cached_data` short-circuit before the `make_*_trainval` call; the
`HEAD` side is the pre-smoke-cached version. Keep the `smoke_cached_data`
branch (with correct 4-space indentation — the conflict introduced
6-space-indented lines that won't parse) and drop the `=======`/`>>>>>>>`
duplicates. After resolution, `python -m py_compile train.py` must pass.

### B2 — Dataset generation cost is ~3.7h, not a hang

Root cause confirmed by timing: `generate_full_trajectory` is pure-Python RK4
over `spinup_steps=10000 + num_steps=300` = 10,300 steps/window, ~8.9 s/window
on CPU. For the V2/V3 config (1000 train + 100 val + 200 s0 + 200 s1 = 1500
windows) that's ~3.7 h of CPU generation before epoch 1. On a shared cluster
node with filesystem contention this can look like a hang, but it is the
expected cost. **The `cached_datasets` kwarg in `make_l96_s0_s1_trainval`
already supports loading a pre-built cache** — the fix is to pre-generate the
training dataset once (or reuse the eval cache `experiments/l96_datasets_obsj2_int100_nwin200.pt`,
which only has test windows — train/val need building) and point
`smoke_cached_data` at it. The `experiments/` dir is currently empty; no
`.pt` cache exists yet.

**Fix:** (1) resolve B1 so `smoke_cached_data` actually loads; (2) build a
train+val cache (e.g. a one-off CPU job or a 1-epoch warmup that saves the
generated dataset) and reference it from V2/V3 configs; or (3) accept the
~3.7h generation as part of the training job wall-clock.

### B3 — No V2/V3 unit tests

`grep` for `TweedieCFM|PredictStateCFM|tweedie_cfm|predict_state_cfm` in
`tests/` returns nothing. Any new model needs at least: forward shape, sample
shape, stage dispatch (V2), loss sanity, τ=0 invariance (V3). Cheap, catches
the B1-adjacent `MeanEstimatorCell` transpose concern before training.

### B4 — Eval pipeline has zero V2/V3 support

`evaluation/neural_inference.py` and `eval_neural_l96.py` have no
`tweedie_cfm`/`predict_state_cfm` branches (grep returns nothing). Even after
training, checkpoints can't be loaded/inferred. Must add: model resolution,
checkpoint loading (stage-1 vs stage-2 ckpt for V2), `sample()` dispatch, and
`--n-members`/`--n-outer` passthrough for the ens30 comparison.

## Cross-cutting requirements

- **Multi-member sampling path** (for ensemble comparison vs L3 ens30×10):
  both `TweedieCFM.sample` and `PredictStateCFM.sample` draw fresh x0 per
  call, so `--n-members` just loops `sample()`. Wire into
  `eval_neural_l96.py`'s existing member loop (mirrors `VanillaCFM`).
- **S1/S0 degradation must remain ≈ 1.0** (matches all other neural models).
- **Report:** regenerate `reports/l96/outputs/l96_consolidated_benchmark.md`
  with V2/V3 rows (single-sample + ens30×10) once eval lands.

## Implementation order (next session)

1. **Fix B1** (resolve `train.py` merge conflicts; `py_compile` must pass).
2. **Fix B3** (add V2/V3 unit tests; confirm `MeanEstimatorCell` shapes).
3. **Fix B2** (build train+val cache OR accept generation cost; update V2/V3
   configs to reference the cache via `smoke_cached_data`).
4. **1-epoch smoke** on login GPU (sl-mee-br-202/204) for V2 and V3 to
   validate the full train path before launching full training.
5. **Launch full training** via `batch/run_l96_cfm_variants_train.sbatch`
   (V2 stage-1 epochs → 100; V3 single-stage 400).
6. **Fix B4** (build eval pipeline for V2/V3 in `evaluation/neural_inference.py`
   + `eval_neural_l96.py`).
7. **Evaluate** single-sample (bar: L1b/L4 0.619/0.622) and ens30×10 (bar:
   L3 0.5643) on the cached S0/S1 test set.
8. **Regenerate** the consolidated report with V2/V3 rows.

## Decision log (2026-08-27)

- V2 stage-1 budget: **100 epochs** (halve from 200; simpler target).
- V3 scope: **keep** (cheap, clean predict-μ comparison; not the doc's
  deferred diffusion).
- Training unblock: **debug the hang first** — root cause found (B1 broken
  `train.py` + B2 ~3.7h generation, not a true hang).
- Execution: **update this doc first**, then implement code in follow-up.
