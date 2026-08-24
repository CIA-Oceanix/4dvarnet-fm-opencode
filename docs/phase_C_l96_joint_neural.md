# Phase C — L96 Joint State-Parameter Neural Estimation (design doc)

**Status:** designed for execution.
**System:** Lorenz-96 (two-scale), observed subspace 24D (`obs_j=2`, Obs30).
**Models:** 3 joint neural estimators (state + 8 params), benchmarked against the
L96 joint DA baselines (`JointEnKFL96` / `JointETKFL96` / `JointStrong4DVarL96`).

---

## Motivation

The state-only L1b/L2b neural models already beat the best state-only DA baseline
on S0/S1 (RMSE 0.62 vs 0.74). The open question is joint state-parameter
estimation: can a neural estimator infer the 24D state **and** the model's
parameters from the observations alone, as well as (or better than) a joint DA
scheme that has the forward model? This mirrors the already-built L96 **joint DA**
extension and the L63 **joint CFM** (`JointCFM`) template.

## Parameters estimated (8, h fixed)

Following the DA baselines' convention (`_L96_JOINT_PARAM_DIM = 8`,
`evaluation/baselines.py:1893`): **F, c1, hx, eps + the 4 fast_weights**
(w1..w4). `h` is fixed at 1.0 (not estimated), exactly as in the joint DA
methods (`_l96_h_fixed`).

Per-window `param_names = [F, c1, hx, eps, w1, w2, w3, w4]` (GD truth read from
`true_F, true_c1, true_hx, true_eps, true_w1..true_w4`).

## Models

| ID | Model | τ mode | Backbone | Epochs | Mirrors state-only |
|---|---|---|---|---|---|
| L7 | `JointCFM` (port of `models/vanilla_cfm.py:75`) | τ=0 | [64,128,256] | 400 | L2b |
| L8 | `JointDirectUNet` (new, `models/direct_unet.py`) | n/a | [64,128,256] | 200 | L1b |
| L9 | `JointCFM` | multi-τ | [64,128,256] | 400 | L3 |

Dimension facts: `state_dim=24`, `param_dim=8`, `cond_extra_dim=1+8=9`
(forcing + 8 params), `output_dim = 24+8 = 32` on the UNet.

**State head:** `v_state = v[..., :24]` — velocity/state output (CFM) or direct
state estimate (UNet).
**Param head:** `params = softplus(mean_t(v[..., 24:]))` — time-averaged UNet
output passed through softplus (positivity), same convention as `JointCFM`.
No separate `nn.Module` head; it is a pooled slice of the UNet output.

**Losses (single-stage, no two-stage freeze):**
- `JointCFM`: `L = MSE(v_pred_state, states − x0) + w · MSE(softplus(mean_t v_params), true_params)`, `w = param_loss_weight = 0.1`.
- `JointDirectUNet`: `L = MSE(pred_state, states) + w · MSE(softplus(mean_t v_params), true_params)`.

## Dataset / param plumbing (WP1)

`fast_weights` is a list of 4 in the window dict. To keep the generic scalar
param-extraction path (`FlowMatchingDataset._extract_params`, which reads
`w.get(n)` / `w.get("true_{n}")`) working unchanged, each L96 window now also
stores per-index scalars via `_set_window_params`:

- current: `w1..w4` (plus existing `fast_weights` list)
- truth: `true_w1..true_w4`
- biased DA (S1): `w1_da..w4_da` (plus existing `fast_weights_da`)

These flatten fast_weights into 4 scalar keys so `param_names=[...,w1,w2,w3,w4]`
resolves without list-aware dataloader changes. The scalar `F, c1, hx, eps`
keys already existed.

## Dispatch wiring (WP3)

- `train.py model_factory`: new `joint_direct_unet` branch; `joint_cfm` branch
  already generic (reads `joint_cfm.param_dim`).
- `training/lightning_module.py _forward_and_loss`: new `joint_direct_unet`
  branch -> `model.compute_loss(batch)`; `joint_cfm` -> `compute_cfm_loss` (existing).
- `train.py evaluate_model` / `save_trajectories`: new `joint_direct_unet`
  branch (mirror the `joint_cfm return_params` path).
- `with_params` dataloader condition widened to both joint types (`train.py:374,385`).
- `is_joint` flag + `hc_src` config-capture widened to `joint_direct_unet`.

## Evaluation (WP5)

`eval_joint_neural_l96.py` (separate script, mirrors `eval_neural_l96.py`):
loads a joint checkpoint via `evaluation/neural_inference.load_model` (extended
to resolve/construct joint model types and infer `param_dim` from the checkpoint
weights), runs on the cached S0/S1 test set, and reports per case:

- state RMSE/EV/ES (pooled, per-group slow/obs_fast/all_obs) via `evaluate_estimates`
- 8-param RMSE vector + mean via `param_rmse` (params from `sample(return_params=True)`)

A **separate** script (not a modification of `eval_joint_comparison_l96.py`)
keeps the DA comparator stable. A report script merges both JSONs
(`joint_neural_eval.json` per model + `l96_joint_comparison.json` from the DA
side) — the merge/report step is a follow-up (possible future PR) since the
neural-training results are required first.

New `evaluation/neural_inference.py` joint support:
- `resolve_model_class`/`create_model`: `JOINTCFM`/`JOINTDIRECTUNET`.
- `load_checkpoint`: infer `param_dim = output_dim − state_dim` when the UNet
  output exceeds state (only joint models have it).
- `collate_joint_eval`: stacks the 8 params + true_params per window.
- `_run_case_inference`: handles joint models (call `sample(..., return_params=True)`,
  collect `params_pred`/`params_true`), with the joint-type `isinstance` check
  placed **before** `VanillaCFM` since `JointCFM` subclasses it.

## Configs (WP4)

`config/experiment/{L7_joint_cfm_s0s1, L8_joint_direct_unet_s0s1, L9_joint_cfm_s0s1_multitau}.yaml`
— base `/lorenz96_default`, `state_dim=24`, `param_dim=8`, `data.param_names` =
the 8-param list, inherit all-5-param randomization ±20%. No `randomize`
override (matches L1b/L2b production). L7 `joint_cfm.train_tau_0_only: true`,
L9 `false`.

## Tests (WP6)

`tests/test_joint_estimation_l96_neural.py` (8 tests): construction/shapes/losses
for both joint models at 24D/8-param; softplus positivity of params; dataloader
`with_params` produces 8-param batches; `_make_eval_batch` extracts the 8
`true_w*` keys; `evaluate_model(return_params=True)` returns 8-param RMSE;
`model_factory` composes all 3 configs to the right class; `LitModel` training
step + backward produces gradients for both joint types. (Also WP1 regression
tests in `tests/test_lorenz96_training.py`.) Added to the CI gate.

## Training + sbatch (WP7)

`batch/run_l96_joint_neural_training.sbatch` (3-task array: L7/L8/L9, rtx8000,
~5h each). `batch/run_l96_joint_neural_eval.sbatch` (3-task array running
`eval_joint_neural_l96.py`; `--train-tau0-only` for L7 only).

## Success criteria

- Train to convergence (val loss plateaus; state RMSE at the ~0.6 state-only
  level, since the state task is unchanged by also predicting params).
- Joint neural **state** RMSE ≲ joint DA state RMSE on S0.
- Joint neural **param** RMSE ≤ joint DA param RMSE on the scalars (F/c1/hx/eps);
  the fast_weights are the hardest (DA has a forward model, neural does not).
- S1/S0 degradation ≈ 1.0 (robustness preserved).

## Status

Infrastructure + doc complete (2026-08-25). Training/eval/report are follow-up
GPU work once launched.
