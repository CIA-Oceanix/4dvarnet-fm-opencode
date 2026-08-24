# 4DVarNet-FM: Implementation Plan

## Overview

Three model families + CS3/CS4 randomized-parameter tests + Experiment G ablation.

- **DirectUNet**: Single UNet pass `obs → state` via MSE (no flow matching) — implemented
- **VanillaCFM**: Standard conditional flow matching, no Tweedie decomposition — implemented
- **TweedieSolver**: Original two-stage solver — legacy, maintained
- **RandomParamDataset**: Per-window randomized `σ,ρ,β` ±20% for robust training — implemented
- **CS3/CS4**: Evaluation on unseen random parameter draws — implemented
- **Exp G (τ=0 CFM)**: Ablate multi-τ training to isolate CFM's source of advantage — implemented (L63; τ=0 runs superseded by the S-series, also L63)

**System naming convention:** the E/F/G/S experiment series are all **Lorenz-63**
(`train_mix: cs1+cs2`, `state_dim=3` checkpoints). The **L-series** is the two-scale
**Lorenz-96** benchmark (`system: lorenz96`, `state_dim=24` observed subspace).

## L96 (two-scale Lorenz-96) — merged to master 2026-08-18

- **Dynamics/DA baselines** (`feat/weighted-fast-coupling` merged into master, SW/MAOOAM excluded):
  - `models/lorenz96_dynamics.py`, `data/lorenz96.py` — two-scale L96 (NO=8, J=4, state_dim=40, weighted fast coupling)
  - `evaluation/run_l96*.py` + `reports/outputs/l96_baseline_report.md` — Waves 1-4 DA sweeps + ETKF ablation, pooled-EV metric
  - `models/dynamics.py` `get_dynamics()` supports only `lorenz63`/`lorenz96` (SW/MAOOAM deferred)
- **Training infrastructure** (built, no runs launched):
  - `conf/schema.py` → `DataConfig.to_lorenz96_config()`
  - `train.py` → `data.system` dispatch (`lorenz96`), `param_names`-generalized eval
  - configs: `config/experiment/L1_direct_unet_s0s1.yaml`, `L2_vanilla_cfm_s0s1.yaml`
  - test: `tests/test_lorenz96_training.py`

## L96 Neural Training (`feat/l96-neural-training`, from master @ 0687e07)

Implements all-5-param randomization + neural S0/S1 training (commit `3a1c8d5`).
See `L96_NEURAL_TRAINING_PROGRESS.md` for the per-WP tracker and handoff.

- **All 5 params randomized ±20%**: `models/lorenz96_dynamics.py` accepts `c1,h,hx,eps`
  + `F` kwargs; `data/lorenz96.py` `RandomParam`/`RandomBias` + `make_l96_s0_s1_trainval`.
- **Neural `param_dim=0` + `cond_extra_dim=0`** (DirectUNet, VanillaCFM-τ=0): obs-only
  input (no forcing/params conditioning; 24D in, 24D out).
- **DA parity**: `evaluation/run_l96.py` + `evaluate_all_l96.py` pass per-window all-5
  params to DA; S1 uses biased `*_da` params.
- Configs: `config/lorenz96_default.yaml`, `L1_direct_unet_s0s1.yaml`,
  `L2_vanilla_cfm_s0s1.yaml` (both `param_dim=0`; L2 is τ=0).
- Tests: `tests/test_lorenz96_training.py` (11 tests).
- sbatch: `run_one_epoch_tests_l96`, `run_l96_da_consistency`,
  `run_l96_neural_training`, `run_l96_evaluate_all`.

### Multi-agent review workflow (git/PR)

Code changes on this branch go through an implementer → reviewer → verifier loop.
Two execution paths:

- **Option A — GitHub PR**: `.github/workflows/ci.yml` runs ruff + pytest on PRs
  to `feat/l96-*`. Agents use `gh pr create` / `gh pr review` / `gh pr merge`.
  Blocked until `gh auth login` is run interactively (W3).
- **Option B — Local**: `scripts/agent_review_loop.sh <STEP> "<desc>" [--review]`
  provides the same loop with local git (works immediately).

**Run-to-completion policy:** the general rules (branch naming, drive-to-merge, reviewer
identity, CI gate) live in `AGENTS.md` under **Git / PR Workflow** and apply to every
session — follow those. This branch additionally uses `feat/l96-*` as its integration
namespaces subject to the repo ruleset.

**REMINDER:** run `gh auth login` and enable branch protection on `feat/l96-*`
(require 1 PR approval + status checks) to unlock the GitHub PR path.

**Status (2026-08-23)**: Step 11c/12 + WP8 are complete — L1b (DirectUNet) and L2b
(VanillaCFM τ=0) trained on the DA-parity config and evaluated on the shared cached
test set (`reports/outputs/neural_benchmark_table.md`): neural beats the best DA
baseline on both S0 and S1 (RMSE 0.62 vs 0.74), degradation ≈1.00 vs DA ≈1.9×.

**Open questions (L96)** — all answered (standalone DA-parity eval, cached test set):
- **Q1 (answered, L3)**: multi-τ CFM does NOT beat conditional-mean estimation —
  L3 0.688/0.690 vs τ=0 L2b 0.633/0.633 (+8.6%); mirrors the L63 G-series finding.
- **Q2 (answered, L4/L5)**: size sensitivity is model-dependent — small DirectUNet
  (L4) slightly beats default L1b (0.619 vs 0.622); small τ=0 CFM (L5) is worse than
  default L2b (+4.3%). CFM benefits from capacity; DirectUNet does not.
- **Q3 (answered, L6)**: corrupted-forcing conditioning is neutral-to-slightly-negative
  (L6 0.639/0.638 vs obs-only L2b 0.633/0.633); neural degradation was already ≈1.00,
  so there was no robustness gap for conditioning to close.

## Experiments

All E/F/G/S rows are **Lorenz-63** (`cs1+cs2` mixes); results live under `experiments/`.
The **L-series** (Lorenz-96) is listed separately below.

### Lorenz-63

| ID | Model | Hidden | Epochs | Train mix | Status |
|---|---|---|---|---|---|
| E1_direct_unet_default | DirectUNet | [64,128,256] | 200 | cs1+cs2 | done |
| E2_direct_unet_small | DirectUNet | [32,64,128] | 200 | cs1+cs2 | done |
| E3_direct_unet_rand | DirectUNet | [32,64,128] | 200 | cs1_rand+cs2_rand | done |
| F1_vanilla_cfm_default | VanillaCFM | [64,128,256] | 400 | cs1+cs2 | done |
| F2_vanilla_cfm_small | VanillaCFM | [32,64,128] | 400 | cs1+cs2 | done |
| F3_vanilla_cfm_rand | VanillaCFM | [32,64,128] | 400 | cs1_rand+cs2_rand | done |
| G1_vanilla_cfm_t0_default | VanillaCFM (τ=0) | [64,128,256] | 400 | cs1+cs2 | done |
| G2_vanilla_cfm_t0_small | VanillaCFM (τ=0) | [32,64,128] | 400 | cs1+cs2 | done |
| G3_vanilla_cfm_t0_rand | VanillaCFM (τ=0) | [32,64,128] | 400 | cs1_rand+cs2_rand | done |
| S1–S10 (incl. τ=0 + joint-CFM variants) | various | various | — | s0_s1 | done |

### Lorenz-96 (DA-parity: all-5 params ±20%, obs_j=2 → 24D, Obs30)

| ID | Model | Hidden | Epochs | τ mode | Status |
|---|---|---|---|---|---|
| L1b_direct_unet_s0s1 | DirectUNet | [64,128,256] | 200 | n/a | done (beats DA on S0+S1) |
| L2b_vanilla_cfm_s0s1 | VanillaCFM | [64,128,256] | 400 | τ=0 | done (≈ L1b) |
| L3_vanilla_cfm_s0s1 | VanillaCFM | [64,128,256] | 400 | multi-τ | done (Q1: worse than τ=0, +8.6%) |
| L4_direct_unet_s0s1_small | DirectUNet | [32,64,128] | 200 | n/a | done (Q2: best overall, 0.619/0.621) |
| L5_vanilla_cfm_s0s1_small_tau0 | VanillaCFM | [32,64,128] | 400 | τ=0 | done (Q2: small hurts CFM, +4.3%) |
| L6_vanilla_cfm_s0s1_forcing_cond | VanillaCFM | [64,128,256] | 400 | τ=0 + forcing cond | done (Q3: neutral vs obs-only) |

**Standalone DA-parity results (cached test set, Obs30, 200 windows)** — S0/S1 RMSE:
L4 **0.619**/0.621 < L1b 0.622/0.625 < L2b 0.633/0.633 ≈ L6 0.639/0.638 < L5 0.660/0.660 < L3 0.688/0.690.
All neural degradation ≈ 1.00; best DA (Strong-4DVar): 0.742/1.432.

## Phases

### Phase 0: Plan
- [x] Initial PLAN.md created
- [x] CS3/CS4 experiment plan in `docs/case_studies.tex`
- [x] Exp G (τ=0) experiment plan in `docs/experiment_G_tau0_cfm.md`

### Phase 1: Implementation (complete)
- [x] `models/direct_unet.py` — DirectUNet nn.Module
- [x] `models/vanilla_cfm.py` — VanillaCFM nn.Module with CFM loss + sampling
- [x] `data/random_param_dataset.py` — Randomized Lorenz-63 parameters per window
- [x] `conf/schema.py` — `DirectUNetConfig`, `VanillaCFMConfig`, `DataConfig` with CS3/CS4 fields
- [x] `training/lightning_module.py` — `LitModel` dispatches all 3 model types
- [x] `training/pipeline.py` — `create_trainer`, `train_stage`, `run_2stage_pipeline`
- [x] `train.py` — `model_factory`, `evaluate_model`, CS3/CS4 evaluation
- [x] 6 experiment YAML configs (E1-E3, F1-F3)
- [x] CS3/CS4 test cases in data generation, evaluation, and report
- [x] `evaluate_all.py` — Unified baseline + CFM comparison script
- [x] `reports/generate_unet_cfm_report.py` — CS3/CS4 report

### Phase 2: sbatch Infrastructure (this session)
- [x] `batch/run_lint.sbatch` — ruff + mypy in batch
- [x] `batch/run_test_suite.sbatch` — pytest (fast) in batch
- [x] `batch/run_config_validation.sbatch` — Hydra config + model factory validation
- [x] Deprecated duplicate `run_vanilla_experiments.sbatch` and interactive `run_tests.sh`

### Phase 3: τ=0 CFM Ablation (Exp G) — complete (2026-07-01)
- [x] `conf/schema.py` — `train_tau_0_only: bool = False` on `VanillaCFMConfig`
- [x] `models/vanilla_cfm.py` — τ=0 logic in `compute_cfm_loss` and `sample`
- [x] `train.py` — `train_tau_0_only` wired through `model_factory`
- [x] 3 config YAMLs: G1_vanilla_cfm_t0_default, G2_vanilla_cfm_t0_small, G3_vanilla_cfm_t0_rand
- [x] `batch/run_one_epoch_tests.sbatch` + `batch/run_new_experiments.sbatch` updated with G1-G3
- [x] Tests for τ=0 mode

### Phase 4: Verify (all via sbatch) — complete (2026-07-01)
- [x] `sbatch batch/run_config_validation.sbatch` — all configs load
- [x] `sbatch batch/run_lint.sbatch` — ruff + mypy pass
- [x] `sbatch batch/run_test_suite.sbatch` — fast tests pass
- [x] `sbatch batch/run_one_epoch_tests.sbatch` — GPU smoke test (E1-F3 + G1-G3, 1 epoch)

### Phase 5: Launch — complete (2026-07)
- [x] `sbatch batch/run_new_experiments.sbatch` — full E1-F3 + G1-G3
- [x] Results collected under `experiments/` (see Experiments tables above)
- [x] CHANGELOG.md entries per change

## Interfaces

### Model forward signatures (for LightningModule dispatch):
```
TweedieSolver:
  training_step(stage=1): model.estimate_mean(obs) → (B,T,D)
  training_step(stage=2): model(obs) → (B,T,D)
  config_optim(stage=1): model.mean_estimator.parameters()
  config_optim(stage=2): model.non_gaussian.parameters()

DirectUNet:
  training_step: model(obs) → (B,T,D)
  loss: StateMSELoss(pred, batch.states)
  config_optim: model.parameters()

VanillaCFM:
  training_step: compute_cfm_loss(batch) → scalar
  config_optim: model.parameters()
  sampling: model.sample(obs, N_outer) → (B,T,D)
```

### Dataset output format:
```python
{
    "true_state": Tensor(T, 3),
    "obs": Tensor(T, 3),
    "obs_mask": Tensor(T,),
    "forcing_true": Tensor(T,),
    "forcing_corrupted": Tensor(T,),
}
```

### results.json format (per experiment):
```json
{
  "experiment_id": "...",
  "config": {...},
  "epochs_trained": ...,
  "total_time_seconds": ...,
  "train_time_seconds": ...,
  "eval_time_seconds": ...,
  "fm_cs1": {"X": {"mean": ..., "std": ...}, "Y": ..., "Z": ..., "mean": ...},
  "fm_cs2": {...},
  "fm_cs3": {...},
  "fm_cs4": {...},
  "fm_degradation": ...,
  "fm_degradation_cs3cs4": ...
}
```
