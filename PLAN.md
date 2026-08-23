# 4DVarNet-FM: Implementation Plan

## Overview

Three model families + CS3/CS4 randomized-parameter tests + Experiment G ablation.

- **DirectUNet**: Single UNet pass `obs → state` via MSE (no flow matching) — implemented
- **VanillaCFM**: Standard conditional flow matching, no Tweedie decomposition — implemented
- **TweedieSolver**: Original two-stage solver — legacy, maintained
- **RandomParamDataset**: Per-window randomized `σ,ρ,β` ±20% for robust training — implemented
- **CS3/CS4**: Evaluation on unseen random parameter draws — implemented
- **Exp G (τ=0 CFM)**: Ablate multi-τ training to isolate CFM's source of advantage — implemented (superseded by S9/S10)

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
- **Neural `param_dim=0`** (DirectUNet, VanillaCFM-τ=0): obs + corrupted forcing only.
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

**To do (next)**: run DA consistency re-run (Step 11c), train L1/L2 (Step 12), then
compare DA vs neural on S0/S1 (WP8) — all via the L96 sbatch scripts.

## Experiments

| ID | Model | Hidden | Epochs | Train mix | Status |
|---|---|---|---|---|---|
| E1_direct_unet_default | DirectUNet | [64,128,256] | 200 | cs1+cs2 | config ready |
| E2_direct_unet_small | DirectUNet | [32,64,128] | 200 | cs1+cs2 | config ready |
| E3_direct_unet_rand | DirectUNet | [32,64,128] | 200 | cs1_rand+cs2_rand | config ready |
| F1_vanilla_cfm_default | VanillaCFM | [64,128,256] | 400 | cs1+cs2 | config ready |
| F2_vanilla_cfm_small | VanillaCFM | [32,64,128] | 400 | cs1+cs2 | config ready |
| F3_vanilla_cfm_rand | VanillaCFM | [32,64,128] | 400 | cs1_rand+cs2_rand | config ready |
| G1_vanilla_cfm_t0_default | VanillaCFM (τ=0) | [64,128,256] | 400 | cs1+cs2 | **to implement** |
| G2_vanilla_cfm_t0_small | VanillaCFM (τ=0) | [32,64,128] | 400 | cs1+cs2 | **to implement** |
| G3_vanilla_cfm_t0_rand | VanillaCFM (τ=0) | [32,64,128] | 400 | cs1_rand+cs2_rand | **to implement** |

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

### Phase 3: τ=0 CFM Ablation (Exp G)
- [ ] `conf/schema.py` — add `train_tau_0_only: bool = False` to `VanillaCFMConfig`
- [ ] `models/vanilla_cfm.py` — τ=0 logic in `compute_cfm_loss` and `sample`
- [ ] `train.py` — wire `train_tau_0_only` flag through `model_factory`
- [ ] 3 config YAMLs: G1_vanilla_cfm_t0_default, G2_vanilla_cfm_t0_small, G3_vanilla_cfm_t0_rand
- [ ] Update `batch/run_one_epoch_tests.sbatch` + `batch/run_new_experiments.sbatch` with G1-G3
- [ ] Tests for τ=0 mode

### Phase 4: Verify (all via sbatch)
- [ ] `sbatch batch/run_config_validation.sbatch` — all 10 configs load
- [ ] `sbatch batch/run_lint.sbatch` — ruff + mypy pass
- [ ] `sbatch batch/run_test_suite.sbatch` — all fast tests pass
- [ ] `sbatch batch/run_one_epoch_tests.sbatch` — GPU smoke test (E1-F3 + G1-G3, 1 epoch)

### Phase 5: Launch
- [ ] `sbatch batch/run_new_experiments.sbatch` — full E1-F3 + G1-G3
- [ ] Collect results → `python reports/generate_experiment_report.py`
- [ ] Merge to master, push
- [ ] Update CHANGELOG.md

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
