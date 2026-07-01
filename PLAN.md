# 4DVarNet-FM: Implementation Plan

## Overview

Three model families + CS3/CS4 randomized-parameter tests + Experiment G ablation.

- **DirectUNet**: Single UNet pass `obs → state` via MSE (no flow matching) — implemented
- **VanillaCFM**: Standard conditional flow matching, no Tweedie decomposition — implemented
- **TweedieSolver**: Original two-stage solver — legacy, maintained
- **RandomParamDataset**: Per-window randomized `σ,ρ,β` ±20% for robust training — implemented
- **CS3/CS4**: Evaluation on unseen random parameter draws — implemented
- **Exp G (τ=0 CFM)**: Ablate multi-τ training to isolate CFM's source of advantage — **to implement**

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
