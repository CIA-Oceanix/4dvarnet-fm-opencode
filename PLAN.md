# 4DVarNet-FM: New Experiments Implementation Plan

## Overview

Two new model families + randomized-parameter dataset, implemented via Hydra+Lightning.

- **DirectUNet**: Single UNet pass `obs → state` via MSE (no flow matching)
- **VanillaCFM**: Standard conditional flow matching, no Tweedie decomposition
- **RandomParamDataset**: Per-window randomized `σ,ρ,β` ±20% for robust training

## Experiments

| ID | Model | Hidden | Epochs | Train mix |
|---|---|---|---|---|
| E1_direct_unet_default | DirectUNet | [64,128,256] | 200 | cs1+cs2 |
| E2_direct_unet_small | DirectUNet | [32,64,128] | 200 | cs1+cs2 |
| E3_direct_unet_rand | DirectUNet | [32,64,128] | 200 | cs1_rand+cs2_rand |
| F1_vanilla_cfm_default | VanillaCFM | [64,128,256] | 400 | cs1+cs2 |
| F2_vanilla_cfm_small | VanillaCFM | [32,64,128] | 400 | cs1+cs2 |
| F3_vanilla_cfm_rand | VanillaCFM | [32,64,128] | 400 | cs1_rand+cs2_rand |

## Phases

### Phase 0: Plan
- [x] PLAN.md created

### Phase 1: Implementation (parallel)
- [x] Agent D: 6 YAML configs + batch/run_new_experiments.sbatch
- [ ] Agent A: models/direct_unet.py, models/vanilla_cfm.py
- [ ] Agent B: data/random_param_dataset.py

### Phase 2: Refactoring + Report (parallel, after Phase 1)
- [ ] Agent C: `conf/schema.py`, `training/lightning_module.py`, `training/pipeline.py`, `train.py`
- [ ] Agent E: `reports/generate_experiment_report.py`

### Phase 3: Verify
- [ ] `train.py` E1 (1 epoch)
- [ ] `train.py` F1 (1 epoch)
- [ ] `train.py` E3 (1 epoch)
- [ ] `generate_experiment_report.py` produces valid PDF

### Phase 4: Commit + Launch
- [ ] Commit 1–4
- [ ] `git push`
- [ ] `sbatch batch/run_new_experiments.sbatch`

### Phase 5: Finalize
- [ ] All 6 `results.json` files received
- [ ] `generate_experiment_report.py` produces final PDF
- [ ] `generate_report.py` updated
- [ ] Commit 5 + push

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
  "fm_degradation": ...
}
```
