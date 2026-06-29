# 4DVarNet-FM Refactoring: Hydra + Lightning

## Objective
Refactor the training pipeline to use Hydra (config management) + PyTorch Lightning (training loop) while preserving full backward compatibility with the existing `checkpoint_stage*.pt` files.

## Progress

| Phase | Status |
|-------|--------|
| 0: Environment setup | **In progress** |
| 1: Config layer — sync YAML with schema, create experiment configs | Pending |
| 2: Model layer — audit, no changes needed | Pending |
| 3: LightningModule — extend Lit4DVarNetFM with DictConfig + stage toggle | Pending |
| 4: Trainer pipeline — training/pipeline.py with pl.Trainer | Pending |
| 5: Entry points — train.py + refactor run_experiments.py | Pending |
| 6: Evaluation/data adapters | Pending |
| 7: Checkpoint management — standardized dirs | Pending |
| 8: Tests — checkpoint compat + pipeline integration | Pending |
| 9: Cleanup — remove stage1.py/stage2.py, final validation | Pending |

---

## Plan

### Phase 0: Environment Setup
- Create a new conda env or extend `fdv`
  - Python 3.10, PyTorch 2.4.1, CUDA 11.8, hydra-core 1.3.2, omegaconf 2.3.0, pytorch-lightning 2.3.3
  - Fix the torch library loading issue
- `requirements.txt` updated

### Phase 1: Config Layer — Hydra Structured Configs
- `conf/schema.py` — already done, keep as-is
- `config/lorenz63_default.yaml` — sync with ExperimentConfig schema
- `config/experiment/` — one YAML per experiment variant (A1, B1, C1, C4, D1)

### Phase 2: Model Layer — Zero Changes
- `models/` — no edits. Must produce identical state_dict() keys

### Phase 3: LightningModule — Backward-Compatible Wrapper
- `training/lightning_module.py` — extend Lit4DVarNetFM:
  - Accept DictConfig instead of flat params
  - set_stage(stage) to freeze/unfreeze sub-networks
  - configure_optimizers reads lr, weight_decay from config
  - LR scheduler support
  - load_legacy_checkpoint() kept and tested

### Phase 4: Trainer Pipeline
- **New** `training/pipeline.py`:
  - create_trainer(cfg, stage) → pl.Trainer with callbacks + logger
  - train_stage(lit_module, loaders, cfg, stage) → fits trainer, saves raw .pt
  - run_2stage_pipeline() → stage 1 + optional stage 2

### Phase 5: Entry Points
- **New** `train.py` — @hydra.main entry point with CLI overrides
- **Refactor** `run_experiments.py` — use Hydra multirun or subprocess calls

### Phase 6: Evaluation / Data — Minimal Adapters
- Bridge DataConfig → Lorenz63Config for backward compat
- evaluation/experiment.py accepts DataConfig

### Phase 7: Checkpoint Management
- Standardized: `checkpoints/{experiment_name}/stage{1,2}.pt`
- Current checkpoints remain loadable
- Both .ckpt and raw .pt saved

### Phase 8: Tests — the Critical Layer
- `tests/test_checkpoint_compat.py`: legacy loading tests
- `tests/test_pipeline.py`: pipeline integration tests
- Update `tests/test_training.py` with Trainer variants

### Phase 9: Cleanup
- Remove `stage1.py`, `stage2.py` after validation
