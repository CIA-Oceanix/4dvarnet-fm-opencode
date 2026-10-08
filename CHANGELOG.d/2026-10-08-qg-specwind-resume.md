## 2026-10-08: QG gyrostat neural — resume an interrupted run; G1-L fine-tune config

**Summary:** `train_qg_neural.py --resume [ckpt]` (sbatch `RESUME=1`) continues an interrupted run from its Lightning checkpoint (epoch, optimizer, LR schedule and best-checkpoint tracking), and the regeneration callback now sets the round from the epoch, so a resumed run draws the round the uninterrupted run would be on. Adds the G1-L counterpart of the G1-M sparse-tilted fine-tune.
**Files modified:**
- `data/qg_specwind_neural.py` — `RegenerateTrainWindows` tracks its round and draws `epoch // every` whenever it changes (unchanged schedule for a run from epoch 0)
- `train_qg_neural.py` — `--resume`, passed as `ckpt_path` to `trainer.fit`; recorded as `training.resume_from`
- `batch/run_qg_specwind_neural_train.sbatch` — `RESUME=1|<path>`; appends to the run's train.log instead of overwriting it
- `config/experiment/G1L_direct_unet_tchannels_specwind_b16_kpow_ft.yaml` — new
- `tests/test_qg_specwind_neural.py` — resume round test
**Rationale:** /Odyssey filled up on 2026-10-08 13:38 and killed the G1-M K^-1.5 and 15-day runs at epochs 279/294 of 1200 (~5 GPU-h each); their last checkpoints were intact. The G1-M fine-tune gained +0.008 at K = 3 (S0 0.820, S1 0.790), so the same recipe is run on G1-L, the best tier.
**Verification:** `pytest -m "not slow" tests/test_qg_specwind_neural.py tests/test_qg_specwind_kpow_window.py tests/test_qg_config_persistence.py tests/test_qg_neural.py` — 125 passed; ruff clean; `bash -n` on the sbatch; both interrupted runs relaunched with `RESUME=1` (59392/59393).
