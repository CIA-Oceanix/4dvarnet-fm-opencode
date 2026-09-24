## 2026-09-23: PredictStateCFM τ-consistency training options + EMA teacher (track B)

**Summary:** Adds opt-in training options to `PredictStateCFM` / `MonaiPredictStateCFM`, plus an EMA teacher in `LitModel`, implementing the training arms of `docs/scoping/cfm_tau_consistency_next_steps.md` v3:
- T1′: `tau0_frac`.
- T1: `tau_sampling: low_mix`.
- T2a/T2b: `boot_*`, bootstrapped targets `(1-α) x1 + α sg D_teacher(x_tau', tau')` on forward-noised pairs.
- T4a: `tau0_zero_input`.

Adds the Batch-1/2 configs and sbatch scripts, including the T0 reseed.
**Files modified:**
- `models/vanilla_cfm.py`: `_init_tau_options`, `pair_from_x1` (exact joint law in x-space), `compute_loss(batch, teacher=None)`, and `tau0_zero_input` in `forward`.
- `models/monai_unet_adapter.py`: `MonaiPredictStateCFM` forwards the options.
- `training/lightning_module.py`: `ema_decay` / `teacher_start_epoch`. The teacher is a frozen deep copy kept outside the module tree, lerp-updated after each batch.
- `train.py`: `psc_tau_options`, `psc_teacher_options`, wired into `model_factory` and `LitModel`.
- `conf/schema.py`: documented fields.
- `config/experiment/A2_predictstatecfm_monaiM_{tau0frac,boot0,bootmid,lowmix,tau0zero}_l96.yaml`.
- `batch/run_l96_a2_ps_m_{seed1,tau0frac,boot0,bootmid,lowmix,tau0zero}_train.sbatch`.
- `tests/test_psc_tau_options.py`.

**Rationale:**
- Every default reproduces today's loss bit for bit, so no existing config retrains.
- The options apply only in training mode. `val_loss`, and hence checkpoint selection, stays the plain uniform-τ loss for every arm.
- The teacher never enters the `state_dict`, so checkpoints and `load_model` are unchanged. A resumed run re-seeds the teacher from the resumed weights.
- Arm configs pin `training.seed: 1` to pair them with the T0 reseed.
- Arm configs set `data.norm_stats_path` to the P1 stats file (md5 identical to the P1 run's) by absolute path: a fresh worktree has no `experiments/l96_norm_stats_obsj2.pt`, and `train.py` does not compute one.
- T0 overrides the seed with `++training.seed`: the key is absent from the composed config, so a plain override is rejected.

**Verification:**
- `pytest tests/test_psc_tau_options.py tests/test_monai_unet_adapter.py`: 29 passed.
- `ruff check` is clean on the touched files.
- A 3-epoch smoke training of the T2a config with the teacher switched on at epoch 1 (see PR).
