## 2026-09-24: Early-fine flow sampler default + T5 second-order (variance) consistency loss

**Summary:**
- `VanillaCFM` / `PredictStateCFM` (and the monai subclasses) now sample on the early-fine Euler grid `tau_k = 1 − (1 − k/N)^0.5` by default (`DEFAULT_STEP_POWER`), a deliberate default recorded in AGENTS.md.
- Adds the T5 loss of `docs/scoping/cfm_tau_consistency_next_steps.md` v4, the configs, and a training + evaluation sbatch array for T5 (λ ∈ {10, 100} × seeds {1, 2}) plus T0b.

**Files modified:**
- `models/vanilla_cfm.py`:
  - `flow_tau_grid` and `DEFAULT_STEP_POWER`, used by both `sample` methods (with a `step_power` argument);
  - `var_*` options, `variance_terms` (finite-difference J u with dropout off, restored afterwards) and `variance_ratio`;
  - the T5 term in `_compute_loss_tau_options`, gated by `var_active`.
- `evaluation/neural_inference.py`, `eval_neural_l96.py`: `--step-power`. `step_power` is passed only when set, so existing `sample` overrides are unaffected, and it is recorded as `sampling.step_power`.
- `training/lightning_module.py`: `var_start_epoch` (sets `model.var_active` each epoch) and `val_var_ratio` logging.
- `train.py`: `var_*` wiring; `step_power` from config; `psc_teacher_options` also returns `var_start_epoch`.
- `conf/schema.py`: documented fields.
- `config/experiment/A2_predictstatecfm_monaiM_var{10,100}_l96.yaml`.
- `batch/run_l96_a2_ps_m_t5_train.sbatch`.
- `AGENTS.md`: sampler-default convention.
- `tests/test_t5_variance_and_sampler.py`.

**Rationale:**
- The early-fine grid improved CRPS by 4.4% / 2.0% at the same cost (`docs/results/cfm_sampler_schedule.md`, #252). Pre-2026-09-24 numbers used the uniform grid; reproduce them with `--step-power 1`.
- T5 targets the operator share of the variance collapse via the per-channel NS1c identity `E[(x1 − D)²] = (b²/τ)·E[∂D/∂x]`:
  - the residual target is stop-gradient, so the mean is trained only by `L_CFM`;
  - the τ-sampling density is unchanged (Batch 1 lesson).
- λ is calibrated on the trained P1 checkpoint, where `L_var/L_CFM` ≈ 0.008, so λ = 10 / 100 gives ≈ 8% / 80%.
- The term is off before epoch 50: at initialisation J ≈ 0 and the residual ≈ 1, so it would dominate `L_CFM` about 10× (seen in the smoke run).

**Verification:**
- `pytest tests/test_t5_variance_and_sampler.py tests/test_psc_tau_options.py tests/test_neural_inference.py tests/test_monai_unet_adapter.py tests/test_lightning_module.py tests/test_checkpoint_compat.py tests/test_hydra_config.py tests/test_config_persistence.py tests/test_fm_sampler.py -m "not slow"`: all pass. The new tests check:
  - step_power = 1 reproduces the old uniform samplers exactly;
  - the finite-difference Jacobian term matches an autograd JVP;
  - the identity holds for an exact Gaussian operator;
  - dropout independence;
  - eval-mode loss unchanged;
  - `var_start_epoch` gating.
- `ruff check` is clean.
- A 3-epoch smoke run of `var10` completes and logs `val_var_ratio`.
