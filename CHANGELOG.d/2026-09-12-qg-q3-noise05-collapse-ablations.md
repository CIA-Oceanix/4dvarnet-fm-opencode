## 2026-09-12: rename Q3-lag5 -> Q3-noise0.05, add gradient-clip/seed ablations

**Summary:** Q3's lag=5/noise=0.05 retrain (previously misnamed "Q3-lag5")
collapsed at epoch ~19-20 -- loss froze at an exact constant for multiple
consecutive epochs, the same signature as the earlier forcing-normalization
collapse but a different, not-yet-confirmed root cause. Renamed the config
to reflect what actually matters: `init_lag_days` has zero effect on Q3 (no
`include_ic`), only `obs_noise_std_frac` does. Added two CLI flags to set
up disambiguating ablations.

**Files modified:**
- `train_qg_neural.py` -- new `--gradient-clip-val` (was hardcoded 10.0 in
  `make_trainer_cfg()`, no override existed); new `--seed` (calls
  `pl.seed_everything()` -- there was no seeding infrastructure at all
  before this, every prior run used whatever random state the process
  happened to start with, undocumented and unreproducible).
- `config/experiment/Q3_direct_unet_s0_oracle_cond_lag5.yaml` ->
  `..._noise05.yaml`, `batch/run_qg_q3_lag5_train.sbatch` ->
  `run_qg_q3_noise05_train.sbatch` (renamed).
- New `batch/run_qg_q3_noise05_gradclip1_train.sbatch` (same config,
  `--gradient-clip-val 1.0`) and `run_qg_q3_noise05_seed_train.sbatch`
  (same config, `--seed 123`) -- two independent single-variable ablations
  of the collapsed run.

**Rationale:** Working hypothesis for the collapse: the 5x higher obs noise
(0.01->0.05) occasionally produces a larger/outlier daily-aggregated obs
value (sparse `random_columns` sampling) whose gradient isn't fully
contained by the existing `gradient_clip_val=10.0`. Notably Q5 (same obs
noise, but `cond_mode="noisy"` + IC input) did NOT collapse at the same
epoch -- these two ablations disambiguate whether tighter clipping alone
fixes it, or whether the original collapse was closer to a rare/stochastic
event tied to that particular (untracked) random draw rather than a
guaranteed function of the noise level.

**Verification:** `pytest tests/test_qg_neural.py -m "not slow"` (fdv env)
-- 34 passed, 1 skipped (monai-gated). `ruff check` clean.
