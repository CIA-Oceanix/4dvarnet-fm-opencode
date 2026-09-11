## 2026-09-12: QG Q5 (IC-conditioned) + Q3-lag5 retrain

**Summary:** Added Q5 -- Q4's noisy forcing+param conditioning plus the raw
initial-condition (IC) snapshot as a fourth input, trained at the DA
reference case's own lag=5.0d/noise=0.05 (not Q1/Q3/Q4's 1.0d/0.01
defaults). Motivated by DA baselines getting real skill from their
background (a rolled-forward init state) that Q1/Q3/Q4 have zero equivalent
of. Paired with a Q3-lag5 retrain (same oracle conditioning, same updated
lag/noise, no IC) so the "oracle vs. IC-augmented" comparison isn't
confounded by also comparing across training distributions.

**Files modified:**
- `data/qg_neural.py` -- new `_ic_field` (inverts `window["init_state"]` to
  psi, z-scores with the existing global `psi_norm_stats`); `QGNeuralDataset`
  gained `include_ic`; `QGBatch` gained `ic`.
- `models/monai_unet_qg2d.py` -- `MonaiDirectUNetQG` gained `ic_dim`: unlike
  `forcing` (varies per day) or `params` (a scalar vector), the IC is one
  static field per window, broadcast identically across all days.
- `train_qg_neural.py` -- new `--include-ic` flag; new `--s1-param-bias`/
  `--s1-amp-bias` CLI overrides (previously training always silently used
  `QGConfig`'s 0.15 class default, no override existed); fixed
  `--obs-noise-std-frac`/`--init-lag-days` to actually read from the
  experiment YAML (previously CLI-only with hardcoded 0.01/1.0 defaults --
  Q5's config alone would have been silently ignored without this).
- New configs: `Q5_direct_unet_s1_noisy_ic_cond.yaml` (`s1_param_bias`/
  `s1_amp_bias=0.1` with `noisy_max=2.0` for an effective `[0,0.2]` bias
  range, `init_lag_days=5.0`, `obs_noise_std_frac=0.05`, `include_ic: true`),
  `Q3_direct_unet_s0_oracle_cond_lag5.yaml`.
- New sbatch: `run_qg_q5_train.sbatch`, `run_qg_q3_lag5_train.sbatch`.
- Tests: `tests/test_qg_neural.py` (include_ic requires psi_norm_stats,
  normalized-scale/determinism, collate stacking), `tests/test_monai_unet_qg2d.py`
  (forward-pass IC conditioning sanity check).

**Rationale:** psi/param/forcing norm stats are reused unchanged from
Q1/Q3/Q4's precompute -- none depend on `obs_noise_std_frac`/`init_lag_days`
(only obs/IC *sampling* does, not the underlying truth/params/forcing).

**Verification:** `pytest tests/test_qg_neural.py -m "not slow"` (fdv env)
-- 32 passed, 1 skipped (monai-gated); `pytest tests/test_monai_unet_qg2d.py
tests/test_qg_neural.py::test_build_model_honors_yaml_param_dim_and_cond_extra_dim`
(fdv-monai-proto env) -- 6 passed. `ruff check` clean.
