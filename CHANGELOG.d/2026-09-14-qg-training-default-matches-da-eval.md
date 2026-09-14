## 2026-09-14: Training default now matches the DA-baseline eval default

**Summary:** `train_qg_neural.py`'s `obs_noise_std_frac`/`init_lag_days`
fallback (used when neither `--obs-noise-std-frac`/`--init-lag-days` nor
the experiment YAML's `data.*` specify a value) changes from `0.01`/`1.0`
to `0.05`/`5.0` -- the DA-baseline reference case's own value (see
`eval_qg_neural_s0_s1.py`), so a new config that omits these fields now
trains apples-to-apples with the DA comparison by default instead of
producing a recurring "not apples-to-apples" caveat on every eval report.
`eval_qg_neural_s0_s1.py`'s own `--lag-days`/`--noise-frac` CLI defaults
are flipped the same way, `1.0`/`0.01` -> `5.0`/`0.05`.

**Files modified:**
- `train_qg_neural.py` -- fallback literals in the `obs_noise_std_frac`/
  `init_lag_days` resolution changed `0.01`/`1.0` -> `0.05`/`5.0`; comments
  updated accordingly. The unrelated `test_cache_cfg` cache-lookup key
  (`obs_noise_std_frac=0.01, init_lag_days=1.0`, matching a specific
  pre-generated truth cache on disk) is untouched.
- `eval_qg_neural_s0_s1.py` -- `--lag-days`/`--noise-frac` CLI defaults
  `1.0`/`0.01` -> `5.0`/`0.05`; help text and module docstring updated.
  The unrelated `CACHE_KW` cache-lookup key is untouched for the same
  reason.
- `config/experiment/Q1_direct_unet_s0.yaml`,
  `Q1_direct_unet_s0_obsdensity_aug.yaml`, `Q2_vanilla_cfm_s0.yaml`,
  `Q3_direct_unet_s0_oracle_cond.yaml`, `Q4_direct_unet_s1_noisy_cond.yaml`,
  `Q6_fourdvarnet_s0.yaml`, `Q7_direct_unet_tchannels_s0.yaml` -- each
  explicitly pins `data.obs_noise_std_frac: 0.01` / `data.init_lag_days:
  1.0` so this code-level default change alters nothing about what these
  already-trained/in-flight runs actually trained at.
  `Q3_direct_unet_s0_oracle_cond_noise05.yaml` and
  `Q5_direct_unet_s1_noisy_ic_cond.yaml` already set `0.05`/`5.0`
  explicitly and needed no change.

**Rationale:** Same precedent as the 2026-09-10 cosine-LR-scheduler
default flip (see `AGENTS.md`): pin every existing config to its current
effective value before changing the shared code-level default, so nothing
already-trained or in-flight is silently altered -- only a genuinely new
config that omits both fields picks up the new default. See PLAN.md's
2026-09-14 "Training default now matches the DA eval default" section.

**Verification:** All 7 pinned YAMLs load via `OmegaConf.load` and carry
`obs_noise_std_frac: 0.01`/`init_lag_days: 1.0` in `data`, confirmed
directly. `ruff check train_qg_neural.py eval_qg_neural_s0_s1.py` clean.
`pytest tests/test_qg_neural.py tests/test_fourdvarnet.py
tests/test_monai_unet_qg2d.py -m "not slow"` (fdv env).
