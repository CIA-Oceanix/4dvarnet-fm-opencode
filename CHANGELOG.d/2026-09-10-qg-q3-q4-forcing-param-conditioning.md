## 2026-09-10: QG Q3 (oracle) / Q4 (noisy) forcing+param-conditioned DirectUNet

**Summary:** Added two new QG neural schemes that condition the MONAI
DirectUNet estimator on the wind-forcing field + physical params
(`U1, rd, rek, beta`), alongside observations -- Q3 uses the exact true
forcing/params (oracle), Q4 uses per-draw resampled-severity corrupted
forcing/params, mirroring the L96 SDA3 CFM study's noisy-params
conditioning approach rather than a single fixed S1 bias.

**Files modified:**
- `train_qg_neural.py` -- fixed `build_model()` to actually read
  `model.param_dim`/`model.cond_extra_dim` from the experiment YAML
  (previously hardcoded to 0/0, silently ignoring those YAML fields for
  Q1/Q2 too); added `--exp-id`/`--cond-mode`/`--param-norm-stats-path`/
  `--noisy-max` CLI flags.
- `data/qg_neural.py` -- `QGNeuralDataset` gained `cond_mode`
  (`"none"`/`"true"`/`"noisy"`), `param_norm_stats`, `noisy_max`; `forcing`
  changed shape project-wide from `(B, days)` scalar to `(B, days, ny, nx)`
  spatial field; `qg_collate` now also stacks `params`.
- `models/monai_unet_qg2d.py` -- `MonaiDirectUNetQG.forward()` reshapes the
  real spatial forcing field instead of scalar-broadcasting it.
- `precompute_qg_norm_stats.py` -- new `--output-params` flag for
  `[U1,rd,rek,beta]` z-score stats.
- New configs: `config/experiment/Q3_direct_unet_s0_oracle_cond.yaml`,
  `Q4_direct_unet_s1_noisy_cond.yaml`.
- New sbatch: `batch/run_qg_param_stats.sbatch`, `run_qg_q3_train.sbatch`,
  `run_qg_q4_train.sbatch`.
- Tests: `tests/test_qg_neural.py`, `tests/test_monai_unet_qg2d.py`.

**Rationale:** Q1's forward pass is scenario-agnostic by design (identical
S0/S1 estimates, see `eval_qg_q1_s0_s1.py`) -- it never sees forcing or
params at all. Q3/Q4 test whether feeding them helps, and whether that
answer changes between oracle and realistic (noisy) conditioning, mirroring
the L96 SDA study's finding that conditioning-regime differences there were
small. See PLAN.md's 2026-09-10 QG Q3/Q4 section for the full design.

**Verification:** `pytest tests/test_qg_neural.py tests/test_qg_data.py -m
"not slow"` (fdv env, no monai) -- 23 passed, 1 skipped (monai-gated);
`pytest tests/test_monai_unet_qg2d.py tests/test_qg_neural.py::
test_build_model_honors_yaml_param_dim_and_cond_extra_dim` (fdv-monai-proto
env) -- 5 passed. `ruff check` clean on all touched `.py` files.
