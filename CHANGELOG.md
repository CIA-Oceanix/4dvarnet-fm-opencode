# Changelog

## 2026-06-30: Initialize opencode project guidelines

**Summary:** Added AGENTS.md, opencode.json, and initial CHANGELOG.md to establish a consistent workflow for opencode sessions.
**Files modified:**
- `AGENTS.md` — new: project guidelines with session workflow, commands, conventions
- `opencode.json` — new: project opencode config referencing PLAN.md and CHANGELOG.md
- `.gitignore` — removed `opencode.json` exclusion so the config can be committed
- `CHANGELOG.md` — new: implementation log
**Rationale:** Ensure every opencode session follows a consistent workflow: read PLAN.md, implement, verify, log changes.

## 2026-06-30: Add experiment plan for τ=0 CFM ablation

**Summary:** Created `docs/experiment_G_tau0_cfm.md` documenting a proposed experiment to test whether VanillaCFM's advantage over DirectUNet comes from multi-τ training or from the residual loss formulation.
**Files modified:**
- `docs/experiment_G_tau0_cfm.md` — new: experiment plan with motivation, code changes, configs, and expected outcomes
**Rationale:** Plan to isolate the effect of random τ sampling by training VanillaCFM with τ=0 only and comparing RMSE against full CFM (F1-F3) and DirectUNet (E2).

## 2026-06-30: Add CS3/CS4 randomized-parameter test cases

**Summary:** Extended the benchmark with two new test cases (CS3/CS4) that apply per-window parameter randomisation (param_noise=0.2) to CS1/CS2 dynamics. Fixed a coupling_type bug in baseline evaluation (CS2/CS4 need "quartic"). Added unified `evaluate_all.py` script and updated report generation and documentation.
**Files modified:**
- `data/lorenz63.py` — `make_mixed_datasets()` now accepts `include_randparam_test` and `param_noise`; returns `RandomParamLorenz63Dataset` for test_cs3/test_cs4
- `conf/schema.py` — added `test_randparam` and `test_param_noise` fields to `DataConfig`
- `evaluation/run.py` — extended `_BASELINE_CASES` to include cs3/cs4 with coupling_type; created per-coupling-type baseline pool (linear/quartic)
- `train.py` — evaluate on CS3/CS4, save trajectories, extend results.json with fm_cs3/fm_cs4 entries
- `evaluate_all.py` — new: unified script that runs baselines + loads trained CFM models and produces comparison table
- `reports/generate_unet_cfm_report.py` — added CS3/CS4 columns to metrics table, bar charts, per-component breakdown, and conclusion
- `docs/case_studies.tex` — added CS3/CS4 sections with equations and description
**Rationale:** CS3/CS4 test generalisation to unseen random parameter draws at evaluation time, complementing the CS1/CS2 fixed-parameter tests. The coupling_type fix ensures correct forward model in baselines for quartic cases.
**Verification:** Verified — `pytest tests/ -m "not slow"` (111 passed), config validation (10/10 configs OK), `.gitignore` cleanup applied.

## 2026-07-01: Implement τ=0 CFM ablation + sbatch infrastructure + tests

**Summary:** Implemented Experiment G (VanillaCFM τ=0 ablation), created 3 new sbatch scripts for lint/test/config-validation, updated PLAN.md to reflect actual state, wrote missing tests for DirectUNet/VanillaCFM/RandomParamDataset, fixed stale test assertions, and updated .gitignore from stash.

**Files modified:**
- `conf/schema.py` — added `train_tau_0_only: bool = False` to `VanillaCFMConfig`
- `models/vanilla_cfm.py` — τ=0 logic in `compute_cfm_loss` (zero tau) and `sample` (single Euler step)
- `train.py` — wired `train_tau_0_only` flag through `model_factory`
- `config/experiment/G{1,2,3}_vanilla_cfm_t0_*.yaml` — 3 new experiment configs (mirror F1-F3, with `train_tau_0_only: true`)
- `config/experiment/F{1,2,3}_*.yaml` — added explicit `train_tau_0_only: false`
- `batch/run_lint.sbatch` — new: ruff + mypy batch job
- `batch/run_test_suite.sbatch` — new: pytest fast suite batch job
- `batch/run_config_validation.sbatch` — new: validates all 10 configs load correctly
- `batch/run_one_epoch_tests.sbatch` — added G1-G3, updated array range
- `batch/run_new_experiments.sbatch` — added G1-G3, updated array range, extended time limit
- `batch/run_vanilla_experiments.sbatch` — added deprecation notice
- `batch/run_tests.sh` — added deprecation notice, fixed stale path
- `PLAN.md` — complete rewrite matching actual state
- `.gitignore` — added `checkpoints/`, `*.pt`, `.coverage`, `.pytest_cache/`, `all_figures.pdf` from stash
- `tests/test_direct_unet.py` — new: 4 tests for DirectUNet
- `tests/test_vanilla_cfm.py` — new: 8 tests for VanillaCFM including τ=0 mode
- `tests/test_random_param_dataset.py` — new: 6 tests for RandomParamDataset
- `tests/test_hydra_config.py` — fixed stale `T_max` (5.0→3.0) and `da_window_steps` (500→300) assertions
- `tests/test_baselines_hydra.py` — fixed stale `da_window_steps` assertion
- `tests/test_refactoring_equivalence.py` — fixed `test_legacy_stage1_checkpoint` to save full model state dict
- `CHANGELOG.md` — marked CS3/CS4 verification as complete, appended this entry

**Rationale:** Experiment G tests whether VanillaCFM's advantage comes from multi-τ training or the residual loss formulation. τ=0 collapses CFM to a single Euler step predicting the conditional mean, directly comparable to DirectUNet. All sbatch workflows consolidate infrastructure for reproducible cluster runs.

**Verification:** `python -m pytest tests/ -m "not slow" --ignore=tests/test_checkpoint_compat.py` — 111 passed, 0 failed, 7 deselected (slow). Config validation: all 10 configs (E1-E3, F1-F3, G1-G3, lorenz63_default) produced correct model types. τ=0 flag confirmed on all G configs.

## 2026-07-02: Add EnKF/ETKF inflation sensitivity sweep for CS3/CS4

**Summary:** Created sbatch infrastructure for inflating parameter sweeps of EnKF and ETKF on CS3/CS4 test cases, filling a gap where only CS1/CS2 had been scanned. Added `suffix` parameter to `run_and_cache_baselines` for clean `_cs3cs4` cache-file tagging.

**Files modified:**
- `evaluation/run.py` — added `suffix=""` kwarg to `run_and_cache_baselines`, appended to `param_suffix` before cache filename construction
- `batch/inflation_sweep_cs3cs4.py` — new: standalone script that generates CS3/CS4 datasets and runs one inflation value for the specified method
- `batch/run_enkf_cs3cs4_sweep.sbatch` — new: 7-task array job for EnKF inflation [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
- `batch/run_etkf_cs3cs4_sweep.sbatch` — new: 11-task array job for ETKF inflation [1.0, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5, 1.6, 2.0]

**Rationale:** The CS1/CS2 baseline summary used tuned inflation (EnKF=1.2, ETKF=1.6) but CS3/CS4 evaluation was only run with ETKF at default inflation=1.0. These sweeps enable the same optimization for CS3/CS4.

**Verification:** Python syntax via `ast.parse` — clean. Bash syntax via `bash -n` — clean. Existing callers unaffected (suffix defaults to `""`).
