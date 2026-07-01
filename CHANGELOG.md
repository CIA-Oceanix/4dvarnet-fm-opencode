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

## 2026-07-01: Baseline summary JSON + experiments/summaries/ folder

**Summary:** Created `experiments/summaries/baselines_summary_cs1_cs2.json` consolidating Weak-4DVar, Strong-4DVar, EnKF (inf=1.2), and ETKF (inf=1.6) metrics and trajectory file paths for CS1/CS2. All methods verified on identical trajectories. Updated `.gitignore` to track `experiments/summaries/` while keeping the rest of `experiments/` ignored. Moved exp/g-tau0-cfm-ablation branch to remote.

**Files modified:**
- `.gitignore` — added `!experiments/summaries/`, `!experiments/summaries/**` to allow tracking
- `experiments/summaries/baselines_summary_cs1_cs2.json` — new: consolidated DA baseline summary

**Rationale:** Summary file provides a single source of truth for comparing baseline methods with learned models. Tracked subfolder enables sharing results while keeping large generated data (npz/pt) gitignored.

**Verification:** Trajectory truths match across all four methods (max_diff=0). JSON contains per-variable X/Y/Z metrics, degradation ratios, and file paths.

## 2026-07-01: Verify G3 inference + forced τ=0 CFM comparison + CS1/CS2 model summary

**Summary:** Verified G3 (τ=0 CFM) inference is robust across multiple x0 noise seeds (variance < 0.001). Evaluated F1/F2/F3 with forced τ=0 single-step sampling to quantify degradation from multi-τ training. Generated comprehensive CS1/CS2 comparison report across all 4 model families (DirectUNet, VanillaCFM 10-step, VanillaCFM forced τ=0, τ=0 CFM).

**Files modified:**
- `batch/verify_g3_inference.py` — new: standalone G3 verification with 3 x0 seeds
- `batch/run_verify_g3_inference.sbatch` — new: sbatch runner for verification
- `batch/compute_forced_tau0.py` — new: forced τ=0 evaluation for F1/F2/F3
- `batch/run_compute_forced_tau0.sbatch` — new: sbatch runner for forced τ=0
- `reports/generate_cs1_cs2_summary.py` — new: report generator with per-group best-in-class
- `experiments/summaries/cs1_cs2_unet_cfm_comparison.json` — new: full comparison data
- `CHANGELOG.md` — appended this entry

**Rationale:** Validate G3's suspiciously good 0.032 RMSE, quantify VanillaCFM's reliance on multi-step Euler integration, and consolidate all CS1/CS2 model results in a single report.

**Verification:** G3 verified across 3 seeds (42, 123, 999): CS1=0.031975±0.000013, CS2=0.032000±0.000024. Forced τ=0 degrades F3 from 0.070→0.094 (CS1). Data independence confirmed: train/val/test use different seeds (42/99/123/124), producing decorrelated Lorenz attractor trajectories after 10k-step spinup.
