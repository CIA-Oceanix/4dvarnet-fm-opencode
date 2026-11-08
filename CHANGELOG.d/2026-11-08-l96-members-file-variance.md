## 2026-11-08: `--members-file variance` stores the ensemble variance

**Summary:** A new `--members-file variance` mode in the L96 neural, SDA and hybrid eval scripts. It writes the usual per-window `scores_<case>.npz`, plus the ensemble variance `(W, T, D)` as float32 with ddof=1, the same convention as the DA trajectory files. A batch script re-evaluates the three 2D learned rows of the P1 tables with it.
**Files modified:**
- `evaluation/estimate_metrics.py`, `evaluation/members_store.py`: the new mode.
- `eval_neural_l96.py`, `eval_sda_l96.py`, `eval_sda_directunet_hybrid_l96.py`: the new mode added to the option's choices.
- `batch/run_l96b_2d_variance_eval.sbatch`: re-evaluates VanillaCFM-2D-M, SDA1-2D-M and the DirectUNet-2D-M -> SDA1-2D-M hybrid (seed 1), with the original sampler arguments and seed.
- Tests: `tests/test_ensemble_window_scores.py`, `tests/test_members_store.py`.
**Rationale:** The pooled spread-skill ratio of the paper (eq. `ss`) and the Gaussian-form FMS need the per-time variance. The per-window `spread`, a mean standard deviation, understates the pooled ratio by 20-30 % (Jensen), and full member dumps are GBs.
**Verification:** Job 70543 reproduces the stored benchmark RMSE and CRPS to 5 decimals for all three rows. The two test files pass (20 tests), and `ruff check` passes.
