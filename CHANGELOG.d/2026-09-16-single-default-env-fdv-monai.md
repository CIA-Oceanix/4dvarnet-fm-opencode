## 2026-09-16: Consolidate onto a single default conda env (fdv-monai-proto)

**Summary:** The project no longer splits work across two conda envs. `fdv-monai-proto`
(torch 2.8.0+cu126 + monai 1.6.0) becomes the single default env for every batch script,
helper script and doc; the older `fdv` env (torch 2.4.1+cu121, no monai) is kept frozen
only as a rollback path. All `torch.load` calls on project-produced artifacts now pass
`weights_only=False` explicitly, which torch >= 2.6 requires for pickled Dataset objects
and numpy arrays.

**Files modified:**
- `requirements.txt` — now pins `torch==2.8.0+cu126` + `monai==1.6.0` (with the cu126
  `--extra-index-url` install note); `requirements-monai.txt` deleted, its content folded in.
- `AGENTS.md` — new "Conda environment" section: default env, per-run override, the frozen
  `fdv` rollback path, and the `torch.load`/`weights_only` gotcha.
- `batch/**` (~174 scripts), `scripts/agent_review_loop.sh`, `batch/run_tests.sh`,
  `demos/README.md`, `demos/IMPLEMENTATION_SUMMARY.md`, `tests/TEST_SUITE_SUMMARY.md` —
  `envs/fdv/bin` -> `envs/fdv-monai-proto/bin`, `CONDA_ENV:-fdv` / `ENV_NAME:-fdv` /
  `CONDA_DEFAULT_ENV=fdv` / `conda activate fdv` all repointed.
- 17 Python files (`eval_baselines.py`, `run_experiment.py`, `run_experiments.py`,
  `data/lorenz63.py`, `data/qg.py`, `data/qg_neural.py`, `training/lightning_module.py`,
  `batch/inflation_sweep_cs3cs4.py`, `batch/inflation_sweep_cs4b.py`,
  `batch/noise_density_sweep.py`, `eval_qg_neural_s0_s1.py`, `eval_qg_q1_lag5_noise05.py`,
  `train_qg_neural.py`, `reports/qg/assemble_qg_windows.py`,
  `reports/qg/fix_qg_test_obs_ic.py`, `tests/test_checkpoint_compat.py`,
  `tests/test_numerical_equivalence.py`) — `torch.load(..., weights_only=False)`.
- `tests/test_qg_neural.py` — `test_ensure_truth_cache_redrawn_matches_plain_when_cfg_unchanged`
  compared `obs` with `torch.equal`; relaxed to a tight `torch.allclose`
  (rtol=1e-5, atol=1e-3), docstring updated with the measurement.
- `models/monai_unet_adapter.py`, `models/monai_unet_qg2d.py`,
  `tests/test_l96_normalization_configs.py`,
  `batch/run_l96_vanilla_cfm_norm_eval_only.sbatch` — docstrings/comments that told readers
  to use a separate monai env updated to the single-env reality.

**Rationale:** The two-env split existed only because an early attempt to install monai into
the shared env dragged torch forward and broke CUDA (CHANGELOG 2026-09-06). That no longer
holds. Measured 2026-09-16: `fdv-monai-proto` is a strict superset of `fdv` — identical
Python 3.10.15 and identical versions for all 356 shared packages; the entire delta is
torch (2.4.1+cu121 -> 2.8.0+cu126), triton, the nvidia-* CUDA libs, and monai. Both envs
expose the same `sm_50..sm_90` arch list, so RTX8000/A40/H100/H200 all keep working. The
remaining hazard was torch 2.6's `weights_only=True` flip, which silently broke loading of
pickled caches (confirmed: `dataset_cache/l96_200w_ens200.pt` fails to load under torch 2.8
before this change) — fixed here.

The one behavioural difference the switch surfaced is test-only:
`test_ensure_truth_cache_redrawn_matches_plain_when_cfg_unchanged` asserted `torch.equal`
between two paths that reach `_generate_obs_ic`'s spectral field extraction with
differently-shaped intermediates. torch 2.4 made them bit-identical; torch >= 2.8 picks a
different kernel for one of them, giving a <=1 ULP float32 difference (measured: 6.1e-05
absolute, 7e-08 relative, on obs of magnitude ~6e+02 — verified against unmodified code in
both envs, so it is a torch-version effect, not a regression from this PR). The assertion
is now `torch.allclose(rtol=1e-5, atol=1e-3)`, still ~6 orders of magnitude tighter than a
genuinely different obs realization.

**Verification:**
- Numerics bit-identical across the two torch versions to 12 significant digits: QG 200-step
  spectral RK4 rollout, L96 500-step RK4 rollout, GPU matmul, Cholesky, `eigvalsh`.
- `pytest tests/test_fourdvarnet.py tests/test_lightning_module.py
  tests/test_neural_inference.py tests/test_checkpoint_compat.py
  tests/test_config_persistence.py tests/test_direct_unet.py tests/test_hydra_config.py
  tests/test_l96_normalization.py tests/test_obs_density.py tests/test_lorenz96_training.py
  -m "not slow"` — 250 passed, 2 skipped, identical in BOTH envs.
- Throughput identical within noise (100 UNet1D train iters 1.26/1.28 s in `fdv` vs
  1.35/1.21 s in `fdv-monai-proto`; 200-step QG rollout 0.66/0.67 vs 0.65/0.66 s).
- `ruff check` on every touched Python file — clean.
