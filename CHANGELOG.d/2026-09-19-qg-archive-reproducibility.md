## 2026-09-19: QG neural benchmark reproducible from stored checkpoints

**Summary:** Ported PR #219's L96 archive conventions to QG. Training now writes
`resolved_config.yaml`; evaluation reads each scheme's architecture from the run
instead of re-declaring it; the normalization stats and checkpoints are
hard-linked into `experiments/qg/` under their `experiment_id`; and the
cross-scenario results JSON the published table is built from is tracked in git.
The four neural rows #208 had silently dropped from
`reports/qg/outputs/qg_neural_report.md` are restored.

**Files modified:**
- `evaluation/archive.py` — resolver generalized to a `RunArchive` layout with
  `L96` and `QG` instances; module-level functions kept as L96's, so no existing
  caller changed. Norm-stats resolution gained an archive-dir step.
- `evaluation/qg_runs.py` (new) — `architecture()` / `checkpoint()` /
  `norm_stats_paths()` for an archived QG run, with a documented fallback chain
  (`resolved_config.yaml` → `results.json` → `config/experiment/<run>.yaml`).
- `train_qg_neural.py` — writes `resolved_config.yaml` before training starts;
  `DEFAULT_HIDDEN_CHANNELS` replaces three repeated literals so the recorded
  capacity is provably the one built.
- `eval_qg_neural_s0_s1.py` — `SCHEMES` reduced to run name + eval-time
  `cond_mode`; architecture, checkpoint and stats resolved through the archive;
  `load_state_dict(strict=True)`; per-row provenance written into the results JSON.
- `scripts/consolidate_qg_archive.py` (new) — `--check[--portable]` audit and
  `--apply --manifest` hard-link consolidation.
- `batch/run_qg_neural_s0s1_eval.sbatch` (new) — regenerates the table's neural
  half from the archive.
- `reports/qg/outputs/qg_neural_report.md` — regenerated, neural rows restored.
- `reports/qg/outputs/qg_neural_s0_s1_cross_scenario/results_lag5_noise0.05_bias0.1.json`
  — now tracked.
- `docs/archive.md` — QG section.
- `tests/test_archive.py` (+7), `tests/test_qg_config_persistence.py` (new, 17),
  `tests/test_qg_report_consistency.py` (new, 10).

**Rationale:** The QG half of the case study had L96's pre-#219 problem in a
sharper form. Three things lived only in the training worktree — the results
JSON the report reads, the `qg_*_norm_stats.pt` files without which a checkpoint
predicts in uninvertible units, and any run not yet moved into `experiments/qg/`
— so the archived Q1-Q4 checkpoints were reachable from master and unusable
there. It had already cost a published result: #208 (2026-09-16) regenerated
`qg_neural_report.md` in a worktree without the JSON, the generator returned
`None` for the neural half, and the benchmark table was published with its four
DA rows alone. Separately, `eval_qg_neural_s0_s1.py` re-declared every scheme's
`param_dim`/`cond_extra_dim`/`ic_dim` as a literal next to a hardcoded
checkpoint path, with nothing tying the two together.

`cond_mode` is deliberately left at the call site: it is an evaluation-time
choice (`"scenario"`), not a property of the checkpoint (trained `"true"`/
`"noisy"`), and reading it from the config would quietly make the S1 comparison
meaningless.

**Verification:**
- CI's own invocation, `pytest tests/ --deselect tests/test_joint_estimation.py
  -m "not slow"` — 876 passed, 5 skipped, 35 deselected (13m44s), matching the
  `pytest` check on the PR. (`test_joint_estimation.py` is CI's one standing
  exclusion, tracked in `docs/scoping/refactor_plan.md`.)
- `ruff check` on every touched file — clean.
- Architecture read from each archived run matches the literals it replaces, and
  all four checkpoints `load_state_dict(strict=True)` into the model built from
  it (14,506,812 / 14,575,932 / 14,575,932 / 14,610,492 params).
- `scripts/consolidate_qg_archive.py --apply --manifest` linked 23 artifacts;
  `--check --portable` went from 5 unresolvable to 0. Hard links, so
  `experiments/qg/` stayed at 2.9 GB on a volume at 98%.
- Jobs 54246/54247 re-ran the full S0/S1 evaluation from the archive. Every
  published EV reproduces at the table's own 4-decimal precision (Q1
  0.9805/0.6238 both scenarios, Q2 0.9827/0.6093 and 0.9490/0.5571, Q3
  0.9798/0.5978 and 0.9763/0.5959, Q4 0.9824/0.6745 and 0.9808/0.6744). It is
  not bit-identical: across all 32 stored metrics the largest relative
  difference is 1.7e-05, consistent with float32 kernel selection differing
  between GPUs, and the tracked JSON is left as published rather than
  overwritten with the re-run.
