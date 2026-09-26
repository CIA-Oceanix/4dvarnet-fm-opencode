## 2026-09-25: L96 benchmark reports regenerate from per-report input bundles, not topic worktrees

**Summary:** The P1, benchmark-default and extended generators no longer hard-code
sibling-worktree paths. Their input roots resolve through `reports/l96/_inputs.py`
to `experiments/l96/report_inputs/<report>/<root>/` in the main checkout. These are
hard-link bundles of exactly the files each report was generated from, built and
verified by `scripts/bundle_report_inputs.py`. A new CI guard fails on any
worktree or absolute repository path in a report script (QG is exempt until its
rework).

**Files modified:**
- `reports/l96/_inputs.py` (new) — the root resolver. `FDV_REPORT_INPUT_ROOTS`
  overrides it, for the bundler only.
- `reports/l96/generate_p1_l96_benchmark.py`,
  `reports/l96/generate_l96_benchmark_default_report.py` and
  `reports/l96/generate_l96_benchmark_extended_report.py` read through `_inputs`.
  Their metrics caches stay in the running checkout's `experiments/`.
- `scripts/bundle_report_inputs.py` (new) — `record` (audit hook on `open`),
  `link` (hard links + `MANIFEST.json`; also flags eval sidecars whose recorded
  dataset path points into a worktree) and `run`. This is the only place that
  still names the topic worktrees, as the record of where the bundles came from.
- `batch/run_l96_bundle_report_inputs.sbatch` (new) — record, then link, then
  a bundle-only run, diffing the outputs after each step.
- `tests/test_report_inputs.py` (new), `reports/README.md`,
  `docs/plans/tech/archive.md`.

**Rationale:** Step R3 of the docs/reports refactor. #267 archived the runs but
the generators still read five sibling worktrees. The per-run archive can't hold
the same run scored under two protocols, nor the extended report's sub-studies
(`eval_factorial/`, `eval_ood/`, `eval_val/`). Hard links use no extra disk and
survive worktree pruning, which makes the pending worktree cleanup safe for these
reports. Replacing the bundled `members_*.npz` with `scores_*.npz`, to free their
disk, is a planned follow-up.

**Verification:** Run as SLURM jobs 55792, 55798, 55809, 55813, in three phases:
- **Record.** Each generator ran against the topic worktrees under the audit
  hook. All three reports came out byte-identical to the checked-in ones,
  figures included.
- **Link.** `link --apply` hard-linked 46 files for P1, 162 for
  benchmark-default and 2,745 for extended. There were no conflicts, and no eval
  sidecar records a dataset path inside a worktree.
- **Bundle-only run.** All three reports again came out byte-identical.

The first bundle-only run differed for the extended report. 176 result
directories lacked their `*_eval.json`: `done()` tests for it by existence but
the generator never opens it. So `link` now also bundles the `*_eval.json`
beside every linked file.

Tests: `pytest tests/test_report_inputs.py tests/test_reports_index.py
tests/test_docs_layout.py tests/test_l96_report_consistency.py` pass.
