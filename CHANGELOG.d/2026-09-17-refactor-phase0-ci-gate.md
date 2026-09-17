## 2026-09-17: Refactor Phase 0c — CI gate covers the whole test tree

**Summary:** Replaced the hand-maintained CI file list (24 of 53 test files)
with `pytest tests/ -m "not slow"`, after triaging all 27 previously-ungated
files and fixing the stale tests that had accumulated in them.

**Files modified:**
- `.github/workflows/ci.yml` — gate is now the whole `tests/` tree, with two documented deselections
- `tests/conftest.py` — new `l63_dynamics` fixture
- `tests/test_baselines_{enkf,strong4dvar,weak4dvar}.py` — inject `dynamics=`
- `tests/test_refactoring_equivalence.py`, `tests/test_random_param_dataset.py` — NaN-aware comparisons, current key set
- `docs/refactor_plan.md` — triage outcome, the L63 seed bug, the orphaned module

**Rationale:** A new test file is now gated by existing, rather than by
remembering to edit `ci.yml`. The previous list was last extended one commit
before this work started (`c2ce104`), which is the trap firing.

Triage of the 27 ungated files found 18 test failures across 6 files. **None
was a product bug except one.** Ten were stale tests fixed here; seven are
deferred; one is a real data bug, recorded but not fixed.

**Verification:** `pytest tests/ --deselect tests/test_joint_estimation.py
--deselect tests/test_lorenz63.py::test_observations_noise -m "not slow"` →
808 tests collected, 36 deselected, 0 collection errors.

**Deliberate exclusions** (both tracked in `docs/refactor_plan.md`):
- `tests/test_joint_estimation.py` — 7 tests against `JointCFM`'s
  pre-restructure API. `JointCFM` is covered by `test_joint_estimation_l96_neural.py`
  (25 tests) and `test_neural_inference.py`. Deferred to a second step.
- `tests/test_lorenz63.py::test_observations_noise` — correctly detects that
  `Lorenz63Dataset` gives every window the same observation-noise realization
  (`data/lorenz63.py:160` uses a constant `obs_seed` where `lorenz96.py` and
  `random_param_dataset.py` use a per-window one). Fixing it re-randomises every
  L63 dataset and moves published L63 numbers, so it is a deliberate scientific
  decision rather than part of a test-gate change. Impact is bounded to
  base-`Lorenz63Dataset` results; L96 and QG are unaffected.

**Note on CI duration:** per-file timings were measured on a machine under
heavy concurrent load and are unreliable as a budget. The real number should
come from this PR's own CI run; add `@pytest.mark.slow` markers afterwards only
if it proves too slow.
