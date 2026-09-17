## 2026-09-17: Refactor Phase 0 — unbreak pytest collection, contributor docs

**Summary:** First phase of the contributor-facing refactor. Moved three
non-test scripts out of `tests/` so `pytest tests/ -m "not slow"` collects
cleanly again (822 tests, 0 errors — previously "Interrupted: 2 errors"),
and added the contributor documentation a newcomer needs before touching code.

**Files modified:**
- `tests/test_numerical_equivalence.py` -> `scripts/verification/check_numerical_equivalence.py` — moved; stale positional `Lorenz63Dynamics.step(s, W, sigma, rho, beta)` call repaired to keyword form
- `tests/test_equiv_report.py` -> `scripts/verification/check_equiv_report.py` — moved
- `tests/compare_rmse_slices.py` -> `scripts/verification/compare_rmse_slices.py` — moved
- `scripts/verification/README.md` — new; explains these are manually-run scripts, not tests
- `docs/refactor_plan.md` — new; the phased plan, findings that drive it, and delegation table
- `docs/CONTRIBUTING.md` — new; one worked L96 loop end to end with real commands
- `README.md` — rewritten; led with the science and the three case studies instead of the worktree list
- `.gitignore` — ignore the nested `4dvarnet-fm-*/` topic worktrees

**Rationale:** The three moved files define **zero** test functions between them
— all their work runs at module import, so pytest collection executed each
script and crashed. `check_equiv_report.py` additionally hardcodes
`torch.device("cuda")` and runs a full DA suite, so it could never have passed
on a CPU runner. They were verification scripts misfiled under `tests/` with
`test_*.py` names, not broken tests.

The `TypeError: Lorenz63Dynamics.step() takes 3 positional arguments but 6 were
given` is **not a live regression**: `step(state, forcing, **kwargs)` takes its
parameters by keyword and the production path already does that. Only this dead
script still called it positionally. Automated equivalence coverage lives in
`tests/test_refactoring_equivalence.py`.

Scripts were moved rather than deleted so the verification record survives.

**Verification:** `pytest tests/ -m "not slow" --collect-only` → `822/844 tests
collected (22 deselected)`, zero errors, down from `Interrupted: 2 errors during
collection`. Full-suite triage run (to size the 29 files not yet in the CI gate)
in progress; flipping the gate to `pytest tests/ -m "not slow"` is the next step
and is deliberately held until that run reports.
