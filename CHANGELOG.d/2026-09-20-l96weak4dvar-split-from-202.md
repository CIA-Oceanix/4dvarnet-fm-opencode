## 2026-09-20: Split `L96Weak4DVar` out of PR #202 onto master

**Summary:** PR #202 (`feature/l96-fdv2-stier-sweep`) had grown to 64 files and
four unrelated lines of work — the FDV2 S-tier gradient sweep, true-ODE priors,
a self-supervised `var_cost` loss, and this weak-constraint 4D-Var class — on one
branch that had gone `DIRTY` and had never once run CI. This lands the
`L96Weak4DVar` half on its own, because it is the item that unblocks **D0**, the
P0 of `docs/scoping/da_paper_structural_hypotheses.md` (§6/§8), and it is
independent of the other three.

**Files modified:**
- `evaluation/baselines.py` — `L96Weak4DVar`, a **pure 209-line addition** after
  `Strong4DVar` with zero edits to existing code (verified: the diff is a single
  hunk). Master's own `_ESAccumulator.step` numpy-guard fix, which postdates
  #202's fork point, is retained.
- `tests/test_baselines_l96weak4dvar.py` — 14 tests, imported unchanged.
- The four `2026-09-17-l96*weak4dvar*` fragments, imported unchanged as the
  provenance of the investigation.

**Rationale:** Nothing in this half depends on the FDV2/true-prior work, and
resolving a 64-file conflict is not a prerequisite for running the D0 benchmark.
Splitting also gets CI onto this code for the first time.

**Not included, and deliberately so:**
- **No wiring.** `L96Weak4DVar` is not referenced by `evaluation/run_l96.py` or
  any other driver (verified by grep), so this PR adds a class and its tests, not
  a benchmark row. The S0/S1 run is the follow-up.
- **No default change.** `optimizer` still defaults to `"adam"`, which the
  investigation showed gives RMSE ~24 on a 500-step window while LBFGS gives
  ~0.98. The imported fragment flags flipping it as an explicit decision; call
  sites must pass `optimizer="lbfgs"` until that decision is made.

**Verification (re-run on current master, not inherited from #202):**
`pytest tests/test_baselines_l96weak4dvar.py -q` — 14 passed. `pytest
tests/test_baselines_enkf.py tests/test_baselines_strong4dvar.py
tests/test_baselines_weak4dvar.py tests/test_energy_score.py -q -m "not slow"` —
25 passed, 7 deselected. `ruff check evaluation/baselines.py
tests/test_baselines_l96weak4dvar.py` — clean. Note the imported port fragment's
own Verification section records 7 *failures* in that neighbouring set; those
were the stale `NoneType .step` constructor failures, which no longer occur on
master — the 7 here are `slow`-marker deselections, not failures.
