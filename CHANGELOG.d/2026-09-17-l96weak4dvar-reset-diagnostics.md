## 2026-09-17: L96Weak4DVar reset diagnostics + first real sanity-check result

**Summary:** Added instrumentation to `L96Weak4DVar` (`n_subwindows`,
`n_resets`, `reset_step_indices`) to distinguish "stable and learning a
real correction" from "stable only because it keeps resetting to the
pure background/dynamics forecast" -- motivated by the first real
sanity-check result below, which is stable but suspicious.
**Files modified:**
- `evaluation/baselines.py` -- `L96Weak4DVar.__init__` gains
  `self.n_subwindows`/`self.n_resets`/`self.reset_step_indices`;
  `_reset_nan` (now an instance method, was `@staticmethod`) increments
  `n_resets` and appends the triggering Adam step index (or `None` for an
  LBFGS reset, which has no single triggering step) to
  `reset_step_indices`; `_optimize` increments `n_subwindows` once per
  call and passes the step index through to `_reset_nan`.
- `tests/test_baselines_l96weak4dvar.py` -- updated `_reset_nan` test
  (no longer callable as a staticmethod); new
  `test_reset_nan_records_diagnostics`, `test_no_reset_when_loss_stays_finite`,
  and updated Adam/LBFGS reset tests to assert the new counters (Adam's
  always-NaN-loss case resets on step 0 as expected;
  LBFGS's always-NaN-loss case never even computes a gradient, so it
  correctly stays at its finite initial value with zero resets -- a real,
  documented edge case, not a bug).
**First real sanity-check result** (3 S0 test windows, `da_window_steps=500`,
`mode="weak"`, `optimizer="adam"`, `opt_steps=150, lr=0.02` -- this
project's own established Weak4DVar default, the exact setting that gave
the plain `Weak4DVar` class RMSE~12-26): all 3 windows finite (**no
NaN/divergence** -- the safety net works), but RMSE~24.4, suspiciously
uniform across all 3 windows and both slow/fast variable groups. This
pattern is consistent with the reset firing so often that the "analysis"
degenerates into an uncorrected free forecast from the noisy background --
stable, but not yet a useful DA scheme. The new diagnostics let the next
run report exactly how often/when resets fire per sub-window, rather than
guessing.
**Verification:** `pytest tests/test_baselines_l96weak4dvar.py -q` (14
passed, `fdv` env); `pytest tests/test_baselines_enkf.py
tests/test_baselines_strong4dvar.py tests/test_baselines_weak4dvar.py
tests/test_baselines_l96weak4dvar.py tests/test_energy_score.py -q -m "not
slow"` (32 passed, same 7 pre-existing unrelated failures as the prior
fragment); `ruff check evaluation/baselines.py
tests/test_baselines_l96weak4dvar.py` clean.
