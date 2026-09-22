## 2026-09-22: Test-suite redesign proposal + Tier A1 golden-value tests for the L96 DA baselines

**Summary:** Adds `docs/scoping/test_suite_redesign.md` (the proposal) and
`tests/test_da_golden_l96.py` (its first phase): 21 deterministic
numerical-regression tests for ETKF, EnKF, Strong4DVar, Weak4DVar and
L96Weak4DVar on a tiny L96, running in 29 s.

**What the audit found.** CI runs 964 of 999 tests across 56 modules. Four
problems:
1. **Effort is on code the papers exclude.** `test_fourdvarnet*.py` is **171
   tests (18%)** for the unrolled-solver family; every DA baseline combined is
   **71**.
2. **The scientific assertions never run** -- six of the 22 `slow`-marked tests
   assert something a paper claim depends on, and `-m "not slow"` excludes them
   from PRs *and* from master.
3. **Three of those six are broken**, failing with `AttributeError: 'NoneType'
   object has no attribute 'step'`: they construct `Strong4DVar(...)`/`EnKF(...)`
   without `dynamics=`, relying on a removed Lorenz-63 default. **The test
   asserting the papers' headline S0->S1 degradation has been silently broken
   for an unknown period** -- exactly the failure the whole-tree CI change was
   meant to end, reopened by the `slow` marker.
4. **The assertions are not statistically meaningful.** Every 4D-Var test
   assimilates *one* window (`ds[0]`, even where the config builds three) and
   asserts a 1.2x margin, against a measured per-window spread of 0.436-1.499.

Also found: `test_weak4dvar_perfect_obs_low_rmse` is documented as comparing
dense against sparse observations and never does -- its assertions are
`rmse > 0`, `rmse < 20`, `not NaN`.

**The design principle.** A PR gate should be deterministic, fast and sensitive;
scientific validation is a different job with a different budget. Conflating them
is what produced single-window tests with invented thresholds. So Tier A pins
**stored values on fixed seeds** (`pytest.approx(0.725394, rel=1e-4)`) instead of
bounds -- strictly stronger at the same cost, since a bound catches catastrophe
while a golden value catches any numerical change. Tier B (nightly, >=30 windows,
tolerances derived from the measured spread) is where the scientific
relationships belong.

**Determinism was verified, not assumed** -- repeated runs of ETKF, Strong4DVar
and L96Weak4DVar reproduce to all printed digits. `rel=1e-4` rather than exact
because CI hardware differs; still four orders tighter than `rmse < 20`.

**Files added:** `docs/scoping/test_suite_redesign.md`;
`tests/test_da_golden_l96.py` (golden value, output contract, determinism,
fully-masked window, and a cost canary per scheme).

**Not done in this PR**, and listed in the doc: scenario-wiring tests (A2),
retiring the broken L63 slow tests once their L96 equivalents exist (A3), golden
values for the neural schemes, and Tier B's nightly workflow.

**One incidental measurement:** `Strong4DVar` at `max_iter=1` pays a one-off
~16 s LBFGS warmup for an RMSE identical to `max_iter=2` (0.670710 both), so the
tests use 2.

**Verification:** `pytest tests/test_da_golden_l96.py -q` -- 21 passed in 29.09 s.
`ruff check` on the new file -- clean.
