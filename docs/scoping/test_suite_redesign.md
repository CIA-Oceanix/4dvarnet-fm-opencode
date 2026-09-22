# Test-suite redesign: what should gate a PR

**Status:** PROPOSAL v1 (2026-09-22). Phase A1 implemented in the same PR as this
doc; the rest is designed, not built.

## 1. What is wrong today

CI runs `pytest tests/ --deselect tests/test_joint_estimation.py -m "not slow" -q`
-- **964 of 999 tests**, 56 modules, 5-8 min on the runner. Gating the whole tree
rather than a hand-maintained file list was the right call and is not in question
here. Four things are.

**1.1 Effort sits on code the papers do not use.** `test_fourdvarnet*.py` is
**171 tests (18% of the suite)** for the unrolled-solver family, which
`da_paper_structural_hypotheses.md` explicitly excludes. Every DA baseline
combined -- EnKF, ETKF, strong/weak-4D-Var, QG, data-leakage -- is **71**.

**1.2 The scientific assertions never run.** Of the 22 `slow`-marked tests, six
assert something a paper claim depends on. They are excluded from every PR and
from master.

**1.3 Three of those six are broken.** `test_strong4dvar_degrades_cs2`,
`test_strong4dvar_better_than_weak_cs1` and `test_enkf_mean_tracks_truth` all
fail with `AttributeError: 'NoneType' object has no attribute 'step'` -- they
construct `Strong4DVar(...)`/`EnKF(...)` without a `dynamics=` argument, relying
on a Lorenz-63 default that was removed. Line 42 of the same file passes it
explicitly, so the fix never reached the slow-marked tests. **The test asserting
the paper's headline S0->S1 degradation has been silently broken for an unknown
period**, which is precisely the failure mode the whole-tree change was meant to
end; `-m "not slow"` reopened it for these 22.

**1.4 The assertions that do exist are not statistically meaningful.** Every
4D-Var test assimilates **a single window** -- `da_window_steps=500`, `T_max=5.0`,
and `ds[0]` only, even where the config builds three. `test_strong4dvar_degrades_cs2`
asserts `rmse_cs2 > rmse_cs1 * 1.2` from one draw each, against a measured
per-window spread of **0.436-1.499** (>3x). A 1.2x margin on one window from that
distribution catches "the model-error wiring is disconnected", not a real
regression. Tolerances elsewhere are looser still: `rmse < 20`, `mean < 30`.

**1.5 One test does not do what it says.**
`test_weak4dvar_perfect_obs_low_rmse` is documented as *"Dense observations
should improve tracking over sparse observations"*; its three assertions are
`rmse > 0`, `rmse < 20`, `not NaN`. It never compares dense against sparse.

## 2. Principle

**A PR gate should be deterministic, fast, and sensitive. Scientific validation
is a different job with a different budget.** Conflating them is what produced
single-window tests with invented thresholds: an attempt to validate science
inside a time budget that cannot hold it.

The key technique is to stop asserting loose bounds and start asserting **stored
values on fixed seeds**:

```python
assert result.rmse.mean() == pytest.approx(0.725394, rel=1e-4)
```

This is strictly stronger than any threshold. A bound catches catastrophe; a
golden value catches *any* numerical change -- a changed default, a reordered
operation, a silent dtype shift -- and costs the same to run. Determinism was
verified for ETKF, Strong4DVar and L96Weak4DVar: repeated runs agree to all
printed digits.

## 3. Proposed tiers

### Tier A -- the PR gate, target <= 10 min

- **A1. Numerical golden-value regression.** *(implemented, this PR)* One test
  per scheme on a tiny L96 (`NO=2, J=4` -> 10-dim state, 20-step window, fixed
  seed), asserting a stored RMSE, plus contract, determinism and
  fully-masked-window checks for each. Measured cost: **21 tests in 29 s** for
  the five DA baselines (a single assimilation pass is ~1 s; the file makes
  about 30 of them).
- **A2. Scenario-wiring tests.** Assert that `S1` actually differs from `S0` --
  that `param_bias` reaches the DA model's parameters and not the truth. This is
  the class of bug that silently invalidates every model-error number in the
  papers, and it is cheap to guard.
- **A3. Sparse/NaN-observation handling** for every DA scheme, moved from L63 to
  tiny L96. The two L63 versions currently cost **92 s**; on tiny L96 this is
  seconds.
- **A4. Contract/unit tests** -- the existing ~800. Unchanged; this part works.

### Tier B -- nightly, no time limit

- **B1. The scientific relationships, done properly.** >=30 windows at realistic
  L96 (40-dim, 500-step), asserting what the papers claim: weak-4D-Var <=
  strong-4D-Var under model error; S1 RMSE > S0 RMSE per scheme; the marginal
  value of observations positive at S0 and collapsing at S1 -- with **tolerances
  derived from the measured per-window spread**, not guessed. This is where
  `test_strong4dvar_degrades_cs2` belongs: it is a 30-window claim, not a
  1-window one.
- **B2. Neural-scheme inference regression** against stored checkpoints.
- **B3. Report/benchmark consistency** end to end.

### Tier C -- manual/release
Full benchmark regeneration.

## 4. Lorenz-96 as the reference system

| system | role |
|---|---|
| **L96** | **primary** -- all DA and neural numerical and scientific tests |
| L63 | contract/API only; its DA science tests move to L96 |
| QG | minimal smoke (64x64 spectral, expensive); keep the psi/q and obs-geometry checks |

L96 is the only system carrying the full scheme matrix (DA baselines, DirectUNet,
CFM, SDA, blending), and it is where the papers' numbers come from. Moving the DA
science tests there also retires the three broken L63 tests by replacement rather
than repair -- their L96 equivalents are written fresh with `dynamics=` correct.

## 5. Costs and open questions

- **Golden values must be regenerated deliberately** when a change is intended.
  This needs a documented workflow and a rule that regenerating requires
  justification in the PR body, or the tests degrade into rubber stamps.
- **`rel=1e-4`, not exact.** CI hardware differs from the development machines.
  Still four orders tighter than `rmse < 20`.
- **Tier B needs infrastructure that does not exist**: a scheduled workflow and
  somewhere to put results.
- **Not yet decided:** whether `test_fourdvarnet.py`'s 142 tests are
  proportionate now that the unrolled family is out of scope for P1. They still
  guard live code, so the answer is probably yes, but it is worth an explicit
  decision rather than drift.

## 6. Order of work

1. **A1 for the DA baselines** -- highest value, small, no new infrastructure. *(this PR)*
2. **A2 + A3**; retire the broken L63 slow tests as their L96 equivalents land.
3. **A1 for the neural schemes.**
4. **Tier B**, once a nightly workflow exists.
