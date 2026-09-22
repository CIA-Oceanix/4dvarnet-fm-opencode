## 2026-09-22: Golden values are not portable for optimiser-driven schemes on chaotic dynamics

**Summary:** The first CI run of `tests/test_da_golden_l96.py` failed on one of
five schemes, which corrects a premise in the same PR's proposal.

**What happened.** Four DA baselines reproduced their pinned RMSE exactly on the
GitHub runner. `L96Weak4DVar` returned **1.921647 against a local 1.072387** --
a 79% difference, not floating-point noise.

**Why, and why it is not a bug.** It is the only one of the five that runs an
iterative optimiser (Adam by default) over a *free per-step model-error control*
through a chaotic unroll: the largest control space of the set, with Lyapunov
amplification between iterations. A platform-level difference of 1e-16 in an
early gradient is enough to land elsewhere after five steps.

**The error was mine, and specific.** I verified determinism by running each
scheme twice *on the same machine* and concluded the values could be pinned.
Determinism on one machine does not imply reproducibility across machines, and
for this class of computation it positively does not.

**Fix.** `GOLDEN` now holds the four schemes verified to reproduce across
machines; `NO_CROSS_PLATFORM_VALUE` holds `L96Weak4DVar`, which keeps every other
Tier A check -- output contract, same-machine determinism, fully-masked-window
handling -- and whose numerics move to the nightly tier, where a multi-window
mean is stable enough to assert on.

**Explicitly not fixed by widening the tolerance.** At 79% no tolerance is
meaningful, and one loose enough to pass would detect nothing -- which would
leave a test that looks like a guard and is not. The code comment and the scoping
doc both say so, because widening it is the obvious wrong move for whoever hits
this next.

**Verification:** `pytest tests/test_da_golden_l96.py -q` -- 20 passed in 32.50 s.
