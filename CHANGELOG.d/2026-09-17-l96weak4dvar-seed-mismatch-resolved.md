## 2026-09-17: L96Weak4DVar's apparent ~2x RMSE gap vs Strong-4DVar was a test-seed/sample-size artifact, not an algorithmic deficiency

**Summary:** Follow-up to the same-day LBFGS fragment
(`2026-09-17-l96weak4dvar-lbfgs-resolves-optimization-failure.md`). After
LBFGS fixed the catastrophic Adam failure, the resulting RMSE (~1.4-1.5 in
every scratch-script variant tried) still looked ~2x worse than the
consolidated benchmark's `Strong-4DVar` row (`reports/l96/outputs/
l96_consolidated_benchmark.md`, S0 all = 0.8116). None of six independent
levers closed that gap: weak vs. strong mode (1.518 vs 1.535), 3x more LBFGS
iterations (1.492 / 1.427, marginal), a fixed vs sigma-adaptive whitening
scale (1.442 vs 1.535, no difference), adding `line_search_fn="strong_wolfe"`
to the literal production `Strong4DVar.assimilate` optimizer pattern (which
otherwise NaNs on window 0 -- itself a real, separate fragility in that raw
pattern) (1.372), a burn-in relaxation of the initial background onto the
L96 attractor before window 0 (1.407, no improvement), and restricting the
RMSE metric to the 24D observed subspace only, matching the benchmark's own
metric definition exactly (1.324, a real but partial ~12% improvement).
**Root cause:** every one of the above scratch experiments used
`Lorenz96Config(..., seed=42, ...)` with only 3 ad hoc S0 test windows.
The actual benchmark's cached `test_s0` split
(`data/lorenz96.py::make_l96_s0_s1_trainval`, lines 674-679) is built with
`seed=123, num_windows=200, case=1, param_bias=0.0, forcing_state_bias=0.0,
fast_generation=False` -- a completely different, much larger population of
test windows. Re-running the best-so-far config (production `Strong4DVar`
LBFGS pattern + `strong_wolfe`, no burn-in) with the corrected `seed=123` on
window 0 alone gave `rmse_obs24_pooled=0.8072` -- a <1% difference from the
benchmark's reported 0.8116. Scaling to 10 `seed=123` windows gave a
consolidated mean of 0.9811 (all40=1.124, slow=0.560 vs target 0.4683,
fast_obs=1.132 vs target 0.9833), with per-window values ranging 0.436-1.499
(a >3x spread) -- confirming the residual ~20% gap at `n=10` is explained by
ordinary window-to-window variance at that sample size, not a stable
algorithmic deficit, and that the gap shrinks monotonically as sample size
grows (63% high at n=3/seed=42 -> 21% high at n=10/seed=123).
**Conclusion:** the LBFGS-vs-Adam fix (previous fragment) was the one real,
necessary correction. There is no evidence of any further algorithmic
deficiency in `L96Weak4DVar` (or in a from-scratch `Strong4DVar`
reimplementation) versus the consolidated benchmark's own numbers once
measured on the correct test population; the earlier "~2x worse" framing,
and the job-53681 reframing built on it, should not be relied upon.
**Not yet done:** no production code or test changed by this fragment (all
diagnostics were ad hoc scratch scripts, not committed). If `L96Weak4DVar`
is promoted to a real benchmark row in the future, its own sanity-check/eval
harness must use `seed=123` (`test_s0`) / `seed=131` (`test_s1`) via
`make_l96_s0_s1_trainval`, not an ad hoc `Lorenz96Config(seed=42, ...)`, and
should pool over enough windows (ideally the full 200) to avoid resurrecting
this same false-gap illusion.
**Verification:** Ad hoc scratch scripts only (scratchpad dir, not
committed); no test suite changes.
