## 2026-09-17: L96Weak4DVar sanity check resolved -- Adam was the actual bottleneck, not the weak-constraint parameterization

**Summary:** The `L96Weak4DVar` (QG4DVar-style whitened-control) sanity check
that previously ran stably (`n_resets: 0`, no NaN) but with a poor RMSE~24.4-25.7
is now traced to Adam's inability to optimize through the ~500-step chaotic
RK4 unroll -- switching to LBFGS (`Strong4DVar`'s own validated
`max_iter=10, lr=0.2`) with everything else unchanged (same whitening,
gradient sanitization/clipping, NaN-reset) brings RMSE down to ~1.5, a real
working weak-constraint DA scheme.
**Diagnosis:** Instrumented one S0 test window (`da_window_steps=500`,
`mode="weak"`, `optimizer="adam"`, `opt_steps=150, lr=0.02`) to print the raw
`Jo`/`Jb`/`Jq` cost breakdown and `w_ctrl`/`u` control norms per sub-window:
`n_obs_t=5/500` (confirms extreme temporal sparsity within each window, as
expected from `obs_interval=100`), but critically `Jo` stayed at ~80,000-90,000
across all 6 sub-windows despite 150 Adam steps -- a good fit to 5 sparse
observed timesteps at `R_var=0.5` should leave `Jo` around 60-100. `u_norm~31`
throughout (not near zero) rules out "controls stay at zero, safety net
degenerates to background forecast" as the cause; `sigma` (background std fed
forward each window) jumped from 2.27 to a stable 10-14 range after window 0,
a real but bounded degradation. This is consistent with Adam's gradient signal
through ~500 sequential chaotic RK4 steps being dominated by Lyapunov-amplified
noise, exactly the regime `Strong4DVar` already needed LBFGS+line-search (not
Adam) to handle well with just 10 iterations.
**Fix validated (not yet a default change):** Re-ran the identical setup with
`optimizer="lbfgs", max_iter=10, lr=0.2` (mode/whitening/reset-safety all
unchanged): single-window `Jo` collapsed to 77-195 (right in the expected
noise-floor range), `window_rmse~1.3-1.4`; full 3-S0-window run gave
`RMSE_mean=1.42/1.84/1.29`, overall mean 1.52 -- versus `Strong4DVar`'s own
~0.81-1.0 ceiling (x0-only, no free `q`) and versus this same class's
Adam-based ~24.4-25.7. Also ~6-10x faster per sub-window (25-44s vs ~7min)
since LBFGS's internal line search needs far fewer effective gradient
evaluations to make progress than 150 blind Adam steps.
**Correction (see the same-day follow-up fragment,
`2026-09-17-l96weak4dvar-seed-mismatch-resolved.md`):** the ~1.52 number
above (and the ~0.81-1.0 `Strong4DVar` "ceiling" it was compared against)
both turned out to be measured on the wrong, ad hoc test seed with only 3
windows -- not a real ~2x algorithmic gap. Do not use ~1.52 as a stable
`L96Weak4DVar` reference number; see that fragment for the corrected
(`seed=123`, 10-window) result of ~0.98, much closer to the `Strong4DVar`
target and consistent with normal window-to-window variance at that sample
size. The 53681 reframing below is therefore also unreliable and should not
be relied on.
**Not yet done:** the class's own `optimizer` default is still `"adam"`
(`test_defaults` in `tests/test_baselines_l96weak4dvar.py` still asserts this)
-- flipping it, or just always passing `optimizer="lbfgs"` at call sites, is
a decision to make explicitly rather than silently, since it also changes
runtime cost characteristics (LBFGS is cheaper here, but that may not
generalize to every window/config). No test or default changed in this
fragment; this is a diagnostic-only result from ad hoc scratch scripts
(not committed) confirming the direction to take before making that change.
**Verification:** Scratch diagnostic scripts only (not part of the test
suite); existing `pytest tests/test_baselines_l96weak4dvar.py -q` (14 passed)
still describes current (Adam-default) behavior unchanged.
