## 2026-09-13: fold Q3/Q4/Q3-noise0.05/Q5 + S1 into the QG neural overview report

**Summary:** `qg_neural_report.md` was S0-only and covered only Q1 (flagged
as not apples-to-apples). Rewrote its generator to add all 4 remaining
neural schemes (Q3, Q4, Q3-noise0.05, Q5) and an S1 column, all evaluated
at the DA baselines' own reference config (lag=5.0d/noise_frac=0.05/
s1_param_bias=s1_amp_bias=0.1, N=100). DA rows (already updated on master
with the `etkf_ridge=1.0` fix) are unchanged; verified those numbers match
the ones this branch had already independently computed before merging.

**Files modified:**
- `reports/qg/generate_qg_neural_report.py` -- `load_da_summary()` now takes
  a scenario dir/key so it can read both `qg_repro_validation/` (S0) and
  `qg_repro_validation_s1/` (S1); replaced `load_q1_summary()` with
  `load_neural_summary()` reading the cross-scenario JSON
  (`qg_neural_s0_s1_cross_scenario/results_lag5_noise0.05_bias0.1.json`);
  added Q3/Q4/Q3-noise0.05/Q5 scheme descriptions; the "not apples-to-apples"
  footnote now applies per-row (Q1/Q3/Q4 -- trained at the easier
  lag=1.0/noise=0.01 default, only re-evaluated here) instead of blanket,
  since Q3-noise0.05/Q5 were actually trained at this exact config.
- `reports/qg/outputs/qg_neural_report.md` -- regenerated.

**Rationale:** Q5 substantially closes the PV-q gap the earlier Q1-only
table couldn't show (S1 q EV 0.385, best of all 9 rows including every DA
method) -- this is the headline result of the Q3/Q4/Q5 work and needed to
be in the actual published report, not just PLAN.md.

**Verification:** `pytest tests/test_qg_neural.py -m "not slow"` (fdv env)
-- 34 passed, 1 skipped (pre-existing, unrelated). `ruff check
reports/qg/generate_qg_neural_report.py` clean. Report regenerated and
diffed by hand against the values already confirmed in this session.
