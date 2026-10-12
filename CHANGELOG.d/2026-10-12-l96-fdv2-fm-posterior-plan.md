## 2026-10-12: L96 FDV2 × score-based flow matching posterior plan

**Summary:** New analysis plan `docs/plans/analysis/l96_fdv2_fm_posterior.md`: a calibrated FDV2 + flow-matching posterior at least as accurate (RMSE and CRPS, one configuration) as the FDV2→SDA hybrid, with a validation-selected protocol, baselines including post-hoc inflation, and multi-agent phases with gates. Supersedes `docs/plans/analysis/l96_fdv2_score_prior.md` before implementation.
**Files modified:** `docs/plans/analysis/l96_fdv2_fm_posterior.md` — new plan; `docs/plans/analysis/l96_fdv2_score_prior.md` — Status SUPERSEDED; `docs/README.md` — index rows.
**Rationale:** Review of the score-prior plan found the fixed σ₀ below the iterate error, a ~100× channel-scale change, a missed training-free test (SDA1-M is already a DSM-equivalent prior), R-unit mismatch for DPS/ΠGDM, and test-set selection of the reference numbers.
**Verification:** `pytest tests/test_docs_layout.py` — passed.
