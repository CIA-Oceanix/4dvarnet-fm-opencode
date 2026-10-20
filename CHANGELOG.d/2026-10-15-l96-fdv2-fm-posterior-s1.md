## 2026-10-15: L96 FDV2 x FM posterior — S1 check of M1bI

**Summary:** Added section 4 to the FDV2 × FM posterior results note: the frozen S0 configurations (hybrid τ0 0.5/0.1, B5, both M1bI configs) scored on the S1 (model-error) test windows, with inflation re-tuned on S1 validation. M1bI keeps its 4–7% CRPS lead over the hybrid and B5 and SSR ≈ 1, RMSE passes in 3 of 4 cells, but rank-histogram reliability fails on the random layout.
**Files modified:** `docs/results/l96_fdv2_fm_posterior.md` — S1 section, status, answer and reading updated; `docs/plans/analysis/l96_fdv2_fm_posterior.md` — status line mentions the S1 check.
**Rationale:** The plan's S-robust criterion asks for the S1 ordering (reported, not gated); S1 was listed as open when the note was written.
**Verification:** `pytest tests/test_docs_layout.py`; the scoring script reproduces the note's S0 numbers exactly before being applied to S1.
