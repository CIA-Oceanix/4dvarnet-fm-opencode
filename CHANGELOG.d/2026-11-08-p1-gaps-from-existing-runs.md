## 2026-11-08: P1 gaps filled from existing runs

**Summary:** Fills the P1 table cells and `\needsrun` items that the stored L96 outputs already cover: ETKS-100 FMSS/SS (Table 5), L96 EnKF and D-Unet mean shares (Table 4), the filter/smoother identity along tau, and the S1-mean/S0-spread counterfactual. Also removes a stale CRPS flag.
**Files modified:**
- `reports/p1/p1_figures.py`: new `extras` step, writing `reports/p1/outputs/l96_extras.json`. The D-Unet tau-series now averages scores over seeds instead of scoring the seed-averaged estimate.
- `reports/l96/p1_mean_bracket.py`: EnKF row, from the stored mean and variance on the probe's scale grid, and D-Unet row.
- `reports/p1/outputs/l96_tau_series.json`, `reports/p1/outputs/l96_extras.json` and `reports/l96/outputs/fm_score/p1_mean_bracket.json`: regenerated.
- `docs/papers/p1_structural_hypotheses/`: synced from Overleaf (Tables 1, 4 and 5, Sects. 4 and 5, `fig_fmss.pdf`).
**Rationale:** Every number in the paper must trace to committed files. The identity along tau applies Proposition B.4 to the context (S, x_tau). The old D-Unet line scored a 3-seed ensemble mean, which is better than any single seed (FMSS_0.75 0.551 vs 0.509).
**Verification:** On ETKF S1, the mean/variance path reproduces the committed probe summary to 0.2 % (FMS_0.75 and the scalar-calibrated bound). The ETKS-30 series reproduces Table 5 (0.41 / -1.56, SS 0.48 / 0.34). Ran `pytest tests/test_docs_layout.py tests/test_reports_index.py tests/test_report_inputs.py` and `ruff check` on the touched scripts.
