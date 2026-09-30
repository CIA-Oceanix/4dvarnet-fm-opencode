## 2026-09-29: Localized EnKS (smoother) on the EnSRF ETKF; QG gyrostat smoother rows

**Summary:** A localized ensemble Kalman smoother, the QG counterpart of the
L96 ETKS, which needs the unlocalized ETKF. The localized EnSRF analysis
records HA, S⁻¹dy, its modified-gain factor and its localization.
`_enks_smooth_localized` applies each analysis to the earlier stored states
within a lag, using the state–observation localization of that analysis.
Without localization it is the exact RTS smoother.
- **Lag:** tuned on val to 12 analyses (4 days): S0 +0.034, S1 +0.028 over
  the filter. Longer lags hurt under model error (whole window below the
  filter in S1).
- **Test, 100 windows:**
  - S0: 0.804 / 0.813 (forced / coupled), against 0.774 / 0.779 for the
    ETKF;
  - S1: 0.602 / 0.610, against ETKF 0.579 / 0.586 and EnKF 0.596 / 0.604;
  - the gain is mostly in q₁ (+0.07).

**Files modified:**
- `evaluation/baselines.py`: `record` in `_ensrf_localized_analysis`;
  `_enks_smooth_localized`; the `EnKS` class.
- `evaluation/run_qg_baselines.py`: `method_name="enks"`, `enks_lag`; the
  column localization is now built for `enks` too.
- `evaluation/run_qg_specwind_da.py`: `--method enks`, `--enks-lag`
  (default 12).
- `batch/run_qg_specwind_da.sbatch`: `ENKSLAG`.
- `evaluation/qg_specwind_etkf_check.py`: lag tasks 16–25.
- `reports/qg/generate_qg_specwind_da_report.py`: EnKS rows.
- `reports/qg/generate_qg_specwind_da_figs.py`: EnKS reconstruction row;
  the S0/S1 figure has one panel per method.
- regenerated report and figures.
- `tests/test_enks_localized.py` (3 tests, including an exact RTS check) and
  an end-to-end test in `tests/test_qg_specwind_da.py`.
- `docs/results/qg_specwind_enks.md` (new);
  `docs/results/qg_specwind_da_s0_s1_test.md` §3b; `reports/README.md`.

**Rationale:** The user asked for a localized EnKS for QG after the review of
the L96 ETKS. Ocean reanalyses are smoothers, so this row is the
reanalysis-type estimate for the case study.

**Verification:**
- SLURM 56925 (val lag sweep) and 56975–56978 (test): all completed.
- `pytest -m "not slow"` on the EnKS, EnSRF, specwind DA and S1 tests and
  the docs/report tests; `ruff check` on the touched files.
