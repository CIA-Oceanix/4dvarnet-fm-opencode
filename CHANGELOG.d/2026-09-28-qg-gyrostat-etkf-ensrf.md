## 2026-09-28: QG gyrostat benchmark switches its ETKF to the exact localized EnSRF; test rows and S1 attribution re-run

**Summary:** The QG gyrostat benchmark's ETKF now uses `loc_mode="ensrf"`
with radius 8 and ridge 0.1. The library default of `ETKF` is unchanged.
Re-run with it:
- all ETKF test rows: S0 and S1 on both datasets, bridge density, density
  curve;
- the realistic-base S1 Shapley attribution (val);
- the report's example figures.

**Test results:**
- **S0 score:** 0.774 forced, 0.779 coupled (old ETKF 0.766 / 0.773). The
  ETKF now beats the EnKF by +0.013 [+0.011, +0.015].
- **S1 score:** 0.579 forced (old 0.559), with q₂ +0.060 [+0.027, +0.098].
  The gap to the EnKF shrinks from −0.037 to −0.018.

**New S1 error budget (base):** obs 31%, res 25%, rd 18%, drag 16%, forcing
9% of the score loss, with interactions about 0. With the legacy ETKF it
was obs 43% with +0.042 interaction: the legacy filter's over-confidence
made the errors compound.

**Files modified:**
- `evaluation/run_qg_specwind_da.py` and `batch/run_qg_specwind_da.sbatch`:
  ETKF defaults EnSRF / ridge 0.1; `LOCMODE`; ETKF output dirs tagged with
  the mode.
- `evaluation/qg_specwind_s1_sweep.py` and `batch/run_qg_specwind_s1.sbatch`:
  EnSRF ETKF; output `experiments/qg_specwind_s1_ensrf`.
- `reports/qg/generate_qg_specwind_da_report.py`: selects the EnSRF ETKF
  runs; filter statement.
- `reports/qg/generate_qg_specwind_da_figs.py`.
- `reports/qg/outputs/qg_specwind_da_report.md` and
  `reports/qg/outputs/figs/qg_specwind_da_*`.
- `reports/README.md`: localized-ETKF caveat for the legacy QG reports.
- `docs/results/qg_specwind_da_s0_s1_test.md`,
  `docs/results/qg_specwind_da_s1_calibration.md`,
  `docs/results/qg_specwind_da2_val_tuning.md`,
  `docs/results/qg_specwind_etkf_loc_update.md`,
  `docs/plans/analysis/qg_specwind_da_s0.md`,
  `docs/plans/analysis/qg_specwind_da_s1.md`.

**Rationale:** #294 showed that the legacy localized ETKF divides R by N − 1
and over-contracts the spread; the user asked to switch and to re-run the
attribution. The legacy QG reports stay as they are, with the caveat in
`reports/README.md`.

**Verification:**
- SLURM 56545 (attribution, 32 tasks, rtx8000) and 56548–56556 (test,
  90 tasks, a100): all completed.
- `pytest -m "not slow"` on the specwind DA, S1, EnSRF and docs/report
  tests; `ruff check` on the touched files.
