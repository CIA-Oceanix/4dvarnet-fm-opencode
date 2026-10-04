## 2026-10-04: Relaxation inflation (RTPS / RTPP) for the ETKF and EnKF; QG gyrostat val sweep (negative result)

**Summary:** This adds relaxation inflation (`relax="rtps"|"rtpp"`,
`relax_alpha`) to the ETKF and EnKF, applied after each analysis. It then
tests the option on val (44 configurations, 20 windows) as an alternative to
the S1-tuned R × 6.
- **It is stable and calibrates the spread** (spread/RMSE ≥ 1).
- **In S1 it does not replace R × 6:**
  - the best is ETKF + RTPP 0.9 at 0.609 and EnKF + RTPS 0.9 at 0.606,
    against 0.641 / 0.636 for R × 6;
  - on top of R × 6 it adds at most +0.004.
- **In S0** RTPS 0.5 gains +0.004 / +0.005.
- **The benchmark is unchanged.**

**Files modified:**
- `evaluation/baselines.py`:
  - `_relax_anomalies` and `RELAX_MODES`;
  - `relax` / `relax_alpha` on `ETKF` (`assimilate`, `assimilate_batch`)
    and `EnKF.assimilate`;
  - `EnKS`, `ETKS` and `EnKF.assimilate_batch` refuse it.
- `evaluation/run_qg_baselines.py`: `run(..., relax, relax_alpha)`.
- `evaluation/run_qg_specwind_da.py`: the `--relax` / `--relax-alpha`
  options, also recorded in the meta file.
- `batch/run_qg_specwind_da.sbatch`: the `RELAX` / `ALPHA` variables and
  their output-dir tag.
- `evaluation/qg_specwind_relax_tuning.py` and
  `batch/run_qg_specwind_relax_tuning.sbatch` (new): the val sweep.
- `tests/test_relax_inflation.py` (new) and `tests/test_qg_specwind_da.py`:
  the helper's properties, end-to-end runs, and a check that the spread
  widens.
- `reports/qg/generate_qg_specwind_da_report.py`: an S1-tuning paragraph and
  a caveat. The report is regenerated.
- `docs/results/qg_specwind_relax_inflation.md` (new);
  `docs/results/qg_specwind_s1_tuning.md` finding 2; `reports/README.md`.

**Rationale:** Multiplicative inflation diverges under S1, and RTPS/RTPP are
the standard alternative a reviewer would expect. The user asked for the
sweep to close this gap in the DA baselines.

**Verification:**
- SLURM 58432: all 44 tasks completed. Its baselines reproduce the S1-tuning
  sweep to within 0.001.
- `pytest -m "not slow"` on the relaxation, specwind DA, ETKS, QG baselines
  and docs/report tests.
- `ruff check` on the touched files.
