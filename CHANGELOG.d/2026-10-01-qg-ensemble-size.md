## 2026-10-01: Ensemble-size study for the QG gyrostat ETKF/EnKS; large-ensemble (N = 320) reference rows

**Summary:** This checks whether N = 80 limits the benchmark ETKF and EnKS.
- **Val convergence** (N = 40–320): rankings hold at every N, with the EnKS
  ahead of the ETKF by about +0.04.
  - S0 is not converged at N = 80: EnKS +0.022 at N = 320.
  - S1-tuned saturates by N = 160 (+0.003 from 160 to 320).
- **S0 re-tuned at N = 320:** radius 12 for both, taper 16 d for the EnKS.
  Wider localization helps (+0.007 for the EnKS).
- **Test, N = 320, forced / coupled:**
  - S0: ETKF 0.794 / 0.800, EnKS 0.841 / 0.847 (+0.02 / +0.03 over
    N = 80);
  - S1-tuned: ETKF 0.641 / 0.650, EnKS 0.675 / 0.683 (about +0.01).
- **N = 80 stays the benchmark.** The report gains a "large-ensemble
  reference (N = 320)" block for quoting converged skill.

**Files modified:**
- `evaluation/qg_specwind_enks_tuning.py`: ensemble-size and N = 320
  re-tune tasks (13–32).
- `batch/run_qg_specwind_da.sbatch`: output dirs tagged `_N<ens>` when
  N ≠ 80.
- `batch/run_qg_specwind_enks_tuning.sbatch`.
- `reports/qg/generate_qg_specwind_da_report.py`: ensemble size per run;
  benchmark tables restricted to N = 80; large-ensemble table. Regenerated
  report.
- `docs/results/qg_specwind_ensemble_size.md` (new);
  `docs/results/qg_specwind_da_s0_s1_test.md` §3d; `reports/README.md`.

**Rationale:** The user asked whether the EnKS is a good smoother baseline
and to confirm it with a larger ensemble.

**Verification:**
- SLURM 57613 and 57698 (val; one re-tune task hit a bus error at exit
  after writing its complete result), 57701–57704 and 57913–57916 (test):
  all completed.
- `pytest -m "not slow"` on the EnKS, specwind DA and docs/report tests;
  `ruff check` on the touched files.
