## 2026-10-05: S1 error budgets for the S1-tuned ETKF and EnKS (QG gyrostat)

**Summary:** This adds two Shapley attributions of the S0 → realistic-S1 loss
(5 error groups, 32 on/off combinations, 20 val windows each), alongside the
existing one for the S0-tuned ETKF:
- the ETKF fixed at R × 6;
- the EnKS fixed at R × 6 with the 8-day taper.

Together they separate the effect of the S1 tuning from the effect of
smoothing:
- **R × 6 is free in the perfect-model corner** (ETKF 0.768 against 0.764;
  EnKS 0.809 against 0.806).
- **The tuning cuts the S1 loss by a third** (0.184 → 0.127). Most of the
  gain is on observation error (−0.025) and rd (−0.015).
- **Smoothing raises the cost of wind-forcing error** (+0.010; +0.030 on
  ψ₂).
- **The S1-tuned budgets are sub-additive** (interaction about −0.03).
- **What limits the best baseline, the S1-tuned EnKS:** observation error on
  q₁, forcing on ψ₂, and drag and resolution on q₂.

**Files modified:**
- `evaluation/qg_specwind_s1_sweep.py`:
  - `--method enks`, `--r-scale` and `--enks-taper-days`;
  - R-scaled runs go under `<out-dir>/<method>_R<scale>`;
  - Shapley output now includes the per-window values and an interaction
    interval.
- `batch/run_qg_specwind_s1.sbatch`: the `METHOD` / `RSCALE` variables.
- `reports/qg/generate_qg_specwind_da_report.py`: §4 gains a three-budget
  comparison with paired contrasts, and the S1-tuned EnKS per-field shares.
  The report is regenerated.
- `docs/results/qg_specwind_da_s1_calibration.md`: new §3.1d, plus updates
  to the status, numbers and caveats.

**Rationale:** The user asked how each S1 error term degrades the S1-tuned
EnKS, then for the S1-tuned ETKF so the comparison is clean.

**Verification:**
- SLURM 58483 (EnKS) and 58529 (ETKF): all 32 tasks completed for each.
  The full-S1 corners reproduce the S1-tuning sweep (EnKS 0.679, ETKF 0.641).
- `pytest -m "not slow"` on the docs, report and specwind S1 tests.
- `ruff check` on the touched files.
