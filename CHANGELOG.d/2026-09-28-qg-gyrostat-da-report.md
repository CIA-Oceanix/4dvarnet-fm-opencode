## 2026-09-28: QG gyrostat DA report in the legacy QG DA report layout, with figures

**Summary:** `reports/qg/outputs/qg_specwind_da_report.md` is restructured
after the legacy QG DA report (`qg_da_report.md`) and becomes the Headline
report for the QG gyrostat case. It has:
- the case study, with the gyrostat-forced animation;
- per-scenario DA tables (S0, realistic S1, forced and coupled), ranked
  among the DA methods with the free forecast as a reference row;
- a configuration synthesis and the S1 error budget (Shapley, val);
- observation density, forced vs coupled, and factor strata sections;
- best/median/worst reconstructions (truth / free / ETKF / EnKF × ψ₁ ψ₂ q₁ q₂)
  with ETKF DA-cycle GIFs, in S0 and S1.

**Files modified:**
- `reports/qg/generate_qg_specwind_da_report.py`: new layout; runs are keyed
  by scenario label, so realistic base and high are not confused.
- `reports/qg/generate_qg_specwind_da_figs.py` (new): re-runs ETKF and EnKF on
  the 3 example windows for the panels and GIFs (the S1 GIF shows the true
  and the DA-model wind curl); summary figures (density, S0 vs S1, S1
  budget).
- `reports/qg/outputs/figs/qg_specwind_da_*` (new figures, GIFs about
  5–6 MB each).
- `reports/README.md`: Headline row and index row.
- `docs/results/qg_specwind_da_s0_s1_test.md`: generator reference; a caveat
  on the localized ETKF anomaly update.

**Rationale:** The user asked for one reference report for the case study,
modelled on the existing QG report with its figures and animations.
#291 (ETKF square-root null space) touches only the unlocalized ETKF
branch; every QG gyrostat run is localized, so the numbers are unchanged.

**Verification:** `pytest -m "not slow" tests/test_docs_layout.py tests/test_report*.py
tests/test_qg_specwind_s1.py` passes (121); `ruff check` on both generators is clean.
