## 2026-09-26: G3 — two-way coupled QG ocean + gyrostat atmosphere (Option C)

**Summary:** Adds `models/qg_coupled.py`.

`CoupledSpectralQG` advances the ocean (`BatchedQGDynamics`, with eddy drag
`r_cf`) and the gyrostat atmosphere in one RK4 step, with the coupling of
decision 4:
- **atmosphere → ocean:** the spectral PV source from the gyrostat modes,
  held over the step as in the one-way path;
- **ocean → atmosphere:** `kappa_fb * gamma_a * r_cf * zeta_{o,k}` on each
  wind amplitude. The ocean's upper-layer vorticity is read from the six
  Fourier coefficients of psi_1 and rotated into the gyrostat's co-moving
  frame. Calm windows get no feedback.

`generate_coupled_windows` produces G2-format windows with a forced,
coupled spin-up. It is kept separate from `data/qg_datasets.py` until G5
merges.

**Files modified:** `models/qg_coupled.py` and `tests/test_qg_coupled.py`
(8 tests) — new; `docs/plans/tech/qg_batched_generation_datasets.md`
(§3.4 note).
**Rationale:** Decision 4 keeps the co-stepped path for a labelled
feedback-gain sweep, `kappa_fb` in {1 (physical), 10, 100}. Design points
and findings:
- **The gyrostat half of the step uses the same arithmetic as
  `BatchedGyrostat`,** so `kappa_fb = 0` reproduces the one-way Option B
  path bit for bit (tested). A reordering would be amplified by the chaos.
- **Measured at batch 256 (RTX 8000):** 0.046 ms per window-step, 1.63×
  one-way.
- **Feedback relative to the gyrostat's own tendency:** 0.75% (median) at
  kappa_fb = 1, 7.5% at 10, 57% at 100. This confirms decision 4's
  estimate that physical mechanical coupling is weak.
- **After 60 days** the gyrostat state differs from kappa_fb = 0 by 4%,
  41% and 173%. The ocean's eddy field diverges even at kappa_fb = 1
  (chaos), so forced response must be measured with ensembles (the plan's
  WP2), not single runs.

**Verification:**
- `pytest tests/test_qg_coupled.py`: 8 passed. It covers the Fourier-read
  vorticity against the grid projection, kappa_fb = 0 bit-exact against
  one-way, the feedback formula, calm windows, linearity in kappa_fb, the
  step limit and the G2-format generator.
- `ruff check`: clean.
