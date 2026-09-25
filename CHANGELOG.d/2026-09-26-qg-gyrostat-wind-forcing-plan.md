## 2026-09-26: Plan — gyrostat wind forcing for the QG case study (WP1–WP2)

**Summary:** Adds `docs/plans/analysis/qg_gyrostat_wind_forcing.md` (DRAFT
v1), the implementation and response-analysis plan for Option A of
`docs/plans/case_study/qg_gyrostat_synoptic_forcing.md`:
- a Lorenz-63 gyrostat driver in a new `models/gyrostat_driver.py`, behind a
  `wind_driver` switch in `QGDynamics`;
- the same `(A, x_c, y_c)` storm state as today, so no consumer changes;
- a multivariate phase-randomized surrogate as the control;
- long forced runs E0–E3 with ensembles, whose diagnostics end in a go/no-go
  on Option B.

**Files modified:** `docs/plans/analysis/qg_gyrostat_wind_forcing.md` — new.
`docs/README.md` — index row.
**Rationale:** Turns the scoping note's first two work packages into an
executable plan grounded in the current code. Two findings shape it:
- **Cache keys:** the truth-cache key hashes the whole `QGConfig`, so naively
  adding driver fields would invalidate every cached truth. The plan keeps
  the default keys unchanged, and a test pins them.
- **Benchmark truth unsuitable for response analysis:** the ocean is spun up
  unforced and forced for only 40 days, so Q1–Q3 need separate long forced
  runs.

Lorenz-63 time scales were measured rather than assumed: `x` has an ACF
e-folding of 0.305 units, lobes last 1.72 units on average, and its kurtosis
is 2.31 (bimodal, not heavy-tailed). They set the time-unit and mapping
decisions.
**Verification:** Docs only. `pytest tests/test_docs_layout.py` passes.
