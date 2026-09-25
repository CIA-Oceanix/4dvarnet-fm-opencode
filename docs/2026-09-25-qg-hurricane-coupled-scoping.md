## 2026-09-25: Scoping note — coupled hurricane–ocean configuration for the QG case study

**Summary:** Adds `docs/plans/case_study/qg_hurricane_coupled.md` (QG-TC), an
"Option F" for the QG gyrostat note: a parametric vortex with an Emanuel-type
intensity equation, a slab mixed layer coupled to the QG interface, and the
gyrostat driver as the storm environment, coupled two-way through the SST
under the storm core. It gives a configuration ladder (H0 Geisler check to
H3 fully coupled), synthetic observation and model-error scenarios, a
discussion of transfer to real wind/ocean observations (reconstruction,
forecasting, calibration), an implementation plan, risks and open decisions.
**Files modified:** `docs/plans/case_study/qg_hurricane_coupled.md` — new.
`docs/README.md` — index row.
**Rationale:** Hurricanes are the sharpest case of local, two-way
ocean–atmosphere coupling that the two-layer QG ocean can partly represent
(the balanced part of Geisler 1970, eddy modulation of cooling). The setting
also bridges the synthetic case studies to real altimetry/SST/wind data.
The QG benchmark default is unchanged.
**Verification:** Docs only. `pytest tests/test_docs_layout.py` passes.
