## 2026-09-26: Plan — gyrostat spectral wind forcing for the QG case study (Option B)

**Summary:** Adds `docs/plans/analysis/qg_gyrostat_wind_forcing.md` (DRAFT
v2), the implementation and response-analysis plan for **Option B** of
`docs/plans/case_study/qg_gyrostat_synoptic_forcing.md`. Option A is skipped.
The plan covers:
- a `FourierWindBasis` (`models/qg_wind_modes.py`) that writes mode
  amplitudes straight into the spectral PV source;
- a gyrostat driver (`models/gyrostat_driver.py`, sparse-chain `chain6`
  preset);
- drivers `spectral_ou`, `gyrostat` and `gyrostat_surrogate` behind
  `wind_driver`, with the default OU storm path and cache keys bit-identical;
- S1 corruption and the 1L/psi DA models generalized to the wider wind
  state;
- long forced runs E0–E4 with ensembles, ending in a go/no-go on Option C.

**Files modified:** `docs/plans/analysis/qg_gyrostat_wind_forcing.md` — new.
`docs/README.md` — index row.
**Rationale:** The user chose to go straight to Option B, which is also the
prerequisite for two-way coupling (Option C). Three measurements and code
findings shape the plan:
- **Today's storm is effectively domain-scale.** Measured with
  `wind_curl_field`, it keeps 99.8% of its curl variance in the 12 Fourier
  amplitudes with `|k| <= 2`, and its spectrum peaks at domain wavenumber 1.
  So the basis contains today's forcing, and every driver can share its
  spatial spectrum, differing only in temporal and cross-mode structure.
- **The truth-cache key hashes the whole `QGConfig`**, so the new fields are
  kept out of the key for the default driver.
- **The benchmark truth is forced for only 40 days after an unforced
  spinup**, hence separate long runs.
**Verification:** Docs only. `pytest tests/test_docs_layout.py` passes.
