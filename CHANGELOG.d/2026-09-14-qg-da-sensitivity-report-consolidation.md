## 2026-09-14: Consolidate 4DVar + obs-density sensitivity findings into the DA sensitivity report

**Summary:** Extends `reports/qg/generate_da_sensitivity_report.py`
(previously ETKF inflation/additive/ridge + EnKF inflation only) with two
new sections: the Strong/Weak-4DVar `b_var_scale`/`q_var_scale`
covariance-scale sweep (negative result, N=5), and the ETKF/EnKF
obs-density/configuration sensitivity study (`cols_per_day`, new psi2
lower-layer point observations, `cols_sampling="random"`) including its
full "initial sweep -> correction" narrative -- both original headline
findings ("more upper-layer density destabilizes S1", "psi2 observations
are uniquely valuable") were retracted after user pushback revealed a
`loc_radius`/ensemble-conditioning mismatch, not a real physical effect.
Also adds a short matching paragraph to `generate_qg_da_report.py`'s
existing condensed "Hyperparameter sensitivity analysis" section, which
already summarized the ETKF-ridge and 4DVar findings but predated (and so
omitted) the obs-density study.

**Files modified:**
- `reports/qg/generate_da_sensitivity_report.py` -- two new sections
  ("4DVar covariance-scale sensitivity", "ETKF/EnKF obs-density
  sensitivity"), a `render_metric_table` helper for the new multi-column
  tables, `metrics()` extended with `q_ev_layer2`, intro/Synthesis/closing
  data-pointer updated to reflect the expanded scope; also corrected a
  stale "not yet decided" note about promoting `etkf_ridge=1.0` to the
  default (that promotion already happened 2026-09-12).
- `reports/qg/generate_qg_da_report.py` -- one new condensed paragraph in
  the "Hyperparameter sensitivity analysis" section for the obs-density
  study, pointing to the dedicated report's full correction trail.
- `reports/qg/outputs/da_sensitivity_s0_s1_report.md`,
  `reports/qg/outputs/qg_da_report.md` -- regenerated.
- `PLAN.md` -- consolidation note.

**Rationale:** The 4DVar negative result and the obs-density
sensitivity-with-correction findings were previously only documented as
PLAN.md prose scattered across three separate sweep output directories
(`qg_da_sensitivity_sweep/`, `qg_4dvar_sensitivity_sweep/`,
`qg_obs_density_sweep/`), with no single generated report covering all
three studies together.

**Verification:** `ruff check` passes on both touched report generators;
both regenerated successfully and manually reviewed against PLAN.md's
underlying numbers (all tables computed directly from the committed JSON
sweep outputs, not transcribed); the CI-gated test list (`pytest ... -m
"not slow"`, 450 passed/1 skipped) is unaffected, as expected for a
report-generator-only change with no QG/neural production code touched.
