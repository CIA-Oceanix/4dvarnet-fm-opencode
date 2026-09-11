## 2026-09-11: QG DA hyperparameter sensitivity report (ETKF/EnKF)

**Summary:** New dedicated report consolidating ETKF/EnKF hyperparameter
sensitivity (multiplicative inflation, additive inflation, ridge
regularization) on both the S0 reference case and the revised S1
combined-error case. Resolves the open question of why EnKF beats ETKF on
S1: not inflation/ensemble-collapse (both methods share that fragility
equally), but an under-regularized `etkf_ridge` default on ETKF's
transform-matrix inversion — `etkf_ridge=1.0` raises ETKF's S1 q EV from
0.253 to 0.332 at N=10, exceeding EnKF's own baseline.

**Files modified:**
- `reports/qg/generate_da_sensitivity_report.py` (new) — JSON-only report
  generator, project bold/italic ranking convention.
- `reports/qg/outputs/da_sensitivity_s0_s1_report.md` (new) — rendered report.
- `reports/qg/outputs/qg_repro_validation_s1/etkf_n10_{additive,inflation}*.json`
  (new, N=10) — finer inflation grid + additive-inflation sweep, S1.
- `reports/qg/outputs/qg_da_sensitivity_sweep/*.json` (new, N=10, 28 files) —
  consolidated S0 inflation/additive, S1 ridge, and EnKF (S0+S1) inflation
  sweeps.
- `PLAN.md` — two new sections documenting the full sensitivity trail.

**Rationale:** `PLAN.md`'s 2026-09-10 "revised S1" campaign found EnKF
consistently beats ETKF on the model-error case with no explanation. A first
inflation sweep found any inflation above 1.0 catastrophic for ETKF, leaving
the gap unexplained. This report finishes that investigation: extends the
sweep to S0, to EnKF's own hyperparameters (never swept before — every prior
EnKF result used the fixed `inflation=1.0` default), and to `etkf_ridge`
(also never exercised), and finds the actual explanation.

**Verification:** `ruff check reports/qg/generate_da_sensitivity_report.py`
passes; `python reports/qg/generate_da_sensitivity_report.py` runs and
renders the report correctly against the committed JSON data.
