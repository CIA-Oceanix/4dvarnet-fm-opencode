## 2026-09-11: QG DA hyperparameter sensitivity report (ETKF/EnKF)

**Summary:** New dedicated report consolidating ETKF/EnKF hyperparameter
sensitivity (multiplicative inflation, additive inflation, ridge
regularization) on both the S0 reference case and the revised S1
combined-error case. Resolves the open question of why EnKF beats ETKF on
S1: not inflation/ensemble-collapse (both methods share that fragility
equally), but an under-regularized `etkf_ridge` default on ETKF's
transform-matrix inversion. Confirmed at full N=100 scale (not just the
N=10 sweep): `etkf_ridge=1.0` raises ETKF's S1 q EV from 0.307 to 0.357 and
S0 psi EV from 0.921 to 0.957, beating EnKF's own N=100 numbers outright on
S1 (both fields) and tying/beating it on S0.

**Files modified:**
- `reports/qg/generate_da_sensitivity_report.py` (new) — JSON-only report
  generator, project bold/italic ranking convention.
- `reports/qg/outputs/da_sensitivity_s0_s1_report.md` (new) — rendered report.
- `reports/qg/outputs/qg_repro_validation_s1/etkf_n10_{additive,inflation}*.json`
  (new, N=10) — finer inflation grid + additive-inflation sweep, S1.
- `reports/qg/outputs/qg_da_sensitivity_sweep/*.json` (new, N=10, 34 files) —
  consolidated S0 inflation/additive, S1 ridge (incl. extended 2.0/5.0
  points), S0 ridge, and EnKF (S0+S1) inflation sweeps.
- `reports/qg/outputs/qg_repro_validation{,_s1}/etkf_ridge1.json` (new,
  N=100) — full-scale confirmation of the `etkf_ridge=1.0` finding, kept
  distinctly-tagged (not overwriting the canonical `etkf.json` reference
  numbers pending a decision on promoting it to the default ETKF config).
- `PLAN.md` — three new sections documenting the full sensitivity trail,
  including the N=100 confirmation.

**Not yet decided:** whether to promote `etkf_ridge=1.0` to the default
ETKF config in the main benchmark table and `run_qg_baselines.py`/
`sweep_qg_baselines.py`'s CLI default — separate follow-up decision.

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
