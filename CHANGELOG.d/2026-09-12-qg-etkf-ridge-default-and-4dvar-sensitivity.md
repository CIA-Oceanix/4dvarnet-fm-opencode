## 2026-09-12: Promote etkf_ridge=1.0 to the ETKF default; 4DVar sensitivity (negative result); DA report extended to S0+S1 with sensitivity/synthesis sections

**Summary:** Promotes `etkf_ridge=1.0` (found in the 2026-09-11 sensitivity
study to close the previously-unexplained EnKF>ETKF gap on S1, confirmed at
N=100) to the actual default in `run_qg_baselines.py`/`sweep_qg_baselines.py`.
Also ran a matching covariance-weighting sensitivity sweep for Strong/Weak-
4DVar (`b_var_scale`/`q_var_scale`) -- a clean negative result, no config
change. The main DA baselines report is significantly extended: covers both
S0 and S1 (was S0-only), gains a "Hyperparameter sensitivity analysis"
section and a closing "Synthesis: best configuration per method" table.

**Files modified:**
- `evaluation/run_qg_baselines.py` — `run()`'s `etkf_ridge` default 0.0→1.0;
  new `--etkf-ridge`/`--etkf-additive` CLI flags.
- `evaluation/sweep_qg_baselines.py` — `--etkf-ridge-list` fallback [0.0]→[1.0].
- `reports/qg/outputs/qg_repro_validation{,_s1}/etkf.json` — now the
  ridge=1.0 N=100 results (old default archived as `etkf_ridge_default.json`
  in both dirs, not deleted).
- `reports/qg/generate_qg_da_report.py` — extended to cover S0+S1, added
  sensitivity-analysis and synthesis sections.
- `reports/qg/generate_qg_neural_report.py` / `reports/qg/outputs/
  qg_neural_report.md` — ETKF description/numbers updated to match.
- `reports/qg/outputs/qg_4dvar_sensitivity_sweep/*.json` (new, N=5, 7
  files) — Strong/Weak-4DVar covariance-weighting sweep, negative result.
- `PLAN.md` — two new sections (default promotion + 4DVar negative result).

**Rationale:** `evaluation/baselines.py`'s shared `ETKF` class default is
deliberately left at 0.0 since it's also used by L96 (unvalidated there);
only QG's own driver-level default changed. See PLAN.md's "ETKF/EnKF
sensitivity" sections for the full sensitivity trail this promotion is
based on.

**Verification:** `ruff check` passes on all touched files;
`python reports/qg/generate_qg_da_report.py` and
`generate_qg_neural_report.py` run and render correctly against the
updated JSON data; full QG test suite (134 tests, `-m "not slow"`) passes
unchanged.
