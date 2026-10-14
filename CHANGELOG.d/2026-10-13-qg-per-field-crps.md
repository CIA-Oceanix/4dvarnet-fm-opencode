## 2026-10-13: QG per-field CRPS (psi1, psi2, q1, q2)

**Summary:** The QG DA and neural scorers now store the CRPS of each field per window (`crps_psi1`, `crps_psi2`, `crps_q1`, `crps_q2`), next to the pooled q CRPS, so that a CRPS skill score can be formed per field like the explained variance.
**Files modified:** `evaluation/run_qg_baselines.py` — optional `ensemble_hook` called on each window's (N, T, D) ensemble; `evaluation/run_qg_specwind_da.py` — `field_crps` (members inverted to psi with the window's true parameters, as in `window_metrics`), wired into `evaluate`; `evaluation/run_qg_specwind_neural.py` — same fields in `score_window`; `tests/test_qg_specwind_da.py` — per-field CRPS agrees with the pooled q CRPS and reduces to the MAE for one member.
**Rationale:** The stored CRPS pooled q1 and q2 in physical units (q1 dominates, 6x the climatological spread of q2) and had no psi counterpart, so it could not be compared with the per-field EV score (paper P1, QG tables).
**Verification:** `pytest -m "not slow" tests/test_qg_specwind_da.py tests/test_qg_specwind_neural_eval.py` (34 passed); `ruff check` on the touched files.
