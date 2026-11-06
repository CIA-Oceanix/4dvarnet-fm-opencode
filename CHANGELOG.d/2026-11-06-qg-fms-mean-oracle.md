## 2026-11-06: QG flow-matching scores — mean-only lower bound

**Summary:** `field_fms` (shared by the QG DA and neural scorers) also stores `fm_<f>_orc_<tau>`, the Gaussian score with the variance set to the realised squared error of the mean, mean(e^2 / (1 + alpha_tau e^2)).
**Files modified:** `evaluation/run_qg_specwind_da.py` — new term and docstring; `tests/test_qg_specwind_da.py` — it equals the MSE at tau = 0, decreases with tau and lies below the best scalar-calibrated score.
**Rationale:** For tau > 0 the flow-matching score mixes the error of the mean with the characterisation of the distribution around it. The best Gaussian with a given mean (variance = conditional MSE) is not computable; its score is bracketed by this term (below) and the scalar-calibrated score (above), which separates the mean part from the spread and shape parts (P1 paper, section 5.3).
**Verification:** `pytest -m "not slow" tests/test_qg_specwind_da.py -k "field_fms or fm_scores"` (2 passed); `ruff check` on the touched files.
