## 2026-09-28: ETKF square-root transform keeps the unobserved anomaly directions

**Summary:** The unlocalised `ETKF` built its symmetric square-root transform
from a thin SVD of the (N, od) obs anomalies, `T = U diag(sqrt(N1/d)) Uᵀ`. When
od < N, U has only od columns, so T zeroed the N − 1 − od anomaly directions
the observations do not constrain at every analysis (L96 benchmark: 24 obs vs
N = 30, analysis anomaly rank 24 instead of 29). The new
`_etkf_sqrt_transform` adds the complement `sqrt(N1/d_null) (I − UUᵀ)`, which
gives the exact `sqrt(N1) (N1 I + YYᵀ + ridge)^(-1/2)`. Both
`ETKF.assimilate` and `ETKF.assimilate_batch` use it; the mean update `w` is
unchanged (HAᵀ has no component on the complement).
**Files modified:**
- `evaluation/baselines.py` — `_etkf_sqrt_transform` helper; the two ETKF ensemble-space branches call it. When od ≥ N the thin SVD is square and the output is unchanged.
- `tests/test_etkf_sqrt_transform.py` (new) — the transform against the exact inverse square root (od < N, = N − 1, = N, > N; ridge 0 and 0.1), the analysis covariance against the Kalman (I − KH)Pᶠ, and full anomaly rank after one `ETKF.assimilate` analysis. 9 of the 15 fail on the old transform.
- `tests/test_da_golden_l96.py` — ETKF golden RMSE 0.725394 → 0.727326 (its config has od = 4 < N = 10, so it is affected by design).
**Rationale:** Prerequisite for the ETKS (`docs/plans/tech/l96_etks_smoother.md`
on `feature/l96-etks`): the smoother multiplies the stored per-cycle transforms,
and rank-deficient factors would give a rank-deficient smoothed ensemble and
break the linear-Gaussian RTS check.
**Impact on the L96 benchmark (small):** 20 test windows, regular grid,
`evaluate_all_l96.py --skip-weak --skip-strong`, per-case inflation defaults:
ETKF S0 RMSE 0.924 → 0.914, S1 1.467 → 1.467. The EnKF, whose code does not
change, moved by about the same amount between the two runs (S0 0.967 → 0.956,
unseeded perturbed obs), so the ETKF change is at that noise level. Forecasts
between analyses (100 steps) regrow the lost directions, and inflation covers the
rest. The published ETKF rows are not rerun here. The L63 ETKF (`evaluation/run.py`,
same class, od ≤ 3 vs N = 30) loses far more directions per cycle and may move
more; not measured here.
**Not changed:** the joint-state ETKFs (L63 `JointETKF` around
`baselines.py:2316/2412`, the L96 joint `_analysis` at `:2572/:2850`) have the
same thin-SVD transform, with od ≤ 3 vs N = 30 on L63. They are left for a
separate change, because fixing them moves the joint-estimation results.
**Verification:** `pytest tests/ -m "not slow"` (CI selection) passed locally;
`ruff check` clean on the touched files.
