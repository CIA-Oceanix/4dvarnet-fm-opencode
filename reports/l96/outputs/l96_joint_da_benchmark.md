# L96 Joint State-Parameter DA Benchmark

**System:** Lorenz-96 two-scale (NO=8, J=4), observed subspace state_dim=24 (Obs30, obs_interval=100), 200 shared test windows.

**Models:** joint ensemble filters (Joint-EnKF / Joint-ETKF) estimate the 24D observed state **and** the 8 model parameters `F, c1, hx, eps, w1..w4` (fast weights per index; h fixed) via an augmented-state ensemble, benchmarked against their state-only vanilla counterparts on the same cached S0/S1 test set. On S1 (reduced 24D J=2 dynamics) the DA only carries `w1,w2`; `w3,w4` default to the reference prior `[1.0, 0.1]` and are marked with a `†` (not estimated).

---

## Benchmarked methods

| Method | Type | Describes state + 8 params? |
|---|---|---|
| ETKF | vanilla ETKF | no (state only) |
| Joint-ETKF | joint ETKF | yes |
| EnKF | vanilla EnKF | no (state only) |
| Joint-EnKF | joint EnKF | yes |

---

## State RMSE (per case)

Pooled RMSE over the observed subspace, grouped slow (8D) / obs_fast (16D) / mean (24D). Lower is better.

| Method | S0 slow | S0 obs_fast | S0 mean | S1 slow | S1 obs_fast | S1 mean |
|---|---|---|---|---|---|---|
| ETKF | 0.5756 | 1.0286 | 0.8776 | 1.2131 | 1.7246 | 1.5541 |
| Joint-ETKF | 0.2979 | 0.8011 | 0.6334 | 1.1678 | 1.6618 | 1.4971 |
| EnKF | 0.4873 | 1.0927 | 0.8909 | 1.2371 | 1.6393 | 1.5052 |
| Joint-EnKF | 0.3507 | 0.9141 | 0.7263 | 1.1916 | 1.5930 | 1.4592 |

*Best is the lowest per column; rendered from the comparator JSON.*

---

## Energy Score (ES, per case)

N=30 ensemble Energy Score on the observed subspace (subsampled to 24D). Lower is better.

| Method | S0 ES | S1 ES |
|---|---|---|
| ETKF | 0.4508 | 0.9988 |
| Joint-ETKF | 0.2977 | 0.9374 |
| EnKF | 0.4581 | 0.8940 |
| Joint-EnKF | 0.3709 | 0.8434 |

---

## Explained variance (EV, per case)

Pooled explained variance over the observed subspace, grouped slow (8D) / obs_fast (16D) / mean (24D). Higher is better.

| Method | S0 slow | S0 obs_fast | S0 mean | S1 slow | S1 obs_fast | S1 mean |
|---|---|---|---|---|---|---|
| ETKF | 0.9083 | 0.5880 | 0.6948 | 0.5821 | -0.1161 | 0.1166 |
| Joint-ETKF | 0.9744 | 0.7448 | 0.8213 | 0.6123 | -0.0333 | 0.1819 |
| EnKF | 0.9343 | 0.5502 | 0.6782 | 0.5660 | -0.0093 | 0.1824 |
| Joint-EnKF | 0.9656 | 0.6785 | 0.7742 | 0.5972 | 0.0467 | 0.2302 |

---

## Parameter RMSE — S0

Per-parameter RMSE (`F, c1, hx, eps, w1..w4`) against the per-window true params. `†` marks fast weights defaulted to the reference prior (not estimated) on the J=2 S1 dynamics.

| Method | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| Joint-ETKF | 0.1306 | 0.0167 | 0.0156 | 0.0016 | 0.1155 | 0.1218 | 0.0119 | 0.0112 | 0.0531 |
| Joint-EnKF | 0.1532 | 0.0180 | 0.0168 | 0.0019 | 0.1157 | 0.1245 | 0.0120 | 0.0113 | 0.0567 |

---

## Parameter RMSE — S1

Per-parameter RMSE (`F, c1, hx, eps, w1..w4`) against the per-window true params. `†` marks fast weights defaulted to the reference prior (not estimated) on the J=2 S1 dynamics.

| Method | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| Joint-ETKF | 0.6082 | 0.1052 | 0.0637 | 0.0106 | 0.1161† | 0.1186† | 0.0000† | 0.0000† | 0.1278 |
| Joint-EnKF | 0.7637 | 0.1053 | 0.0640 | 0.0112 | 0.1194† | 0.1197† | 0.0000† | 0.0000† | 0.1479 |

---

## Context: L9 joint neural baseline (ens30 × 10)

Single-sample L9 `JointCFM` multi-tau state RMSE and param-RMSE mean, for reference against the DA rows. (Full neural tables live in 
`l96_joint_neural_benchmark.md`.)

| Case | L9 state RMSE | L9 param-RMSE mean | Best DA state RMSE |
|---|---|---|---|
| S0 | 0.6257 | 0.0591 | 0.6334 (Joint-ETKF) |
| S1 | 0.6313 | 0.0615 | 1.4592 (Joint-EnKF) |

*Best DA state RMSE is the minimum across the benchmarked methods for that case.*

---

## Consistency check

The comparator loads the cached `l96_datasets_obsj2_int100_nwin200.pt` and runs `evaluate_baseline` (the same code path as the vanilla DA caches). State RMSE/EV/ES are pooled over all 200 windows and subsampled to `obs_var_indices` (24D); param RMSE compares each joint filter's 8-wide estimate against `true_*` (padded to 8 with the reference prior on S1).
