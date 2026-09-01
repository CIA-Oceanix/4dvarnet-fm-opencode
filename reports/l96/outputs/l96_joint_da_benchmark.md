# L96 Joint State-Parameter DA Benchmark

**System:** Lorenz-96 two-scale (NO=8, J=4), observed subspace state_dim=24 (Obs30, obs_interval=100), 200 shared test windows.

**Models:** joint ensemble filters (Joint-EnKF / Joint-ETKF) estimate the 24D observed state **and** the 8 model parameters `F, c1, hx, eps, w1..w4` (fast weights per index; h fixed) via an augmented-state ensemble, benchmarked against their state-only vanilla counterparts on the same cached S0/S1 test set. On S1 (reduced 24D J=2 dynamics) the DA only carries `w1,w2`; `w3,w4` default to the reference prior `[1.0, 0.1]` and are marked with a `†` (not estimated).

---

## Benchmarked methods

| Method | Type | Describes state + 8 params? |
|---|---|---|
| Joint-EnKF | joint EnKF | yes |
| Joint-ETKF | joint ETKF | yes |
| Joint-Strong-4DVar | joint Strong-4DVar | yes |
| Strong-4DVar | vanilla Strong-4DVar | no (state only) |
| ETKF | vanilla ETKF | no (state only) |
| EnKF | vanilla EnKF | no (state only) |

---

## State RMSE (per case)

Pooled RMSE over the observed subspace, grouped slow (8D) / obs_fast (16D) / mean (24D). Lower is better.

| Method | S0 slow | S0 obs_fast | S0 mean | S1 slow | S1 obs_fast | S1 mean |
|---|---|---|---|---|---|---|
| Joint-EnKF | 0.3473 | 0.9130 | 0.7244 | 1.1899 | 1.5954 | 1.4602 |
| Joint-ETKF | 0.3006 | 0.8019 | 0.6348 | 1.1665 | 1.6631 | 1.4976 |
| Joint-Strong-4DVar | 0.3324 | 0.8919 | 0.7054 | 0.6343 | 1.4827 | 1.1999 |
| Strong-4DVar | 0.4559 | 0.8817 | 0.7398 | 1.0553 | 1.6202 | 1.4319 |
| ETKF | 0.5756 | 1.0286 | 0.8776 | 1.2131 | 1.7246 | 1.5541 |
| EnKF | 0.4873 | 1.0927 | 0.8909 | 1.2371 | 1.6393 | 1.5052 |

*Best is the lowest per column; rendered from the comparator JSON.*

---

## Energy Score (ES, per case)

N=30 ensemble Energy Score on the observed subspace (subsampled to 24D). Lower is better.

| Method | S0 ES | S1 ES |
|---|---|---|
| Joint-EnKF | 0.3703 | 0.8438 |
| Joint-ETKF | 0.2991 | 0.9382 |
| Joint-Strong-4DVar | 0.4575 | 0.8100 |
| Strong-4DVar | 0.4852 | 0.9817 |
| ETKF | 0.4508 | 0.9988 |
| EnKF | 0.4581 | 0.8940 |

---

## Explained variance (EV, per case)

Pooled explained variance over the observed subspace, grouped slow (8D) / obs_fast (16D) / mean (24D). Higher is better.

| Method | S0 slow | S0 obs_fast | S0 mean | S1 slow | S1 obs_fast | S1 mean |
|---|---|---|---|---|---|---|
| Joint-EnKF | 0.9663 | 0.6793 | 0.7750 | 0.5985 | 0.0443 | 0.2291 |
| Joint-ETKF | 0.9739 | 0.7441 | 0.8207 | 0.6133 | -0.0349 | 0.1811 |
| Joint-Strong-4DVar | 0.9682 | 0.6528 | 0.7579 | 0.8815 | 0.1790 | 0.4132 |
| Strong-4DVar | 0.9405 | 0.6532 | 0.7490 | 0.6796 | 0.0202 | 0.2400 |
| ETKF | 0.9083 | 0.5880 | 0.6948 | 0.5821 | -0.1161 | 0.1166 |
| EnKF | 0.9343 | 0.5502 | 0.6782 | 0.5660 | -0.0093 | 0.1824 |

---

## Parameter RMSE — S0

Per-parameter RMSE (`F, c1, hx, eps, w1..w4`) against the per-window true params. `†` marks fast weights defaulted to the reference prior (not estimated) on the J=2 S1 dynamics.

| Method | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| Joint-EnKF | 0.1444 | 0.0192 | 0.0178 | 0.0015 | 0.1139 | 0.1226 | 0.0121 | 0.0114 | 0.0553 |
| Joint-ETKF | 0.1472 | 0.0173 | 0.0168 | 0.0018 | 0.1149 | 0.1249 | 0.0118 | 0.0114 | 0.0558 |
| Joint-Strong-4DVar | 0.8513 | 0.0023 | 0.1202 | 0.0116 | 0.2513 | 0.1997 | 0.1603 | 0.2083 | 0.2256 |

---

## Parameter RMSE — S1

Per-parameter RMSE (`F, c1, hx, eps, w1..w4`) against the per-window true params. `†` marks fast weights defaulted to the reference prior (not estimated) on the J=2 S1 dynamics.

| Method | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| Joint-EnKF | 0.7868 | 0.1063 | 0.0647 | 0.0108 | 0.1170† | 0.1179† | 0.0000† | 0.0000† | 0.1504 |
| Joint-ETKF | 0.6248 | 0.1098 | 0.0616 | 0.0107 | 0.1137† | 0.1177† | 0.0000† | 0.0000† | 0.1298 |
| Joint-Strong-4DVar | 1.4336 | 0.1011 | 0.2674 | 0.0412 | 0.2671† | 0.2678† | 0.0000† | 0.0000† | 0.2973 |

---

## Context: L9 joint neural baseline (ens30 × 10)

Single-sample L9 `JointCFM` multi-tau state RMSE and param-RMSE mean, for reference against the DA rows. (Full neural tables live in 
`l96_joint_neural_benchmark.md`.)

| Case | L9 state RMSE | L9 param-RMSE mean | Best DA state RMSE |
|---|---|---|---|
| S0 | 0.6257 | 0.0591 | 0.6348 (Joint-ETKF) |
| S1 | 0.6313 | 0.0615 | 1.1999 (Joint-Strong-4DVar) |

*Best DA state RMSE is the minimum across the benchmarked methods for that case.*

---

## Consistency check

The comparator loads the cached `l96_datasets_obsj2_int100_nwin200.pt` and runs `evaluate_baseline` (the same code path as the vanilla DA caches). State RMSE/EV/ES are pooled over all 200 windows and subsampled to `obs_var_indices` (24D); param RMSE compares each joint filter's 8-wide estimate against `true_*` (padded to 8 with the reference prior on S1).
