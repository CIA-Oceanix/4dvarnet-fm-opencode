# L96 Joint State-Parameter DA Benchmark (ETKF)

**System:** Lorenz-96 two-scale (NO=8, J=4), observed subspace state_dim=24 (Obs30, obs_interval=100), 200 shared test windows.

**Models:** Joint-ETKF estimates the 24D observed state **and** the 8 model parameters `F, c1, hx, eps, w1..w4` (fast weights per index; h fixed) via an augmented-state ensemble, benchmarked against the state-only vanilla ETKF on the same cached S0/S1 test set. On S1 (reduced 24D J=2 dynamics) the DA only carries `w1,w2`; `w3,w4` default to the reference prior `[1.0, 0.1]` and are marked with a `†` (not estimated).

---

## Benchmarked methods

| Method | Type | Describes state + 8 params? |
|---|---|---|
| ETKF | Vanilla ETKF | no (state only) |
| Joint-ETKF | Joint ETKF | yes |

---

## State RMSE (per case)

Pooled RMSE over the observed subspace, grouped slow (8D) / obs_fast (16D) / mean (24D). Lower is better.

| Method | S0 slow | S0 obs_fast | S0 mean | S1 slow | S1 obs_fast | S1 mean |
|---|---|---|---|---|---|---|
| ETKF | 0.5756 | 1.0286 | 0.8776 | 1.2131 | 1.7246 | 1.5541 |
| Joint-ETKF | 0.2979 | 0.8011 | 0.6334 | 1.1678 | 1.6618 | 1.4971 |

*Best is the lowest per column; rendered from the comparator JSON.*

---

## Energy Score (ES, per case)

N=30 ensemble Energy Score on the observed subspace (subsampled to 24D). Lower is better.

| Method | S0 ES | S1 ES |
|---|---|---|
| ETKF | 0.4508 | 0.9988 |
| Joint-ETKF | 0.2977 | 0.9374 |

---

## Parameter RMSE — S0

Per-parameter RMSE (`F, c1, hx, eps, w1..w4`) against the per-window true params. `†` marks fast weights defaulted to the reference prior (not estimated) on the J=2 S1 dynamics.

| Method | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| Joint-ETKF | 0.1306 | 0.0167 | 0.0156 | 0.0016 | 0.1155 | 0.1218 | 0.0119 | 0.0112 | 0.0531 |

---

## Parameter RMSE — S1

Per-parameter RMSE (`F, c1, hx, eps, w1..w4`) against the per-window true params. `†` marks fast weights defaulted to the reference prior (not estimated) on the J=2 S1 dynamics.

| Method | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| Joint-ETKF | 0.6082 | 0.1052 | 0.0637 | 0.0106 | 0.1161† | 0.1186† | 0.0000† | 0.0000† | 0.1278 |

---

## Context: L9 joint neural baseline (ens30 × 10)

Single-sample L9 `JointCFM` multi-tau state RMSE and param-RMSE mean, for reference against the DA rows. (Full neural tables live in 
`l96_joint_neural_benchmark.md`.)

| Case | L9 state RMSE | L9 param-RMSE mean | Best DA state RMSE |
|---|---|---|---|
| S0 | 0.6257 | 0.0591 | 0.6334 (Joint-ETKF) |
| S1 | 0.6313 | 0.0615 | 1.4971 (Joint-ETKF) |

*Best DA state RMSE is the minimum across the benchmarked methods for that case.*

---

## Consistency check

The comparator loads the cached `l96_datasets_obsj2_int100_nwin200.pt` and runs `evaluate_baseline` (the same code path as the vanilla DA caches). State RMSE/EV/ES are pooled over all 200 windows and subsampled to `obs_var_indices` (24D); param RMSE compares the joint ETKF's 8-wide estimate against `true_*` (padded to 8 with the reference prior on S1).
