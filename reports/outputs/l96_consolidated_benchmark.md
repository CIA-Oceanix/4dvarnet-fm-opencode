# L96 Consolidated Benchmark — DA baselines vs neural models

Setup: two-scale L96, Obs30 (`obs_interval=100`, `obs_j=2` → 24D observed space), dws=500, 200 shared cached test windows; S1 = ±20% params + ±10% bias (DA forward model uses biased `*_da`).

All table values are recomputed from the stored trajectory arrays via `evaluation/estimate_metrics.py`; **bold** marks the best value per column.

## RMSE (pooled, lower is better)

### RMSE by variable group

| Method | S0 all | S0 slow | S0 fast | S1 all | S1 slow | S1 fast | S1/S0 |
|---|---|---|---|---|---|---|---|
| ETKF | 0.8867 | 0.4765 | 1.0917 | 1.4961 | 1.2477 | 1.6203 | 1.687 |
| EnKF | 0.9110 | 0.4930 | 1.1200 | 1.5314 | 1.2590 | 1.6675 | 1.681 |
| Strong-4DVar | 0.8142 | 0.4634 | 0.9896 | 1.4566 | 1.0792 | 1.6453 | 1.789 |
| L1b | 0.6221 | **0.4019** | 0.7321 | 0.6256 | **0.4006** | 0.7381 | 1.006 |
| L2b | 0.6330 | 0.4188 | 0.7401 | 0.6336 | 0.4154 | 0.7426 | 1.001 |
| L3 | 0.6877 | 0.4706 | 0.7963 | 0.6906 | 0.4715 | 0.8001 | 1.004 |
| L4 | **0.6189** | 0.4037 | **0.7265** | **0.6211** | 0.4024 | **0.7304** | 1.004 |
| L5 | 0.6603 | 0.4858 | 0.7476 | 0.6604 | 0.4831 | 0.7490 | 1.000 |
| L6 | 0.6390 | 0.4136 | 0.7517 | 0.6381 | 0.4097 | 0.7523 | 0.999 |

Note on conventions: the DA metric cache stores the **mean of per-window RMSEs** (evaluation/run_l96.py), while this table uses the **pooled** convention (`sqrt(mean sq err)` over all windows/timesteps) for every method — the same convention as the neural evaluation. Pooled RMSE is ≤ mean-of-window RMSE, so DA values here are slightly lower (more favorable) than in the legacy cache; both orderings agree.

## Explained Variance (higher is better)

### EV by variable group

| Method | S0 all | S0 slow | S0 fast | S1 all | S1 slow | S1 fast |
|---|---|---|---|---|---|---|
| ETKF | 0.6932 | 0.9391 | 0.5702 | 0.2206 | 0.5716 | 0.0450 |
| EnKF | 0.6767 | 0.9349 | 0.5477 | 0.1815 | 0.5638 | -0.0096 |
| Strong-4DVar | 0.7452 | 0.9424 | 0.6466 | 0.2399 | 0.6795 | 0.0200 |
| L1b | 0.8567 | **0.9567** | 0.8067 | 0.8538 | **0.9558** | 0.8028 |
| L2b | 0.8527 | 0.9530 | 0.8025 | 0.8510 | 0.9525 | 0.8003 |
| L3 | 0.8278 | 0.9406 | 0.7714 | 0.8251 | 0.9388 | 0.7682 |
| L4 | **0.8586** | 0.9563 | **0.8097** | **0.8564** | 0.9554 | **0.8068** |
| L5 | 0.8445 | 0.9364 | 0.7985 | 0.8430 | 0.9354 | 0.7969 |
| L6 | 0.8489 | 0.9541 | 0.7963 | 0.8480 | 0.9538 | 0.7951 |

## Energy Score (lower is better)

### ES by variable group

| Method | S0 all | S0 slow | S0 fast | S1 all | S1 slow | S1 fast |
|---|---|---|---|---|---|---|
| ETKF | 0.6109 | 0.3704 | 0.7312 | 1.0262 | 0.9460 | 1.0664 |
| EnKF | 0.6359 | 0.3873 | 0.7602 | 1.0745 | 0.9732 | 1.1251 |
| Strong-4DVar | 0.4859 | 0.3531 | 0.5523 | 0.9817 | 0.8169 | 1.0640 |
| L1b | **0.3914** | **0.2704** | 0.4520 | 0.3949 | **0.2702** | 0.4572 |
| L2b | 0.4056 | 0.2867 | 0.4650 | 0.4069 | 0.2850 | 0.4679 |
| L3 | 0.4438 | 0.3249 | 0.5033 | 0.4469 | 0.3282 | 0.5063 |
| L4 | **0.3915** | 0.2726 | **0.4509** | **0.3940** | 0.2725 | **0.4548** |
| L5 | 0.4321 | 0.3525 | 0.4720 | 0.4329 | 0.3507 | 0.4740 |
| L6 | 0.4068 | 0.2814 | 0.4695 | 0.4071 | 0.2803 | 0.4706 |

Caveat: EnKF/ETKF cached ES values are computed from their forecast ensembles (proper scoring rule, N=30); neural models and Strong-4DVar are deterministic, so their ES reduces to a per-dim MAE proxy (N=1). The two are not strictly comparable — deterministic ES ignores sharpness.

## Consistency checks

- DA cached metrics vs recomputed-from-npz (42 values): max |Δ| = 2.12e-04 → PASS
- Neural stored truth vs dataset true_state[:, obs_var_indices]: max |Δ| = 0.00e+00 → PASS

## Reconstruction examples (Hovmöller)

Windows ranked by per-window pooled 24D RMSE of Strong-4DVar (best DA scheme); each figure shows rows = Truth/methods and columns = state / |error| maps for the slow X (8D) and fast Y (16D) blocks. State colors share one scale per figure; error maps share one scale across all rows/methods (99.5th-percentile cap, noted on the colorbar). Dotted vertical lines on the truth row mark observation times.

| Case | Rank | Window | 4DVar win-RMSE | Strong-4DVar | EnKF | ETKF | L4 | L2b |
|---|---|---|---|---|---|---|---|---|
| S0 | worst | 138 | 1.388 | 1.388 | 1.244 | 1.186 | 0.808 | 0.818 |
| S0 | median | 197 | 0.817 | 0.817 | 0.949 | 0.840 | 0.551 | 0.548 |
| S0 | best | 134 | 0.419 | 0.419 | 0.795 | 0.754 | 0.572 | 0.573 |
| S1 | worst | 75 | 1.974 | 1.974 | 2.039 | 1.981 | 0.832 | 0.850 |
| S1 | median | 178 | 1.475 | 1.475 | 1.534 | 1.497 | 0.661 | 0.655 |
| S1 | best | 35 | 0.965 | 0.965 | 0.998 | 0.987 | 0.485 | 0.495 |

![s0-worst](figs/l96_hovm_s0_worst.png)

![s0-median](figs/l96_hovm_s0_median.png)

![s0-best](figs/l96_hovm_s0_best.png)

![s1-worst](figs/l96_hovm_s1_worst.png)

![s1-median](figs/l96_hovm_s1_median.png)

![s1-best](figs/l96_hovm_s1_best.png)
