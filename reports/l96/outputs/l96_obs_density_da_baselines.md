# L96 DA-Baseline Observation Density: obsj2 (24D) vs slow-only obsj0 (8D)

**System:** Lorenz-96 two-scale (NO=8, J=4), 200 shared cached S0/S1 test windows, Obs30 (obs_interval=100). Same dynamics/truth/params; only the observation changes.

**Configurations:**
* **obsj2 (canonical):** 24D observed = 8 slow X + 16 fast Y1,Y2 per node.
* **slow-only obsj0:** only the **8 slow X** observed (no fast Y); S1 reduced dynamics kept at J=2 (24D state).

**Eval subspace:** both are scored on the identical 24D group (slow + first-2-fast), so the metrics are directly comparable. On obsj0 the `obs_fast` group reflects fast variables **not observed** by the DA (slow-only stress test).

**S1 forcings:** all rows use the corrected `case=2` config, i.e. the DA is fed the **corrupted** forcing `forcing_corrupted` on S1 (and `forcing_true` on S0).

---

## State-only DA baselines (S0/S1)

| Case | Method | obsj2 mean | obsj0 mean | Δ mean | obsj2 slow | obsj0 slow | obsj2 obs_fast | obsj0 obs_fast |
|---|---|---|---|---|---|---|---|---|
| S0 | Strong-4DVar | 0.7383 | 1.4382 | 0.6998 | 0.4539 | 0.4131 | 0.8805 | 1.9507 |
| S0 | EnKF | 0.8943 | 1.2735 | 0.3792 | 0.4892 | 0.4491 | 1.0968 | 1.6858 |
| S0 | ETKF | 0.8662 | 1.2481 | 0.3819 | 0.4677 | 0.4607 | 1.0655 | 1.6418 |
| S1 | Strong-4DVar | 1.4369 | 1.6165 | 0.1796 | 1.0522 | 0.5262 | 1.6293 | 2.1616 |
| S1 | EnKF | 1.5123 | 1.6978 | 0.1855 | 1.2298 | 0.5958 | 1.6535 | 2.2488 |
| S1 | ETKF | 1.4748 | 1.7073 | 0.2325 | 1.2202 | 0.6229 | 1.6021 | 2.2495 |

---

## Joint state-parameter DA baselines (S0/S1)

| Case | Method | obsj2 mean | obsj0 mean | Δ mean | obsj2 EV | obsj0 EV | obsj2 param* | obsj0 param* |
|---|---|---|---|---|---|---|---|---|
| S0 | Joint-EnKF | 0.7237 | 1.2079 | 0.4843 | 0.7760 | 0.3517 | 0.0544 | 0.0464 |
| S0 | Joint-ETKF | 0.6414 | 1.1865 | 0.5451 | 0.8157 | 0.3523 | 0.0545 | 0.0449 |
| S0 | Joint-Strong-4DVar | 0.7100 | 1.5805 | 0.8705 | 0.7578 | -0.2244 | 0.2282 | 0.2900 |
| S1 | Joint-EnKF | 1.4646 | 1.6687 | 0.2042 | 0.2229 | -0.2292 | 0.1481 | 0.1570 |
| S1 | Joint-ETKF | 1.5125 | 1.6008 | 0.0883 | 0.1623 | -0.0966 | 0.1296 | 0.1579 |
| S1 | Joint-Strong-4DVar | 1.1997 | 1.3187 | 0.1190 | 0.4131 | 0.1867 | 0.3142 | 0.2725 |
*param = mean of the (identifiable) per-parameter RMSE (8 on S0, 6 on S1 — w3/w4 pinned to the reference prior at J=2).*

---

## Joint-DA per-parameter RMSE (S0/S1)

### S0 — per-parameter RMSE

| Method / config | mean | F | c1 | hx | eps | w1 | w2 | w3 | w4 |
|---|---|---|---|---|---|---|---|---|---|
| Joint-EnKF / obsj2 | 0.0544 | 0.1368 | 0.0174 | 0.0163 | 0.0017 | 0.1167 | 0.1226 | 0.0120 | 0.0117 |
| Joint-EnKF / obsj0 | 0.0464 | 0.0897 | 0.0103 | 0.0096 | 0.0009 | 0.1143 | 0.1230 | 0.0118 | 0.0114 |
| Joint-ETKF / obsj2 | 0.0545 | 0.1370 | 0.0163 | 0.0175 | 0.0017 | 0.1164 | 0.1240 | 0.0121 | 0.0113 |
| Joint-ETKF / obsj0 | 0.0449 | 0.0806 | 0.0096 | 0.0101 | 0.0011 | 0.1124 | 0.1227 | 0.0118 | 0.0113 |
| Joint-Strong-4DVar / obsj2 | 0.2282 | 0.8606 | 0.0026 | 0.1182 | 0.0116 | 0.2534 | 0.2122 | 0.1633 | 0.2038 |
| Joint-Strong-4DVar / obsj0 | 0.2900 | 1.1494 | 0.0016 | 0.3046 | 0.0254 | 0.2685 | 0.2651 | 0.1528 | 0.1526 |

### S1 — per-parameter RMSE

| Method / config | mean | F | c1 | hx | eps | w1 | w2 | w3 | w4 |
|---|---|---|---|---|---|---|---|---|---|
| Joint-EnKF / obsj2 | 0.1481 | 0.7683 | 0.1025 | 0.0668 | 0.0111 | 0.1176 | 0.1185 | 0.0000 | 0.0000 |
| Joint-EnKF / obsj0 | 0.1570 | 0.8162 | 0.1014 | 0.1004 | 0.0100 | 0.1145 | 0.1137 | 0.0000 | 0.0000 |
| Joint-ETKF / obsj2 | 0.1296 | 0.6177 | 0.1047 | 0.0675 | 0.0110 | 0.1182 | 0.1182 | 0.0000 | 0.0000 |
| Joint-ETKF / obsj0 | 0.1579 | 0.8244 | 0.1007 | 0.1002 | 0.0102 | 0.1134 | 0.1141 | 0.0000 | 0.0000 |
| Joint-Strong-4DVar / obsj2 | 0.3142 | 1.5396 | 0.1418 | 0.2674 | 0.0415 | 0.2624 | 0.2608 | 0.0000 | 0.0000 |
| Joint-Strong-4DVar / obsj0 | 0.2725 | 1.1585 | 0.1106 | 0.2232 | 0.0254 | 0.3336 | 0.3285 | 0.0000 | 0.0000 |

---

## Neural (SDA, score-based DA) — same obsj2 checkpoint, guidance-only restriction

No retraining and no new dataset: the slow-only column restricts which channels the DPS guidance cost may see (`evaluation/sda_sampler.py`'s `obs_indices`), reusing the identical obsj2-trained checkpoint and cached test set. All rows are the 30-member-ensemble (`ens30×10`) convention, same as the consolidated benchmark.

| Case | Method | obsj2 mean | obsj0 mean | Δ mean | obsj2 slow | obsj0 slow | obsj2 obs_fast | obsj0 obs_fast |
|---|---|---|---|---|---|---|---|---|
| S0 | SDA1 | 0.7185 | 1.1350 | 0.4165 | 0.4431 | 0.4543 | 0.8562 | 1.4753 |
| S0 | SDA2-mixed | 0.7074 | 1.1158 | 0.4084 | 0.3863 | 0.4066 | 0.8679 | 1.4704 |
| S0 | SDA2-nominal | 0.7046 | 1.1098 | 0.4052 | 0.3801 | 0.4040 | 0.8668 | 1.4627 |
| S1 | SDA1 | 0.7168 | 1.1369 | 0.4201 | 0.4387 | 0.4514 | 0.8558 | 1.4797 |
| S1 | SDA2-mixed | 0.7051 | 1.1187 | 0.4136 | 0.3803 | 0.4021 | 0.8674 | 1.4769 |
| S1 | SDA2-nominal | 0.7033 | 1.1121 | 0.4087 | 0.3763 | 0.4020 | 0.8669 | 1.4671 |

