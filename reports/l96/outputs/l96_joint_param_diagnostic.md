# L96 joint state-parameter — per-parameter diagnostic (offline recompute)

**Source:** the eval arrays stored by `eval_joint_neural_l96.py` (`joint_estimates_{case}.npz` / `..._ens30.npz`), recomputed offline — no inference re-run. Pooled over the 200 cached windows (Obs30, observed subspace 24D).

**Metrics:** per-parameter `RMSE = sqrt(mean((pred-true)^2))`, `EV = 1 - mean((pred-true)^2)/var(true)`, `NRMSE = RMSE / mean(|true|)`; free forecast = 300-step rollout RMSE/EV between estimated- and true-parameter trajectories from the same x0 + forcing (observed subspace).

*Reading note:* per-parameter EV is dominated by the parameter's own scale — the D-subsystem params (`eps, w3, w4`) have very small true variance (~0.1 vs F~8), so even small absolute errors yield large negative EV there. **NRMSE is the fair cross-parameter comparison** (normalizes by mean(|true|)); free-forecast EV is the most physically meaningful summary of a parameter block.

---

## L7_joint_cfm_s0s1 — JointCFM tau=0

### Single-sample (n_members=1) per-parameter metrics

Pooled over the 200 windows. RMSE = `sqrt(mean((pred-true)^2))`; EV = `1 - mean((pred-true)^2)/var(true)`; NRMSE = RMSE / mean(|true|).

| Case | P | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 | RMSE | 0.3860 | 0.1271 | 0.0781 | 0.0337 | 0.1200 | 0.1269 | 0.0526 | 0.0503 | 0.1219 |
| S0 | EV | 0.8117 | -0.2044 | 0.5605 | -6.8554 | -0.1180 | -0.0806 | -19.3772 | -19.0373 | -5.5376 |
| S0 | NRMSE | 0.0479 | 0.1289 | 0.0793 | 0.3378 | 0.1188 | 0.1260 | 0.5177 | 0.5063 | 0.2328 |
| S1 | RMSE | 0.4696 | 0.2011 | 0.1794 | 0.0561 | 0.2686 | 0.1469 | 0.0769 | 0.0710 | 0.1837 |
| S1 | EV | 0.7449 | -1.7121 | -1.4334 | -25.3986 | -4.5759 | -0.6460 | -41.0648 | -37.8105 | -13.9871 |
| S1 | NRMSE | 0.0590 | 0.2006 | 0.1799 | 0.5631 | 0.2668 | 0.1464 | 0.7822 | 0.7052 | 0.3629 |

---

### Free forecast (single-sample, 300-step)

Free forecast RMSE / EV between a rollout with the **estimated** params and one with the **true** params, from the same x0 and forcing (observed subspace). High RMSE / negative EV => parameter error destroys short-term forecast skill.

| Case | RMSE | EV |
|---|---|---|
| S0 | 3.1491 | -2.5842 |
| S1 | 2.7176 | -1.8472 |

---

### ens30 (n_members=30, k=1) per-parameter metrics

`params_pred` is the member-mean across the 30 members.

| Case | P | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 | RMSE | 0.3699 | 0.1269 | 0.0722 | 0.0214 | 0.1116 | 0.1249 | 0.0509 | 0.0379 | 0.1145 |
| S0 | EV | 0.8271 | -0.2010 | 0.6245 | -2.1650 | 0.0334 | -0.0455 | -18.0951 | -10.3711 | -3.6741 |
| S0 | NRMSE | 0.0459 | 0.1287 | 0.0733 | 0.2144 | 0.1105 | 0.1239 | 0.5011 | 0.3814 | 0.1974 |
| S1 | RMSE | 0.4440 | 0.1975 | 0.1821 | 0.0497 | 0.2643 | 0.1418 | 0.0784 | 0.0642 | 0.1777 |
| S1 | EV | 0.7720 | -1.6139 | -1.5070 | -19.7073 | -4.3982 | -0.5333 | -42.7340 | -30.7630 | -12.5606 |
| S1 | NRMSE | 0.0558 | 0.1969 | 0.1826 | 0.4987 | 0.2625 | 0.1413 | 0.7976 | 0.6380 | 0.3467 |

---

### ens30 (n_members=30, k=10) per-parameter metrics

`params_pred` is the member-mean across the 30 members.

| Case | P | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 | RMSE | 0.3699 | 0.1269 | 0.0722 | 0.0214 | 0.1116 | 0.1249 | 0.0509 | 0.0379 | 0.1145 |
| S0 | EV | 0.8271 | -0.2010 | 0.6245 | -2.1650 | 0.0334 | -0.0455 | -18.0951 | -10.3711 | -3.6741 |
| S0 | NRMSE | 0.0459 | 0.1287 | 0.0733 | 0.2144 | 0.1105 | 0.1239 | 0.5011 | 0.3814 | 0.1974 |
| S1 | RMSE | 0.4440 | 0.1975 | 0.1821 | 0.0497 | 0.2643 | 0.1418 | 0.0784 | 0.0642 | 0.1777 |
| S1 | EV | 0.7720 | -1.6139 | -1.5070 | -19.7073 | -4.3982 | -0.5333 | -42.7340 | -30.7630 | -12.5606 |
| S1 | NRMSE | 0.0558 | 0.1969 | 0.1826 | 0.4987 | 0.2625 | 0.1413 | 0.7976 | 0.6380 | 0.3467 |

---

### Free forecast (ens30, k=1, 300-step)

| Case | RMSE | EV |
|---|---|---|
| S0 | 0.7831 | 0.6874 |
| S1 | 1.0964 | 0.3943 |

---

### Free forecast (ens30, k=10, 300-step)

| Case | RMSE | EV |
|---|---|---|
| S0 | 0.7831 | 0.6874 |
| S1 | 1.0964 | 0.3943 |

---


## L8_joint_direct_unet_s0s1 — JointDirectUNet

### Single-sample (n_members=1) per-parameter metrics

Pooled over the 200 windows. RMSE = `sqrt(mean((pred-true)^2))`; EV = `1 - mean((pred-true)^2)/var(true)`; NRMSE = RMSE / mean(|true|).

| Case | P | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 | RMSE | 0.3600 | 0.1363 | 0.1559 | 0.0220 | 0.1089 | 0.1232 | 0.0140 | 0.0134 | 0.1167 |
| S0 | EV | 0.8363 | -0.3838 | -0.7497 | -2.3407 | 0.0789 | -0.0187 | -0.4329 | -0.4232 | -0.4292 |
| S0 | NRMSE | 0.0447 | 0.1382 | 0.1583 | 0.2203 | 0.1078 | 0.1223 | 0.1373 | 0.1349 | 0.1330 |
| S1 | RMSE | 0.5463 | 0.1380 | 0.1407 | 0.0302 | 0.1213 | 0.1220 | 0.0241 | 0.0150 | 0.1422 |
| S1 | EV | 0.6548 | -0.2761 | -0.4983 | -6.6746 | -0.1366 | -0.1355 | -3.1434 | -0.7360 | -1.3682 |
| S1 | NRMSE | 0.0686 | 0.1376 | 0.1412 | 0.3036 | 0.1205 | 0.1216 | 0.2455 | 0.1492 | 0.1610 |

---

### Free forecast (single-sample, 300-step)

Free forecast RMSE / EV between a rollout with the **estimated** params and one with the **true** params, from the same x0 and forcing (observed subspace). High RMSE / negative EV => parameter error destroys short-term forecast skill.

| Case | RMSE | EV |
|---|---|---|
| S0 | 0.8476 | 0.6412 |
| S1 | 0.9259 | 0.5730 |

---

### ens30 (n_members=30, k=1) per-parameter metrics

`params_pred` is the member-mean across the 30 members.

| Case | P | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 | RMSE | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| S1 | RMSE | -- | -- | -- | -- | -- | -- | -- | -- | -- |

---

### ens30 (n_members=30, k=10) per-parameter metrics

`params_pred` is the member-mean across the 30 members.

| Case | P | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 | RMSE | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| S1 | RMSE | -- | -- | -- | -- | -- | -- | -- | -- | -- |

---

### Free forecast (ens30, k=1, 300-step)

| Case | RMSE | EV |
|---|---|---|
| S0 | -- | -- |
| S1 | -- | -- |

---

### Free forecast (ens30, k=10, 300-step)

| Case | RMSE | EV |
|---|---|---|
| S0 | -- | -- |
| S1 | -- | -- |

---


## L9_joint_cfm_s0s1_multitau — JointCFM multi-tau

### Single-sample (n_members=1) per-parameter metrics

Pooled over the 200 windows. RMSE = `sqrt(mean((pred-true)^2))`; EV = `1 - mean((pred-true)^2)/var(true)`; NRMSE = RMSE / mean(|true|).

| Case | P | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 | RMSE | 3.1023 | 0.4477 | 0.4995 | 0.3639 | 0.4691 | 0.4675 | 0.3186 | 0.3338 | 0.7503 |
| S0 | EV | -11.1602 | -13.9393 | -16.9556 | -912.6987 | -16.0882 | -13.6575 | -746.1912 | -880.5987 | -326.4112 |
| S0 | NRMSE | 0.3852 | 0.4540 | 0.5071 | 3.6428 | 0.4645 | 0.4641 | 3.1349 | 3.3585 | 1.5514 |
| S1 | RMSE | 4.0048 | 0.8054 | 0.5446 | 0.3856 | 0.6696 | 0.5189 | 0.3695 | 0.3486 | 0.9559 |
| S1 | EV | -17.5492 | -42.4781 | -21.4313 | -1246.1704 | -33.6396 | -19.5286 | -971.1744 | -935.4407 | -410.9265 |
| S1 | NRMSE | 0.5029 | 0.8030 | 0.5462 | 3.8702 | 0.6650 | 0.5169 | 3.7605 | 3.4642 | 1.7661 |

---

### Free forecast (single-sample, 300-step)

Free forecast RMSE / EV between a rollout with the **estimated** params and one with the **true** params, from the same x0 and forcing (observed subspace). High RMSE / negative EV => parameter error destroys short-term forecast skill.

| Case | RMSE | EV |
|---|---|---|
| S0 | 10.2623 | -36.8967 |
| S1 | 9.0984 | -28.3942 |

---

### ens30 (n_members=30, k=1) per-parameter metrics

`params_pred` is the member-mean across the 30 members.

| Case | P | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 | RMSE | 3.1185 | 0.2672 | 0.3941 | 0.1278 | 0.2934 | 0.3178 | 0.0849 | 0.0696 | 0.5842 |
| S0 | EV | -11.2877 | -4.3208 | -10.1819 | -111.7624 | -5.6859 | -5.7753 | -52.0575 | -37.3291 | -29.8001 |
| S0 | NRMSE | 0.3872 | 0.2710 | 0.4002 | 1.2797 | 0.2905 | 0.3155 | 0.8354 | 0.7003 | 0.5600 |
| S1 | RMSE | 4.0439 | 0.6961 | 0.4690 | 0.1150 | 0.5683 | 0.3213 | 0.1273 | 0.1196 | 0.8076 |
| S1 | EV | -17.9124 | -31.4764 | -15.6363 | -109.9977 | -23.9564 | -6.8698 | -114.4085 | -109.1975 | -53.6819 |
| S1 | NRMSE | 0.5078 | 0.6940 | 0.4704 | 1.1546 | 0.5644 | 0.3201 | 1.2957 | 1.1884 | 0.7744 |

---

### ens30 (n_members=30, k=10) per-parameter metrics

`params_pred` is the member-mean across the 30 members.

| Case | P | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 | RMSE | 3.1185 | 0.2672 | 0.3941 | 0.1278 | 0.2934 | 0.3178 | 0.0849 | 0.0696 | 0.5842 |
| S0 | EV | -11.2877 | -4.3208 | -10.1819 | -111.7624 | -5.6859 | -5.7753 | -52.0575 | -37.3291 | -29.8001 |
| S0 | NRMSE | 0.3872 | 0.2710 | 0.4002 | 1.2797 | 0.2905 | 0.3155 | 0.8354 | 0.7003 | 0.5600 |
| S1 | RMSE | 4.0439 | 0.6961 | 0.4690 | 0.1150 | 0.5683 | 0.3213 | 0.1273 | 0.1196 | 0.8076 |
| S1 | EV | -17.9124 | -31.4764 | -15.6363 | -109.9977 | -23.9564 | -6.8698 | -114.4085 | -109.1975 | -53.6819 |
| S1 | NRMSE | 0.5078 | 0.6940 | 0.4704 | 1.1546 | 0.5644 | 0.3201 | 1.2957 | 1.1884 | 0.7744 |

---

### Free forecast (ens30, k=1, 300-step)

| Case | RMSE | EV |
|---|---|---|
| S0 | 0.7375 | 0.7333 |
| S1 | 1.1100 | 0.4086 |

---

### Free forecast (ens30, k=10, 300-step)

| Case | RMSE | EV |
|---|---|---|
| S0 | 18.5879 | -120.6307 |
| S1 | 18.5818 | -119.5533 |

---

