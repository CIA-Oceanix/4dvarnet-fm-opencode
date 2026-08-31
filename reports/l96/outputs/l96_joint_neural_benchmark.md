# L96 Joint State-Parameter Neural Estimation Benchmark

**System:** Lorenz-96 two-scale (NO=8, J=4), observed subspace state_dim=24 (Obs30, obs_interval=100), 200 shared test windows.

**Models:** JointCFM + JointDirectUNet jointly estimate the 24D observed state **and** the 8 model parameters `F, c1, hx, eps, w1..w4` (fast weights per index; h fixed), matching the L96 joint DA convention. Each model's predictions are evaluated on the same cached S0/S1 test set used by the DA baselines.

**Oracle-free retrain (2026-08-31):** the numbers below are from the retrained checkpoints produced with the true-parameter oracle removed — the state UNet conditions on `[obs, forcing]` only (`cond_extra_dim=1`, `output_dim=state_dim`) and a dedicated parameter head (`ParamFlowCNN` / `ParamHeadCNN`) reads the params from that oracle-free state estimate; `true_params` appear only as the regression target. Earlier published per-parameter rows came from oracle-contaminated runs (true params fed into the UNet conditioning) and are **not** a valid baseline — the correct comparison is the **joint DA baselines** table below.

**Per-parameter detail:** `reports/l96/outputs/l96_joint_param_diagnostic.md` gives the full offline per-parameter RMSE / EV / NRMSE and free-forecast tables (single and ens30, all runs), recomputed from the stored eval arrays.

---

## Consolidated summary — neural vs DA (S0/S1)

Single-sample state RMSE (S0/S1), S1/S0 degradation, and **mean** per-parameter RMSE over the 8 params (F, c1, hx, eps, w1..w4). State RMSE beats DA on S1 (robust ≈1.0 degradation) but DA filters recover the parameters far better on S0 (Joint-ETKF mean per-param RMSE 0.053 vs best neural 0.122). L9's multi-τ param head is the notable failure (mean 0.750 on S0).

| Method | S0 state RMSE | S1 state RMSE | S1/S0 | S0 paramRMSE mean | S1 paramRMSE mean |
|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.6332 | 0.6513 | 1.0286 | 0.1219 | 0.1837 |
| L8_joint_direct_unet_s0s1 | 0.6247 | 0.9190 | 1.4710 | 0.1167 | 0.1422 |
| L9_joint_cfm_s0s1_multitau | 0.6619 | 0.6658 | 1.0059 | 0.7503 | 0.9559 |
| Joint-ETKF | 0.6334 | 1.4971 | 2.3638 | 0.0531 | 0.1278 |
| Joint-EnKF | 0.7263 | 1.4592 | 2.0091 | 0.0567 | 0.1479 |

*Lower is better for every column: state RMSE, S1/S0 degradation, and mean per-param RMSE. DA S1 paramRMSE average includes the pinned-to-prior `w3/w4` = 0, so it is not fully apples-to-apples (see the per-column parameter tables below).*

---

## Benchmarked models

| ID | Type | τ mode | Description |
|---|---|---|---|
| L7_joint_cfm_s0s1 | JointCFM | tau=0 | Conditional flow matching (state + 8-param joint output) trained at tau=0 only; sampled with a single Euler step. Hidden [64,128,256], 400 epochs. |
| L8_joint_direct_unet_s0s1 | JointDirectUNet | n/a | Single-pass joint regression obs -> (state, 8 params). Deterministic. Hidden [64,128,256], 200 epochs. |
| L9_joint_cfm_s0s1_multitau | JointCFM | multi-tau | Standard multi-tau conditional flow matching (state + 8-param joint output); sampled as a 30-member ensemble with 10 Euler steps (ens30 x 10, N=30). Hidden [64,128,256], 400 epochs. |

---

## Single-sample results (n_members=1, k=1)

State metrics over the observed subspace for the neural models (single-sample) and the joint-DA filters. S1/S0 is the degradation ratio (>1 means worse on the parameter-biased S1 setup). ES for the deterministic neural models and DA rows is the N=1 mean-absolute-error proxy; the DA filters' ES is N=30 (see DA note).

| ID | S0 RMSE | S0 EV | S0 ES | S1 RMSE | S1 EV | S1 ES | S1/S0 |
|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.6332 | 0.8520 | 0.4034 | 0.6513 ** | 0.8427 ** | 0.4163 ** | 1.0286 |
| L8_joint_direct_unet_s0s1 | 0.6247 ** | 0.8555 ** | 0.3931 | 0.9190 | 0.7069 | 0.5498 | 1.4710 |
| L9_joint_cfm_s0s1_multitau | 0.6619 | 0.8355 | 0.4155 | 0.6658 | 0.8325 | 0.4203 | 1.0059 ** |
| Joint-ETKF | 0.6334 | 0.8213 | 0.2977 ** | 1.4971 | 0.1819 | 0.9374 | 2.3638 |
| Joint-EnKF | 0.7263 | 0.7742 | 0.3709 | 1.4592 | 0.2302 | 0.8434 | 2.0091 |

*Best per column: lowest RMSE / ES / degradation, highest EV. The joint-DA rows (Joint-ETKF / Joint-EnKF) come from `l96_joint_comparison.json`; their ES is the N=30 ensemble score while the neural single-sample ES is an N=1 MAE proxy (not strictly comparable, flagged).*

---

## Ensemble results (n_members=30, k=1)

State RMSE / explained variance (EV) / energy score (ES) over the observed subspace, computed on the member-mean trajectory; ES is the proper N=30 ensemble scoring rule.

| ID | S0 RMSE | S0 EV | S0 ES | S1 RMSE | S1 EV | S1 ES |
|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.6293 ** | 0.8537 ** | 0.3994 ** | 0.6474 | 0.8445 | 0.4125 |
| L8_joint_direct_unet_s0s1 | -- | -- | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | 0.6339 | 0.8512 | 0.4009 | 0.6390 ** | 0.8479 ** | 0.4041 ** |

*Only ens30 runs present on disk are shown; missing runs render as -- (L8 is deterministic and is not run as an ensemble). Best per column: lowest RMSE/ES, highest EV.*

---

## Ensemble results (n_members=30, k=10)

State RMSE / explained variance (EV) / energy score (ES) over the observed subspace, computed on the member-mean trajectory; ES is the proper N=30 ensemble scoring rule.

| ID | S0 RMSE | S0 EV | S0 ES | S1 RMSE | S1 EV | S1 ES |
|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.6293 | 0.8537 | 0.3994 | 0.6474 | 0.8445 | 0.4125 |
| L8_joint_direct_unet_s0s1 | -- | -- | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | 0.5644 ** | 0.8810 ** | 0.3552 ** | 0.5727 ** | 0.8766 ** | 0.3617 ** |

*Only ens30 runs present on disk are shown; missing runs render as -- (L8 is deterministic and is not run as an ensemble). Best per column: lowest RMSE/ES, highest EV.*

---

## Parameter EV — ens30 (n_members=30, k=1)

Per-parameter explained variance from the **member-mean** parameters of the 30-member ensemble (offline from the stored eval arrays). Deep integration (k=10) of the multi-tau JointCFM parameter head can **collapse** the EV (hugely negative) even when the ensemble-mean state improves.

| ID | Case | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | S0 | 0.8271 ** | -0.2010 ** | 0.6245 ** | -2.1650 ** | 0.0334 ** | -0.0455 ** | -18.0951 ** | -10.3711 ** | -3.6741 ** |
| L7_joint_cfm_s0s1 | S1 | 0.7720 | -1.6139 | -1.5070 | -19.7073 | -4.3982 | -0.5333 | -42.7340 | -30.7630 | -12.5606 |
| L8_joint_direct_unet_s0s1 | S0 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L8_joint_direct_unet_s0s1 | S1 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | S0 | -11.2877 | -4.3208 | -10.1819 | -111.7624 | -5.6858 | -5.7753 | -52.0574 | -37.3291 | -29.8001 |
| L9_joint_cfm_s0s1_multitau | S1 | -17.9124 | -31.4764 | -15.6363 | -109.9977 | -23.9565 | -6.8698 | -114.4085 | -109.1974 | -53.6819 |

*Best per cell (highest EV) is bolded. L8 is deterministic (no ensemble).*

---

## Parameter EV — ens30 (n_members=30, k=10)

Per-parameter explained variance from the **member-mean** parameters of the 30-member ensemble (offline from the stored eval arrays). Deep integration (k=10) of the multi-tau JointCFM parameter head can **collapse** the EV (hugely negative) even when the ensemble-mean state improves.

| ID | Case | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | S0 | 0.8271 ** | -0.2010 ** | 0.6245 ** | -2.1650 ** | 0.0334 ** | -0.0455 ** | -18.0951 ** | -10.3711 ** | -3.6741 ** |
| L7_joint_cfm_s0s1 | S1 | 0.7720 | -1.6139 | -1.5070 | -19.7073 | -4.3982 | -0.5333 | -42.7340 | -30.7630 | -12.5606 |
| L8_joint_direct_unet_s0s1 | S0 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L8_joint_direct_unet_s0s1 | S1 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | S0 | -11.2877 | -4.3208 | -10.1819 | -111.7624 | -5.6858 | -5.7753 | -52.0574 | -37.3291 | -29.8001 |
| L9_joint_cfm_s0s1_multitau | S1 | -17.9124 | -31.4764 | -15.6363 | -109.9977 | -23.9565 | -6.8698 | -114.4085 | -109.1974 | -53.6819 |

*Best per cell (highest EV) is bolded. L8 is deterministic (no ensemble).*

---

## Parameter RMSE — S0 (single-sample)

Per-parameter RMSE (`F, c1, hx, eps, w1..w4`) and its mean across the 8 params.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.3860 | 0.1271 | 0.0781 | 0.0337 | 0.1200 | 0.1269 | 0.0526 | 0.0503 | 0.1219 |
| L8_joint_direct_unet_s0s1 | 0.3600 | 0.1363 | 0.1559 | 0.0220 | 0.1089 | 0.1232 | 0.0140 | 0.0134 | 0.1167 |
| L9_joint_cfm_s0s1_multitau | 3.1023 | 0.4477 | 0.4995 | 0.3639 | 0.4691 | 0.4675 | 0.3186 | 0.3338 | 0.7503 |
| Joint-ETKF | 0.1306 | 0.0167 | 0.0156 | 0.0016 | 0.1155 | 0.1218 | 0.0119 | 0.0112 | 0.0531 |
| Joint-EnKF | 0.1532 | 0.0180 | 0.0168 | 0.0019 | 0.1157 | 0.1245 | 0.0120 | 0.0113 | 0.0567 |

*Joint-DA rows are the co-estimated 8-param RMSE from `l96_joint_comparison.json` (S1 `w3`/`w4` are pinned to the reference prior, not estimated). Per-parameter EV and the free forecast are **not** stored for DA (the per-window predictions were not archived), so those tables show DA as `--`.*

---

## Parameter RMSE — S1 (single-sample)

Per-parameter RMSE (`F, c1, hx, eps, w1..w4`) and its mean across the 8 params.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.4696 | 0.2011 | 0.1794 | 0.0561 | 0.2686 | 0.1469 | 0.0769 | 0.0710 | 0.1837 |
| L8_joint_direct_unet_s0s1 | 0.5463 | 0.1380 | 0.1407 | 0.0302 | 0.1213 | 0.1220 | 0.0241 | 0.0150 | 0.1422 |
| L9_joint_cfm_s0s1_multitau | 4.0048 | 0.8054 | 0.5446 | 0.3856 | 0.6696 | 0.5189 | 0.3695 | 0.3486 | 0.9559 |
| Joint-ETKF | 0.6082 | 0.1052 | 0.0637 | 0.0106 | 0.1161 | 0.1186 | 0.0000 | 0.0000 | 0.1278 |
| Joint-EnKF | 0.7637 | 0.1053 | 0.0640 | 0.0112 | 0.1194 | 0.1197 | 0.0000 | 0.0000 | 0.1479 |

*Joint-DA rows are the co-estimated 8-param RMSE from `l96_joint_comparison.json` (S1 `w3`/`w4` are pinned to the reference prior, not estimated). Per-parameter EV and the free forecast are **not** stored for DA (the per-window predictions were not archived), so those tables show DA as `--`.*

---

## Normalized parameter RMSE (NRMSE) — S0 (single-sample)

Per-parameter NRMSE = `param_RMSE / mean(|true_param|)`, which normalizes away the scale difference between parameters (e.g. F~8 vs eps~0.1) so each competes equally. Mean is across the 8 params; lower is better.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.0479 | 0.1289 ** | 0.0793 ** | 0.3378 | 0.1188 | 0.1260 | 0.5177 | 0.5063 | 0.2328 |
| L8_joint_direct_unet_s0s1 | 0.0447 ** | 0.1382 | 0.1583 | 0.2203 ** | 0.1078 ** | 0.1223 ** | 0.1373 ** | 0.1349 ** | 0.1330 ** |
| L9_joint_cfm_s0s1_multitau | 0.3852 | 0.4540 | 0.5071 | 3.6428 | 0.4645 | 0.4641 | 3.1349 | 3.3585 | 1.5514 |
| Joint-ETKF | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Joint-EnKF | -- | -- | -- | -- | -- | -- | -- | -- | -- |

*Best per column (lowest NRMSE) is bolded. Joint-DA rows render as `--`: per-parameter NRMSE needs the true-parameter scale (`mean(|true|)`) which is not archived for DA (only the aggregated 8-param RMSE in `l96_joint_comparison.json`).*

---

## Normalized parameter RMSE (NRMSE) — S1 (single-sample)

Per-parameter NRMSE = `param_RMSE / mean(|true_param|)`, which normalizes away the scale difference between parameters (e.g. F~8 vs eps~0.1) so each competes equally. Mean is across the 8 params; lower is better.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.0590 ** | 0.2006 | 0.1799 | 0.5631 | 0.2668 | 0.1464 | 0.7822 | 0.7052 | 0.3629 |
| L8_joint_direct_unet_s0s1 | 0.0686 | 0.1376 ** | 0.1412 ** | 0.3036 ** | 0.1205 ** | 0.1216 ** | 0.2455 ** | 0.1492 ** | 0.1610 ** |
| L9_joint_cfm_s0s1_multitau | 0.5029 | 0.8030 | 0.5462 | 3.8702 | 0.6650 | 0.5169 | 3.7605 | 3.4642 | 1.7661 |
| Joint-ETKF | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Joint-EnKF | -- | -- | -- | -- | -- | -- | -- | -- | -- |

*Best per column (lowest NRMSE) is bolded. Joint-DA rows render as `--`: per-parameter NRMSE needs the true-parameter scale (`mean(|true|)`) which is not archived for DA (only the aggregated 8-param RMSE in `l96_joint_comparison.json`).*

---

## Parameter EV — S0 (single-sample)

Per-parameter explained variance `EV_p = 1 - mean((pred-true)^2)/var(true)` pooled over the 200 windows (computed offline from the stored eval arrays). Negative => parameter estimate is worse than a time-constant mean prediction. Note: `eps/w3/w4` have very small true variance, so even good absolute errors give large negative EV there — **NRMSE above is the fair cross-param metric**.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.8117 | -0.2044 ** | 0.5605 ** | -6.8554 | -0.1180 | -0.0806 | -19.3772 | -19.0373 | -5.5376 |
| L8_joint_direct_unet_s0s1 | 0.8363 ** | -0.3838 | -0.7497 | -2.3407 ** | 0.0789 ** | -0.0187 ** | -0.4329 ** | -0.4232 ** | -0.4292 ** |
| L9_joint_cfm_s0s1_multitau | -11.1602 | -13.9393 | -16.9556 | -912.6989 | -16.0882 | -13.6574 | -746.1907 | -880.5985 | -326.4111 |
| Joint-ETKF | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Joint-EnKF | -- | -- | -- | -- | -- | -- | -- | -- | -- |

*Best per column (highest EV) is bolded. Joint-DA rows render as `--`: per-parameter EV is not archived for the DA baselines (only the aggregated 8-param RMSE in `l96_joint_comparison.json`).*

---

## Parameter EV — S1 (single-sample)

Per-parameter explained variance `EV_p = 1 - mean((pred-true)^2)/var(true)` pooled over the 200 windows (computed offline from the stored eval arrays). Negative => parameter estimate is worse than a time-constant mean prediction. Note: `eps/w3/w4` have very small true variance, so even good absolute errors give large negative EV there — **NRMSE above is the fair cross-param metric**.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.7449 ** | -1.7121 | -1.4334 | -25.3987 | -4.5759 | -0.6460 | -41.0648 | -37.8105 | -13.9871 |
| L8_joint_direct_unet_s0s1 | 0.6548 | -0.2761 ** | -0.4983 ** | -6.6746 ** | -0.1366 ** | -0.1355 ** | -3.1434 ** | -0.7360 ** | -1.3682 ** |
| L9_joint_cfm_s0s1_multitau | -17.5492 | -42.4780 | -21.4313 | -1246.1711 | -33.6396 | -19.5286 | -971.1740 | -935.4407 | -410.9266 |
| Joint-ETKF | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Joint-EnKF | -- | -- | -- | -- | -- | -- | -- | -- | -- |

*Best per column (highest EV) is bolded. Joint-DA rows render as `--`: per-parameter EV is not archived for the DA baselines (only the aggregated 8-param RMSE in `l96_joint_comparison.json`).*

---

## Trajectory forecast skill — S0 (single-sample, 300-step)

State RMSE / EV between a short forecast rolled with the **estimated** parameters and one rolled with the **true** parameters, from the same initial state and forcing (L96 truth dynamics, 300-step horizon, observed subspace). This quantifies the sensitivity of short-term forecast quality to parameter estimation error; higher EV / lower RMSE is better.

| ID | RMSE slow | RMSE obs_fast | RMSE all | EV slow | EV obs_fast | EV all |
|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 2.3230 | 3.5622 | 3.1491 | -0.4802 | -3.6363 | -2.5842 |
| L8_joint_direct_unet_s0s1 | 0.1062 ** | 1.2183 ** | 0.8476 ** | 0.9969 ** | 0.4633 ** | 0.6412 ** |
| L9_joint_cfm_s0s1_multitau | 7.4986 | 11.6442 | 10.2623 | -14.3819 | -48.1541 | -36.8967 |
| Joint-ETKF | -- | -- | -- | -- | -- | -- |
| Joint-EnKF | -- | -- | -- | -- | -- | -- |

*Best per column is bolded (lowest RMSE, highest EV). Joint-DA rows render as `--`: the free forecast needs per-window predicted params (`x0`/`forcing` rollouts), which are not archived for DA.*

---

## Trajectory forecast skill — S1 (single-sample, 300-step)

State RMSE / EV between a short forecast rolled with the **estimated** parameters and one rolled with the **true** parameters, from the same initial state and forcing (L96 truth dynamics, 300-step horizon, observed subspace). This quantifies the sensitivity of short-term forecast quality to parameter estimation error; higher EV / lower RMSE is better.

| ID | RMSE slow | RMSE obs_fast | RMSE all | EV slow | EV obs_fast | EV all |
|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 1.6180 | 3.2675 | 2.7176 | 0.2201 | -2.8809 | -1.8472 |
| L8_joint_direct_unet_s0s1 | 0.1260 ** | 1.3259 ** | 0.9259 ** | 0.9956 ** | 0.3616 ** | 0.5730 ** |
| L9_joint_cfm_s0s1_multitau | 7.1136 | 10.0908 | 9.0984 | -13.1563 | -36.0132 | -28.3942 |
| Joint-ETKF | -- | -- | -- | -- | -- | -- |
| Joint-EnKF | -- | -- | -- | -- | -- | -- |

*Best per column is bolded (lowest RMSE, highest EV). Joint-DA rows render as `--`: the free forecast needs per-window predicted params (`x0`/`forcing` rollouts), which are not archived for DA.*

---

## Trajectory forecast skill — ens30 (n_members=30, k=1, 300-step)

Same parameter-sensitivity metric computed on the **member-mean** parameter estimates from the ens30 ensemble (300-step rollouts, observed subspace). Higher EV / lower RMSE is better.

| ID | S0 EV all | S0 RMSE all | S1 EV all | S1 RMSE all |
|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.6874 | 0.7831 | 0.3943 | 1.0964 ** |
| L8_joint_direct_unet_s0s1 | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | 0.7333 ** | 0.7375 ** | 0.4086 ** | 1.1100 |

*Best per column is bolded (highest EV, lowest RMSE). L8 is deterministic and not run as an ensemble → --.*

---

## Trajectory forecast skill — ens30 (n_members=30, k=10, 300-step)

Same parameter-sensitivity metric computed on the **member-mean** parameter estimates from the ens30 ensemble (300-step rollouts, observed subspace). Higher EV / lower RMSE is better.

| ID | S0 EV all | S0 RMSE all | S1 EV all | S1 RMSE all |
|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.6874 ** | 0.7831 ** | 0.3943 ** | 1.0964 ** |
| L8_joint_direct_unet_s0s1 | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | -120.6307 | 18.5879 | -119.5533 | 18.5818 |

*Best per column is bolded (highest EV, lowest RMSE). L8 is deterministic and not run as an ensemble → --.*

---

## DA baselines (joint)

Joint augmented-state DA filters (state **and** 8 params) benchmarked on the same cached S0/S1 test set, for direct comparison against the oracle-free neural rows above. Rows are read from `experiments/l96_joint_comparison.json`; missing methods render as --. For a per-parameter DA table (Joint-ETKF/EnKF/Strong-4DVar) see `l96_joint_da_benchmark.md`.

| Method | S0 RMSE | S0 ES | S1 RMSE | S1 ES |
|---|---|---|---|---|
| Joint-ETKF | 0.6334 | 0.2977 | 1.4971 | 0.9374 |
| Joint-EnKF | 0.7263 | 0.3709 | 1.4592 | 0.8434 |

*ES is the N=30 ensemble Energy Score for the filters; Joint-Strong-4DVar is a deterministic solve so its ES is the N=1 MAE proxy (marked per the DA report). Lower is better for RMSE and ES. Rows are read from `experiments/l96_joint_comparison.json`.*

---

## Consistency check

The eval script stores each run's predictions against the observed-subspace truth subsampled from the cached `true_state[:, obs_var_indices]`. When the numpy arrays are accessible (same `experiments/` dir), the report would recompute a metric from them and compare against the stored JSON to detect cache drift. Here we only assert the JSONs are internally consistent (one `s0`/`s1` entry per run).
