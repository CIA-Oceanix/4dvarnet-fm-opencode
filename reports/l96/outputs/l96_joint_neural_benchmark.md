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
| L7_joint_cfm_s0s1 | 0.6704 | 1.3086 | 1.9519 | 0.0970 | 0.3269 |
| L8_joint_direct_unet_s0s1 | 0.6629 | 1.8759 | 2.8299 | 0.0983 | 0.1928 |
| L9_joint_cfm_s0s1_multitau | 0.6515 | 0.6589 | 1.0113 | 0.1338 | 0.1410 |
| L10_joint_cfm_coupled_multitau | 0.6511 | 0.6536 | 1.0038 | 0.1166 | 0.1799 |
| L12_joint_direct_unet_unethead | 0.6659 | 1.5510 | 2.3291 | 0.0965 | 0.3424 |
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
| L10_joint_cfm_coupled_multitau | JointCFMCoupled | multi-tau | Coupled joint conditional flow: BOTH x_tau=(1-tau)x0+tau*x1 and theta_tau=(1-tau)theta0+tau*theta1 condition both velocity fields u_theta(x_tau,theta_tau,tau,obs,forcing) and v_phi(...) -> (theta1-theta0). UNet param flow [32,64,128], state [64,128,256], 400 epochs. |
| L12_joint_direct_unet_unethead | JointDirectUNet | n/a | JointDirectUNet (deterministic) with a UNet param head (param_head_backbone=unet) regressing 8 params from [obs, forcing, x_hat_state] (stop-grad), attention-pooled. State [64,128,256], param head [32,64,128], 200 epochs. |

---

## Single-sample results (n_members=1, k=1)

State metrics over the observed subspace for the neural models (single-sample) and the joint-DA filters. S1/S0 is the degradation ratio (>1 means worse on the parameter-biased S1 setup). ES for the deterministic neural models and DA rows is the N=1 mean-absolute-error proxy; the DA filters' ES is N=30 (see DA note).

| ID | S0 RMSE | S0 EV | S0 ES | S1 RMSE | S1 EV | S1 ES | S1/S0 |
|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.6704 | 0.8319 | 0.4096 | 1.3086 | 0.4297 | 0.8669 | 1.9519 |
| L8_joint_direct_unet_s0s1 | 0.6629 | 0.8354 | 0.4039 | 1.8759 | -0.1882 | 1.3448 | 2.8299 |
| L9_joint_cfm_s0s1_multitau | 0.6515 | 0.8393 | 0.4061 | 0.6589 | 0.8348 | 0.4131 ** | 1.0113 |
| L10_joint_cfm_coupled_multitau | 0.6511 | 0.8395 ** | 0.4155 | 0.6536 ** | 0.8375 ** | 0.4191 | 1.0038 ** |
| L12_joint_direct_unet_unethead | 0.6659 | 0.8338 | 0.4054 | 1.5510 | 0.1827 | 1.0286 | 2.3291 |
| Joint-ETKF | 0.6334 ** | 0.8213 | 0.2977 ** | 1.4971 | 0.1819 | 0.9374 | 2.3638 |
| Joint-EnKF | 0.7263 | 0.7742 | 0.3709 | 1.4592 | 0.2302 | 0.8434 | 2.0091 |

*Best per column: lowest RMSE / ES / degradation, highest EV. The joint-DA rows (Joint-ETKF / Joint-EnKF) come from `l96_joint_comparison.json`; their ES is the N=30 ensemble score while the neural single-sample ES is an N=1 MAE proxy (not strictly comparable, flagged).*

---

## Ensemble results (n_members=30, k=1)

State RMSE / explained variance (EV) / energy score (ES) over the observed subspace, computed on the member-mean trajectory; ES is the proper N=30 ensemble scoring rule.

| ID | S0 RMSE | S0 EV | S0 ES | S1 RMSE | S1 EV | S1 ES |
|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | -- | -- | -- | -- | -- | -- |
| L8_joint_direct_unet_s0s1 | -- | -- | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | -- | -- | -- | -- | -- | -- |
| L10_joint_cfm_coupled_multitau | -- | -- | -- | -- | -- | -- |
| L12_joint_direct_unet_unethead | -- | -- | -- | -- | -- | -- |

*Only ens30 runs present on disk are shown; missing runs render as -- (L8 is deterministic and is not run as an ensemble). Best per column: lowest RMSE/ES, highest EV.*

---

## Ensemble results (n_members=30, k=10)

State RMSE / explained variance (EV) / energy score (ES) over the observed subspace, computed on the member-mean trajectory; ES is the proper N=30 ensemble scoring rule.

| ID | S0 RMSE | S0 EV | S0 ES | S1 RMSE | S1 EV | S1 ES |
|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | -- | -- | -- | -- | -- | -- |
| L8_joint_direct_unet_s0s1 | -- | -- | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | -- | -- | -- | -- | -- | -- |
| L10_joint_cfm_coupled_multitau | -- | -- | -- | -- | -- | -- |
| L12_joint_direct_unet_unethead | -- | -- | -- | -- | -- | -- |

*Only ens30 runs present on disk are shown; missing runs render as -- (L8 is deterministic and is not run as an ensemble). Best per column: lowest RMSE/ES, highest EV.*

---

## Parameter EV — ens30 (n_members=30, k=1)

Per-parameter explained variance from the **member-mean** parameters of the 30-member ensemble (offline from the stored eval arrays). Deep integration (k=10) of the multi-tau JointCFM parameter head can **collapse** the EV (hugely negative) even when the ensemble-mean state improves.

| ID | Case | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | S0 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L7_joint_cfm_s0s1 | S1 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L8_joint_direct_unet_s0s1 | S0 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L8_joint_direct_unet_s0s1 | S1 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | S0 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | S1 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L10_joint_cfm_coupled_multitau | S0 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L10_joint_cfm_coupled_multitau | S1 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L12_joint_direct_unet_unethead | S0 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L12_joint_direct_unet_unethead | S1 | -- | -- | -- | -- | -- | -- | -- | -- | -- |

*Best per cell (highest EV) is bolded. L8 is deterministic (no ensemble).*

---

## Parameter EV — ens30 (n_members=30, k=10)

Per-parameter explained variance from the **member-mean** parameters of the 30-member ensemble (offline from the stored eval arrays). Deep integration (k=10) of the multi-tau JointCFM parameter head can **collapse** the EV (hugely negative) even when the ensemble-mean state improves.

| ID | Case | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | S0 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L7_joint_cfm_s0s1 | S1 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L8_joint_direct_unet_s0s1 | S0 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L8_joint_direct_unet_s0s1 | S1 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | S0 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | S1 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L10_joint_cfm_coupled_multitau | S0 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L10_joint_cfm_coupled_multitau | S1 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L12_joint_direct_unet_unethead | S0 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L12_joint_direct_unet_unethead | S1 | -- | -- | -- | -- | -- | -- | -- | -- | -- |

*Best per cell (highest EV) is bolded. L8 is deterministic (no ensemble).*

---

## Parameter RMSE — S0 (single-sample)

Per-parameter RMSE (`F, c1, hx, eps, w1..w4`) and its mean across the 8 params.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.3192 | 0.1248 | 0.0536 | 0.0124 | 0.1182 | 0.1221 | 0.0130 | 0.0126 | 0.0970 |
| L8_joint_direct_unet_s0s1 | 0.3202 | 0.1360 | 0.0425 | 0.0131 | 0.1200 | 0.1280 | 0.0133 | 0.0129 | 0.0983 |
| L9_joint_cfm_s0s1_multitau | 0.5250 | 0.1523 | 0.0926 | 0.0124 | 0.1193 | 0.1383 | 0.0150 | 0.0153 | 0.1338 |
| L10_joint_cfm_coupled_multitau | 0.4026 | 0.1513 | 0.0736 | 0.0125 | 0.1303 | 0.1332 | 0.0143 | 0.0147 | 0.1166 |
| L12_joint_direct_unet_unethead | 0.3165 | 0.1341 | 0.0464 | 0.0122 | 0.1166 | 0.1200 | 0.0127 | 0.0133 | 0.0965 |
| Joint-ETKF | 0.1306 | 0.0167 | 0.0156 | 0.0016 | 0.1155 | 0.1218 | 0.0119 | 0.0112 | 0.0531 |
| Joint-EnKF | 0.1532 | 0.0180 | 0.0168 | 0.0019 | 0.1157 | 0.1245 | 0.0120 | 0.0113 | 0.0567 |

*Joint-DA rows are the co-estimated 8-param RMSE from `l96_joint_comparison.json` (S1 `w3`/`w4` are pinned to the reference prior, not estimated). Per-parameter EV and the free forecast are **not** stored for DA (the per-window predictions were not archived), so those tables show DA as `--`.*

---

## Parameter RMSE — S1 (single-sample)

Per-parameter RMSE (`F, c1, hx, eps, w1..w4`) and its mean across the 8 params.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 1.5046 | 0.3347 | 0.2251 | 0.0163 | 0.3436 | 0.1538 | 0.0227 | 0.0141 | 0.3269 |
| L8_joint_direct_unet_s0s1 | 0.7277 | 0.1719 | 0.1838 | 0.0351 | 0.1703 | 0.1892 | 0.0245 | 0.0395 | 0.1928 |
| L9_joint_cfm_s0s1_multitau | 0.5315 | 0.1621 | 0.0937 | 0.0120 | 0.1312 | 0.1597 | 0.0199 | 0.0178 | 0.1410 |
| L10_joint_cfm_coupled_multitau | 0.5925 | 0.3546 | 0.0894 | 0.0192 | 0.2059 | 0.1445 | 0.0150 | 0.0178 | 0.1799 |
| L12_joint_direct_unet_unethead | 1.0941 | 0.4951 | 0.5040 | 0.0213 | 0.2468 | 0.3089 | 0.0390 | 0.0301 | 0.3424 |
| Joint-ETKF | 0.6082 | 0.1052 | 0.0637 | 0.0106 | 0.1161 | 0.1186 | 0.0000 | 0.0000 | 0.1278 |
| Joint-EnKF | 0.7637 | 0.1053 | 0.0640 | 0.0112 | 0.1194 | 0.1197 | 0.0000 | 0.0000 | 0.1479 |

*Joint-DA rows are the co-estimated 8-param RMSE from `l96_joint_comparison.json` (S1 `w3`/`w4` are pinned to the reference prior, not estimated). Per-parameter EV and the free forecast are **not** stored for DA (the per-window predictions were not archived), so those tables show DA as `--`.*

---

## Normalized parameter RMSE (NRMSE) — S0 (single-sample)

Per-parameter NRMSE = `param_RMSE / mean(|true_param|)`, which normalizes away the scale difference between parameters (e.g. F~8 vs eps~0.1) so each competes equally. Mean is across the 8 params; lower is better.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.0396 | 0.1266 ** | 0.0544 | 0.1240 | 0.1171 | 0.1212 | 0.1279 | 0.1270 ** | 0.1047 |
| L8_joint_direct_unet_s0s1 | 0.0398 | 0.1379 | 0.0432 ** | 0.1307 | 0.1189 | 0.1270 | 0.1309 | 0.1302 | 0.1073 |
| L9_joint_cfm_s0s1_multitau | 0.0652 | 0.1544 | 0.0940 | 0.1238 | 0.1181 | 0.1373 | 0.1478 | 0.1539 | 0.1243 |
| L10_joint_cfm_coupled_multitau | 0.0500 | 0.1535 | 0.0747 | 0.1247 | 0.1290 | 0.1322 | 0.1409 | 0.1480 | 0.1191 |
| L12_joint_direct_unet_unethead | 0.0393 ** | 0.1360 | 0.0471 | 0.1220 ** | 0.1155 ** | 0.1191 ** | 0.1253 ** | 0.1334 | 0.1047 ** |
| Joint-ETKF | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Joint-EnKF | -- | -- | -- | -- | -- | -- | -- | -- | -- |

*Best per column (lowest NRMSE) is bolded. Joint-DA rows render as `--`: per-parameter NRMSE needs the true-parameter scale (`mean(|true|)`) which is not archived for DA (only the aggregated 8-param RMSE in `l96_joint_comparison.json`).*

---

## Normalized parameter RMSE (NRMSE) — S1 (single-sample)

Per-parameter NRMSE = `param_RMSE / mean(|true_param|)`, which normalizes away the scale difference between parameters (e.g. F~8 vs eps~0.1) so each competes equally. Mean is across the 8 params; lower is better.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.1890 | 0.3337 | 0.2257 | 0.1633 | 0.3413 | 0.1532 | 0.2309 | 0.1400 ** | 0.2221 |
| L8_joint_direct_unet_s0s1 | 0.0914 | 0.1714 | 0.1844 | 0.3523 | 0.1691 | 0.1885 | 0.2494 | 0.3927 | 0.2249 |
| L9_joint_cfm_s0s1_multitau | 0.0667 ** | 0.1617 ** | 0.0940 | 0.1205 ** | 0.1303 ** | 0.1591 | 0.2029 | 0.1769 | 0.1390 ** |
| L10_joint_cfm_coupled_multitau | 0.0744 | 0.3535 | 0.0896 ** | 0.1930 | 0.2045 | 0.1439 ** | 0.1525 ** | 0.1772 | 0.1736 |
| L12_joint_direct_unet_unethead | 0.1374 | 0.4936 | 0.5055 | 0.2137 | 0.2451 | 0.3077 | 0.3970 | 0.2992 | 0.3249 |
| Joint-ETKF | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Joint-EnKF | -- | -- | -- | -- | -- | -- | -- | -- | -- |

*Best per column (lowest NRMSE) is bolded. Joint-DA rows render as `--`: per-parameter NRMSE needs the true-parameter scale (`mean(|true|)`) which is not archived for DA (only the aggregated 8-param RMSE in `l96_joint_comparison.json`).*

---

## Parameter EV — S0 (single-sample)

Per-parameter explained variance `EV_p = 1 - mean((pred-true)^2)/var(true)` pooled over the 200 windows (computed offline from the stored eval arrays). Negative => parameter estimate is worse than a time-constant mean prediction. Note: `eps/w3/w4` have very small true variance, so even good absolute errors give large negative EV there — **NRMSE above is the fair cross-param metric**.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.8712 | -0.1607 ** | 0.7932 | -0.0590 | -0.0854 | 0.0003 | -0.2445 | -0.2614 ** | 0.1067 ** |
| L8_joint_direct_unet_s0s1 | 0.8704 | -0.3774 | 0.8698 ** | -0.1768 | -0.1191 | -0.0981 | -0.3029 | -0.3260 | 0.0425 |
| L9_joint_cfm_s0s1_multitau | 0.6518 | -0.7276 | 0.3834 | -0.0549 | -0.1045 | -0.2835 | -0.6607 | -0.8517 | -0.2060 |
| L10_joint_cfm_coupled_multitau | 0.7952 | -0.7069 | 0.6103 | -0.0706 | -0.3188 | -0.1902 | -0.5098 | -0.7124 | -0.1379 |
| L12_joint_direct_unet_unethead | 0.8735 ** | -0.3398 | 0.8451 | -0.0249 ** | -0.0565 ** | 0.0340 ** | -0.1937 ** | -0.3912 | 0.0933 |
| Joint-ETKF | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Joint-EnKF | -- | -- | -- | -- | -- | -- | -- | -- | -- |

*Best per column (highest EV) is bolded. Joint-DA rows render as `--`: per-parameter EV is not archived for the DA baselines (only the aggregated 8-param RMSE in `l96_joint_comparison.json`).*

---

## Parameter EV — S1 (single-sample)

Per-parameter explained variance `EV_p = 1 - mean((pred-true)^2)/var(true)` pooled over the 200 windows (computed offline from the stored eval arrays). Negative => parameter estimate is worse than a time-constant mean prediction. Note: `eps/w3/w4` have very small true variance, so even good absolute errors give large negative EV there — **NRMSE above is the fair cross-param metric**.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | -1.6183 | -6.5104 | -2.8317 | -1.2192 | -8.1237 | -0.8035 | -2.6647 | -0.5291 ** | -3.0376 |
| L8_joint_direct_unet_s0s1 | 0.3876 | -0.9815 | -1.5561 | -9.3332 | -1.2407 | -1.7290 | -3.2755 | -11.0352 | -3.5954 |
| L9_joint_cfm_s0s1_multitau | 0.6733 ** | -0.7622 ** | 0.3363 | -0.2094 ** | -0.3295 ** | -0.9448 | -1.8290 | -1.4407 | -0.5632 ** |
| L10_joint_cfm_coupled_multitau | 0.5940 | -7.4268 | 0.3959 ** | -2.1018 | -2.2766 | -0.5912 ** | -0.5986 ** | -1.4499 | -1.6819 |
| L12_joint_direct_unet_unethead | -0.3845 | -15.4296 | -18.2110 | -2.8033 | -3.7055 | -6.2752 | -9.8349 | -5.9872 | -7.8289 |
| Joint-ETKF | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Joint-EnKF | -- | -- | -- | -- | -- | -- | -- | -- | -- |

*Best per column (highest EV) is bolded. Joint-DA rows render as `--`: per-parameter EV is not archived for the DA baselines (only the aggregated 8-param RMSE in `l96_joint_comparison.json`).*

---

## Trajectory forecast skill — S0 (single-sample, 300-step)

State RMSE / EV between a short forecast rolled with the **estimated** parameters and one rolled with the **true** parameters, from the same initial state and forcing (L96 truth dynamics, 300-step horizon, observed subspace). This quantifies the sensitivity of short-term forecast quality to parameter estimation error; higher EV / lower RMSE is better.

| ID | RMSE slow | RMSE obs_fast | RMSE all | EV slow | EV obs_fast | EV all |
|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.0609 | 0.7950 | 0.5503 | 0.9990 | 0.7713 | 0.8472 |
| L8_joint_direct_unet_s0s1 | 0.0583 | 0.7979 | 0.5514 | 0.9991 | 0.7685 | 0.8454 |
| L9_joint_cfm_s0s1_multitau | 0.0777 | 0.7248 ** | 0.5091 ** | 0.9984 | 0.8102 ** | 0.8729 ** |
| L10_joint_cfm_coupled_multitau | 0.0747 | 0.8110 | 0.5655 | 0.9985 | 0.7616 | 0.8406 |
| L12_joint_direct_unet_unethead | 0.0569 ** | 0.8057 | 0.5561 | 0.9991 ** | 0.7650 | 0.8430 |
| Joint-ETKF | -- | -- | -- | -- | -- | -- |
| Joint-EnKF | -- | -- | -- | -- | -- | -- |

*Best per column is bolded (lowest RMSE, highest EV). Joint-DA rows render as `--`: the free forecast needs per-window predicted params (`x0`/`forcing` rollouts), which are not archived for DA.*

---

## Trajectory forecast skill — S1 (single-sample, 300-step)

State RMSE / EV between a short forecast rolled with the **estimated** parameters and one rolled with the **true** parameters, from the same initial state and forcing (L96 truth dynamics, 300-step horizon, observed subspace). This quantifies the sensitivity of short-term forecast quality to parameter estimation error; higher EV / lower RMSE is better.

| ID | RMSE slow | RMSE obs_fast | RMSE all | EV slow | EV obs_fast | EV all |
|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.2039 | 0.8678 | 0.6465 | 0.9885 | 0.7262 | 0.8136 |
| L8_joint_direct_unet_s0s1 | 0.2114 | 1.7330 | 1.2258 | 0.9877 | -0.0902 | 0.2691 |
| L9_joint_cfm_s0s1_multitau | 0.0806 ** | 0.7698 ** | 0.5401 ** | 0.9982 ** | 0.7847 ** | 0.8558 ** |
| L10_joint_cfm_coupled_multitau | 0.1091 | 0.9854 | 0.6933 | 0.9967 | 0.6471 | 0.7637 |
| L12_joint_direct_unet_unethead | 0.1734 | 1.3783 | 0.9766 | 0.9917 | 0.3090 | 0.5366 |
| Joint-ETKF | -- | -- | -- | -- | -- | -- |
| Joint-EnKF | -- | -- | -- | -- | -- | -- |

*Best per column is bolded (lowest RMSE, highest EV). Joint-DA rows render as `--`: the free forecast needs per-window predicted params (`x0`/`forcing` rollouts), which are not archived for DA.*

---

## Trajectory forecast skill — ens30 (n_members=30, k=1, 300-step)

Same parameter-sensitivity metric computed on the **member-mean** parameter estimates from the ens30 ensemble (300-step rollouts, observed subspace). Higher EV / lower RMSE is better.

| ID | S0 EV all | S0 RMSE all | S1 EV all | S1 RMSE all |
|---|---|---|---|---|
| L7_joint_cfm_s0s1 | -- | -- | -- | -- |
| L8_joint_direct_unet_s0s1 | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | -- | -- | -- | -- |
| L10_joint_cfm_coupled_multitau | -- | -- | -- | -- |
| L12_joint_direct_unet_unethead | -- | -- | -- | -- |

*Best per column is bolded (highest EV, lowest RMSE). L8 is deterministic and not run as an ensemble → --.*

---

## Trajectory forecast skill — ens30 (n_members=30, k=10, 300-step)

Same parameter-sensitivity metric computed on the **member-mean** parameter estimates from the ens30 ensemble (300-step rollouts, observed subspace). Higher EV / lower RMSE is better.

| ID | S0 EV all | S0 RMSE all | S1 EV all | S1 RMSE all |
|---|---|---|---|---|
| L7_joint_cfm_s0s1 | -- | -- | -- | -- |
| L8_joint_direct_unet_s0s1 | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | -- | -- | -- | -- |
| L10_joint_cfm_coupled_multitau | -- | -- | -- | -- |
| L12_joint_direct_unet_unethead | -- | -- | -- | -- |

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
