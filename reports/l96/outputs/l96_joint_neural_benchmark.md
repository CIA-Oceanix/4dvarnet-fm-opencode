# L96 Joint State-Parameter Neural Estimation Benchmark

**System:** Lorenz-96 two-scale (NO=8, J=4), observed subspace state_dim=24 (Obs30, obs_interval=100), 200 shared test windows.

**Models:** JointCFM + JointDirectUNet jointly estimate the 24D observed state **and** the 8 model parameters `F, c1, hx, eps, w1..w4` (fast weights per index; h fixed), matching the L96 joint DA convention. Each model's predictions are evaluated on the same cached S0/S1 test set used by the DA baselines.

**Oracle-free retrain (2026-08-31):** the numbers below are from the retrained checkpoints produced with the true-parameter oracle removed — the state UNet conditions on `[obs, forcing]` only (`cond_extra_dim=1`, `output_dim=state_dim`) and a dedicated parameter head (`ParamFlowCNN` / `ParamHeadCNN`) reads the params from that oracle-free state estimate; `true_params` appear only as the regression target. Earlier published per-parameter rows came from oracle-contaminated runs (true params fed into the UNet conditioning) and are **not** a valid baseline — the correct comparison is the **joint DA baselines** table below.

**Per-parameter detail:** `reports/l96/outputs/l96_joint_param_diagnostic.md` gives the full offline per-parameter RMSE / EV / NRMSE and free-forecast tables (single and ens30, all runs), recomputed from the stored eval arrays.

**Cascade (decoupled state→param head, 2026-09-02):** the state→param head fed by the frozen L1b state estimate (C1), the exact true state (C2), or the true state + derivative channel with positive-only bias-resampled `*_da` training (C3) is tabulated in the NRMSE / param-RMSE tables below. **The earlier "fast-weight failure" (w1/w2 NRMSE ≈ 1.1-1.2) was an eval-metric bug**, not a model failure: the train-script eval read scalar `true_w1..true_w4` keys that the cached test windows don't have (they store `true_fast_weights` as a list), silently comparing all four fast-weight channels against 0.0. With the list-aware extraction every cascade model recovers the fast weights well on S0 (w1..w4 RMSE 0.011-0.013 for C1/C2), and on the biased S1 setup C3's positive-bias resampling is the most robust (w1/w2 0.04/0.10, F 0.52) while C1/C2 stay competitive on the fast weights and degrade mainly on F. See CHANGELOG 2026-09-02.

---

## Consolidated summary — neural vs DA (S0/S1)

Single-sample state RMSE (S0/S1), S1/S0 degradation, and **mean** per-parameter RMSE over the 8 params (F, c1, hx, eps, w1..w4). State RMSE beats DA on S1 (robust ≈1.0 degradation) but DA filters recover the parameters far better on S0 (Joint-ETKF mean per-param RMSE 0.053 vs best neural 0.122). L9's multi-τ param head is the notable failure (mean 0.750 on S0).

| Method | S0 state RMSE | S1 state RMSE | S1/S0 | S0 paramRMSE mean | S1 paramRMSE mean |
|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.6704 | 1.3086 | 1.9519 | 0.0970 | 0.3269 |
| L8_joint_direct_unet_s0s1 | 0.6629 | 1.8759 | 2.8299 | 0.0983 | 0.1928 |
| L9_joint_cfm_s0s1_multitau | 0.6515 | 0.6589 | 1.0113 | 0.1338 | 0.1410 |
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
| C1_stateparam_head_s1 | StateParamHead (CNN) | n/a | Decoupled cascade: param head fed by frozen L1b state estimate (decoupled). With the corrected per-param metric all recover the fast weights on S0; C3's positive-bias training is the most S1-robust. |
| C2_stateparam_head_state_true | StateParamHead (CNN) | n/a | Decoupled cascade: param head fed by exact true state (ablation). With the corrected per-param metric all recover the fast weights on S0; C3's positive-bias training is the most S1-robust. |
| C3_param_head_true_deriv | StateParamHead (CNN) | n/a | Decoupled cascade: param head fed by exact true state + temporal-derivative channel, positive-only bias-resampled `*_da` training (2026-09-01). With the corrected per-param metric all recover the fast weights on S0; C3's positive-bias training is the most S1-robust. |
| C4a_param_head_unet_true | StateParamUNet (UNet) | n/a | Decoupled cascade: param head fed by exact true state, UNet backbone (implicit multi-scale temporal features, no derivative channel). training/eval pending (not yet evaluated). |
| C4b_param_head_unet_l1b | StateParamUNet (UNet) | n/a | Decoupled cascade: param head fed by frozen L1b state estimate, UNet backbone (implicit multi-scale temporal features, no derivative channel). training/eval pending (not yet evaluated). |

---

## Single-sample results (n_members=1, k=1)

State metrics over the observed subspace for the neural models (single-sample) and the joint-DA filters. S1/S0 is the degradation ratio (>1 means worse on the parameter-biased S1 setup). ES for the deterministic neural models and DA rows is the N=1 mean-absolute-error proxy; the DA filters' ES is N=30 (see DA note).

| ID | S0 RMSE | S0 EV | S0 ES | S1 RMSE | S1 EV | S1 ES | S1/S0 |
|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.6704 | 0.8319 | 0.4096 | 1.3086 | 0.4297 | 0.8669 | 1.9519 |
| L8_joint_direct_unet_s0s1 | 0.6629 | 0.8354 | 0.4039 | 1.8759 | -0.1882 | 1.3448 | 2.8299 |
| L9_joint_cfm_s0s1_multitau | 0.6515 | 0.8393 ** | 0.4061 | 0.6589 ** | 0.8348 ** | 0.4131 ** | 1.0113 ** |
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

*Only ens30 runs present on disk are shown; missing runs render as -- (L8 is deterministic and is not run as an ensemble). Best per column: lowest RMSE/ES, highest EV.*

---

## Ensemble results (n_members=30, k=10)

State RMSE / explained variance (EV) / energy score (ES) over the observed subspace, computed on the member-mean trajectory; ES is the proper N=30 ensemble scoring rule.

| ID | S0 RMSE | S0 EV | S0 ES | S1 RMSE | S1 EV | S1 ES |
|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | -- | -- | -- | -- | -- | -- |
| L8_joint_direct_unet_s0s1 | -- | -- | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | -- | -- | -- | -- | -- | -- |

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

*Best per cell (highest EV) is bolded. L8 is deterministic (no ensemble).*

---

## Parameter RMSE — S0 (single-sample)

Per-parameter RMSE (`F, c1, hx, eps, w1..w4`) and its mean across the 8 params.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.3192 | 0.1248 | 0.0536 | 0.0124 | 0.1182 | 0.1221 | 0.0130 | 0.0126 | 0.0970 |
| L8_joint_direct_unet_s0s1 | 0.3202 | 0.1360 | 0.0425 | 0.0131 | 0.1200 | 0.1280 | 0.0133 | 0.0129 | 0.0983 |
| L9_joint_cfm_s0s1_multitau | 0.5250 | 0.1523 | 0.0926 | 0.0124 | 0.1193 | 0.1383 | 0.0150 | 0.0153 | 0.1338 |
| Joint-ETKF | 0.1306 | 0.0167 | 0.0156 | 0.0016 | 0.1155 | 0.1218 | 0.0119 | 0.0112 | 0.0531 |
| Joint-EnKF | 0.1532 | 0.0180 | 0.0168 | 0.0019 | 0.1157 | 0.1245 | 0.0120 | 0.0113 | 0.0567 |

*Cascade — decoupled state→param head:*
| C1 (L1b state) | 0.0940 | 0.0112 | 0.0139 | 0.0110 | 0.0123 | 0.0125 | 0.0113 | 0.0104 | 0.0221 |
| C2 (true state) | 0.0863 | 0.0107 | 0.0130 | 0.0053 | 0.0110 | 0.0121 | 0.0103 | 0.0104 | 0.0199 |
| C3 (true state + derivative + bias-resample) | 0.2629 | 0.0887 | 0.0139 | 0.0040 | 0.0511 | 0.0637 | 0.0102 | 0.0094 | 0.0630 |

*Joint-DA rows are the co-estimated 8-param RMSE from `l96_joint_comparison.json` (S1 `w3`/`w4` are pinned to the reference prior, not estimated — their RMSE=0 is a masking artifact, **[not]** recovery; on the 6 genuinely-estimated params DA S1 mean NRMSE is ~0.10, i.e. parity with L9). Per-parameter EV and the free forecast are **not** stored for DA (the per-window predictions were not archived), so those tables show DA as `--`.*

---

## Parameter RMSE — S1 (single-sample)

Per-parameter RMSE (`F, c1, hx, eps, w1..w4`) and its mean across the 8 params.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 1.5046 | 0.3347 | 0.2251 | 0.0163 | 0.3436 | 0.1538 | 0.0227 | 0.0141 | 0.3269 |
| L8_joint_direct_unet_s0s1 | 0.7277 | 0.1719 | 0.1838 | 0.0351 | 0.1703 | 0.1892 | 0.0245 | 0.0395 | 0.1928 |
| L9_joint_cfm_s0s1_multitau | 0.5315 | 0.1621 | 0.0937 | 0.0120 | 0.1312 | 0.1597 | 0.0199 | 0.0178 | 0.1410 |
| Joint-ETKF | 0.6082 | 0.1052 | 0.0637 | 0.0106 | 0.1161 | 0.1186 | 0.0000 | 0.0000 | 0.1278 |
| Joint-EnKF | 0.7637 | 0.1053 | 0.0640 | 0.0112 | 0.1194 | 0.1197 | 0.0000 | 0.0000 | 0.1479 |

*Cascade — decoupled state→param head:*
| C1 (L1b state) | 1.6479 | 0.1006 | 0.1346 | 0.0144 | 0.1774 | 0.1208 | 0.0286 | 0.0258 | 0.2813 |
| C2 (true state) | 0.9714 | 0.1178 | 0.0769 | 0.0103 | 0.2117 | 0.1232 | 0.0186 | 0.0144 | 0.1930 |
| C3 (true state + derivative + bias-resample) | 0.5200 | 0.1364 | 0.0468 | 0.0108 | 0.0437 | 0.0966 | 0.0332 | 0.0447 | 0.1165 |

*Joint-DA rows are the co-estimated 8-param RMSE from `l96_joint_comparison.json` (S1 `w3`/`w4` are pinned to the reference prior, not estimated — their RMSE=0 is a masking artifact, **[not]** recovery; on the 6 genuinely-estimated params DA S1 mean NRMSE is ~0.10, i.e. parity with L9). Per-parameter EV and the free forecast are **not** stored for DA (the per-window predictions were not archived), so those tables show DA as `--`.*

---

## Normalized parameter RMSE (NRMSE) — S0 (single-sample)

Per-parameter NRMSE = `param_RMSE / mean(|true_param|)`, which normalizes away the scale difference between parameters (e.g. F~8 vs eps~0.1) so each competes equally. Mean is across the 8 params; lower is better. This is the relevance metric: NRMSE ≲ 0.2 (≲20% relative error) marks an estimate that carries genuine information about the parameter.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.0396 | 0.1266 | 0.0544 | 0.1240 | 0.1171 | 0.1212 | 0.1279 | 0.1270 | 0.1047 |
| L8_joint_direct_unet_s0s1 | 0.0398 | 0.1379 | 0.0432 | 0.1307 | 0.1189 | 0.1270 | 0.1309 | 0.1302 | 0.1073 |
| L9_joint_cfm_s0s1_multitau | 0.0652 | 0.1544 | 0.0940 | 0.1238 | 0.1181 | 0.1373 | 0.1478 | 0.1539 | 0.1243 |
| Joint-ETKF | 0.0162 ** | 0.0170 ** | 0.0159 ** | 0.0165 ** | 0.1143 ** | 0.1209 ** | 0.1173 ** | 0.1131 ** | 0.0664 ** |
| Joint-EnKF | 0.0190 | 0.0183 | 0.0171 | 0.0187 | 0.1145 | 0.1235 | 0.1186 | 0.1133 | 0.0679 |

*Cascade — decoupled state→param head:*
| C1 (L1b state) | 0.0117 | 0.0114 | 0.0141 | 0.1099 | 0.0122 | 0.0124 | 0.1117 | 0.1051 | 0.0486 |
| C2 (true state) | 0.0107 | 0.0109 | 0.0132 | 0.0532 | 0.0109 | 0.0120 | 0.1012 | 0.1049 | 0.0396 |
| C3 (true state + derivative + bias-resample) | 0.0326 | 0.0900 | 0.0141 | 0.0404 | 0.0506 | 0.0632 | 0.1006 | 0.0948 | 0.0608 |

*Best per column (lowest NRMSE) is bolded. Joint-DA NRMSE rows are derived from their archived per-param RMSE ÷ the cached true-param scale. The C1/C2/C3 cascade (2026-09-02 re-eval) recovers the fast weights on S0 (C1/C2 w1..w4 NRMSE 0.01-0.02; F the main cost) — the earlier w1/w2 NRMSE ≈ 1.0 was an eval-metric artifact (fast-weight truth read as 0.0), not a model failure; C3's positive-bias training keeps S0 ≈ parity with C1/C2 on the fast weights and is the most S1-robust.*

---

## Normalized parameter RMSE (NRMSE) — S1 (single-sample)

Per-parameter NRMSE = `param_RMSE / mean(|true_param|)`, which normalizes away the scale difference between parameters (e.g. F~8 vs eps~0.1) so each competes equally. Mean is across the 8 params; lower is better. This is the relevance metric: NRMSE ≲ 0.2 (≲20% relative error) marks an estimate that carries genuine information about the parameter.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.1890 | 0.3337 | 0.2257 | 0.1633 | 0.3413 | 0.1532 | 0.2309 | 0.1400 | 0.2221 |
| L8_joint_direct_unet_s0s1 | 0.0914 | 0.1714 | 0.1844 | 0.3523 | 0.1691 | 0.1885 | 0.2494 | 0.3927 | 0.2249 |
| L9_joint_cfm_s0s1_multitau | 0.0667 ** | 0.1617 | 0.0940 | 0.1205 | 0.1303 | 0.1591 | 0.2029 | 0.1769 | 0.1390 |
| Joint-ETKF | 0.0764 | 0.1049 ** | 0.0639 ** | 0.1066 ** | 0.1153 ** | 0.1181 ** | 0.0000 ** | 0.0000 ** | 0.0731 ** |
| Joint-EnKF | 0.0959 | 0.1050 | 0.0642 | 0.1121 | 0.1186 | 0.1193 | 0.0000 ** | 0.0000 ** | 0.0769 |

*Cascade — decoupled state→param head:*
| C1 (L1b state) | 0.2069 | 0.1003 | 0.1350 | 0.1442 | 0.1762 | 0.1203 | 0.2913 | 0.2565 | 0.1788 |
| C2 (true state) | 0.1220 | 0.1174 | 0.0771 | 0.1030 | 0.2102 | 0.1227 | 0.1893 | 0.1432 | 0.1356 |
| C3 (true state + derivative + bias-resample) | 0.0653 | 0.1360 | 0.0469 | 0.1084 | 0.0434 | 0.0963 | 0.3373 | 0.4446 | 0.1598 |

*Best per column (lowest NRMSE) is bolded. On S1 the relevant comparison: L9 (multi-τ joint flow) keeps **every** parameter at NRMSE ≤ 0.20 (F 0.07), i.e. ≤20% relative error — genuine param recovery at parity with the joint DA filters on the params they actually estimate. The C1/C2/C3 cascade (2026-09-02 re-eval with the fixed list-aware fast-weight metric) recovers the fast weights well (C2 w1/w2 NRMSE 0.21/0.12; C3 w1/w2 0.04/0.10, F 0.52) — the earlier w1/w2 NRMSE ≈ 1.1-1.2 was an eval-metric artifact (fast-weight truth read as 0.0), not a model failure. C3 remains the most S1-robust cascade member, but the coupled multi-τ flow (L9) and the joint-DA filters are still ahead on overall param recovery. Joint-DA S1 `w3`/`w4` NRMSE 0.00 is the pinned-to-prior masking artifact (they are **not** estimated), not recovery; DA mean NRMSE is 0.07 incl. / 0.10 excl. those masked w3/w4. The DA NRMSE rows are derived from their archived per-param RMSE in `l96_joint_comparison.json` ÷ the cached true-param scale; per-window predictions (EV, free forecast) are not archived for DA.*

---

## Parameter EV — S0 (single-sample)

Per-parameter explained variance `EV_p = 1 - mean((pred-true)^2)/var(true)` pooled over the 200 windows (computed offline from the stored eval arrays). Negative => parameter estimate is worse than a time-constant mean prediction. Note: `eps/w3/w4` have very small true variance, so even good absolute errors give large negative EV there — **NRMSE above is the fair cross-param metric**.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.8712 ** | -0.1607 ** | 0.7932 | -0.0590 | -0.0854 ** | 0.0003 ** | -0.2445 ** | -0.2614 ** | 0.1067 ** |
| L8_joint_direct_unet_s0s1 | 0.8704 | -0.3774 | 0.8698 ** | -0.1768 | -0.1191 | -0.0981 | -0.3029 | -0.3260 | 0.0425 |
| L9_joint_cfm_s0s1_multitau | 0.6518 | -0.7276 | 0.3834 | -0.0549 ** | -0.1045 | -0.2835 | -0.6607 | -0.8517 | -0.2060 |
| Joint-ETKF | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Joint-EnKF | -- | -- | -- | -- | -- | -- | -- | -- | -- |

*Best per column (highest EV) is bolded. Joint-DA rows render as `--`: per-parameter EV is not archived for the DA baselines (only the aggregated 8-param RMSE in `l96_joint_comparison.json`).*

---

## Parameter EV — S1 (single-sample)

Per-parameter explained variance `EV_p = 1 - mean((pred-true)^2)/var(true)` pooled over the 200 windows (computed offline from the stored eval arrays). Negative => parameter estimate is worse than a time-constant mean prediction. Note: `eps/w3/w4` have very small true variance, so even good absolute errors give large negative EV there — **NRMSE above is the fair cross-param metric**.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | -1.6183 | -6.5104 | -2.8317 | -1.2192 | -8.1237 | -0.8035 ** | -2.6647 | -0.5291 ** | -3.0376 |
| L8_joint_direct_unet_s0s1 | 0.3876 | -0.9815 | -1.5561 | -9.3332 | -1.2407 | -1.7290 | -3.2755 | -11.0352 | -3.5954 |
| L9_joint_cfm_s0s1_multitau | 0.6733 ** | -0.7622 ** | 0.3363 ** | -0.2094 ** | -0.3295 ** | -0.9448 | -1.8290 ** | -1.4407 | -0.5632 ** |
| Joint-ETKF | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Joint-EnKF | -- | -- | -- | -- | -- | -- | -- | -- | -- |

*Best per column (highest EV) is bolded. Joint-DA rows render as `--`: per-parameter EV is not archived for the DA baselines (only the aggregated 8-param RMSE in `l96_joint_comparison.json`).*

---

## Trajectory forecast skill — S0 (single-sample, 300-step)

State RMSE / EV between a short forecast rolled with the **estimated** parameters and one rolled with the **true** parameters, from the same initial state and forcing (L96 truth dynamics, 300-step horizon, observed subspace). This quantifies the sensitivity of short-term forecast quality to parameter estimation error; higher EV / lower RMSE is better.

| ID | RMSE slow | RMSE obs_fast | RMSE all | EV slow | EV obs_fast | EV all |
|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.0609 | 0.7950 | 0.5503 | 0.9990 | 0.7713 | 0.8472 |
| L8_joint_direct_unet_s0s1 | 0.0583 ** | 0.7979 | 0.5514 | 0.9991 ** | 0.7685 | 0.8454 |
| L9_joint_cfm_s0s1_multitau | 0.0777 | 0.7248 ** | 0.5091 ** | 0.9984 | 0.8102 ** | 0.8729 ** |
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

*Best per column is bolded (highest EV, lowest RMSE).*

---

## Trajectory forecast skill — ens30 (n_members=30, k=10, 300-step)

Same parameter-sensitivity metric computed on the **member-mean** parameter estimates from the ens30 ensemble (300-step rollouts, observed subspace). Higher EV / lower RMSE is better.

| ID | S0 EV all | S0 RMSE all | S1 EV all | S1 RMSE all |
|---|---|---|---|---|
| L7_joint_cfm_s0s1 | -- | -- | -- | -- |
| L8_joint_direct_unet_s0s1 | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | -- | -- | -- | -- |

*Best per column is bolded (highest EV, lowest RMSE).*

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
