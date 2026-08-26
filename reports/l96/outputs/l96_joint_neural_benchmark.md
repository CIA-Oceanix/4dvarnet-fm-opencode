# L96 Joint State-Parameter Neural Estimation Benchmark

**System:** Lorenz-96 two-scale (NO=8, J=4), observed subspace state_dim=24 (Obs30, obs_interval=100), 200 shared test windows.

**Models:** JointCFM + JointDirectUNet jointly estimate the 24D observed state **and** the 8 model parameters `F, c1, hx, eps, w1..w4` (fast weights per index; h fixed), matching the L96 joint DA convention. Each model's predictions are evaluated on the same cached S0/S1 test set used by the DA baselines.

---

## Benchmarked models

| ID | Type | τ mode | Description |
|---|---|---|---|
| L7_joint_cfm_s0s1 | JointCFM | tau=0 | Conditional flow matching (state + 8-param joint output) trained at tau=0 only; sampled with a single Euler step. Hidden [64,128,256], 400 epochs. |
| L8_joint_direct_unet_s0s1 | JointDirectUNet | n/a | Single-pass joint regression obs -> (state, 8 params). Deterministic. Hidden [64,128,256], 200 epochs. |
| L9_joint_cfm_s0s1_multitau | JointCFM | multi-tau | Standard multi-tau conditional flow matching (state + 8-param joint output); sampled as a 30-member ensemble with 10 Euler steps (ens30 x 10, N=30). Hidden [64,128,256], 400 epochs. |

---

## Single-sample results (n_members=1, k=1)

State RMSE over the observed subspace. S1/S0 is the degradation ratio (>1 means the model is worse on the parameter-biased S1 setup).

| ID | S0 RMSE | S1 RMSE | S1/S0 |
|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.6062 ** | 0.6620 | 1.0920 |
| L8_joint_direct_unet_s0s1 | 0.6096 | 0.6611 | 1.0846 |
| L9_joint_cfm_s0s1_multitau | 0.6257 | 0.6313 ** | 1.0089 |

*Best per column is bolded (lowest RMSE; S1/S0 degradation >1 means worse on the parameter-biased S1 setup).*

---

## Ensemble results (n_members=30, k=1)

State RMSE / explained variance (EV) / energy score (ES) over the observed subspace, computed on the member-mean trajectory; ES is the proper N=30 ensemble scoring rule.

| ID | S0 RMSE | S0 EV | S0 ES | S1 RMSE | S1 EV | S1 ES |
|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.6021 | 0.8624 | 0.3669 | 0.6579 | 0.8384 | 0.3969 |
| L8_joint_direct_unet_s0s1 | -- | -- | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | 0.6009 ** | 0.8642 ** | 0.3615 ** | 0.6037 ** | 0.8622 ** | 0.3636 ** |

*Only ens30 runs present on disk are shown; missing runs render as -- (L8 is deterministic and is not run as an ensemble). Best per column: lowest RMSE/ES, highest EV.*

---

## Ensemble results (n_members=30, k=10)

State RMSE / explained variance (EV) / energy score (ES) over the observed subspace, computed on the member-mean trajectory; ES is the proper N=30 ensemble scoring rule.

| ID | S0 RMSE | S0 EV | S0 ES | S1 RMSE | S1 EV | S1 ES |
|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.6021 | 0.8624 | 0.3669 | 0.6579 | 0.8384 | 0.3969 |
| L8_joint_direct_unet_s0s1 | -- | -- | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | 0.5251 ** | 0.8932 ** | 0.3115 ** | 0.5308 ** | 0.8903 ** | 0.3153 ** |

*Only ens30 runs present on disk are shown; missing runs render as -- (L8 is deterministic and is not run as an ensemble). Best per column: lowest RMSE/ES, highest EV.*

---

## Parameter RMSE — S0 (single-sample)

Per-parameter RMSE (`F, c1, hx, eps, w1..w4`) and its mean across the 8 params.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 6.3826 | 0.1338 | 0.2543 | 0.6942 | 0.4033 | 0.3776 | 0.7224 | 0.7296 | 1.2122 |
| L8_joint_direct_unet_s0s1 | 0.1295 | 0.1174 | 0.0165 | 0.0159 | 0.0837 | 0.0987 | 0.0149 | 0.0127 | 0.0611 |
| L9_joint_cfm_s0s1_multitau | 0.1765 | 0.0803 | 0.0386 | 0.0118 | 0.0371 | 0.1054 | 0.0117 | 0.0118 | 0.0591 |

---

## Parameter RMSE — S1 (single-sample)

Per-parameter RMSE (`F, c1, hx, eps, w1..w4`) and its mean across the 8 params.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 6.3826 | 0.1338 | 0.2543 | 0.6942 | 0.4033 | 0.3776 | 0.7224 | 0.7296 | 1.2122 |
| L8_joint_direct_unet_s0s1 | 0.1295 | 0.1174 | 0.0165 | 0.0159 | 0.0837 | 0.0987 | 0.0149 | 0.0127 | 0.0611 |
| L9_joint_cfm_s0s1_multitau | 0.1765 | 0.0803 | 0.0386 | 0.0118 | 0.0371 | 0.1054 | 0.0117 | 0.0118 | 0.0591 |

---

## Normalized parameter RMSE (NRMSE) — S0 (single-sample)

Per-parameter NRMSE = `param_RMSE / mean(|true_param|)`, which normalizes away the scale difference between parameters (e.g. F~8 vs eps~0.1) so each competes equally. Mean is across the 8 params; lower is better.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.7925 | 0.1357 | 0.2582 | 6.9495 | 0.3993 | 0.3748 | 7.1088 | 7.3396 | 2.9198 |
| L8_joint_direct_unet_s0s1 | 0.0161 ** | 0.1191 | 0.0167 ** | 0.1588 | 0.0829 | 0.0979 ** | 0.1467 | 0.1273 | 0.0957 |
| L9_joint_cfm_s0s1_multitau | 0.0219 | 0.0814 ** | 0.0392 | 0.1177 ** | 0.0367 ** | 0.1046 | 0.1149 ** | 0.1183 ** | 0.0793 ** |

*Best per column (lowest NRMSE) is bolded.*

---

## Normalized parameter RMSE (NRMSE) — S1 (single-sample)

Per-parameter NRMSE = `param_RMSE / mean(|true_param|)`, which normalizes away the scale difference between parameters (e.g. F~8 vs eps~0.1) so each competes equally. Mean is across the 8 params; lower is better.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 0.7920 | 0.1854 | 0.2493 | 6.7882 | 0.4397 | 0.3795 | 7.2024 | 7.4960 | 2.9416 |
| L8_joint_direct_unet_s0s1 | 0.0244 | 0.1198 | 0.0211 ** | 0.1548 | 0.0967 | 0.1128 ** | 0.1540 | 0.1324 | 0.1020 |
| L9_joint_cfm_s0s1_multitau | 0.0220 ** | 0.0731 ** | 0.0416 | 0.1162 ** | 0.0411 ** | 0.1232 | 0.1313 ** | 0.1195 ** | 0.0835 ** |

*Best per column (lowest NRMSE) is bolded.*

---

## Trajectory forecast skill — S0 (single-sample, 300-step)

State RMSE / EV between a short forecast rolled with the **estimated** parameters and one rolled with the **true** parameters, from the same initial state and forcing (L96 truth dynamics, 300-step horizon, observed subspace). This quantifies the sensitivity of short-term forecast quality to parameter estimation error; higher EV / lower RMSE is better.

| ID | RMSE slow | RMSE obs_fast | RMSE all | EV slow | EV obs_fast | EV all |
|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 1.1214 | 2.0796 | 1.7602 | 0.6561 | -0.5616 | -0.1557 |
| L8_joint_direct_unet_s0s1 | 0.0570 | 0.9728 | 0.6675 | 0.9991 | 0.6559 | 0.7703 |
| L9_joint_cfm_s0s1_multitau | 0.0552 ** | 0.7555 ** | 0.5220 ** | 0.9992 ** | 0.7935 ** | 0.8620 ** |

*Best per column is bolded (lowest RMSE, highest EV).*

---

## Trajectory forecast skill — S1 (single-sample, 300-step)

State RMSE / EV between a short forecast rolled with the **estimated** parameters and one rolled with the **true** parameters, from the same initial state and forcing (L96 truth dynamics, 300-step horizon, observed subspace). This quantifies the sensitivity of short-term forecast quality to parameter estimation error; higher EV / lower RMSE is better.

| ID | RMSE slow | RMSE obs_fast | RMSE all | EV slow | EV obs_fast | EV all |
|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | 1.1046 | 2.0956 | 1.7653 | 0.6638 | -0.5936 | -0.1745 |
| L8_joint_direct_unet_s0s1 | 0.0560 | 0.9650 | 0.6620 | 0.9991 | 0.6610 | 0.7737 |
| L9_joint_cfm_s0s1_multitau | 0.0536 ** | 0.7055 ** | 0.4882 ** | 0.9992 ** | 0.8194 ** | 0.8793 ** |

*Best per column is bolded (lowest RMSE, highest EV).*

---

## Trajectory forecast skill — ens30 (n_members=30, k=1, 300-step)

Same parameter-sensitivity metric computed on the **member-mean** parameter estimates from the ens30 ensemble (300-step rollouts, observed subspace). Higher EV / lower RMSE is better.

| ID | S0 EV all | S0 RMSE all | S1 EV all | S1 RMSE all |
|---|---|---|---|---|
| L7_joint_cfm_s0s1 | -0.1557 | 1.7603 | -0.1745 | 1.7652 |
| L8_joint_direct_unet_s0s1 | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | 0.8399 ** | 0.5596 ** | 0.8585 ** | 0.5254 ** |

*Best per column is bolded (highest EV, lowest RMSE). L8 is deterministic and not run as an ensemble → --.*

---

## Trajectory forecast skill — ens30 (n_members=30, k=10, 300-step)

Same parameter-sensitivity metric computed on the **member-mean** parameter estimates from the ens30 ensemble (300-step rollouts, observed subspace). Higher EV / lower RMSE is better.

| ID | S0 EV all | S0 RMSE all | S1 EV all | S1 RMSE all |
|---|---|---|---|---|
| L7_joint_cfm_s0s1 | -0.1557 | 1.7603 | -0.1745 | 1.7652 |
| L8_joint_direct_unet_s0s1 | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | 0.8658 ** | 0.5150 ** | 0.8818 ** | 0.4833 ** |

*Best per column is bolded (highest EV, lowest RMSE). L8 is deterministic and not run as an ensemble → --.*

---

## DA baselines (joint)

| Method | S0 RMSE | S1 RMSE |
|---|---|---|
| Joint-EnKF | -- | -- |
| Joint-ETKF | -- | -- |
| Joint-Strong-4DVar | -- | -- |

*Joint DA baselines have not been run successfully for this benchmark; their rows are deferred and shown as --. Once the joint DA regeneration completes, this report should add them apples-to-apples against the neural rows.*

---

## Consistency check

The eval script stores each run's predictions against the observed-subspace truth subsampled from the cached `true_state[:, obs_var_indices]`. When the numpy arrays are accessible (same `experiments/` dir), the report would recompute a metric from them and compare against the stored JSON to detect cache drift. Here we only assert the JSONs are internally consistent (one `s0`/`s1` entry per run).
