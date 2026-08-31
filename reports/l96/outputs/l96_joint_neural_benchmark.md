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
| L7_joint_cfm_s0s1 | -- | -- | -- |
| L8_joint_direct_unet_s0s1 | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | 0.6257 ** | 0.6313 ** | 1.0089 |

*Best per column is bolded (lowest RMSE; S1/S0 degradation >1 means worse on the parameter-biased S1 setup).*

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

## Parameter RMSE — S0 (single-sample)

Per-parameter RMSE (`F, c1, hx, eps, w1..w4`) and its mean across the 8 params.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L8_joint_direct_unet_s0s1 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | 0.1765 | 0.0803 | 0.0386 | 0.0118 | 0.0371 | 0.1054 | 0.0117 | 0.0118 | 0.0591 |

---

## Parameter RMSE — S1 (single-sample)

Per-parameter RMSE (`F, c1, hx, eps, w1..w4`) and its mean across the 8 params.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L8_joint_direct_unet_s0s1 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | 0.1765 | 0.0803 | 0.0386 | 0.0118 | 0.0371 | 0.1054 | 0.0117 | 0.0118 | 0.0591 |

---

## Normalized parameter RMSE (NRMSE) — S0 (single-sample)

Per-parameter NRMSE = `param_RMSE / mean(|true_param|)`, which normalizes away the scale difference between parameters (e.g. F~8 vs eps~0.1) so each competes equally. Mean is across the 8 params; lower is better.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L8_joint_direct_unet_s0s1 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | -- | -- | -- | -- | -- | -- | -- | -- | -- |

*Best per column (lowest NRMSE) is bolded.*

---

## Normalized parameter RMSE (NRMSE) — S1 (single-sample)

Per-parameter NRMSE = `param_RMSE / mean(|true_param|)`, which normalizes away the scale difference between parameters (e.g. F~8 vs eps~0.1) so each competes equally. Mean is across the 8 params; lower is better.

| ID | F | c1 | hx | eps | w1 | w2 | w3 | w4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L8_joint_direct_unet_s0s1 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | -- | -- | -- | -- | -- | -- | -- | -- | -- |

*Best per column (lowest NRMSE) is bolded.*

---

## Trajectory forecast skill — S0 (single-sample, 300-step)

State RMSE / EV between a short forecast rolled with the **estimated** parameters and one rolled with the **true** parameters, from the same initial state and forcing (L96 truth dynamics, 300-step horizon, observed subspace). This quantifies the sensitivity of short-term forecast quality to parameter estimation error; higher EV / lower RMSE is better.

| ID | RMSE slow | RMSE obs_fast | RMSE all | EV slow | EV obs_fast | EV all |
|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | -- | -- | -- | -- | -- | -- |
| L8_joint_direct_unet_s0s1 | -- | -- | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | -- | -- | -- | -- | -- | -- |

*Best per column is bolded (lowest RMSE, highest EV).*

---

## Trajectory forecast skill — S1 (single-sample, 300-step)

State RMSE / EV between a short forecast rolled with the **estimated** parameters and one rolled with the **true** parameters, from the same initial state and forcing (L96 truth dynamics, 300-step horizon, observed subspace). This quantifies the sensitivity of short-term forecast quality to parameter estimation error; higher EV / lower RMSE is better.

| ID | RMSE slow | RMSE obs_fast | RMSE all | EV slow | EV obs_fast | EV all |
|---|---|---|---|---|---|---|
| L7_joint_cfm_s0s1 | -- | -- | -- | -- | -- | -- |
| L8_joint_direct_unet_s0s1 | -- | -- | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | -- | -- | -- | -- | -- | -- |

*Best per column is bolded (lowest RMSE, highest EV).*

---

## Trajectory forecast skill — ens30 (n_members=30, k=1, 300-step)

Same parameter-sensitivity metric computed on the **member-mean** parameter estimates from the ens30 ensemble (300-step rollouts, observed subspace). Higher EV / lower RMSE is better.

| ID | S0 EV all | S0 RMSE all | S1 EV all | S1 RMSE all |
|---|---|---|---|---|
| L7_joint_cfm_s0s1 | -- | -- | -- | -- |
| L8_joint_direct_unet_s0s1 | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | -- | -- | -- | -- |

*Best per column is bolded (highest EV, lowest RMSE). L8 is deterministic and not run as an ensemble → --.*

---

## Trajectory forecast skill — ens30 (n_members=30, k=10, 300-step)

Same parameter-sensitivity metric computed on the **member-mean** parameter estimates from the ens30 ensemble (300-step rollouts, observed subspace). Higher EV / lower RMSE is better.

| ID | S0 EV all | S0 RMSE all | S1 EV all | S1 RMSE all |
|---|---|---|---|---|
| L7_joint_cfm_s0s1 | -- | -- | -- | -- |
| L8_joint_direct_unet_s0s1 | -- | -- | -- | -- |
| L9_joint_cfm_s0s1_multitau | -- | -- | -- | -- |

*Best per column is bolded (highest EV, lowest RMSE). L8 is deterministic and not run as an ensemble → --.*

---

## DA baselines (joint)

Joint augmented-state DA filters (state **and** 8 params) benchmarked on the same cached S0/S1 test set, vs the best neural joint estimator (L9 single-sample). Rows are read from `experiments/l96_joint_comparison.json`; missing methods render as --.

| Method | S0 RMSE | S0 ES | S1 RMSE | S1 ES |
|---|---|---|---|---|
| Joint-ETKF | 0.6334 | 0.2977 | 1.4971 | 0.9374 |
| Joint-EnKF | 0.7263 | 0.3709 | 1.4592 | 0.8434 |
| Joint-Strong-4DVar | 0.7122 | 0.4623 | 1.2001 | 0.8102 |

*ES is the N=30 ensemble Energy Score for the filters; Joint-Strong-4DVar is a deterministic solve so its ES is the N=1 MAE proxy (marked per the DA report). Lower is better for RMSE and ES. Rows are read from `experiments/l96_joint_comparison.json`.*

---

## Consistency check

The eval script stores each run's predictions against the observed-subspace truth subsampled from the cached `true_state[:, obs_var_indices]`. When the numpy arrays are accessible (same `experiments/` dir), the report would recompute a metric from them and compare against the stored JSON to detect cache drift. Here we only assert the JSONs are internally consistent (one `s0`/`s1` entry per run).
