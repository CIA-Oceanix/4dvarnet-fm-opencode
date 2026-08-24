# Joint State-Parameter Estimation: Vanilla vs Joint Baseline Results

**Date:** 2026-07-02  
**Branch:** `exp/cs4b-randombias` → porting to `feature/joint-estimation`  
**Data:** CS3 / CS4 datasets (randomized Lorenz-63 parameters, param_noise=0.2)  
**Settings:** da_window_steps=50, batch_size=200, opt_steps=200 (4DVar methods)

---

## 1. State RMSE: Vanilla vs Joint

| Method | CS3 Vanilla | CS3 Joint | Ratio | CS4 Vanilla | CS4 Joint | Ratio |
|--------|:----------:|:---------:|:----:|:----------:|:---------:|:----:|
| Weak-4DVar | 0.71 | 0.95 | 1.35× | 0.81 | 0.96 | 1.26× |
| Strong-4DVar | 0.68 | 1.04 | 1.54× | 0.79 | 1.34 | 1.71× |
| EnKF | 1.26 | **1.04** | **0.82×** | 2.48 | **1.24** | **0.49×** |
| ETKF | 1.15 | 1.36 | 1.17× | 2.17 | 2.01 | 0.93× |

## 2. Parameter RMSE (σ / ρ / β)

| Method | CS3 | CS4 |
|--------|:---:|:---:|
| Joint-Weak-4DVar | 2.24 / 1.95 / 0.33 | 2.66 / 1.72 / 0.30 |
| Joint-Strong-4DVar | 15.05 / 3.66 / 0.44 | 13.39 / 3.56 / 0.61 |
| Joint-EnKF | **0.64 / 0.50 / 0.11** | **0.94 / 0.78 / 0.19** |
| Joint-ETKF | 0.40 / 0.58 / 0.12 | 1.05 / 0.83 / 0.18 |

## 3. Key Findings

1. **EnKF is the best joint estimation method** — the 6D ensemble handles state-parameter coupling naturally. Joint estimation *improves* state RMSE (0.82× CS3, 0.49× CS4) compared to the vanilla 3D EnKF.

2. **All 4DVar methods degrade 1.3–1.7×** when estimating parameters jointly. The extra degrees of freedom (params + model error q) make the optimization landscape harder.

3. **Strong-4DVar has the worst σ RMSE** (13–15) — σ is the hardest parameter to identify jointly because it directly scales the dynamics.

4. **β is consistently the best-estimated parameter** (RMSE 0.11–0.61) — least coupled to the state trajectory.

5. **More optimizer steps help Joint-Strong-4DVar** (ratio improved from 1.98× to 1.54× with 60→200 steps) but **hurt Joint-Weak-4DVar** (param σ RMSE increases due to overfitting q).

## 4. Next Steps

- Train JointCFM model (H1 config: default CFM, H2 config: τ=0 CFM) and compare against these baselines
- Expected: JointCFM should outperform Joint-EnKF on state RMSE and match it on param RMSE, given CFM's advantage on the vanilla state estimation task
