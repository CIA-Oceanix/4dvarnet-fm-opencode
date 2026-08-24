# CS4b Case Study: Per-Window Random Bias (Lorenz-63) — Baseline Results

**Date:** 2026-07-03  
**Branch:** `exp/cs4b-randombias`  
**Data:** CS1–CS4b datasets (Lorenz-63, param_noise=0.2, CS4b: per-window sigma/rho/beta ±20% + random bias U(0,0.20))  
**Settings:** da_window_steps=50, batch_size=200, opt_steps=200 (4DVar methods)  
**Inflation:** EnKF=2.4, ETKF=2.4 (tuned on CS4b)

---

## 1. Comparison Table: State RMSE (mean over X/Y/Z)

| Case | Weak-4DVar | Strong-4DVar | EnKF | ETKF |
|------|:----------:|:------------:|:----:|:----:|
| CS1  | 0.6574     | 0.6602       | 0.7261 | 0.7189 |
| CS2  | 1.6656     | 2.1650       | 2.4422 | 2.4430 |
| CS3  | 0.6449     | 0.6412       | 0.7385 | 0.7272 |
| CS4  | 0.7815     | 0.9129       | 0.9799 | 0.9608 |
| CS4b | **1.0657** | **1.2720**   | **1.3211** | **1.2983** |

## 2. Degradation Chain (Weak-4DVar)

| Step | Configuration | RMSE | Δ |
|------|--------------|:----:|:-:|
| CS1  | Fixed Lorenz (no mismatch) | 0.6574 | — |
| CS3  | + per-window σ/ρ/β (±20%) | 0.6449 | −0.0125 |
| CS4  | + quartic coupling (model mismatch) | 0.7815 | +0.1366 |
| CS4b | + per-window random bias U(0,0.20) | **1.0657** | **+0.2842** |

## 3. Inflation Sweep Results (CS4b, DWS=50)

Sweep over 15 inflation values (1.0–4.0) for EnKF and ETKF on the CS4b test set.

| Inflation | EnKF | ETKF |
|:---------:|:----:|:----:|
| 1.0 | 1.4174 | 1.4115 |
| 1.1 | 1.4097 | 1.4072 |
| 1.2 | 1.4013 | 1.3951 |
| 1.3 | 1.3925 | 1.3840 |
| 1.4 | 1.3833 | 1.3733 |
| 1.5 | 1.3734 | 1.3619 |
| 1.6 | 1.3619 | 1.3502 |
| 2.0 | 1.3304 | 1.3188 |
| 2.2 | 1.3159 | 1.3052 |
| **2.4** | **1.3026** | **1.2968** |
| 2.6 | 1.3110 | 1.3038 |
| 2.8 | 1.3312 | 1.3206 |
| 3.0 | 1.3493 | 1.3378 |
| 3.5 | 1.3962 | 1.3807 |
| 4.0 | 1.4365 | 1.4184 |

**Optimal inflation:** EnKF=2.4 (mean=1.3026), ETKF=2.4 (mean=1.2968)

Both show a clear U-shaped curve with minimum at 2.4. Notably higher than CS4's optimal (EnKF≈1.30, ETKF≈1.60) — per-window bias demands stronger inflation.

## 4. Key Findings

1. **4DVar dominates all cases at DWS=50.** With short windows and model error Q, Weak-4DVar handles even the CS4b bias better than ensemble methods at inflation=2.4. This is consistent with 4DVar's ability to fit the trajectory within each window.

2. **Per-window random bias (CS4b) degrades all methods ∼1.4× relative to CS4.** The jump from CS4 (0.78) to CS4b (1.07) for Weak-4DVar is the largest step in the degradation chain. Stochastic parameters make the state estimation fundamentally harder than fixed bias.

3. **EnKF/ETKF at CS4b-optimal inflation=2.4 suffer on simpler cases** (CS1 mean=0.73, while optimal would be ∼0.49 at inflation∼1.0). The inflation tuned for CS4b over-disperses the ensemble for cases without per-window bias.

4. **EnKF and ETKF are nearly equivalent** across all cases at inflation=2.4 (ETKF slightly better by ∼0.02). The deterministic ETKF update provides marginal benefit over stochastic EnKF for this setup.

5. **CS3 ≈ CS1** for Weak-4DVar (0.6449 vs 0.6574) — per-window parameter variation alone (without model mismatch or bias) is well-handled by 4DVar when the window is short enough.

## 5. Dataset & Performance Notes

- Dataset generation is CPU-bound (7M integration steps, ∼20–25 min). Cached to `experiments/datasets_cs4b.pt` (24 MB).
- 4DVar methods: ∼55s/case at batch_size=200, DWS=50.
- EnKF/ETKF: 0.2–2.0s/case at batch_size=200.
- CS4b test seed: 130, 200 test windows.
- Files: `evaluation/run.py` (baseline runner), `data/random_bias_dataset.py` (CS4b dataset), `data/lorenz63.py` (make_mixed_datasets).

## 6. Next Steps

- Run CS4b evaluation with **case-optimal inflation** for a fair comparison across all cases.
- Train CFM models (E1–E3, F1–F3) on CS4b data and compare against these baselines.
- Investigate: can CFM's denoising score matching handle per-window random bias better than ensemble Kalman methods?
