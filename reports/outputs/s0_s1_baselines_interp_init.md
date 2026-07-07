# S0/S1 Baseline Results — Data Leakage Fix & Interpolation Initialization

**Date**: 2026-07-08
**Dataset**: `make_s0_s1_trainval` with `RandomParamLorenz63Dataset` (per-window random σ, ρ, β ±20%)
**Windows**: 200 test windows per case
**DA window steps**: 50
**EnKF inflation**: 2.0
**ETKF inflation**: 2.0
**Observation settings**: R_var=0.5, obs_interval=20 (14 obs / 300-step window)

## Background

Two issues were identified and fixed:

1. **Data leakage**: `generate_observations()` previously cloned `true_fluid` at all 300 steps, then overwrote only ~14 observed steps with noisy data. The remaining ~286 unobserved steps contained exact truth, giving DA methods an unfair advantage (identity-mapping lower bound ~0.08–0.16 RMSE).

2. **Initialization**: After the NaN fix, `observations[0]` became NaN (step 0 isn't observed — first obs at step 20), breaking the original init `observations[0].clone()`. The initial fix used `zeros + noise`, which starts far from the attractor. The improved fix uses **linear interpolation** of the 14 sparse observations across 300 steps, with extrapolation using nearest-observed values.

## Results: Interpolation Init vs Zeros+Noise Init

### S0 (Perfect Model, a=1.6 truth, a=1.6 DA)

| Method | Zeros+Noise Init RMSE | Interp Init RMSE | Δ |
|--------|:---------------------:|:----------------:|:-:|
| Weak-4DVar | 2.89 ± 1.74 | **1.63** ± 1.24 | **−44%** |
| Strong-4DVar | 4.73 ± 2.87 | **1.66** ± 1.87 | **−65%** |
| EnKF (infl=2.0) | 3.66 ± 1.01 | **2.45** ± 1.14 | **−33%** |
| ETKF (infl=2.0) | 3.68 ± 1.01 | **2.45** ± 1.14 | **−34%** |

### S1 (Model Mismatch, a=1.6 truth, a=1.0 DA)

| Method | Zeros+Noise Init RMSE | Interp Init RMSE | Δ |
|--------|:---------------------:|:----------------:|:-:|
| Weak-4DVar | 3.13 ± 1.58 | **2.19** ± 1.05 | **−30%** |
| Strong-4DVar | 4.44 ± 2.88 | **2.57** ± 1.61 | **−42%** |
| EnKF (infl=2.0) | 4.20 ± 0.84 | **3.19** ± 0.90 | **−24%** |
| ETKF (infl=2.0) | 4.22 ± 0.84 | **3.19** ± 0.90 | **−24%** |

## Per-Component Breakdown (Interpolation Init)

### S0

| Component | Weak-4DVar | Strong-4DVar | EnKF | ETKF |
|-----------|:----------:|:------------:|:----:|:----:|
| X | 1.32 ± 1.24 | 1.54 ± 1.87 | 1.87 ± 1.14 | 1.86 ± 1.14 |
| Y | 1.55 ± 1.56 | 1.60 ± 2.15 | 2.51 ± 1.74 | 2.51 ± 1.74 |
| Z | 2.02 ± 1.14 | 1.84 ± 1.54 | 2.97 ± 1.42 | 2.96 ± 1.42 |
| **Mean** | **1.63** | **1.66** | **2.45** | **2.45** |

### S1

| Component | Weak-4DVar | Strong-4DVar | EnKF | ETKF |
|-----------|:----------:|:------------:|:----:|:----:|
| X | 1.42 ± 1.05 | 1.68 ± 1.61 | 2.00 ± 0.90 | 2.00 ± 0.90 |
| Y | 1.88 ± 1.20 | 2.07 ± 1.94 | 3.01 ± 1.33 | 3.04 ± 1.31 |
| Z | 3.27 ± 0.78 | 3.98 ± 1.15 | 4.58 ± 0.93 | 4.54 ± 0.93 |
| **Mean** | **2.19** | **2.57** | **3.19** | **3.19** |

## Comparison with Old (Leaky) Results

The old leaky results (from `baselines_dws50_inf2.0_etkf_inf2.0.json`) used fixed σ/ρ/β (Not RandomParamLorenz63Dataset) and leaked truth at all 300 steps. They are **not directly comparable** — both the test dataset and the observation model changed. The old results are shown for reference only:

### Old vs New — S0

| Method | Old (leaky, fixed params) | New (fixed NaN, random params, interp init) |
|--------|:-------------------------:|:-------------------------------------------:|
| Weak-4DVar | 0.54 ± 0.10 | 1.63 ± 1.24 |
| Strong-4DVar | 0.56 ± 0.10 | 1.66 ± 1.87 |
| EnKF (infl=2.0) | 0.74 ± 0.08 | 2.45 ± 1.14 |
| ETKF (infl=2.0) | 0.72 ± 0.08 | 2.45 ± 1.14 |

The old results had 4–5× lower RMSE but used a fundamentally different (and invalid) observation model. The tight std (0.08–0.10) vs new (1.1–1.9) reflects both the per-window param variation in the new test set and the observation sparsity effect without data leakage.

## Discussion

1. **Interpolation init provides major improvement (24–65%)** over zeros+noise across all methods and both cases. The effect is strongest for Strong-4DVar (65% S0), which has no model-error term and is most dependent on good initialization.

2. **Weak-4DVar now slightly edges out Strong-4DVar** on S0 (1.63 vs 1.66), which is reasonable — the model-error term q helps handle the per-window random-param variation even when the DA params match on average.

3. **EnKF/ETKF are identical** (2.45 S0, 3.19 S1) — both use inflation=2.0 and N=30 ensemble members, and they converge to the same RMSE within the noise of a 200-window sample.

4. **S1 degradation is consistent**: methods degrade 1.3–1.5× from S0 to S1 (Weak: 1.63→2.19, Strong: 1.66→2.57, EnKF: 2.45→3.19, ETKF: 2.45→3.19). Strong-4DVar degrades most (1.5×) — the model mismatch without a q-term is hardest for it.

5. **Runtime** is essentially the same (~256s) with or without interpolation — the interpolation is negligible overhead.

## Next Steps

- Submit S3 vanilla CFM training (the `_make_eval_batch` crash is now fixed by on-the-fly obs generation in `__getitem__`)
- The RMSE values (~1.6–3.2) are the correct baselines for the S0/S1 setup with NaN-fixed observations