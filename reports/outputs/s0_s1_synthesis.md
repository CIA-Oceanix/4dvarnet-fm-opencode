# S0/S1 Benchmark Synthesis: Vanilla UNet vs DA Baselines

**Date**: 2026-07-08
**Dataset**: `make_s0_s1_trainval` with `RandomParamLorenz63Dataset` (per-window random σ, ρ, β ±20%)
**Windows**: 200 test windows per case
**DA window steps**: 50
**Obs settings**: R_var=0.5, obs_interval=20 (14 obs / 300-step window)
**Truth coupling exponent**: a=1.6
**S0 DA exponent**: 1.6 (perfect model)
**S1 DA exponent**: 1.0 (mismatch — param_bias=0.15, forcing_state_bias=0.1)

---

## 1. Background: Bug Fixes Applied

Three issues were fixed before these baselines were produced:

1. **Data leakage** (Jul 7): `generate_observations()` previously cloned `true_fluid` at all 300 steps, overwriting only ~14 observed steps with noise. The remaining ~286 unobserved steps contained exact truth, giving DA methods an unfair identity-mapping lower bound.

2. **NaN observations** (Jul 7): `observations[0]` was NaN (step 0 isn't observed — first obs at step 20), breaking the original background initialization.

3. **Initialization** (Jul 7): The initial NaN fix used `zeros + noise`, which starts far from the attractor. Replaced with **linear interpolation** of the 14 sparse observations across 300 steps, improving all methods by 24–65%.

4. **Lazy obs regeneration** (Jul 7, 22:33): `__getitem__` methods added to `random_bias_dataset.py` and `random_param_dataset.py` to regenerate `"obs"` on-the-fly if stripped by `_strip_obs()` in cached S0/S1 data.

---

## 2. DA Baselines (Interpolation Init, Inflation=2.0)

### S0 (Perfect Model, a=1.6)

| Method | X | Y | Z | **Mean** |
|--------|:---:|:---:|:---:|:--------:|
| Weak-4DVar | 1.32 ± 1.24 | 1.55 ± 1.56 | 2.02 ± 1.14 | **1.63** ± 1.24 |
| Strong-4DVar | 1.54 ± 1.87 | 1.60 ± 2.15 | 1.84 ± 1.54 | **1.66** ± 1.87 |
| EnKF (infl=2.0) | 1.87 ± 1.14 | 2.51 ± 1.74 | 2.97 ± 1.42 | **2.45** ± 1.14 |
| ETKF (infl=2.0) | 1.86 ± 1.14 | 2.51 ± 1.74 | 2.96 ± 1.42 | **2.45** ± 1.14 |

### S1 (Model Mismatch, a=1.0 DA)

| Method | X | Y | Z | **Mean** |
|--------|:---:|:---:|:---:|:--------:|
| Weak-4DVar | 1.42 ± 1.05 | 1.88 ± 1.20 | 3.27 ± 0.78 | **2.19** ± 1.05 |
| Strong-4DVar | 1.68 ± 1.61 | 2.07 ± 1.94 | 3.98 ± 1.15 | **2.57** ± 1.61 |
| EnKF (infl=2.0) | 2.00 ± 0.90 | 3.01 ± 1.33 | 4.58 ± 0.93 | **3.19** ± 0.90 |
| ETKF (infl=2.0) | 2.00 ± 0.90 | 3.04 ± 1.31 | 4.54 ± 0.93 | **3.19** ± 0.90 |

### Degradation (S1 / S0)

| Method | S0 | S1 | Degradation |
|--------|:--:|:--:|:-----------:|
| Weak-4DVar | 1.63 | 2.19 | 1.34x |
| Strong-4DVar | 1.66 | 2.57 | 1.55x |
| EnKF | 2.45 | 3.19 | 1.30x |
| ETKF | 2.45 | 3.19 | 1.30x |

**Best DA method**: Weak-4DVar on both S0 (1.63) and S1 (2.19).

---

## 3. Vanilla UNet (DirectUNet) Results — Jul 8

Configs: 200 epochs, lr=0.001, gradient_clip_val=10.0, trained on CS1+CS2 mixed data.

### S1 — Large UNet ([64, 128, 256] channels)

| Component | S0 RMSE | S1 RMSE |
|-----------|:-------:|:-------:|
| X | 0.542 | 0.892 |
| Y | 0.741 | 1.268 |
| Z | 1.056 | 2.865 |
| **Mean** | **0.780** | **1.675** |
| Degradation | — | 2.15x |
| Train time | 484s (8 min) | — |

### S2 — Small UNet ([32, 64, 128] channels)

| Component | S0 RMSE | S1 RMSE |
|-----------|:-------:|:-------:|
| X | 0.621 | 0.787 |
| Y | 0.655 | 0.998 |
| Z | 0.891 | 2.587 |
| **Mean** | **0.722** | **1.457** |
| Degradation | — | 2.02x |
| Train time | 433s (7.2 min) | — |

---

## 4. Head-to-Head: UNet vs Best DA

| Scenario | Best DA (Weak-4DVar) | S1 Large UNet | S2 Small UNet |
|----------|:--------------------:|:-------------:|:-------------:|
| **S0** | 1.63 | **0.78** (−52%) | **0.72** (−56%) |
| **S1** | 2.19 | 1.68 (−23%) | **1.46** (−33%) |
| Degradation | 1.34x | 2.15x | 2.02x |

### Key Findings

1. **Both UNets outperform ALL DA baselines** on both S0 and S1 by wide margins (23–56%).
2. **The small UNet (S2) beats the large one (S1)** on both S0 (0.722 vs 0.780) and S1 (1.457 vs 1.675), suggesting the larger architecture may overfit given the training data size.
3. **S2 beats even the best DA method (Weak-4DVar)** by 56% on S0 and 33% on S1.
4. **Degradation is higher for UNets (2.02–2.15x) vs DA methods (1.30–1.34x)** in relative terms, but in absolute RMSE the UNet is far superior on both regimes.
5. **Training is fast**: ~7–8 minutes for 200 epochs on GPU.

---

## 5. Pending Work

- **S3/S4 (VanillaCFM on S0/S1)**: Training completed but evaluation crashed with `KeyError: 'obs'` — fixed by lazy obs regeneration on Jul 7 @ 22:33. Needs re-running.
- **S5 (JointCFM on S0/S1)**: Diverged to NaN around epoch 399. Needs debugging (learning rate, gradient clipping, or data dynamics).
- **S4/S6**: CUDA device contention — needs isolated single-GPU runs.
