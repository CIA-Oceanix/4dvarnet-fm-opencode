# S0/S1 Benchmark Synthesis: Vanilla UNet vs DA Baselines

**Date**: 2026-07-08
**Branch**: `exp/obs-at-step0`
**Dataset**: `make_s0_s1_trainval` with `RandomParamLorenz63Dataset` (per-window random σ, ρ, β ±20%)
**Windows**: 200 test windows per case
**DA window steps**: 50
**Obs settings**: R_var=0.5, obs_interval=20 (15 obs / 300-step window, now includes step 0)
**Truth coupling exponent**: a=1.6
**S0 DA exponent**: 1.6 (perfect model)
**S1 DA exponent**: 1.0 (mismatch — param_bias=0.15, forcing_state_bias=0.1)

---

## 1. Background: Bug Fixes Applied

Three issues were fixed before these baselines were produced:

1. **Data leakage** (Jul 7): `generate_observations()` previously cloned `true_fluid` at all 300 steps, overwriting only ~14 observed steps with noise. The remaining ~286 unobserved steps contained exact truth, giving DA methods an unfair identity-mapping lower bound.

2. **NaN observations** (Jul 7, fixed in this branch): `observations[0]` was NaN because step 0 wasn't observed (first obs at step 20), breaking the background initialization for DA methods. **This branch adds observation at step 0** (`np.arange(0, ...)` instead of `np.arange(obs_interval, ...)`), giving 15 instead of 14 observations.

3. **Initialization** (Jul 7): The initial NaN fix used `zeros + noise`, which starts far from the attractor. Replaced with **linear interpolation** of sparse observations across 300 steps, improving all methods by 24–65%.

4. **Lazy obs regeneration** (Jul 7, 22:33): `__getitem__` methods added to `random_bias_dataset.py` and `random_param_dataset.py` to regenerate `"obs"` on-the-fly if stripped by `_strip_obs()` in cached S0/S1 data.

---

## 2. DA Baselines (Obs at Step 0, Interpolation Init, Inflation=2.0)

### S0 (Perfect Model, a=1.6)

| Method | X | Y | Z | **Mean** |
|--------|:---:|:---:|:---:|:--------:|
| Weak-4DVar | 0.45 ± 0.45 | 0.66 ± 0.73 | 0.81 ± 0.57 | **0.64** ± 0.47 |
| Strong-4DVar | 0.51 ± 0.50 | 0.72 ± 0.58 | 0.95 ± 0.56 | **0.73** ± 0.45 |
| EnKF (infl=2.0) | 0.53 ± 0.56 | 0.85 ± 0.71 | 0.95 ± 0.57 | **0.78** ± 0.52 |
| ETKF (infl=2.0) | 0.52 ± 0.52 | 0.85 ± 0.71 | 0.95 ± 0.56 | **0.77** ± 0.50 |

### S1 (Model Mismatch, a=1.0 DA)

| Method | X | Y | Z | **Mean** |
|--------|:---:|:---:|:---:|:--------:|
| Weak-4DVar | 0.84 ± 0.52 | 1.30 ± 0.73 | 2.77 ± 0.58 | **1.64** ± 0.45 |
| Strong-4DVar | 1.14 ± 0.66 | 1.60 ± 0.90 | 3.69 ± 0.67 | **2.14** ± 0.58 |
| EnKF (infl=2.0) | 1.14 ± 0.62 | 1.94 ± 0.81 | 3.73 ± 0.70 | **2.27** ± 0.55 |
| ETKF (infl=2.0) | 1.14 ± 0.62 | 1.99 ± 0.78 | 3.69 ± 0.72 | **2.28** ± 0.55 |

### Improvement over No-Obs-at-Step0

| Method | S0 (was→now) | S1 (was→now) |
|--------|:------------:|:------------:|
| Weak-4DVar | 1.63 → **0.64** (−61%) | 2.19 → **1.64** (−25%) |
| Strong-4DVar | 1.66 → **0.73** (−56%) | 2.57 → **2.14** (−17%) |
| EnKF | 2.45 → **0.78** (−68%) | 3.19 → **2.27** (−29%) |
| ETKF | 2.45 → **0.77** (−69%) | 3.19 → **2.28** (−29%) |

### Degradation (S1 / S0)

| Method | S0 | S1 | Degradation |
|--------|:--:|:--:|:-----------:|
| Weak-4DVar | 0.64 | 1.64 | 2.56x |
| Strong-4DVar | 0.73 | 2.14 | 2.94x |
| EnKF | 0.78 | 2.27 | 2.91x |
| ETKF | 0.77 | 2.28 | 2.94x |

**Best DA method**: Weak-4DVar on both S0 (0.64) and S1 (1.64).

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

## 4. Head-to-Head: UNet vs Best DA (Obs at Step 0)

| Scenario | Best DA (Weak-4DVar) | S1 Large UNet | S2 Small UNet |
|----------|:--------------------:|:-------------:|:-------------:|
| **S0** | 0.64 | 0.78 (−22% vs W4D) | **0.72** (−13% vs W4D) |
| **S1** | 1.64 | 1.68 (+2% vs W4D) | **1.46** (−11% vs W4D) |
| Degradation | 2.56x | 2.15x | **2.02x** |

### Key Findings

1. **With obs at step 0, the gap between UNet and DA narrows dramatically** — Weak-4DVar now achieves 0.64 RMSE on S0 (was 1.63 without step 0) and 1.64 on S1 (was 2.19).
2. **On S0, the best DA (Weak-4DVar at 0.64) now outperforms BOTH UNets** (S1 large: 0.78, S2 small: 0.72). The DA benefits more from the step-0 observation because its model is perfect on S0.
3. **On S1 (model mismatch), S2 UNet still leads** (1.46 vs 1.64 for Weak-4DVar, a 11% advantage), but the margin is smaller than before (33% without step 0).
4. **S1 Large UNet is now slightly worse than Weak-4DVar** on S1 (1.68 vs 1.64), flipping the previous result.
5. **The small UNet (S2) remains the best overall method** on S1, and the most robust to degradation (2.02x vs 2.56x for Weak-4DVar).
6. **Training is fast**: ~7–8 minutes for 200 epochs on GPU.

---

## 5. Notes

- **Obs at step 0** (`exp/obs-at-step0` branch): This report reflects results with an observation at time step 0 (via `np.arange(0, num_steps, obs_interval)`). This significantly improves DA baseline performance by eliminating the unknown initial condition. UNet results are unchanged (same model weights).
- **Unsure what to think**: the new best DA outperforms the UNets on S0 but not on S1. The DA methods now seem to compete directly with the learned approaches.

## 6. Pending Work

- **S3/S4 (VanillaCFM on S0/S1)**: Training completed but evaluation crashed with `KeyError: 'obs'` — fixed by lazy obs regeneration on Jul 7 @ 22:33. Needs re-running.
- **S5 (JointCFM on S0/S1)**: Diverged to NaN around epoch 399. Needs debugging (learning rate, gradient clipping, or data dynamics).
- **S4/S6**: CUDA device contention — needs isolated single-GPU runs.
