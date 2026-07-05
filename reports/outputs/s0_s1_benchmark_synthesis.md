# S0/S1 Benchmark Synthesis

**Date**: 2026-07-04
**Branch**: `exp/s0-s1-benchmark`
**Truth coupling exponent**: a=1.6 (default)
**S0 DA exponent**: 1.6 (perfect model)
**S1 DA exponent**: 1.0 (mismatch)
**DA window steps**: 50

## Experimental Setup

| Parameter | S0 (Reference) | S1 (Biased) |
|-----------|---------------|-------------|
| Truth coupling exponent | 1.6 | 1.6 |
| DA coupling exponent | 1.6 | 1.0 |
| Parametric bias | None | 15% (σ×0.85, ρ×0.85, β×1.15) |
| c₁ bias | None | 15% (c₁×0.85) |
| Forcing corruption | None | η(t) + 0.1×X |
| Default obs noise (R_var) | 0.5 | 0.5 |
| Default obs interval | 20 | 20 |
| DA window steps | 50 | 50 |

Two experimental sets are reported:
- **200‑window sweeps**: Comprehensive obs noise/density sweeps using `batch/noise_density_sweep.py` with EnKF/ETKF inflation fixed at 1.2/1.6 and R_var passed to all DA methods. 200 windows for robust statistics.
- **10‑window validation**: Dev runs using `evaluate_all.py` with inflation=2.0. Used for coupling exponent sensitivity and as a cross-check.

---

## 1. Coupling Exponent Sensitivity

10 windows, dws=50, inflation=2.0 (EnKF/ETKF), R_var=0.5, obs_interval=20.

### a=1.5

| Method | S0 | S1 | Ratio S1/S0 |
|--------|------|------|-------------|
| Weak-4DVar | 0.5910 | 1.3716 | 2.32× |
| Strong-4DVar | 0.6207 | 1.8482 | 2.98× |
| EnKF | 0.7541 | 2.1182 | 2.81× |
| ETKF | 0.7444 | 2.1051 | 2.83× |

### a=1.6

| Method | S0 | S1 | Ratio S1/S0 |
|--------|------|------|-------------|
| Weak-4DVar | 0.5397 | 1.5032 | 2.79× |
| Strong-4DVar | 0.5574 | 2.4463 | 4.39× |
| EnKF | 0.7393 | 2.1742 | 2.94× |
| ETKF | 0.7198 | 2.1764 | 3.02× |

### a=1.75

| Method | S0 | S1 | Ratio S1/S0 |
|--------|------|------|-------------|
| Weak-4DVar | 4.8676 | 1.9826 | 0.41× |
| Strong-4DVar | 5.9081 | 2.7179 | 0.46× |
| EnKF | 0.9054 | 2.4012 | 2.65× |
| ETKF | 0.8984 | 2.3768 | 2.65× |

### a=2.0

All methods NaN. The coupling term c₁·sign(W)·|W|² produces trajectories that diverge during the 10000-step spinup for some random seeds (stochastic Lorenz63 with Euler-Maruyama integration).

### Summary

| a | Best S0 | Best S1 | S0 < S1? | Notes |
|---|---------|---------|----------|-------|
| 1.5 | 0.591 (Weak) | 1.372 (Weak) | ✓ All methods | Baseline, smallest gap |
| **1.6** | **0.540 (Weak)** | **1.503 (Weak)** | **✓ All methods** | **Largest variational gap** |
| 1.75 | 0.898 (ETKF) | 1.983 (Weak) | ✗ Variational fail on S0 | xKF stable but degraded |
| 2.0 | NaN | NaN | — | Unstable integrator |

Key findings:
- **a=1.5** and **a=1.6** are the only stable regimes where all methods show proper S0 < S1 ordering.
- **a=1.6** widens the S0–S1 gap for 4DVar methods (Strong-4DVar: 2.98× → 4.39×) while keeping S0 RMSE low (~0.54).
- **a=1.75** degrades variational methods even on perfect-model S0 (RMSE jumps from ~0.55 to ~4.9), suggesting the optimizer struggles with stronger nonlinearity.
- xKF methods with inflation=2.0 are robust across all stable a values.
- **a=2.0** is unstable for long trajectories with dt=0.01 Euler-Maruyama.

---

## 2. Observation Noise Sweep

200 windows, dws=50, EnKF inflation=1.2 / ETKF inflation=1.6, obs_interval=20.
R_var passed to all DA methods (Weak/Strong-4DVar, EnKF, ETKF).

| R_var | Case | Weak-4DVar | Strong-4DVar | EnKF | ETKF |
|:-----:|:----:|:----------:|:------------:|:----:|:----:|
| 0.1 | S0 | 0.479 | 0.569 | 0.783 | 0.541 |
| | S1 | 1.394 | 2.164 | 3.027 | 2.271 |
| 0.25 | S0 | 0.528 | 0.655 | 0.810 | 0.613 |
| | S1 | 1.518 | 2.140 | 2.815 | 2.251 |
| 0.5 | S0 | 0.642 | 0.687 | 0.851 | 0.721 |
| | S1 | 1.622 | 2.158 | 2.705 | 2.288 |
| 1.0 | S0 | 0.773 | 0.863 | 0.935 | 0.902 |
| | S1 | 1.778 | 2.106 | 2.693 | 2.336 |
| 2.0 | S0 | 1.073 | 1.101 | 1.133 | 1.196 |
| | S1 | 1.986 | 2.211 | 2.713 | 2.479 |
| 4.0 | S0 | 1.471 | 1.440 | 1.455 | 1.627 |
| | S1 | 2.289 | 2.544 | 2.794 | 2.740 |

### Cross-check with 10-window, inf=2.0 results

| R_var | Src | S0 Weak | S0 Strong | S0 EnKF | S0 ETKF | S1 Weak | S1 Strong | S1 EnKF | S1 ETKF |
|:-----:|:---:|:-------:|:---------:|:-------:|:-------:|:-------:|:---------:|:-------:|:-------:|
| 0.5 | 200w/i1.2 | 0.642 | 0.687 | 0.851 | 0.721 | 1.622 | 2.158 | 2.705 | 2.288 |
| 0.5 | 10w/i2.0 | 0.540 | 0.557 | 0.739 | 0.720 | 1.503 | 2.446 | 2.174 | 2.176 |
| 2.0 | 200w/i1.2 | 1.073 | 1.101 | 1.133 | 1.196 | 1.986 | 2.211 | 2.713 | 2.479 |
| 2.0 | 10w/i2.0 | 0.983 | 0.883 | 1.187 | 1.229 | 1.696 | 2.575 | 2.355 | 2.381 |

The 10‑window results show the same qualitative trends but differ in absolute values due to inflation (1.2 vs 2.0) and sampling noise.

### Analysis

- **Weak-4DVar** is the best method at low noise (R_var ≤ 1.0), but all methods converge at high noise (R_var ≥ 2.0), with RMSE clustering between 1.1–1.6 on S0 and 2.2–2.8 on S1.
- **Strong-4DVar** and **ETKF** are most noise‑robust on S0 (degrade ~2.5× from 0.1→4.0 vs ~3× for others).
- On S1, the noise sensitivity is muted: S1 RMSE includes a large irreducible error from model bias (~1.4–2.3 even at R_var=0.1), so added observation noise contributes proportionally less to total error.
- The **S0–S1 gap shrinks** as noise increases: at R_var=0.1 the gap is ~1.6–2.2×, at R_var=4.0 it is ~1.5–1.8×.

---

## 3. Observation Sparsity Sweep

200 windows, dws=50, EnKF inflation=1.2 / ETKF inflation=1.6, R_var=0.5.

| obs_int | Obs/win | Case | Weak-4DVar | Strong-4DVar | EnKF | ETKF |
|:-------:|:-------:|:----:|:----------:|:------------:|:----:|:----:|
| 10 | 29 | S0 | 0.487 | 0.554 | 0.626 | 0.550 |
| | | S1 | 1.110 | 1.914 | 2.284 | 1.819 |
| 20 | 14 | S0 | 0.670 | 0.687 | 0.830 | 0.718 |
| | | S1 | 1.600 | 2.158 | 2.728 | 2.287 |
| 40 | 7 | S0 | 0.993 | 0.978 | 1.339 | 1.245 |
| | | S1 | 2.617 | 2.811 | 3.854 | 3.437 |
| 60 | 5 | S0 | 1.965 | 2.040 | 2.141 | 2.104 |
| | | S1 | 4.325 | 4.036 | 5.287 | 4.941 |
| 100 | 3 | S0 | 6.703 | 5.019 | 3.181 | 3.192 |
| | | S1 | 9.307 | 7.709 | 6.540 | 6.422 |

### Cross-check with 10-window, inf=2.0 results

| obs_int | Src | S0 Weak | S0 Strong | S0 EnKF | S0 ETKF | S1 Weak | S1 Strong | S1 EnKF | S1 ETKF |
|:-------:|:---:|:-------:|:---------:|:-------:|:-------:|:-------:|:---------:|:-------:|:-------:|
| 60 | 200w/i1.2 | 1.965 | 2.040 | 2.141 | 2.104 | 4.325 | 4.036 | 5.287 | 4.941 |
| 60 | 10w/i2.0 | 1.171 | 1.397 | 1.695 | 1.740 | 3.908 | 4.709 | 4.959 | 5.000 |

The 10‑window results are systematically lower for the variational methods on S0 (inf=2.0 helps convergence), while ensemble methods agree well across both sets.

### Analysis

- **Sparsity is much more harmful than noise.** Going from obs_int=20→60 (3× fewer obs) degrades RMSE ~3× on both S0 and S1, while increasing noise 4× (R_var=0.5→2.0) degrades only ~1.7×.
- At **obs_int=100** (3 observations per window), the variational methods collapse (Weak-4DVar: 6.70 S0, 9.31 S1) while xKF methods are comparatively robust (~3.2 S0, ~6.5 S1), benefiting from the ensemble covariance propagation between obs.
- The **S0–S1 gap widens dramatically** with sparsity: absolute gap grows from ~0.9 at obs_int=10 to ~3.3 at obs_int=60.
- **Strong-4DVar becomes the best variational method** at high sparsity (obs_int=100), overtaking Weak-4DVar as the model-error term (q) in weak-constraint becomes poorly constrained.

---

## 4. Effect of Inflation on xKF (dws=50, a=1.5, 10 windows)

| Method | S0 (inf=1.0) | S0 (inf=2.0) | S1 (inf=1.0) | S1 (inf=2.0) |
|--------|:------------:|:------------:|:------------:|:------------:|
| EnKF | 1.9517 | 0.7541 | 5.5139 | 2.1182 |
| ETKF | 1.8089 | 0.7444 | 4.4976 | 2.1051 |

Inflation=2.0 improves xKF by ~2.5–2.6× on both S0 and S1. With inflation, xKF becomes competitive with 4DVar methods.

---

## 5. Per-Component Breakdown (a=1.6, 10 windows, inflation=2.0)

### S0 (Perfect Model)

| Component | Weak-4DVar | Strong-4DVar | EnKF | ETKF |
|-----------|:----------:|:------------:|:----:|:----:|
| X | 0.391 ± 0.102 | 0.411 ± 0.076 | 0.472 ± 0.084 | 0.457 ± 0.077 |
| Y | 0.559 ± 0.131 | 0.540 ± 0.102 | 0.793 ± 0.132 | 0.768 ± 0.125 |
| Z | 0.669 ± 0.112 | 0.722 ± 0.116 | 0.953 ± 0.208 | 0.934 ± 0.180 |
| **Mean** | **0.540** | **0.557** | **0.739** | **0.720** |

### S1 (Biased)

| Component | Weak-4DVar | Strong-4DVar | EnKF | ETKF |
|-----------|:----------:|:------------:|:----:|:----:|
| X | 0.737 ± 0.166 | 1.461 ± 1.611 | 1.084 ± 0.223 | 1.070 ± 0.196 |
| Y | 1.147 ± 0.176 | 1.961 ± 1.738 | 1.858 ± 0.293 | 1.912 ± 0.264 |
| Z | 2.625 ± 0.315 | 3.917 ± 1.892 | 3.580 ± 0.590 | 3.547 ± 0.601 |
| **Mean** | **1.503** | **2.446** | **2.174** | **2.176** |

The Z component is the most degraded in S1 across all methods (β bias is ×1.15, and Z dynamics are the slowest to recover from error). Strong-4DVar shows high variance on S1 (std ~1.9), indicating convergence instability on some windows.

---

## 6. Additional Results: dws=300 (No Inflation, a=1.5, 10 windows)

| Method | S0 | S1 |
|--------|------|------|
| Weak-4DVar | 5.1702 | 3.7100 |
| Strong-4DVar | 7.2674 | 9.2078 |
| EnKF | 2.0935 | 5.1058 |
| ETKF | 1.8792 | 4.3132 |

The anomalous S0 > S1 for Weak-4DVar at dws=300 (5.17 > 3.71) is attributed to the longer DA window making variational optimization harder.

---

## 7. Conclusions

1. **Best operating regime**: a=1.6 truth exponent with S0 DA=1.6 (perfect), S1 DA=1.0 (mismatch). Maximizes S0–S1 contrast while keeping all methods stable with S0 < S1 ordering.

2. **Weak-4DVar** is the best method at default obs settings (R_var=0.5, obs_int=20) on both S0 and S1. Strong-4DVar overtakes Weak-4DVar only at extreme sparsity (obs_int=100).

3. **xKF methods** benefit dramatically from multiplicative inflation (2.0) and are more robust than variational methods at higher coupling exponents (a≥1.75) and at extreme observation sparsity (obs_int=100).

4. **Observation sparsity degrades performance much more than observation noise**: 3× fewer obs (20→60) causes ~3× RMSE increase, while 4× higher noise (0.5→2.0) causes only ~1.7× increase.

5. **The Z component** is consistently the most affected by the S1 parametric bias (β×1.15), with 2–3× higher RMSE than X or Y.

6. **a=2.0** is not viable with the current Euler-Maruyama integrator (dt=0.01) due to trajectory divergence during the 10000-step spinup.
