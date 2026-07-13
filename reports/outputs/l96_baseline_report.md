# L96 Baseline Report — Observation Config & Trajectory Examples

**Date**: 2026-07-13
**System**: Two-scale Lorenz96 (NO=8, J=4, state_dim=40)
**Model**: `Lorenz96Dynamics` with RK4 integration, `dt=0.001`, T_max=3.0 (3000 steps/window)
**Forcing**: F=8.0, coupling exponent=1.6, AR(1) stochastic forcing with sinusoidal component
**Observations**: Full-state (all 40 dims) for Wave 1/2; partial (2/4 fast vars per node → 24D) for Wave 3, `obs_interval=5` (every 0.005 tu), R_var=0.5
**Climatological variance** (200k-step free run on truth dynamics): slow vars 1.052 ± 0.001, fast vars 1.360 ± 0.004, overall ≈ 1.30. For J-mismatch experiments (Wave 3), EV is computed on the common 24 variables shared between S0 and S1 (first 24 dims: X[0-7], Y1[0-7], Y2[0-7]), with truth variance computed from the dataset (slow: 1.184, fast1: 1.541, fast2: 1.584, overall: 1.436). Explained variance is EV = 1 − MSE / Var<sub>truth</sub>.

---

## Annex: L96 Trajectory and Observation Patterns

The figures below illustrate a single 3.0-tu window (3000 steps) of the Lorenz96 system with the observation pattern used in the S0/S1 experiments.

### Figure 1: Field heatmap, observations, and slow variable traces

![L96 field + observations](figs/l96_trajectory_field.png)

Three panels:
1. **True L96 state** — 2D spacetime Hovmöller of the full 40-dim state (8 slow + 32 fast); vertical white dashed lines mark observation times (every 0.2 tu, 15 per window)
2. **Observations** — same format, showing only the noisy observed values at observation times (NaN = unobserved interleaved steps)
3. **Slow variable line plots** — X1..X8 with observation markers (colored dots) overlaid

### Figure 2: Per-variable slow trajectories with observations

![L96 slow variable trajectories](figs/l96_slow_vars_trajectories.png)

All 8 slow variables individually, showing truth (black line) and observations (orange dots, R_var=0.5). The system oscillates around a mean of ~2.13 with variance ~1.05. Observations at 0.2 tu intervals capture the primary oscillation modes.

### Figure 3: Slow vs fast variable Hovmöller

![L96 Hovmöller slow/fast](figs/l96_hovmoller_fast.png)

Side-by-side comparison of the slow variables (X1..X8) and the first fast variable (Y₁^1..Y₈^1) per slow node. Fast variables oscillate at smaller spatial scales and higher frequency (time scale ε=0.1), with comparable variance (1.36 vs 1.05 for slow).

---

## Observation Config Summary

| Parameter | Value | Notes |
|---|---|---|
| Integration dt | 0.001 | Fine — needed for fast-scale (ε=0.1) stability |
| Steps per window | 3000 | T_max=3.0 / dt=0.001 |
| Obs interval | 200 steps = 0.2 tu | 4× sparser than classic DAPPER setup (0.05 tu) |
| Obs count per window | 15 | Includes step 0 |
| Obs noise (R_var) | 0.5 | Corresponds to ~σ=0.71 (~30% of slow var std ≈1.03) |
| Obs operator | Full 40D identity | All slow+fast variables observed |
| Model | Two-scale L96 | NO=8 slow, J=4 fast per node |
| Classic DAPPER ref | dt=0.05, obs every step, R_var=1.0, single-scale L96 | Main diffs: finer dt (two-scale), sparser obs, lower noise |

---

## Baseline DA Results (5 windows, Strong-4DVar / EnKF / ETKF)

All methods use `da_window_steps=500`, EnKF/ETKF with inflation=2.0, Strong-4DVar with max_iter=10, lr=0.2.

Results are reported as **RMSE** (root-mean-squared error) and **EV** (explained variance = 1 − MSE / Var<sub>clim</sub>). Climatological variance per variable group: slow ≈ 1.05, fast ≈ 1.36, overall ≈ 1.30 (200k-step free run). Positive EV > 0.5 indicates useful skill; EV < 0 means the DA is worse than climatology.

### Wave 1: Model parameter bias sweep

S1 uses biased `F_da = F_true × (1 - param_bias)`, and biased stochastic forcing (`forcing_state_bias` scales the AR(1) forcing innovations). DA dynamics use the correct two-scale Lorenz96.

| Config | Method | S0 RMSE | S0 EV | S1 RMSE | S1 EV | Δ% |
|---|---|---|---|---|---|---|
| **a1**: pb=0.3, fsb=0.3, F_da=5.6 | Strong-4DVar | 0.3646 | 0.894 | 0.4157 | 0.857 | +14.0% |
| | EnKF | 0.5080 | 0.792 | 0.5249 | 0.780 | +3.3% |
| | ETKF | 0.5226 | 0.780 | 0.5074 | 0.794 | -2.9% |
| **a2**: pb=0.3, fsb=0.5, F_da=5.6 | Strong-4DVar | 0.3924 | 0.875 | 0.4631 | 0.825 | +18.0% |
| | EnKF | 0.5103 | 0.789 | 0.5454 | 0.762 | +6.9% |
| | ETKF | 0.5073 | 0.793 | 0.5091 | 0.793 | +0.4% |
| **a3**: pb=0.4, fsb=0.3, F_da=4.8 | Strong-4DVar | 0.4153 | 0.858 | 0.4357 | 0.843 | +4.9% |
| | EnKF | 0.5165 | 0.784 | 0.5335 | 0.770 | +3.3% |
| | ETKF | 0.4999 | 0.798 | 0.5004 | 0.798 | +0.1% |
| **a4**: pb=0.4, fsb=0.5, F_da=4.8 | Strong-4DVar | 0.3999 | 0.872 | 0.4214 | 0.858 | +5.4% |
| | EnKF | 0.5220 | 0.780 | 0.5207 | 0.782 | -0.2% |
| | ETKF | 0.5103 | 0.791 | 0.5298 | 0.775 | +3.8% |
| **a5**: pb=0.5, fsb=0.3, F_da=4.0 | Strong-4DVar | 0.4164 | 0.858 | 0.4388 | 0.845 | +5.4% |
| | EnKF | 0.5194 | 0.782 | 0.5333 | 0.771 | +2.7% |
| | ETKF | 0.5063 | 0.793 | 0.5135 | 0.788 | +1.4% |
| **a6**: pb=0.5, fsb=0.5, F_da=4.0 | Strong-4DVar | 0.4039 | 0.869 | 0.4372 | 0.842 | +8.3% |
| | EnKF | 0.5264 | 0.777 | 0.5521 | 0.755 | +4.9% |
| | ETKF | 0.5093 | 0.791 | 0.5057 | 0.796 | -0.7% |

All S0 and S1 EV values are 0.75–0.89, confirming good-to-excellent DA skill. **Conclusion**: Model parameter bias alone (param_bias up to 0.5, forcing bias up to 0.5) is **insufficient** — EV drops by at most ~0.05 (a2, Strong-4DVar: 0.875→0.825). EnKF/ETKF with inflation=2.0 are nearly unaffected.

### Wave 2: Wrong dynamics model (single-scale DA)

S1 uses single-scale Lorenz96 dynamics (NO=40, J=0, h=0.0 — no coupling to fast variables) inside the DA, while the truth remains two-scale (NO=8, J=4). This is a fundamental model mismatch.

| Config | Method | S0 RMSE | S0 EV | S1 RMSE | S1 EV | Δ% |
|---|---|---|---|---|---|---|
| **b1**: no single-scale, F_da=1.6 | Strong-4DVar | 0.3675 | 0.893 | 0.4077 | 0.867 | +10.9% |
| | EnKF | 0.5152 | 0.785 | 0.5299 | 0.774 | +2.9% |
| | ETKF | 0.5115 | 0.789 | 0.5200 | 0.785 | +1.7% |
| **b2**: single-scale DA, F_da=8.0 ✅ | Strong-4DVar | 0.4048 | 0.868 | 1.7662 | **−1.518** | **+336.3%** |
| | EnKF | 0.5183 | 0.784 | 2.9433 | **−5.821** | **+467.9%** |
| | ETKF | 0.5058 | 0.794 | 2.4041 | **−3.589** | **+375.3%** |
| **b3**: single-scale DA, F_da=1.6 | Strong-4DVar | 0.3991 | 0.873 | 1.7690 | −1.525 | +343.2% |
| | EnKF | 0.5209 | 0.782 | 2.9636 | −5.891 | +469.0% |
| | ETKF | 0.5078 | 0.793 | 2.4431 | −3.700 | +381.1% |
| **b4**: single-scale + F_da=1.6 + no infl | Strong-4DVar | 0.4204 | 0.856 | 1.7666 | −1.519 | +320.2% |
| | EnKF | 0.5266 | 0.777 | 3.7682 | **−10.094** | **+615.5%** |
| | ETKF | 0.5121 | 0.788 | 3.6984 | **−9.697** | **+622.2%** |

S0 EV remains 0.78–0.87 (good skill even with single-scale DA). On S1, EV drops to **negative values** for all methods, meaning all perform worse than climatology. Strong-4DVar: EV ≈ −1.5 (MSE is 2.5× climatological variance). EnKF/ETKF: EV ≈ −3.6 to −5.9 (with inflation 2.0), plunging to −9.7 to −10.1 when inflation is removed (b4).

**Winner**: **b2** — pure model mismatch (single-scale DA with correct F=8.0, inflation=2.0). Degradation >300% for all methods with negative explained variance.

### b2 Per-variable-group RMSE and EV (5 windows)

| Method | S0 RMSE → S1 RMSE | S0 EV → S1 EV | Slow S0→S1 EV | Fast S0→S1 EV |
|---|---|---|---|---|
| Strong-4DVar | 0.40 → 1.77 (**+336%**) | 0.868 → **−1.518** | 0.922 → 0.002 | 0.858 → −1.812 |
| EnKF | 0.52 → 2.94 (**+468%**) | 0.784 → **−5.821** | 0.913 → −3.386 | 0.759 → −6.291 |
| ETKF | 0.51 → 2.40 (**+375%**) | 0.794 → **−3.589** | 0.914 → −1.706 | 0.771 → −3.953 |

All S1 EV values are negative. Strong-4DVar's slow-variable EV is near-zero (0.002 — barely matches climatology) while fast variables go strongly negative (−1.812). EnKF/ETKF are deeply negative across both groups, indicating the single-scale model fundamentally cannot represent the two-scale dynamics regardless of variable type.

### S1 Definition

The final **S1** (model-mismatch case) is:

- **DA dynamics**: Single-scale Lorenz96 (40 slow variables, no fast variables, `NO=40`, `J=0`, `h=0.0`)
- **Truth dynamics**: Two-scale Lorenz96 (8 slow, 32 fast, `NO=8`, `J=4`, `h=1.0`)
- **Forcing**: Same F=8.0 and stochastic AR(1) forcing for both (no parameter bias)
- **Observation**: Full 40D state, `obs_interval=200`, `R_var=0.5` (same as S0)
- **EnKF/ETKF**: inflation=2.0 (same as S0)

The single-scale DA observes the full 40-dimensional state (slow + fast variables) but models it as 40 coupled slow variables with no fast-scale parametrization. This is a strong violation of the model-error-correctly-specified assumption.

---

## 200-Window Validation — b2 (single-scale DA)

200 windows, same config as b2. Results confirm the 5-window sweep with slightly moderated degradation (better statistics).

### Overall RMSE and EV

| Method | S0 RMSE | S0 EV | S1 RMSE | S1 EV | Δ% |
|---|---|---|---|---|---|
| Strong-4DVar | 0.3981 | 0.875 | 1.5696 | **−0.970** | +294.3% |
| EnKF | 0.4940 | 0.804 | 2.6514 | **−4.472** | +436.7% |
| ETKF | 0.4844 | 0.812 | 2.1383 | **−2.590** | +341.4% |

### Per-group EV

| Method | Slow S0→S1 EV | Fast S0→S1 EV |
|---|---|---|
| Strong-4DVar | 0.930 → 0.095 | 0.864 → −1.176 |
| EnKF | 0.921 → −3.346 | 0.781 → −4.690 |
| ETKF | 0.919 → −1.366 | 0.792 → −2.827 |

Strong-4DVar's slow-variable EV drops to 0.095 (barely above climatology), while fast variables go strongly negative (−1.176). All ensemble methods are deeply negative across both groups.

### Comparison with 5-window results

| Metric | Strong-4DVar (5w → 200w) | EnKF (5w → 200w) | ETKF (5w → 200w) |
|---|---|---|---|
| S0 RMSE | 0.405 → 0.398 | 0.518 → 0.494 | 0.506 → 0.484 |
| S1 RMSE | 1.766 → 1.570 | 2.943 → 2.651 | 2.404 → 2.138 |
| S0 EV | 0.868 → 0.875 | 0.784 → 0.804 | 0.794 → 0.812 |
| S1 EV | −1.518 → −0.970 | −5.821 → −4.472 | −3.589 → −2.590 |

The 200-window run gives slightly better S0 and S1 numbers (less extreme degradation) due to better sampling, but the qualitative finding holds: **all methods degrade by 290%+ with negative explained variance on S1**.

---

## Wave 3: J-mismatch with partial observations

S1 uses a **lower-resolution dynamics model** (J=2 instead of J=4) inside the DA, while the truth remains two-scale (J=4). Observations are **partial** — only 2 out of 4 fast variables per slow node are observed, giving 24 observed dimensions from the 40D state. This introduces both model mismatch (J=2 vs J=4) and a rectangular observation operator for S0 (H: 40→24).

### Setup

| Component | S0 | S1 |
|---|---|---|
| Truth dynamics | J=4, NO=8 (40D) | J=4, NO=8 (40D) |
| DA dynamics | J=4, NO=8 (40D) | **J=2, NO=8 (24D)** |
| Observation | 2/4 fast vars per node (24D) | 2/4 fast vars per node (24D) |
| Obs operator | Rectangular H (40→24) | Identity H (24→24) |
| Obs interval | 5 steps (0.005 tu) | 5 steps |
| EnKF/ETKF init | Independent noise v3 (all 4 init sites) | same |
| Inflation | 1.1 (tuned) | same |

### Common 24 variables

RMSE and EV are computed on the **common 24 variables** shared by both S0 and S1: the first 24 dimensions of the 40D truth (X[0-7], Y1[0-7], Y2[0-7]). Truth variance on this subset: slow=1.184, fast1=1.541, fast2=1.584, overall=1.436.

### Representative results (5 windows, inf=1.1)

| Config | Method | S0 RMSE | S0 EV | S1 RMSE | S1 EV |
|---|---|---|---|---|---|
| **N=30, no loc** | Strong-4DVar | 0.149 | **0.985** | 1.229 | **−0.051** |
| | EnKF | 0.265 | **0.951** | 1.034 | **0.256** |
| | ETKF | 0.300 | **0.937** | 0.881 | **0.460** |
| **N=100, no loc** | EnKF | 0.333 | 0.923 | 0.928 | 0.400 |
| | ETKF | 0.296 | **0.939** | 0.879 | **0.462** |
| **N=100, loc=4, square_root** | EnKF | 0.342 | 0.918 | 0.914 | 0.418 |
| | ETKF | 0.318 | 0.930 | 1.008 | 0.293 |
| **N=100, loc=2, per_member** | ETKF | 0.979 | 0.333 | **0.703** | **0.656** |

### Per-group EV breakdown (inf=1.1, N=30, no loc)

| Method | S0 Slow EV | S0 Fast1 EV | S0 Fast2 EV | S1 Slow EV | S1 Fast1 EV | S1 Fast2 EV |
|---|---|---|---|---|---|---|
| Strong-4DVar | 0.995 | 0.978 | 0.979 | 0.402 | −0.319 | −0.270 |
| EnKF | **0.978** | **0.931** | **0.941** | **0.683** | −0.004 | 0.022 |
| ETKF | 0.972 | 0.920 | 0.915 | **0.881** | **0.165** | **0.189** |

### Key findings

1. **Strong-4DVar S1 EV is negative** (−0.05 overall, −0.27 to −0.32 on fast vars) — same pattern as the single-scale model mismatch (Wave 2). The J-mismatch fundamentally breaks the gradient-based optimization.

2. **EnKF is the most balanced** (S0 EV=0.951, S1 EV=0.256). Slow variables retain moderate skill (S1 Slow EV=0.68), fast variables are near zero. Runs in 11s vs Strong-4DVar's 510s.

3. **ETKF without localization gives the best S1 EV** (0.460) among the three methods, with slow EV=0.881 — nearly retains slow-variable skill. Fast variables show modest but positive EV (0.17-0.19). S0 EV=0.937.

4. **Localization with per-member stochastic update** pushes ETKF S1 EV to **0.656** (best S1 by far, fast var EV=0.67-0.69) but collapses S0 EV to 0.333 (per-member stochastic noise overwhelms the rectangular-H cross-covariances).

5. **Square-root localization** fixes the S0 collapse (S0 EV=0.93) but loses the S1 benefit (S1 EV=0.29). Localization tapers cross-covariances that are already rank-deficient for the rectangular H case.

6. **Larger ensemble (N=100)** without localization does not significantly improve S0 or S1 ETKF EV over N=30, suggesting the benefit of larger ensembles is marginal given the current init and inflation tuning.

### Best config per method

| Method | Config | S0 EV | S1 EV | Runtime |
|---|---|---|---|---|
| Strong-4DVar | max_iter=10, lr=0.2 | 0.985 | −0.051 | 510s |
| **EnKF** | N=30, inf=1.1 | **0.951** | **0.256** | **11s** |
| ETKF (best S1) | N=100, inf=1.1, no loc | 0.939 | **0.462** | 9s |
| ETKF (best S1 with loc) | N=100, loc=2, per_member | 0.333 | **0.656** | 48s |

### Comparison with Wave 2 (single-scale model mismatch)

Unlike Wave 2 where all methods had deeply negative S1 EV (−1.5 to −10), the J-mismatch case preserves some skill:
- EnKF/ETKF maintain **positive S1 EV** (0.26–0.46) — the J=2 model still captures the slow dynamics reasonably well
- The J-mismatch affects fast variables most (EV near zero or slightly positive), while Wave 2's single-scale model had no fast variables at all
- Strong-4DVar fails similarly in both waves (negative S1 EV) — iterative 4DVar cannot self-correct when the dynamics model is wrong

---

## Files

- **`experiments/l96_sweep_b2-validate.json`**: 200-window validation results (Wave 2)
- **`reports/outputs/l96_clim_var.json`**: Per-variable climatological variance (40 dims)
- **`reports/compute_explained_var.py`**: Script to compute EV = 1 − MSE / Var<sub>clim</sub> from sweep JSON
- **`experiments/l96_sweep_{a1..a6,b1..b4}.json`**: 5-window sweep results (Wave 1/2)
- **`experiments/l96_sweep_kf_inf11.json`**: Best EnKF/ETKF config (N=30, inf=1.1, Wave 3)
- **`experiments/l96_sweep_kf_inf11_ens100.json`**: Large ensemble reference (N=100, Wave 3)
- **`experiments/l96_sweep_kf_loc_sr.json`**: Square-root localization (loc=2, Wave 3)
- **`experiments/l96_sweep_kf_loc4_sr.json`**: Square-root localization (loc=4, Wave 3)
- **`experiments/l96_sweep_kf_loc.json`**: Per-member localization baseline (Wave 3)