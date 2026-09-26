# Spectral wind forcing for the QG case study — basis and gyrostat calibration

**Status:** RESULTS (2026-09-26). Measurements made while implementing PR-1
of the Option B wind-forcing plan (the Fourier basis and the gyrostat
driver). Context: `docs/plans/case_study/qg_gyrostat_synoptic_forcing.md`
(Option B). The simulations are in
`reports/qg/outputs/qg_spectral_wind_report.md`.

**Numbers:** `reports/qg/outputs/qg_spectral_wind_report.md`.

## 1. Today's storm lives in 12 Fourier amplitudes

The current forcing is a σ = 250 km Mexican hat summed over periodic images
on the 1000 km, 64 × 64 grid. Its curl power spectrum was averaged over 64
random positions with `QGDynamics.wind_curl_field`:

| basis | independent real amplitudes | share of curl variance |
|---|---|---|
| `|k| ≤ 1` | 4 | 74.0% |
| `|kx|, |ky| ≤ 1` | 8 | 99.1% |
| `|k| ≤ 2` | 12 | 99.8% |
| `|k| ≤ 3` | 28 | 100.0% |

The radial spectrum peaks at domain wavenumber 1. In practice the storm is a
translating domain-scale pattern, not a localized synoptic system.

**Analytic spectrum.** The 2D Fourier transform of `(1 − r²/2σ²) e^{−r²/2σ²}`
is `π σ⁴ k² e^{−σ²k²/2}`. A storm of amplitude A at `r_c` therefore has, on
wavevector k, a cos/sin pair of radius

```
R_k = 2π σ⁴ k² e^{−σ²k²/2} / (L W)
```

and phase `k·r_c`. Averaged over positions, the per-amplitude std is
`A R_k / √2`.
- This reproduces the 74% share at `|k| ≤ 1`.
- It matches the grid projection of `wind_curl_field` to 2×10⁻³ relative
  error (`tests/test_qg_wind_modes.py`).
- `FourierWindBasis.ricker_mode_std` uses it as the target spatial spectrum
  of every spectral driver.

## 2. Gyrostat topologies

The time scales are in gyrostat units, from 2000–4000-unit runs with RK4 at
`dt = 0.005`.

| topology | modes | largest Lyapunov exponent | outcome |
|---|---|---|---|
| sparse nested chain of 6 Lorenz-63 gyrostats (roles stacked on shared modes) | 13 | −0.88 | **collapses to a fixed point.** Shared modes are over-damped; every variance is 0 |
| fan: 6 (x, y) pairs sharing one z | 13 | 0.90 | **pairs synchronize** (pairwise correlation 1.0): only 3 effective degrees of freedom |
| 4 independent Lorenz-63 copies | 12 | 0.91 | chaotic, but no cross-copy structure (mean \|corr\| 0.07) |
| **`l63ring4`**: 4 Lorenz-63 copies on a ring, energy-conserving coupling triads `(x_k, y_{k+1}, z_{k+1})` of strength 0.5 | **12** | **1.25** | **chosen**: chaotic, cross-copy structure (mean \|corr\| 0.19) |

The planned 13-mode chain (`chain6`) is therefore replaced by `l63ring4`.
This is the fallback for risk R1 in the plan. Its 12 modes map one-to-one
onto the 12 amplitudes of `|k| ≤ 2`.

## 3. Attractor constants (stored in `models/gyrostat_driver.py`)

| preset | mode | mean | std | ACF e-folding (units) | kurtosis |
|---|---|---|---|---|---|
| `l63` | x, y, Z | 0, 0, −14.452 | 7.924, 9.011, 8.624 | 0.31, 0.22, 0.14 | 2.30, 2.85, 2.14 |
| `l63ring4` | x, y, Z (per copy, pooled) | 0, 0, −10.195 | 7.320, 8.758, 7.327 | 0.241, 0.165, 0.115 | 1.95, 2.61, 2.59 |

Notes:
- The x and y means are 0 by the `(x, y) → (−x, −y)` symmetry; measured
  values were 0.03–0.17, which is sampling noise.
- Ring statistics are pooled over the four copies, since the ring is
  invariant under cyclic shifts.
- `Z = z − (ρ + σ)`.
- Lobe residence (time between sign changes of x): `l63` mean 1.74, median
  1.53; `l63ring4` mean 0.83, median 0.46.
- A slow test re-measures the constants on an independent 1000-unit run
  (std within 5%, mean within 0.1 std, Lyapunov exponent > 0.5).

## 4. Calibration of the drivers

**Time unit:** 15 d / 0.241 = **62.2 days per gyrostat unit**, so that the
gyrostat amplitudes have the storm's 15-day memory.

**Mode mapping** (`xz_pairs`):
- wavevectors (1,0), (0,1), (1,1), (1,−1) take the (x, Z) modes of copies
  0–3;
- (2,0) and (0,2) take (y₀, y₁) and (y₂, y₃).

The bimodal x modes therefore feed the most energetic, lowest wavenumbers.

**Validation** on 40-year amplitude-only series (6-hour sampling, no drift),
for the (1,0) cos amplitude:

| driver | e-folding (days) | kurtosis | rms per wavevector / storm target |
|---|---|---|---|
| `ou` (current storm) | 14.8 | 2.95 | 1.00 (all six) |
| `spectral_ou` | 14.8 | 3.20 | 1.00–1.03 |
| `gyrostat` | 15.8 | 1.98 | 0.98–1.00 |
| `gyrostat_surrogate` | 15.0 | 2.93 | 1.00–1.02 |

What this shows:
- The memory and the spatial spectrum are matched across drivers.
- **The gyrostat amplitude is bimodal** (kurtosis 1.98); its surrogate is
  Gaussian (2.93) at the same spectrum. This is the contrast the E1 vs E2
  comparison is built to test.
- **The gyrostat amplitudes also have a long-memory tail.** Their ACF is
  still ≈ 0.3 at 60–90 days, while the OU drivers' ACF has decayed to about 0
  by 50 days. The surrogate shares the tail, since it shares the spectrum.
  That makes E0b vs E2 a test of time-spectrum shape at fixed e-folding.
