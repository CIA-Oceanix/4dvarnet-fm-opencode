# QG S0/S1 DA Baselines — Consolidated Report (cross-resolution S1)

**Date:** 2026-08-31
**Branch (report):** master
**Provenance (jobs, A40 `sl-mee-br-205`):** S0 1%-noise matrix (prior session, jobs 50927/50930/50932); S1 @ da_nx=16 (job 51069); S1 @ da_nx=32 (job 51075).
**Result JSONs (committed on master):** `reports/qg/outputs/qg_matrix_{c4,c8}_{q,psi}/`, `reports/qg/outputs/qg_s1_lag{1,2}p0/`, `reports/qg/outputs/qg_s1_da32_lag{1,2}p0/`.

## 1. Full S0/S1 settings

Two-layer quasi-geostrophic (Phillips-channel double-periodic β-plane) model, `da_model='qg2l'` (2-layer) / `qg2l_lores` (coarse 2-layer DA model). The DA truth is generated on the fly per the `QGConfig` snapshot (`reports/qg/outputs/qg_settings.json`); the expensive spinup is cached under `reports/qg_cache/` (gitignored, available locally on master).

### Base configuration (QGConfig snapshot)

| Parameter | Value |
|---|---|
| Domain length L | 1000000 m |
| Grid | nx = ny = 64, state_dim = 8192 (2×64×64, layer-major) |
| Time step | dt = 7200 s, steps_per_day = 12 |
| Assimilation window | 30 days (360 steps) |
| Spinup | 2 years |
| Windows | 5 |
| Physics β | 1.5e-11 |
| Physics rd | 15000 m |
| Physics δ (layer-depth ratio) | 0.25 |
| Physics U₁ | 0.05 |
| Physics U₂ | 0.0 |
| Physics rek (linear drag) | 5.787e-07 |
| Spectral filter | filterfac = 23.6 |
| Seed | 7 |

### Moving-storm wind forcing (upper-layer PV source)

- Wind-stress-curl amplitude `wind_amp = 1e-11` (Ornstein–Uhlenbeck, `wind_tau_days = 15` d; storm width `wind_sigma = 250` km).
- Storm-track drift `wind_cx = 0.5`, `wind_cy = 0.03` m/s; position OU jitter `wind_drift_tau_days = 10` d, `wind_drift_sigma = 50` km.

### Per-window truth randomization

- U₁, rd, rek drawn once per window as `U[1 ± param_range]` (`param_range = 0.15`); β/δ fixed.
- Independent storm per window: start `(x0,y0) ~ U(0,L)²`, track `cx ~ U[0.25, 0.75]`, `cy ~ U[−0.06, 0.06]`.
- Wind amplitude drawn from discrete levels `{0, 3e-12, 1e-11, 2e-11, 3e-11}` round-robin `i % 5`.
- Initial state at `t₀ − U(0, init_lag_days)` (lagged-truth first guess).

### Observations

- Geometry `random_columns`: `cols_per_day` distinct meridional columns of the upper-layer field, one simultaneous event/day at a random step.
- Observed field: upper-layer streamfunction ψ₁ (psi-obs) — the baseline `ObsOperator` inverts ψ to PV after spectral upsampling on the DA grid.
- Noise: `sigma = obs_noise_std_frac × std(field)`, `frac = 0.01` (1%).
- Coverage (production): cols = 4 → ~0.52% of space-time gridpoints.

### Scenarios

- **S0** (error-free): `da_params = true_params`, `da_model = 'qg2l'`, `da_nx = cfg.nx` (DA at full resolution).
- **S1** (model error): `da_model = 'qg2l_lores'`, `da_nx ∈ {16, 32}` (cross-resolution, truth at 64×64); param bias `rd,rek ← rd,rek × (1 − s1_param_bias)` (`s1_param_bias = 0.15`);
  corrupted wind: storm-centre OU jitter (`s1_loc_sigma_frac = 0.25` × wind_sigma, τ = 10 d) and amplitude `A(1+s1_amp_bias)+η` (`s1_amp_bias = 0.15`, OU η with `s1_sigma_eta_frac = 0.3`, τ = 10 d).

### DA filter

- **ETKF**, ensemble N = 80, inflation 1.0, Gaspari–Cohn localization radius 6 (physical coords on the DA grid).
- Init: lagged-truth shared by the DA ensemble and the free-forecast reference; `disp_frac = 1.0` (background-error-scaled), band ±0.25 d.
- Lags tested: 1.0 d and 2.0 d.

## 2. S0 results (1% obs noise, error-free)

| obs | cols | lag | DA RMSE | Free RMSE | improv | EV_full | EV_free |
|---|---|---|---|---|---|---|---|
| q | 4 | 1.0 | 9.47e-06 | 7.33e-06 | 0.77 | -1.400 | +0.727 |
| q | 4 | 2.0 | 1.20e-05 | 1.30e-05 | 1.08 | -2.612 | +0.301 |
| q | 8 | 1.0 | 7.43e-06 | 7.33e-06 | 0.99 | -0.539 | +0.727 |
| q | 8 | 2.0 | 8.67e-06 | 1.30e-05 | 1.50 | +0.053 | +0.301 |
| psi | 4 | 1.0 | 6.32e-06 | 7.33e-06 | 1.16 | +0.752 | +0.727 |
| psi | 4 | 2.0 | 8.11e-06 | 1.30e-05 | 1.61 | +0.652 | +0.301 |
| psi | 8 | 1.0 | 5.18e-06 | 7.33e-06 | 1.42 | +0.777 | +0.727 |
| psi | 8 | 2.0 | 6.89e-06 | 1.30e-05 | 1.89 | +0.689 | +0.301 |

## 3. S1 results — cross-resolution (da_nx=16 vs da_nx=32)

### Headline (S1, psi-obs, cols=4, 1% noise, truth 64×64)

| da_nx | ratio | lag | DA RMSE | Free RMSE | improv | EV_full | EV_free |
|---|---|---|---|---|---|---|---|
| 16 | 4:1 | 1.0 | 1.71e-05 | 1.88e-05 | 1.10 | -0.106 | -0.456 |
| 16 | 4:1 | 2.0 | 1.74e-05 | 1.93e-05 | 1.11 | -0.135 | -0.542 |
| 32 | 2:1 | 1.0 | 1.24e-05 | 1.83e-05 | 1.47 | +0.340 | -0.348 |
| 32 | 2:1 | 2.0 | 1.30e-05 | 1.92e-05 | 1.48 | +0.260 | -0.471 |

### Per-field metrics

#### S1 @ da_nx=16

**lag 1.0**

| field | layer | DA RMSE | Free RMSE | improv | EV | EV_free |
|---|---|---|---|---|---|---|
| PV q | upper (layer 1) | 2.49e-05 | 2.84e-05 | 1.14 | +0.026 | -0.267 |
| PV q | lower (layer 2) | 4.30e-06 | 4.90e-06 | 1.14 | -0.237 | -0.644 |
| PV q | full state | 1.79e-05 | 2.04e-05 | 1.14 | -0.106 | -0.456 |
| streamfunction ψ | upper (layer 1) | 1.01e+04 | 1.11e+04 | 1.11 | +0.479 | +0.334 |
| streamfunction ψ | lower (layer 2) | 8.14e+03 | 7.88e+03 | 0.97 | +0.496 | +0.502 |
| streamfunction ψ | full state | 9.15e+03 | 9.65e+03 | 1.05 | +0.488 | +0.418 |

**lag 2.0**

| field | layer | DA RMSE | Free RMSE | improv | EV | EV_free |
|---|---|---|---|---|---|---|
| PV q | upper (layer 1) | 2.53e-05 | 2.92e-05 | 1.15 | -0.002 | -0.345 |
| PV q | lower (layer 2) | 4.37e-06 | 5.07e-06 | 1.16 | -0.268 | -0.739 |
| PV q | full state | 1.82e-05 | 2.10e-05 | 1.15 | -0.135 | -0.542 |
| streamfunction ψ | upper (layer 1) | 1.07e+04 | 1.17e+04 | 1.10 | +0.406 | +0.273 |
| streamfunction ψ | lower (layer 2) | 8.75e+03 | 8.28e+03 | 0.95 | +0.399 | +0.452 |
| streamfunction ψ | full state | 9.75e+03 | 1.01e+04 | 1.04 | +0.402 | +0.362 |

#### S1 @ da_nx=32

**lag 1.0**

| field | layer | DA RMSE | Free RMSE | improv | EV | EV_free |
|---|---|---|---|---|---|---|
| PV q | upper (layer 1) | 1.86e-05 | 2.78e-05 | 1.49 | +0.458 | -0.221 |
| PV q | lower (layer 2) | 3.41e-06 | 4.68e-06 | 1.37 | +0.221 | -0.476 |
| PV q | full state | 1.34e-05 | 1.99e-05 | 1.49 | +0.340 | -0.348 |
| streamfunction ψ | upper (layer 1) | 6.24e+03 | 1.05e+04 | 1.69 | +0.801 | +0.406 |
| streamfunction ψ | lower (layer 2) | 5.13e+03 | 7.96e+03 | 1.55 | +0.806 | +0.493 |
| streamfunction ψ | full state | 5.71e+03 | 9.34e+03 | 1.63 | +0.804 | +0.449 |

**lag 2.0**

| field | layer | DA RMSE | Free RMSE | improv | EV | EV_free |
|---|---|---|---|---|---|---|
| PV q | upper (layer 1) | 1.96e-05 | 2.92e-05 | 1.49 | +0.398 | -0.348 |
| PV q | lower (layer 2) | 3.63e-06 | 4.89e-06 | 1.35 | +0.122 | -0.594 |
| PV q | full state | 1.41e-05 | 2.09e-05 | 1.49 | +0.260 | -0.471 |
| streamfunction ψ | upper (layer 1) | 6.8e+03 | 1.11e+04 | 1.64 | +0.764 | +0.344 |
| streamfunction ψ | lower (layer 2) | 5.64e+03 | 8.33e+03 | 1.48 | +0.761 | +0.449 |
| streamfunction ψ | full state | 6.25e+03 | 9.83e+03 | 1.57 | +0.763 | +0.396 |

## 4. Interpretation

- **Milder resolution mismatch (da_nx=32) is much easier DA than da_nx=16**: forecast-improv rises ~1.10 → ~1.48 and pooled EV flips from negative to positive.
- At da_nx=32 every layer beats the free forecast (`improv > 1`), with the streamfunction (ψ) most informative (upper-ψ improv ≈ 1.6–1.7; q₁ improv ≈ 1.49).
- The DA-vs-free-forecast skill scales monotonically with the cross-resolution ratio (4:1 → 2:1): the free forecast itself diverges strongly under model error (EV_free < 0), and DA recovers a large share of that signal.
