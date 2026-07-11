# L96 Baseline Report — Observation Config & Trajectory Examples

**Date**: 2026-07-11
**System**: Two-scale Lorenz96 (NO=8, J=4, state_dim=40)
**Model**: `Lorenz96Dynamics` with RK4 integration, `dt=0.001`, T_max=3.0 (3000 steps/window)
**Forcing**: F=8.0, coupling exponent=1.6, AR(1) stochastic forcing with sinusoidal component
**Observations**: Full-state (all 40 dims), `obs_interval=200` (every 0.2 tu), R_var=0.5
**Climatological variance** (200k-step free run): slow vars ~1.05, fast vars ~1.36, overall ~1.30

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