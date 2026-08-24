# L96 S0/S1 DA Baseline: Observation Density — S0/S1-Obs200 vs S0-Obs100/S1-Obs100

**Date:** 2026-08-19
**Config:** two-scale Lorenz-96, NO=8, J=4, obs_j=2 → 24D observed subspace
(8 slow + 16 fast Y1,Y2 per node); truth 40D, `fast_weights=[1,1,0.1,0.1]`.
Per-window all-5-param randomization (F, c1, h, hx, eps) ±20%; S1 adds ±10% model bias
(DA uses biased `*_da` params). DWS=500, inflation=2.0, 200 test windows.
Groundtruth identical across densities (same seeds); only the observation time grid differs
(`obs_interval` 200 → 100 gives 15 → 30 obs/window).

Metrics are pooled over windows: RMSE (dim-mean of pooled MSE)^1/2 and pooled explained
variance `EV = 1 − mean(SqErr)/var(ref)` on the 24D observed subspace, split into
slow (8D), obs_fast (16D), all_obs (24D).

## S0 (clean, all-5 ±20%)

| Method | RMSE(all) | EV all | EV slow | EV obs_fast |
|---|---|---|---|---|
| **Obs200** | | | | |
| EnKF | 1.0927 | +0.544 | +0.890 | +0.371 |
| ETKF | 1.0973 | +0.538 | +0.892 | +0.361 |
| Strong-4DVar | 0.9701 | +0.586 | +0.923 | +0.417 |
| **Obs100 (2× denser)** | | | | |
| EnKF | 0.9046 | +0.680 | +0.935 | +0.552 |
| ETKF | 0.8815 | +0.692 | +0.940 | +0.569 |
| Strong-4DVar | 0.7788 | +0.729 | +0.941 | +0.622 |

## S1 (model bias +10% on DA params)

| Method | RMSE(all) | EV all | EV slow | EV obs_fast |
|---|---|---|---|---|
| **Obs200** | | | | |
| EnKF | 1.6503 | +0.022 | +0.498 | −0.215 |
| ETKF | 1.6367 | +0.036 | +0.516 | −0.204 |
| Strong-4DVar | 1.4751 | +0.205 | +0.637 | −0.011 |
| **Obs100 (2× denser)** | | | | |
| EnKF | 1.5022 | +0.201 | +0.545 | +0.029 |
| ETKF | 1.4680 | +0.236 | +0.561 | +0.074 |
| Strong-4DVar | 1.4276 | +0.256 | +0.674 | +0.047 |

## Findings

- **S0:** Densifying observations (200 → 100) improves every method: all_obs EV rises
  +0.10–0.14 (Strong-4DVar best, +0.729), RMSE drops ~0.19. Skill concentrates in the slow
  variables (EV ~0.94); the fast-variable subspace improves most (obs_fast EV +0.55–0.62 vs
  +0.36–0.42).
- **S1:** The density gain is relatively larger. All_obs EV climbs from ~0.02–0.04 (Obs200) to
  ~+0.20–0.26 (Obs100), mainly because obs_fast EV turns **positive** (+0.03–0.07 vs −0.20)
  while slow EV is largely unchanged. More data partially compensates for model error.
- Strong-4DVar remains the strongest method at both densities; EnKF ≈ ETKF.

## Artifacts

- `experiments/l96_baselines_dws500_inf2.0_etkf_inf2.0_obsj2.json` — S0/S1 Obs200 cache (EV backfilled)
- `experiments/l96_baselines_dws500_inf2.0_etkf_inf2.0_obsj2_int100.json` — S0-Obs100/S1-Obs100 cache
- `experiments/l96_datasets_obsj2_nwin200.pt` / `..._int100_nwin200.pt` — datasets (same trajectories)
