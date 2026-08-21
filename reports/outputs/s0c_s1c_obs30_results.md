# S0c/S1c DA Baseline Results — Obs30 (obs_interval=100)

**Configuration:** obs_j=2 (24D observed subspace), dws=500, 200 windows, obs_interval=100 (30 obs/window)
**S0c:** all params randomized ±20% EXCEPT h (h fixed at reference); DA uses true params
**S1c:** same true dynamics; DA forward has biased non-h params (+10%), h unbiased

## RMSE

| Method | Case | RMSE (all) | RMSE (slow) | RMSE (obs_fast) |
|--------|------|-----------|------------|----------------|
| Strong-4DVar | S0 | 0.7418 | 0.4504 | 0.8876 |
| EnKF | S0 | 0.8916 | 0.4847 | 1.0951 |
| ETKF | S0 | 0.8641 | 0.4690 | 1.0617 |
| Strong-4DVar | S1 | 1.4319 | 1.0553 | 1.6202 |
| EnKF | S1 | 1.5059 | 1.2397 | 1.6390 |
| ETKF | S1 | 1.4715 | 1.2288 | 1.5929 |

## Explained Variance (EV)

| Method | Case | EV (all) | EV (slow) | EV (obs_fast) |
|--------|------|---------|-----------|--------------|
| Strong-4DVar | S0 | 0.7453 | 0.9424 | 0.6467 |
| EnKF | S0 | 0.6768 | 0.9349 | 0.5478 |
| ETKF | S0 | 0.6932 | 0.9391 | 0.5703 |
| Strong-4DVar | S1 | 0.2400 | 0.6796 | 0.0202 |
| EnKF | S1 | 0.1817 | 0.5639 | -0.0094 |
| ETKF | S1 | 0.2207 | 0.5717 | 0.0452 |

## Key Observations

- **S0 (parametric variability):** All methods achieve EV > 0.67. Strong-4DVar leads (RMSE 0.74, EV 0.75).
- **S1 (model error):** EV drops to 0.18-0.24 for all methods. Slow variables retain skill (EV 0.56-0.68) but obs_fast variables are near zero.
- **Obs30 vs Obs15:** Denser observations improve S0 skill significantly (Strong-4DVar RMSE 0.93→0.74) but S1 improvement is modest (1.48→1.43), confirming that model error dominates at S1 regardless of observation density.
