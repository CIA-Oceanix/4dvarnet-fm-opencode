## 1. DA baselines

Lorenz-63 with wind forcing, 200 test windows (seed 123, per-window parameters ±20%), 3 state channels, 15 observations per 300-step window (obs_interval 20, R = 0.5), DA sub-windows of 50 steps, 30 members. Every RMSE cell is the per-window RMSE (channel-averaged over time) on X, Y, Z, mean ± sd across windows. Hyper-parameters are selected per case on 50 validation windows (seed 99).

| scheme | params | seeds | S0 | S1 | S1/S0 | CRPS S0 (S1) | spread/RMSE (S0) | note |
|---|---|---|---|---|---|---|---|---|
| ETKF | λ 2.0 / 1.5 | — | 0.954 ± 0.503 | 2.332 ± 0.397 | 2.45 | 0.555 (1.674) | 1.27 |  |
| EnKF | λ 2.0 / 2.0 | — | 0.969 ± 0.493 | 2.333 ± 0.363 | 2.41 | 0.552 (1.560) | 1.23 |  |
| Strong-4DVar | — | — | 0.724 ± 0.888 | 2.112 ± 0.966 | 2.92 | — (—) | — |  |
| Weak-4DVar | q 0.03 / 0.3, 150 it | — | **0.637 ± 0.519** | **1.511 ± 0.479** | 2.37 | — (—) | — |  |
