## 1. DA baselines

Lorenz-63 with wind forcing, 200 test windows (seed 123, per-window parameters ±20%), 3 state channels, 30 observations per 300-step window (obs_interval 10, R = 0.5), DA sub-windows of 50 steps, 30 members. Every RMSE cell is the per-window RMSE (channel-averaged over time) on X, Y, Z, mean ± sd across windows. Hyper-parameters are selected per case on 50 validation windows (seed 99).

| scheme | params | seeds | S0 | S1 | S1/S0 | CRPS S0 (S1) | spread/RMSE (S0) | note |
|---|---|---|---|---|---|---|---|---|
| ETKF | λ 2.0 / 4.0 | — | 0.720 ± 0.355 | 1.502 ± 0.248 | 2.09 | 0.432 (0.930) | 1.47 |  |
| EnKF | λ 2.0 / 4.0 | — | 0.730 ± 0.344 | 1.510 ± 0.246 | 2.07 | 0.432 (0.940) | 1.42 |  |
| Strong-4DVar | — | — | 0.616 ± 0.793 | 1.864 ± 0.314 | 3.02 | — (—) | — |  |
| Weak-4DVar | q 0.03 / 1.0, 150 it | — | **0.471 ± 0.194** | **0.895 ± 0.161** | 1.90 | — (—) | — |  |
| ETKS | λ 1.5 / 4.0 | — | 0.521 ± 0.454 | 1.203 ± 0.180 | 2.31 | 0.292 (0.717) | 0.65 |  |
| *ETKS, 100 members* | λ 1.5 / 4.0 | — | 0.512 ± 0.435 | 1.192 ± 0.174 | 2.33 | 0.285 (0.707) | 0.66 | ensemble size: 100 members (the others use 30), inflation re-selected on the validation windows |
