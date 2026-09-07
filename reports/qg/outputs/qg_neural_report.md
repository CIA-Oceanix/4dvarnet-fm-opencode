# QG Neural Baseline (S0) — Q1 DirectUNet / Q2 VanillaCFM(τ=0)

Neural estimators are trained on the daily-mean full 2-layer **streamfunction** (ψ) with an auxiliary PV-q loss, and scored on the same daily-mean fields (ψ and PV q). The PV-q RMSE/EV are directly comparable to the QG DA baselines (`qg_repro_validation`), which are also scored on the full PV q field.

## DA baselines (S0 reference, PV-q)

| method | PV RMSE | free RMSE | improv | PV EV | PV q1 EV | PV q2 EV | ψ EV |
|---|---|---|---|---|---|---|---|
| EnKF | 6.34e-06 | 7.33e-06 | 1.1556 | 0.7487 | 0.8170 | 0.6803 | 0.9707 |
| ETKF | 6.40e-06 | 7.33e-06 | 1.1443 | 0.7470 | 0.8151 | 0.6788 | 0.9689 |
| Weak-4DVar | 4.88e-06 | 7.33e-06 | 1.5005 | 0.8048 | 0.8819 | 0.7277 | 0.9927 |
| Strong-4DVar | 5.10e-06 | 7.33e-06 | 1.4363 | 0.7727 | 0.8743 | 0.6711 | 0.9932 |
| _free forecast_ | 7.33e-06 | — | 1.0 | -- | -- | -- | — |

## QG neural estimators (S0)

Metrics on the daily-mean full 2-layer ψ and PV q. `--` = run not yet available (training/eval pending).

| model | nx | ψ RMSE | ψ EV | q RMSE | q EV | q1 EV | q2 EV |
|---|---|---|---|---|---|---|---|
| Q1 DirectUNet | -- | -- | -- | -- | -- | -- |
| Q2 VanillaCFM(τ=0) | -- | -- | -- | -- | -- | -- |

## Experiment settings

- **Q1 DirectUNet** — `--` (not run yet).
- **Q2 VanillaCFM(τ=0)** — `--` (not run yet).

