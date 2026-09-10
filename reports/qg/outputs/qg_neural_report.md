# QG Case Study: Benchmarked Schemes Overview

Two-layer quasi-geostrophic (QG) Phillips-channel case study -- the four DA baselines (ETKF, EnKF, Strong-4DVar, Weak-4DVar) plus the Q1 neural estimator (DirectUNet), on the S0 (no model error) reference case.

## 1. Benchmarked schemes

| scheme | type | key config | description |
|---|---|---|---|
| ETKF | Ensemble DA (deterministic square-root) | N=80, inflation=1.0, loc_radius=6.0 | Ensemble Transform Kalman Filter -- deterministic ensemble-square-root analysis update, sequentially cycled over the assimilation window. No stochastic observation perturbation. |
| EnKF | Ensemble DA (stochastic, perturbed-obs) | N=80, inflation=1.0, loc_radius=6.0 | Perturbed-observation Ensemble Kalman Filter -- each ensemble member assimilates an independently perturbed observation. Same hyperparameters as ETKF for a controlled comparison. |
| Strong-4DVar | Variational DA (deterministic, perfect-model) | window=60 steps, LBFGS, max_iter=60, b_var_scale=1.0 | 4D-Var assuming the DA dynamical model is exact over the assimilation window (strong constraint) -- optimizes only the initial condition. |
| Weak-4DVar | Variational DA (deterministic, weak-constraint) | window=60 steps, LBFGS, max_iter=60, b_var_scale=1.0, q_var_scale=0.1 | 4D-Var with an added per-step model-error control term (weak constraint) -- can partially compensate for a biased/mismatched dynamical model, at the cost of a larger control space. |
| Q1 (DirectUNet) | Neural (deterministic, single-pass, supervised) | MONAI circular 2D U-Net, hidden=[64,128,256] (M tier), cosine LR, 200 epochs | Direct single forward-pass estimator mapping observations to the full state (no iterative assimilation cycle, no dynamical model at inference time). Circular-padded Conv2d over the doubly-periodic (ny,nx) grid; trained via supervised regression on a combined psi + weighted PV-q loss. |

DA-method reference-case detail (full lag/noise sensitivity analysis, per-layer breakdown, reconstruction figures) is in `qg_da_report.md` (`generate_qg_da_report.py`) -- not repeated here.

## 2. Summary metrics (S0 reference case, N=100)

Pooled EV (higher is better) on ψ (streamfunction, both layers) and PV-q (both layers), plus normalized CRPS where available.

| scheme | ψ EV | PV-q EV | CRPS (norm.) |
|---|---|---|---|
| EnKF | 0.9474 | 0.4812 | 0.2802 |
| ETKF | 0.9212 | 0.4050 | 0.3200 |
| Weak-4DVar | 0.9660 | -0.0345 | 0.4524* |
| Strong-4DVar | 0.9714 | -0.1257 | 0.4538* |
| Q1 (DirectUNet) † | 0.8971 | 0.1605 | -- |

\* CRPS is deterministic (degenerates to MAE, no ensemble spread).

† **Not apples-to-apples**: Checkpoint trained at lag=1.0d/noise=0.01 (train_qg_neural.py defaults); this eval redraws obs at lag=5.0d/noise=0.05 to match the DA baselines' reference case, but the model was NOT retrained for this distribution -- not a fully fair comparison.

