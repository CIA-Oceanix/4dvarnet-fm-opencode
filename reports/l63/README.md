# Lorenz-63 benchmark results

Replication of the L96 benchmark summary (`reports/l96/outputs/p1_l96_benchmark.md`) on Lorenz-63 with wind forcing.

**Setup.** 200 test windows (seed 123, per-window parameters ±20%) of 300 steps, 3 state channels, 30 observations per window (obs_interval 10, R = 0.5). S0 = perfect model; S1 = biased DA parameters (param_bias 0.15, forcing_state_bias 0.1, corrupted forcing). Learned models with `param_dim: 0` ignore forcing and parameters, so their S1 equals S0 by construction. Learned models: 3 seeds, averaged per window. RMSE is the per-window RMSE (channel-averaged over time), mean ± sd across windows. CRPS is the analysis CRPS. spread/RMSE is the window-mean ensemble spread over the window-mean RMSE.

Flow ensembles: 30 members × 20 early-fine Euler steps (`step_power` 0.5). SDA: 30 members × 10 steps with DPS-style guidance; guidance weight (and tau0 for the hybrid) tuned on 50 validation windows (seed 99), seed 1.

## 1. DA baselines

| scheme | S0 | S1 | S1/S0 | CRPS S0 (S1) | spread/RMSE (S0) |
|---|---|---|---|---|---|
| ETKF | 0.720 ± 0.355 | 1.502 ± 0.248 | 2.09 | 0.432 (0.930) | 1.47 |
| EnKF | 0.730 ± 0.344 | 1.510 ± 0.246 | 2.07 | 0.432 (0.940) | 1.42 |
| Strong-4DVar | 0.616 ± 0.793 | 1.864 ± 0.314 | 3.02 | n/a | n/a |
| Weak-4DVar | **0.471 ± 0.194** | **0.895 ± 0.161** | 1.90 | n/a | n/a |
| ETKS | 0.521 ± 0.454 | 1.203 ± 0.180 | 2.31 | 0.292 (0.717) | 0.65 |
| *ETKS, 100 members* | 0.512 ± 0.435 | 1.192 ± 0.174 | 2.33 | 0.285 (0.707) | 0.66 |

Details and hyper-parameters: `outputs/l63_da_baselines_table.md`.

## 2. Deterministic point estimators

| model | S0 = S1 |
|---|---|
| DirectUNet-M | 0.437 ± 0.053 |
| PredictStateCFM-M, tau=0 mean | 0.438 ± 0.056 |
| VanillaCFM-M, tau=0 mean | 0.419 ± 0.051 |

## 3. Flow matching

| model | S0 = S1 RMSE | CRPS | spread/RMSE |
|---|---|---|---|
| PredictStateCFM-M, 1200 ep | 0.436 ± 0.066 | 0.258 | 0.60 |
| VanillaCFM-M, 1200 ep | 0.389 ± 0.050 | 0.221 | 0.83 |
| PredictStateCFM-M, 400 ep | 0.402 ± 0.057 | 0.233 | 0.67 |
| VanillaCFM-M, 400 ep | **0.377 ± 0.051** | **0.213** | 0.87 |
| PredictStateCFM-M ×3 (velocity average) | 0.396 ± 0.060 | 0.232 | 0.63 |

## 4. SDA (score-based prior + guidance)

| model | S0 RMSE | S1 RMSE | S1/S0 | CRPS S0 | spread/RMSE S0 | tuned |
|---|---|---|---|---|---|---|
| SDA1-M | 0.727 ± 0.178 | 0.727 ± 0.178 | 1.00 | 0.409 | 0.67 | gw 25 |
| SDA2-M | 0.689 ± 0.082 | 0.938 ± 0.153 | 1.36 | 0.424 | 0.62 | gw 25 |
| SDA3-fix-M | 0.656 ± 0.086 | 0.728 ± 0.124 | 1.11 | 0.408 | 0.55 | gw 25 |
| Hybrid DirectUNet-M → SDA3-fix-M | 0.402 ± 0.053 | 0.398 ± 0.053 | 0.99 | 0.239 | 0.58 | tau0 0.2, gw 0.5 |

SDA2 is conditioned on forcing and the true parameters, SDA3-fix on forcing and noisy DA parameters. Both are trained on windows whose DA parameters and forcing are randomly biased relative to the truth (`train_bias_mode: random_da`). The hybrid uses the DirectUNet-M estimate as a warm start for SDA3-fix-M guided sampling.

## Observations

- VanillaCFM-M at 400 epochs is the best learned model on RMSE (0.377) and CRPS (0.213).
- On L63, 400 epochs beat 1200 for both flows, the opposite of L96. Whether this comes from overfitting to the frozen training noise has not been tested.
- Vanilla beats PredictState at both training lengths. Every flow ensemble except PredictState at 1200 epochs beats DirectUNet-M (0.437).
- Flow ensembles are under-dispersed (spread/RMSE 0.60-0.87); Vanilla is the closest to calibrated.
- All DA baselines degrade by a factor of 1.9-3.0 on S1; the learned models do not see the biased parameters.
- Plain SDA (0.66-0.73) is clearly worse than the flows and DirectUNet-M. The plain SDA2/SDA3-fix S1 ensembles are overconfident (spread/RMSE 0.28 and 0.40).
- The hybrid reaches 0.402 on S0 and 0.398 on S1, but relies on the DirectUNet warm start. Its tuned gw (0.5) is at the lower edge of the grid.

## Reproduction

Per-model JSON outputs are in `outputs/` (`L63B_*_eval.json`, `*_tau0_mean.json`, `*_ens30_no20.json`, `*_x3_ens30_no20.json`, `*_guided_ens30_no10.json`). Scripts: `generate_da_baselines_table.py`, `eval_directunet.py`, `eval_tau0_mean.py`, `eval_flow_ensemble.py`, `eval_flow_x3.py`, `eval_sda.py`. Configs: `config/experiment/L63B_*.yaml`.
