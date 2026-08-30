# L96 TweedieCFM Benchmark — V2 family vs V3 reference

Setup: two-scale L96, Obs30 (`obs_interval=100`, `obs_j=2` → 24D observed space), dws=500, 200 shared cached test windows; S1 = ±20% params + ±10% bias. All schemes are the [V2/V3 CFM variants](phase_B_l96_cfm_variants.md); DA baselines are covered in `l96_consolidated_benchmark.md`.

Single-sample (N=1) metrics are read from each experiment's root `neural_eval.json`; the ens30×10 (N=30) tables use the shared `ens30_no10` run (10 Euler steps, fresh x₀ per member). ES/spread for ens30 rows are proper ensemble scores (MAE − 0.5·pairwise). **bold** marks the best value per column.

## Schemes

| ID | Type | K_inner | σ_prior | Description |
|---|---|---|---|---|
| **V2** | TweedieCFM | 5 | 0.5 | published |
| **V2rerun** | TweedieCFM | 5 | 0.5 | rerun (post-fix) |
| **V2s0p2** | TweedieCFM | 5 | 0.2 | σ_prior=0.2 ablation (#7) |
| **V2kinner1** | TweedieCFM | 1 | 0.5 | K_inner=1 ablation (#4) |
| **V3** | PredictStateCFM | n/a | n/a | PredictStateCFM reference |
| **L2b** | vanilla CFM | n/a | n/a | vanilla CFM, τ=0 (conditional-mean) |
| **L3** | vanilla CFM | n/a | n/a | vanilla CFM, multi-τ |

## Single-sample (N=1)

### RMSE (lower is better)

| Method | S0 | S1 | S1/S0 |
|---|---|---|---|
| V2 | 0.7219 | 0.7230 | 1.001 |
| V2rerun | 0.4865 | 0.4838 | 0.994 |
| V2s0p2 | 0.4788 | 0.4768 | 0.996 |
| V2kinner1 | 0.5360 | 0.5417 | 1.011 |
| V3 | 0.6527 | 0.6576 | 1.008 |
| L2b | 0.6329 | 0.6334 | 1.001 |
| L3 | 0.6876 | 0.6904 | 1.004 |

### EV (higher is better)

| Method | S0 | S1 |
|---|---|---|
| V2 | 0.8173 | 0.8153 |
| V2rerun | 0.9070 | 0.9076 |
| V2s0p2 | 0.9101 | 0.9103 |
| V2kinner1 | 0.8890 | 0.8860 |
| V3 | 0.8423 | 0.8391 |
| L2b | 0.8527 | 0.8511 |
| L3 | 0.8278 | 0.8251 |

## ens30×10 (N=30, proper ensemble)

### RMSE (lower is better)

| Method | S0 | S1 | S1/S0 |
|---|---|---|---|
| V2 | 0.5156 | 0.5170 | 1.003 |
| V2rerun | 0.4693 | 0.4665 | 0.994 |
| V2s0p2 | 0.4728 | 0.4707 | 0.996 |
| V2kinner1 | 0.5097 | 0.5152 | 1.011 |
| V3 | 0.5715 | 0.5728 | 1.002 |
| L2b | 0.6290 |   —   | n/a |
| L3 | 0.5643 | 0.5667 | 1.004 |

### EV (higher is better)

| Method | S0 | S1 |
|---|---|---|
| V2 | 0.8974 | 0.8962 |
| V2rerun | 0.9132 | 0.9133 |
| V2s0p2 | 0.9117 | 0.9118 |
| V2kinner1 | 0.8976 | 0.8944 |
| V3 | 0.8765 | 0.8749 |
| L2b | 0.8544 |   —   |
| L3 | 0.8788 | 0.8771 |

### ES (lower is better)

| Method | S0 | S1 |
|---|---|---|
| V2 | 0.2664 | 0.2681 |
| V2rerun | 0.2222 | 0.2208 |
| V2s0p2 | 0.2332 | 0.2337 |
| V2kinner1 | 0.2438 | 0.2471 |
| V3 | 0.2762 | 0.2766 |
| L2b | 0.3711 |   —   |
| L3 | 0.2649 | 0.2671 |

### Spread (higher = more diverse ensemble)

| Method | S0 | S1 |
|---|---|---|
| V2 | 0.4971 | 0.4972 |
| V2rerun | 0.1833 | 0.1832 |
| V2s0p2 | 0.1385 | 0.1386 |
| V2kinner1 | 0.2017 | 0.2018 |
| V3 | 0.2484 | 0.2499 |
| L2b | 0.0617 |   —   |
| L3 | 0.2776 | 0.2776 |

## Findings

- **Group A fix materially improves V2**: the post-fix rerun (K_inner=5, σ_prior=0.5) at ens30×10 S0/S1 0.4693/0.4665 beats the pre-fix published V2 (0.5156/0.5170), because the correct stage-2 checkpoint selection yields a genuinely better model. Both runs use identical config.
- **σ_prior=0.2 (#7) is neutral-to-marginal**: essentially ties the rerun on RMSE (S0 0.4728 vs 0.4693) with a tighter ensemble (spread 0.139 vs 0.183) but a marginally higher ES.
- **K_inner=1 (#4) is clearly worse**: S0 0.5097 vs rerun 0.4693 (+8.6%), higher spread — iterative mean refinement (K_inner=5) matters.
- **TweedieCFM beats vanilla CFM at ens30×10**: the best TweedieCFM (rerun/s0p2, S0 0.4728) is below the vanilla τ=0 CFM L2b (0.6290) and multi-τ L3 (0.5643), and V3 PredictStateCFM (0.5715).

## Consistency check

- Neural stored truth vs dataset true_state[:, obs_var_indices]: max |Δ| = 0.00e+00 → PASS