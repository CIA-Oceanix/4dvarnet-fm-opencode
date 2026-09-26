# L96 PredictStateCFM with a linear skip connection

**Status:** RESULTS (2026-09-26). Two 1200-epoch PredictStateCFM-M runs (benchmark-default data, seed 1) with D = τ·x_τ + (1 - τ)·net, one per loss. Code: `skip_connection` / `skip_loss` options of `PredictStateCFM` (`models/vanilla_cfm.py`), configs `L96B_predictstatecfm_monaiM_ep1200_skip_{state,net}loss`, `batch/run_l96b_psc_skip.sbatch`.

**Numbers:** `batch/run_l96b_psc_skip.sbatch` (`eval_neural_l96.py`, benchmark sampler); per-run eval JSONs and estimates are archived under the canonical L96 archive (`experiments/l96/L96B_predictstatecfm_monaiM_ep1200_skip_*`), not checked in.

## Question

Does an EDM-style skip parameterization of PredictStateCFM's denoiser -- exact at τ = 1, unchanged
at τ = 0, sampler velocity (D - x)/(1 - τ) = net - x -- improve the flow? Two losses:
`skip_loss: state` trains MSE(D, x1) (net weighted by (1 - τ)²); `skip_loss: net` trains the
unweighted MSE(net, (1 + τ) x1 - τ x0). Validation always uses MSE(D, x1).

## Results

Benchmark flow sampler (30 members x 20 early-fine steps), 200 test windows, per-window means
(RMSE / CRPS / spread-over-RMSE):

| 1200 ep, seed 1 | regular S0 | regular S1 | random S0 | random S1 |
|---|---|---|---|---|
| PredictStateCFM-M | 0.340 / 0.150 / 0.63 | 0.337 | 0.441 / 0.192 / 0.65 | 0.445 |
| PredictStateCFM-M, 3-seed mean | 0.341 / 0.150 / 0.63 | 0.338 | 0.443 / 0.194 / 0.64 | 0.447 |
| skip + state loss | 0.343 / 0.151 / 0.68 | 0.341 | 0.441 / 0.192 / 0.68 | 0.447 |
| skip + net loss | 0.350 / 0.150 / 0.71 | 0.348 | 0.459 / 0.199 / 0.70 | 0.466 |
| VanillaCFM-M | 0.351 / 0.151 / 0.72 | 0.346 | 0.464 / 0.203 / 0.69 | 0.470 |
| VanillaCFM-M, 3-seed mean | 0.351 / 0.152 / 0.71 | 0.349 | 0.462 / 0.202 / 0.68 | 0.468 |

The net-loss run had the lowest validation loss (0.0032 vs 0.0038-0.0039) but the worst RMSE of
the three PredictStateCFM variants: validation MSE(D, x1) over uniform τ does not rank sampled
ensembles.

## Why: the skip changes the parameterization, the loss weight decides the model

With v̂ = net - x_τ and the velocity target v* = x1 - x0:

- net loss: the target (1 + τ) x1 - τ x0 = v* + x_τ, so MSE(net, target) = MSE(v̂, v*) -- exactly
  VanillaCFM's loss, with VanillaCFM's sampler (v = net - x). Only the output parameterization
  differs (v + x instead of v).
- state loss: D - x1 = (1 - τ)(v̂ - v*), so MSE(D, x1) is the same velocity regression weighted by
  (1 - τ)² -- plain PredictStateCFM's loss, skip or not.

The two runs land on VanillaCFM and PredictStateCFM respectively. The skip parameterization changes
little (at most slightly more spread, 0.68 vs 0.63, at equal RMSE); what separates PredictStateCFM
from VanillaCFM is the τ weighting of one velocity regression -- (1 - τ)² emphasises early τ and
gives the better RMSE and tighter ensembles, the uniform weight ~3-4% worse RMSE and more spread.
The follow-up on combining the two families at sampling time is
`docs/results/l96_cfm_velocity_ensembles.md`. Single seed per arm: differences below ~0.005 RMSE are
within seed noise.
