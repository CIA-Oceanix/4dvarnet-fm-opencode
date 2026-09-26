# L96 flow blends and velocity ensembles (VanillaCFM / PredictStateCFM)

**Status:** RESULTS (2026-09-26). Sampling-time blends and ensembles of trained 1200-epoch benchmark-default flows. Code: `models/cfm_blend.py`, `eval_neural_l96.py --blend-with ... [--blend-schedule | --blend-weights]`.

**Numbers:** `eval_neural_l96.py` blend runs (`--blend-with`); their per-run eval JSONs are not checked in. Reference rows: `reports/l96/outputs/l96_benchmark_extended.md`.

## Question

VanillaCFM (VC) and PredictStateCFM (PSC) integrate the same probability-flow ODE from the same
prior on the same grid; they differ only in the velocity parameterization (v directly vs
v = (D - x)/(1 - τ)), and their losses are the same velocity regression weighted 1 (VC) and
(1 - τ)² (PSC). PSC has the better RMSE, VC the larger spread. Does a τ-varying blend of the two
velocity fields, v = λ(τ) v_PSC + (1 - λ(τ)) v_VC, beat both? And does a τ-varying weighting of
several networks beat their plain average?

## Protocol

Benchmark flow sampler (30 members x 20 early-fine steps), per-window metrics of the benchmark
reports (`G(...)["all_obs"]`: RMSE on the 24D observed space, ensemble CRPS, spread/RMSE).
Schedules selected on the 50 validation windows (regular + random layout, S0 + S1, mean CRPS),
then scored once on the 200 test windows (regular grid + canonical random set). Two-network rows
pool three seed pairs per window; three-network rows use the only combination of the three seeds.

## Results

**VC + PSC blend (validation, seed 1).** The best schedules hand over from PSC to VC:
λ = (1 - τ)² (selected; CRPS 0.1678) and a switch at τ = 0.2 (0.1680), ahead of a constant
50/50 mix (0.1694) and far ahead of the reverse order λ = τ (0.1773); PSC alone 0.1745, VC alone
0.1809. The same-family control -- PSC seed 1 + seed 2 at 50/50 -- reaches 0.1681: the blend's
gain is two-network model averaging, not the τ schedule.

**Test** (RMSE / CRPS / spread-over-RMSE; mean over windows):

| scheme (1200 ep) | networks | regular S0 | regular S1 | random S0 | random S1 |
|---|---|---|---|---|---|
| VanillaCFM-M | 1 | 0.351 / 0.152 / 0.71 | 0.349 / 0.151 / 0.72 | 0.462 / 0.202 / 0.68 | 0.468 / 0.205 / 0.66 |
| PredictStateCFM-M | 1 | 0.341 / 0.150 / 0.63 | 0.338 / 0.148 / 0.63 | 0.443 / 0.194 / 0.64 | 0.447 / 0.196 / 0.61 |
| VC x2 (50/50) | 2 | 0.344 / 0.148 / 0.70 | 0.341 / 0.147 / 0.71 | 0.453 / 0.198 / 0.67 | 0.459 / 0.202 / 0.65 |
| VC + PSC, λ = (1 - τ)² | 2 | 0.333 / 0.143 / 0.66 | 0.330 / 0.142 / 0.66 | 0.437 / 0.189 / 0.65 | 0.442 / 0.192 / 0.62 |
| PSC x2 (50/50) | 2 | 0.330 / 0.144 / 0.63 | 0.327 / 0.143 / 0.63 | 0.433 / 0.188 / 0.63 | 0.436 / 0.191 / 0.60 |
| VC x3 | 3 | 0.341 / 0.146 / 0.70 | 0.338 / 0.146 / 0.70 | 0.450 / 0.197 / 0.67 | 0.456 / 0.200 / 0.64 |
| **PSC x3** | 3 | **0.327 / 0.142** / 0.63 | **0.323 / 0.141** / 0.63 | **0.429 / 0.186** / 0.63 | **0.432 / 0.189** / 0.60 |
| PSC x3 + VC x3 | 6 | 0.329 / 0.141 / 0.66 | 0.325 / 0.140 / 0.66 | 0.434 / 0.189 / 0.65 | 0.439 / 0.192 / 0.62 |

Paired per window: PSC x2 beats a single PSC in ~100% of windows; the VC + PSC blend beats PSC x2
on RMSE in only 21-26% (CRPS tied); PSC x3 beats PSC x2 in 99-100%.

**Random τ-varying weights over PSC x3.** Ten schedules (Dirichlet(1) simplex weights at
τ = 0, 0.25, ..., 1, linearly interpolated; `random_weight_schedule`, seeds 0-9): on validation
none beats equal weights (0.1668); the best (seed 8) ties, seven are worse (up to 0.1687). CRPS
tracks the mean effective number of models 1/Σw² (correlation -0.47; equal weights = 3). On test,
seed 8 ties equal weights (mean RMSE difference ≤ 3e-4, ~50% of windows either way).

## Conclusions

1. What improves a flow at sampling time is **averaging independently trained networks with equal
   weights**: PSC regular S0 RMSE 0.341 -> 0.330 -> 0.327 for 1, 2, 3 networks, calibration
   unchanged, diminishing returns.
2. **τ-dependent weighting adds nothing** over that average -- neither the VC -> PSC hand-over nor
   random schedules. The seeds are exchangeable, so unequal weights only shrink the effective
   ensemble. The hand-over direction (PSC early, VC late) is still the better one within a VC + PSC
   pair, consistent with PSC's (1 - τ)² loss weighting.
3. VC ensembles gain less than PSC ensembles; PSC x3 is the best flow ensemble (`PredictStateCFM-M x3`
   row of `reports/l96/outputs/l96_benchmark_extended.md`). Any N-network row costs N x a single model.
