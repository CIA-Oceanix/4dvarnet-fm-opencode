# L3 multi-tau CFM: 5-seed reproducibility (S0, N=30 members)

Five independent 30-member ensembles (seeds 1-5) vs the original seed-0 run, for 1-step (n_outer=1, `x0 + v(x0;tau=0)`) and 10-step (n_outer=10, full tau-march 0->1) integration of the L3 multi-tau CFM checkpoint on the cached S0 test set (200 windows, Obs30, 24D). Fresh x0 per member; tau schedule is deterministic (not random) at inference.

**RMSE convention**: mean over 24 dims of sqrt(mean over (W,T) of err^2), matching `evaluate_estimates`.

## 1-step (n_outer=1)

| run | RMSE | EV | ES |
|---|---|---|---|
| seed1          | 0.6501 | 0.8451 | 0.4143 |
| seed2          | 0.6501 | 0.8451 | 0.4142 |
| seed3          | 0.6500 | 0.8452 | 0.4142 |
| seed4          | 0.6506 | 0.8449 | 0.4145 |
| seed5          | 0.6503 | 0.8450 | 0.4144 |
| seed0 (orig)   | 0.6503 | 0.8450 | 0.4144 |
| **seeds 1-5** | **0.6502+-0.0002** | **0.8451+-0.0001** | **0.4143+-0.0001** |
| seed0 (orig) | 0.6503 | 0.8450 | 0.4144 |

RMSE range across all 6 runs: [0.6500, 0.6506]

## 10-step (n_outer=10)

| run | RMSE | EV | ES |
|---|---|---|---|
| seed1          | 0.5641 | 0.8789 | 0.3577 |
| seed2          | 0.5639 | 0.8790 | 0.3575 |
| seed3          | 0.5637 | 0.8791 | 0.3573 |
| seed4          | 0.5650 | 0.8785 | 0.3582 |
| seed5          | 0.5642 | 0.8789 | 0.3578 |
| seed0 (orig)   | 0.5643 | 0.8788 | 0.3578 |
| **seeds 1-5** | **0.5642+-0.0005** | **0.8789+-0.0002** | **0.3577+-0.0003** |
| seed0 (orig) | 0.5643 | 0.8788 | 0.3578 |

RMSE range across all 6 runs: [0.5637, 0.5650]

## Summary

- 10-step vs 1-step RMSE ratio (mean): **0.8677** (13.2% reduction)
- 1-step RMSE (seeds 1-5): 0.6502 +- 0.0002
- 10-step RMSE (seeds 1-5): 0.5642 +- 0.0005
- Cross-seed std < 0.001 for both schemes -> the multi-tau advantage is NOT a seed artifact.

## Context (single-run anchors, not rerun)

| scheme | RMSE |
|---|---|
| L2b tau=0 (30x1) | 0.6290 |
| DirectUNet L4 (single) | 0.6189 |
| Strong-4DVar (DA) | 0.7420 |

The 10-step multi-tau result (0.564) beats every deterministic scheme including the best neural (DirectUNet L4, 0.619) and the best DA (Strong-4DVar, 0.742), reproduced across 6 independent seeds.
