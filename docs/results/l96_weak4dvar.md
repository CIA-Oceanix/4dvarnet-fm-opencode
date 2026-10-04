# L96 — weak-constraint 4D-Var: tuning and benchmark rows (2026-09-30 / 10-04)

**Status:** RESULTS — model-error scale tuned per case on the validation windows; benchmark rows on the 200-window regular test set (S0, S1). No random-layout row yet.

**Numbers:** `reports/l96/outputs/l96_benchmark_extended.md` (main table, finding 10), `reports/l96/outputs/p1_l96_benchmark.md` (section 1); the validation sweeps are inline below (per-run JSONs from `scripts/weak4dvar_sweep_l96.py`).

## Scheme

`L96Weak4DVar` (`evaluation/baselines.py`) replaces the generic `Weak4DVar`, which diverges on L96 and was always skipped (`--skip-weak`). It has:
- **whitened controls**: `x₀ = x_b + b·σ·w` and a model-error increment `q·σ·u_t` at every step of the 500-step sub-window (`q_var_scale = q`);
- **loss** `½ Jo / r_var + ½‖w‖² + ½ Σ‖u_t‖²`, with `r_var = 0.5`;
- **optimiser**: LBFGS (lr 0.2), with gradient clipping and a NaN reset;
- **benchmark setup**: per-window parameters and DA fast weights, 6 cycled sub-windows per 3000-step window.

`mode = strong` drops `u` (hard constraint). `scripts/weak4dvar_sweep_l96.py` swaps it into the benchmark's Weak-4DVar slot of `run_and_cache_baselines`.

## Validation sweeps (8 regular validation windows, RMSE, slow / fast in brackets)

**LBFGS budget.** At the benchmark's 10 iterations, the strong mode is far from the benchmark Strong-4D-Var (0.721 vs 0.545 at S0). At 40 iterations it matches it:

| run | S0 | S1 |
|---|---|---|
| benchmark Strong-4D-Var | 0.545 (0.252 / 0.692) | 1.465 (1.117 / 1.638) |
| L96Weak4DVar strong, 10 it | 0.721 (0.381 / 0.890) | 1.484 (1.125 / 1.664) |
| L96Weak4DVar strong, 40 it | 0.545 (0.263 / 0.686) | 1.473 (1.124 / 1.648) |

At 10 iterations the weak sweep conflates the model-error term with an under-converged optimiser (S0 0.73–0.84). It was redone at 40 iterations.

**Model-error scale q, 40 iterations:**

| q | S0 | S1 |
|---|---|---|
| 0.003 | 0.620 (0.299 / 0.781) | 1.471 (1.122 / 1.645) |
| 0.01 | 0.557 (0.285 / 0.692) | 1.452 (1.108 / 1.625) |
| 0.03 | **0.553** (0.313 / 0.673) | 1.359 (1.026 / 1.526) |
| 0.1 | 0.663 (0.483 / 0.753) | 1.130 (0.769 / 1.311) |
| 0.3 | 0.791 (0.600 / 0.887) | **1.099** (0.746 / 1.275) |
| 1.0 | 0.781 (0.598 / 0.872) | 1.124 (0.752 / 1.310) |

No NaN resets in any run.

**Selection: one value per case**, as for the DA inflation: **q = 0.03 at S0, q = 0.3 at S1**.
- **Why not one shared value.** The min-max rule would pick q 0.1, costing 20% at S0 for 3% at S1.
- **S0.** The model-error term only costs accuracy at a perfect model; q 0.01–0.03 ties the strong constraint.
- **S1.** RMSE falls until q ≈ 0.1 and then plateaus over 0.1–1.0 (1.10–1.13, within the noise of 8 windows). The gain is mostly on the slow variables (1.12 → 0.75).
- **q 0.003 at S0** (0.620) is worse than the strong limit. This is likely under-convergence of the tighter controls at 40 iterations.

## Benchmark rows (200 regular test windows)

`batch/run_l96_weak4dvar_bench.sbatch` (SLURM 58092/58093): 20 chunks of 10 windows per case, LBFGS 40 iterations. The chunks are assembled and rescored with `scripts/assemble_weak4dvar_bench.py`: the windows cover 0–199 once, and the RMSE matches the chunks to 1e-4. There were no NaN resets.

| scheme | S0 | S1 | S1/S0 |
|---|---|---|---|
| Strong-4D-Var (benchmark) | 0.703 | 1.436 | 2.04 |
| **Weak-4D-Var (q 0.03 / 0.3)** | **0.695** ± 0.191 | **1.064** ± 0.182 | 1.53 |
| ETKS (30 members) | 0.497 | 1.338 | 2.69 |
| best learned (benchmark default) | 0.340 | 0.338 | 0.99 |

**Reading.**
- **S0.** Weak-4D-Var ties Strong-4D-Var (−1%), as it should at a perfect model.
- **S1.** It cuts the error by 26% and becomes the best DA row, ahead of the ETKS. Representing model error as a soft constraint converts part of the hard constraint's misspecification into restriction.
- **What remains.** It is still about 3× the learned schemes at S1: the per-step additive error with a scalar scale cannot absorb a parametric bias.

**Cost.** About 300 s per window per case on an L40S, 540 s on an RTX 8000, and about 1.6 h per 10-window chunk on an A40. Windows are solved sequentially (no batching).

## Open

- **Random layout.** `L96Weak4DVar._obs_cost` does not mask missing channels; a `torch.isfinite` mask (and a test) is needed before the canonical random-layout row.
- **Marginal value of observations** (15 → 30 obs) for Weak-4D-Var, next to the Strong-4D-Var collapse (extended report finding 7).
- **Batching** windows (with a per-window optimiser) would cut the cost about 10×.
