# Relaxation inflation (RTPS / RTPP) for the QG gyrostat ETKF and EnKF: a negative result

**Status:** RESULTS (2026-10-04). Under the realistic S1, multiplicative inflation
diverges, and the S1-tuned filters absorb model error with R × 6 instead
(`docs/results/qg_specwind_s1_tuning.md`). This note tests the standard
alternative: relaxation inflation, applied once per analysis.

**Numbers:** `evaluation/qg_specwind_relax_tuning.py` (`--summarize`) over
`experiments/qg_specwind_relax_tuning/`. SLURM 58432, 44 tasks, rtx8000, 20 val
windows of the forced dataset.

**Settings:**
- ETKF: EnSRF, radius 8, ridge 0.1.
- EnKF: radius 6.
- Both: N = 80, cross-layer weight 1, bred init.
- S1 is the realistic base.

The sweep's no-relaxation baselines reproduce the S1-tuning sweep to within
0.001.

**Method** (`_relax_anomalies` in `evaluation/baselines.py`;
Whitaker & Hamill 2012): after each analysis, and before any multiplicative
inflation:
- **RTPP** blends the analysis anomalies with the forecast anomalies:
  A ← (1 − α) Aₐ + α A_f.
- **RTPS** rescales each variable's analysis spread toward its forecast
  spread: σ ← σₐ + α (σ_f − σₐ).

Both are available for the ETKF and EnKF, via `relax=` / `relax_alpha=`, the
`--relax` / `--relax-alpha` options and the `RELAX` / `ALPHA` sbatch
variables. The EnKS and ETKS refuse it: their retrospective update assumes
the unrelaxed analysis.

## Results (val, 20 windows)

**S1, as a replacement for R × 6 (at R × 1).** Best α per row; differences are
paired, bootstrap 95%:

| | score | spread/RMSE q₁ | q₂ | vs R × 1 | vs R × 6 |
|---|---|---|---|---|---|
| ETKF, R × 1 | 0.580 | 0.68 | 0.229 | — | −0.061 [−0.068, −0.055] |
| ETKF + RTPS 0.9 | 0.582 | 1.10 | 0.248 | +0.003 [−0.007, +0.013] | −0.058 |
| ETKF + RTPP 0.9 | 0.609 | 1.29 | 0.343 | +0.030 [+0.019, +0.039] | −0.032 [−0.041, −0.023] |
| **ETKF, R × 6** | **0.641** | 0.90 | 0.374 | | — |
| EnKF, R × 1 | 0.591 | 0.68 | 0.274 | — | −0.046 [−0.052, −0.040] |
| EnKF + RTPS 0.9 | 0.606 | 1.09 | 0.319 | +0.015 [+0.008, +0.023] | −0.031 [−0.037, −0.024] |
| EnKF + RTPP 0.25 | 0.579 | 0.73 | 0.267 | −0.012 | −0.057 |
| **EnKF, R × 6** | **0.636** | 0.88 | 0.379 | | — |

**Other RTPP values for the EnKF are worse:** 0.549, 0.511 and 0.554 at
α = 0.5, 0.75 and 0.9.

**S1, on top of a larger R** (RTPS):

| | α 0.25 | α 0.5 | α 0.75 | vs no relaxation (best α) |
|---|---|---|---|---|
| ETKF, R × 3 (0.627) | 0.628 | 0.627 | 0.625 | +0.001 [−0.000, +0.002] |
| ETKF, R × 6 (0.641) | 0.642 | 0.642 | 0.640 | +0.001 [+0.000, +0.003] |
| EnKF, R × 3 (0.626) | 0.629 | 0.631 | 0.630 | +0.005 [+0.001, +0.008] |
| EnKF, R × 6 (0.636) | 0.639 | **0.640** | 0.636 | +0.004 [+0.001, +0.007] |

**S0** (R × 1):

| | score | vs no relaxation | spread/RMSE q₁ |
|---|---|---|---|
| ETKF | 0.764 | — | 0.74 |
| ETKF + RTPS 0.25 / 0.5 | 0.767 / 0.768 | +0.003 / +0.004 [+0.002, +0.007] | 0.83 / 0.94 |
| ETKF + RTPP 0.25 / 0.5 | 0.767 / 0.764 | +0.003 / −0.000 | 0.85 / 0.97 |
| EnKF | 0.753 | — | 0.70 |
| EnKF + RTPS 0.25 / 0.5 | 0.757 / 0.758 | +0.004 / +0.005 [+0.003, +0.008] | 0.79 / 0.91 |
| EnKF + RTPP 0.25 / 0.5 | 0.751 / 0.736 | −0.002 / −0.017 [−0.027, −0.008] | 0.77 / 0.89 |

## Findings

1. **Relaxation is stable where multiplicative inflation is not.** Spread
   rises smoothly to spread/RMSE ≥ 1, with no divergence.
2. **It does not replace R × 6 in S1.** At a calibrated or over-dispersed
   spread, the best relaxation still trails R × 6:
   - by 0.03 for the ETKF (RTPP);
   - by 0.03 for the EnKF (RTPS).

   **So the S1 deficit is not a spread problem.** R × 6 damps every update.
   The filter then stops fitting observations to a wrong model and spreading
   those errors into the unobserved deep layer: q₂ is 0.37 with R × 6,
   against 0.25 with RTPS for the ETKF. Relaxation leaves the gain of the
   analysis mean unchanged at that analysis, so it can't do this.
3. **RTPP helps the deterministic EnSRF and hurts the stochastic EnKF.**
   RTPP pulls the anomalies back toward the forecast ones. In the EnKF this
   reintroduces forecast anomalies that the perturbed-observation update
   has already partly replaced with observation noise, and the ψ fields
   degrade (ψ₂ 0.65–0.75).
4. **On top of R × 6, and in S0, gains are at most +0.005.** They are
   statistically significant on val but negligible in practice. RTPS 0.5
   mostly calibrates the spread: in S0, spread/RMSE goes from 0.72 to 0.93.

## Decision

- **No change to the benchmark.** S0 stays at R × 1 and S1-tuned at R × 6,
  both without relaxation. No test runs are needed.
- **The relaxation option stays in the code.** If a calibrated S0 spread is
  needed (for example, for CRPS comparisons), RTPS 0.5 gives one at no skill
  cost.

## Caveats

- 20 val windows, the forced dataset only, N = 80.
- The S1 grid at R × 1 stops at α = 0.9; α → 1 for RTPS is the full forecast
  spread.
- Additive inflation and adaptive schemes (for example, Anderson's or ESTKF
  adaptive inflation) were not tried.
