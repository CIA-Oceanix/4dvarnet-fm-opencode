# L96 ETKS at the benchmark inflation (ETKF 1.15 / 2.5)

**Status:** RESULTS (2026-09-29). Adds the ETKS row at the benchmark ETKF inflation retuned in #295 (S0 1.15 / S1 2.5, `docs/results/l96_da_inflation_post291.md`). `docs/results/l96_etks_benchmark.md` measured the ETKS only at λ 1.5 / 2.0 and 1.1 / 2.5. Same protocol: `correct` retro-inflation, full window, the 200 P1 test windows, the ETKF rerun in the same job for paired comparisons.

**Numbers:** `reports/l96/outputs/l96_benchmark_extended.md` (the "ETKS" DA row, directly under the benchmark ETKF). Runs: `batch/run_l96_etks_test.sbatch` tasks 2–3 with `ETKF_INF=ETKS_INF=s0=1.15,s1=2.5` (SLURM 56736). Paired statistics from the same bundle files.

## Results

Benchmark metric (per-window mean over the 24 observed channels of the per-channel RMSE), paired against the ETKF of the same run, with 95% intervals:

| | ETKF | ETKS | ΔRMSE | windows ETKS better | ΔCRPS | spread/RMSE ETKF → ETKS |
|---|---|---|---|---|---|---|
| regular S0 | 0.612 | **0.497** | −18.7% ± 2.3 | 186/200 | −11.9% ± 3.1 | 0.61 → 0.41 |
| regular S1 | 1.410 | **1.338** | −5.1% ± 0.2 | 200/200 | +1.6% ± 0.4 | 0.61 → 0.34 |
| random S0 | 0.677 | **0.572** | −15.6% ± 1.5 | 186/200 | −12.1% ± 1.8 | 0.72 → 0.54 |
| random S1 | 1.407 | **1.348** | −4.2% ± 0.7 | 172/200 | −5.2% ± 1.0 | 0.93 → 0.53 |

The ETKF of this run reproduces the benchmark ETKF rows of #298 (0.610 / 1.409 / 0.679 / 1.407) to within 0.3%.

## Findings

1. **At the benchmark inflation the smoother gains 16–19% RMSE at S0 and 4–5% at S1**, in line with the other two inflation settings. The F3 (filter vs smoother) term does not depend on the inflation within 1.1–1.5 at S0.
2. **This is the canonical ETKS row: regular S0 0.497, random S0 0.572.**
   - Regular S0 is better than the ETKS at 1.1 / 2.5 (0.514), because the regular grid prefers λ slightly above 1.1 (validation sweep). Random S0 is slightly worse (0.566). S1 is identical, since both use 2.5.
   - Against the learned schemes (≈0.34 regular, ≈0.44 random S0), DA's S0 deficit is 1.46× regular and 1.3× random.
3. **Smoothing costs spread.** The ETKS is under-dispersed everywhere (spread/RMSE 0.34–0.54). At regular S1 its CRPS is 1.6% worse than the filter's, despite 5% lower RMSE, the same pattern as at λ 1.1 / 2.5.
