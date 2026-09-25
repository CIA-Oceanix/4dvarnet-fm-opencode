# Rank histograms on L96 (P1 paper C4) — results

**Status:** RESULTS (2026-09-25). This is the experiment flagged as `\needsrun` in P1 §5 ("the single experiment that would settle C4").

**C4, as the paper states it:** schemes that cannot represent `Psi_NG` (the ensemble-Kalman class) are *better* calibrated on spread/RMSE than the schemes carrying a full non-Gaussian branch. The paper's reconciliation is that spread/RMSE is a second-moment diagnostic and `Psi_NG` is not a second-moment object. So the missing branch should show up in the *shape* of the predictive distribution, which only a rank histogram can see.

**Verdict:** **not supported as a discriminating claim.**
- Most of the ensemble filters' apparent shape defect under the P1 protocol is a **mean bias**: model error in their forward model.
- Once spread and mean bias are both corrected, their remaining shape error is comparable to PredictStateCFM's and smaller than SDA's, with *opposite* tails.
- Only VanillaCFM is close to flat.
- Shape errors are scheme-specific; they are not determined by which operator slots a scheme occupies.

## Protocol

- **Data:** L96 S0, the 200 regular-grid test windows (`l96_datasets_obsj2_int100_nwin200.pt`), 24-D observed subspace, 30 members. Every scheme is scored against the same truth.
- **DA:** ETKF and EnKF with 30 members, re-run with the new opt-in member output (`evaluate_all_l96.py --save-members`; `store_ensemble` on the batched assimilators).
  - **P1 protocol:** inflation 2.0, no DA fast weights, the setting of `p1_l96_benchmark.md`.
  - **Benchmark default:** S0 inflation 1.5, with DA fast weights.
  - The re-runs reproduce the stored DA RMSEs to within 0.4% (ETKF 0.8681 vs 0.8662, EnKF 0.8932 vs 0.8943; benchmark 0.6876 vs 0.6851, 0.7114 vs 0.7112). The ensembles are unseeded.
- **Learned schemes:** the stored ensembles:
  - P1 VanillaCFM-M and PredictStateCFM-M, on the current 20-step early-fine sampler (`ens30_no20`) and on the old 10-step uniform one;
  - P1 SDA1-M and SDA2-M (`ens30_gw20`);
  - benchmark-default VanillaCFM-M and PredictStateCFM-M, 3 seeds each (`ens30_no20`).
- **Script:** `reports/l96/probe_rank_histograms.py`; job `batch/run_l96_rank_histograms.sbatch` (SLURM 55459).
  - The DA members are ~1.7 GB per method, so they were produced, used and deleted on the compute node's local `/tmp`.
  - Only the JSON/PNG outputs in `reports/l96/outputs/rank_histograms/` were kept.

**Diagnostics** (each pooled over windows × time × channels, per group):

| diagnostic | definition |
|---|---|
| `RI` | reliability index, `sum_k |h_k − 1/31|` (0 = flat) |
| `extreme` | mass in the two outer bins ÷ its flat value: > 1 means a U shape (tails too light / under-dispersed), < 1 a hump (tails too heavy / over-dispersed) |
| `RI_shape` | after rescaling members about the ensemble mean by one scalar per group, so that the second moment is calibrated |
| `RI_db` | after also removing each channel's mean error (ensemble mean − truth, averaged over windows and time) and recalibrating |

A pure mean bias is a ramp that survives the spread correction but not the de-biasing. Whatever survives both is higher-moment shape.

The spread and bias corrections use the truth. They are diagnostics, not usable post-processing.

## Results (all_obs; per group in the JSON)

| scheme | protocol | RI raw | extreme raw | RI_shape | RI_db | extreme_db |
|---|---|---|---|---|---|---|
| ETKF | P1 (inf 2.0) | 0.41 | 0.26 | 0.30 | **0.14** | 0.64 |
| EnKF | P1 (inf 2.0) | 0.36 | 0.36 | 0.29 | **0.10** | 0.83 |
| VanillaCFM-M | P1, 20 early-fine | 0.20 | 2.00 | 0.05 | **0.03** | 1.00 |
| VanillaCFM-M | P1, 10 uniform | 0.29 | 2.60 | 0.09 | 0.08 | 0.81 |
| PredictStateCFM-M | P1, 20 early-fine | 0.49 | 4.09 | 0.14 | **0.13** | 1.79 |
| PredictStateCFM-M | P1, 10 uniform | 0.66 | 5.42 | 0.14 | 0.12 | 1.81 |
| SDA1-M | P1 | 0.68 | 5.33 | 0.20 | **0.18** | 1.89 |
| SDA2-M | P1 | 0.48 | 3.89 | 0.27 | **0.27** | 2.47 |
| ETKF | benchmark (inf 1.5, DA fast weights) | 0.30 | 0.36 | 0.23 | **0.18** | 0.58 |
| EnKF | benchmark | 0.27 | 0.40 | 0.14 | **0.11** | 0.84 |
| VanillaCFM-M, seeds 1/2/3 | benchmark default | 0.09 / 0.08 / 0.02 | 1.34 / 1.30 / 1.07 | 0.12 / 0.10 / 0.02 | **0.11 / 0.09 / 0.04** | 1.51 / 1.40 / 1.09 |
| PredictStateCFM-M, seeds 1/2/3 | benchmark default | 0.18 / 0.21 / 0.18 | 1.94 / 2.14 / 1.98 | 0.12 / 0.13 / 0.12 | **0.12 / 0.13 / 0.12** | 1.63 / 1.74 / 1.66 |

**By group:**
- **P1 DA:**
  - Slow channels: RI_db = 0.04 (ETKF) and 0.02 (EnKF), i.e. flat once de-biased. Their raw defect was entirely a mean bias (channel bias RMS 0.17 / 0.20).
  - Observed fast channels: RI_db = 0.11 / 0.12, with a hump (extreme_db 0.69 / 0.89).
- **Benchmark DA:** the slow-channel bias is gone (RMS 0.005 / 0.009; DA fast weights remove the model error). The fast channels still carry a bias (RMS 0.13–0.15) and a hump after de-biasing.
- **Flows:** channel biases ≤ 0.06 everywhere. Their defects are spread and tail shape, not the mean.

## Reading

1. **The DA ramp is mostly first-moment.**
   - Under the P1 protocol the ensemble filters' histograms are humped and skewed, with the truth above the ensemble (rank bias +0.08).
   - Correcting spread leaves a ramp (RI_shape 0.29–0.30, the worst of any scheme). Removing each channel's mean bias halves or better that residual (0.10–0.14) and flattens the slow channels completely.
   - That bias is the forward-model error the P1 DA runs carry at S0: it runs without per-window fast weights. It shrinks when the fast weights are used (the benchmark DA, rank bias +0.03).
   - This is a model-error effect, not evidence about `Psi_NG`.
2. **After correcting spread and mean, no class is singled out.**
   - The ensemble filters' residual (0.10–0.18) sits in the same range as PredictStateCFM (0.12–0.13) and below SDA (0.18–0.27).
   - The *direction* differs by class:
     - DA ensembles are humped: tails too heavy relative to the truth (extreme_db 0.6–0.8), mainly in the fast channels;
     - PredictStateCFM and SDA are U-shaped: tails too light (extreme_db 1.6–2.5).
   - The operator-slot partition does not predict the ranking. SDA, which fills all three slots, has the worst shape.
3. **VanillaCFM's posterior shape is essentially right.**
   - P1, 20 steps: RI_db = 0.03 and extreme_db = 1.00. Its only defect was spread, and the early-fine sampler largely fixes that; the old 10-step uniform sampler also distorted shape (RI_db 0.08).
   - Benchmark default: 0.04–0.11 across seeds, and seed 3 is nearly flat even raw (RI 0.02).
4. **PredictStateCFM's light tails are structural.** RI_db ≈ 0.12–0.13 and extreme_db ≈ 1.6–1.8 are the same under the P1 and benchmark recipes, for every seed and both samplers. This is consistent with its late-τ Jacobian defect (NS1c ≈ 0.07 at τ = 0.9, `docs/results/l96_cfm_tau_consistency_l96b.md`).

## Consequences for the P1 paper (proposed; not edited here)

- **C4's refutation of the naive expectation stands:** the ensemble filters are the better *spread*-calibrated class.
- **Its reconciliation needs rewriting.** "`Psi_NG` shows up in shape instead" is not what the rank histograms show. The filters' apparent shape defect is mostly mean bias from forward-model error. After correcting mean and spread, shape errors of similar size appear in *every* class, with class-specific direction, and one flow (VanillaCFM) is essentially calibrated in shape.
- **A defensible statement:** posterior shape is not ordered by operator-slot occupancy. The one clear shape difference is *within* the flow family: VanillaCFM vs PredictStateCFM (0.03 vs 0.13 under P1). That points to the parameterisation, not the slot.
- **The `\needsrun` in §5 can be replaced by this table.** How to reword C4 is an authorial decision.

## Caveats

- **Coverage:** S0, regular grid only. DA is one run per protocol; the flows are one checkpoint (P1) or 3 seeds (benchmark).
- **Pooling:** the histograms pool strongly correlated coordinates (3000 steps × 24 channels × 200 windows). Differences of a few hundredths in RI between neighbouring rows are not meaningful; the table's contrasts are all ≥ 0.05.
- **Corrections:** the de-biasing uses a per-channel, window- and time-averaged bias. A bias that varies within a window would partly survive it and be read as shape.
- **Sampler dependence:** SDA uses its own guided 10-step sampler; the flows use their current benchmark sampler. Shape can depend on the sampler, as VanillaCFM's 10-uniform row shows.
