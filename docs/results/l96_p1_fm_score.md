# L96 P1 — flow-matching operator score FMS_τ (2026-09-30 / 10-01)

**Status:** RESULTS — current benchmark rows (1200-epoch learned schemes, 3 seeds; benchmark
ETKF/ETKS re-run with their ensembles), regular grid and random layouts, S0 and S1; plus the
earlier pilot on the 400-epoch P1 rows.

**Numbers:** `reports/l96/outputs/fm_score/fm_score_current.md` (tables, from the per-run JSONs in `reports/l96/outputs/fm_score/current/`), pilot `reports/l96/outputs/fm_score/fm_score_p1_400ep.json` and `reports/l96/outputs/fm_score/scale/`.

## Metric

Path x_τ = τ x₁ + (1−τ) x₀, x₀ ~ N(0, I), states z-scored per channel
(`experiments/l96_norm_stats_obsj2.pt`). For a candidate q with operator
Ψ_q(x_τ, τ) = E_q[x₁ | x_τ] and the truth x₁*,

    FMS_τ(q) = E ‖Ψ_q(x_τ) − x₁*‖² = E ‖Ψ_q − Ψ_p‖² + mmse_p(τ)

where p is the reference posterior given everything (common to all schemes).
Differences between schemes are therefore exact operator-divergence
differences; the constant mmse_p(τ) is unknown. Properties used below:

- τ = 0: squared error of the ensemble mean (MSE).
- Each τ ∈ (0, 1): strictly proper (Gaussian smoothing is injective).
- Gaussian candidate N(m, v), error e, snr = τ²/(1−τ)²:
  FMS = (e² + snr·v²)/(1 + snr·v)², minimised at v = E e² for every τ — the
  spread/skill identity.
- τ → 1: FMS = 1/snr + snr⁻²·(e²/v² − 2/v) + …, i.e. a universal floor plus
  the Hyvärinen score. τ = 0.95 is dominated by that floor and by the
  heavy-tailed e²/v² term; τ ∈ [0.25, 0.75] is the informative range.
- Integrated with weight dsnr, the score gives KL(p‖q) up to a constant
  (Verdú 2010, mismatched estimation), i.e. the log score; diverges for an
  atomic ensemble.
- A point estimate scores its MSE at every τ.

Variants: `gauss` (per-element N(mean, var), closed form, the only one
available for ETKF/ETKS, which store mean + per-time variance), `ens`
(empirical 30-member operator, univariate blocks), `kde` (Silverman-smoothed
members), `snap` (empirical operator on 24-channel snapshot blocks). Monte
Carlo over 4 x₀ draws with common random numbers across schemes; pooled over
200 windows × 3000 steps × 24 observed channels.

Code: `evaluation/fm_score.py` (+ `tests/test_fm_score.py`),
`reports/l96/probe_fm_score.py`, `reports/l96/probe_fm_score_scale.py`.
Outputs: `reports/l96/outputs/fm_score/`.

## Current benchmark rows

Members are re-sampled in memory with the benchmark protocols (flows 30 × 20 early-fine
steps; SDA gw 25, 10 steps; hybrid DU-M(1200, seeds 1–3) → SDA3-fix-M(1200, seed 1),
τ₀ 0.1, gw 2) and scored in the eval job (`reports/l96/probe_fm_score_eval.py`,
`batch/run_l96_fm_score_current.sbatch`, SLURM 57643/57649). ETKF/ETKS (λ S0 1.15 / S1 2.5,
ETKS `correct`, full window) are re-run with `--save-members` and their ensembles scored on
node-local storage (`reports/l96/probe_fm_score_da.py`, `batch/run_l96_fm_score_da.sbatch`,
SLURM 57648/57650): a fresh ensemble realisation, RMSE within sampling noise of the
benchmark files (regular S0 ETKF 0.615 vs 0.610, ETKS 0.493 vs 0.497; random 0.679 / 0.579 vs
0.679 / 0.572). Every learned row reproduces its benchmark RMSE.

Gaussian FMS (normalised units²), mean over seeds; full tables with seed ranges, spread/skill
and the non-Gaussian column in `fm_score_current.md`. Seed ranges are ≤ 0.006 on FMS.

| scheme | reg S0 τ=0 | 0.5 | 0.75 | reg S1 τ=0 | 0.5 | 0.75 | rnd S0 τ=0 | 0.5 | 0.75 | rnd S1 τ=0 | 0.5 | 0.75 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ETKF | 0.187 | 0.125 | 0.050 | 0.715 | 0.498 | 0.127 | 0.236 | 0.127 | 0.040 | 0.826 | 0.471 | 0.121 |
| ETKS | 0.129 | 0.107 | 0.059 | 0.631 | 0.548 | 0.255 | 0.194 | 0.122 | 0.047 | 0.761 | 0.530 | 0.186 |
| VanillaCFM-M | 0.057 | 0.042 | 0.0196 | 0.056 | 0.042 | 0.0196 | 0.121 | 0.071 | 0.025 | 0.122 | 0.072 | 0.026 |
| PredictStateCFM-M | 0.054 | 0.040 | **0.0195** | 0.053 | 0.040 | **0.0192** | 0.113 | 0.066 | **0.025** | 0.113 | 0.067 | **0.025** |
| SDA1-M | 0.091 | 0.071 | 0.038 | 0.091 | 0.070 | 0.038 | 0.159 | 0.106 | 0.047 | 0.169 | 0.112 | 0.049 |
| SDA2-M | 0.089 | 0.064 | 0.031 | 0.093 | 0.067 | 0.032 | 0.149 | 0.085 | 0.035 | 0.164 | 0.089 | 0.036 |
| SDA3-fix-M | 0.088 | 0.067 | 0.034 | 0.088 | 0.066 | 0.034 | 0.155 | 0.097 | 0.041 | 0.162 | 0.098 | 0.041 |
| Hybrid DU-M → SDA3-fix-M | **0.039** | **0.034** | 0.0200 | **0.038** | **0.033** | 0.0196 | **0.081** | **0.062** | 0.029 | **0.080** | **0.062** | 0.029 |

Calibration loss (share of FMS removed by the best scalar variance rescaling) at τ = 0.75:

| scheme | reg S0 | reg S1 | rnd S0 | rnd S1 |
|---|---|---|---|---|
| ETKF | 14% | 23% | 3% | 14% |
| ETKS | 34% | 63% (c* ≥ 8) | 21% | 47% (c* ≥ 8) |
| VanillaCFM-M / PredictStateCFM-M | 0% / 2% | 0% / 2% | 1% / 2% | 2% / 3% |
| SDA1-M / SDA2-M / SDA3-fix-M | 18 / 7 / 14% | 18 / 3 / 13% | 22 / 5 / 15% | 26 / 2 / 15% |
| Hybrid | 16% | 15% | 28% | 28% |

### Readings (current rows)

1. **The most accurate scheme is not the best distribution.** The hybrid has the best mean
   (τ ≤ 0.5 everywhere) but the flows overtake it at τ = 0.75 — on the regular grid by a
   hair (0.0195 vs 0.0200), on the random layouts clearly (0.025 vs 0.029). Its sampler is
   under-dispersed (spread/skill 0.47–0.58, calibration loss 15–28%); the flows are
   calibrated (0–3%). RMSE alone ranks the hybrid first; the FMS curve says it buys its
   mean with an over-confident ensemble — the blending composes Ψ_mean from DirectUNet and
   Ψ_anom from a tempered-likelihood sampler, and the score separates the two.
2. **Misspecification signature, current DA.** DA calibration loss grows from S0 to S1 on
   both layouts (ETKF 14 → 23% regular, 3 → 14% random; ETKS 34 → 63%, 21 → 47%); every
   learned row is S0/S1-invariant, except SDA2, whose loss *falls* (7 → 3%, 5 → 2%) — the
   accidental widening from conditioning on the biased θ (structure doc §3).
3. **ETKS vs ETKF.** ETKS has the better mean (τ = 0) everywhere and the worse score from
   τ = 0.75 at S0 and from τ = 0.5 at S1: the smoother's over-confidence costs more than its
   accuracy gain once the score weighs the anomaly.
4. **Non-Gaussian structure.** At 1200 epochs the flows' KDE beats the Gaussian at the same
   variance by 1.4–3.4% (more on the random layouts), SDA2 by 2–5%; SDA1/SDA3-fix and the
   hybrid ≈ 0 at τ = 0.5. The DA ensembles go the other way at τ = 0.75 (+3 to +9% at S0,
   +18 to +31% at S1): the truth falls outside the members more often than a Gaussian with
   the same variance allows — a short-tailed, over-confident ensemble, worst under model
   error. (The 400-epoch pilot showed no shape gain for the learned rows.)
5. **Training budget.** PredictStateCFM-M at 1200 vs 400 epochs: calibration loss at τ = 0.75
   from 11% to 2%, FMS(0.75) 0.0237 → 0.0195; SDA3-fix-M 17% → 14%.

## Pilot: 400-epoch P1 rows (2026-09-30)

### Results (all 24 channels, normalised units²)

Gaussian FMS; the RMSEs reproduce the benchmark (ETKF 0.610 / 1.409, ETKS
0.497 / 1.338, DirectUNet-M 0.470).

| scheme | S0 τ=0 | 0.25 | 0.5 | 0.75 | S1 τ=0 | 0.25 | 0.5 | 0.75 |
|---|---|---|---|---|---|---|---|---|
| DirectUNet-M (point) | 0.091 | 0.091 | 0.091 | 0.091 | 0.092 | 0.092 | 0.092 | 0.092 |
| ETKF (λ 1.15 / 2.5) | 0.179 | 0.167 | 0.119 | 0.046 | 0.714 | 0.682 | 0.497 | 0.127 |
| ETKS (λ 1.15 / 2.5) | 0.133 | 0.129 | 0.110 | 0.060 | 0.631 | 0.620 | 0.548 | 0.255 |
| VanillaCFM-M | 0.052 | 0.050 | 0.041 | 0.021 | 0.051 | 0.049 | 0.040 | 0.020 |
| PredictStateCFM-M | 0.055 | 0.053 | 0.044 | 0.024 | 0.054 | 0.052 | 0.044 | 0.023 |
| SDA1-M | 0.108 | 0.104 | 0.083 | 0.043 | 0.108 | 0.104 | 0.084 | 0.043 |
| SDA3-fix-M | 0.105 | 0.100 | 0.078 | 0.039 | 0.108 | 0.103 | 0.080 | 0.039 |

**Calibration loss** = share of FMS(c = 1) removed by the best scalar variance
rescaling c·v (log grid 1/8–8), and the optimal c*:

| scheme | pooled spread/skill S0 / S1 | S0 τ=0.5 | S0 τ=0.75 | S1 τ=0.5 | S1 τ=0.75 |
|---|---|---|---|---|---|
| ETKF | 0.78 / 0.59 | 3% (c* 1.7) | 12% (2.6) | 16% (2.8) | 23% (3.5) |
| ETKS | 0.47 / 0.34 | 13% (4.3) | 36% (6.5) | 29% (≥8) | 63% (≥8) |
| VanillaCFM-M | 0.78 / 0.79 | 2% (1.7) | 4% (1.7) | 2% (1.5) | 3% (1.5) |
| PredictStateCFM-M | 0.65 / 0.66 | 5% (2.3) | 11% (2.6) | 5% (2.3) | 10% (2.3) |
| SDA1-M | 0.57 / 0.57 | 9% (2.8) | 21% (4.8) | 9% (2.8) | 22% (4.8) |
| SDA3-fix-M | 0.63 / 0.65 | 6% (2.3) | 17% (3.9) | 6% (2.3) | 16% (3.5) |

### Readings (pilot)

1. **The τ-curve separates accuracy from calibration.** ETKS beats ETKF at
   τ ≤ 0.5 (better mean) and loses at τ = 0.75 at S0, already at τ = 0.5 at
   S1: its smoothed ensemble is over-confident. The point estimate is beaten by
   ETKF's ensemble from τ = 0.75 at S0 despite a mean twice as bad.
2. **Misspecification signature, from a proper score.** The DA calibration
   loss grows from S0 to S1 (ETKF 12 → 23%, ETKS 36 → 63% at τ = 0.75);
   the learned rows' loss is S0/S1-invariant (VanillaCFM 3–4%, SDA3-fix
   16–17%). This is K6 of the P1 structure doc restated with a proper score.
3. **c* rises with τ.** Larger τ weights elements where the spread is small
   relative to the error (the e²/v² term), so c*(τ) > 1/ss² flags errors
   concentrated where the scheme is confident (ETKF S0: c* 1.2 → 2.6 from τ =
   0.25 to 0.75, against 1/ss² = 1.6). The curve reads global calibration at
   small τ and local calibration at large τ.
4. **No univariate non-Gaussian gain.** The KDE beats the plain Gaussian, but
   the Gaussian with the KDE's inflated variance (+29%) recovers all of it: at
   τ = 0.5 KDE vs inflated Gaussian is −0.3 to −0.8%, at τ = 0.75 +0.5 to
   +1.5%. Consistent with the rank-histogram result (no class-specific shape
   deficit).
5. **Snapshot blocks** (24 channels jointly) keep a usable weight ESS at
   τ ≤ 0.5 (27–28 and 18–21 of 30) and degenerate by τ = 0.75 (5–7). Their
   scores are not comparable to univariate ones (different mmse_p constant);
   rankings within the block type agree with the univariate ones.
6. The atomic `ens` operator tracks `gauss` up to τ = 0.5 and is penalised
   above (nearest-member snapping); use `gauss` / `kde` for τ ≥ 0.75.

### Caveats (pilot)

- Learned rows are the 400-epoch P1 protocol; the 1200-epoch benchmark
  flows are better calibrated (pooled 0.84–0.94). Cross-class conclusions wait
  for re-sampled current rows.
- DA rows are Gaussian by construction (no stored members); the learned rows'
  `gauss` column is the like-for-like comparison.
- c* for ETKS at S1 hits the grid edge (8).

## Next

- Integrated summary over τ (uniform dτ, and dsnr-weighted with truncation ≈ KDE log score).
- Breakdown by time since the last observation and by slow / fast group (both already in
  the JSONs per group; time bins need a probe change).
- Fold an FMS(τ = 0.5 / 0.75) column into the P1 report and the structure doc (§4.5 metrics,
  §5.5 / §6.3 calibration).
