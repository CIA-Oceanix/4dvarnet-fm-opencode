# L96 P1 — flow-matching operator score, pilot (2026-09-30)

**Status:** RESULTS — pilot on the rows that still have stored ensemble members (400-epoch
P1 protocol for the learned rows). Not a benchmark table: the current benchmark
rows (1200 epochs, benchmark-default training) have no stored members and need
in-memory re-sampling first.

**Numbers:** `reports/l96/outputs/fm_score/fm_score_p1_400ep.json` (scores per variant, τ, group) and `reports/l96/outputs/fm_score/scale/` (variance-rescaling sweep).

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

## Results (all 24 channels, normalised units²)

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

## Readings

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

## Caveats

- Learned rows are the 400-epoch P1 protocol; the 1200-epoch benchmark
  flows are better calibrated (pooled 0.84–0.94). Cross-class conclusions wait
  for re-sampled current rows.
- DA rows are Gaussian by construction (no stored members); the learned rows'
  `gauss` column is the like-for-like comparison.
- c* for ETKS at S1 hits the grid edge (8).

## Next

- Re-sample the current benchmark rows (CFM 1200 ep, SDA 1200 ep, hybrid) and
  score them in memory, as the pooled-calibration probe does.
- DA with member dumps (node-local /tmp) to score ETKF/ETKS beyond Gaussian.
- A per-time-since-observation breakdown and an integrated summary over τ.
