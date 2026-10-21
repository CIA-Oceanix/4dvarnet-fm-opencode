# Neural schemes on the QG gyrostat benchmark: end-to-end, SDA and SDA hybrids across observation configurations

**Status:** RESULTS (2026-10-21).
- **Question 1:** is there an observing configuration where SDA sampled from pure noise
  (the published sampler) falls behind the end-to-end schemes?
- **Question 2:** does the SDA hybrid (SDA started from an end-to-end estimate) stay ahead
  everywhere?

**Numbers:** test rows in `reports/qg/outputs/qg_specwind_da_report.md` §9–10.
- Rows: `experiments/qg_specwind_da/qg_specwind_gyrostat_v1/test/neural_*`.
- Val sweeps: `.../val/neural_*`.
- Design: `docs/plans/analysis/qg_specwind_neural.md`.

## Setting

**Schemes:** all are 3D U-Nets over (time, y, x), trained on 15-day crops.
- **G1 DirectUNet:** M tier, 16.8 M parameters, and L tier.
- **G2 CFM:** an observation-conditioned flow, matched to G1-M-3D.
- **G3 SDA prior:** an unconditional score prior.
  - The run reported is 10.0 M parameters.
  - A tier-matched 16.8 M retrain gives the same scores, within 0.002 for pure-noise SDA and
    for the hybrids.

**Training observing system** (the end-to-end nets only; the prior never sees
observations):
- 3–30 random (step, column) slots per day, drawn with a K^-1.5 tilt towards sparse.
- Observation error drawn between the S0 model and the S1 one.
- So 6 and 12 columns per day and 15% noise are in distribution, and 2% noise is below the
  training error range.

**Evaluation:**
- 100 forced test windows of 30 days, with the same observation draws as the DA rows.
- Daily-mean estimates are cubic-interpolated to 2 h.
- Score = mean EV over ψ₁, ψ₂, q₁ and q₂.
- Comparisons use paired bootstrap intervals over windows.

**Pure-noise SDA:** the published sampler (Rozet & Louppe).
- γ 0.01, 100 steps, 1 corrector, 30 members.
- Observation σ = the noise level.

**Hybrid:** the same prior's SDA sampler, started by SDEdit at τ₀ from an end-to-end estimate.
- 10 steps.
- The guidance is a normalized step of fixed length (the "weight").
- Warm starts:
  - G1-M-3D or G1-L-3D: their full-window estimate.
  - G2-3D: one CFM sample per member (`--cfm-warm-start member`, PR #355).

## Findings

### 1. Pure-noise SDA falls behind the end-to-end schemes off the reference case

Paired score difference against the best end-to-end row (G1-L-3D, stitched or full window,
whichever is better):

| config | pure-noise SDA − best end-to-end [95% CI] |
|---|---|
| S0, 3 col/day, 5% (reference) | −0.009 [−0.013, −0.006] |
| S1 (realistic base) | **−0.165** [−0.212, −0.124] |
| 6 col/day | −0.009 [−0.013, −0.006] |
| 12 col/day | −0.020 [−0.030, −0.013] |
| 15% noise | −0.030 [−0.036, −0.024] |
| 2% noise | −0.001 [−0.005, +0.003] |

**S1 collapse:**
- The likelihood assumes white noise at σ = 5%.
- The S1 altimetry error is 15% white plus 15% correlated per pass.
- 15 windows score below 0.5, and q₂ EV drops to 0.35.

**Against the M tier:** pure-noise SDA (0.881) ties G2-3D (0.881) at the reference density and
beats G1-M-3D (0.870–0.874).
- Its −0.009 at the reference case therefore comes from the L-tier end-to-end net.
- The other gaps hold against the M tier too: S1, 12 columns and 15% noise.

**Calibration:** pure-noise SDA is the best calibrated scheme (spread/RMSE 0.8–0.9).

### 2. The hybrid leads the benchmark, with one setting

The G1-L-3D warm start with τ₀ 0.5 and weight 2 beats the best end-to-end row in 5 of 6
configurations:

| config | best end-to-end | hybrid | hybrid − best end-to-end [95% CI] |
|---|---|---|---|
| S0, 3 col/day | 0.890 (full window) | **0.901** | +0.010 [+0.009, +0.011] |
| S1 | 0.849 (full window) | **0.857** | +0.008 [+0.007, +0.009] |
| 6 col/day | 0.925 (stitched) | **0.929** | +0.004 [+0.002, +0.005] |
| 12 col/day | **0.943** (stitched) | 0.940 | −0.002 [−0.003, −0.001] |
| 15% noise | 0.862 (full window) | **0.872** | +0.010 [+0.009, +0.011] |
| 2% noise | 0.896 (full window) | **0.905** | +0.010 [+0.009, +0.011] |

**c12, the one loss:**
- The warm start is G1-L-3D's full-window estimate, which is its weaker variant at dense
  observations (0.939 against 0.943 stitched).
- τ₀ 0.7 at 12 columns (val-selected there) ties instead: test 0.9431, +0.0005
  [−0.0006, +0.0015].
- A stitched warm start at dense observations is untested.

**Against DA (section 9 of the report):**
- **S0:** the hybrid is +0.060 over the best DA reference (EnKS, N = 320).
- **S1:** the hybrid is +0.18 over it.
- See the report's caveat: the networks are trained on the true model, so in S1 they carry
  no model error.

### 3. The warm start sets the hybrid's level

Hybrid score on test by warm start (old prior, τ₀ 0.5, weight 5 unless stated):

| warm start | S0 3 col | S1 (weight 2) | 6 col | 12 col | 15% | 2% |
|---|---|---|---|---|---|---|
| G1-M-3D, full window | 0.891 | 0.848 | 0.923 | 0.938 | 0.850 | 0.899 |
| G2-3D, per-member CFM samples | 0.888 | 0.844 | 0.925 | 0.940 | 0.852 | 0.894 |
| G1-L-3D, full window | 0.898 | 0.857 | 0.926 | 0.939 | 0.856 | 0.907 |

**What this shows:**
- At the reference case the hybrid adds +0.007 to +0.017 to its warm start's own score, and the
  stronger end-to-end net gives the stronger hybrid.
- With an M-tier warm start, the hybrid only ties G1-L-3D (3 and 6 columns, S1, 2% noise) or
  trails it (12 columns, 15% noise).
- **Mean start:** starting all G2 members from the CFM ensemble mean scores the same
  (0.8878 against 0.8876) but loses spread. Per-member starts are the default.

### 4. Hybrid tuning (val, 100 windows)

**Guidance weight at 15% noise.** Δ against G2-3D:

| warm start | weight 1 | weight 2 | weight 5 |
|---|---|---|---|
| G2-3D | +0.002 | +0.005 | −0.008 |
| G1-M-3D | +0.007 | +0.008 | −0.010 |

**τ₀ × weight at 3 columns per day** (G1-L-3D warm start). Δ against G1-L-3D stitched (0.885
val):

| τ₀ | weight 2 | weight 5 |
|---|---|---|
| 0.5 | **+0.020** (spread 0.57 / 0.52) | +0.016 |
| 0.7 | +0.020 (spread 0.43 / 0.42) | +0.016 |
| 0.8 | +0.019 (spread 0.35 / 0.37) | +0.015 |

Weight 2 beats weight 5 by +0.004 [+0.0025, +0.0054]. τ₀ 0.5 matches 0.7 on score and keeps
the most spread.

**τ₀ × weight at 12 columns per day.** Δ against G1-L-3D stitched (0.945 val):

| τ₀ | weight 2 | weight 5 |
|---|---|---|
| 0.5 | −0.0005 | −0.003 |
| 0.7 | +0.0018 | −0.0016 |
| 0.8 | +0.0021 | −0.0003 |

**Decision:** τ₀ 0.5 with weight 2 is used everywhere. On test, moving from weight 5 to weight
2 gains:

| config | gain |
|---|---|
| 3 col/day | +0.0025 |
| 6 col/day | +0.0023 (from a tie with G1-L-3D to a win) |
| 15% noise | +0.017 |
| 2% noise | −0.001 |
| 12 col/day | +0.0008 |

### 5. End-to-end side

**Tier:** the best G1-L-3D variant beats the best G1-M-3D variant in every configuration, by
+0.010 to +0.017.

**Full window against stitched:**
- Full window helps when observations are sparse or noisy: 3 columns, S1, 15% and 2% noise.
- It does not help at 6 and 12 columns.

**G2-3D:**
- Stitched, it matches the G1-L-3D full window at 12 columns (0.939) and has the best CRPS in
  three configurations (S1, 12 columns, 15% noise).
- On the full window it fails: −0.038 at 3 columns (0.844 against 0.881), and over-dispersed.

### 6. Calibration remains the hybrid's weak point

**Hybrid:**
- The G1-L-3D hybrid's spread/RMSE is 0.47–0.74 for q₁ and q₂.
- The deterministic warm start pins the members together.
- It still has the best CRPS at 3 and 6 columns.

**Elsewhere on CRPS:**
- G2-3D or its hybrid lead at S1, 12 columns and 15% noise.
- Pure-noise SDA leads at 2% noise.

## Caveats and open items

**Caveats:**
- One training seed per network.
- **Tier mismatch:** the G1-L-3D hybrid uses an L-tier warm start, while G2 and G3 exist at
  M tier only.
- **DA rows:** they exist only at the reference density and S1 (plus the ETKF at 6 columns).
  The stress configurations have no DA counterpart yet.

**Open items:**
- **Stitched warm start at dense observations:** for the 12-column gap.
- **L-tier G2 and G3:** about 2 days of training each.
- **Spread of the hybrid:** perturbed warm starts, or a τ₀/weight trade-off on CRPS.
- **DA rows at 12 columns and at 15% / 2% noise:** EnKS and 4D-Var.
