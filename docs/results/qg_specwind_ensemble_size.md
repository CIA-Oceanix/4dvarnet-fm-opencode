# Ensemble size for the QG gyrostat DA: is N = 80 enough for the ETKF and the EnKS?

**Status:** RESULTS (2026-10-01). Asked when checking whether the localized
EnKS (`docs/results/qg_specwind_enks.md`) is a good smoother baseline. With
80 members for an 8,192-dimensional state, localization and sampling noise
could limit both the filter and the smoother.

**Numbers:** `evaluation/qg_specwind_enks_tuning.py` (`--summarize`, tasks
13–32) over `experiments/qg_specwind_enks_tuning/`. SLURM 57613 and 57698,
rtx8000, 20 val windows of the forced dataset. Test rows:
`reports/qg/outputs/qg_specwind_da_report.md` (large-ensemble block), from
`experiments/qg_specwind_da/` (`ENS=320`, SLURM 57701–57704 and
57913–57916).

**Settings:** the benchmark ones. ETKF: EnSRF, radius 8, ridge 0.1. EnKS:
time taper 8 d, no lag. Both: cross-layer weight 1, bred init. R × 1 in S0;
R × 6 in S1-tuned (realistic base).

## Step 1: convergence at the benchmark settings (val)

| N | S0 ETKF | S0 EnKS | S1-tuned ETKF | S1-tuned EnKS |
|---|---|---|---|---|
| 40 | 0.734 | 0.770 | 0.632 | 0.667 |
| **80 (benchmark)** | 0.764 | 0.806 | 0.641 | 0.679 |
| 160 | 0.775 | 0.820 | 0.647 | 0.685 |
| 320 | 0.781 | **0.828** | 0.649 | **0.688** |

| paired (bootstrap 95%) | S0 | S1-tuned |
|---|---|---|
| EnKS, N 320 − 80 | +0.022 [+0.019, +0.025] | +0.009 [+0.006, +0.012] |
| EnKS, N 320 − 160 | +0.008 [+0.006, +0.010] | +0.003 [+0.001, +0.004] |
| ETKF, N 320 − 80 | +0.017 [+0.014, +0.021] | +0.008 [+0.005, +0.011] |
| EnKS − ETKF at N = 40 / 80 / 160 / 320 | +0.036 / +0.042 / +0.045 / +0.046 | +0.035 / +0.039 / +0.039 / +0.040 |

Spread/RMSE q₁ of the EnKS (median), N = 40 / 80 / 160 / 320: S0 0.56 / 0.67
/ 0.74 / 0.77; S1-tuned 0.81 / 0.87 / 0.89 / 0.90.

## Step 2: re-tuning at N = 320, S0 only (val)

| N = 320, S0 | score | vs radius 8 (taper 8) |
|---|---|---|
| EnKS radius 8, taper 8 | 0.828 | — |
| EnKS radius 8, taper 16 | 0.831 | +0.004 [+0.003, +0.005] |
| EnKS radius 12, taper 8 | 0.832 | +0.004 [+0.001, +0.007] |
| **EnKS radius 12, taper 16** | **0.835** | +0.007 [+0.004, +0.010] |
| EnKS radius 16, taper 16 | 0.825 | −0.003 [−0.008, +0.003] |
| ETKF radius 8 | 0.781 | — |
| **ETKF radius 12** | **0.785** | +0.004 [+0.001, +0.007] |
| ETKF radius 16 | 0.777 | −0.004 [−0.010, +0.001] |

**Selected for the S0 large-ensemble reference:** ETKF radius 12; EnKS radius
12 with taper 16 d. A larger ensemble supports weaker localization, with an
optimum inside the grid at radius 12. Taper 16 is the top of the taper grid,
but its step over 8 is small (+0.003 at radius 12).

S1 is not re-tuned: it saturates by N = 160, so it keeps radius 8 and taper
8 d at N = 320.

## Step 3: test (100 windows)

**S1-tuned, N = 320** (benchmark settings; forced / coupled):

| | N = 80 | N = 320 | N 320 − 80 |
|---|---|---|---|
| ETKF | 0.633 / 0.642 | 0.641 / 0.650 | +0.008 [+0.007, +0.009] / +0.008 [+0.007, +0.010] |
| EnKS | 0.665 / 0.673 | **0.675 / 0.683** | +0.010 [+0.009, +0.012] / +0.011 [+0.009, +0.012] |

**S0, N = 320** (re-tuned settings; forced / coupled):

| | N = 80 | N = 320 | N 320 − 80 | CRPS q (×10⁻⁶), N = 80 → 320 |
|---|---|---|---|---|
| ETKF (radius 12) | 0.774 / 0.779 | 0.794 / 0.800 | +0.020 [+0.018, +0.023] / +0.021 [+0.019, +0.024] | 5.88 → 5.53 / 5.74 → 5.37 |
| EnKS (radius 12, taper 16 d) | 0.812 / 0.819 | **0.841 / 0.847** | +0.029 [+0.026, +0.032] / +0.028 [+0.025, +0.031] | 5.32 → 4.78 / 5.16 → 4.62 |

- **Test matches val:** S0 +0.020 / +0.029 against the predicted +0.021 /
  +0.029; S1 +0.008 / +0.010.
- **The smoother's margin over the filter grows slightly with N** in S0
  (+0.038 / +0.040 at N = 80 → +0.047 / +0.047 at N = 320).
- **CRPS improves by about 10% in S0** and about 1.5% in S1.

## Findings

1. **The rankings hold at every ensemble size.** The EnKS beats the ETKF
   by about +0.04 at all N (S0 +0.036 → +0.046, S1 +0.035 → +0.040). The
   benchmark's comparative conclusions do not depend on N.
2. **S0 is not converged at N = 80.** Absolute skill keeps rising with each
   doubling, and spread calibration improves. In the perfect-model case,
   sampling error is a real part of the remaining error. At N = 80 the S0
   EnKS sits about 0.02 below its N = 320 value, and about 0.03 below it once
   re-tuned.
3. **Under model error, N barely matters past 160** (+0.003 from 160 to
   320; +0.01 from 80 to 320 on test). Model error, not sampling error,
   limits skill in S1.
4. **A larger ensemble wants weaker localization.** In S0 at N = 320 the best
   radius moves from 8 to 12 cells and the taper widens, which is
   consistent with less sampling noise in long-range covariances.

## Recommendation

- **Keep N = 80 as the benchmark.** It is operationally realistic,
  comparable to the legacy QG setup, and every comparison holds there.
- **Quote the large-ensemble (N = 320) rows as converged-skill references**,
  in particular when comparing with learned methods.

## Caveats

- 20 val windows. The N = 320 S0 runs take about 3 h each on an rtx8000
  (about 9 min per window).
- Step 2 was S0 only; the taper grid stopped at 16 d.
- One task of the re-tuning (EnKS radius 12, taper 16) ended with a bus
  error **after** writing its complete result (20 windows, scores printed).
  The result is valid.
