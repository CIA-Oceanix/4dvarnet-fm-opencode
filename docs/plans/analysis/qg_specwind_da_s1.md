# ETKF/EnKF DA on the spectral-wind QG datasets, S1 (model error) — design

**Status:** DRAFT v2 (2026-09-27). v2 adds the realism-anchored scenario
(§2b), which replaced the legacy-analogue κ scan as the S1. **Selected:
realistic base** (`docs/results/qg_specwind_da_s1_calibration.md`), whose
ETKF ψ₁ of about 0.85–0.9 matches the user's revised aim. Test runs done
(`docs/results/qg_specwind_da_s0_s1_test.md`); the ETKF rows and the base
attribution were re-run with the exact localized EnSRF update on 2026-09-28. Follows the S0 design
(`docs/plans/analysis/qg_specwind_da_s0.md`) and the val-tuned S0 settings
(`docs/results/qg_specwind_da2_val_tuning.md`). 4D-Var is deliberately out of
scope for now; ETKF and EnKF only.

## 1. Question

How much of the S0 skill survives when the DA model is wrong in the ways a
real ocean reanalysis is wrong — an imperfect atmospheric forcing and biased
ocean parameters — and which error source costs the most?

## 2. Error components

The spectral analogues of the legacy QG S1 (`data/qg.py`), implemented in
`evaluation/qg_specwind_s1.py`. Only the DA side is corrupted; the truth,
observations and metrics are the S0 ones.

| component | what the DA model gets | reference level (κ = 1) | legacy QG analogue |
|---|---|---|---|
| `amp` — wind amplitude bias | every mode amplitude × (1 + b) | b = 0.15 | storm amplitude +15% |
| `noise` — random wind error | + OU noise per mode, std f × that mode's climatological RMS, τ = 10 d | f = 0.3 | OU amplitude noise, 0.3 × std, τ = 10 d |
| `shift` — wind position error | pattern translated by an OU displacement per axis, std s × L, τ = 10 d | s = 0.05 (50 km) | OU storm location error |
| `param` — ocean parameter bias | rd and bottom drag rek × (1 − p) | p = 0.15 | rd, rek −15% |

- **Mode RMS** (`MODE_RMS`): the per-mode amplitude RMS over windy windows of
  the **train** split (no val/test information), so the noise keeps the
  wind's spectrum and calm windows also receive a wrong wind, as a
  reanalysis forcing would.
- **Continuity:** the wind is corrupted as one series over the lead buffer
  and the window, so the bred initial ensemble is integrated with the DA
  model's wrong wind and parameters.
- **Common random numbers:** each component has its own generator seeded by
  the window's dataset index, so switching a component on or off never
  changes another's draw. Attribution differences are then pure effects,
  not sampling noise.

## 2b. Realism-anchored scenario (v2)

At κ = 2 the legacy mix reaches the target only by inflating rd to −30%
(about 3× real uncertainty), and it has no observation or structural error.
The attribution then mostly measures that rd error. The realistic scenario
(`REALISTIC_VARIANTS`) instead uses magnitudes defensible for an ocean
reanalysis, and five attribution groups:

| group | base | range tested (low → high) |
|---|---|---|
| `forcing` (amplitude, random, position) | +15%, 0.25 × RMS, 50 km | +10%, 0.15, 30 km → +20%, 0.35, 70 km |
| `rd` | −10% | −5% → −15% |
| `drag` (bottom) | −50% | −30% → −60% |
| `obs` (white + correlated per pass, fraction of ψ₁ std) | 15% + 15% | 10% + 10% → 20% + 20% |
| `res` (DA grid, truth 64) | 32 | 48 → 32 (plus 48 and 64 at base) |

The filter's R is the total observation-error variance, diagonal; the
per-pass correlation is not represented, as in operational systems. The
bred ensemble breeds on the DA grid from the downsampled truth.

## 3. Calibration target

The S1 intensity κ scales all four reference levels together. Target (user
request, 2026-09-27): per-window analysis EVs **typically** in

- upper-layer q: **0 to 0.25** (S0 val: 0.57);
- upper-layer ψ: **0.7 to 0.9** (S0 val: 0.93).

"Typically" is read as: the median over the 20 val windows inside the
range, and as many windows as possible inside it (fractions reported).

**Procedure** (`evaluation/qg_specwind_s1_sweep.py`, val, 20 windows, S0-tuned
DA):
1. `--phase calib`: each component alone at κ = 1, 2, 4, and all four at κ =
   0.5 to 4. Gives each component's dose-response and the full-S1 curve.
2. Choose κ* from the full-S1 curve. If no κ meets both targets at once
   (the two metrics need not degrade together), iterate the **mix**: rescale
   the components that mostly hit ψ₁ (expected: `amp`, `shift`) against
   those that mostly hit q₁ (expected: `noise`, `param`), guided by step 1,
   and re-run the full-S1 points.

## 4. Attribution

At the calibrated levels, run all 16 on/off combinations of the four
components (`--phase shapley --kappa κ*`) and report, per metric (ψ₁, ψ₂,
q₁, q₂, score):
- the **Shapley value** of each component: its average marginal EV loss over
  all orders of switching components on. The four values sum exactly to the
  S0 → full-S1 loss;
- the **stand-alone loss** of each component;
- the **interaction** term (full loss minus the sum of stand-alone losses):
  positive when errors compound, negative when they overlap.

Intervals: bootstrap over windows (the per-window Shapley values are
linear, so the bootstrap is exact per window).

## 5. DA settings under S1

The S0-tuned settings (ETKF radius 8, ridge 1, cross-layer 1, bred; EnKF
radius 6). No S1 re-tuning in this step: the point is how S0-tuned DA
degrades. Inflation (> 1) is the natural S1 knob and is a follow-up once
the attribution is known.

## 6. Test runs

At the selected levels (realistic base; "high" also run for reference): ETKF and EnKF, 100 test windows
of the forced and coupled datasets, with the free forecast, alongside the S0
DA-3 runs (`batch/run_qg_specwind_da.sbatch` with `PRESET=realistic KAPPA=1`).

## 7. Out of scope (for now)

- **4D-Var** (strong/weak constraint): excluded at the user's request.
- **Other structural errors:** the one-layer DA model (needs a spectral wind
  hook for `QG1LDynamics`), and the missing-feedback error (coupled truth,
  forced DA model), which is physically small (about 1% of the gyrostat
  tendency). Resolution is covered by the `res` group (v2).

## 8. Compute

About 17 min per configuration of 20 val windows on an rtx8000: 19
calibration + up to 14 new Shapley configurations ≈ 9 GPU-hours; test runs
≈ 4 GPU-hours.
