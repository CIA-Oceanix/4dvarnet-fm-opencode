# QG DA — q-state vs psi-state comparison

**Date:** 2026-09-03
**Scope:** ETKF (N=80, inflation 1.0, Gaspari–Cohn radius 6), psi-obs (upper-layer streamfunction, 4 random meridional columns/day), lag 1.0 d. S0 (error-free) and S1 (param bias + corrupted wind); S1 at same-res da_nx=64 and cross-res da_nx=32 (vs 64×64 truth).

## Decision: q-state is the default DA configuration

The **q-state** DA formulation (PV `q` as the forecast/analysis state, with the streamfunction computed on demand via spectral inversion for psi-obs) is the **production default** for QG data assimilation. The **psi-state** variant (`obs_var="psi_state"`), which carries the streamfunction as the state and converts q↔ψ each integration step, is a research alternative retained for comparison, not the default.

- Code default: `--obs-var` resolves to `"q"` (q-state) in both `evaluation/run_qg_baselines.py` and `evaluation/sweep_qg_baselines.py`.
- The q-state DA scenario (psi-obs, 4 cols/day) is the configuration used for the production S0/S1 QG DA-baseline runs and for the `reports/qg/generate_qg_s0s1_report.py` consolidated report.

## 1. Representations

- **q-state (default):** the DA state is potential vorticity `q`. The psi-obs operator inverts `q → ψ` spectrally each observation step. The metrics measure the PV q field directly (well-conditioned).
- **psi-state:** the DA state is the streamfunction `ψ`. The psi-obs operator is an H-mode read of the psi-state (`_PsiMixin.streamfunctions` is the identity reshape) spectrally upsampled to the obs grid. The metrics convert the psi analysis back to PV via `forward_pv` (q ≈ ∇²ψ) before computing q-field skill.

## 2. Per-case results

### S0 (error-free)

| field | metric (layer1 / layer2 / full) |
|---|---|
| q-state (default) q | +0.815 / +0.688 / +0.752 |
| q-state (default) ψ | +0.966 / +0.971 / +0.968 |
| psi-state q | +0.762 / +0.403 / +0.583 |
| psi-state ψ | +0.976 / +0.978 / +0.977 |

- Headline `expvar_full` (qall): q-state +0.752 vs psi-state +0.583.

### S1 same-res (da_nx=64)

| field | metric (layer1 / layer2 / full) |
|---|---|
| q-state (default) q | +0.552 / +0.304 / +0.428 |
| q-state (default) ψ | +0.435 / +0.222 / +0.328 |
| psi-state q | -1.186 / -4.664 / -2.925 |
| psi-state ψ | +0.814 / +0.398 / +0.606 |

- Headline `expvar_full` (qall): q-state +0.428 vs psi-state -2.925.

### S1 cross-res (da_nx=32)

| field | metric (layer1 / layer2 / full) |
|---|---|
| q-state (default) q | +0.458 / +0.221 / +0.340 |
| q-state (default) ψ | +0.801 / +0.806 / +0.804 |
| psi-state q | -1.371 / -5.061 / -3.216 |
| psi-state ψ | +0.795 / +0.392 / +0.594 |

- Headline `expvar_full` (qall): q-state +0.340 vs psi-state -3.216.

## 3. Summary table

| case | q-state qall | psi-state qall | q-state ψ1/ψ2 | psi-state ψ1/ψ2 |
|---|---|---|---|---|
| S0 (error-free) | +0.752 | +0.583 | +0.966 / +0.971 / +0.968 | +0.976 / +0.978 / +0.977 |
| S1 same-res (da_nx=64) | +0.428 | -2.925 | +0.435 / +0.222 / +0.328 | +0.814 / +0.398 / +0.606 |
| S1 cross-res (da_nx=32) | +0.340 | -3.216 | +0.801 / +0.806 / +0.804 | +0.795 / +0.392 / +0.594 |

## 4. Interpretation

- **The q-state (default) wins on the PV q field in every case.** On S0 at noise 0.01 qall is 0.752 vs psi-state 0.583 (gap dominated by q2, 0.688 vs 0.403); on S1 same-res it is 0.428 vs −2.93 and on da_nx=32 0.340 vs −3.22, where the psi-state q field collapses to strongly negative EV.
- **psi-state is competitive on the streamfunction field.** On S0 its ψ1/ψ2 (0.976/0.978) is the best per-field result, slightly above q-state (0.966/0.971); on S1 same-res its ψ1 (0.814) clearly beats q-state (0.435). But on S1 cross-res da_nx=32 q-state ψ2 (0.806) exceeds psi-state ψ2 (0.392).
- **Why the psi-state q field is degenerate:** the PV diagnostic q ≈ ∇²ψ (via `forward_pv`) amplifies high-wavenumber psi-analysis error by K², so a psi analysis that is skilful in bulk (large-scale dominated, positive ψ EV) still has small-scale error that explodes under the PV conversion. This is most severe on the corrupted S1 case and grows with the psi→q round-trip; it is a physical ψ↔q representation limitation, not a code bug (free forecasts are bit-identical).
- **Conclusion:** the q-state representation is the robust default: it keeps the PV q field — the physical variable the benchmark scores — well-conditioned across S0 and the corrupted cross-resolution S1 alike. The psi-state variant remains useful for studying streamfunction-direct observations but is not the production DA configuration.
