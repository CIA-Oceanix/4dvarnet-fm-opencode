## 2026-09-17: FM-prior conditional sampler — blended-flow family + diagnosed negatives

**Summary:** Consolidates the L96 FM-prior *sampler* investigation. Positive result:
a deterministic mean estimator can be read as a degenerate flow (constant operator
`Psi_det = m_hat`, hence `v_det = (m_hat - x_tau)/(1-tau)`), which makes the SDEdit
warm start the discontinuous member `lam(tau) = 1{tau < tau_0}` of a one-parameter
family `lam(tau) = lam_0 (1-tau)^p`. The continuous member `lam_0 = 0.5, p = 2`
strictly dominates the warm start on both RMSE and spread. Six other schemes were
implemented, run, and rejected; their diagnoses are recorded.

**Files modified:**
- `reports/l96/fm_sampler_common.py` — new: shared gain/prior-mean/scoring helpers
- `reports/l96/run_blended_flow_sampler.py` — new: the blended-flow sweep
- `reports/l96/probe_sampler_covariance_control.py` — new: decoupled mean/gain/NG
  control (documented negative result + regression harness for the general velocity)
- `docs/fm_prior_sampler_results.md` — new: derivation, sweep, full-test-set table,
  and the six negatives with diagnoses
- `reports/l96/sweep_pigdm_guidance.py` — adds `--variance-form {exact,sda}` (the
  Rozet & Louppe Eq. (15) form `Gamma alpha^2/beta^2` alongside the exact isotropic
  posterior variance) and the missing `--baseline-weight` argparse entry, which the
  script already read but never declared

**Rationale:** The scoping calls for a *sampler* baseline built on an FM prior, not
another conditional-mean estimator. The cold sampler is the calibrated baseline
(spread/RMSE 1.01 against a 0.845 target) but trails on RMSE; the blend is the only
configuration found that improves both axes at once relative to the warm start.

`estimate_prior_gain_variance` deliberately exposes **no** `guidance` flag: `P_prior`
must be measured on an unguided pass, and estimating it on a guided ensemble
(`P_prior = 0.153` instead of `0.804`) silently moved a calibration ratio from 0.854
to 1.190 three separate times during this work. Inlining the unguided integration
removes the flag there was to forget.

**Verification:** `ruff check` clean on all new files; CI-gate test files run via
`batch/` on a dedicated node. Full 200-window test set
(`experiments/l96_datasets_obsj2_int100_nwin200.pt`, N = 10, `N_outer = 50`):

| config | rmse_repo | spread | ratio (target 0.905) |
|---|---|---|---|
| mean-only | 0.4715 | 0 | — |
| cold | 0.5583 | 0.6599 | 1.104 |
| warm (`tau_0 = 0.3`) | 0.3990 | 0.1168 | 0.273 |
| **blend `lam_0 = 0.5, p = 2`** | **0.3813** | 0.2285 | 0.557 |

The `mean-only` row reproduces the benchmark's `DirectUNet-M(obsdensity)` = 0.4727 to
0.25%, which validates the harness against the consolidated benchmark. The blend beats
its own warm-start baseline by 4.4% on RMSE with 96% more spread — the headline result.
It also lands 2.0% ahead of the closest benchmark row (`DirectUNet(aug)+SDA3` = 0.3889),
but the guidance rule, `N_outer` and member count all differ there, so the doc presents
that as indicative rather than a like-for-like claim.

**Correction recorded in the doc:** the cold sampler is ~22% *over-dispersive*
(1.104 against a 0.905 target), not "calibrated" as earlier stated in this
investigation; it remains the closest to reliable of anything measured.
