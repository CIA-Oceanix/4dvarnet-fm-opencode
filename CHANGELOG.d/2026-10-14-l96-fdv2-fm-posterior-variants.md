## 2026-10-14: L96 FDV2 x FM posterior — M1/M1b/M1bI/M1c methods and the validation sweeps (Phase 1)

**Summary:** Registers the FM-posterior candidates of the plan in the harness. M1 runs `fm_posterior_sample` (SDA1-M prior) from the FDV2 mean. M1b runs it from the perturbed-obs FDV2 ensemble (B2), with `obs_pert` scaling the obs perturbation. M1bI is M1b inflated to SSR = 1 like B5: tuned in-sample on val, fixed through `inflation_from` on test. M1c picks a per-window tau0 from the FDV2 prior misfit J_prior; `starts=perturbed` gives it perturbed-obs starts. The batch script can now run several rows per process.
**Files modified:**
- `eval_fdv_fm_posterior_l96.py`:
  - M1/M1b/M1bI/M1c. `r_units=norm` is the default, R_norm = R_var / sigma_clim^2 per channel; `r_units=raw` reproduces B3's scalar R.
  - The `MethodSpec.prepare` pre-pass hook, used by M1c.
  - `--argv-file` multi-row mode.
  - A per-process FDV2 mean cache.
- `batch/run_l96_fdv2_fm_posterior.sbatch`: optional WINDOWS/NAME row fields, `ROWS_PER_TASK`, `SKIP_EXISTING`.
- `tests/test_posterior_scores.py`: M1 == B3 equivalence, stochastic spread, M1b/M1c (both start modes), `obs_pert`, M1bI registration, the mean cache.
**Rationale:**
- Plan `docs/plans/analysis/l96_fdv2_fm_posterior.md`, Phases 0-1.
- M1's `n_steps` counts steps on [tau0, 1], whereas B3's counts the full [0, 1] grid. So B3(tau0=0.5, n_steps=10) equals M1(tau0=0.5, n_steps=5).
- Validation sweeps: 488 rows, 30 members, both layouts, FDV2 seeds 1 and 2. FM rows are scored on val windows 0-39 and paired-bootstrapped against the hybrid on the same windows.
  - The hybrid's best settings are gw = 1: tau0 = 0.5 for RMSE, tau0 = 0.1 for CRPS.
  - The sampler noise gamma, correctors and n_steps = 20 add little.
  - DPS gives the same numbers as PiGDM at a third of the cost.
  - Perturbed-obs starts plus inflation carry the gain. M1bI (DPS, gamma = 0.5, tau0 = 0.7, `obs_pert` = 0.75) gives CRPS -4.9..-6.7% vs the hybrid's CRPS-best and B5, with SSR 1 and cov90 0.90-0.93. Its RMSE is -0.4..+1.1% vs the hybrid's RMSE-best.
  - `obs_pert` = 0.5 matches the RMSE but over-disperses the tails.
  - No configuration met all the pre-registered G1 criteria in every cell; two frozen configurations go to test.
**Verification:** `pytest tests/test_posterior_scores.py tests/test_fm_posterior_sampler.py -m "not slow"`: 39 passed. GPU smoke (job 60515, A40, 10 val windows): M1(gamma=0, normalized, gw=2, tau0=0.5, n_steps=5, r_units=raw) equals B3(tau0=0.5, gw=2) to 3e-7 per window.
