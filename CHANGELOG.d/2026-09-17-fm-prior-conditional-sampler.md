## 2026-09-17: FM-prior conditional sampler — blended flows, one integrator, six negatives

**Summary:** Adds a configurable conditional sampler for flow-matching priors and the
L96 study built on it. A deterministic mean estimator is a flow: its operator is the
constant map `Psi_det = m_hat`, so `v_det = (m_hat - x_tau)/(1-tau)`, which integrates
exactly to the SDEdit warm-start state. The warm start is therefore the discontinuous
member `lam(tau) = 1{tau < tau_0}` of the family `lam(tau) = lam_0 (1-tau)^p`, and
`p >= 1` is forced by `v_det`'s `1/(1-tau)` factor. The continuous member
`lam_0 = 0.5, p = 2` strictly dominates the warm start on both RMSE and spread.

**Files modified:**
- `evaluation/fm_sampler.py` — new: one integrator (`sample`) with pluggable schemes
  (`Cold`, `Warm`, `Blend`, `Decoupled`), `SamplerConfig`/`SchemeContext`, the Gaussian
  helpers (`prior_gain`, `prior_mean`, `estimate_prior_gain_variance`) and
  `EnsembleScores`/`reliability_target`
- `tests/test_fm_sampler.py` — new: 24 unit tests
- `reports/l96/run_sda_sampler_experiments.py` — new: driver, `--suite {blend,decoupled,all}`
- `reports/l96/generate_l96_fm_sampler_report.py` — new: renders the report from run JSON
- `reports/l96/outputs/l96_fm_sampler_benchmark.md` — new: the report
- `batch/run_l96_blended_flow_sampler.sbatch` — new: SLURM wrapper
- `reports/l96/sweep_pigdm_guidance.py` — adds `--variance-form {exact,sda}` (the
  Rozet & Louppe Eq. (15) form alongside the exact isotropic posterior variance) and
  the missing `--baseline-weight` argparse entry, which the script already read but
  never declared
- `docs/fm_prior_sampler_results.md` — pointer at the generated report, so results
  live in exactly one place

**Results.** Full 200-window test set, `N = 30` members (matching the benchmark's
ensemble size), `N_outer = 50`, SDA3 prior + obsdensity-augmented DirectUNet-M mean:

| config | rmse_repo | spread | ratio (target 0.967) |
|---|---|---|---|
| mean-only | 0.4714 | 0 | — |
| cold (`lam_0 = 0`) | 0.5295 | 0.6833 | 1.202 |
| warm (`tau_0 = 0.3`) | 0.3978 | 0.1211 | 0.284 |
| **blend `lam_0 = 0.5, p = 2`** | **0.3764** | 0.2369 | 0.585 |

`mean-only` reproduces the benchmark's `DirectUNet-M(obsdensity)` = 0.4727 to 0.3%,
which validates the harness. The blend beats its own warm-start baseline by 5.4% on
RMSE with 96% more spread — the headline result — and lands 3.2% ahead of the closest
benchmark row (`DirectUNet(aug)+SDA3` = 0.3889). The `warm` row decomposes that: at
0.3978 it is the same *scheme* as the benchmark row under this report's settings and
is 2.3% worse, so the guidance rule and `N_outer` (still unmatched) cost something and
the blend more than recovers it.

**Correction.** The cold sampler is consistently ~20-25% **over-dispersive**
(ratio/target 1.20 at `N = 6`, 1.22 at `N = 10`, 1.24 at `N = 30`), not "calibrated"
as stated earlier in this investigation. Stability across `N` shows this is a property
of the sampler, not of ensemble size.

**Six negative results** are recorded with diagnoses in the report: variance-corrected
warm start (the missing term is ~8% of the noise term), operator mean-shift (Eq. (11)
is output-only and valid only for affine `Psi`; the correct input-shifted form reduces
to post-hoc addition), post-hoc recentering, low-rank covariance (`N = 6` in 72000
dims), Desroziers calibration (`m_hat` is trained to fit the observations, so its
departures understate its error: `dep_fit` 0.147 < `R` 0.163), observation
cross-validation (~15x over-estimate under temporal holdout), and decoupled
mean/gain/NG control (`Psi_NG` is defined relative to `(mu_p, K_p)` and is not
invariant under a gain change).

**Two hazards made structural rather than documented.** `estimate_prior_gain_variance`
exposes **no** `guidance` flag — `P_prior` must come from an unguided pass, and
measuring it on a guided ensemble (0.153 instead of 0.804) silently moved a
calibration ratio from 0.854 to 1.190 three separate times during this work. And
writing the schemes as a family surfaced an exact identity: `Blend(lam0, p)` *is*
`Decoupled(lam0, p, gain="prior", scale_ng=True)`, now asserted as a test so the two
cannot diverge.

**Reproducibility defect found and fixed.** `prior_mean` drew from PyTorch's ambient
generator with no seeding, so `mu_p` varied run to run. It hid well:
`estimate_prior_gain_variance` seeds internally and every scheme's `sample` is
preceded by `torch.manual_seed`, so `P_prior` and all the blend rows were bit-for-bit
stable — and only `Decoupled` reads `mu_p`. Identical code gave `decoup a0=.5` =
0.6098 / 0.6102 / 0.6113 across three runs. `prior_mean` now takes a `seed`, with two
regression tests asserting independence from the ambient RNG state.

**Verification:** `ruff check` clean on all new files. `pytest` on the CI-gate file
list: **454 passed, 4 skipped**. `pytest tests/test_fm_sampler.py`: 24 passed — these
found a real division-by-zero in `EnsembleScores.summary()` when RMSE is exactly zero
(inherited from the pre-refactor code; the undefined ratio is now NaN). Determinism
verified by running the full suite twice and diffing all ten rows: bit-for-bit
identical. The refactor reproduces the pre-refactor blend results exactly on all seven
rows (`cold` 0.5951/0.6365/0.6452/1.014, `warm` 0.4048/0.4376/0.1124/0.257,
`lam0=0.5,p=2` 0.3898/0.4223/0.2224/0.527).
