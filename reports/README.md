# `reports/` — index

**`reports/` holds generated numbers (tables, figures and the scripts that write them);
the written findings that interpret them are in `docs/results/`, and each names the
outputs its numbers come from.**

Every checked-in report under `reports/*/outputs/` is listed here with the
question it answers, its status, the protocol its numbers were produced under,
and the script that writes it. `tests/test_reports_index.py` fails when a report
is missing from this index or a listed generator does not exist.

Outputs are generated: fix the generator **and** the checked-in output together,
or the next regeneration reverts you. Hand-written findings belong in
`docs/results/`, not here.

## A CURRENT report is regenerated when its model configs change

The configs a report depends on are derived from its generator's source. That
means every `config/experiment/<name>.yaml` whose name it mentions, plus any
`config/*.yaml` path it names, followed through each config's Hydra `defaults:`
(so `config/lorenz96_default.yaml` and `config/l96_benchmark_default.yaml` count).
`tests/test_reports_index.py` fails when any of them differs, as parsed YAML,
from the version at the report's last commit. Comment-only edits therefore do
not trip it.

When it fails, re-train or re-evaluate the affected rows and regenerate the
report in the same PR as the config change. If the report deliberately stays on
the old configuration, change its status to **FROZEN**. The check applies to
CURRENT L96 reports; QG and L63 are exempt until their rework (below).

## Where report inputs come from

The three L96 benchmark generators (`generate_p1_l96_benchmark.py`,
`generate_l96_benchmark_default_report.py`, `generate_l96_benchmark_extended_report.py`)
read their inputs only through `reports/l96/_inputs.py`. Each named input root
(`here`, `bench`, `p1`, `n20`, `da_random_layout`, `random_obs_times`) resolves to
that report's **input bundle**,
`<main checkout>/experiments/l96/report_inputs/<report>/<root>/`. A bundle holds
hard links of exactly the files the report was generated from, with a
`MANIFEST.json`. Test sets and shared DA caches come from the main checkout's
`experiments/` (`_inputs.shared()`). So these reports regenerate from any
checkout, and pruning a topic worktree cannot break them.
`tests/test_report_inputs.py` fails if any report script outside `reports/qg/`
contains a worktree path or an absolute repository path.

To add or refresh rows, produce the new estimates, then rebuild the bundle with
`scripts/bundle_report_inputs.py`:
- `record` runs the generator against the source directories under an audit hook;
- `link --apply` hard-links what it read;
- `run` regenerates from the bundle alone.

`batch/run_l96_bundle_report_inputs.sbatch` does all three and diffs the outputs
after each step. A migration is complete only when the report regenerated from
the bundle is byte-identical.

## Written findings that cite these outputs

Derived from the `**Numbers:**` line of each note in `docs/results/`; `tests/test_docs_layout.py`
fails if a note cites an output that is not listed here against it.

| output | discussed in |
|---|---|
| `reports/l96/outputs/cfm_tau_consistency/` | `docs/results/l96_cfm_tau_consistency.md`, `docs/results/l96_cfm_tau_consistency_ns1.md` |
| `reports/l96/outputs/cfm_tau_consistency/batch1/` | `docs/results/l96_cfm_tau_consistency_batch1.md` |
| `reports/l96/outputs/cfm_tau_consistency/l96b/` | `docs/results/l96_cfm_tau_consistency_l96b.md` |
| `reports/l96/outputs/cfm_tau_consistency/sampler/` | `docs/results/l96_cfm_sampler_schedule.md` |
| `reports/l96/outputs/cfm_tau_consistency/t5/` | `docs/results/l96_cfm_tau_consistency_t5.md` |
| `reports/l96/outputs/fm_score/fm_score_bootstrap.md` | `docs/results/l96_p1_fm_score.md` |
| `reports/l96/outputs/fm_score/current/` | `docs/results/l96_p1_fm_score.md` |
| `reports/l96/outputs/fm_score/fm_score_current.md` | `docs/results/l96_p1_fm_score.md` |
| `reports/l96/outputs/fm_score/fm_score_p1_400ep.json` | `docs/results/l96_p1_fm_score.md` |
| `reports/l96/outputs/fm_score/scale/` | `docs/results/l96_p1_fm_score.md` |
| `reports/l96/outputs/l96_benchmark_default.old.md` | `docs/results/l96_benchmark_default_paper_audit.md` |
| `reports/l96/outputs/l96_benchmark_extended.md` | `docs/results/l96_benchmark_default_paper_audit.md`, `docs/results/l96_cfm_velocity_ensembles.md`, `docs/results/l96_etks_benchmark.md`, `docs/results/l96_etks_default_inflation.md`, `docs/results/l96_weak4dvar.md` |
| `reports/l96/outputs/p1_l96_benchmark.md` | `docs/results/l96_weak4dvar.md` |
| `reports/l96/outputs/l96_da_obs_count_dafw.md` | `docs/results/l96_benchmark_default_paper_audit.md` |
| `reports/l96/outputs/rank_histograms/` | `docs/results/l96_rank_histograms_c4.md` |
| `reports/l96/outputs/da_inflation_sweep/` | `docs/results/l96_da_inflation_post291.md` |
| `reports/qg/outputs/qg_specwind_da_report.md` | `docs/results/qg_specwind_da_s0_s1_test.md`, `docs/results/qg_specwind_enks.md`, `docs/results/qg_specwind_s1_tuning.md`, `docs/results/qg_specwind_ensemble_size.md`, `docs/results/qg_specwind_4dvar.md` |
| `reports/qg/outputs/qg_spectral_wind_report.md` | `docs/results/qg_spectral_wind_calibration.md` |

## Status vocabulary

| status | meaning |
|---|---|
| **CURRENT** | Produced under the current protocol; cite it for current numbers. |
| **FROZEN** | A finished study that is valid under the protocol it states, and is not refreshed when defaults change. Cite it for its own question, not for headline numbers. |
| **STALE** | A later bug fix or default change invalidated its numbers, and it has not been re-run. Do not cite it until it is. |
| **SUPERSEDED** | Replaced by the report named in its row. Kept for provenance. |

A note after the status says which cells it qualifies.

## Headline reports

| case | cite for current numbers |
|---|---|
| L96 | `reports/l96/outputs/l96_benchmark_extended.md` (the benchmark-default report it followed up is frozen as `l96_benchmark_default.old.md`). `p1_l96_benchmark.md` is the P1 paper's table, on the same benchmark framework, with the P1-protocol tier study as its annex. |
| QG | `reports/qg/outputs/qg_neural_report.md` (DA and neural rows); `qg_da_report.md` for the DA configuration. |
| QG gyrostat (spectral wind) | `reports/qg/outputs/qg_specwind_da_report.md`: DA baselines S0 + realistic S1 on the forced and coupled test sets, with the S1 error budget and reconstruction examples. |
| L63 | none — see the L63 section. |

## L96

Current protocol (as of #257/#258): flows are scored with 30 members × 20
early-fine Euler steps (`ens30_no20`), SDA with the guided `ens30_gw20` protocol
(guidance weight 25 in the extended report), and ETKF/EnKF with inflation
1.5 (S0) / 2.0 (S1) and CRPS on the analysis ensemble. The consolidated report
uses the older protocol (`ens30_no10` uniform steps, DA inflation 2.0).

The two current benchmark reports (`l96_benchmark_extended.md`, `p1_l96_benchmark.md`) end with the same section, a per-window RMSE / EV / CRPS summary of the current benchmark rows (`reports/l96/per_window_summary.py`, `benchmark_extended` inputs). The consolidated report is not updated: it uses the old protocol, and the scoping docs quote its digits.

| report | question | status | protocol | generator |
|---|---|---|---|---|
| `l96_benchmark_default.old.md` | DA vs learned schemes under the benchmark-default training recipe (400 epochs), on the regular and the random observing system | **FROZEN** (2026-10-02) — superseded by `l96_benchmark_extended.md`, which carries its protocol and every row at the current settings (DA before #291/#295, no ETKS, SDA gw 20) | 400-epoch, pre-#291 DA | `reports/l96/generate_l96_benchmark_default_report.py` |
| `l96_benchmark_extended.md` | The L96 benchmark: DA vs learned schemes on the regular and the random observing system, plus training budget, observing-system dependence, SDA and hybrids, DA CRPS and ensemble size, marginal value of observations | **CURRENT** — SDA S1 rows re-evaluated in #265 | current (SDA gw 25) | `reports/l96/generate_l96_benchmark_extended_report.py` |
| `p1_l96_benchmark.md` | P1 paper table: DA vs deterministic, flow-matching and SDA under the benchmark-default framework (regular + random test sets, 1200 epochs, 3 seeds); annex A = the P1 protocol (fixed observing system, 400 epochs) for the S+/M/L tier, CFM-parameterization and SDA-conditioning comparisons | **CURRENT** — sections 1-4 moved to the benchmark framework 2026-10-02 | current | `reports/l96/generate_p1_l96_benchmark.py` |
| `l96_da_random_layout.md` | ETKF/EnKF/4D-Var under the random observing system | **CURRENT** | current DA | `reports/l96/generate_l96_da_random_layout_report.py` |
| `l96_da_obs_count_dafw.md` | ETKF/EnKF RMSE vs number of observation times per window | **CURRENT** | current DA | `reports/l96/generate_l96_da_obs_count_report.py` |
| `l96_consolidated_benchmark.md` | Earlier full benchmark: DA baselines vs every neural family, reproducibility-audited | **SUPERSEDED** as the headline by `l96_benchmark_extended.md`; still the report the paper-scoping docs quote by digit | old (`ens30_no10`, inflation 2.0) | `reports/l96/generate_l96_consolidated_report.py` |
| `l96_consolidated_benchmark.old.md` | The consolidated benchmark before the 2026-09-18 reproducibility refactor | **SUPERSEDED** by `l96_consolidated_benchmark.md` (frozen, not regenerated) | pre-2026-09-18 | none (frozen snapshot) |
| `p1_mean_component.md` | `Ψ_mean` alone vs the full CFM | **FROZEN**; its `full RMSE` column is `ens30_no10`, from before #257 | old flow sampler | `reports/l96/eval_mean_component.py` (`--output`) |
| `l96_fm_sampler_benchmark.md` | Conditional sampling from a flow-matching prior (Cold / Warm / Blend / Decoupled) | **FROZEN** (measurements 2026-09-17) | own protocol, stated in the report | `reports/l96/generate_l96_fm_sampler_report.py` (data: `reports/l96/run_sda_sampler_experiments.py`) |
| `l96_obs_density_augmented_training.md` | Does fast-Y observation-density augmentation during training buy robustness at reduced density? | **FROZEN** | own protocol | `reports/l96/generate_l96_obs_density_augmented_report.py` |
| `l96_normalization_ablation.md` | Per-channel z-score normalization ablation (led to the `data.normalize` default) | **FROZEN** | pre-monai | `reports/l96/generate_l96_normalization_ablation.py` |
| `l96_tweediecfm_benchmark.md` | TweedieCFM (V2 family) vs PredictStateCFM (V3) | **FROZEN** | pre-monai, `ens30_no10` | `reports/l96/generate_l96_tweediecfm_report.py` |
| `ens30_seed_report.md` | 5-seed reproducibility of the L3 multi-τ CFM ensemble | **FROZEN** | pre-monai | `reports/l96/generate_ens30_seed_report.py` |
| `l96_joint_da_benchmark.md` | Joint state–parameter ETKF/EnKF (Phase C) | **FROZEN** | pre-monai | `reports/l96/generate_l96_joint_da_report.py` |
| `l96_joint_neural_benchmark.md` | Joint state–parameter neural estimation (Phase C) | **FROZEN** | pre-monai | `reports/l96/generate_l96_joint_neural_report.py` |
| `l96_joint_param_diagnostic.md` | Per-parameter diagnostic of the joint neural estimates | **FROZEN** | pre-monai | `reports/l96/diagnose_joint_params.py` |
| `l96_da_baselines_obsj2_vs_obsj0.md` | DA baselines with 24D vs slow-only 8D observations | **FROZEN** | DA inflation 2.0 | `reports/l96/generate_l96_obs_density_report.py` |
| `s0_s1_obs_density_da_baselines.md` | DA baselines at Obs200 vs Obs100 (2026-08-19) | **FROZEN** | DA inflation 2.0 | none in the repo |
| `l96_grad_checkpoint_benchmark.md` | Memory/time cost of gradient checkpointing in the FDV solvers (engineering) | **FROZEN** | synthetic batches | `reports/l96/generate_l96_grad_checkpoint_benchmark.py` |

The `reports/l96/probe_*.py` scripts write the data behind `docs/results/` notes,
not reports; their outputs live in subfolders of `reports/l96/outputs/` (for
example `cfm_tau_consistency/`, `rank_histograms/`).

## QG

> **Rework planned.** The QG reports will be reorganized later; the statuses
> below are provisional and QG is exempt from the config-freshness check until
> then.

The QG DA defaults changed on 2026-09-12 (`etkf_ridge`) and 2026-09-16
(`loc_radius` 2.0 / 1.0, `etkf_ridge` 0.1). Reports from before those dates keep
their own stated configuration.

**Localized ETKF caveat (2026-09-28).** Every localized-ETKF number in the
QG reports below, except `qg_specwind_da_report.md`, used the legacy
`ETKF(loc_mode="square_root")` update. Its gain divides R by N − 1
(unnormalized covariances), and it applies the full gain to the anomalies,
which over-contracts the spread. The tuned `etkf_ridge` partly stood in for
R, so those ETKF rows reflect that configuration, not an exact ETKF; the
EnKF rows are unaffected (`docs/results/qg_specwind_etkf_loc_update.md`).
The reports are left as they are. `qg_specwind_da_report.md` uses the exact
EnSRF update (`loc_mode="ensrf"`).

| report | question | status | protocol | generator |
|---|---|---|---|---|
| `qg_neural_report.md` | QG benchmark: DA baselines vs DirectUNet Q1–Q4 across S0/S1 | **CURRENT** | current DA defaults | `reports/qg/generate_qg_neural_report.py` |
| `qg_da_report.md` | QG DA baselines under the current defaults | **CURRENT** | current DA defaults | `reports/qg/generate_qg_da_report.py` |
| `qg_obs_density_report.md` | QG DA vs observation density (cols 1–64) — the study that set `loc_radius` | **CURRENT** | current DA defaults | `reports/qg/generate_qg_obs_density_report.py` |
| `da_sensitivity_s0_s1_report.md` | QG ETKF/EnKF/4D-Var sensitivity sweeps (inflation, ridge, `b_var_scale`/`q_var_scale`) | **FROZEN** — its conclusions were promoted to the defaults | sweep | `reports/qg/generate_da_sensitivity_report.py` |
| `qg_s0s1_report.md` | QG S0/S1 DA comparison (2026-09-06) | **FROZEN**; predates the 2026-09-12/16 DA defaults | old DA defaults | `reports/qg/generate_qg_s0s1_report.py` |
| `qg_psi_state_report.md` | Assimilating ψ instead of q (the H3a evidence) | **FROZEN**; predates the 2026-09-12/16 DA defaults | old DA defaults | `reports/qg/generate_qg_psi_state_report.py` |
| `qg1l_report.md` | One-layer QG variant | **FROZEN** | old DA defaults | `reports/qg/generate_qg1l_report.py` |
| `figure_qg_lag_loc_sweep_report.md` | Lag × localization sweep figure | **FROZEN** | old DA defaults | none in the repo |
| `qg_spectral_wind_report.md` | First QG simulations under the Option B spectral wind drivers (storm, spectral OU, gyrostat, surrogate) | **CURRENT** — illustrative single realizations, no statistical claims | simulation only (no DA), nominal ocean, 2-year unforced spin-up + 120 days | `reports/qg/simulate_qg_spectral_wind.py` |
| `qg_specwind_da_report.md` | QG gyrostat case study: ETKF/EnKF on the spectral-wind test sets (forced and coupled), S0 and realistic S1, in the layout of `qg_da_report.md` (per-scenario tables with ETKF, EnKF and the localized EnKS smoother, configuration synthesis + S1 error budget, density, forced vs coupled, strata, best/median/worst reconstructions + DA-cycle GIFs) | **CURRENT** | val-tuned DA-2 settings (ETKF radius 8 / ridge 1, EnKF radius 6, cross-layer 1, bred init); S1 realistic base | `reports/qg/generate_qg_specwind_da_report.py`; figures `reports/qg/generate_qg_specwind_da_figs.py` |

## L63

> **Rework planned.** Like QG, L63 will be reworked later; the statuses below
> are provisional and L63 is exempt from the config-freshness check.

**Every L63 report predates the 2026-09-18 fix of the shared observation-noise
bug** in `data/lorenz63.py` (`docs/plans/tech/multi_refactor_plan.md`, "Known
bug found during Phase 0"). Only datasets built on the base `Lorenz63Dataset`
are affected. The randomized-parameter and random-bias datasets always drew
per-window noise.

| report | question | status | protocol | generator |
|---|---|---|---|---|
| `s0_s1_synthesis.md` | L63 S0/S1 synthesis of DA baselines and the E/F/G/S series | **STALE** (base dataset) | pre-2026-09-18 | none in the repo |
| `s0_s1_baselines_interp_init.md` | L63 S0/S1 DA baselines with interpolated initialization | **STALE** (base dataset) | pre-2026-09-18 | none in the repo |
| `cs4b_randombias_baselines_results.md` | CS1–CS4b DA baselines, per-window random bias | **STALE** for the CS1/CS2 columns (base dataset); the CS3/CS4/CS4b columns stand | pre-2026-09-18 | none in the repo |
| `joint_estimation_baselines_results.md` | Joint vs state-only DA on CS3/CS4 (randomized parameters) | **FROZEN** — randomized datasets, unaffected by the bug | own protocol | none in the repo |

The L63 PDFs in `reports/l63/outputs/` are written by the `reports/l63/generate_*.py`
scripts and inherit the same staleness as their inputs.
