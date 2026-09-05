# 4DVarNet-FM: Implementation Plan

## Overview

Three model families + CS3/CS4 randomized-parameter tests + Experiment G ablation.

- **DirectUNet**: Single UNet pass `obs → state` via MSE (no flow matching) — implemented
- **VanillaCFM**: Standard conditional flow matching, no Tweedie decomposition — implemented
- **TweedieSolver**: Original two-stage solver — legacy, maintained
- **RandomParamDataset**: Per-window randomized `σ,ρ,β` ±20% for robust training — implemented
- **CS3/CS4**: Evaluation on unseen random parameter draws — implemented
- **Exp G (τ=0 CFM)**: Ablate multi-τ training to isolate CFM's source of advantage — implemented (L63; τ=0 runs superseded by the S-series, also L63)

**System naming convention:** the E/F/G/S experiment series are all **Lorenz-63**
(`train_mix: cs1+cs2`, `state_dim=3` checkpoints). The **L-series** is the two-scale
**Lorenz-96** benchmark (`system: lorenz96`, `state_dim=24` observed subspace).

## QG (two-layer quasi-geostrophic) — integrated to master 2026-09-02

DA-baseline case study on a two-layer Phillips channel (pyqg-compatible), now with
the **full executable codebase integrated to master** (previously only the JSON-only
report + generator were on master).

- **Dynamics**: `models/qg_dynamics.py` (`QGDynamics`, 2-layer) + `models/qg1l_dynamics.py`
  (`QG1LDynamics`, reduced-gravity 1-layer structural-error model) + `models/qg_interp.py`
  (spectral resize). Native torch port of pyqg v0.4.0 (RK4, flux-form advection, pyqg
  exponential filter, masked PV inversion), moving-storm wind-stress curl forcing.
- **Data**: `data/qg.py` (`QGConfig`, `QGDataset`, `make_qg_s0_s1_datasets`) — S0/S1
  along-track + random-column obs, corrupted wind/param bias, qg1l structural-error scenario.
  **Random-column geometry (2026-09-05):** `cols_per_day` distinct meridional columns are now
  each observed exactly once per day at its **own** randomly-sampled intra-day step (no two
  columns of a day share a step) — `obs (T,ny)`, `obs_columns (T,)`, `OBS_GEOMETRY_VERSION=2`
  folded into the truth-cache key so a geometry change auto-invalidates old obs caches.
  DA re-run vs the old single-simultaneous-event-per-day constellation: S0 cols=8/lag 1.0
  clearly improves (RMSE −7.7%, EV +0.777→+0.812), cols=4 essentially unchanged; S1 da_nx
  32/64 slightly degrades (RMSE +6–7%, EV −9 to −12 pts) since dispersing the simultaneous
  multi-column updates reduces each update's spatial info under model error.
- **DA baselines**: `evaluation/run_qg_baselines.py` (+ `eval_qg1l_rscale_probe.py`,
  `sweep_qg_baselines.py`) drive EnKF/ETKF on ψ/q. `evaluation/baselines.py` carries the
  shared `ObsOperator` H-mode, `_build_qg_loc_matrices`/`_build_qg_col_loc_matrices`,
  and per-time `loc_Lx_t`/`loc_Ly_t` localization + `init_ensemble` in ETKF/EnKF, merged
  with the L96/joint/ES work.
- **Psi-state variant**: `models/qg_psi_dynamics.py` (`QGPsiDynamics`/`QG1LPsiDynamics`,
  `wrap_psi`) integrates with the **streamfunction as the state variable**, so the psi
  observation operator reduces to a trivial index lookup (`obs_var="psi_state"`). The
  q-space physics is bit-identical to the q-state model (windows around the same
  `_rk4_step`/filter/clip), so free forecasts match to ~1e-6 relative and ETKF skill is
  comparable to the legacy H-function psi-obs. Works for same-resolution S0 + full-res
  qg1l. **Cross-resolution S1 (da_nx 32/16) is now supported** (2026-09-03) via an H-mode
  psi obs operator (`_psi_h` spectrally upsamples the DA-model psi-state to the obs grid
  before column selection); same-res behavior is bit-identical (re-verified EV −2.925).
  **Caveat:** psi_state DA gives a skilful **streamfunction** analysis (psi full EV +0.59
  at da_nx=32) but a degenerate **PV (q) field** (q full EV −3.2 vs the q-state da_nx=32
  ref +0.34): `forward_pv` (q ≈ ∇²ψ) amplifies high-wavenumber psi-analysis error by K²,
  so the q-field skill score (which `expvar_full` reports) collapses while the psi field
  itself is well-estimated. This is a physical psi↔q representation limitation, not a code
  bug (free forecasts still match exactly). On S0 at the q-state-matching noise 0.01 the
  psi-state q-field EV is 0.583 (q1 0.76 / q2 0.40) vs the q-state+psi-obs 0.752 (0.82/0.69),
  while psi-state's streamfunction fields are the best per-field (psi1/psi2 ≈ 0.98); at
  default 0.05 noise the S0 psi-state qall is 0.487 vs the q-state 0.660.
  **Decision (2026-09-03): q-state is the default DA config** — recorded in the dedicated
  report (`reports/qg/generate_qg_psi_state_report.py` → `qg_psi_state_report.md`) and in
  the `--obs-var` default (`'q'`) / help text of `run_qg_baselines.py` and
  `sweep_qg_baselines.py`. psi_state/psi remain research alternatives.
- **Report**: `reports/qg/generate_qg_s0s1_report.py` (JSON-only) renders from the result
  JSONs under `reports/qg/outputs/` → `reports/qg/outputs/qg_s0s1_report.md` (revised:
  governing equations, case-study table, S0 / S1-QG2L da_nx 16/32/64 / S1-QG1L sections,
  psi-obs focus). Dedicated `reports/qg/generate_qg1l_report.py` → `qg1l_report.md` for
  the reduced-gravity structural-error case (r-scale sweep).
- **Illustrations (S0 / S1-QG2L da_nx=32)**: `reports/qg/generate_qg_s0s1_figs.py` (no
  DA-cache dependency) runs a single-window production ETKF (nx=64, N=80, psi-obs,
  cols=4, 1% noise, lag 1.0) and writes per-scenario obs-days 2×2 panel + full-window
  obs Hovmöller, forcing, truth psi/q, analysis-vs-free-vs-truth, and a DA-cycle GIF to
  `reports/qg/outputs/figs/`. `generate_qg_s0s1_report.py` §8 embeds them
  (`![](figs/...)`, still JSON-only). The generator now (2026-09-03 fix) selects the
  **first window with a non-zero wind amplitude** (the S1 wind-levels list starts at
  0.0, so window 0 previously rendered a flat all-zero wind-curl forcing), draws the
  **corrupted** wind-curl (so the S1 figure shows the actual corrupted moving storm),
  fixes the DA-cycle GIF's per-panel `ax` handling (truth q₁ and DA-analysis q₁ panels
  were blank), and rebuilds the obs Hovmöller as a time×column storm-track field with
  cross-time interpolation (previously ~96% blank). Regenerated production figures
  committed alongside the fix. A follow-up DA-cycle obs-panel fix (2026-09-04) renders
  the **raw observations** (not a per-day aggregate): observed meridional columns are
  drawn **vertical** (removed the `img.T` transpose that had rotated them horizontal),
  at a single window-wide **fixed color scale** (previously re-normalized per frame),
  and each frame shows the nearest preceding raw obs event so the panel is never blank.
- **Tests**: 7 QG test files (`test_qg_dynamics`, `test_qg_data`, `test_qg_baselines`,
  `test_qg_s0s1`, `test_qg_random_columns`, `test_qg1l_dynamics`,
  `test_qg_psi_state`) — all in the master CI gate.
- **sbatch**: 32 `batch/run_qg_*.sbatch` for the S0/S1 matrix + S1-resolution + qg1l sweeps
  (+ `run_qg_figs.sbatch`, the illustration/DA-cycle regeneration job).
- QG is DA-baseline-only (no QG neural estimator; not wired into `train.py`/`get_dynamics()`).

## L96 (two-scale Lorenz-96) — merged to master 2026-08-18

- **Dynamics/DA baselines** (`feat/weighted-fast-coupling` merged into master, SW/MAOOAM excluded):
  - `models/lorenz96_dynamics.py`, `data/lorenz96.py` — two-scale L96 (NO=8, J=4, state_dim=40, weighted fast coupling)
  - `evaluation/run_l96*.py` + `reports/outputs/l96_baseline_report.md` — Waves 1-4 DA sweeps + ETKF ablation, pooled-EV metric
  - `models/dynamics.py` `get_dynamics()` supports only `lorenz63`/`lorenz96` (SW/MAOOAM deferred)
- **Training infrastructure** (built, no runs launched):
  - `conf/schema.py` → `DataConfig.to_lorenz96_config()`
  - `train.py` → `data.system` dispatch (`lorenz96`), `param_names`-generalized eval
  - configs: `config/experiment/L1_direct_unet_s0s1.yaml`, `L2_vanilla_cfm_s0s1.yaml`
  - test: `tests/test_lorenz96_training.py`

## L96 Neural Training (`feat/l96-neural-training`, from master @ 0687e07)

Implements all-5-param randomization + neural S0/S1 training (commit `3a1c8d5`).
See `L96_NEURAL_TRAINING_PROGRESS.md` for the per-WP tracker and handoff.

- **All 5 params randomized ±20%**: `models/lorenz96_dynamics.py` accepts `c1,h,hx,eps`
  + `F` kwargs; `data/lorenz96.py` `RandomParam`/`RandomBias` + `make_l96_s0_s1_trainval`.
- **Neural `param_dim=0` + `cond_extra_dim=0`** (DirectUNet, VanillaCFM-τ=0): obs-only
  input (no forcing/params conditioning; 24D in, 24D out).
- **DA parity**: `evaluation/run_l96.py` + `evaluate_all_l96.py` pass per-window all-5
  params to DA; S1 uses biased `*_da` params.
- Configs: `config/lorenz96_default.yaml`, `L1_direct_unet_s0s1.yaml`,
  `L2_vanilla_cfm_s0s1.yaml` (both `param_dim=0`; L2 is τ=0).
- Tests: `tests/test_lorenz96_training.py` (11 tests).
- sbatch: `run_one_epoch_tests_l96`, `run_l96_da_consistency`,
  `run_l96_neural_training`, `run_l96_evaluate_all`.

### Multi-agent review workflow (git/PR)

Code changes on this branch go through an implementer → reviewer → verifier loop.
Two execution paths:

- **Option A — GitHub PR**: `.github/workflows/ci.yml` runs ruff + pytest on PRs
  to `feat/l96-*`. Agents use `gh pr create` / `gh pr review` / `gh pr merge`.
  Blocked until `gh auth login` is run interactively (W3).
- **Option B — Local**: `scripts/agent_review_loop.sh <STEP> "<desc>" [--review]`
  provides the same loop with local git (works immediately).

**Run-to-completion policy:** the general rules (branch naming, drive-to-merge, reviewer
identity, CI gate) live in `AGENTS.md` under **Git / PR Workflow** and apply to every
session — follow those. This branch additionally uses `feat/l96-*` as its integration
namespaces subject to the repo ruleset.

**REMINDER:** run `gh auth login` and enable branch protection on `feat/l96-*`
(require 1 PR approval + status checks) to unlock the GitHub PR path.

**Status (2026-08-24)**: Q1–Q3 closed; the canonical L96 benchmark artifact is
`reports/l96/outputs/l96_consolidated_benchmark.md` (all-metric tables RMSE/EV/ES ×
{all/slow/fast} + consistency checks + Hovmöller reconstruction examples, regenerated by
`reports/l96/generate_l96_consolidated_report.py`). Headline: neural beats the best DA
baseline on both S0 and S1 (RMSE 0.62 vs 0.74), degradation ≈1.00 vs DA ≈1.9×.
Note: tables use the **pooled** RMSE convention for every method (the DA cache stores
mean-of-per-window RMSE). ES note (**DA performance pending update**): a normalization
bug in `_ESAccumulator` (accuracy term divided by N twice) deflated cached EnKF/ETKF ES
to spread-dominated values, and separately `Strong4DVar.assimilate_batch` never computed
ES in-run (historical Strong values came from a since-deleted offline backfill). Fixes
merged to `feat/l96-neural-eval-fix`: PR #67 (accumulator formula + esfix infra),
#68 (batch-path Strong ES wiring + relative gate tolerance), #70 (dim-compatible truth
for reduced-dynamics S1 methods; #69 was an accidental empty merge).
**All DA ES values in the canonical s0c cache have been updated** (bug-fixed
accumulator, swapped 2026-08-24); RMSE/EV rankings are unaffected (fresh runs agree
within ~1%) and neural numbers are unaffected. The consolidated report now shows
proper ensemble ES for DA EnKF/ETKF (N=30) and L3 (ens30×10, N=30); deterministic
methods (Strong-4DVar, L1b/L2b/L4/L5/L6) show N=1 MAE ES (marked `*`). The remaining
legacy caches (s0c int200 fw, fw6 int100/int200, legacy int100/int200, dws50) were
**not** swapped: the original esfix array (49383) **failed** (ran against the pre-fix
`rerun_l96_esfix.py` without missing-`es` handling / strict 5e-3 tolerance). Their
regeneration is deferred — the `_ESAccumulator` fix needed to produce correct textbook
ES is now on master (PR #74), so resubmitting `batch/run_l96_esfix.sbatch` against
master and swapping the passing caches is open follow-up (see
"Deferred future work (Phases B & C)" → Phase C-adjacent DA regen note).

**Open questions (L96)** — all answered (standalone DA-parity eval, cached test set):
- **Q1 (REVISED 2026-08-24, L3 ens30 study — see below)**: the original answer
  (multi-τ worse than τ=0, +8.6%) was an artifact of single-sample × 1-step evaluation.
  With N=30 members and proper integration, multi-τ CFM **beats** conditional-mean
  estimation: L3 0.5643 vs τ=0 L2b 0.6290 (−10.4%), also beating DirectUNet L4
  (0.6189) — the new overall best on S0. Decomposition of the published L3 0.688:
  −5.5% from 30-member averaging (sampling variance), −13.2% further from 10 Euler
  steps (integration coarseness). τ=0 control is exactly invariant to n_outer
  (single-Euler-step shortcut), confirming the effect is specific to multi-τ.
- **Q1-original (superseded)**: multi-τ CFM does NOT beat conditional-mean estimation —
  L3 0.688/0.690 vs τ=0 L2b 0.633/0.633 (+8.6%); mirrors the L63 G-series finding
  **at 1 member × 1 step only**; see the ens30 revision above before quoting this.
- **Q2 (answered, L4/L5)**: size sensitivity is model-dependent — small DirectUNet
  (L4) slightly beats default L1b (0.619 vs 0.622); small τ=0 CFM (L5) is worse than
  default L2b (+4.3%). CFM benefits from capacity; DirectUNet does not. (Note: Q2's
  "best overall" ranking is superseded for S0 by L3 ens30×10 = 0.5643.)
- **Q3 (answered, L6)**: corrupted-forcing conditioning is neutral-to-slightly-negative
  (L6 0.639/0.638 vs obs-only L2b 0.633/0.633); neural degradation was already ≈1.00,
  so there was no robustness gap for conditioning to close.

### L3 ensemble study (`ens30`, S0 + S1, job arrays 49350 / 49447)

N=30 members to match DA EnKF/ETKF `N_ensemble=30`; outputs in
`experiments/{L3,L2b}_vanilla_cfm_s0s1/ens30_no{1,10}/` (`estimates_s0.npz` member mean,
`members_s0.npz` (200,3000,24,30) f32, `neural_eval.json` with a `sampling` block).
ES note: this S0 study ran while the DA `_ESAccumulator` still had its normalization bug,
so the S0 JSONs store both conventions ("cache" = legacy buggy `mae/M − 0.5·pairwise`,
"textbook" = proper scoring rule); after the 2026-08-24 fix there is a single ES
(the textbook formula) everywhere — the table below keeps both columns as record.
The S1 ens30 study (`experiments/L3_vanilla_cfm_s0s1/ens30_s1_no{1,10}/`, job 49447,
2026-08-24) ran against the bug-fixed code, so its JSONs store the single textbook
ensemble ES (matched by the DA EnKF/ETKF N=30 caches).

| Model | members × steps | RMSE | EV | ESens(cache)* | ESens(textbook) | spread |
|---|---|---|---|---|---|---|
| L3 multi-τ | 30 × 1 | 0.6503 | 0.845 | −0.094 | 0.336 | 0.194 |
| L3 multi-τ | 30 × 10 | **0.5643** | 0.879 | −0.140 | 0.265 | 0.278 |
| L2b τ=0 | 30 × 1 | 0.6290 | 0.854 | −0.021 | 0.371 | 0.062 |
| L2b τ=0 | 30 × 10 | 0.6290 (≡ no1 bitwise) | 0.854 | −0.021 | 0.371 | 0.062 |

(*legacy buggy convention, kept for provenance only.)

Reference points (single-sample): L3 0.688, L2b 0.633, L4 DirectUNet 0.6189;
best DA Strong-4DVar 0.742 (S1 1.432). Multi-τ spread (~0.28 at 10 steps) is ~4.5×
the τ=0 spread — the τ-sampled velocity field yields genuinely diverse members whose
mean beats every deterministic scheme; whether that diversity helps probabilistic
scores (CRPS vs the ES conventions here) is open follow-up work. The S1 ens30 study
is complete (2026-08-24, see below); other models' ensemble runs remain open follow-up.

### 5-seed reproducibility (S0, job array 49419)

Five independent 30-member ensembles (seeds 1–5) confirm the multi-τ advantage is
not a seed artifact. Report: `reports/l96/outputs/ens30_seed_report.md`.

| scheme | seeds 1-5 mean±std | seed0 (orig) | range (6 runs) |
|---|---|---|---|
| 1-step (n_outer=1) | 0.6502 ± 0.0002 | 0.6503 | [0.6500, 0.6506] |
| 10-step (n_outer=10) | 0.5642 ± 0.0005 | 0.5643 | [0.5637, 0.5650] |

10-step/1-step ratio = 0.868 (−13.2%), cross-seed std < 0.001 for both schemes.
At inference τ is a deterministic schedule (k/N_outer), not random; all member
diversity comes from fresh x₀ noise. The improvement comes from proper ODE
integration of the multi-τ-trained field across τ∈(0,1], not from τ=0 evaluations
(the 1-step result at 0.650 is worse than the τ=0-trained L2b control at 0.629).

### L3 ens30 on S1 (job array 49447, 2026-08-24)

S1 counterpart of the S0 ens30 study, run against the bug-fixed single-ES code
(`--cases s1 --n-members 30 --seed 0`, n_outer ∈ {1,10}). Outputs in
`experiments/L3_vanilla_cfm_s0s1/ens30_s1_no{1,10}/` (`members_s1.npz` (200,3000,24,30) f32,
`estimates_s1.npz`, `neural_eval.json` with a single textbook `ensemble.es`).

| Model | members × steps | RMSE | EV | ES (ens, N=30) | spread |
|---|---|---|---|---|---|
| L3 multi-τ | 30 × 1 | 0.6528 | 0.843 | 0.338 | 0.194 |
| L3 multi-τ | 30 × 10 | **0.5667** | 0.877 | 0.267 | 0.278 |

The 10-step/1-step ratio is 0.868 (−13.2%) — the integration-coarseness effect is
statistically identical to S0. S1/S0 degradation at 30×10 is ≈ **1.004** (S1 0.5667 vs
S0 0.5643): the multi-τ ensemble is essentially as good on S1 as on S0, consistent
with the neural models' known robustness to the parameter-biased S1 test setup.

## Deferred future work (Phases B & C)

These were proposed alongside the S1 ens30 study (Phase A, done 2026-08-24) but are
**not committed for execution** — recorded so a future session can pick them up.

### Phase B — CFM architecture variants (low priority; **requires a design doc first**)

Investigate whether a Tweedie-style two-stage decomposition or a diffusion-style
variant improves on VanillaCFM for L96. **Design doc drafted 2026-08-25**
(`docs/phase_B_l96_cfm_variants.md`) — defines V1 (L96 TweedieSolver port,
obs-only, `use_energy=false`, 2-stage) and V2 (CFM + Tweedie residual hybrid)
precisely with reference bars vs L2b/L3/L4 and the open design questions to
resolve (cond_extra_dim plumbing, energy flag, stage-1 budget, 24D cell
validity, multi-member sampling, single-sample vs ens30×10 bar). **V3 diffusion
  deferred** (no scaffolding exists). **No Phase B code implements these yet** — a
  future session must resolve the doc's open questions before implementing.

**V2/V3 worktree state (checked 2026-08-28) — for the Phase B session:**
The V2/V3 topic worktree (`4dvarnet-fm-cfm-v2v3`, branch `feature/l96-v2v3-pure`)
is separate from master and **not yet trainable**:
- Its `train.py` still has **unresolved git merge-conflict markers** (B1 blocker;
  `py_compile` fails with `SyntaxError: unmatched ')'`) from commit `f7749a9`,
  so no training/eval can run from that branch as-is.
- It does **not** carry the vectorized data-gen fix merged to master as PR #105
  (`data/lorenz96.py` there has no `fast_generation` path). When the Phase B
  session resumes it should re-sync master first (which includes both #105 and the
  joint-DA work).
- The **parallel-merge CHANGELOG conflict** on master (joint-DA 2026-08-28 entry vs
  the V2/V3-sourced #105 2026-08-27 data-gen entry) is already **resolved cleanly**
  on `origin/master` — no markers remain. The V2/V3 tree did NOT contribute the
  fix to master (it merged independently); only its own branch needs the sync.


### Phase C — L96 joint state-parameter neural estimation (infra done 2026-08-25; training + eval done 2026-08-26)

Extend the existing **L63 joint infrastructure** to L96. Currently only L96 **Joint DA
baselines** exist; the L96 joint **neural** models were missing.
- **Existing (real) pieces:** `JointCFM` (`models/vanilla_cfm.py:75`, L63-shaped,
  `output_dim = state_dim + param_dim`); `JointCFMConfig` (`conf/schema.py:162`);
  L63 configs `H1_joint_cfm_default.yaml`, `H2_joint_cfm_tau0.yaml`, `S5/S6`; L96
  Joint DA baselines `JointEnKFL96`/`JointETKFL96`/`JointStrong4DVarL96`
  (`evaluation/baselines.py`) + `eval_joint_comparison_l96.py` (ready DA comparator).
- **Done (2026-08-25):** design doc `docs/phase_C_l96_joint_neural.md`; L96 joint
  neural models `JointCFM` (port) + `JointDirectUNet` (new); 3 configs L7/L8/L9;
  `data/lorenz96.py` flattens `fast_weights` to per-index `w1..w4`/`true_w1..`/`_da`
  scalar keys; `train.py`/`lightning_module.py` dispatch; `eval_joint_neural_l96.py`
  (extended `evaluation/neural_inference.py` for joint types); 8 joint-neural tests
  + WP1 dataset-key tests; 2 sbatch (training 3-task array, eval). `param_dim=8`,
  h fixed. Training completed; standalone eval + ens30 completed (2026-08-26, below).
- **Eval bugs fixed (2026-08-26):** PR #81 (`state_dim` inference, `obs_var_indices`,
  member-param collection), #83 (ens30 sbatch IFS, ensemble evaluator, `N_outer`
  passthrough), #85 (`collate_joint_eval` legacy-`fast_weights`-list support for the
  pre-flattening cached dataset), #89 (ens30 `params_pred` member-mean shape fix),
  #90 (report-generator table column-order/separator/best-marking fixes).
- **Results (2026-08-26, cached S0/S1 test set, Obs30, 200 windows):** see the L96
  joint neural benchmark at `reports/l96/outputs/l96_joint_neural_benchmark.md`.
  Single-sample: L7 0.606/0.662, L8 0.610/0.661, L9 0.626/0.631 (S0/S1 state RMSE).
  L9 ens30×10 is the best joint estimator (S0 **0.525**/S1 **0.531**), matching L3's
  multi-τ integration advantage. **L9 recovers the 8 params (paramRMSE 0.058) while L7
  τ=0 fails (1.21) despite matching state RMSE**; L8 deterministic recovers them well
  (0.061). **Joint ETKF DA baseline run (2026-08-28, Job 50577)** — see
  `l96_joint_da_benchmark.md`: S0 Joint-ETKF 0.633/param 0.053 (≈ L9 parity); S1 1.497/
  param 0.128 after the inflation/stability fix (see "L96 joint state-parameter DA
  baseline" under Experiments). **Joint-EnKF added 2026-08-28 (job 50655)** — the
  state-only-inflation fix was ported to `JointEnKFL96` (RC: it inflated the whole
  augmented state like the old ETKF, risking the same S1 divergence) and a joint
  `assimilate_batch` was added (the inherited parent batch silently dropped params);
  S0 0.726/param 0.057, S1 **1.459**/param 0.148 (S1 is the best DA row, stable after
  the fix). **Joint-Strong-4DVar run (2026-08-31, Job 51000)** — batched pure-gradient
  Adam solve (no LBFGS) over all 200 windows: S0 0.712/param 0.226, S1 **1.200**/param
  0.299. Joint-Strong-4DVar is now the **best DA row on S1** (beats Joint-EnKF 1.459 /
  Joint-ETKF 1.497 / vanilla Strong-4DVar 1.432) and close to Joint-ETKF on S0; its
  param recovery (esp. F 0.85) is weaker than the filters, but its S1 state skill is
  the joint-DA best.

### Phase C-adjacent (blocked, unblocks DA-parity ES): L96 DA cache ES regeneration

The original esfix array (job 49383 + resubmissions) failed before regenerating the
non-canonical baseline caches. A 2026-08-24 resubmission (job 49488, `--array=1-7`,
against fixed master) **still cannot complete**: it *resumed from the stale partial
`*_esfix*` caches* written by 49383 rather than doing a clean regeneration, and the
validation gate then **failed on 6 of 7 caches** (RMSE mismatch vs originals: e.g. dws50
EnKF S0 1.009→1.102, Strong-4DVar and S1 EnKF/ETKF drifting). Only the **legacy int100**
cache passes cleanly and was swapped (`.bak` + promoted esfix). Rather than force-swap
gate-failing caches (would corrupt RMSE consistency), the rest stay stale.

**Important scoping note:** the consolidated report's DA candidates
(`DA_JSON_CANDIDATES` in `reports/l96/generate_l96_consolidated_report.py`) point at the
**canonical s0c int100 fw** cache first, which was already swapped bug-fixed and is
correct (ETKF/EnKF proper textbook ES) — so **the report is already correct on the DA
side** and none of the 6 non-report caches affect it. The 6 remaining caches (s0c int200
fw, fw6 int100/int200, legacy int200, dws50, dws50 fw) serve other lineages and are
**not** referenced by the consolidated report. To finish them correctly, delete the stale
`*_esfix*` files for those caches and do a clean full regeneration (hours GPU each) —
outcome uncertain given the CHANGELOG's note that some L96 caches "are not reproducible
under current code semantics."

### Decoupled cascade (C1/C2) — documented negative (2026-09-01)

A **decoupled state→param cascade** (`model_type=param_head`, `models/param_head.py`) was
tested as an alternative to the coupled joint flow for recovering the 8 L96 params: a
`StateParamHead` reads obs + biased `*_da` params + forcing + a state estimate, trained
under two state sources — **C1** = frozen L1b state-only DirectUNet estimate, **C2** =
exact true state (ablation). **Both fail the fast weights `w1/w2` on S1 (NRMSE ≈ 1.1-1.2,**
i.e. relative error > 100%), even with the exact true state (C2) — an
**information/architecture bottleneck**, not a state-quality issue (F is partly state-quality:
true state halves it 1.67→0.86). Only the coupled **multi-τ flow (L9)** recovers all 8 params
(S1 per-param NRMSE all ≤ 0.20, F 0.07) at parity with the joint DA filters on the params they
actually estimate. **Recorded as a documented negative, not a benchmark win.** Details + NRMSE
table: CHANGELOG 2026-09-01 and `reports/l96/outputs/l96_joint_neural_benchmark.md` (which now
also carries computed DA NRMSE rows with a `w3/w4`=pinned-prior masking footnote).



All E/F/G/S rows are **Lorenz-63** (`cs1+cs2` mixes); results live under `experiments/`.
The **L-series** (Lorenz-96) is listed separately below.

### Lorenz-63

| ID | Model | Hidden | Epochs | Train mix | Status |
|---|---|---|---|---|---|
| E1_direct_unet_default | DirectUNet | [64,128,256] | 200 | cs1+cs2 | done |
| E2_direct_unet_small | DirectUNet | [32,64,128] | 200 | cs1+cs2 | done |
| E3_direct_unet_rand | DirectUNet | [32,64,128] | 200 | cs1_rand+cs2_rand | done |
| F1_vanilla_cfm_default | VanillaCFM | [64,128,256] | 400 | cs1+cs2 | done |
| F2_vanilla_cfm_small | VanillaCFM | [32,64,128] | 400 | cs1+cs2 | done |
| F3_vanilla_cfm_rand | VanillaCFM | [32,64,128] | 400 | cs1_rand+cs2_rand | done |
| G1_vanilla_cfm_t0_default | VanillaCFM (τ=0) | [64,128,256] | 400 | cs1+cs2 | done |
| G2_vanilla_cfm_t0_small | VanillaCFM (τ=0) | [32,64,128] | 400 | cs1+cs2 | done |
| G3_vanilla_cfm_t0_rand | VanillaCFM (τ=0) | [32,64,128] | 400 | cs1_rand+cs2_rand | done |
| S1–S10 (incl. τ=0 + joint-CFM variants) | various | various | — | s0_s1 | done |

### Lorenz-96 (DA-parity: all-5 params ±20%, obs_j=2 → 24D, Obs30)

| ID | Model | Hidden | Epochs | τ mode | Status |
|---|---|---|---|---|---|
| L1b_direct_unet_s0s1 | DirectUNet | [64,128,256] | 200 | n/a | done (beats DA on S0+S1) |
| L2b_vanilla_cfm_s0s1 | VanillaCFM | [64,128,256] | 400 | τ=0 | done (≈ L1b) |
| L3_vanilla_cfm_s0s1 | VanillaCFM | [64,128,256] | 400 | multi-τ | done (ens30×10 best on S0 0.564 + S1 0.567, job 49447) |
| L4_direct_unet_s0s1_small | DirectUNet | [32,64,128] | 200 | n/a | done (Q2: best overall, 0.619/0.621) |
| L5_vanilla_cfm_s0s1_small_tau0 | VanillaCFM | [32,64,128] | 400 | τ=0 | done (Q2: small hurts CFM, +4.3%) |
| L6_vanilla_cfm_s0s1_forcing_cond | VanillaCFM | [64,128,256] | 400 | τ=0 + forcing cond | done (Q3: neutral vs obs-only) |
| L7_joint_cfm_s0s1 | JointCFM | [64,128,256] | 400 | τ=0, joint 8-param | done (state 0.606/0.662; paramRMSE 1.21 — τ=0 fails params) |
| L8_joint_direct_unet_s0s1 | JointDirectUNet | [64,128,256] | 200 | joint 8-param | done (state 0.610/0.661; paramRMSE 0.061) |
| L9_joint_cfm_s0s1_multitau | JointCFM | [64,128,256] | 400 | multi-τ, joint 8-param | done (ens30×10 best joint: state 0.525/0.531; paramRMSE 0.058) |

**Standalone DA-parity results (cached test set, Obs30, 200 windows)** — S0/S1 RMSE
(single-sample convention; L3's S0 ranking is superseded by the ens30 study above):
L4 **0.619**/0.621 < L1b 0.622/0.625 < L2b 0.633/0.633 ≈ L6 0.639/0.638 < L5 0.660/0.660 < L3 0.688/0.690.
All neural degradation ≈ 1.00; best DA (Strong-4DVar): 0.742/1.432.

**L96 joint state-parameter estimation (Phase C, 2026-08-26)** — cached S0/S1, Obs30,
200 windows; 24D state + 8 params (F, c1, hx, eps, w1..w4). Single-sample S0/S1 state
RMSE: L7 0.606/0.662, L8 0.610/0.661, L9 0.626/0.631. Best joint = **L9 multi-τ at
ens30×10: S0 0.525 / S1 0.531** (matches L3's multi-τ integration advantage). Param
recovery is model-dependent: L9 recovers the 8 params (paramRMSE 0.058) as does L8
JointDirectUNet (0.061), but **L7 τ=0 fails to recover params (1.21)** despite matching
state RMSE. Full tables: `reports/l96/outputs/l96_joint_neural_benchmark.md`.

**L96 joint state-parameter DA baseline (2026-08-28, ETKF + EnKF)** — see
`reports/l96/outputs/l96_joint_da_benchmark.md`. Joint-ETKF vs vanilla ETKF, and
Joint-EnKF vs vanilla EnKF, on the cached S0/S1 set (200 windows). **S0** Joint-ETKF
state RMSE **0.633** (EV 0.82, ES 0.30) vs vanilla 0.878; paramRMSE mean **0.053** —
**at parity with the L9 neural model** (state 0.626 / param 0.059 single-sample).
Joint-EnKF S0 0.726 (EV 0.77, ES 0.37) beats vanilla EnKF 0.891 but is worse than
Joint-ETKF. **S1** Joint-EnKF **1.459** (EV 0.23, ES 0.84) is the best DA row, ahead of
Joint-ETKF 1.497 & vanilla EnKF 1.505 / ETKF 1.554 — both joint filters stabilized by
the state-only-inflation fix (RC: the filters were inflating the unobserved param
block, growing spread into the reduced J=2 forecast). Neural (L9) still clearly ahead
on S1 (0.631) via its ≈1.00 bias robustness. **Joint-Strong-4DVar run 2026-08-31**
(job 51000, batched Adam over 200 windows): S0 state RMSE **0.712** / S1 **1.200** —
beats vanilla Strong-4DVar (0.750/1.432) on both cases and is the **best DA row on S1**
(ahead of Joint-EnKF 1.459 / Joint-ETKF 1.497); param RMSE mean 0.226 (S0) / 0.299
(S1), weaker than the filters (F 0.85 dominates).

**Multi-method reconstruction artifacts (merged 2026-09-01, PR #134)** —
`eval_joint_comparison_l96.py` persists per-window reconstruction `.npz` arrays
(`trajectories`, per-member `ensemble_variance`, `params`, `es`) for every benchmarked
method, merged into `experiments/l96_joint_baselines_trajectories.npz`. Re-ran the 3
joint methods (jobs 51098/51131) + vanilla Strong-4DVar (51294, so it appears in the
comparator schema) on the canonical cached S0/S1 test set. `experiments/l96_joint_comparison.json`
is now the full **6-method** comparison (vanilla ETKF/EnKF/Strong-4DVar + Joint-ETKF/EnKF/
Strong-4DVar; fresh re-run values: Joint-ETKF S0 0.6348, Joint-EnKF 0.7244, Joint-Strong-4DVar
0.7054/1.1999). Both joint reports regenerate against it, so the neural report's DA-baselines
table now lists all 6 DA methods (incl. Joint-Strong-4DVar). Note the master committed DA
report retains oracle-free neural values; a stale local regeneration against oracle-era JSONs
on the master worktree was discarded in favor of the merged oracle-free content.

### S1 DA corrupted-forcing fix (2026-09-02) + slow-only obsj0 DA baselines

**S1 forcing bug fixed:** `cfg_s1` in both DA eval paths (`run_and_cache_baselines` and
`eval_joint_comparison_l96.py`) was built without `case=2`, so `Lorenz96Config.use_corrupted_forcing`
returned False and `evaluate_baseline` fed the DA the **true** forcing (`forcing_true`) on S1 —
the `forcing_state_bias=0.1` corruption designed into S1 was silently never applied to the DA
(the cached S1 windows *do* hold a genuine `forcing_corrupted`; it was just never read). Fixed by
setting `case=2` in `cfg_s1` (S0 unchanged). Re-ran the canonical obsj2 S1 DA (state-only + joint)
with the fix; **S1 changes only mildly** (filters <1%: Joint-ETKF 1.4976→1.5125, EnKF 1.5044→1.5123,
Strong-4DVar 1.4319→1.4369) — the DA is robust to the forcing corruption. **S0 reproduced within
noise** (S0 gate <2%), confirming the fix did not disturb S0. Canonical caches swapped (`.bak`
backups: `l96_baselines_dws500_s0c_*_obsj2_int100_fw.json*` + `l96_joint_comparison.json`); the
consolidated / joint-DA / joint-neural reports regenerated with the corrected S1 rows.

**New slow-only obsj0 configuration (2026-09-02):** decoupled the observation count from the S1
reduced-dynamics J and the eval metric group. Now `run_and_cache_baselines` / the comparator take
`obs_j` (fast vars observed; 0 = slow-only), `s1_j` (S1 reduced dynamics J, kept=2 for the obsj0
study), and `eval_j` (eval metric group, kept=2 → the same 24D slow+first-2-fast group) — all three
independent. This lets a **slow-only** observation (only the 8 slow X) be scored on the identical
24D eval subspace as the canonical obsj2 config, apples-to-apples. Ran state-only + joint DA
baselines (obsj0, Obs30, 200 windows): S0 EnKF 1.27 / ETKF 1.25 / Strong-4DVar 1.44 / Joint-ETKF
1.19; S1 EnKF 1.70 / ETKF 1.71 / Strong-4DVar 1.62 / Joint-ETKF 1.60. vs obsj2 the degradation is
dominated by the **unobserved** obs_fast group (S0 obs_fast ≈ 1.6–1.95 vs 0.88–1.10 obsj2) while the
**slow subgroup is preserved** (S0 slow ≈ 0.41–0.46). Joint-DA param recovery: S1 Joint-ETKF
0.130→0.158 (hx/F degrade), S0 slightly improves (0.045 vs 0.054, F-driven). Full tables:
`reports/l96/outputs/l96_obs_density_da_baselines.md`.

## Phases

### Phase 0: Plan
- [x] Initial PLAN.md created
- [x] CS3/CS4 experiment plan in `docs/case_studies.tex`
- [x] Exp G (τ=0) experiment plan in `docs/experiment_G_tau0_cfm.md`

### Phase 1: Implementation (complete)
- [x] `models/direct_unet.py` — DirectUNet nn.Module
- [x] `models/vanilla_cfm.py` — VanillaCFM nn.Module with CFM loss + sampling
- [x] `data/random_param_dataset.py` — Randomized Lorenz-63 parameters per window
- [x] `conf/schema.py` — `DirectUNetConfig`, `VanillaCFMConfig`, `DataConfig` with CS3/CS4 fields
- [x] `training/lightning_module.py` — `LitModel` dispatches all 3 model types
- [x] `training/pipeline.py` — `create_trainer`, `train_stage`, `run_2stage_pipeline`
- [x] `train.py` — `model_factory`, `evaluate_model`, CS3/CS4 evaluation
- [x] 6 experiment YAML configs (E1-E3, F1-F3)
- [x] CS3/CS4 test cases in data generation, evaluation, and report
- [x] `evaluate_all.py` — Unified baseline + CFM comparison script
- [x] `reports/generate_unet_cfm_report.py` — CS3/CS4 report

### Phase 2: sbatch Infrastructure (this session)
- [x] `batch/run_lint.sbatch` — ruff + mypy in batch
- [x] `batch/run_test_suite.sbatch` — pytest (fast) in batch
- [x] `batch/run_config_validation.sbatch` — Hydra config + model factory validation
- [x] Deprecated duplicate `run_vanilla_experiments.sbatch` and interactive `run_tests.sh`

### Phase 3: τ=0 CFM Ablation (Exp G) — complete (2026-07-01)
- [x] `conf/schema.py` — `train_tau_0_only: bool = False` on `VanillaCFMConfig`
- [x] `models/vanilla_cfm.py` — τ=0 logic in `compute_cfm_loss` and `sample`
- [x] `train.py` — `train_tau_0_only` wired through `model_factory`
- [x] 3 config YAMLs: G1_vanilla_cfm_t0_default, G2_vanilla_cfm_t0_small, G3_vanilla_cfm_t0_rand
- [x] `batch/run_one_epoch_tests.sbatch` + `batch/run_new_experiments.sbatch` updated with G1-G3
- [x] Tests for τ=0 mode

### Phase 4: Verify (all via sbatch) — complete (2026-07-01)
- [x] `sbatch batch/run_config_validation.sbatch` — all configs load
- [x] `sbatch batch/run_lint.sbatch` — ruff + mypy pass
- [x] `sbatch batch/run_test_suite.sbatch` — fast tests pass
- [x] `sbatch batch/run_one_epoch_tests.sbatch` — GPU smoke test (E1-F3 + G1-G3, 1 epoch)

### Phase 5: Launch — complete (2026-07)
- [x] `sbatch batch/run_new_experiments.sbatch` — full E1-F3 + G1-G3
- [x] Results collected under `experiments/` (see Experiments tables above)
- [x] CHANGELOG.md entries per change

## Interfaces

### Model forward signatures (for LightningModule dispatch):
```
TweedieSolver:
  training_step(stage=1): model.estimate_mean(obs) → (B,T,D)
  training_step(stage=2): model(obs) → (B,T,D)
  config_optim(stage=1): model.mean_estimator.parameters()
  config_optim(stage=2): model.non_gaussian.parameters()

DirectUNet:
  training_step: model(obs) → (B,T,D)
  loss: StateMSELoss(pred, batch.states)
  config_optim: model.parameters()

VanillaCFM:
  training_step: compute_cfm_loss(batch) → scalar
  config_optim: model.parameters()
  sampling: model.sample(obs, N_outer) → (B,T,D)
```

### Dataset output format:
```python
{
    "true_state": Tensor(T, 3),
    "obs": Tensor(T, 3),
    "obs_mask": Tensor(T,),
    "forcing_true": Tensor(T,),
    "forcing_corrupted": Tensor(T,),
}
```

### results.json format (per experiment):
```json
{
  "experiment_id": "...",
  "config": {...},
  "epochs_trained": ...,
  "total_time_seconds": ...,
  "train_time_seconds": ...,
  "eval_time_seconds": ...,
  "fm_cs1": {"X": {"mean": ..., "std": ...}, "Y": ..., "Z": ..., "mean": ...},
  "fm_cs2": {...},
  "fm_cs3": {...},
  "fm_cs4": {...},
  "fm_degradation": ...,
  "fm_degradation_cs3cs4": ...
}
```
