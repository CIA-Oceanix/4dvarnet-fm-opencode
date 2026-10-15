# L96 FDV2 × score-based flow matching: a calibrated posterior at least as accurate as the FDV2→SDA hybrid

**Status:** CLOSED (2026-10-14) after Phases 0–1; Phase 2 (M2, M3) not run. Outcome in `docs/results/l96_fdv2_fm_posterior.md`: the training-free M1bI (perturbed-obs FDV2 starts + DPS-guided stochastic FM sampler + validation-tuned inflation) beats the hybrid and B5 on CRPS by 4–7% and is calibrated in every test cell, but misses S-acc RMSE on the regular layout by +0.6–0.8%. An S1 test of the frozen configurations (2026-10-15, results note section 4) keeps the CRPS lead and SSR ≈ 1, but fails reliability on the random layout. Plan v1 (2026-10-12) Supersedes `docs/plans/analysis/l96_fdv2_score_prior.md` (whose evidence, sections 1.1–1.5, it reuses) and fixes the issues found in that plan's review (section 7 below).

## 0. Goal and success criteria

**Goal:** one method combining the FDV2 unrolled solver with a score-based flow-matching (FM) prior that, on the L96 benchmark test sets, matches or beats the FDV2→SDA hybrid in both RMSE and CRPS **with a single configuration**, and is calibrated where the hybrid is not.

### 0.1 Reference: the hybrid (2026-10-12 probe, `sda_warm.json`)

Setup: FDV2 coupled deep-supervision fine-tune (seed 2) as the mean, warm start of the SDA1-M prior (`B4_sda1_monaiM_ep1200_l96_seed1`), gw = 2, 10 steps, 30 members. S0, 200 test windows, physical units.

| τ0 | regular RMSE / CRPS / spread/RMSE | random RMSE / CRPS / spread/RMSE |
|---|---|---|
| 0.1 | 0.2411 / **0.1166** / 0.57 | 0.2810 / **0.1347** / 0.49 |
| 0.3 | 0.2310 / 0.1202 / 0.34 | 0.2737 / 0.1423 / 0.28 |
| 0.5 | 0.2268 / 0.1237 / 0.24 | 0.2694 / 0.1463 / 0.20 |
| 0.7 | **0.2262** / 0.1297 / 0.15 | **0.2685** / 0.1522 / 0.13 |
| perturbed-obs FDV2 ensemble | ≈ 0.24 / 0.1147 / 0.57 | – / 0.1333 / – |

The hybrid trades RMSE against CRPS through τ0, and it is under-dispersed at every τ0. These numbers were selected on the test windows, from one seed, on S0 only. Phase 0 re-establishes them under the protocol in section 0.3.

### 0.2 Success criteria (held-out test, regular and random layouts, S0)

- **S-acc:** ensemble-mean RMSE ≤ the hybrid's best RMSE over τ0, and CRPS ≤ the hybrid's best CRPS over τ0, from the *same* configuration. "≤" means the paired window-bootstrap 95% CI of the difference lies at or below 0, or includes 0 with a point estimate ≤ +0.5%.
- **S-cal:** spread-skill ratio SSR ∈ [0.85, 1.15] **and** a rank-histogram reliability index at most half the hybrid's (best-CRPS τ0) **and** 90% central-interval coverage in [0.85, 0.95].
- **S-robust:** the ordering against the hybrid holds on S1 (reported, not gated) and for both FDV2 seeds (seed 1 = `..._deepsup_ft_coupled_s1philr_seed1`).
- **Calibration must not be free:** the method must also beat **B5**, the hybrid with post-hoc spread inflation about the ensemble mean, tuned on validation to SSR = 1 (section 3). If B5 already meets S-cal at equal CRPS, the scientific claim shifts from calibration to the posterior's *structure* (rank histograms per variable, spatial correlation of members, energy score), and we decide with the user before continuing.

### 0.3 Evaluation protocol (applies to every row, baselines included)

- Tune on the **100 validation windows** (`num_val_windows: 100`); score on the 200 test windows; never select on test.
- 30 members; the same observation draws and member seeds across methods (common random numbers).
- Units: scores in physical units (as the probe). Inside samplers, **R in normalised units**: R_norm = R_var / σ_clim² (≈ 0.14 for R_var = 0.5). The current SDA sampler's normalised-step guidance hides this mismatch; DPS/ΠGDM do not.
- Metrics:
  - ensemble-mean RMSE;
  - CRPS (fair estimator);
  - SSR = √((M+1)/M) · RMS spread / RMSE (note: the probe's "spread" is the mean member std, not RMS spread; the harness uses RMS);
  - rank histogram + reliability index;
  - 50/90% coverage;
  - energy score;
  - FMS_τ, optional (`docs/results/l96_p1_fm_score.md`).
- Uncertainty: paired window bootstrap (2000 resamples), as in `docs/plans/analysis/l96_p1_fm_score_bootstrap.md`.
- Member arrays go to node-local `/tmp`; only per-window scores are kept on `/Odyssey`.

## 1. Mathematics: one FM prior, three uses

Linear interpolant x_τ = (1−τ)x₀ + τx₁, with x₀ ~ N(0, σ_p²I) and σ_p = 0.5 (SDA1-M). A trained FM prior, parameterised either by velocity v or by state prediction Ψ, gives:

- **Denoiser (Tweedie):** Ψ(x_τ, τ) = E[x₁ | x_τ] = x_τ + (1−τ)v.
  - Since x_τ/τ = x₁ + σ(τ)ε with σ(τ) = σ_p(1−τ)/τ, the prior is a denoiser at every noise level: Φ(x̃, σ) = Ψ(τ x̃, τ) with τ = σ_p/(σ_p + σ).
  - **SDA1-M is therefore already a DSM-equivalent prior.** No new training is needed to test the idea.
- **Score:** ∇log p_τ(x_τ) = (τΨ − x_τ) / ((1−τ)²σ_p²).
- **Posterior state prediction:**
  - Exactly, Ψ_y = E[x₁ | x_τ, y] = Ψ + ((1−τ)²σ_p²/τ) ∇_{x_τ} log p(y | x_τ).
  - With the ΠGDM approximation, p(y | x_τ) ≈ N(y; HΨ, R + r_τ² H J_Ψ Hᵀ), with r_τ² = σ(τ)²/(1+σ(τ)²) (σ_d = 1).
  - Posterior velocity: v_y = (Ψ_y − x_τ)/(1−τ).
  - DPS is the r_τ = 0 special case. The current SDA sampler replaces the likelihood term by a fixed-norm step gw·g/‖g‖.
- **Stochastic sampler with the same marginals:** dx = [v_y + ½ g(τ)² ∇log p_τ(x_τ | y)] dτ + g(τ) dW.
  - g ≡ 0 gives the current (deterministic) hybrid sampler.
  - Use g(τ)² = 2γ σ_p²(1−τ)/τ, which vanishes at τ = 1; one knob γ, plus optional Langevin corrector steps at fixed τ.
- **Why the hybrid is under-dispersed:** it integrates a deterministic ODE from x_{τ0} = (1−τ0)x₀ + τ0 x̂. Its only randomness is x₀, which a contracting guided ODE then shrinks. The stochastic term (γ > 0) re-injects variance at every step at the rate the prior allows. That is the mechanism this plan bets on.
- **The FDV2 prior channel as an FM quantity:** x − Φ(x, σ) = −σ²∇log p_σ(x). Feed it to the solver at a noise level σ_k **matched to the iterate error at iteration k**, not at a fixed σ₀ below the observation noise: FDV2 iterates are 0.23–1 away from the manifold (normalised), where Φ(·, 0.1) is out of distribution. Normalise the channel to O(1): feed (x − Φ(x, σ_k))/σ_k = −σ_k·score.

## 2. Candidate methods (cheapest first)

| id | combination | training | what it tests |
|---|---|---|---|
| **M1** | FDV2 mean → SDA1-M prior, **stochastic** posterior sampler (γ > 0, ΠGDM, correct R_norm), warm start at τ0 | none | is the hybrid's miscalibration a sampler artefact? |
| **M1b** | M1 started from perturbed-obs FDV2 members (y + R^½ε per member) instead of one mean | none | randomised-maximum-likelihood-style starts + prior noise |
| **M1c** | M1 with a per-window warm-start level τ0_w, from the truth-free indicator J_prior(x̂_w) (ρ = 0.63–0.80 with error, `docs/plans/analysis/l96_fdv2_score_prior.md` 1.3) | none | conditional spread; improves the rank histogram rather than the SSR alone |
| **M2** | "Score-prior FDV2": FDV2 solver with the FM-prior channel (x − Φ(x, σ_k))/σ_k, σ_k following the measured iterate-error curve; Φ = SDA1-M frozen; solver-only fine-tune from the deep-sup checkpoint; then M1 sampling with the same prior | solver fine-tune, ~13 h × 2 seeds | does one shared prior help the point estimate too (one model, coherent mean + samples)? |
| **M2'** | M2 with an FDV2-native FM prior (state prediction, trained on the training states with the benchmark recipe; memorisation check) instead of SDA1-M | prior ~1200 epochs + solver fine-tune | only if M2 is limited by SDA1-M's prior quality |
| **M3** | "FDV2-CFM": the FDV2 solver *is* the conditional state predictor Ψ_θ(x_τ, y, τ). K inner iterations from x_τ/τ, flow time as the noise level of the prior channel (σ(τ)) and as an embedding; initialised from FDV2 deep-sup, trained with the PredictStateCFM loss; early-fine grid, stochastic sampler | ~1–2 days × 2 seeds | the conditional-flow route (calibrated by construction, SSR ≈ 1 for PSC-M), now with FDV2's accuracy; overlaps P2 M4a (`docs/plans/paper/l96_p2_unrolled_solvers_and_flows.md`) |

**Expected outcome:** M1/M1b reach S-cal and approach S-acc. M2 recovers the RMSE gap: the hybrid's best RMSE comes from refinement at τ0 = 0.7, which the shared prior can supply inside the solver. M3 is the stretch goal and runs only if G1 shows that warm-start methods cannot reach S-acc and S-cal together.

## 3. Baselines (all re-scored under 0.3)

- **B1:** FDV2 point estimate (deep-sup, 10 / 15 iterations).
- **B2:** perturbed-obs FDV2 ensemble.
- **B3:** hybrid, τ0 ∈ {0.1, 0.3, 0.5, 0.7}, gw ∈ {1, 2, 5}, selected on validation, once for RMSE and once for CRPS.
- **B4:** SDA1-M from noise (gw = 20).
- **B5:** B3 (CRPS-selected) + post-hoc inflation to SSR = 1.
- **B6:** PredictStateCFM-M benchmark flow (ens30_no20), for scale only.
- **B7:** ETKS (the DA reference row) from the existing benchmark outputs, for context.

## 4. Phases, gates and agents

Execution is split into independent tracks, each run by one agent in its own git worktree (`isolation: "worktree"`). Every track branches from master and opens its own master-based PR; PRs are not stacked, because stacked PRs get no CI. At most ~8 agents in total.

**GPU budget:** at most half the Odyssey_GPU nodes across all tracks. Track A uses ≤ 2 GPUs (evaluation sweeps); Tracks B and C use ≤ 2 training jobs each. Use array throttles.

### Phase 0: harness and baselines (1 agent, ~0.5 day)

- **Agent P0 "harness":**
  - Code:
    - `evaluation/posterior_scores.py`: SSR, rank histogram + reliability index, coverage, fair CRPS, energy score; per-window outputs plus paired bootstrap; reuses `evaluation/metrics.py`.
    - `eval_fdv_fm_posterior_l96.py`: method registry (B1–B7, M*), validation tuning, test scoring, JSON per row.
  - Tests:
    - SSR ≈ 1 for an ensemble drawn from the true Gaussian posterior;
    - the rank histogram is flat for an exchangeable ensemble;
    - the CRPS matches `evaluation/metrics.crps`;
    - B3 reproduces `sda_warm.json` within Monte Carlo noise at fixed seeds (regression test on 4 windows).
  - Runs B1–B5 (both FDV2 seeds, S0/S1, regular/random).
- **Gate G0:** the harness reproduces the probe numbers. The val-selected B3/B5 rows become the targets of 0.2. We then check whether B5 already meets S-cal; if it does, we stop and ask the user (0.2, last bullet).

### Phase 1: training-free tracks, in parallel (2 agents, ~1–2 days)

- **Agent A1 "sampler":**
  - `evaluation/fm_posterior_sampler.py`: posterior-velocity guidance (DPS / ΠGDM via a Hutchinson-diagonal or a few CG steps on the observed entries), the stochastic term γ, corrector steps, warm start from a mean or from per-member starts, and the per-window τ0.
  - Toy tests: on a linear-Gaussian prior with an exact Ψ, ΠGDM with γ > 0 must reproduce the analytic posterior mean and variance; γ = 0 with the normalised step must reproduce `sda_guided_sample` exactly.
- **Agent A2 "sweep":**
  - Validation sweep of M1 / M1b / M1c over γ ∈ {0, 0.25, 0.5, 1}, τ0, guidance type, corrector steps ∈ {0, 1, 2}, and N ∈ {10, 20}.
  - Two-stage search: coarse on 40 windows, refined on all 100.
  - Records cost (NFE, wall-clock).
- **Agent V "verifier":** adversarial review of A1/A2:
  - apples-to-apples configs against the B rows (the same mean checkpoint, members, windows, R);
  - no test-set selection;
  - correct units.
  - Blocks the PR on any mismatch.
- **Gate G1:**
  - M1* meets S-acc and S-cal → write up and **ask the user** whether M2/M3 are still wanted for the "one model" story.
  - Only S-cal met → Phase 2 (M2) targets the RMSE.
  - Neither → diagnose (rank histograms, spread per τ) before Phase 2; M3 becomes the main bet.

### Phase 2: trained tracks (2 agents, started after G0; launched at G1 depending on the outcome)

- **Agent B "score-prior FDV2" (M2, then M2'):**
  1. Measure the iterate-error curve ‖x_k − x_true‖ (normalised) of the deep-sup FDV2 on validation → σ_k schedule.
  2. Measure the RMS of the old prior channel against the new channel at σ_k.
  3. `prior_fm` option in `models/fourdvarnet.py`: Φ = a frozen FM prior loaded from a checkpoint, channel (x − Φ(x, σ_k))/σ_k, default off.
  4. Tests.
  5. Configs `L96B_fdvsubgradmask_monaiM_fmprior_ft_s{1,2}.yaml` (solver-only fine-tune from deep-sup, stable recipe, 300 epochs).
  6. Two jobs.
  - Accuracy gate G2a: point RMSE at N = 10/15 within +1% of B1. Then M1 sampling with the same prior.
  - M2' (native prior with the state-prediction FM recipe, plus a nearest-training-window memorisation check) only if G2a passes and the prior quality limits the samples.
- **Agent C "FDV2-CFM" (M3, conditional on G1):**
  1. `FourDVarNetSolver` flow mode: forward(x_τ, batch, τ) with the τ embedding and the σ(τ) channel; PSC loss; reuses `FourDVarNetPredictStateCFM` plumbing where possible.
  2. Initialise from deep-sup FDV2.
  3. Tests: τ → 1 identity, and the shapes.
  4. Two seeds.
  - Gate G2b: validation RMSE within +2% of B1 at τ = 0 (mean head), SSR ∈ [0.85, 1.15].
- Each trained model is evaluated by A2's harness and checked by agent V.

### Phase 3: selection and write-up (1 agent)

- **Agent S "synthesis":**
  - Pick one configuration per method on validation and run a single test scoring (both seeds, S0/S1, both layouts) with bootstrap CIs.
  - Write the results note (`l96_fdv2_fm_posterior.md` under `docs/results/`): the table with bold best and italic second-best per metric; rank-histogram and SSR-vs-lead-time figures.
  - Archive the ckpt + config of benchmark rows to `experiments/l96/`.
  - PR, stopping at CI green + reviewer approval for the user to merge.

## 5. Deliverables and expected PRs

1. Plan (this doc) + the `docs/README.md` row; the old plan marked SUPERSEDED.
2. Harness + baselines (P0).
3. FM posterior sampler + tests (A1), M1 sweep results note (A2).
4. Score-prior FDV2 option + configs (B).
5. FDV2-CFM (C), if launched.
6. Synthesis results note (S).

Each PR includes a CHANGELOG.d fragment.

## 6. Risks

- **The stochastic sampler degrades RMSE** (members wander off the mean). Mitigation: corrector steps only at τ ≥ τ0, γ tuned jointly on the CRPS + RMSE criteria, M1b starts.
- **ΠGDM cost** with ~720 observations/window: the Hutchinson diagonal (4 probes) before CG; the cap is ~0.25 s/step for 60 chains on an A40 (Langevin probe).
- **SDA1-M prior quality** (pure SDA RMSE 0.46): could limit the posterior far from the mean. M2' trains a better native prior; M1's warm start keeps it local.
- **Selection noise** (1 seed of each prior, 100 validation windows): two FDV2 seeds, bootstrap CIs, and a pre-registered grid (section 4).
- **FDV2 training instability** (3/5 full-rate runs diverged): fine-tune only, stable recipe.

## 7. Changes from `docs/plans/analysis/l96_fdv2_score_prior.md`

- **σ₀:** fixed, below the observation noise → a per-iteration σ_k matched to the iterate error; the channel normalised by σ_k (the old plan's x − Φ shrinks ~100× and the "initialise from the old Φ" argument was wrong).
- **Prior:** a new EDM/VE stack → the existing FM parameterisation (σ ↔ τ). SDA1-M is used first, with no training.
- **Sampler:** the probability-flow ODE (the deterministic warm start is the source of under-dispersion) → a stochastic sampler with ΠGDM and R in normalised units.
- **Protocol:** test-selected numbers → validation selection, 2 seeds, bootstrap CIs, post-hoc-inflation baseline B5, plus the conditional-flow baseline B6.
- **Memorisation:** added a memorisation check for any newly trained prior (1000 training windows).
