# L96 FDV2 with a score-based prior: denoising-score-matching Φ, 4D-Var prior channel and posterior sampling

**Status:** SUPERSEDED (2026-10-12) by `docs/plans/analysis/l96_fdv2_fm_posterior.md`, before any implementation; its sections 1.1–1.5 (evidence) remain the reference cited there. Was PLAN v1 (2026-10-12); gate (section 4, step 0) passed. Follows `docs/results/l96_fdv_m_mask_prior_unroll.md` (FDV2 + mask, frozen Φ, deep-supervision fine-tune) and the uncertainty probes of 2026-10-12 summarised in section 1.

## 0. Goal

Make the FDV2 prior Φ a proper (noise-conditional) score model so that one trained model gives:

1. **the estimate** — the unrolled solver, as today;
2. **posterior samples** — guided annealed sampling with the model's *own* prior, warm-started from the solver estimate;
3. **a truth-free per-window error indicator** — the prior residual at the estimate (already found useful, see 1.3).

This is P2's admission criterion for the FDV family ("samplable and posterior-tractable").

## 1. Evidence motivating the plan (2026-10-12, M tier, S0, 200 test windows)

All probe scripts and JSON outputs are in the `4dvarnet-fm-l96-fdv-mask` worktree, `experiments/diag_resid_collapse/` (not checked in): `jcurves_per_window.py`, `ensembles.py`, `langevin.py`, `weighted_pobs.py`, `sda_warm.py`.

### 1.1 The current Φ is not a density

Φ is trained by an aux reconstruction loss ‖x − Φ(x)‖² on true states and solver outputs. Its L2 minimiser is the identity; capacity and dropout make it an approximate projection onto the data manifold. U(x) = ½‖x − Φ(x)‖² is ≈ 0 along the whole manifold, so it has no normalisable density there.

**Langevin on E(x) = ½ Σ m(x − y)²/R + λU(x)** (100 steps, η = 0.01, 30 chains, from the FDV2 estimate) is worse than every baseline at every (λ, T) tried (calibration: 40 regular-grid S0 windows):
- best CRPS 0.1318 (λ = 1, T = 0.01) vs 0.1162 for the perturbed-observation ensemble;
- spread/RMSE ≈ 1 only by moving members away from the estimate (ensemble-mean RMSE 0.27 vs 0.24);
- λ ≥ 100: chains stall or diverge.

### 1.2 Baseline ensembles (seed-2 coupled deep-supervision FDV2 fine-tune, 30 members)

| ensemble | regular CRPS | regular spread/RMSE | random CRPS |
|---|---|---|---|
| random initial states | 0.1414 | 0.08 | 0.1609 |
| perturbed observations (y + R^½ε) | **0.1147** | 0.57 | **0.1333** |
| perturbed obs, tempered-likelihood weights (best β = 0.03) | 0.1145 | 0.56 | 0.1330 |
| perturbed obs, full-likelihood weights (β = 1) | 0.1517 | 0.23 | 0.1744 (2 effective members) |

The perturbed-observation ensemble is the reference to beat: FDV2 is 12% better than FDV1-M with it (0.1142 vs 0.1296). It is 2× under-dispersed because nothing represents prior uncertainty. Likelihood weighting cannot fix this: it only reweights existing members, and full weights collapse.

### 1.3 Truth-free error indicator

J_prior = ‖x̂ − Φ(x̂)‖² of the final estimate (own Φ), coupled deep-supervision fine-tune, both seeds:
- regular grid: Spearman ρ = 0.63–0.80 with per-window RMSE; AUROC 0.90–0.91 for flagging the worst 10% windows (S0 and S1);
- random layouts: ρ = 0.59–0.67 within observation-count strata.

J_obs is uninformative (ρ ≈ −0.2). The indicator is weaker or inconsistent for the frozen-Φ fine-tune and the pretrained model.

### 1.4 Stopping rules

Deep-supervised solvers are best at about 15 iterations. A stationarity rule on J_obs matches fixed 15; the per-window oracle is only 0.7% better. A discrepancy rule fails because J_obs falls below its expected value before the error minimum.

### 1.5 SDA warm start (idea 3a, job 60316, `sda_warm.json`)

The benchmark's SDA1-M prior (`B4_sda1_monaiM_ep1200_l96_seed1`), via `sda_guided_sample(mean_estimate=x̂, tau0=τ0)`; S0, 30 members, 200 windows. Deterministic FDV2 (10 iterations): 0.2321 regular / 0.2759 random.

| start | τ0, guidance weight | regular: ensemble-mean RMSE / CRPS / spread/RMSE | random: ensemble-mean RMSE / CRPS / spread/RMSE |
|---|---|---|---|
| pure SDA (noise) | 0, 20 | 0.4630 / 0.2374 / 0.39 | 0.5717 / 0.2974 / 0.36 |
| FDV2 coupled deep-sup. (seed 2) | 0.1, 2 | 0.2411 / 0.1166 / 0.57 | 0.2810 / 0.1347 / 0.49 |
| FDV2 | 0.3, 2 | 0.2310 / 0.1202 / 0.34 | 0.2737 / 0.1423 / 0.28 |
| FDV2 | 0.5, 2 | 0.2268 / 0.1237 / 0.24 | 0.2694 / 0.1463 / 0.20 |
| FDV2 | 0.7, 2 | **0.2262** / 0.1297 / 0.15 | **0.2685** / 0.1522 / 0.13 |
| FDV1-M (seed 2) | 0.5, 2 | 0.2333 / 0.1278 / 0.23 | 0.2778 / 0.1504 / 0.19 |

As in `docs/results/l96_psi_decomposition.md`, the warm start is refinement, not posterior sampling. τ0 = 0.5–0.7 gives the best point estimate so far (−2.5% regular, −2.7% random vs deterministic FDV2), but almost no spread. No τ0 beats the perturbed-observation ensemble's CRPS (0.1147 / 0.1333): τ0 = 0.1 comes closest at 0.1166 / 0.1347. **Gate outcome: 3b's uncertainty target stands (section 5).**

## 2. Mathematics

Notation: x ∈ ℝⁿ (n = 3000 × 24) is the normalised window state, p(x) its prior, y = Hx + η with η ~ N(0, R), H the observation selection (mask m), R diagonal.

### 2.1 Gaussian smoothing and Tweedie's formula

Corrupted state x̃ = x + σε, ε ~ N(0, I). Smoothed prior p_σ(x̃) = ∫ p(x) N(x̃; x, σ²I) dx.

Since ∇_x̃ N(x̃; x, σ²I) = N(x̃; x, σ²I) (x − x̃)/σ²:

  ∇ log p_σ(x̃) = (E[x | x̃] − x̃)/σ²,  equivalently E[x | x̃] = x̃ + σ² ∇ log p_σ(x̃)   (Tweedie).

Second order: Cov[x | x̃] = σ² ∂E[x | x̃]/∂x̃.

### 2.2 Denoising score matching

  min_θ E_{x ~ data, σ ~ π, ε} λ(σ) ‖Φ_θ(x + σε, σ) − x‖².

The minimiser is Φ*(x̃, σ) = E[x | x̃], hence the score estimate

  s_θ(x̃, σ) = (Φ_θ(x̃, σ) − x̃)/σ² ≈ ∇ log p_σ(x̃)   (Vincent 2011).

**EDM parameterisation** (Karras et al. 2022), F_θ = the current Φ MONAI UNet:
- Φ(x̃, σ) = c_skip x̃ + c_out F_θ(c_in x̃, c_noise);
- c_skip = σ_d²/(σ² + σ_d²), c_out = σσ_d/√(σ² + σ_d²), c_in = 1/√(σ² + σ_d²), c_noise = ¼ ln σ;
- σ_d = 1 (normalised data); loss weight λ(σ) = (σ² + σ_d²)/(σσ_d)²;
- π: log-uniform on [σ_min, σ_max] = [0.01, 5] (or EDM's log-normal, P_mean = −1.2, P_std = 1.2).

**Data: true states only.** Drop the current aux term on solver outputs: they are not samples of p(x), and that term made Φ task-shaped.

### 2.3 The solver's prior channel becomes a 4D-Var gradient

At a fixed small σ₀: x − Φ(x, σ₀) = −σ₀² s(x, σ₀) ≈ σ₀² ∇[−log p_σ₀(x)].

**Gaussian check**, p = N(μ, B): E[x | x̃] = μ + B(B + σ²I)⁻¹(x̃ − μ), so

  x − Φ(x, σ₀) = σ₀²(B + σ₀²I)⁻¹(x − μ) → σ₀² B⁻¹(x − μ) as σ₀ → 0,

i.e. the 4D-Var background gradient. Keep the input layout `subgrad+mask+state` = [(y − x)·m, x − Φ(x, σ₀), x, m]. Choose σ₀ in [0.05, 0.1]: below the observation noise (≈ 0.37 normalised), but not → 0, where the score of near-manifold data blows up like 1/σ² normal to the manifold. Optional variant: 2–3 levels as extra channels.

### 2.4 Posterior score and guidance

  ∇ log p_σ(x̃ | y) = s_θ(x̃, σ) + ∇_x̃ log p_σ(y | x̃),  with p_σ(y | x̃) = ∫ p(y | x) p(x | x̃) dx intractable.

- **DPS:** p_σ(y | x̃) ≈ N(y; H Φ(x̃, σ), R) ⇒ ∇ ≈ J_Φᵀ Hᵀ R⁻¹ (y − H Φ(x̃, σ)), where J_Φ = ∂Φ/∂x̃ (one vector-Jacobian product through Φ). The current SDA sampler uses a heuristic normalised step with weight `gw`.
- **ΠGDM:** p_σ(y | x̃) ≈ N(y; H Φ, R + σ² H J_Φ Hᵀ) (Tweedie covariance). No guidance weight to tune. Implement with a few conjugate-gradient steps on the observed entries, or a diagonal approximation of H J_Φ Hᵀ estimated by Hutchinson probes.

### 2.5 Sampler (annealed Langevin / reverse VE-SDE)

- Schedule σ_K > … > σ_1 = σ_min, geometric (K ≈ 10–20).
- At each level, L steps (L = 1–3) of
  x ← x + η_i [s_θ(x, σ_i) + ∇ log p_σi(y | x)] + √(2η_i) ξ,  with η_i = ε_step σ_i²/σ_min².
- Alternatively the EDM Heun sampler on the probability-flow ODE plus noise churn.

**Warm start:** x_K = x̂_FDV2 + σ* ε, annealing from σ* to σ_min. Members = independent ε and ξ.
- σ* global (calibrated): about 0.1–0.5 normalised.
- **σ* per window**, from the error indicator: σ*_w = a · √J_prior(x̂_w) + b, with a and b calibrated on held-out windows.

## 3. Code changes (all default off)

1. **`models/fourdvarnet.py`:**
   - `prior_sigma_conditioning: bool` builds Φ with the EDM wrapper (2.2).
     - The MONAI backbone already has a timestep embedding: pass c_noise(σ) as `tau` to `MonaiUNet1D.forward`. Today Φ is called with `tau=None`, i.e. zeros.
     - The existing `prior_tau_conditioning` flag raises for MONAI (legacy-only). Add a new flag; do not reuse it.
   - `prior_dsm_weight` replaces `aux_var_cost_weight` when DSM is on. The aux loss becomes the DSM loss on `batch.states` only, with σ ~ π and the EDM λ(σ). Keep `aux_detach_x_final` irrelevant (no solver-output term).
   - `prior_channel_sigma: float` (σ₀): `_build_update_input`'s `subgrad+mask+state` branch calls Φ(x, σ₀) via `_prior_for_solver` (keep its `grad_scale` / `freeze_prior_in_solver` semantics).
   - Helpers:
     - `_prior_denoise(prior_unet, x, sigma)`, the EDM-wrapped forward;
     - `prior_score(x, sigma) = (Φ(x, σ) − x)/σ²`.
2. **`evaluation/score_sampler.py`** (new):
   - `annealed_guided_sample(model, batch, sigmas, n_members, x_init=None, sigma_star=None, guidance="dps"|"pigdm", R_var=0.5, L=1, eps_step=…)`, returning (B, T, D, M).
   - Reuse `evaluation/sda_sampler.py`'s structure and its observation-cost helper; members can be batched along a new axis.
3. **Configs:**
   - `L96B_fdvsubgradmask_monaiM_dsm_ft_s2.yaml`: fine-tune from `L96B_fdvsubgradmask_monaiM_aux01det_frozen_seed2` stage1_best. `freeze_prior_in_solver: true`; `prior_unet_lr_scale: 0.1` (the stable recipe); `tau_schedule: clamp`; `train_n_blocks: 2`; 300 epochs at lr 1e-4; `prior_sigma_conditioning: true`; `prior_channel_sigma: 0.1`; `prior_dsm_weight: 1.0`.
   - A seed-1 twin from `..._frozen_philr01_seed1`.
   - Note: the existing Φ weights do not match the EDM-wrapped network's expected input scaling at all σ. Initialise from them anyway (c_skip → 1, c_out → 0 at small σ keeps the solver channel close at the start). Alternatively, pre-train Φ alone first (step 2).
4. **Tests** (`tests/test_fdv_score_prior.py`):
   - EDM coefficients;
   - DSM loss on a 1-D Gaussian toy: the learned denoiser ≈ the analytic E[x | x̃] after a few hundred steps;
   - the score of a Gaussian prior equals −(B + σ²I)⁻¹(x − μ) when Φ is the analytic denoiser;
   - `prior_channel_sigma` wiring;
   - sampler shapes;
   - a warm start with σ* → 0 returns x_init;
   - DPS / ΠGDM guidance on a linear-Gaussian toy reproduces the analytic posterior mean and variance within tolerance.

## 4. Experiments, in order

0. **Gate (done 2026-10-12):** warm-started SDA does not beat the perturbed-observation CRPS (1.5), so the uncertainty target stands. Its τ0 = 0.5–0.7 refinement (RMSE 0.2262 regular S0) is the point-estimate bar for 3b's sampler mean.
1. **Toy validation (CPU, minutes):** the tests in 3.4, including the linear-Gaussian posterior check of both guidance schemes.
2. **Φ-only DSM pre-training** (about 2 h on an A100): train the EDM-wrapped Φ on L96 training states (`data.train_cache`), solver frozen. Checks:
   - denoising MSE vs σ;
   - Tweedie consistency: the residual of denoised samples is ≈ σ-scaled noise;
   - **unconditional samples** from the annealed sampler without guidance: their statistics match the climatology (per-channel mean, variance, lag-autocorrelation, energy spectrum). This is the sanity check that Φ is a density.
3. **FDV2 DSM fine-tune** (step 3.3 configs), seed 2 then seed 1: about 13 h each, 2 jobs in parallel.
   - **Accuracy gate:** N = 10 and 15 RMSE within ±1% of the current coupled deep-supervision fine-tune (0.2321 / 0.2287 regular S0 on seed 2). The solver channel changes, so accuracy must not regress.
   - Also re-run the J_prior error-indicator AUROC with the DSM Φ.
4. **Posterior sampling evaluation:**
   - Calibrate on 40 even regular-S0 windows: σ* (global vs per-window), K, L, ε_step, DPS vs ΠGDM.
   - Score on held-out odd windows, regular and random, S0 and S1: CRPS, spread/RMSE, rank histograms, ensemble-mean RMSE.
   - Baselines, all with 30 members: the perturbed-observation ensemble with the same model (CRPS 0.1147 / 0.1333), SDA from noise, and the SDA warm start (3a: best point estimate 0.2262 / 0.2685 at τ0 = 0.7; best CRPS 0.1166 / 0.1347 at τ0 = 0.1).
   - Combined variant: perturbed y plus annealed sampling (closest to randomised maximum likelihood with a prior).
5. **Write-up:** a new results note named `l96_fdv2_score_prior.md` under `docs/results/` (append-only convention) and a PR.

## 5. Success criteria

- Accuracy: no regression beyond ±1% (step 3).
- Uncertainty: held-out CRPS at least 5% below the perturbed-observation ensemble with spread/RMSE in [0.8, 1.1], on both test sets.
- Error indicator: AUROC ≥ 0.85 (regular, both seeds).
- Unconditional samples pass the climatology checks of step 2.

## 6. Risks

- **Solver-channel regression:** the DSM-trained Φ at σ₀ changes the solver's input distribution. Mitigation: fine-tune with Φ frozen in the solver and slow (the stable recipe); try σ₀ ∈ {0.05, 0.1, 0.2}.
- **FDV2 instability:** 3 of 5 full-rate FDV2 runs diverged. Keep `prior_unet_lr_scale: 0.1` and fine-tune rather than train from scratch.
- **Guidance stiffness with ~720 observations per window:** DPS with exact R⁻¹ can overshoot at small σ. ΠGDM's covariance term and step-size control (ε_step) address it; the fallback is the SDA sampler's normalised step.
- **Cost:** annealed sampling with backprop through Φ is about 0.25 s per step for 60 chains on an A40 (as in the Langevin probe). Keep K × L ≈ 20–40 and batch members.

## 7. Pointers

- Code: `models/fourdvarnet.py` (`_prior_ae`, `_prior_for_solver`, `_build_update_input` `subgrad+mask+state` branch, `compute_loss` aux block, the `prior_tau_conditioning` guard for MONAI); `evaluation/sda_sampler.py` (warm start, guidance); `models/monai_unet_adapter.py` (timestep embedding).
- Checkpoints:
  - FDV2 frozen-Φ seed 2: `4dvarnet-fm-l96-fdv-mask/experiments/L96B_fdvsubgradmask_monaiM_aux01det_frozen_seed2`;
  - FDV2 slow-Φ seed 1: `.../L96B_fdvsubgradmask_monaiM_aux01det_frozen_philr01_seed1`;
  - deep-supervision fine-tunes: `..._deepsup_ft_coupled_seed2`, `..._deepsup_ft_coupled_s1philr_seed1`;
  - SDA1-M prior: main `experiments/l96/B4_sda1_monaiM_ep1200_l96_seed1`.
- Batch: `batch/run_l96b_fdv_subgrad_aux.sbatch` (`FDV_CFG`, `FDV_SEED`; submit from the worktree).
