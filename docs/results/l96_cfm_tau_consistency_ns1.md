# τ-consistency on true marginals (NS1) and the sampler-side cost grid (NS0)

**Status:** RESULTS (2026-09-24). Track A of `docs/plans/analysis/l96_cfm_tau_consistency_next_steps.md` (v3).
Diagnostics only: no model, training or evaluation code is changed. Follows
`docs/results/l96_cfm_tau_consistency.md` (property labels A1, B4, D1, E1 are that doc's).

**Checkpoints.** The P1 M-tier monai flows, archived under
`4dvarnet-fm-fdv-tau-aware/experiments/`:
- `A2_predictstatecfm_monaiM_l96/checkpoints/stage1_best.ckpt` (PredictStateCFM-M, "PSC")
- `A1_vanillacfm_monaiM_l96/checkpoints/stage1_best.ckpt` (VanillaCFM-M)

**Data and settings.** L96 S0, 200 cached test windows. Scripts:
- `reports/l96/probe_tau_consistency_truth_marginal.py` (NS1): normalized space; 30 ODE members (N=10) for `g` and `D0`; 4 draws of x0 per window for each truth-built `x_tau`.
- `reports/l96/probe_ode_mean_along_path.py --x0-zero --steps-grid --members-grid --skip-main` (NS0): pooled RMSE in physical units, the convention of `docs/results/l96_cfm_tau_consistency.md` §2.

Outputs are in `reports/l96/outputs/cfm_tau_consistency/{ns1_truth_marginal,ns0_cost_grid}_{psc_m,vanilla_m}_s0.json`.
95% CIs are from a window bootstrap with 200 resamples.

**Why NS1.** #245's B1/B4 were measured on the ODE's own states, so a violation
mixed operator inconsistency with a wrong ODE marginal. NS1 builds `x_tau` from the
test-set truth, `x_tau = tau·x1 + (1 − tau)·x0`. That is exactly the training
distribution, and no sampler is involved.

---

## 1. NS1a — per-τ MMSE orthogonality on true marginals

Slope of the residual `x1 − D(x_tau^true, tau, y)` on `g(y) = ODE ensemble mean − D0`.
It is 0 at every τ for an exact operator. At τ = 0 it is the E1 slope. Monte-Carlo
noise in `g` is small (2.7% of `g`'s energy for PSC, 4.0% for VanillaCFM); noise-corrected
slopes differ by ≤ 0.03 (PSC) and ≤ 0.05 (VanillaCFM), largest at τ = 0.

| τ | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|---|---|---|
| **PSC** all_obs | **+0.97** | **+0.29** [0.28, 0.30] | +0.04 [0.03, 0.05] | −0.02 | −0.04 | −0.05 | −0.06 | −0.05 |
| PSC slow / fast | +0.99 / +0.97 | +0.43 / +0.27 | +0.11 / +0.03 | +0.01 / −0.03 | −0.03 / −0.05 | −0.05 / −0.05 | −0.08 / −0.05 | −0.08 / −0.05 |
| **VanillaCFM** all_obs | **+1.17** | **+0.32** [0.31, 0.33] | +0.06 [0.05, 0.06] | −0.00 | −0.01 | −0.01 | −0.01 | −0.00 |
| VanillaCFM slow / fast | +1.21 / +1.16 | +0.50 / +0.29 | +0.16 / +0.04 | +0.06 / −0.01 | +0.02 / −0.02 | +0.00 / −0.02 | −0.00 / −0.01 | −0.00 / −0.00 |

**Reading.**
- The first-moment inconsistency is **confined to τ ≲ 0.2**. The slope drops about 70% by τ = 0.1 and is near 0 from τ ≈ 0.2 (all_obs) or ≈ 0.3 (slow).
- For PSC, a small, statistically significant **negative** plateau (−0.04 to −0.06) holds for τ ≥ 0.4. On its own training distribution the operator is slightly more conservative than the ODE correction `g`, by about 5%.
- The slow variables are consistently the last to become consistent.

## 2. NS1b — the pairwise martingale on true marginals

`x_s` is a further-noised copy of `x_tau^true` (s < τ), built with the x-space pair
construction `x_s = (s/τ) x_tau + sqrt(b_s² − (s/τ)² b_tau²) ξ`. Its law given `x1`
is checked in `tests/test_tau_pair_construction.py`. We regress
`Δ = D(x_tau, tau) − D(x_s, s)` on two features of `x_s`. Every slope must be 0.

| s | feature `D(x_s, s) − D0`: PSC (all / slow / fast) | VanillaCFM (all / slow / fast) |
|---|---|---|
| 0 | −0.97 / −0.97 / −0.97 (every τ) | −0.96 / −0.97 / −0.96 |
| 0.1 | +0.07 / −0.00 / +0.08 | −0.02 / **−0.11 to −0.15** / −0.01 |
| 0.2 | +0.01 to +0.02 | −0.01 to −0.02 (slow −0.03 to −0.05) |
| 0.3 | ≤ +0.01 | −0.01 to −0.02 |

The second feature, `x_s` itself, gives |slope| ≤ 0.026 and |corr| ≤ 0.09 in every
cell, for both models.

**Reading.**
- **s = 0: slope ≈ −1.** The x0-dependence of the τ=0 prediction (A1) is **entirely spurious**. No later τ follows it, i.e. `E[D(x_tau) | x0]` does not move with `D(x0, 0)`. This is the pairwise view of A1 and of defect 1.
- **s ≥ 0.2: ≈ 0 for both models.** The operator satisfies the martingale identity on true marginals.
- **s = 0.1: small** (PSC fast +0.07; VanillaCFM slow −0.11 to −0.15), consistent with NS1a's residual at τ = 0.1.

## 3. NS1c — second-order consistency on true marginals (added; not in v3)

v3's NS1 tests only first-moment identities, while defect 2 (the B4 variance decay)
is second order. The truth-marginal second-order identity combines MMSE with
second-order Tweedie:

  `E|x1 − D(x_tau)|² = (b_tau² / tau) · E[tr ∇_x D] / n`

It holds at every τ for an exact operator. The trace uses Hutchinson estimation with
central finite-difference JVPs (ε = 1e-2, validated against autograd in #245). The
table gives the ratio of the Jacobian term to the residual MSE; the target is 1.

| τ | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|
| **PSC** all_obs | 0.90 | 1.07 | 1.24 | 1.28 | 1.16 | 0.92 | **0.60** | **0.30** | **0.08** |
| PSC slow / fast | 1.08 / 0.88 | 1.06 / 1.07 | 1.07 / 1.28 | 0.93 / 1.37 | 0.71 / 1.31 | 0.46 / 1.11 | 0.26 / 0.80 | 0.11 / 0.43 | 0.03 / 0.12 |
| **VanillaCFM** all_obs | **1.90** | 1.62 | 1.32 | 1.31 | 1.35 | 1.28 | 1.15 | 1.14 | 1.26 |
| VanillaCFM slow / fast | 4.12 / 1.65 | 2.80 / 1.45 | 1.74 / 1.25 | 1.51 / 1.28 | 1.49 / 1.33 | 1.34 / 1.27 | 1.19 / 1.14 | 1.31 / 1.09 | 1.42 / 1.19 |

**Reading.** Second-order consistency is **violated on true marginals too**, but
with a model-dependent sign and τ-profile:

- **PSC's Jacobian under-states the residual variance at τ ≥ 0.6**, 12× at τ = 0.9 and 30× for slow.
  - D is nearly insensitive to `x_tau` late in the path. Its implied posterior covariance there is far too small, and the Euler map contracts deviations accordingly.
  - Over [0.3, 0.5] it over-states mildly (1.2–1.4× for fast).
- **VanillaCFM over-states everywhere**: 1.1–1.9× all_obs, 4.1× for slow at τ = 0.1.
- **Implication.** #245's ODE-based B4 decayed about 3× on *both* models with the same profile, while the truth-marginal pattern differs in sign between them. So the B4 decay cannot be attributed to a single operator defect.
  - It is at least partly a second-order operator problem (NS1c ≠ 1), and partly the ODE marginal: the same Jacobian evaluated on the ODE's states rather than the data's.
  - The split is not identified by these runs.

**Caveat.** Hutchinson uses one Rademacher probe per draw (4 per window-τ), pooled
over 200 windows × 3000 steps × channels. The estimator is unbiased, and pooled
noise is small against the ratios quoted.

## 4. NS0 — sampler side: refinement vs ensembling, and cheap point estimators

Endpoint ensemble-mean RMSE (physical units, pooled all_obs) and network calls
(steps N × members M):

| N × M (calls) | 1×1 (1) | 1×30 (30) | 2×1 (2) | 2×8 (16) | 2×30 (60) | 3×8 (24) | 3×30 (90) | 10×30 (300, P1) |
|---|---|---|---|---|---|---|---|---|
| **PSC** | 0.454 | 0.444 | **0.430** | 0.417 | 0.416 | **0.411** | 0.409 | 0.400 |
| **VanillaCFM** | 0.474 | 0.446 | **0.441** | 0.415 | 0.413 | **0.407** | 0.404 | 0.389 |

The 10×30 column is from `docs/results/l96_cfm_tau_consistency.md` §2.

- **NS0a (a path from x0 = 0) is uninformative, as v3's caveat anticipated.**
  - The deterministic path scores 0.510 (PSC) and 0.799 (VanillaCFM), worse than a single random member.
  - Its τ=0 prediction is already 0.59 / 0.94: x0 = 0 is far outside the input distribution (norm 0 vs about `sqrt(n)·s0`).
  - The grid answers the question instead. **A single 2-step member (2 calls) beats the 30-draw τ=0 mean (30 calls)**: 0.430 vs 0.444, and 0.441 vs 0.446. So refinement alone accounts for part of the gain.
  - Averaging over members adds the rest (2×30: 0.416 / 0.413). The gain is both refinement and ensembling, with refinement first.
- **NS0c.** **3 steps × 8 members (24 calls, 8% of P1's 300)** reaches 0.411 / 0.407, within 2.6% / 4.7% of the full ensemble mean. N = 3 captures most of the step gain, and M = 8 most of the member gain. This is the honest cheap point estimator for the flows, and a replacement candidate for the P1 "τ=0 mean" row.

---

## 5. Reading against v3's NS1 decision rule

v3 §3 NS1: *"NS1a falls to 0 by τ ≈ 0.1 and NS1b ≈ 0 → defect confined to τ = 0:
Batch 1 stands, Batch 2 is dropped … NS1a/NS1b clearly nonzero over [0.1, 0.4] →
Batch 1, then Batch 2."*

| question | answer |
|---|---|
| Does NS1a fall to 0 by τ ≈ 0.1? | **Almost.** 0.29 / 0.32 at τ = 0.1, ≈ 0.04–0.06 at τ = 0.2, ≈ 0 by 0.3. The first-moment defect lives in τ ∈ [0, 0.2] |
| Is NS1b ≈ 0 over [0.1, 0.4]? | **Yes for s ≥ 0.2; small at s = 0.1** (≤ 0.08, except VanillaCFM slow at −0.11 to −0.15) |
| **Batch 1 (T0, T1′, T2a)** | **Stands, and its teacher range is validated.** T2a draws teacher targets from τ' ∈ [0.2, 0.5], exactly where NS1a and NS1b are ≈ 0 on both models |
| **Batch 2 as designed (T2b: a first-moment martingale target over [0.1, 0.4])** | **No-go.** There is little first-moment inconsistency left to fix at s ≥ 0.2. The residual at τ ∈ (0, 0.2] is better covered by widening T2a's student range to τ ∈ [0, 0.1] than by T2b |
| Defect 2 (B4 decay) | **Not resolved by NS1, and not purely sampler-side.** NS1c shows second-order inconsistency on true marginals, with a model-dependent sign. A first-moment target such as T2b would not address it |

**Implication for the plan (proposal only; not implemented).**
- The truth-marginal second-order identity is **per-sample trainable from a single `x1`**, unlike the per-`y` B4 of v3's T3. A second-order denoising objective of the form `‖(b_tau²/tau) J u − (x1 − D)(x1 − D)ᵀ u‖²` (Meng et al., NeurIPS 2021 *(recall)*) has the NS1c identity as its expectation.
- That makes it the natural Batch-2 candidate for defect 2, in place of T2b. It should be decided before Batch 2.

## 6. Caveats

- One checkpoint per model, S0 only. P1 measured 15–21% checkpoint-selection noise, so the magnitudes can move; the qualitative pattern (defect ≤ τ 0.2; NS1b ≈ 0 for s ≥ 0.2) is shared by both parameterisations.
- NS1a uses `g = ODE mean − D0` as its only test feature. Orthogonality to one feature is necessary, not sufficient.
- NS1b uses two features. The martingale could fail along directions they do not span.
- NS1c's ratios use the pooled trace (the sum over all coordinates). The diagonal Jacobian over a 3000-step window is dominated by directions where the posterior covariance is tiny (see `docs/results/l96_cfm_tau_consistency.md` §3 reading 5), but the identity is exact in trace, so the ratio stays valid.
- NS0 uses the same seed and draws for every grid cell within a step count N, but cells with different N are independent draws.
