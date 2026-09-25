# τ-consistency of the L96 flow-matching operators — synthesis

**Status:** RESULTS / SYNTHESIS (2026-09-25). This note **closes** the τ-consistency line: it is the reference to cite. It supersedes, for citation purposes, the seven notes it summarises. They stay the detailed record; results docs are append-only.

| note | PR | content |
|---|---|---|
| `docs/results/cfm_tau_consistency.md` | #245 | first probes: per-step predictions, E1, B2/B4, A1/A3, C1 |
| `docs/results/cfm_tau_consistency_ns1.md` | #250 | truth-marginal tests NS1a–c; sampler cost grid |
| `docs/results/cfm_tau_consistency_batch1.md` | #251 | training arms T0 / T1′ / T2a |
| `docs/results/cfm_sampler_schedule.md` | #252 | stochastic and step-schedule samplers (S1) |
| `docs/results/cfm_tau_consistency_t5.md` | #255 | second-order training loss (T5) |
| `docs/results/cfm_tau_consistency_l96b.md` | #256 | re-probe on the benchmark-default flows, 400 vs 1200 epochs |
| `docs/results/l96_rank_histograms_c4.md` | #260 | rank histograms (P1 paper C4) |

The plan that drove the line, `docs/scoping/cfm_tau_consistency_next_steps.md` (v1–v4, #245–#252), is closed by this note. Its open items are listed in §6.

**Scope:**
- **System:** two-scale L96, 24-D observed subspace, S0, 200 cached test windows.
- **Models:** PredictStateCFM-M (**PSC**, endpoint target) and VanillaCFM-M (**Van**, velocity target), monai backbone. Two training recipes:
  - **P1**: the regular observation grid, 400 epochs;
  - **L96B** (benchmark default): the random observing system, 400 epochs × 3 seeds, plus 1200 epochs × 1 seed.
- **Conventions:** the probe numbers use the notes' own conventions (pooled RMSE, normalised space for the identities). Benchmark numbers are quoted from the regenerated reports (#257).

---

## 1. The question and the answer

**Question.** The flow's ODE ensemble mean beats the same network's τ=0 prediction `D(x0, 0, y)`, whose exact target *is* `E[x1|y]`, the MSE-optimal estimator. Why? Is the flow averaging a noisy estimate of `E[x1|y]` over τ?

**Answer.**
- **It is not averaging.** Per-step errors correlate at 0.90–0.99 across τ; an explicit average over τ gains ≈ 0.3%; and the uniform-Euler endpoint *is* the last per-step prediction.
- **The τ=0 prediction is biased,** and the flow **self-refines** it within 1–2 Euler steps: at τ = 0.1 the network sees its own previous estimate through `x_tau` and corrects it (implicit self-conditioning).
- **This held on every model and recipe, and the gain grows with training** (§2.1). The τ=0 prediction should not be used as a point estimator. If a cheap estimator is needed, 3 steps × 8 members (24 network calls) is within 2.6–4.7% of the 300-call ensemble mean (#250).

## 2. What the exact operator requires, and what the trained ones do

`D(x, τ, y) = E[x1 | x_tau = x, y]` on the linear path `x_tau = τ x1 + b_τ ε`, with `b_τ = (1 − τ) σ0`.

| property (label) | exact operator | PSC | Van | notes |
|---|---|---|---|---|
| τ=0 is the posterior mean (E1: residual slope on the ODE correction) | 0 | ≈ 1 (0.89–0.99) | ≈ 1 (1.06–1.14) | **violated everywhere** |
| τ=0 ignores its noise input (A1) | 0 | 0.12–0.21 | 0.20–0.31 | shrinks with training |
| first-moment consistency on true marginals, τ > 0 (NS1a/b) | 0 | P1: 0.29 at τ=0.1, ≈ 0 from 0.2; L96B: ≈ 0.1 at τ=0.1 | same | fine except at τ = 0 |
| second-order, Jacobian ↔ residual variance (NS1c; target 1) | 1 | 0.55–1.3 over τ 0.1–0.5 (0.9 at τ=0.1 for P1, 0.55–0.59 for L96B), **0.07–0.09 at τ = 0.9 (every run)** | 1.1–1.9 (P1 and L96B 400 ep), 0.8–1.3 (1200 ep) | model-specific |
| total variance conserved along τ (B4 = B4(0.1)/B4(0.9)) | 1 | 2.6 (P1), 1.5–1.7 (L96B) | 3.2 (P1), 1.4–2.1 (L96B) | mostly sampler (§2.3) |
| Jacobian symmetric, i.e. the velocity is a gradient field (C1; ‖(J−Jᵀ)u‖/‖Ju‖) | 0 | 1.2 → 0.35 | 1.2 → 0.40 | expected for an unconstrained net; no measured harm |
| posterior shape after spread and bias correction (rank-histogram RI_db; outer-bin ratio) | 0; 1.0 | 0.12–0.13; **1.6–1.8** | **0.03** (P1), 0.04–0.11 (L96B); 1.0–1.5 | see §3 |

### 2.1 Defect 1 — the biased τ=0 prediction

| model | τ=0 gap vs the ODE endpoint (pooled RMSE) |
|---|---|
| P1 PSC / Van | +11% / +15% |
| L96B, 400 epochs (3 seeds) | +14% / +17–18% |
| L96B, 1200 epochs | **+18% / +21%** |

- Longer training improves the endpoint more than the τ=0 prediction, so the refinement matters more as models improve.
- **Training the τ=0 prediction harder fails.** Adding 25% of each batch at τ = 0 (T1′), even with lower-variance EMA-teacher targets (T2a), makes the network **memorise** each training window's `x1` at 1000 windows. Ensemble-mean RMSE got worse by 10–19% vs the paired baseline (#251).
- The uniform-τ network's τ=0 bias is partly a *regularised* compromise. It is benign for sampling, which uses the τ=0 prediction only for the first step.
- **With the random observing system (L96B),** the first-moment defect sits at τ = 0 alone (NS1a ≈ 0.1 at τ = 0.1, vs 0.3 for P1).

### 2.2 Defect 2 — under-dispersion

| sampler | P1 PSC / Van | L96B PSC / Van (400 ep) |
|---|---|---|
| uniform Euler, N = 10 (the pre-#253 benchmark) | 0.53 / 0.66 | 0.70 / 0.77 |
| **early-fine** `τ_k = 1 − (1 − k/N)^0.5`, N = 10 | 0.61 / 0.73 | 0.84 / 0.92 |
| **early-fine, N = 20 (the benchmark since #257)** | — | 0.86–0.92 / 0.95–1.02 |
| uniform, N = 80 (≈ the ODE's own dispersion) | 0.64 / 0.81 | 0.93–0.95 / 1.04–1.07 |

Spread/RMSE, pooled convention (#252, #256).

**Readings:**
- **The collapse was mostly a sampler effect** once training covered the observing-system variability. For L96B the ODE's own dispersion is near-calibrated. What was lost at N = 10 was Euler discretisation, concentrated in the *early* steps: late-fine schedules are clearly worse. Early-fine steps recover most of it at no cost.
- On the benchmark reports (per-window convention), spread/RMSE went 0.53 → 0.70 (PSC) and 0.62 → 0.79 (Van) at regular S0, with CRPS −4%/−2.5% (#257).
- **For the P1 recipe** there was also a real *operator* share: limits of 0.64 / 0.81. It points to training-data coverage, not the method.
- **Longer training takes some dispersion back:** the 1200-epoch PSC limit is 0.88 (vs 0.93–0.95).
- **Stochastic (DDIM-η / ancestral) sampling makes it worse:** it discards the carried noise in favour of the late-τ predictions, which under-state variance (#252).
- **A second-order training loss (T5) did not help, and says nothing either way.** The single-probe objective has expectation `(a − t)² + Var_u(â)`. The probe variance rewards shrinking the whole Jacobian: NS1c fell 10–15% at every τ, and λ = 100 diverged at switch-on (#255). A corrected, two-probe version is specified in that note but was not run. For L96B there is little operator share left for it to fix.

## 3. The one clear model failure: PredictStateCFM's late-τ Jacobian

**Consistent across both recipes, every seed, both samplers and both epoch budgets:**
- **NS1c ≈ 0.07–0.09 at τ = 0.9:** the prediction barely reacts to its input near τ = 1, so its implied posterior covariance there is ≈ 12× too small.
- **Lower ODE-limit dispersion than Van:** 0.88–0.95 vs 1.01–1.07.
- **Rank-histogram tails too light** after spread and mean-bias correction: outer-bin ratio 1.6–1.8, RI_db 0.12–0.13. Van reaches RI_db 0.03 under P1 (#260).
- Van shares the backbone, data and training, and shows none of this, so the failure is in the **parameterisation**, not the architecture or the data.

**Likely mechanism (interpretation, not yet tested).**
- Near τ = 1 the exact operator is close to the identity (`D ≈ x_tau`, `∇D ≈ I`).
- Van predicts the velocity, and `D = x + (1 − τ) v` gets the identity for free.
- PSC must learn the identity map itself, and a regularised network returns a smoothed, input-insensitive version.
- The last Euler step returns `D` directly, so the members are pulled together at the end.
- This is the motivation of EDM-style skip preconditioning.

**Test (open, §6).** One PSC run with `D = τ·x + (1 − τ)·net`. If the parameterisation is the cause, three things move:
- NS1c at τ = 0.9 rises toward 1;
- the rank-histogram tails flatten;
- the ODE-limit dispersion matches Van's.

**Van's own second-order error runs the other way** (NS1c 1.1–1.9 for P1 and the 400-epoch L96B runs, i.e. Jacobian too large). It shrinks with training (0.8–1.3 at 1200 epochs) and shows up only as mild light tails in some L96B seeds (outer-bin ratio 1.1–1.5), not in P1 (1.0).

## 4. Posterior shape (rank histograms, P1 paper C4)

After correcting spread and per-channel mean bias (#260, S0 regular grid, 30 members):

| scheme | P1 protocol | benchmark default | tails vs truth |
|---|---|---|---|
| ETKF / EnKF | 0.14 / 0.10 | 0.18 / 0.11 | too heavy (hump), fast channels |
| **VanillaCFM-M** | **0.03** | 0.04–0.11 | ≈ right (P1); mildly light in some L96B seeds |
| PredictStateCFM-M | 0.13 | 0.12–0.13 | too light (§3) |
| SDA1-M / SDA2-M | 0.18 / 0.27 | — | too light |

- The ensemble filters' apparent shape defect is **mostly mean bias** from forward-model error: the P1 DA runs without per-window fast weights, and its slow channels are flat once de-biased.
- After both corrections no class stands out. Shape errors are scheme-specific, not ordered by operator-slot occupancy.
- **C4, as currently phrased, is not supported as a discriminating claim.** The rewording is an open authorial decision (§6).

## 5. What to carry into the papers

- **P2 — the mechanism claim.** A conditional flow self-refines its own amortised τ=0 estimate within 1–2 steps: an implicit unrolled refinement, structurally what a 4DVarNet solver does explicitly. The gain is robust across parameterisations and recipes, and **grows** with training (+11% → +21%).
- **P1 — calibration.** The flows' under-dispersion was mostly a sampler artefact. With training that covers the observing system, the ODE's own dispersion is ≈ calibrated. Early-fine Euler steps (the benchmark default since #253/#257) recover most of it at no cost, and posterior *shape* is right for VanillaCFM. The remaining calibration gap is PSC-specific (§3).
- **Negative results worth stating:**
  - over-weighting τ=0 memorises;
  - stochastic sampling reduces spread;
  - a single-probe second-order loss is biased toward shrinking the Jacobian;
  - the Jacobian is not symmetric, i.e. not conservative, with no measured cost.
- **Seed noise:** identical configs differ by up to **10.6%** in ensemble-mean RMSE (P1 vs T0), consistent with P1's 15–21% checkpoint-selection noise. Single-run differences below that are not interpretable.

## 6. Open items (the plan's remaining work)

1. **PSC skip-connection test (§3):** the one controlled experiment that would confirm or refute the identified failure. A one-line model change plus one training run and evaluation.
2. **P1 paper C4 rewording (§4):** proposed text in `l96_rank_histograms_c4.md`; authorial decision.
3. **Corrected T5 (two-probe unbiased product, λ ramp): low priority.** For the L96B flows the operator share of under-dispersion is small, and a diagonal, time-averaged constraint barely sees the late-τ structure in §3.
4. **Coverage:** everything here is S0; S1 and QG were not probed.

## Caveats

- One checkpoint per model for the P1 recipe; 3 seeds for L96B at 400 epochs; 1 seed at 1200 epochs.
- The probe identities (E1, NS1, B4) are evaluated on the regular-grid test set, which lies inside the L96B training distribution. The random-layout set reproduced every pattern it was checked on (#256).
- The mechanism in §3 is an interpretation of consistent measurements. It is not a controlled result until the §6.1 test runs.
