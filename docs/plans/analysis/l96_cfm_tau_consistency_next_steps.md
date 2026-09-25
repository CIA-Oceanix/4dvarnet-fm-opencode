# τ-consistency of the CFM operator — next steps

**Status:** **CLOSED 2026-09-25** — outcomes and open items in `docs/results/l96_cfm_tau_consistency_synthesis.md`. Last revision: v4 2026-09-24. v4 re-targets the plan at
**defect 2 (variance collapse)**, after v3's Batch 1 came back negative. v3's
material that is still relevant is summarised in §1; the full v3 text is in git
history (#248).

**Version history:**
- **v1** (#245): probes, reseed, low-τ sampling, bootstrapped targets.
- **v2** (#246): tower-property derivation, truth-marginal test NS1, split T2a/T2b.
- **v3** (#248): T1′ control, shared pair construction, parallel execution.
- **v4 (this):**
  - records the v3 outcomes;
  - adds the **second-order consistency identity** as the organising principle for defect 2;
  - adds a **no-retraining sampler test (S1)** first, then the **second-order training arm T5**;
  - keeps **T2a′** as an optional last test of the first-moment line.

Property labels (A1, B4, D1, E1, NS1a–c) are those of
`docs/results/l96_cfm_tau_consistency.md` and `docs/results/l96_cfm_tau_consistency_ns1.md`.

---

## 1. Where v3 left us

| item | outcome | record |
|---|---|---|
| Defect 1 — biased τ=0 prediction | **First-moment consistency holds on the training distribution from τ ≈ 0.2**: NS1a ≈ 0, NS1b ≈ 0 for s ≥ 0.2. The defect is confined to τ ≲ 0.1, most strongly in the slow channels | `cfm_tau_consistency_ns1.md` (#250) |
| Batch 1 (T0, T1′, T2a) | **Negative.** Extra τ=0 training mass makes the network memorise each training window's `x1` (1000 windows). Both arms are worse than their paired T0 (ensemble-mean RMSE +19% / +10%) and more under-dispersed. The teacher target beats `x1` at equal mass (T2a > T1′, −7.5% RMSE) but cannot compensate. Seed noise (P1 vs T0, identical configs) is **10.6%** | `cfm_tau_consistency_batch1.md` (#251) |
| Defect 2 — variance collapse | B4 decays ≈ 2.6–2.7× along the ODE on both models. NS1c shows the operator's **second-order** identity is violated on true marginals too, with a model-dependent sign: PredictStateCFM's Jacobian is 12× too small at τ = 0.9, VanillaCFM's 1.1–1.9× too large. So the collapse is **partly operator, partly sampler** (the ODE's own marginals), in unidentified proportions | `cfm_tau_consistency_ns1.md` §3 |
| Cheap point estimator | 3 steps × 8 members (24 calls) is within 2.6–4.7% of the 300-call ensemble mean. The member-averaged prediction at τ ≈ 0.3–0.4 beats the endpoint by 2.5–5% in the seed-1 runs | #250, #251 |

**Lessons carried forward:**
1. **Do not add training mass at a fixed τ.** At 1000 windows that becomes memorisation. Every v4 arm leaves the τ-sampling density unchanged.
2. **Two seeds per arm minimum**, compared against a T0 seed pair. Differences below ~10% are noise.
3. **A consistency metric can improve because the whole operator degrades.** Every arm is judged on RMSE, CRPS and spread first; consistency metrics only explain the result.

---

## 2. Defect 2 in one identity

For the exact operator `D = E[x1 | x_tau, y]`, with `b_tau = (1 − tau) s0`, second-order Tweedie gives

  `Cov(x1 | x_tau, y) = (b_tau² / tau) · ∇_x D(x_tau, tau, y)`,

and two consequences follow:

- **B4 (law of total variance):** `P(y) = Var_{x_tau}(D) + (b_tau²/tau)·E[∇_x D]` must be flat in τ. On the ODE's states it decays ≈ 2.7×.
- **NS1c (second-order MMSE identity on true marginals, per coordinate):** `E[(x1 − D)_i²] = (b_tau²/tau)·E[∂D_i/∂x_i]`. The ratio of right to left side should be 1; the measured ratio is PredictStateCFM 0.9 → 0.08 over τ ∈ [0.1, 0.9], and VanillaCFM 1.1–1.9.

**Why the CFM loss cannot fix it.** The loss regresses `D` onto `x1`, so it trains only the first moment. `∇_x D` — how strongly the prediction reacts to `x_tau` — is left unconstrained. It is the only carrier of the remaining uncertainty, and it is also what keeps ODE members apart: a small Jacobian makes each Euler step contract member differences. A finite-data regressor is pulled toward shrinkage, i.e. a small Jacobian.

**Two levers, in the order below:**
1. **Sampler (S1):** stop the ODE from accumulating contraction by re-injecting noise. This acts on the marginal part and costs no training.
2. **Operator (T5):** train the NS1c identity. It is **per-sample trainable from a single `x1`**, unlike the per-`y` B4 (v3's T3), because it needs no posterior samples.

---

## 3. Steps

### S1 — stochastic sampler on the existing P1 checkpoints (no retraining; **done**)

> **Outcome (2026-09-24, `docs/results/l96_cfm_sampler_schedule.md`): the "mixed" row of the decision rule below.**
> - **Stochasticity (η > 0) lowers the spread** on both models: rejected.
> - **With uniform Euler steps, spread/RMSE saturates with N**, at ≈ 0.64 (PSC) and ≈ 0.81 (Vanilla). The gap from there to 1 is the operator's share.
> - **Fine early steps are a free win at the current cost.** With the schedule `tau_k = 1 − (1 − k/N)^0.5`, N = 10: CRPS −4.4% (PSC) / −2.0% (Vanilla), RMSE slightly better, spread/RMSE 0.53 → 0.61 / 0.66 → 0.73.
> - Hence **T5 stays the main lever**, and p = 0.5 becomes T5's second evaluation sampler (§4).
>
> The original S1 specification follows.

A DDIM-η family on the linear path (`reports/l96/probe_stochastic_sampler.py`).
One step s → τ, with `D = D(x_s, s, y)` and `eps_hat = (x_s − s D)/b_s`:

  `x_tau = tau D + sqrt(b_tau² − sigma²)·eps_hat + sigma·xi`, with `sigma = eta·sqrt(v)`,

where `v` is the variance of `q(x_tau | x_s, x1)` under the forward-noising kernel.

- **η = 0 is exactly the current Euler step** (tested).
- **η = 1 is the ancestral sampler**; at s = 0 it draws fresh noise.
- **Every η keeps the path marginal when `D` is exact** (tested), so η trades the ODE's accumulated contraction against injected noise.

**Grid:** `(N, η)` ∈ {10} × {0, 0.25, 0.5, 0.75, 1} ∪ {20} × {0, 0.5, 1}, with 30 members, S0, on P1 PredictStateCFM-M and VanillaCFM-M. It reports ensemble-mean RMSE, spread, spread/RMSE, fair CRPS and single-member RMSE, per group. It runs as SLURM job 55216.

**Decision rule:**

| outcome | meaning | next |
|---|---|---|
| spread/RMSE → ≈ 1 at some η, with RMSE within 2% and CRPS better | the collapse is mostly sampler-side | **Ship η** as the flows' sampler (a P1 benchmark update). T5 is optional, only for the residual operator part |
| spread/RMSE rises but stays ≪ 1, or RMSE degrades > 2% before calibration | mixed: the operator limits what noise can do | run T5; use the best η as T5's evaluation sampler |
| η changes nothing | the collapse is operator-side | T5 is the main lever |

The flat-in-τ B4 check is reported for the chosen η too, via `probe_tau_consistency.py` once it takes the sampler as an option (a small follow-up).

### T0b — second baseline seed (1 run)

`A2_predictstatecfm_monaiM_l96` with `++training.seed=2`. With P1 and T0, this gives three baseline runs, the minimum for a noise band. The sbatch pattern is the existing `run_l96_a2_ps_m_seed1_train.sbatch`.

### T5 — second-order (variance) consistency loss (4 runs: 2 values of λ × 2 seeds)

> **Outcome (2026-09-24, `docs/results/l96_cfm_tau_consistency_t5.md`): negative as implemented.**
> - λ = 10 is inside the noise band, with no calibration change; it scaled the Jacobian term *down* ≈ 10–15% at every τ.
> - λ = 100 diverged at switch-on.
> - Cause: the single-probe squared loss is biased; its expectation adds the Hutchinson variance, which rewards shrinking the whole Jacobian.
> - A corrected T5 (two-probe unbiased product, λ ramp, optionally an autograd JVP) is specified in §5 of that note. It is not launched and needs a decision.


- **Loss:** `L = L_CFM + λ · L_var`, where

  `L_var = mean_c ( mean_t [ (b_tau²/tau) · u ⊙ (J u) − sg(x1 − D)² ] )²`

  - **Per-channel trace matching:** the Hutchinson diagonal `u ⊙ J u` (Rademacher `u`), averaged over time within each channel, then squared and averaged over channels.
  - Its expectation vanishes exactly when the NS1c identity holds per channel. That is needed because NS1c's slow/fast errors differ several-fold.
  - The target uses the residual **directly** (`|x1 − D|²`, not `(u·r)²`), which removes probe noise from the target.
  - `sg` = stop-gradient: the mean is trained only by `L_CFM`, and `L_var` shapes only the Jacobian.
- **`J u`:** central finite differences (ε = 1e-2, validated against autograd in #245): two extra forward passes, differentiated through.
- **Where:** rows with τ ∈ [0.05, 0.95] of the **unchanged** uniform-τ batch, so no extra mass anywhere (lesson 1). Rows outside the range get `L_var = 0`.
- **Scale:** both sides are per-coordinate variances (O(P) ≈ 0.05 normalized), so `L_var` is O(P²). Take λ ∈ {10, 100} to start, so that `λ·L_var` is 10–50% of `L_CFM` at initialisation (to be checked in the smoke run and adjusted before launch).
- **Monitoring:** log the NS1c ratio on validation batches (`val_var_ratio`, per group) next to `val_loss`. Checkpoint selection stays on `val_loss`.
- **Cost:** ≈ 2–3× per training step, i.e. ≈ 1.5–2 h per run on an A100.
- **Code** (extends #249's options; off by default, bit-identical when off):
  - `var_weight`, `var_tau_min`, `var_tau_max`, `var_fd_eps` on `PredictStateCFM`;
  - the loss term in `compute_loss`;
  - `val_var_ratio` logging in `LitModel`;
  - configs `A2_predictstatecfm_monaiM_var{10,100}_l96.yaml`;
  - tests: bit-identical when off, the finite-difference Jacobian term agrees with autograd, and on a linear-Gaussian toy the loss vanishes at the exact operator.
- **Risks:**
  - The residual target is a noisy per-window quantity. Averaged over 3000 steps it is far less noisy than a single `x1` at τ=0, but memorisation is possible, so watch `val_loss` against T0.
  - `L_var` could also be lowered by shrinking the residual (the mean) rather than fixing the Jacobian. That is blocked by the stop-gradient on the target.
  - It fixes only the operator part. S1 tells how large the sampler part is.

### T5b — full probe-vector second-order loss (deferred)

`‖(b_tau²/tau) J u − r (r·u)‖²` with `r = sg(x1 − D)`. It matches the full covariance, not just its diagonal, but its target is rank-one and much noisier. Run only if T5 moves NS1c to ≈ 1 but spread/RMSE stays low, which would mean the off-diagonal structure matters.

### T2a′ — first-moment line, last test (optional; 2 runs)

Teacher targets on the rows the uniform draw already places at τ ∈ [0, 0.1] (relabel, don't add), τ' ~ U[0.2, 0.5], EMA teacher from epoch 50. This needs a `boot_mode: relabel` option on #249's code. Run only if you want the first-moment question closed. The v4 default is to stop that line: the τ=0 bias is ≈ 7–11% on the point estimate and is already bypassed by the cheap 3×8 estimator.

---

## 4. Evaluation (every run)

1. The P1 `ens30_no10` protocol with the **Euler** sampler (for comparability), plus the **S1 early-fine sampler** (η = 0, N = 10, p = 0.5).
2. `probe_ode_mean_along_path.py`, `probe_tau_consistency.py`, and the truth-marginal probe from #250 (NS1a–c).

**Success criteria for T5** vs the T0/T0b pair (plus P1):

| metric | target |
|---|---|
| spread/RMSE (P1 convention) | from ≈ 0.38 toward 1 |
| CRPS / Energy Score | better than the baseline band |
| ensemble-mean RMSE | inside the baseline band (guard-rail) |
| NS1c ratio | toward 1 at every τ, per group |
| B4 flatness `B4(0.1) / B4(0.9)` | from 2.7 toward 1 |

---

## 5. Execution and cost

| step | depends on | compute |
|---|---|---|
| S1 sampler grid | nothing (running) | ≈ 1 A100-hour |
| T0b reseed | nothing | 40 min A100 |
| T5 code PR | nothing (in parallel with S1) | dev + CI |
| T5 runs (4) | T5 code | 4 × ≈ 2 h A100, in parallel |
| T5 evaluation | T5 runs, and S1 for the η | 4 × ≈ 1 h |
| T2a′ (optional) | a `boot_mode` option | 2 × 45 min |

**Gate:** the S1 outcome decides between "ship η" and "T5 is the main lever"; it does not block writing the T5 code. As before, PRs are based on master, never stacked.

## 6. Decisions needed

1. **After S1:** adopt the early-fine schedule (p = 0.5, same N) as the flows' default sampler in the P1 benchmark? It improves CRPS and RMSE on both P1 flows at no extra cost, but it has been verified on one checkpoint per model only.
2. **T5:** launch 2 values of λ × 2 seeds plus T0b (proposed), or a single λ first?
3. **T2a′:** run it, or close the first-moment line as recorded (proposed: close)?
4. **Paper home** (unchanged): the mechanism and self-conditioning go to P2; the variance-collapse diagnosis, S1 and T5 go to P1's calibration section.
