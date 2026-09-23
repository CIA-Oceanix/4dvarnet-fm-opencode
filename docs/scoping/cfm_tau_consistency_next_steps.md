# τ-consistency of the CFM operator — next steps

**Status:** DRAFT for review, **v2 2026-09-23**. Nothing implemented. Follows
`docs/results/cfm_tau_consistency.md`; section references (§2, §3) and property
labels (A1, B1, B4, D1, E1…) are that doc's.

**What changed from v1.** v2 re-derives the plan from the tower property. That
separates what is *measurable* from what is *trainable*, and it changes three things:
- A truth-marginal diagnostic (NS1) now comes before any training. It separates "the operator is inconsistent" from "the ODE's states have the wrong distribution", a caveat that v1's probes could not resolve.
- The bootstrapped-target arm is split. **T2a** (τ=0 only) is the cheapest direct attack on defect 1; **T2b** is the general version.
- The per-`y` tower penalty is ruled out as not trainable (§2). A structural arm for A1 (T4) is added as a fallback.

v1's T1 (low-τ sampling) and T3 (B4 penalty) are unchanged.

---

## 1. The two defects, restated through the tower property

The exact operator `D(x, tau, y) = E[x1 | x_tau = x, y]` satisfies the tower property
at every level of conditioning:

- **Marginal form (B1):** `E_{x_tau ~ p_tau(·|y)}[D(x_tau, tau, y)] = m(y)` for every τ.
- **Pairwise form (D1, the martingale property):** for `s < tau`, where `x_s` is a further-noised copy of `x_tau`, `E[D(x_tau, tau, y) | x_s, y] = D(x_s, s, y)`.
- **Second moment (B4, the law of total variance):** `P(y) = Var(D(x_tau)) + E[Cov(x1 | x_tau, y)]`, and the last term equals `(b_tau²/tau)·E[∇_x D]`.

Measured (`docs/results/cfm_tau_consistency.md` §3), on both P1 M-tier flows:

1. **Defect 1 — a biased τ=0 prediction.** B1 drifts by 0.20 RMSE between τ=0 and the endpoint. The E1 slope is ≈ 1 where it should be 0. The flow corrects the τ=0 estimate within 1–2 Euler steps.
2. **Defect 2 — variance destroyed along τ ∈ [0.1, 0.4].** B4 decays 2.6–3.2× where it should be flat. The transport is self-consistent (B2), so this is where the ensemble under-dispersion (spread/RMSE ≈ 0.4–0.55) comes from.

**The unresolved caveat.** B1 and B4 were evaluated with `x_tau` from the model's
own ODE, not from the true `p_tau(·|y)`. A violation therefore mixes two sources: the
operator's inconsistency, and a wrong distribution of the ODE's states. B2 points to
the operator, but only indirectly. NS1 resolves it.

---

## 2. What the tower property allows us to train

Each training window supplies one `(x1, y)` pair, i.e. **one** draw from `p(x1|y)`.
That settles which forms can become losses:

| form | needs | trainable? |
|---|---|---|
| B1 per `y`: mean over x_τ of `D(x_tau)` = the same `m(y)` at every τ | several draws from `p_tau(·|y)`, hence several posterior samples of `x1` | **No.** Only with self-generated ODE samples: expensive, off-policy, and the samples inherit the defect being fixed |
| D1 pairwise: `D(x_s) ≈ sg D(x_tau)`, with `x_s` forward-noised from a data `x_tau` | one `x1`, two noise draws | **Yes, exactly.** Forward noising from a data `x_tau` gives the pair `(x_s, x_tau)` the correct joint law |
| D1 at `s = 0`: `D(x0, 0, y) ≈ sg D(x_tau', tau', y)` | one `x1`; `x0` independent | **Yes.** It is the τ=0 special case of the row above |
| B4 per `y` | K ≥ 4 draws of x0 per `y` plus a JVP | yes, but costly and noisy (T3) |

So the tower property's trainable form is the **pairwise, martingale** form. Every
intervention below is a variant of it, a change in how τ is sampled, or a
structural constraint.

---

## 3. Steps, in order

All training arms use **PredictStateCFM-M** with the exact P1 recipe:
- monai backbone, `data.normalize: true`, cosine LR, 400 epochs, batch 16, lr 1e-3, clip 10.
- Base config: `config/experiment/A2_predictstatecfm_monaiM_l96.yaml`, plus one change per arm.

PredictStateCFM outputs `D` directly, which keeps the target constructions simple.
New options all default to today's behaviour, so no existing config retrains.

### NS1 — truth-marginal tower test (no retraining; do first; ≈ 30 GPU-min)

Build `x_tau = tau·x1 + (1 − tau)·x0` from the **test-set truth** `x1`. This is the
exact distribution the network was trained on, so no ODE is involved. Then:

- **NS1a. Per-τ orthogonality curve on true marginals.** Regress the residual `x1 − D(x_tau^true, tau, y)` on `g(y) = ODE mean − D0`, pooled over windows, for τ ∈ {0, 0.1, …, 0.9}.
  - For an exact operator the slope is 0 at every τ: `g(y)` is a function of `(x_tau, y)`, so the pooled tower gives `E[D(x_tau) g(y)] = E[x1 g(y)]`.
  - At τ = 0 the curve is the measured E1 slope (≈ 1). As τ → 1 it goes to 0 trivially, since `D` → `x1`.
  - **The τ at which it falls to 0 is where the operator becomes consistent on its own training distribution, independently of the sampler.**
  - Regressing `D(x_tau) − D0` on `g` instead would be wrong: it inherits `D0`'s bias and reads ≈ 1 even for a perfect `D(·, tau > 0)`.
- **NS1b. Truth-marginal martingale test.** Forward-noise `x_tau^true` to `x_s` (s < τ). Regress `D(x_tau, tau) − D(x_s, s)` on features of `x_s` (`D(x_s, s) − D0` and `x_s` itself). Every slope must be 0.
  - This is D1 with no sampler at all, over a grid of (s, τ) pairs.
  - It also checks the construction T2b relies on.
- **Reading the outcome:**
  - NS1a falls to 0 by τ ≈ 0.1 and NS1b is ≈ 0: the operator is consistent on its training distribution except at τ = 0. Defect 1 is then purely a τ=0-prediction problem, which points to T2a/T1. The B4 decay would come from the ODE's states rather than the operator, which **reorders the plan: T2b drops, and a sampler-side check comes first.**
  - NS1a and NS1b clearly nonzero over [0.1, 0.4] (expected, given B4): the operator is at fault across low/mid τ. Proceed with T2a, then T2b.
- **Cost:** one new script (or a flag on `probe_tau_consistency.py`) and forward passes only.

### NS0 — carried over from v1 (no retraining; ≈ 1 GPU-hour)

- **NS0a. Deterministic path from `x0 = 0`.** Is the ODE gain refinement or ensembling? §2 predicts refinement.
- **NS0b. Martingale check under the model's own stochastic sampler.** Complementary to NS1b: the same identity, evaluated on the sampler's distribution.
- **NS0c. Accuracy vs cost of cheap point estimators.** An RMSE grid over N ∈ {1, 2, 3} steps × M ∈ {1, 4, 8, 30} members.
  - N = 2 already recovers two-thirds of the gain, so a 2-step, few-member estimator may be the practical point estimate for the P1 tables at ~1% of the ensemble cost.

### T0 — baseline reseed (1 run)

This is the P1 PredictStateCFM-M config with a new `training.seed`. The family's
checkpoint-selection noise is 15–21% (P1), so no single-run arm difference below
~5% is interpretable without it.

### T2a — bootstrapped τ=0 prediction (1 run; the direct attack on defect 1)

- For a fraction `f0 = 0.25` of each batch, set `tau = 0` and use the target `sg D_teacher(x_tau', tau', y)` instead of `x1`.
  - Here `x_tau' = tau' x1 + (1 − tau') x0'`, with `tau' ~ U[0.2, 0.5]` and `x0'` independent of the τ=0 input `x0`.
  - The rest of the batch keeps the standard loss.
- **Why it is unbiased.** By the tower property, `E[D(x_tau', tau', y) | y] = m(y)`: the minimiser is still `m(y)` if the teacher is exact.
- **Why it should help.** The target's spread about `m(y)` falls from `Var(x1|y)` to `Var(D(x_tau'))`.
  - Measured on the current model at τ' = 0.2 (member variance of D ≈ 0.0155 vs `P` ≈ 0.054, normalized), that is **≈ 3.5× lower target variance**.
  - The training set has 1000 windows, and the τ=0 prediction learns `m(y)` from one noisy `x1` per window. Soft targets from a teacher that is already good at τ' ∈ [0.2, 0.5] (§2) supply many effective draws per window. This is the distillation / Stable-Target-Field argument.
- **Teacher:** an EMA of the online weights (decay 0.999) if `LitModel` supports one — check first; otherwise stop-gradient of the online network. Warm-start: switch on after 50 epochs.
- **Risk:** teacher bias is self-reinforcing. Mitigated by the τ' range (where the model is best), the warm-start, and an E1 check on the 100-epoch checkpoint.

### T1 — low-τ sampling density (1 run; unchanged from v1)

- `tau ~ (1 − f)·U[0,1] + f·U[0, tau_max]`, with `f = 0.5`, `tau_max = 0.4`.
- Schema: `tau_sampling: uniform | low_mix`, `tau_low_frac`, `tau_low_max`, all defaulting to today's behaviour.
- A capacity-allocation control for T2a: it tests whether *more* low-τ training fixes the bias, versus *better targets* at low τ.

### T2b — general pairwise martingale target (1–2 runs; conditional on NS1b and T2a)

- For a fraction of each batch with `tau < tau_b = 0.4`: build `x_tau'` from data with `tau' = tau + Δ`, Δ ~ U[0.1, 0.3]. Obtain `x_tau` by forward-noising in `z = x/tau` coordinates: `z_tau = z_tau' + sqrt(λ_tau² − λ_tau'²) ξ`, with `λ = (1 − tau) s0 / tau`.
- Loss: `‖D(x_tau, tau) − [(1 − α) x1 + α·sg D_teacher(x_tau', tau')]‖²`, with `α ∈ {0.5, 1.0}`.
- **The arm aimed at defect 2.** It propagates the network's mid-τ behaviour down to [0.1, 0.4], where B4 collapses. T2a only fixes the τ=0 end.
- Run only if NS1b shows the pairwise identity broken over [0.1, 0.4]. Its first run uses the α that the T2a lessons suggest.

### T3 — B4 variance-conservation penalty (deferred; unchanged)

- A truth-free penalty on `(V(tau) − V(tau'))²`, with `V = Var_{x0}(D) + (b²/tau)·tr J / n`.
- Needs K ≥ 4 draws of x0 per window plus a finite-difference JVP: ≈ 3–5× the training cost.
- Only if T2a/T2b fix defect 1 but B4 still decays.

### T4 — structural A1 (fallback; not scheduled)

- Parameterise `D(x, tau, y) = m_phi(y) + c(tau)·r_theta(x, tau, y)` with `c(0) = 0`. The τ=0 output is then x0-independent **by construction**, and `m_phi` shares the trunk.
- **Caution:** the τ=0-only control (0.489, ≈ DirectUNet) shows that simply removing the x0 input loses the gain. The trunk sharing is what must preserve it.
- Only if T2a fixes the RMSE but A1 stays large, or for P2, where this is exactly the "mean slot + flow" architecture.

---

## 4. Evaluation (identical for every run)

1. P1 `ens30_no10`: RMSE, CRPS, spread/RMSE — S0 only.
2. `reports/l96/probe_ode_mean_along_path.py` and `reports/l96/probe_tau_consistency.py`, unchanged.
3. **NS1**, which is the cheapest and most direct readout of whether the operator became more consistent.

**Success criteria vs the T0 reseed pair:**

| metric | target | aimed at |
|---|---|---|
| τ=0 prediction RMSE | closes ≥ 50% of the gap to the ODE ensemble mean | defect 1 (T2a, T1) |
| E1 slope; NS1a curve | E1 from ≈ 1 toward 0; the curve flat at 0 for τ ≤ 0.5 | defect 1 |
| NS1b slopes over τ ∈ [0.1, 0.4] | toward 0 | defect 2 (T2b) |
| B4 flatness, `B4(0.1) / B4(0.9)` | from 2.6 toward 1 | defect 2 |
| spread/RMSE (P1 convention) | rises from 0.40 | defect 2 |
| ODE ensemble-mean RMSE / CRPS | no regression beyond the T0 reseed spread | guard-rail |

**Reading the outcomes:**
- **T2a fixes defect 1 with the ODE unchanged:** a one-call point estimator as good as the 300-call ensemble mean.
- **T2a beats T1:** target variance, not capacity, was the problem. That is a clean, citable mechanism for P2.
- **B4 flattens only under T2b:** calibration is a mid-τ consistency problem. That goes to P1's calibration section.

---

## 5. Cost

- NS1 + NS0: about 1.5 GPU-hours, no training.
- T0 + T2a + T1: **3 P1-recipe training runs**.
- T2b: 1–2 more runs, conditional. T3 and T4 are not scheduled.

## 6. Decisions needed

1. **Run NS1 + NS0 now**, before committing to training? Proposed yes: NS1 can reorder everything below it.
2. **First training batch: T0 + T2a + T1** (proposed), or T2a alone?
3. **Seeds:** one per arm plus the T0 pair (proposed), or two per arm (≈ 2× cost).
4. **Paper home** (unchanged from v1): the self-conditioning mechanism and T2a go to P2 (`docs/scoping/p2_unrolled_solvers_and_flows.md`); the B4 variance-decay diagnosis and T2b go to P1's calibration section.
