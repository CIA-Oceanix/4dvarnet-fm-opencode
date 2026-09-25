# What a flow adds to an unrolled variational solver, and what it cannot — DA-venue scoping (P2)

**Status:** SCOPING v1 (2026-09-18), targeting **JAMES** as a sequel to the
4DVarNet line. Companion to `docs/plans/paper/multi_p1_structural_hypotheses.md`
(P1, the diagnosis paper) and `docs/plans/paper/multi_ml_prior_knowledge.md`
(the non-DA ML paper). §9 fixes the boundary between the three.

**S0 only.** This paper makes no claim about model error; see §5.0 for why that
is a design decision rather than a simplification.

Every figure below is from this repo's own L96 runs. Nothing is quoted from a
draft or preprint.

> **Read §7 before drafting anything.** A literature check on 2026-09-18 found
> that two of the three things this paper was going to claim are **already
> published**. The contribution below has been narrowed accordingly, and the
> word "unification" deliberately removed from the title.

---

## 1. The question

An unrolled variational solver (4DVarNet) and a conditional flow (CFM) are
usually presented as different kinds of object: one a learned optimizer, the
other a generative model. They are not. Both are **posterior-mean operators**,
and the flow is the `τ`-indexed family of which the solver is one member.

The questions this paper answers:

- **Q1.** What exactly does a conditional flow add to an unrolled variational
  solver, expressed as named operator components rather than as a benchmark delta?
- **Q2.** Can the two be combined *continuously* rather than by the usual hard
  warm start, and does that buy anything on both accuracy and dispersion?
- **Q3.** Which of those components can be manipulated post hoc, and which
  require retraining?

Q3 is the one with a negative answer, and it is the most useful result here.

---

## 2. The operator family

With the linear interpolant `x_τ = (1−τ)x₀ + τx₁`, `x₀ ~ N(0, σ₀²I)`, the flow
is determined by

```
Ψ(x_τ, y, τ) = E[x₁ | x_τ, y] ,      v = (Ψ − x_τ)/(1 − τ)
```

### 2.1 A deterministic estimator is a degenerate flow

Take any deterministic estimator `m̂(y)` and read it as an operator that ignores
its state argument, `Ψ_det(x_τ, y) = m̂(y)`. Then

```
v_det = (m̂ − x_τ)/(1 − τ)    ⇒    x_τ = (1 − τ)x₀ + τ m̂
```

exactly. **The SDEdit-style warm start is not a separate trick — it is the
trajectory of the degenerate flow whose operator is constant.** 4D-Var, OI,
DirectUNet and 4DVarNet all live in this class.

### 2.2 The partition, and what each method zeroes

`Ψ` admits the exact additive partition `Ψ = Ψ_mean + Ψ_G + Ψ_NG` (Tweedie),
with `Ψ_G` an affine gain term and `Ψ_NG` the residual. **See P1 and the ML
doc §2.2 for the orthogonality condition this rests on** — it forces an `L²`
inner product and `Ψ_G` defined as the `L²` projection.

| method class | `Ψ_mean` | `Ψ_G` | `Ψ_NG` |
|---|---|---|---|
| 4D-Var / OI / DirectUNet / **4DVarNet** | `x̂(y)`, constant in `x_τ` | **0** | **0** |
| EnKF / ETKF / EnKS | ensemble mean | ✓ (the Kalman gain *is* `K_τ`) | **0 by construction** |
| conditional flow / diffusion | ✓ | ✓ | ✓ |

This is the paper's organizing table: it says precisely what an unrolled solver
is missing, and that what it is missing is *two named things*, not one.

**Boundary conditions — `Ψ_NG` vanishes at both ends, exactly.** The ML doc
records `K_0 = 0`, `K_1 = I` but stops there. The consequence for the
non-Gaussian term is two lines and is the architecturally useful part:

- **`τ = 0`:** `x₀ ⊥ x₁`, so `Ψ = E[x₁|x₀,y] = E[x₁|y] = μ`. With `K_0 = 0`,
  `Ψ_G = 0`. Hence **`Ψ_NG(0) = μ − μ − 0 = 0`.**
- **`τ = 1`:** `x_τ = x₁`, so `Ψ = x₁`. With `K_1 = I` and `β_1 = 1`,
  `Ψ_mean + Ψ_G = μ + (x₁ − μ) = x₁`. Hence **`Ψ_NG(1) = 0`.**

So the non-Gaussian branch is a *bump* on `(0,1)` pinned to zero at both ends.
Any parameterization that lets a network spend capacity there is spending it
where the answer is already known — and, worse, permits the model to be wrong
where it is certain. §2.4 and M4 act on this.

### 2.3 The blend

Velocities are combinable, so the two classes can be mixed continuously:

```
v_λ = (1 − λ(τ)) v_FM + λ(τ) v_det ,      λ(τ) = λ₀ (1 − τ)^p
```

**`p ≥ 1` is forced, not chosen.** `v_det` carries a `1/(1−τ)` factor, so the
blend contains `λ(τ)/(1−τ) = λ₀(1−τ)^{p−1}`, bounded as `τ → 1` only for
`p ≥ 1`. The hard warm start is the discontinuous member
`λ(τ) = 1{τ < τ₀}`.

### 2.4 What the current architecture cannot represent

Two gaps follow directly from §2.2, and each is a testable prediction rather
than a preference.

**(a) The unrolled solver is `τ`-blind, so `K_τ` is not representable.**
`FourDVarNetPredictStateCFM` (the `FDV1CFM` row) already does the right *outer*
thing — it computes `μ = E[x₁|x_τ,y]` at **every** flow time, by `K_inner`
weight-tied refinement steps started from the current `x_τ`, rather than freezing
a single `m̂(y)`. But its update rule never sees `τ`:

```python
def forward(self, x_t, batch, tau):        # tau is accepted...
    for k in range(self.K_inner):
        tau_k = torch.full((x.shape[0],), k / denom, ...)   # ...and never used
        gmod = checkpoint(_solver_iteration, self.unet, ..., tau_k, tau_k, ...)
```

The network is conditioned on the **inner iteration index** `k`, not the flow
time; the class docstring calls this "a documented simplification", with `tau`
accepted "only for interface parity". So `τ` enters **only through the
initialization** `x = x_t`, never through the operator.

Since the exact gain `K_τ = β_τΣ(β²_τΣ + α²_τI)⁻¹` runs from `0` to `I`, a
`τ`-blind update rule **cannot represent it** except indirectly through where it
starts. *Possible corroboration already in hand:* the closed-form gain matches
the fitted optimum to **~1.5% for V3** (whose net carries a standard CFM time
embedding) against **~4% for FDV1CFM** (whose net does not) — nearly 3× worse in
exactly the model that cannot see `τ`. **This is a hypothesis consistent with a
two-checkpoint comparison, not an established attribution**; other differences
exist and M4a is what tests it.

**(b) `Ψ_NG` is unconstrained where it is exactly zero.** The affine doc's §5
proposal,

```
v_θ = a_θ(τ)·μ_φ(y) + b_θ(τ)·x_τ + c_θ(τ)·HᵀR⁻¹(y − Hx_τ) + r_ψ(x_τ,τ,y)
```

makes the *affine* part `τ`-dependent but leaves the residual `r_ψ` free at the
endpoints, where §2.2 shows the answer is `0`.

---

## 3. Claims

- **C1 (Q1).** An unrolled variational solver is the `Ψ_mean`-only member of the
  flow family; a flow adds exactly `Ψ_G + Ψ_NG`. The deficit of each DA method
  class attributes to named components (§2.2).
- **C2 (Q2).** The continuous blend **strictly dominates** the hard warm start it
  replaces on **both** accuracy and dispersion — one scalar schedule, no
  retraining.
- **C3 (Q3, negative).** `Ψ_NG` is defined *relative to* the pair `(μ_p, K_p)`
  and is **not invariant** under a change of gain. The components are a
  **read-out, not a design space**: re-using a trained `Ψ_NG` beneath a different
  affine part is inconsistent, and the inconsistency dominates. Real decoupling
  requires retraining `Ψ_NG` against a prescribed `(μ, K)`.
- **C4.** Putting the unrolled solver in the `Ψ` slot is the right place for it,
  and the measured non-affine fraction bounds what the flow can add.
- **C5 (constructive, and the paper's one architectural claim).** The
  decomposition is not only a read-out — it **prescribes** a parameterization.
  Making the solver `τ`-aware lets `K_τ` be represented at all, and writing the
  residual as `Ψ_NG = g(τ)·r_ψ` with `g(0) = g(1) = 0` imposes §2.2's exact
  boundary conditions by construction instead of hoping they are learned. C3 says
  the components cannot be recombined *post hoc*; C5 says they can be *trained*
  jointly with the known structure built in.

---

## 4. Evidence already in hand

All S0, L96, 200 windows unless stated.

### 4.1 The decomposition is validated on trained checkpoints

From `docs/results/l96_cfm_affine_velocity_decomposition.md` (on `master` since #211),
on two trained checkpoints (V3, FDV1CFM):

- the closed-form gain matches the least-squares optimum to **~1.5% (V3) / ~4%
  (FDV1CFM)** from a single parameter;
- the structural constraint `(1 − β_τK_τ)` holds to **<0.015**;
- the predicted peak `τ* = √(s/(s+p))` matches the observed argmax on both.
- **`Ψ_NG` is 19–31% of velocity amplitude**, roughly flat over `τ ∈ [0.1, 0.8]`,
  rising sharply past 0.9 — the empirical justification for a non-Gaussian branch
  existing at all, and the bound behind C4 (payoff ≈ 3×, not 5×).
  - **Caveat that matters for §2.2's boundary conditions.** That figure is
    **velocity-relative**, and velocity carries the `1/(1−τ)` factor, so near
    `τ = 1` what is reported is `Ψ_NG/(1−τ)` — a `0/0`. The rise past 0.9 is
    therefore **not** evidence against `Ψ_NG(1) = 0`: a ratio can grow while its
    numerator vanishes, if the denominator vanishes faster. It does mean **the
    existing probe cannot test the constraint**, which is what M4c is for. The
    affine doc already half-suspects the parameterization here — "never regress
    `v` directly, or the `1/(1−τ)` inflates the loss near `τ = 1`".

### 4.2 The blend is a measured Pareto improvement (C2)

From `reports/l96/outputs/l96_fm_sampler_benchmark.md` (#211), `N = 30`,
`N_outer = 50`, 200 windows:

| config | rmse_repo | spread | spread/RMSE (target 0.967) |
|---|---|---|---|
| mean-only | 0.4714 | 0.0000 | 0.000 |
| cold (λ₀ = 0, pure FM) | 0.5295 | 0.6833 | 1.202 |
| warm (τ₀ = 0.3) | 0.3978 | 0.1211 | 0.284 |
| **blend λ₀ = 0.5, p = 2** | **0.3764** | 0.2369 | **0.585** |

Against its own warm-start baseline at identical settings the blend improves
**both** axes: RMSE **−5.4%** with spread **+96%**. This is the one Pareto
improvement in the whole body of work and it holds at both ensemble sizes tested.

### 4.3 The negative that carries C3

Decoupling the mean, gain and non-Gaussian amplitude into separate knobs is
**worse on both axes** than the coupled blend (`decoup a0=.5,p=2`: 0.6117 /
spread 0.1913, against the blend's 0.3898 / 0.2224 at matched settings). The
cause is structural, not an implementation fault: `Ψ_NG` is the residual relative
to the network's *own* `(μ_p, K_p)`.

The same code path reproduces the blend rows exactly, which is what establishes
this as a genuine negative result.

### 4.4 Five further negatives, each diagnosed

#211 §6 carries five more, each implemented, run and rejected with a diagnosis —
variance-corrected warm start (negligible: the missing term is ~8%); operator
mean-shift (the algebra is output-only, and reduces exactly to post-hoc
addition); post-hoc recentering + diagonal covariance (works, but is a two-stage
construction, not a sampler); low-rank covariance (rank-deficient at `N = 6` in
72000 dimensions); Desroziers calibration (invalid because the estimator is
*trained to fit the observations*: measured `dep_fit = 0.147` against a known
`R = 0.163`); observation cross-validation (over-shoots ~15×).

**These are the paper's substance as much as C2 is.** Several follow from the
operator decomposition rather than from bugs.

### 4.5 The honest state of the sampler

**No scheme measured is simultaneously accurate and calibrated.** The only
configuration carrying a full non-Gaussian branch is ~20–25% over-dispersive
(ratio/target 1.20 / 1.22 / 1.24 at `N` = 6 / 10 / 30 — stable in `N`, so a
sampler property), while everything that closes the RMSE gap collapses to
0.28–0.59 against a target of 0.967. **Caveat:** calibration was measured at a
single `Γ = 1e-2` and `N_outer = 50`, so the over-dispersion is confounded with
guidance strength. The `Γ × N_outer` sweep is P0 here as well as in the ML doc.

---

## 5. Experiments

### 5.0 Why S0 only

This paper's claims are about **operator structure**: which components a method
represents, and whether they can be recombined. None of that involves model
error. Restricting to S0 is therefore not a weakening — it is what keeps the
claims clean, and it immunizes the paper against P1's R1, the S0/S1 asymmetry
that is the largest liability in the whole programme. **Model error is P1's
axis; this paper does not touch it.**

The shift axis here is **observation sparsity**, which is S0-compatible and for
which the machinery and the augmented-training results already exist.

### 5.1 Experiments

**M0 — `Γ × N_outer` guidance sweep.** *Prerequisite.* Until it exists, §4.5's
calibration numbers rest on one operating point and cannot be attributed to the
prior rather than the guidance.

**M1 — the taxonomy, measured.** For each method class of §2.2, construct `Ψ̃`
and report `S(Ψ_mean)`, `S(Ψ_G)`, `S(Ψ_NG)`. Reuses the existing probe
(`reports/l96/probe_cfm_affine_decomposition.py`). Report on **two axes** — a
point-estimate score cannot price the non-Gaussian branch.

**M2 — the mean-slot comparison (moved here from the ML doc).** Hold sampler,
objective, parameter count and NFE fixed; vary **only** what computes the
posterior mean — generic UNet vs obs-conditioned unrolled vs
gradient-conditioned variational — under **observation sparsification**. The
ingredients are existing benchmark rows (`FDV1CFM`, `FDV1+FDV1CFM`,
`CFM-M(monai,cos)`); what is missing is the *matched-compute* version.

**M3 — blend schedule sweep.** `(λ₀, p)` beyond the four points measured, and
the crossover against the hard warm start as a function of observation density.

**M4 — the decomposition-aware architecture (C5).** The constructive answer to
C3 and the paper's one piece of genuinely new modelling. Three separable pieces,
in increasing cost:

- **M4c — re-probe `Ψ_NG` at the `μ` level, not velocity-relative.** *Cheapest,
  do first.* The probe exists; this is a change of normalization. Check both
  endpoints empirically. Either outcome is publishable: the constraint is
  satisfied, and the `τ → 1` blow-up is a pure parameterization artifact; or it
  is violated, and the network is wrong where the answer is exactly known — which
  is the motivation for M4b.
- **M4a — make the solver `τ`-aware.** Feed the outer flow time to the update
  rule alongside (or in place of) the inner index `k/(K_inner−1)`, retrain
  FDV1CFM, re-run the gain probe, and test whether the fitted-vs-closed-form gain
  error falls from ~4% toward V3's ~1.5%. Directly tests §2.4(a).
- **M4b — impose the `Ψ_NG` envelope.** Parameterize
  `Ψ_NG = g(τ)·r_ψ(x_τ,y,τ)` with `g(0) = g(1) = 0` (e.g. `τ(1−τ)`, `sin πτ`),
  on top of the affine doc's §5 form. Tests whether spending no capacity where
  the answer is known buys accuracy, calibration, or neither.

**Ordering matters:** M4c is a measurement and gates the motivation for M4b;
M4a is independent of both and is the cheaper of the two training runs.

**M5 — ADDA cross-check** (optional, cheap). Run ADDA's differentiable
strong/weak 4D-Var on L96 and confirm the repo's classical baselines agree. See
§7.4.

---

## 6. Risks

**R1 — novelty. This is the existential risk and it has grown since
2026-09-15.** The literature check in §7 found the Tweedie/Kalman identification
and the closed-form flow posterior covariance already published, and the
deterministic warm start published as a method. *Mitigation:* narrow to the
DA-side reading — the taxonomy of §2.2, the continuous blend, and the negatives —
and **cite rather than claim** the decomposition itself. A DA venue is essential:
the derivation half is textbook to an ML audience and new to a DA one.

**R2 — "you re-derived SDEdit."** The warm start is old and WSD (§7.2) published
the deterministic-warm-start method in 2025. *Mitigation:* the contribution is
not the warm start but that it is the **degenerate member of a family**, which
makes the continuous blend available; WSD is a hard cut with no velocity
interpolation, and explicitly excludes DA and chaotic dynamics.

**R3 — the 4DVarNet line already has UQ.** Ensemble-based 4DVarNet, SPDE priors
and a VAE-prior variational cost all exist (§7.3). *Mitigation:* this paper does
not propose a new route to UQ; it says **what kind** of uncertainty each route
supplies, in one operator language, and what the deterministic scheme is
structurally missing. That framing needs to be explicit from the abstract on, or
a referee from that line will read it as a competing method.

**R4 — single system.** Everything here is L96. *Mitigation:* QG is available at
moderate cost; decide in §10.

**R5 — the calibration confound** (§4.5). M0 is a prerequisite, not a nicety.

---

## 7. Prior art — literature check, 2026-09-18

*All URLs verified by retrieval on 2026-09-18. Verify author lists and venues
again before a bibliography exists.*

### 7.1 Already published: the decomposition itself

- **"Tweedie's Formula and Score-Driven Updating"** (arXiv 2605.15902) — the
  Kalman correction is an inverse-Fisher-scaled conditional-score correction;
  Tweedie, Kalman and score updates **coincide** in the Gaussian location model.
  **This is §2.2's `Ψ_G` identification.** Cite; do not claim.
- **"Divergence is Uncertainty: A Closed-Form Posterior Covariance for Flow
  Matching"** (arXiv 2605.00941) — closed-form posterior covariance under flow
  interpolants via a Tweedie identity, extended from diffusion to flow matching
  where the parameterization is a velocity rather than a denoiser. **Closest
  published relative of our gain derivation.**
- **"Support-Conditioned Flow Matching Is Kernel Smoothing"** (2605.13386) and
  **FlowDPS** (ICCV 2025) — flow-driven posterior sampling for inverse problems.

### 7.2 Already published: the deterministic warm start

- **"Warm Starts Accelerate Conditional Diffusion"** (arXiv 2507.09212) — a
  lightweight deterministic network predicts the **mean and diagonal variance**
  of `p(x₀|C)` to initialize sampling. Verified by retrieval: it is a **hard
  initial condition, not a continuous velocity blend**; its "warmth blending" is
  a *training-level* task weight, not a `τ`-schedule; testbeds are image
  inpainting and ERA5; it **explicitly excludes data assimilation and chaotic
  dynamics**.
  - *Point of contact worth reporting:* WSD predicts a variance as well as a
    mean. §4.4's variance-corrected warm start found that correction
    **negligible** in this regime (the missing term is ~8% of the noise
    variance). That is a small, citable empirical result against a published
    method's design choice.

### 7.3 The 4DVarNet line's own UQ work — the parent this paper extends

- **Beauchamp, Febvre & Fablet**, "Ensemble-based 4DVarNet uncertainty
  quantification for the reconstruction of sea surface height dynamics"
  (*Environmental Data Science*) — deep-ensemble 4DVarNet whose spread targets
  the posterior.
- **"Neural variational Data Assimilation with Uncertainty Quantification using
  SPDE priors"** (arXiv 2402.01855; *AIES* 4(3), 2025) — SPDE-based 4DVarNet
  extension for UQ; a VAE prior in the variational cost gives a probabilistic
  structure enabling posterior sampling.
- **4DVarNet-SSH** (*GMD* 16, 2119, 2023) and the original 4DVarNet
  learning-variational-solvers paper — the line this sequel attaches to.

**Consequence:** three routes to UQ for 4DVarNet already exist (deep ensembles,
SPDE priors, VAE prior). P2 must position as *explaining what each supplies*,
not as a fourth competitor. See R3.

### 7.4 Flow matching for DA, and infrastructure

- SDA (Rozet & Louppe), **FlowDAS** (2508.13313), **DAISI** (2512.00252,
  stochastic interpolants; L63 + QG turbulence + SEVIR, PF comparisons),
  **PnP-DA** (2508.00325), **ForcingDAS** (2605.14285), **Neural Incremental
  Data Assimilation** (2406.15076), 4DVarGAN (*JAMES* 2025).
- **ADDA** (arXiv 2608.23297, `github.com/m-dml/ADDA`) — modular, end-to-end
  differentiable DA framework: KF/RTS/EnKF/EnKS, strong- **and weak-constraint
  4D-Var**, latent DA, 10 systems incl. two-scale L96 and QG. **Assessment:** it
  is infrastructure, *not* an evaluation framework — its own paper says it is
  "not a benchmarking study", and it ships RMSE / error growth / filter
  divergence, **no CRPS, energy score, rank histogram or spread-skill**. So it
  does not close this project's measurement gap. Its value here is (i) a
  differentiable 4D-Var whose solver could literally be unrolled, which would
  make §2.1's identity *constructive* rather than asserted, and (ii) an
  independent cross-check of the classical baselines (M5). Early-stage (3 stars,
  62 commits, under review), so adopting it does **not** buy external-comparability
  credibility yet, and porting is not recommended.

### 7.5 The gap this paper occupies

None of the above uses the operator partition to **classify data-assimilation
methods** by which components they zero, identifies an **unrolled variational
solver as the degenerate member** of the flow family, schedules a **continuous
blend** between a deterministic and a generative flow with the `p ≥ 1`
boundedness constraint, or reports that **`Ψ_NG` is not invariant under a gain
change**. That is the contribution, and it is narrower than "unification".

---

## 8. Scope control

**Minimum viable: L96 only**, S0 only. M0 → M1 → M3 are cheap and mostly
assemble existing machinery; M2 is the matched-compute work; M4 is the single
new-modelling item.

**Phasing.** P0: **M0** (the guidance sweep — gates every calibration statement).
P1: M1 + M3 (taxonomy and blend schedule; both reuse existing probes and rows).
P2: M2, the mean-slot comparison, **and M4c** (a re-normalization of an existing
probe, so it does not wait on the training items). P3: M4a then M4b — the only
parts that could fail expensively, so they go last and the paper stands without
them.
P4: M5 / QG if §10.3 says yes.

**This is the nearest-term paper of the three.** It needs no Weak-4DVar, no
symmetric model-error arm, no new testbed — and §4 is already most of its
results section.

---

## 9. Boundary with the other two papers

| | testbeds | owns |
|---|---|---|
| **ML** (`ml_paper_exploiting_prior_knowledge.md`) | A Gaussian / B deconvolution / C Fourier-MRI. **No DA.** | is `p*` the best prior under budget (C1/E2) |
| **P1** (`da_paper_structural_hypotheses.md`) | L96, QG | H1/H2/H3; the **model-error** axis; marginal value of observations; C6 |
| **P2** (this doc) | L96, S0 | the operator family; the **observation-sparsity** axis at fixed model; the blend; the negatives |

Two boundaries that must not blur:

1. **Calibration appears in P1 and P2.** P1 owns *"DA's posterior
   characterization is structurally limited"* (diagnosis). P2 owns *"here is a
   sampler family, what it achieves and where it fails"* (method).
2. **Observation density appears in P1 and P2.** P1 owns the *density × model
   error* grid; P2 owns the *density sweep at S0*. **They must share one
   obs-density protocol** or the two papers' numbers will not be comparable —
   this is the apples-to-apples trap, and it is easy to fall into because the
   two sweeps look identical.

The ML doc's §10 decision 6 asks where the mean-slot comparison belongs. **This
doc answers it: here (M2)**, because it is inherently L96/QG and the ML paper is
deliberately non-DA. That decision should be closed in the ML doc.

---

## 10. Open decisions

1. **Title and framing.** "Unification" is not defensible after §7. Current
   working title frames it as *what a flow adds and what it cannot* — diagnosis
   again. Is that too modest for a methods paper, and if so what replaces it?
2. **How much of M4 belongs in v1?** M4c is a measurement and should be in
   regardless. M4a and M4b are training runs and are the only items that can fail
   expensively. *Leaning:* M4c in v1; M4a in if it lands (it is the cheaper run
   and tests a sharp prediction); M4b optional, and the paper stands without it.
   Note that C5 is the only claim making the decomposition *prescriptive* rather
   than diagnostic — dropping all of M4 would leave P2 a diagnosis paper, which is
   P1's genre, so at least M4c and M4a should survive.
3. **QG as a second system?** Adds external validity and a second geometry at
   moderate cost; L96-only is defensible for a mechanism paper but invites the
   single-system criticism.
4. **Is the ADDA cross-check (M5) worth it in v1**, given ADDA is early-stage and
   the benefit is a paragraph of external validity?
5. **Relationship to the 4DVarNet UQ line (§7.3) must be settled before the
   abstract is written** — explaining versus competing. This is R3 and it is the
   difference between a friendly and a hostile review from the nearest referees.
