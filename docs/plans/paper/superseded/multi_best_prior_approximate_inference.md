# Is the true prior the best prior? — scoping doc

**Status:** SUPERSEDED in framing — SCOPING, v1 (2026-09-16). **Superseded in framing by
`docs/plans/paper/multi_ml_prior_knowledge.md` (2026-09-16) — read that
one first; it is the current scoping document.** This doc's testbed ladder and
risk register were folded into it. Defines the scientific question, the testbed
families, and the experiment set. No implementation yet. Supersedes the
*framing* of `docs/plans/analysis/multi_phaseD_physics_priors.md`; Phase D's work
packages survive as the dynamical-systems instantiation and evidence base.

> **Question numbering differs between the two scoping docs — do not cross-cite
> by number.** Here the observation-configuration question ("does the optimal
> prior depend on `H`?") is **Q2**; in the ML-venue doc the same question is
> **Q3**, which inserts a new Q2 on the posterior-targeting objective. So §7.1's
> "Q2 — partially answered" and the ML doc's "Q3 partially answered" are the
> *same* finding on the *same* evidence
> (`l96_obs_density_augmented_training.md`), not two results.

---

## 1. The question

Inverse problem `y = H(x) + ε`, true prior `x ~ p*`.

Under **exact** inference the answer is settled and uninteresting: `p*` is
optimal, and the prior is independent of `H` (the observation process enters
only through the likelihood). The question acquires content only under
**approximate** inference — finite iterations, finite ensemble, finite sampler
steps, finite capacity:

- **Q1.** Is `p*` the best prior *to use* when exact inference is unavailable?
- **Q2.** Should the prior be independent of the observation configuration `H`?
- **Q3a.** When `p*` is a *known marginal* over model uncertainty
  (`p*(x) = ∫p(x|θ)π(θ)dθ`, π known), should one marginalize or condition on θ?
- **Q3b.** When the prior is *misspecified* (M-open; π wrong, or truth outside
  the model class), does the learned prior's advantage grow or shrink?

None of these is a violation of Bayes. All are statements about the
approximation-theoretic problem, not the statistical one.

**Ordering matters for reception.** Q1 is the surprise — a learned
representation winning *even when the true prior is exactly available and
exactly correct*. Q3b alone would reduce to "learned methods are robust to
misspecification," which is well known. Lead with Q1; Q3 establishes practical
relevance.

---

## 2. Why a testbed is hard to find, and what that buys us

The question requires a setting where `p*` is **known exactly and usable**.
Almost no ML inverse problem qualifies — on natural images one can only compare
learned priors to each other, so "is the true prior best?" is unaskable. That
constraint, not the application domain, should drive testbed selection, and it
is the paper's central affordance. State it in the abstract.

---

## 3. The mechanism, and where it is provable

**General statement (operator-agnostic).** Hard-constraining an estimate to an
exact forward model transfers model error and the operator's own
ill-conditioning into the estimate along the operator's **most-amplifying
directions**. Those directions are fixed by the operator's spectrum — not by
where observations are informative, nor by where accuracy is needed. A learned
prior is free to adopt a basis aligned with information instead.

In dynamical systems these are the unstable Lyapunov directions (see §7.2, AUS);
in general they are the small-singular-value directions of the forward operator.

**In the Gaussian family the mechanism is provable, not merely measurable.**
With `x ~ N(0,Σ)`, `y = Hx + ε`, `ε ~ N(0,R)`, the posterior mean is

```
μ = argmin_x  ½‖y − Hx‖²_{R⁻¹} + ½‖x‖²_{Σ⁻¹}
```

solved by CG on `A x = HᵀR⁻¹y` with `A = HᵀR⁻¹H + Σ⁻¹`. If `Σ` has spectrum
`σ_j ~ j^{−α}`, then `Σ⁻¹` has spectrum `~ j^{α}` and **using the exact prior is
precisely what makes the system ill-conditioned.** Reparameterizing
`x = Σ^{1/2} z`, `z ~ N(0,I)` gives the same optimum with a conditioning governed
by `HΣHᵀ + R` instead. At finite budget the two differ arbitrarily.

A flow trained on samples from `N(0,Σ)` implicitly learns `Σ^{1/2}` — i.e. **in
the Gaussian case, a generative representation of the prior is equivalent to
preconditioning the variational problem.**

*Honest accounting:* that equivalence is textbook (cf. centered vs. non-centered
parameterizations, Papaspiliopoulos–Roberts–Sköld). Family 1 is therefore the
anchor that makes the mechanism undeniable, **not** the contribution. The
contribution is that the phenomenon persists where no such equivalence is
available — non-Gaussian priors, nonlinear operators, chaotic dynamics — and that
it predicts *when* learned priors beat exact ones.

---

## 4. Testbed families

Deliberately not DA-only: the effect must be shown to be a property of the
**misalignment between `p*` and `H`**, not of chaotic dynamics or of any one
operator. Hence the design crosses prior family × observation operator.

| # | family | why it earns a place |
|---|---|---|
| **1** | **Gaussian random fields**, `x ~ N(0,Σ)`, `σ_k ~ k^{−α}` | exact posterior in closed form → gap to exact Bayes is measurable, not just error vs. truth; `α` is a continuous **conditioning dial**; the mechanism is provable here |
| **2** | **Non-Gaussian, exactly known**: Ising/Potts at known β; Bernoulli–Gaussian sparse in a known dictionary | tests survival beyond Gaussianity; `−log p*` usable directly as a regularizer; classical CS theory available |
| **3** | **Chaotic dynamical systems**: L63 / L96 / QG | nonlinear, temporally structured, dimension-scalable; the original motivation; all infrastructure exists |
| **4** | **Operator families** crossed with 1–3: deconvolution (known PSF), Fourier masks, limited-angle CT | varies `H` independently of `p*` — the cross is what makes the claim generic |

### 4.1 The prior-knowledge ladder instantiates in every family

| family | Q1 (exact) | Q3a (hierarchical, known) | Q3b (misspecified) |
|---|---|---|---|
| Gaussian field | Σ known | Σ known up to hyperparameters, prior on them | wrong α / wrong kernel |
| Deconvolution | exact PSF | PSF family, uncertain width | wrong PSF |
| Sparse / CS | known dictionary | dictionary with uncertainty | wrong dictionary |
| Dynamical | fixed θ | `θ ~ π` known (**= S0**) | biased θ (**= S1**) |

Q3 therefore costs no generality — it is the semi-blind/blind ladder that already
exists in each of these literatures, which also supplies natural baselines.

---

## 5. Experiments

All four are stated generically and instantiate in every family.

### E1 — Budget sweep: exact `p*` vs. a learned representation of the *same* distribution

**The headline.** Arms:

- **(a)** `p*` used exactly — as covariance, cost, constraint, or energy.
- **(b)** a flow / generative model trained on samples from `p*` — *correct
  distribution*, learned representation, obs-agnostic.
- **(c)** no prior (data-fit only), as a floor.

Sweep the inference budget `B` (CG iterations / ensemble members / sampler steps
/ solver unrolls, whichever the family's inference uses).

**Why this is the right experiment:** it isolates *representation* from
*distributional correctness* completely. Arm (b) has the same distribution as
arm (a), up to training error. Any gap is purely computational.

**It is self-validating.** As `B → ∞`, arm (a) *must* win — it is optimal under
exact inference. If it does not, that is a bug, not a result. The deliverable is
therefore the **crossover budget `B*`**: the budget below which a learned
representation of the very same distribution beats using that distribution
exactly.

### E2 — Conditioning sweep

Vary the ill-conditioning: `α` (Gaussian), the Lyapunov spectrum / `n₀`
(dynamical), OTF decay (deconvolution), dictionary coherence (CS).
**Prediction:** `B*` increases with ill-conditioning. This *tests* the mechanism
of §3 rather than assuming it, and is where the spectral-alignment diagnostic
(§6) earns its place.

### E3 — Does the optimal prior depend on `H`? (Q2)

For each observation configuration, find the best prior within a parameterized
family. Measure: does the argmin move with `H`, and by how much? Is a single
**config-distribution-aware** prior near-optimal across all configurations?

*Already half-answered on L96* (see §7.1): config-distribution-awareness
dominates both config-specific tuning and architectural agnosticism. The task is
to establish whether that shape is generic.

### E4 — Prior-knowledge ladder (Q3)

Run E1 at each rung — exact → hierarchical-known → misspecified, with a
*continuous* misspecification magnitude. Measure `B*` and the learned-prior
advantage at each.

**Prediction:** advantage increases monotonically as prior knowledge degrades;
`B*` decreases. **If the advantage is largest at Q1** — where the true prior is
exactly available — that inverts the expected ordering and is the more
interesting result.

Within Q3a, cross with **marginalize vs. condition on θ**.

---

## 6. Measurement protocol

**Primary responses.** (i) gap to the exact posterior mean — *available only in
Family 1, which is why it is co-equal, not an appendix*; (ii) RMSE / EV vs.
truth; (iii) calibration (rank histogram, spread–skill); (iv) budget `B` and
wall-clock/NFE.

**The unifying diagnostic.** For the (linearized) forward operator `A = UΣVᵀ`,
project onto its singular basis: (i) the truth's energy, (ii) the observation
information, (iii) each method's error. The claim predicts model-constrained
methods place error where `σ` is small, misaligned with (i) and (ii). One figure,
reproducible across families — this is what makes a genericity claim credible
rather than asserted.

For the dynamical family the analogue is the covariant-Lyapunov-vector basis
(Ginelli forward–backward), and `n₀` = number of non-negative exponents.

**Under Q3b, "best" must be redefined.** There is no exact-Bayes reference when
the model is wrong. Family 1 rescues this: compute both the posterior under the
*assumed* Σ and the truth under the *actual* Σ. The divergence between the two
references **is** the cost of misspecification, quantified. Another argument for
Family 1 as co-equal.

---

## 7. What is already in hand

### 7.1 Q2 — partially answered (L96, `l96_obs_density_augmented_training.md`)

| prior | keep_k=16 | keep_k=0 |
|---|---|---|
| config-specific | 0.486 | 1.292 |
| config-**distribution**-aware | 0.473 | 1.081 |
| architecturally agnostic (SDA3) | 0.537 | 1.082 |

Reading: **the prior should be aware of the *distribution* of observation
configurations but need not be specialized to any one.** Currently framed as
engineering (data augmentation); reframing it as Q2 is what makes it a
contribution.

### 7.2 Q3a — a candidate mechanism unifying two findings

**Hypothesis (untested).** Conditioning a learned prior on the true θ pushes it
to approximate `p(x|θ)` — the narrow per-θ manifold — which is precisely the
ill-conditioned object Q1 says not to use. Without θ, the model must learn the
marginal `p*(x)`, which is better conditioned because it averages over manifolds.

If true, two separate observations collapse into one mechanism: the exact per-θ
model is a poor *computational* prior, and handing a network θ makes it
reconstruct that same poor object.

**Test:** a θ-conditioned model should show error *more* concentrated along the
ill-conditioned / unstable directions than an unconditional one — measurable with
the §6 diagnostic applied to learned models. Clean falsification if not.

**Existing supporting observations** (L96 S0/S1 RMSE, and QG PV-q EV):
conditioning on *true* parameters is the worst of {none, true, noisy} across
three independent L96 families (SDA2 0.5588 vs SDA1 0.5532 vs SDA3 0.5365); on
QG, oracle-conditioned Q2 (0.6093 / 0.5571) is beaten by obs-only Q1 (0.6238 /
0.6238) despite carrying strictly more and strictly correct information.

### 7.3 Q1 — the decisive arm does not exist yet

No run compares the *exact* prior against a *learned representation of the same
distribution* at matched budget. This is E1 and it is the centrepiece.

---

## 8. Risks

**R1 — `B*` lands where nobody operates.** If the crossover sits at a budget far
above realistic operating points, the claim deflates to "under tight budgets,
learned priors help" — true, unsurprising, weak.
*Defense, which must be quantitative:* in high-dimensional geophysical inference
the budget is *permanently* tight — a 10⁸-dimensional 4D-Var will never be
converged, which is exactly why this repo's `Strong-4DVar` runs at
`max_iter=10`. If `B*` sits above realistic budgets across several families, the
tight-budget regime is the only regime that exists in practice. This argument is
available but must be made with measured `B*` values, not asserted.

**R2 — baseline fairness is now existential.** Under earlier framings the
under-converged variational baselines (`evaluate_all_l96.py`: Strong-4DVar
`max_iter=10`; Weak-4DVar `opt_steps=50, lr=0.1`) were a cosmetic weakness. Here
they are the entire paper: if a reviewer believes the exact-prior arm was
under-optimized, the claim evaporates. Converged baselines with a reported
convergence diagnostic are a **prerequisite**.
*Note the inversion:* E1's asymptote makes the converged arm part of the result
rather than a threat to it.

**R3 — Family 1 is not novel by itself** (§3). It must be presented as the
anchor, with the contribution resting on families 2–4.

**R4 — genericity claims invite "why not more domains?"** Mitigate by choosing
the minimum set and stating the selection criterion explicitly (§9).

**R5 — "learned prior" is under-specified.** Arm (b) of E1 must be a genuine
representation of `p*`, verified by a two-sample / distributional test against
`p*` samples, or the comparison is not "same distribution, different
representation."

---

## 9. Scope control

**Minimum viable: Family 1 (Gaussian) + Family 3 (L96).** One analytic, where
the claim is provable and the gap to exact Bayes measurable; one nonlinear,
where it matters. Add Family 2 or 4 only if those two agree — **and if they
disagree, the disagreement is the more interesting paper.**

L63 stays as the cheap spectrum-diagnostics system; QG as external validity.

**Phasing.**

- **P0** — converged variational baselines + convergence diagnostics (R2); `n₀`
  measurement on L63/L96; metric/convention harmonization. *Prerequisite for
  everything.*
- **P1** — Family 1 built end-to-end: E1 + E2 analytically and empirically. This
  is the cheapest possible full test of the central claim and should be done
  before committing compute anywhere else.
- **P2** — E1 on L96 (Q1, the decisive arm, §7.3); E4 ladder using the existing
  S0/S1 machinery.
- **P3** — E3 genericity (Q2) beyond L96; §6 diagnostic across families.
- **P4** — optional third family.

---

## 10. Prior art to position against

*Cited from memory — verify before any bibliography.*

- **Gribonval**, "Should penalized least squares regression be interpreted as
  maximum a posteriori estimation?" (IEEE TSP, ~2011) — closest hit: penalized-LS
  estimators correspond to MAP under a *different* prior than the data-generating
  one. Our claim is the empirical, computational counterpart.
- **Grünwald**, Safe Bayes / generalized Bayes — under misspecification the
  correct-by-construction posterior is not the best one.
- **PAC-Bayes** — the prior is explicitly a regularizer chosen to tighten a
  bound, with no requirement to be true.
- **Papaspiliopoulos, Roberts & Sköld** — centered vs. non-centered
  parameterizations: statistically identical, computationally divergent.
- **AUS / unstable subspace**: Trevisan & Uboldi (JAS 2004); Bocquet & Carrassi
  (Tellus A 2017); Gurumoorthy, Grudzien, Apte, Carrassi & Jones on covariance
  convergence onto the unstable subspace.
- VAE prior / aggregate-posterior mismatch, VampPrior; plug-and-play and learned-
  regularizer literature.

None asks this question with a *known ground-truth prior* and a *controllable
observation process*. That is the gap.

---

## 11. Open decisions

1. Is Family 1 co-equal (recommended) or an appendix-level sanity check?
2. Is Q2 co-headline with Q1, or a secondary section? This determines how much
   obs-config machinery must be built in the non-DA families.
3. Venue target — ICLR/TMLR (favours Q1 as a foundational question) vs. TNNLS
   (favours the architectural reading, tolerates the length four questions need).
4. Does flow matching / stochastic interpolants serve as the common substrate for
   arm (b) across all families? It gives a fixed sampler with the prior varying —
   the single-factor control this design needs — but sequential filters do not
   express as flows and must remain external anchors.

---

## 12. Relation to `phase_D_physics_priors_under_uncertainty.md`

Phase D's *framing* (C1–C5, physics representation vs. inference algorithm) is
superseded. Its *work packages* survive as the Family-3 instantiation:

- WP-0.1 (L96 Weak-4DVar) → still the top gap; now also the exact-prior arm of
  E1 on L96, and part of R2's converged-baseline prerequisite.
- WP-1.1 (prior swap, incl. the `aux_var_cost_weight` entry-mode finding) → the
  L96 instantiation of E1, with entry mode as an extra factor.
- WP-1.2 (conditioning ablation) → E4's marginalize-vs-condition axis, and the
  test of the §7.2 hypothesis.
- WP-1.3 (obs-config × physics representation) → E3.
- WP-1.4 (window-length sweep) → the dynamical-family instance of **E1's budget
  sweep**; window length is a budget-like knob and the crossover is `B*`.
- WP-0.2 / WP-0.3 (conventions, calibration, cost) → P0.
