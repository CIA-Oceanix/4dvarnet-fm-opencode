## 2026-09-18: Specify the ML paper's case studies against an explicit admission criterion

**Summary:** §4 rewritten. Every family now has a known prior *and* a defined
exact-inference arm; the real-data MRI arm is dropped; dynamical systems return
as **L63 weak-constraint**. §8.1's earlier withdrawal is narrowed from "family D"
to "L96/QG". Docs only.

**Files modified:** `docs/scoping/ml_paper_exploiting_prior_knowledge.md` — new
§4.0–§4.5 replacing the old §4/§4.1; E0, E3, §7, §8 scope + phasing, §8.1 header
and operative scope, §9 positioning, §10 decisions 1, 3 and 5.

**Rationale:**

1. **An admission criterion, stated so it can be checked (§4.0).** E2's two arms
   need different things: arm (b) needs *sampling*, arm (a) needs *tractable
   posterior inference*. A black-box sampler gives only (b). A family is admitted
   as an E2 host only if `p*` is exactly samplable **and** admits near-exact
   posterior inference under that family's `H`. This is the criterion that
   disqualified L96/QG, and it disqualifies the real-data MRI arm too.

2. **Family C split, then halved.** Real fastMRI/natural images cannot host E2 —
   there is no `p*` — and Q3, the one thing they looked useful for, is answerable
   *more* sharply on synthetic data, where it can be asked against a reference
   posterior. The real-data arm is dropped rather than kept as a secondary
   family: one that visibly omits the headline experiment advertises the
   synthetic-only limitation instead of covering it. **C1** (phantom generative
   program, Gibbs-tractable) is retained as the only **null-space** instance.

3. **B specified as a Gaussian scale mixture.** Exactly samplable, genuinely
   non-Gaussian, posterior-tractable by Gibbs under a linear-Gaussian likelihood,
   and the standard wavelet-domain deconvolution prior, so the family is
   literature-grounded. It also makes `Ψ_mean/Ψ_G/Ψ_NG` computable
   **semi-analytically** by Rao-Blackwellising over the latent scales — the only
   place in the whole programme where §2's partition can be validated rather than
   fitted, which serves E1/M1 in the DA papers too.

4. **D′ restored, and §8.1 narrowed.** The 2026-09-17 correction withdrew
   "family D" when its argument only ever applied to **L96/QG**: at 40+
   dimensions there is no tractable exact posterior, but **L63 in weak-constraint
   (SDE) form** has an evaluable and samplable trajectory prior and a near-exact
   particle-smoother reference in 3-D. §5's E0 already said this; flattening it
   removed C4's state-dependent arm, which is the half the mechanism claim rests
   on — A, B and C1 are all fixed linear operators where preconditioning should
   work. D′ is also the **cheapest** family, not the most expensive
   (`data/lorenz63.py` is already an SDE simulator), so phasing moves it to P1.

5. **Prior art checked (§4.4).** Benchmarks with analytically known posteriors
   exist (arXiv 2503.03007, code on Zenodo/Dataverse), but retrieval confirmed
   they buy tractability with **Gaussian priors and linear operators** — i.e.
   family A, where `Ψ_NG ≡ 0`. Useful as an external cross-check for A; cannot
   host the interesting regime. That every known-posterior benchmark lives where
   the effect vanishes is itself an argument for the paper.

**Verification:** Docs only, no code touched. All external claims verified by
retrieval on 2026-09-18 (2503.03007's Gaussian construction; the Gaussian
scale-mixture Gibbs literature). Downstream references swept: E0, E3, §7's
"S0/S1 machinery" bullet, §8's minimum-viable and phasing, §8.0's superseded
table rows, §9, and §10 decisions 1/3/5 all updated, so no section still
describes the pre-2026-09-18 family set as operative.
