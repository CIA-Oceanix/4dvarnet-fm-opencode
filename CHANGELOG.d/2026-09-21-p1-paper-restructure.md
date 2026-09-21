## 2026-09-21: Restructure the P1 draft around CFM as a posterior-diagnosis instrument

**Summary:** Rewrites `docs/papers/p1_structural_hypotheses/` from the
hypothesis-ordered outline of #233 into the conventional
Introduction / Background / Methods / Evaluation / Results / Discussion shape,
with a new Methods section introducing conditional flow matching as a **generic
representation of posteriors** used as a diagnostic instrument -- not as a
proposed estimator. 13 pages, compiles clean, zero undefined references.

**Why a Methods section at all.** The #233 draft asserted that EnKF/ETKF carry
`Psi_NG = 0` by construction and promised attribution "to a named term of the
operator partition", but **never defined that partition** -- it referred out to a
companion doc. That was a real hole: the paper invoked machinery it did not
establish. Section 3 now derives the interpolant, the ODE, the exact additive
partition `Psi = Psi_mean + Psi_G + Psi_NG`, its boundary conditions
(`Psi_NG` vanishes at both ends), and the table of what each scheme class zeroes.
The deterministic-estimator-as-degenerate-flow identity is what licenses putting
4D-Var, OI and a regression network on one axis.

**Scope changes, all requested:**
- **State estimation only.** Section 2 states the target as `p(x_1 | C)` with
  `C = (y, z*, theta*)`, and says explicitly that joint state-parameter inference
  is out of scope and that `theta*` is an input, never an estimand. The
  `joint_estimation_*` evidence is dropped; parameter *conditioning* is retained,
  since SDA1/2/3 and QG Q1-Q4 estimate state while differing in what they are
  told.
- **No unrolled solvers.** Every FDV/subgrad row is gone. The headline L96 table
  is rebuilt from DA baselines + DirectUNet + CFM + SDA + DirectUNet+SDA; the best
  scheme is now `DirectUNet+SDA3` at 0.4204 pooled S0 (was `subgrad+state-Stier
  +SDA3` at 0.3411).
- **L63 added** as a third case study, and it earns its place: it is the only
  system where weak-constraint 4D-Var has been run, and **it wins** -- S0 0.642 /
  S1 1.633 against Strong-4DVar's 0.767 / 2.100, i.e. 22% better under model
  error.
- **C7 (the adjoint corollary) dropped to P2.** With unrolled solvers out, its
  learned-side ladder is gone and only the intra-classical half remained.

**The L63 result forces C1 to be narrowed now, not later.** The scoping doc
treats weak-4DVar as a pending prerequisite (D0) that *might* soften C1. L63
shows it recovering most of the model-error loss, so the draft states C1 for the
strong-constraint and filtering families rather than for the classical framework
as a whole, and flags the missing L96 measurement with `\needsrun`. Better to own
this than to have a referee find it.

**Notation** follows `previous_draft.tex` (uploaded via Overleaf): `\mathbf{x}_1`
for the target state, `\mathbf{x}_0`/`\mathbf{x}_\tau` for the base sample and
interpolant, `\mathbf{C}` for the context, `\mathcal{M}_\Delta`/`\Phi`/`\mathcal{H}`
for the dynamics and observation operators.

**Files:** `main.tex` (notation macros, section list); new
`sections/02_background.tex`, `03_methods.tex`, `04_evaluation.tex`,
`05_results.tex`, `06_discussion.tex`; rewritten `01_introduction.tex`; updated
`07_conclusion.tex`; removed the superseded `02_hypotheses` / `03_setup` /
`04_evidence` / `05_claims` / `06_limitations`.

**Verification:** `pdflatex` x3 -- **13 pages, 0 undefined references or
citations**. Every `\ref{sec:...}` checked against a matching `\label`: none
dangling. Build artefacts gitignored, none committed.
