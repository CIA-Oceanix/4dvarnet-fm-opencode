# Research notes — CFM/DA originality, physical-prior framing, and benchmark scope (discussion, 2026-09-02)

**Status:** DISCUSSION NOTES, not an execution plan. Captures a working conversation
about how to position and extend the CFM/4DVarNet work for publication. Each section
ends with concrete next steps that could seed a future phase design doc (in the style
of `phase_B_l96_cfm_variants.md` / `phase_C_l96_joint_da.md`).

---

## 1. Originality of `TweedieCFM` (V2) relative to CFM state of the art

- V3 (`PredictStateCFM`, predict-`μ = E[x1|xτ,y]` then derive velocity algebraically)
  is a standard, well-known reparameterization in flow matching / diffusion (x1- vs
  v- vs ε-prediction). Not novel by itself.
- V2 (`TweedieCFM`, mean-estimator + CFM-residual hybrid) matches an established
  pattern seen under other names — residual/cascaded diffusion correction on top of
  a deterministic mean (e.g. CorrDiff-style residual correction, GenCast's residual
  correction) — applied here with a flow-matching residual model instead of diffusion,
  in a DA context instead of forecasting/downscaling. Incremental, not a new
  mechanism.
- **Naming caveat (from `docs/phase_B_l96_cfm_variants.md:55`):** the residual stage
  uses the standard CFM loss (`v = residual − x0`), explicitly **not** Tweedie's
  formula. The "Tweedie" name is inherited from the legacy `TweedieSolver`
  (`models/solver.py`) and is a misnomer for the current V2 architecture — worth
  fixing in nomenclature before publication, or fixing in substance (see §2).

## 2. 4DVarNet-style gradient-conditioned unrolled solver as the mean/velocity parameterization

- `MeanEstimatorCell` (`models/residual.py:63`, used by `TweedieCFM` stage 1) is a
  plain obs-conditioned UNet iterated `K_inner` times — **no** gradient-of-cost
  conditioning.
- `IterativeUpdateCell` + `TweedieSolver.energy_terms` (`models/solver.py`) **does**
  condition each step on residual/gradient-like terms (`y_diff`, `phi_diff`,
  `bg_diff`) of a 3-term cost — structurally 4DVarNet-like — but is orphaned from
  the CFM path (legacy `TweedieSolver`, not used by V2/V3).
- Rating: unrolled, gradient-conditioned solvers as MAP/mean estimators are a mature
  idea (Recurrent Inference Machines, learned primal-dual, MoDL/E2E-VarNet, and
  4DVarNet itself, ~2017–2021) — not novel as a general pattern. Using that solver
  specifically as the amortized-mean parameterization feeding a **conditional
  flow-matching residual model** is less explored and closer to the current
  frontier (physics-/optimization-informed denoisers inside diffusion/flow priors;
  score-based DA, see §5) — this combination, if actually wired up and ablated
  against the current plain-UNet mean estimator, is a more defensible originality
  claim than either piece alone.
- **Next step:** replace `MeanEstimatorCell` with a solver built on genuine
  `torch.autograd.grad` of a real variational cost — see §4 (ocean4dvarnet
  `GradSolver`) — and ablate plain-UNet-mean vs. gradient-conditioned-mean inside
  the same residual-CFM stage 2.

## 3. Axis 1 — sequential DA's implicit Markov hypothesis breaks under model error

- Sequential filters (EnKF/ETKF, incl. joint state-parameter augmentation) assume
  the (augmented) state is a sufficient statistic updated recursively
  `p(x_t,θ_t|y_1:t) ← p(x_{t-1},θ_{t-1}|y_1:t-1)`. Under S1 model error (biased
  params + reduced/corrupted forcing) this breaks: the true generator mismatch is
  not a clean, locally-observable, one-step-at-a-time signal.
- **Evidence already in the repo** (`reports/l96/outputs/l96_consolidated_benchmark.md`):
  S1/S0 RMSE degradation — **ETKF 1.68×, EnKF 1.68×, Strong-4DVar 1.77×**, vs.
  **~1.00–1.01× for every neural scheme** (L1b/L2b/L3/V2/V3).
- **Mechanism, not just symptom:** `l96_joint_da_benchmark.md` — Joint-EnKF/Joint-ETKF
  cannot estimate `w3,w4` on S1 at all (`†`, defaulted to the prior) — the reduced
  forward model gives the filter no local channel to identify them online. Neural/
  amortized models instead train across the whole S0/S1 perturbation distribution
  offline, so they output something closer to a marginal posterior over the nuisance
  parameters rather than requiring online identifiability.
- **Framing for the paper:** "amortized/marginal inference vs. recursive-Markov
  filtering," not "neural nets are smarter." Connects to weak-constraint 4D-Var
  literature (Trémolet 2006; Bocquet et al. on colored/structural model error) and
  to amortized/simulation-based inference framings of robustness to nuisance
  parameters.

## 4. Axis 2 — physical prior: hard constraint vs. soft/weighted

- **Negative control already in the data:** Strong-4DVar (window-based, non-
  sequential, still physics-constrained) degrades *worse* than ETKF/EnKF under S1
  (1.765× vs. 1.68×) — having the physical model in the loop is not sufficient by
  itself. What Strong-4DVar shares with ETKF/EnKF is using Φ as a **hard
  constraint** (`evaluation/baselines.py:565`, `Strong4DVar.assimilate`: only `x0`
  is a free variable, `traj = Φ^t(x0)` exactly). Forcing consistency with a biased
  Φ actively hurts.
- **Weak-constraint reformulation:** make `x_1..x_T` free control variables and add
  a penalized model-error term instead of exact substitution:
  `J = 0.5‖x0−x_bg‖²/B + 0.5·Σ‖H(x_t)−y_t‖²/R + 0.5·Σ‖x_{t+1}−Φ(x_t)‖²/Q`.
  `Q` (precision on trusting Φ) is the "weight." `Q→0` recovers current hard-
  constraint `Strong4DVar`; `Q→∞` drops the dynamical prior entirely.
- **Weighting options, increasing sophistication:**
  1. Fixed scalar `Q` — cheapest test of the hypothesis, same LBFGS loop.
  2. Structured/heteroscedastic `Q` (e.g. more trust for L96 slow modes / QG
     large-scale streamfunction, less for the fast/obs_fast block or eddy scales
     where S1's reduced dynamics actually differ) — informed by the S0/S1 design
     itself.
  3. Learned/adaptive weight — feed `phi_diff = x − Φ(x)` (already computed in
     `TweedieSolver.energy_terms`) into a learned update (`IterativeUpdateCell`)
     instead of enforcing it to be zero; the "weight" becomes an implicit, learned,
     context-dependent function rather than a fixed `Q`.
- **Proposed ablation:** same optimizer/window/Φ, three treatments — (a) hard
  constraint (current `Strong4DVar`), (b) fixed/structured weak-constraint `Q`,
  (c) learned weak-constraint via the gradient-conditioned solver (§2/§4-ocean).
  Showing (b)/(c) close most of the S1 gap that (a) fails to close is a much
  stronger claim than "physics helps" or "physics doesn't matter."

## 4b. Reuse vs. rewrite: `ocean4dvarnet` (CIA-Oceanix)

Checked the actual repo (`gh api repos/CIA-Oceanix/ocean4dvarnet`), not from memory.

- `ocean4dvarnet/models.py` has `GradSolver` + `ConvLstmGradModel` — a genuine
  `torch.autograd.grad`-based unrolled solver:
  `var_cost = prior_cost(state) + lbd**2 * obs_cost(state, batch)`;
  `grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]`;
  ConvLSTM-modulated update blended with an annealed raw-gradient term. This is
  more rigorous than the repo's current `IterativeUpdateCell` (hand-fed residual
  features, no actual cost differentiation).
- Package is otherwise built around gridded ocean remote-sensing data (`xarray`/
  `netCDF4` pinned deps, patch-reconstruction `Lit4dVarNet`) — no Lorenz63/96/QG
  dynamics, no DA-baseline harness, no CFM/flow-matching code. Nothing there is
  reusable for the surrounding infrastructure this repo already has and has
  tested.
- **Recommendation:** neither full rewrite nor full migration. Port just
  `GradSolver` + `ConvLstmGradModel` into `models/solver.py`, replace
  `IterativeUpdateCell`'s hand-fed `energy_terms` with real autodiff `∇J`, adapt
  2D convs → 1D (trivial, `UNet1D` is already channels-first `(B,C,L)`). Write our
  own `obs_cost` (from `y_diff`) and `prior_cost` (start from `phi_diff` as a
  fixed-weight quadratic = the weak-constraint term from §4, later graduate to a
  learned prior like `BilinAEPriorCost`).
- **License:** CeCILL-C (IMT Atlantique/OceaniX, R. Fablet contact) — not a
  practical blocker given shared authorship, but keep attribution on ported files.

## 5. Should score-based DA (SDA-style) be included in the benchmark?

**Yes — it's a third, non-redundant axis**, not just another baseline row.

- Everything currently benchmarked is either sequential-Markov, hard-constraint-
  variational, or end-to-end conditionally trained. Score-based DA (Rozet &
  Louppe, "Score-based Data Assimilation," NeurIPS 2023 — Lorenz-63, Kolmogorov
  flow) trains an **unconditional** diffusion/flow prior `p(x_1:T)` with no
  observations at training time, and combines it with a likelihood/guidance term
  (Tweedie-type posterior-mean estimate) only at inference. Orthogonal to both
  existing axes: not sequential (samples the window jointly, more smoother-like),
  not end-to-end conditioned.
- **Sharp narrative link to §1:** this is where Tweedie's formula is genuinely
  load-bearing (unlike our own `TweedieCFM`, see §1) — V3's `x̂1 = xτ + (1−τ)v(xτ,τ)`
  is exactly the flow-matching analogue of the estimator SDA-style guidance needs.
  Benchmarking against it makes the "Tweedie" naming honest instead of aspirational.
- **Model-error interaction — must be an explicit, reported design decision:**
  - Prior trained on nominal (S0) dynamics only, guided toward S1 observations →
    expect a Strong-4DVar-like failure mode (right manifold, wrong under S1,
    guidance can't fix the manifold).
  - Prior trained on the same S0/S1-perturbed parameter distribution as our CFM
    models → fair comparison isolating "conditional end-to-end" vs. "unconditional
    prior + inference-time guidance" as the only remaining difference.
  - Report both variants; don't silently pick one.
- **Cost axis to report alongside accuracy:** guidance sampling is typically far
  more NFE-expensive (per-step gradients through the score/velocity network) than
  our amortized `N_outer≈10` Euler sampling — "comparable accuracy at 5–10× cost"
  is a real finding, put it in the same table.
- **Scope:** mostly reuses existing code — same `UNet1D` + `LinearInterpolant`,
  trained unconditionally (drop obs conditioning), plus a new DPS/ΠGDM-style
  guidance sampler (nudge via `∇_x‖y − H(x̂1(x_τ))‖²`). New pieces: the guidance
  step and the guidance-weight/likelihood-covariance choice (itself another
  instance of the §4 soft-weighted-prior knob). Recommend scoping the first pass
  to L96 only before extending to QG.

## Open items / candidate next phase docs

1. Ablation: plain-UNet mean estimator vs. gradient-conditioned (`ocean4dvarnet`
   `GradSolver`-derived) mean estimator inside `TweedieCFM` stage 1.
2. Weak-constraint `Strong4DVar` variant (fixed → structured → learned `Q`) vs.
   current hard-constraint baseline, on the existing S0/S1 L96 set.
3. SDA-style unconditional prior + guidance sampler for L96, trained both on
   nominal-only and on the S0/S1-perturbed distribution; report accuracy + NFE/cost.
4. Nomenclature/substance fix for `TweedieCFM` naming (§1) once (1)-(3) clarify
   what the architecture should actually be.
