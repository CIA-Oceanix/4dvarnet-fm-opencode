## 2026-09-16: Paper scoping — "is the true prior the best prior?" + Phase D experiment plan

**Summary:** Three research-planning documents.
`scoping_ml_paper_exploiting_prior_knowledge.md` is the ML-venue scoping: it
adopts the `Psi_mean / Psi_G / Psi_NG` flow-matching operator partition (from the
two drafts added on master in #209) as its *organizing framework*, turning the
contribution into a domain-agnostic diagnostic that attributes any inversion
method's deficit to named components, and spans four case-study families
(Gaussian negative control, deconvolution with a non-Gaussian prior, Fourier
subsampling / MRI, chaotic dynamics). `scoping_best_prior_approximate_inference.md`
defines the scientific question (under *approximate* inference, is the true prior
the best prior to use, and should it be independent of the observation
configuration?), the Q1/Q2/Q3a/Q3b ladder, four testbed families spanning well
beyond data assimilation, and four experiments (E1 budget sweep, E2 conditioning
sweep, E3 obs-config dependence, E4 prior-knowledge ladder).
`phase_D_physics_priors_under_uncertainty.md` is the dynamical-systems
instantiation: targeted contributions, the four-axis design space, and work
packages tiered by cost.

**Files modified:**
- `docs/scoping_ml_paper_exploiting_prior_knowledge.md` — new ML-venue scoping doc (Q1-Q4, the operator decomposition as organizing framework and method taxonomy, four case-study families, E0-E5, risks R1-R6); supersedes the *framing* of the doc below
- `docs/scoping_best_prior_approximate_inference.md` — new scoping doc (question, testbeds, experiments, risks, phasing, prior art)
- `docs/phase_D_physics_priors_under_uncertainty.md` — new design doc (C1-C5, work packages WP-0.1 … WP-3.1); its *framing* is superseded by the scoping doc, its work packages survive as the Family-3 instantiation (see scoping §12)

**Rationale:** Consolidates a long design discussion before any compute is
committed. Two findings shaped the plan and are recorded so they are not
rediscovered: (a) every currently-benchmarked method differs on three or four
design axes simultaneously, so the existing leaderboard cannot attribute any
effect to an axis — hence the harness-plus-corner-anchors discipline; (b) the
learned prior as an *explicit* auxiliary cost (`aux_var_cost_weight`,
`models/fourdvarnet.py::_prior_ae`) has already been tried and **hurt**
(val_loss ≈ 0.39 vs ≈ 0.20-0.29 aux-free), while the same prior entering through
`subgrad+state`'s gradient channel gives the best non-hybrid row — so *how* a
prior enters the cost may matter more than *what* the prior is, and the decisive
experiment must be a 3x2 grid rather than a 3-cell swap.

Also records that the under-converged variational baselines
(`evaluate_all_l96.py`: Strong-4DVar `max_iter=10`, Weak-4DVar
`opt_steps=50, lr=0.1`) shift from a cosmetic weakness to a prerequisite under
this framing, since the central claim is "learned representation beats exact
prior at finite budget."

**Verification:** Documentation only — no code, config or test changes. CI gate
test list unaffected (last full run on this tree: 454 passed, 4 skipped).

**Note:** `phase_D_*.md` (WP-3.1) cross-references
`docs/cfm_affine_velocity_decomposition.md`, which lands via the separate
`feature/cfm-affine-velocity-decomposition` branch; the reference resolves once
that PR merges.
