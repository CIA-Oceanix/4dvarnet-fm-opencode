## 2026-09-17: DA-venue (JAMES) scoping doc + ML-venue scope retargeted

**Summary:** New scoping doc for a DA-venue diagnosis paper on the three
structural hypotheses of classical DA, and a scope revision to the ML-venue doc
retargeting its headline away from an experiment its assets cannot host.

**Files modified:**
`docs/scoping/da_paper_structural_hypotheses.md` — new, ~350 lines.
`docs/scoping/ml_paper_exploiting_prior_knowledge.md` — new §8.0 scope revision;
§10 decisions 5 (headline retargeted) and 6 (boundary with the DA paper, open).
`docs/scoping/README.md` — index entry plus a standing caveat on the S0/S1 axis.

**Rationale:** The ML doc's headline (C1 via E2, exact vs learned prior) cannot
run on family D: for a chaotic system "p* known exactly" means it can be
simulated, not evaluated, so there is no tractable exact-prior arm at L96's
dimensionality. Meanwhile R6's measured ~93%/~7% split instructs moving the
emphasis to the conditional-mean operator — the matched-budget mean-slot
comparison, which an independent 2026-09-15 positioning review had already named
from novelty grounds. Two routes converging on one experiment.

The DA doc is built only on this repo's own numbers. Its spine is a previously
unreported effect: under model error, doubling the observations buys
Strong-4DVar **3.2%** RMSE against **19.7%** without it — a 6.2x collapse in the
marginal value of observations (EnKF/ETKF 1.9x). Model error does not merely add
an error floor, it decouples the analysis from the data.

**Verification:** Every figure re-derived from
`l96_consolidated_benchmark.md` and `s0_s1_obs_density_da_baselines.md`;
percentages recomputed rather than copied. Three claims were checked against
source and corrected before landing: (1) the S0/S1 axis is invisible to
model-free methods — `RandomBiasLorenz96Dataset._compute_params_da` biases the
DA model's parameters only, so a neural `S1/S0 ~ 1.00` is definitional, and §4.1
was rewritten to present the neural rows as a control rather than a competitor;
(2) no code implements a rank histogram (the term appears only in planning
docs), so the claim was narrowed from "nothing in this repo has one";
(3) `train_forcing_state_bias` defaults to 0.1, matching `test_s1`, which is a
second compounding asymmetry now recorded in R1.

**Known gaps recorded, not fixed:** L96 Weak-4DVar still absent and now gating
both papers (it re-anchors the 93%/7% split and is the classical method allowed
to know about model error); the DA and neural obs-density studies vary
*different* sparsity axes (temporal vs channel) and must not be tabled together.
