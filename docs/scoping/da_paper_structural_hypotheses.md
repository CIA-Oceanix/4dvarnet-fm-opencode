# Three structural hypotheses that bound data assimilation — DA-venue scoping

**Status:** SCOPING v1 (2026-09-17), targeting **JAMES**. Companion to
`docs/scoping/ml_paper_exploiting_prior_knowledge.md`, which after its
2026-09-17 revision (§8.0) keeps the `Ψ_mean/Ψ_G/Ψ_NG` operator partition and
the matched-budget mean-slot comparison. **This doc owns the model-error and
observation-sparsity axes**; see §10 decision 1 for the split, which is not yet
settled.

Every figure below is from this repo's own L96/QG runs. Nothing is quoted from a
draft or preprint.

---

## 1. The question

Data assimilation rests on three commitments so standard they are rarely written
down as hypotheses:

- **H1 (sequential).** The analysis is produced by cycling: propagate a state
  estimate, update when observations arrive, repeat. The analysis at time `t`
  uses observations up to `t` (filtering), or within one window (4D-Var,
  smoothers), and each cycle is re-initialized from the last.
- **H2 (model fidelity is the route to skill).** *The better the dynamical
  model, the better the assimilation.* Model development and DA development are
  therefore the same programme.
- **H3 (ODE/PDE state representation).** The state is a point in a phase space
  whose prior is the flow map of a differential equation; uncertainty is carried
  as a perturbation of that point (a covariance, or an ensemble of points).

Each buys something real. The thesis of this paper is that **the three jointly
explain two persistent and widely-observed symptoms**, and that each hypothesis
can be relaxed *separately* and the gain attributed:

- **S-a.** DA converts additional sparse observations into skill inefficiently,
  and the inefficiency *worsens sharply* under model error.
- **S-b.** DA's posterior characterization is structurally impoverished — not
  merely mis-tuned.

This is a **diagnosis paper**, not a new-method paper. That is deliberate: a
method paper invites "your baseline was under-tuned", whereas the claim here is
about what the baselines *cannot* do however well tuned, with the tuning
sensitivity studies already done (§4.5) as the evidence that they were.

---

## 2. The hypotheses stated so they can be falsified

| | asserts | buys | costs | relaxed by |
|---|---|---|---|---|
| **H1** | analysis is a cycled/windowed recursion | O(1) memory, operational streaming, no training set | cannot adapt to the *observation configuration*; no amortization across windows | window-amortized estimators; smoothers |
| **H2** | skill is bounded by, and increasing in, model fidelity | the model is a physically-consistent, transferable prior | the model *is* the prior, so model error enters with **no buffer** | weak-constraint formulations; learned/hybrid priors; model-error covariance `Q` |
| **H3** | state = point in ODE phase space; uncertainty = perturbation | tractable propagation; physical interpretability | the *choice of state variable* fixes the conditioning of the inverse problem; Gaussian update ⇒ `Ψ_NG ≡ 0` | alternative state variables; non-Gaussian / generative posterior representations |

**H2 is the load-bearing one, and it is not simply false.** The hypothesis is
true as stated — better models do assimilate better. The claim here is that it is
*binding*: because the model supplies the entire prior, DA has no mechanism to
absorb model discrepancy, and the penalty is not a constant error floor.

---

## 3. The symptoms, operationalized

- **S-a — marginal value of observations.** Not "RMSE at a given density" but
  **dSkill/d(density)**: how much a doubling of observations buys, measured at
  matched everything, under S0 (no model error) and S1 (model error).
- **S-b — posterior quality.** Spread/RMSE against the reliability target
  `sqrt((N−1)/(N+1))`, rank histograms, CRPS/ES with a *uniform* ensemble-size
  convention, and — the structural part — the `S(Ψ_NG, 0)` term that the
  ensemble-Kalman class carries **in full, by construction**.

---

## 4. Evidence already in hand

### 4.1 The price of H2, with a built-in control

L96, S0→S1 (S1 = ±10% bias on the DA model's parameters), 200 windows, pooled
RMSE and explained variance, from `reports/l96/outputs/l96_consolidated_benchmark.md`:

| scheme | S0 RMSE | S1 RMSE | S1/S0 | EV S0 → S1 |
|---|---|---|---|---|
| Strong-4DVar | 0.8116 | 1.4617 | **1.801** | 0.748 → 0.233 |
| ETKF | 0.8883 | 1.4998 | **1.688** | 0.692 → 0.215 |
| EnKF | 0.9131 | 1.5381 | **1.684** | 0.676 → 0.172 |
| DirectUNet-M(monai,cos) | 0.5064 | 0.5072 | 1.002 | 0.899 → 0.898 |
| FDV1-Stier(monai) | 0.4011 | 0.3993 | 0.996 | 0.936 → 0.936 |
| subgrad+state-Stier+SDA3(monai) | 0.3410 | 0.3402 | 0.997 | 0.952 → 0.952 |

**Read this table with the three method classes separated.**
`data/lorenz96.py:614-622` builds `test_s1` with `param_bias=0.15` through
`RandomBiasLorenz96Dataset`, whose `_compute_params_da` applies the bias to the
**DA model's** parameters; the truth is drawn from the same distribution in both
splits (`param_noise=0.2` either way). What S1 means therefore depends on
whether, and how, a method consumes model information:

| class | what S1 changes for it | ratio ~1.00 means |
|---|---|---|
| classical DA (4D-Var, EnKF/ETKF) | integrates a **biased forward model**; bias compounds through the integration | n/a — they degrade 1.68-1.80x |
| model-free learned (DirectUNet, FDV, SDA1) | **nothing** — no params, no forward model | **definitional**; the 0.992-1.002 scatter is resampling noise |
| conditioned learned (SDA2, SDA3; QG Q2-Q5) | **the conditioning inputs are biased/corrupted** | **a genuine robustness result** |

The third row is the one to be careful about, and it is *not* a no-op.
`data/dataloader.py:99-108` (`FlowMatchingDataset._extract_params`) is explicit:
S0 windows carry no `*_da` entries so the conditioning falls back to the true
values, while on S1 the conditioning vector is
`true + frac * (da - true)`, `frac ~ U(0, noisy_da_max)`. **So SDA2/SDA3 are
handed wrong physical parameters at inference on S1 and true ones on S0** — and
their ratios are 1.004 and 0.998. That is model-error exposure at inference, and
they absorb it.

### 4.1.1 Why they absorb it: on L96, parameter conditioning is nearly inert

The conditioning ladder was built to test exactly this (`SDA1_*`, `SDA2_cond_mixed`,
`SDA2_cond_nominal`, `SDA3_cond_noisy`; QG's `Q2` oracle / `Q3` noisy / `Q4`
noisy+IC). Its S0 numbers are the interesting part:

| scheme | conditioning | S0 RMSE | S1 RMSE | S1/S0 |
|---|---|---|---|---|
| SDA1(monai) | none (unconditional prior) | 0.5532 | 0.5521 | 0.998 |
| SDA2(monai) | **true** params + corrupted forcing | 0.5588 | 0.5610 | 1.004 |
| SDA3(monai) | **noisy** params (0-1.5x of the true→DA bias) | **0.5365** | 0.5355 | 0.998 |

Three further runs already on disk make this a replicated observation across
independent families (within-family comparisons only -- the backbones and
normalization differ, so the absolute levels are not comparable across rows):

| family | uncond. | + model-information conditioning | effect |
|---|---|---|---|
| SDA (monai, guided) | SDA1 0.5532 | SDA2 (true params) 0.5588 | **+1.0% worse** |
| SDA (non-monai, guided, `ens30_no10`) | SDA1 0.7185 | SDA2-mixed 0.7074 | −1.5% better |
| VanillaCFM tau=0 (`L6` vs `L2b`) | L2b 0.6329 | L6 (corrupted forcing) 0.6389 | **+0.9% worse** |
| SDA (monai, guided) | SDA1 0.5532 | SDA3 (**noisy** params) 0.5365 | −3.0% better |

**Conditioning on model information moves S0 skill by at most ~1.5%, and the
sign is not even consistent across families.** It is, to a good approximation,
**inert**. The one variant that reliably helps is *noisy* conditioning (−3.0%),
which looks like regularization rather than information use. The network extracts
little usable information from the physical parameters and forcings.

**This is the sharpest available statement of the paper's thesis.** The learned
schemes are robust to corrupted model parameters largely *because they barely
depend on them* — the observations carry the information. Classical DA has no
such fallback: its prior **is** the integrated model, so a parameter bias is not
a perturbed input but a corrupted prior, compounding over the window. H2 is
binding not because model information is unhelpful, but because DA has no way to
down-weight it.

**The training-exposure control has also been run, and it does not explain the
invariance.** `SDA2_cond_nominal_l96` is trained at `forcing_state_bias=0.0` --
the only configuration that never sees S1-level model error during training. Its
guided 30-member evaluation gives S0 0.7046 / S1 0.7033, **ratio 0.998**,
indistinguishable from SDA2-mixed (0.997) and SDA1 (0.998). Removing training-time
exposure to model error entirely leaves the S1 invariance intact. This retires
what was the third concern in R1.

**The caveat a referee will raise, and it is fair.** Conditioning on biased
parameters is not the same exposure as *integrating* a biased model: one is an
input perturbation, the other compounds through time. The two are not
interchangeable, and §6 D2 is what puts them on a common axis. §4.1.2 is what
makes that distinction the point rather than a nuisance.

### 4.1.2 Corollary: ODE sensitivity is not a proxy for posterior sensitivity

Write the model-uncertainty pair as `(theta, z)` -- physical parameters and
forcings. Two different sensitivities are in play, and **classical DA cannot
separate them**:

- **Model sensitivity**, `d x_ODE / d(theta, z)` -- how much the *free-running*
  trajectory moves. In a chaotic system this is Lyapunov-amplified and large;
  §4.1's 1.68-1.80x degradation is an instance of it.
- **Posterior sensitivity**, `d p(x | y, theta, z) / d(theta, z)` at **fixed
  `y`** -- how much the actual object of inference moves.

Classical DA conflates the two **by construction**, because its estimator *is*
the ODE: every route from `(theta, z)` to the analysis passes through the
integration, so the analysis inherits the model sensitivity whether or not the
posterior has it. Its S0→S1 degradation therefore measures **its own estimator's
sensitivity**, and cannot be read as a measurement of how much `(theta, z)`
uncertainty matters for `p(x | y, theta, z)`.

A conditioned amortized estimator measures the *other* quantity directly: it
takes `(theta, z)` as inputs at fixed `y`, so perturbing the conditioning while
holding the observations fixed is a finite-difference estimate of posterior
sensitivity. Across three families that perturbation moves skill by **≤1.5%**,
and full corruption (S1) by **~0.2%**.

**Consequence, and it is the sharpest methodological point in this document.**
The standard practice of assessing parameter/forcing uncertainty by *propagating
it through the model* systematically **overstates** its effect on the posterior
whenever the observations are informative, and the overstatement is exactly the
gap between the two sensitivities. It also reframes H2: "the better the
dynamical model, the better the DA" is a true statement about DA's *estimator*,
and its apparent force is partly an artifact of **H3** -- the ODE representation
that routes all prior information through the integration. H2 and H3 are not
independent hypotheses; H3 is what makes H2 bind.

**Prediction.** Posterior sensitivity should *grow* as observations sparsify,
since less of `(theta, z)` is screened off by `y`. §4.2's collapse in the
marginal value of observations is the shadow of the same mechanism: at low
density DA has nothing but the model, so model error and observation sparsity
**compound rather than add**. **D7** measures both sensitivities on one grid.

### 4.2 H2 × S-a: model error destroys the *marginal value* of observations

The finding that makes S-a more than a restatement of 4.1. From
`reports/l96/outputs/s0_s1_obs_density_da_baselines.md` (a *different* config —
NO=8, all-5-parameter ±20% randomization, DWS=500, `obs_j=2`; densities are
**temporal**, `obs_interval` 200 → 100, i.e. 15 → 30 obs/window):

| method | S0: RMSE gain from 2× obs | S1: same gain | collapse |
|---|---|---|---|
| Strong-4DVar | 0.9701 → 0.7788 (**19.7%**) | 1.4751 → 1.4276 (**3.2%**) | **6.2×** |
| ETKF | 1.0973 → 0.8815 (19.7%) | 1.6367 → 1.4680 (10.3%) | 1.9× |
| EnKF | 1.0927 → 0.9046 (17.2%) | 1.6503 → 1.5022 (9.0%) | 1.9× |

Under model error, **doubling the observations buys Strong-4DVar 3.2%**. In the
fast-variable subspace the S0→S1 story is starker still: `EV obs_fast` goes
+0.417 → −0.011 at Obs200 and +0.622 → +0.047 at Obs100.

This is the paper's spine: **H2 does not merely set an error floor, it decouples
the analysis from the observations.** A method whose prior is a wrong model
cannot be rescued by more data — which is exactly the regime of sparse-observation
geophysical DA.

### 4.3 H3: the *choice of state variable* changes DA skill, at identical physics

QG two-layer. The `psi_state` variant integrates the same dynamics with the
streamfunction as the state variable; the q-space physics is **bit-identical**
(free forecasts match to ~1e-6 relative). Only the representation differs — and
with it the observation operator, which becomes a trivial index lookup.

Result: `psi_state` gives the **best per-field streamfunction analysis**
(ψ1/ψ2 EV ≈ 0.98) and a **degenerate PV field** (q full EV **−3.2** at
`da_nx=32`, against the q-state reference **+0.34**), because `forward_pv`
(`q ≈ ∇²ψ`) amplifies high-wavenumber ψ-analysis error by `K²`. At matched S0
noise 0.01 the psi-state q-field EV is 0.583 against the q-state 0.752.

**This is H3 in one experiment.** Identical dynamics, identical information, and
a large skill difference produced purely by which variable is called "the state".
The ODE representation is not a neutral container.

### 4.4 H1 + H3: the ensemble-Kalman class carries `S(Ψ_NG, 0)` in full

In the operator partition adopted by the companion doc, every inversion method
defines `Ψ̃ = Ψ̃_mean + Ψ̃_G + Ψ̃_NG`. For EnKF/ETKF/ensemble smoothers,
**`Ψ̃_NG = 0` identically** — the Gaussian update is the method. This is not a
tuning deficiency; it is a statement about the class.

Measured consequences (`reports/l96/outputs/l96_fm_sampler_benchmark.md`, #211):
**no scheme measured is simultaneously accurate and calibrated.** The only
configuration carrying a full non-Gaussian branch is ~20–25% over-dispersive
(ratio/target 1.20 / 1.22 / 1.24 at `N` = 6 / 10 / 30 — stable in `N`, so a
sampler property), while every scheme that closes the RMSE gap does so by
importing a deterministic mean and collapses to ratios 0.28–0.59 against a
target of 0.967.

### 4.5 Localization and inflation as H3 symptoms, with the sensitivity work done

The QG study found `loc_radius = 6.0` was untuned **even at the reference
density**, and that the optimum moves with observation density
(`loc_radius = 2.0` for cols ≤ 32, `1.0` at cols = 64; `etkf_ridge` 1.0 → 0.1).
Localization is a patch for an ensemble's inability to represent the covariance
at finite `N` — an H3 cost — and the fact that its optimum tracks the
*observation configuration* is a small, clean instance of what H1 forecloses.

This work also does double duty as the fairness evidence for R2: the tuning
sensitivity of the classical baselines has been mapped, not assumed.

### 4.6 A motivating ordering (not yet evidence)

L96 S0: sequential filters (ETKF 0.8883, EnKF 0.9131) < window variational
(Strong-4DVar 0.8116) < window-amortized learned (0.3410–0.5064). Suggestive of
H1, but **confounded** — amortization, learning, and capacity all vary at once.
D1 is what turns this into evidence.

---

## 5. Claims

- **C1 (H2, headline).** Because the model *is* the prior, model error enters the
  analysis unbuffered, and its cost is not a constant offset: it **collapses the
  marginal value of observations** (6.2× for Strong-4DVar on L96). This is the
  mechanism behind DA's limited return on sparse observations.
- **C2 (H3).** The ODE state representation fixes the conditioning of the inverse
  problem. Identical physics in a different state variable produces large,
  predictable skill differences (`q ≈ ∇²ψ` ⇒ `K²` error amplification).
- **C3 (H1).** Sequential/cycled processing forecloses adaptation to the
  observation configuration. A configuration-distribution-aware estimator
  dominates both configuration-specific and configuration-agnostic ones
  (L96: 0.473 / 1.081 at `keep_k` 16 / 0, vs 0.486 / 1.292 and 0.537 / 1.082).
- **C4 (H1+H3 ⇒ S-b).** The ensemble-Kalman class carries `S(Ψ_NG, 0)` in full by
  construction, so its posterior characterization has a **structural** ceiling —
  and the alternatives measured so far buy calibration or accuracy, never both.
- **C5 (attribution).** Each hypothesis can be relaxed separately, and the
  resulting gain attributed to a named term of the operator partition. This is
  what makes the paper a diagnosis rather than a benchmark.
- **C6 (H2×H3, the corollary).** The ODE representation's sensitivity to
  `(theta, z)` is **not a proxy** for the sensitivity of `p(x | y, theta, z)` to
  `(theta, z)`. DA conflates them because its estimator is the model; a
  conditioned amortized estimator separates them; and the gap is large — 1.68-1.80x
  against ≤1.5%. Assessing forcing/parameter uncertainty by propagating it
  through the model therefore overstates its effect on the posterior whenever the
  observations are informative.

---

## 6. Experiments

**D0 — L96 Weak-4DVar. Prerequisite, blocks C1.** Weak-constraint 4D-Var is the
formulation that *explicitly relaxes H2*, so without it C1 is argued against a
strong-constraint strawman. Implemented (`evaluation/baselines.py`, wired in
`evaluation/run_l96.py`) but `--skip-weak` is set in essentially every L96 sbatch;
L63 and QG both have it and it is best-or-near-best under model error.
**Highest value, lowest cost item in this document.**

**D1 — H1 isolated.** Filter vs window-smoother vs window-amortized at matched
model, matched observations, matched compute. Turns §4.6's ordering into
evidence by removing the capacity/learning confound.

**D2 — H2 dose-response, on a common axis.** Sweep model error continuously
(parameter bias, structural error, resolution) rather than the binary S0/S1, and
measure dSkill/d(model error) per method class. The axis must be one both classes
feel: the *same* biased parameter set given to the DA method's forward model and
to the learned scheme's conditioning inputs, swept together. `noisy_da_max`
already parameterizes exactly this for the conditioned learned rows
(`data/dataloader.py:99-108`), so the sweep is largely a matter of running it.
Prediction: classical DA's slope is steep, the conditioned learned slope is
near-flat, and §4.1.1 predicts *why* — the learned schemes barely use the
parameters.

**D2b — close the conditioning ladder on the deterministic schemes.** Partly
done already: `L6` is a forcing-conditioned `tau=0` VanillaCFM (i.e. a
conditional-mean estimator) with a matched unconditioned control in `L2b`, and QG
has the DirectUNet ladder Q2-Q5. What is **missing on L96 is params/forcing
conditioning for DirectUNet and FDV specifically** —
`cond_extra_dim`/`state_cond_extra_dim` already exist in `conf/schema.py`, so
this is cheap. It is the arm that lets the mean-slot family be compared to DA on
the model-information axis rather than only on accuracy.

**D3 — S-a, the marginal-value surface.** dSkill/d(density) over a 2-D grid of
(observation density × model error), for every method class, on matched axes.
§4.2 is one row of this table for three classical methods; the neural rows do
not exist at *temporal* densities (see R3).

**D4 — H3 by re-parameterization.** The QG ψ/q result generalized: assimilate the
same system in several state variables and relate the skill difference to the
conditioning of the map between them. L96 admits a slow/fast analogue.

**D5 — S-b, done properly.** Rank histograms, spread/skill against
`sqrt((N−1)/(N+1))`, CRPS/ES at a single uniform ensemble convention, across all
method classes. **No code in this repo implements a rank histogram** -- the term
appears only in three planning docs, always as a gap -- and the consolidated
benchmark still mixes `N=1` proxies with proper `N=30` scores under one
bold-best marker. This is the largest new-code item in the plan.

**D7 — The two sensitivities on one grid (C6).** Measure model sensitivity
(spread of free forecasts under `(theta, z)` perturbation) and posterior
sensitivity (change in the conditioned estimator's posterior at fixed `y` under
the *same* perturbation) over a grid of observation density × perturbation
amplitude. `noisy_da_max` already parameterizes the perturbation
(`data/dataloader.py:99-108`) and the obs-density machinery already exists, so
this is mostly assembly. **Prediction:** the two curves diverge as density rises
and converge as it falls; the crossover density is the paper's second deliverable
number after §4.2's collapse factor.

**D6 — Attribution.** Report `S(Ψ_mean)`, `S(Ψ_G)`, `S(Ψ_NG)` per method per
regime, on both axes (point estimate *and* dispersion — a point-estimate score
cannot price the non-Gaussian branch; see the companion doc's R6).

---

## 7. Risks

**R1 — the S0/S1 axis means different things per method class. Must be stated,
not cashed in.** Three issues, in descending severity:

1. **S1 is a no-op for model-free rows.** DirectUNet, FDV and the unconditional
   SDA1 receive no parameters and run no forward model, so their S1/S0 ~ 1.00
   carries no information about robustness. QG's Q1 is the starkest case: its S1
   numbers are bit-identical to its S0 numbers. **These rows must never be tabled
   as robustness evidence.** The *conditioned* rows (SDA2/SDA3, QG Q2-Q5) are a
   different matter — §4.1 shows they genuinely do receive corrupted parameters
   at inference.
2. **Input perturbation is not model-error propagation.** Even for the
   conditioned rows, being handed biased parameters is not the same as
   integrating a biased model for a 500-step window. Any cross-class claim has to
   put the two on a common axis (**D2**) rather than compare a conditioning
   corruption against a compounding one.
3. ~~**The learned schemes' training set already contains S1-level error.**~~
   **RESOLVED — the control was already run.** `train_forcing_state_bias`
   defaults to 0.1, matching `test_s1`, so most rows do see S1-like error in
   training. But `SDA2_cond_nominal_l96` is trained at `forcing_state_bias=0.0`
   and its guided 30-member evaluation gives S0 0.7046 / S1 0.7033,
   **ratio 0.998** — indistinguishable from the S1-exposed SDA2-mixed (0.997).
   Training-time exposure does **not** explain the invariance. The result exists
   on disk and is simply **not in the consolidated benchmark**, which is a
   reporting gap, not an experimental one.

*Mitigation:* (a) state 1 and 2 rather than cash them in; (b) run **D0**, since
weak-constraint 4D-Var is the classical method *allowed* to know about model
error; (c) **report** the SDA2-nominal / SDA2-mixed / L6-vs-L2b rows that already
exist; (d) **D2** and **D7** to put conditioning corruption and model-integration
error on one axis. §4.2's intra-DA result needs none of this and is publishable
on its own.

**R2 — "your baselines were under-tuned".** The standard rebuttal to any
DA-vs-ML comparison. *Mitigation:* the QG localization/inflation sensitivity
studies (§4.5) and the 4D-Var `b_var_scale`/`q_var_scale` sensitivity study —
which was a **negative** result, finding the existing defaults already best — are
exactly the evidence that the tuning was mapped. Fold both in as an appendix
rather than leaving them as separate reports.

**R3 — apples-to-apples across the sparsity axis.** The DA obs-density study
varies *temporal* density (`obs_interval` 200/100); the neural obs-density study
varies *channel* density (`keep_k` ∈ {16,8,4,0}). **These are different axes and
must not be tabled together.** D3 exists to fix this. Also note §4.1 and §4.2
come from different configs (Obs30/DWS=500 vs NO=8/all-5-param) and may not be
cross-quoted.

**R4 — the circularity objection.** An amortized estimator needs a simulator to
generate training data. If that simulator is the truth model, the comparison has
handed it the truth — and in operations there is no truth model. This is the
deepest objection to the whole line and it is **not** fully answerable.
*Mitigation:* frame the contribution as being about **where prior information
enters and what it costs**, not as "learned beats physical". A hybrid that trains
on a biased model and is tested against a different truth is the honest arm.

**R5 — H2 is a strawman as stated.** No practitioner believes model fidelity is
the *only* route to skill; `Q` matrices, inflation and weak-constraint
formulations all exist precisely because model error is understood.
*Mitigation:* state H2 as *binding*, not as naive — the claim is that the
available buffers are weak, which D0 and D2 measure rather than assert. If D0
shows weak-constraint 4D-Var closes most of the gap, **C1 must be softened to a
statement about strong-constraint and filtering methods**, and that is a
publishable result too.

**R6 — QG's own limits.** The `psi_state` q-field degeneracy (§4.3) is a real
representation effect, but `expvar_full` reports the q-field score, so some of
the dramatic numbers are metric choice as much as physics. Report per-field.

---

## 8. Scope control

**Minimum viable: L96 + QG**, with D0, D2, D3, D5 as the core and D1/D4 as the
mechanism sections. Both systems already have the S0/S1 ladder, the DA
baselines, the obs-density machinery and the tuning sensitivity studies.

**Phasing.**

**P0a — pure reporting, no compute.** Three results already sit on disk and are
absent from `l96_consolidated_benchmark.md` (all four rows are `—` placeholders):
`L2b` vs `L6` (forcing conditioning, plain VanillaCFM), and `SDA2-mixed` vs
`SDA2-nominal` (the training-exposure control). These carry §4.1.1 and R1.3 and
cost only an evaluation-to-table pass.

**P0b — one cheap run: D0** (L96 Weak-4DVar). It gates C1 and re-anchors the
companion doc's 93%/7% split. Still the single highest-value item here.

P1: **D7** then **D2 + D3** — D7 (the two sensitivities on one grid) is the
cheapest route to C6, the paper's sharpest methodological claim, and largely
assembles existing machinery. P2: D5 (posterior diagnostics — the largest
new-code item; no rank histogram is implemented anywhere in the repo). P3: D1,
D2b and D4. P4: D6 attribution, shared with the companion doc.

---

## 9. Positioning

*Cited from memory; verify every one before a bibliography exists.*
**Carrassi, Bocquet, Bertino & Evensen** (WIREs Clim. Change 2018) for the DA
review framing; **Trevisan & Uboldi** (JAS 2004) and **Bocquet & Carrassi**
(Tellus A 2017) on assimilation in the unstable subspace — the closest existing
account of *why* DA's effective degrees of freedom are limited; **Trémolet**
(QJRMS 2006) on weak-constraint 4D-Var; **Desroziers et al.** (2005) for the
innovation diagnostics (whose assumptions #211 found violated when the estimator
is trained to fit the observations); **Bonavita / Laloyaux / Geer** on
operational model-error treatment. On the ML side: **Brajard et al.** (2020),
**Bocquet et al.** (2020) and **Farchi et al.** (2021) on combining DA with
learned model-error correction; **Fablet et al.** (4DVarNet); **Rozet & Louppe**
(SDA, 2023).

**The gap.** The model-error literature asks how to *correct* the model; the
unstable-subspace literature explains the dimensionality of the analysis; the
ML-DA literature proposes hybrids. **None of them measures the marginal value of
observations as a function of model error**, or attributes the deficit to named
components of the inference operator. §4.2 is, as far as this repo's survey
goes, an unreported effect.

**Venue.** JAMES fits: methodological, geophysical case studies, and it publishes
diagnosis papers. The two case studies (L96, QG) are idealized, which JAMES
accepts for mechanism papers, but a referee will ask about realism — own it in
scope rather than over-claim transfer to operational systems.

---

## 10. Open decisions

1. **The split with the companion ML doc.** The matched-budget mean-slot
   comparison (now the ML paper's headline) and D1/D2 here share evidence.
   Decide the boundary before either set of experiments is run twice. *Leaning:*
   ML paper owns the **architecture/inductive-bias** axis at fixed model error;
   this paper owns the **model-error and observation-sparsity** axes.
2. Does the paper claim anything **prescriptive**, or is diagnosis the whole
   contribution? Diagnosis is safer and matches the evidence; a prescription
   invites the method-paper critique this framing was chosen to avoid.
3. **Is QG's `psi_state` result strong enough to carry C2 alone**, or does D4
   need the L96 analogue before C2 is claimed?
4. How much of the S1 asymmetry (R1) can be fixed within this project's compute
   budget, versus stated as a limitation? This is the single largest determinant
   of how strong C1 can be.
5. Reporting convention for S-b: adopt the DA literature's rank
   histogram + spread/skill, or the ML literature's CRPS/ES? *Leaning:* both,
   with rank histograms as the primary figure, since the target audience reads
   those.
