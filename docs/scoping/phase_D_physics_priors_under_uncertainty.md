# Phase D — How physics knowledge matters under model error and observation sparsity (design doc)

**Status:** DESIGN. No code written for this phase yet. Supersedes the paper-
positioning sections of `docs/results/research_notes_cfm_da_originality_and_benchmarking.md`
(2026-09-02), whose four open items are now all executed and benchmarked.

**Question:** when the forward model is uncertain and observations are sparse,
*how much* does each of the following matter, and in what order —
(i) the inference algorithm, (ii) the representation of physics knowledge
(prior + observation model), (iii) the learning paradigm (end-to-end amortization
vs. solving an unsupervised variational criterion), and (iv) whether the
prior/solver is adapted to the observation configuration?

The headline suspicion this phase tests: **the true ODE is not the best prior**,
and the corollary that reducing forcing/parameter uncertainty — the obvious lever
*within* an ODE-based representation — is not the critical lever for state-estimation
skill. This follows up the 4DVarNet JAMES result (a trainable, distributionally
imperfect prior outperforming the true ODE prior).

---

## 1. Targeted contributions

Numbered so every work package below can name which one it serves. A WP that
serves none is out of scope.

- **C1 — Physics representation dominates the inference algorithm.** Under model
  error, *how* physics knowledge is represented (hard constraint / soft constraint /
  learned prior / none) accounts for more of the performance spread than which
  inference algorithm consumes it.
- **C2 — The true ODE is not the best prior**, even when it is exactly correct (S0),
  and even when used as a soft rather than hard constraint.
- **C3 — Parameter/forcing knowledge is the wrong lever; background state is the
  right one.** Conditioning on true parameters/forcing does not improve state skill
  (and can hurt); conditioning on a background state does. Corollary: better
  parameter recovery does not imply better state estimation.
- **C4 — Adaptation to an observation-configuration *distribution* beats both
  configuration-specific tuning and architectural agnosticism.**
- **C5 (secondary) — End-to-end amortization vs. solving the variational criterion.**
  Quantify the gap between a truncated unrolled solver trained end-to-end (an MMSE
  amortizer with variational-shaped architecture) and a genuinely solved bi-level
  problem (MAP), and relate it to measured posterior non-Gaussianity.

C5 is marked secondary because it is the most expensive and the least certain to
produce a clean result; it should not block C1-C4.

---

## 2. The design space, and why the current benchmark cannot resolve it

Four axes:

| axis | levels |
|---|---|
| **A. inference algorithm** | sequential filter / variational / feed-forward amortized / unrolled / prior+guidance |
| **B. physics representation** | true ODE hard / true ODE soft (weak-constraint) / learned prior / none |
| **C. learning paradigm** | none (classical) / supervised end-to-end / unsupervised criterion (bi-level) |
| **D. obs-config adaptation** | config-specific / config-distribution-augmented / agnostic |

**The problem.** Every method currently benchmarked differs from every other on
three or four axes simultaneously:

| method | A | B | C | D |
|---|---|---|---|---|
| Strong-4DVar | variational | true ODE, hard | none | agnostic |
| Weak-4DVar | variational | true ODE, soft | none | agnostic |
| ETKF / EnKF | seq. filter | true ODE, hard | none | agnostic |
| DirectUNet | feed-forward | none | supervised e2e | config-specific |
| SDA1/2/3 | guidance | learned, unconditional | unsup. prior + unsup. inference | agnostic |
| FDV1 (`obs+state`) | unrolled | none | supervised e2e | config-specific |
| FDV2 (`grad+state`, `subgrad+state`) | unrolled | learned prior cost | supervised e2e | config-specific |

A leaderboard over these rows says which method wins. It cannot attribute the win
to an axis, and C1/C2 are attribution claims. **This is the blocking design issue
for the phase** — it is invisible once runs are launched, so it must be fixed first.

### 2.1 Design principle: harness + corner anchors

Do not treat methods as experimental units. Use **`FourDVarNetSolver` / FDV2 as a
harness** in which each axis can be varied singly with everything else — solver,
optimizer, budget, observation term, capacity tier, LR schedule — held fixed:

- **axis B** ← swap the prior cost: learned `prior_unet` (exists, `_PRIOR_MODES`) /
  true-ODE weak-constraint `‖x − Φ(x)‖²` (**new**, shares its implementation with
  `Weak4DVar`) / none (`update_input='obs+state'`, exists).
- **axis C** ← swap the training criterion: supervised end-to-end (exists) /
  outer criterion with a convergent inner solve (**new**).
- **axis D** ← swap the training obs sampling: fixed density (exists) /
  `data/obs_density.py::sample_training_density_mask` (exists) / prior-only, obs
  entering at inference only (exists via SDA).

The classical DA baselines and the pure feed-forward nets remain in the paper as
**corner anchors** that locate the harness in the wider space — not as the units
from which effects are estimated.

---

## 3. Reference bars (existing, L96 DA-parity set: 24D, Obs30, dws=500, 200 windows)

Pooled RMSE, S0 / S1. Source: `reports/l96/outputs/l96_consolidated_benchmark.md`.

| method | S0 | S1 | S1/S0 |
|---|---|---|---|
| Strong-4DVar (best classical) | 0.8116 | 1.4617 | 1.80 |
| ETKF | 0.8883 | 1.4998 | 1.69 |
| EnKF | 0.9131 | 1.5381 | 1.68 |
| **Weak-4DVar** | **— not run —** | **— not run —** | — |
| DirectUNet-M(monai,cos) | 0.5064 | 0.5072 | 1.00 |
| CFM-M(monai,flat) | 0.4810 | 0.4784 | 1.00 |
| FDV1(monai) | 0.4251 | 0.4218 | 0.99 |
| subgrad+state-Stier(monai) | 0.3729 | 0.3730 | 1.00 |
| subgrad+state-Stier+SDA3(monai) | **0.3411** | **0.3403** | 1.00 |

### 3.1 Evidence already in hand for C3

**L96 — conditioning on true parameters is the *worst* of three options,
replicated in three independent families:**

| prior conditioning | SDA alone | +DirectUNet mean | +FDV1 mean |
|---|---|---|---|
| SDA1 — none | 0.5532 | 0.4274 | 0.3852 |
| SDA2 — **true** params+forcing | 0.5588 | 0.4288 | 0.3845 |
| SDA3 — **noisy** params | **0.5365** | **0.4204** | **0.3787** |

**QG — replicates, and separates *which* knowledge helps** (PV-q EV, from
`reports/qg/outputs/qg_neural_report.md`):

| scheme | S0 | S1 |
|---|---|---|
| Q1 — obs only | 0.6238 | 0.6238 |
| Q2 — **oracle** forcing+params | 0.6093 | 0.5571 |
| Q3 — noisy forcing+params | 0.5978 | 0.5959 |
| Q4 — noisy + **initial condition** | **0.6745** | **0.6744** |

Q2 carries strictly more, and strictly correct, information than Q1 and is worse
on both cases — this is the form of the result that cannot be dismissed as a
noise-robustness artifact. Q4 supplies the positive half: background *state*
knowledge helps substantially where parameter/forcing knowledge does not.

**Decoupling of parameter recovery from state skill:** Joint-ETKF attains the best
parameter RMSE (0.053 on S0, `l96_joint_da_benchmark.md`) with state RMSE 0.635,
while the best state estimator (0.341) estimates no parameters at all.

### 3.2 Evidence already in hand for C4

From `reports/l96/outputs/l96_obs_density_augmented_training.md` (S0 RMSE):

| | keep_k=16 | keep_k=8 | keep_k=4 | keep_k=0 |
|---|---|---|---|---|
| DirectUNet-L, config-specific | 0.486 | 0.906 | 1.104 | 1.292 |
| DirectUNet-M, density-augmented | 0.473 | 0.704 | 0.898 | 1.081 |
| SDA3, agnostic | 0.537 | 0.781 | 0.921 | 1.082 |
| DirectUNet(aug)+SDA3 | **0.389** | **0.633** | **0.848** | **1.065** |

C4 is essentially established on L96; it needs replication elsewhere, not more
L96 work.

---

## 4. Model-error constructs (must be harmonized across systems)

The three systems currently use **different** S1 definitions, which makes any
cross-system table non-comparable as it stands:

- L96 (`data/lorenz96.py:628-686`): `param_bias` biases the **DA forward model
  only**; the truth is unchanged. Train/val already run at
  `forcing_state_bias=0.1`, i.e. test_s1's level.
- QG: same convention — hence Q1's S0 and S1 numbers are *bit-identical*.
- L63 (`reports/l63/outputs/s0_s1_synthesis.md`): perturbs the **truth**
  (coupling exponent 1.6 → 1.0, param/forcing bias), so neural degradation there
  is 1.9-2.3×, not ≈1.00.

**Decision: define two constructs and label every result with which one it uses.**

- **E1 — biased forward model, truth unchanged.** Isolates axis B: only methods
  that *consume* a forward model are exposed. This is the L96/QG convention.
  Note for the write-up: under E1 an obs-only amortized model's S1/S0 ≈ 1.00 is
  partly definitional and must be presented as such, not as evidence of robustness.
- **E2 — truth structurally changed** (reduced/perturbed dynamics generating the
  data). Tests genuine generalization; all methods are exposed. This is the L63
  convention.

At least one system must run **both**. L63 is the affordable place to do it.

---

## 5. Work packages

Each WP states: contribution served · what varies · what is held fixed · status.

### Tier 0 — unblocks everything, no new modelling

**WP-0.1 · L96 Weak-4DVar row** — C1, C2
*Varies:* physics representation, hard → soft, within the variational algorithm.
*Fixed:* everything else (same windows, same biased params, same eval).
*Status:* implemented (`evaluation/baselines.py:357`, wired at
`evaluation/run_l96.py:326`) but excluded by `--skip-weak` in essentially every
L96 sbatch. Drop the flag and run.
*Why blocking:* without it, C2 is refutable by "you only showed that *rigidly*
enforcing a biased ODE is bad, which is already known." Weak-4DVar is the
steelman of the physics prior and it is best-or-near-best under model error in
both L63 and QG.

**WP-0.2 · Harmonize metric conventions and close reporting gaps** — all
*Status:* new analysis, no new compute.
- One convention (pooled) everywhere; the obs-density and obsj0 studies currently
  disagree (pooled vs mean-of-window), which alone can flip a ratio.
- Split ES/CRPS tables by ensemble convention (N=1 MAE proxy vs N=30) — currently
  mixed under one bold-best marker.
- Add calibration: rank histograms + spread-skill. None exist anywhere in the repo;
  raw members are on disk (`members_s0.npz`, `(200,3000,24,30)`), so this is
  analysis, not compute. Required because C2/C3 are partly claims about priors.
- Paired bootstrap / Wilcoxon over the 200 windows; the top rows sit within ~5%.

**WP-0.3 · Cost/NFE accounting** — C1, C5
Params, train GPU-h, NFE/sample, inference s/window for every row. Needed for any
claim comparing a 30-member guided sampler against a 30-member ensemble filter.

### Tier 1 — the core factorial (L96, FDV2 harness)

**WP-1.1 · Prior swap** — **C2 (decisive)**
*Varies:* prior cost only ∈ {learned `prior_unet`, true-ODE weak-constraint
`‖x − Φ(x)‖²`, none}.
*Fixed:* solver, `N_outer`, capacity tier, obs term, optimizer, LR schedule, data.
*New code:* the true-ODE prior cost term — shares its implementation with
`Weak4DVar`'s model-error term, so WP-0.1 is a prerequisite, not a parallel task.
*Reads:* if the learned prior still wins with everything else pinned, C2 is
established by single-variable control rather than cross-method inference.
Run under **both S0 and E1-S1** — the S0 cell is the strong form of the claim
(the ODE is exactly right there).

> **Existing negative result that reshapes this WP.** The learned prior has
> already been tried as an *explicit auxiliary cost*: `aux_var_cost_weight`
> (`models/fourdvarnet.py:183-202`, `_prior_ae`, an AE-style `‖x − AE(x)‖²`
> term) in `FDV1_obsstate_monai_l96_initvar01{,_Stier}_auxpriorcost01.yaml`.
> It **hurt** — job 53289 plateaued at val_loss ≈ 0.39 vs ≈ 0.20-0.29 for every
> aux-loss-free FDV1 variant. Meanwhile the learned prior entering *implicitly*,
> through `subgrad+state`'s gradient channel, gives the best non-hybrid row
> (0.3729).
>
> So the axis is not simply learned-vs-ODE: **how the prior enters the cost
> (explicit penalty vs. gradient conditioning) appears to matter more than what
> the prior is.** WP-1.1 must therefore be a 3 × 2 grid — prior ∈ {learned,
> true-ODE, none} × entry mode ∈ {explicit penalty, gradient conditioning} —
> not a 3-cell swap. Without the entry-mode factor, a true-ODE-beats-learned
> result would be uninterpretable (it could just be reproducing the
> `aux_var_cost_weight` degradation). This is the confounding trap of §2
> reappearing inside the harness itself.

**WP-1.2 · Conditioning-channel ablation** — **C3**
*Varies:* conditioning input ∈ {none, true params+forcing, noisy params+forcing,
background state (IC), true params + IC}.
*Fixed:* architecture, budget, data.
*Status:* partially exists (SDA1/2/3 on L96; Q1-Q4 on QG) but the **background-state
(IC) arm does not exist on L96** — it is the arm that carries C3's positive half and
it is currently QG-only. Add it.

**WP-1.3 · Obs-config adaptation × physics representation** — C4 × C2
*Varies:* axis D ∈ {fixed density, density-augmented, agnostic} crossed with
axis B ∈ {learned prior, true-ODE prior, none}.
*Fixed:* solver and budget.
*Status:* axis D alone is done (§3.2); the cross with axis B is new. This is the
cell that tests whether a *physics* prior is intrinsically more obs-config-robust
than a learned one — a claim worth settling and currently unaddressed.

### Tier 2 — cross-system validation

**WP-2.1 · L63 re-run under the current protocol** — C1-C4 + E1/E2 contrast
L63 is stale (2026-07): pre-monai, pre-normalization, no SDA/FDV legs, and on the
E2 convention. At 7-16 min/model it is the cheapest system by two orders of
magnitude. Re-run the full arm set, and run **both E1 and E2** so the
model-error-construct contrast is available somewhere.

**WP-2.2 · QG replication of headline effects** — C2, C3, C4
QG has DA baselines (incl. Weak-4DVar) and Q1-Q4 DirectUNet, but **no CFM, SDA or
unrolled arm**; Q6 is parked for sound reasons (`PLAN.md` 2026-09-14) and should
stay parked. Add **one** τ=0 CFM leg and **one** SDA prior leg — enough to
replicate C2/C3 at realistic dimensionality without attempting the factorial.
Q1-obsdensity (in flight, `PLAN.md` §1095) covers C4.

### Tier 3 — secondary, do not let it block

**WP-3.1 · End-to-end vs. solved bi-level** — **C5**
*Varies:* learning paradigm only — truncated unrolling (`N_outer=10`, supervised
MSE) vs. a convergent inner solve with the prior fitted by an outer criterion
(implicit differentiation, or many more inner iterations).
*Note:* FDV1/FDV2 **as currently trained sit on the end-to-end side of this axis**
despite their variational shape — the inner loop is truncated and the weights are
fit so the 10-step output matches truth, not so that it minimizes J. Truncated
unrolling is a biased surrogate for the bi-level gradient.
*Target mismatch to report:* supervised MSE targets the posterior mean (MMSE);
variational assimilation targets the mode (MAP). They coincide only under
Gaussianity, and `docs/results/cfm_affine_velocity_decomposition.md` measured that
non-Gaussianity (19-31% of velocity amplitude non-affine) — hedge appropriately
(ρ measures non-affineness of E[x₁|x_τ,y] in flow coordinates, not non-Gaussianity
of p(x₁|y) directly), but the link is real.
*Where:* **L63 only.** It is the one system where a convergent inner solve is
affordable.

---

## 6. System division of labour

| system | role | why |
|---|---|---|
| **L96** | the full factorial (Tier 0-1) | only system where every arm exists |
| **QG** | replication of 2-3 headline effects | realistic dimensionality; lacks CFM/SDA/unrolled arms, Q6 parked |
| **L63** | cheap control + the expensive C5 treatment + E1/E2 contrast | 7-16 min/model |

Deliberately **not** a crossed 4-axis × 2-stressor × 3-system design — that is not
runnable. The structure is: one reference cell per system, single-factor
perturbations on L96, headline replication on QG/L63.

---

## 7. Success criteria / falsification

Each contribution needs a stated way to fail, or the phase is unfalsifiable.

- **C1** fails if the spread across axis B (at fixed algorithm) is smaller than the
  spread across axis A (at fixed representation), measured with WP-0.2's paired
  bootstrap.
- **C2** fails if the true-ODE weak-constraint prior (WP-1.1) matches or beats the
  learned prior on **S0**, where the ODE is exactly correct, *at matched entry
  mode*. A win for the ODE on S0 and a loss on S1 would be a *different*, weaker,
  still-publishable claim ("the ODE prior is right but brittle") and should be
  reported as such. If the entry-mode factor dominates the prior-identity factor,
  C2 is superseded by a sharper claim — *how* physics knowledge is injected into
  the cost matters more than *which* physics knowledge it is — which is arguably
  a better paper and should be adopted rather than resisted.
- **C3** fails if the IC arm (WP-1.2) does not reproduce QG's Q4 advantage on L96,
  or if SDA2 > SDA3 reverses under the paired bootstrap.
- **C4** is already supported; it fails only if QG/L63 replication contradicts it.
- **C5** has no failure criterion yet — it is exploratory, which is why it is Tier 3.

---

## 8. Risks

- **Scoping of C2.** "The true ODE is not the best prior" must be stated as
  conditional on the observation densities and model-error magnitudes studied.
  Two chaotic toy systems plus one QG channel do not license a general claim.
- **Three mechanisms, one slogan.** C2 and C3 rest on *different* mechanisms —
  hard-constraining a biased model, versus a conditioning channel that lets a
  network shortcut around the observations. They point the same way but a referee
  will separate them, so the paper must too. The Q2 < Q1 result needs an explicit
  proposed mechanism, not just the number.
- **Confounding creep.** Every added arm is a temptation to vary two things at
  once. The harness discipline of §2.1 is the mitigation; it must survive contact
  with the sbatch queue.
- **Pre-existing bug on the path.** `evaluation/baselines.py:299`
  (`_ESAccumulator.step` calls `.detach()` on a `numpy.ndarray`) is reachable from
  `Strong4DVar.assimilate` and breaks `tests/test_equiv_report.py`. That test is
  not in the CI gate list, so it rots silently. Fix before Tier 1, since WP-1.1
  exercises the same variational path.

---

## 9. Status

DESIGN ONLY — nothing here is implemented. Immediate next action is **WP-0.1**
(L96 Weak-4DVar), which is load-bearing under every framing considered, requires
no new modelling, and is a prerequisite for WP-1.1's true-ODE prior cost.
