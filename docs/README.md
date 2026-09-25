# `docs/` — layout and index

## Layout

Split first by **purpose**, then named `<case>_<topic>.md`, where `<case>` is
`l63`, `l96`, `qg`, `cgoa` or `multi` (spans several case studies).

| folder | holds | lifecycle | `Status:` vocabulary |
|---|---|---|---|
| `plans/paper/` | paper scoping: claims, the evidence each claim needs, venue, the boundary with the other papers | superseded in chains; superseded docs move to `plans/paper/superseded/`, whose Status must be SUPERSEDED | SCOPING, SUPERSEDED |
| `plans/analysis/` | question-driven experiment / analysis plans: hypothesis, runs, controls, decision criteria | superseded in chains; CLOSED once a synthesis in `results/` answers it | PLAN, DESIGN, DRAFT, CLOSED, SUPERSEDED |
| `plans/case_study/` | proposals for a new testbed: physics, scenarios, implementation plan | superseded in chains | PROPOSAL, SCOPING, CLOSED, SUPERSEDED |
| `plans/tech/` | codebase, infrastructure, test design, repo layout — no science | edited in place | DESIGN, DRAFT, PROPOSAL, SCOPING, NOTES |
| `results/` | measurement records and investigation notes — *what was found* | **append-only.** Do not edit one to reflect a later finding; write a new doc that states the retraction (as `docs/results/l96_psi_decomposition.md` does for conclusion 5 of `docs/results/l96_cfm_affine_velocity_decomposition.md`). | RESULTS, NOTES |
| `papers/` | paper drafts (tex / pdf) and their Overleaf sync — not repo prose | external | — |

`CONTRIBUTING.md` stays at this level. Generated number tables live under
`reports/*/outputs/`, not here.

## Conventions

- **Every doc under `plans/` and `results/` has a `**Status:**` line** directly
  under its title, whose first word is in its folder's vocabulary above, and
  which says by what it is superseded when it is.
- **Cite docs by repo-relative path** (`docs/plans/analysis/foo.md`), never by bare
  filename. Bare names were what let `models/sda.py:24` come to cite
  `docs/phase_D_l96_sda.md`, a file that has never existed on any branch.
- Both are enforced by `tests/test_docs_layout.py`, which also fails on any
  cited `docs/...` path that does not exist.
- Report outputs under `reports/*/outputs/` are generated; fix the generator
  **and** the checked-in output together, or the next regeneration reverts you.

## Paper scoping — which link is live

### The ML paper ("is the true prior the best prior?")

Read top to bottom; only the first is current.

| doc | status |
|---|---|
| **`docs/plans/paper/multi_ml_prior_knowledge.md`** | **CURRENT.** ML-venue scoping, v1 2026-09-16, revised 2026-09-17 to absorb the FM-prior sampler results (#211, #212). |
| `docs/plans/paper/superseded/multi_best_prior_approximate_inference.md` | Superseded **in framing** by the above; its testbed ladder and risk register were folded in. Retained for provenance. |
| `docs/plans/analysis/multi_phaseD_physics_priors.md` | Framing superseded in turn, but its **work packages are live** — it is the dynamical-systems instantiation and evidence base, hence filed under `analysis/`. |
| `docs/plans/paper/superseded/multi_cfm_da_originality_notes.md` | Discussion notes (2026-09-02) whose positioning sections Phase D superseded. |

> **Do not cross-cite the first two by question number.** The
> observation-configuration question ("does the optimal prior depend on `H`?")
> is **Q2** in the superseded doc and **Q3** in the current one, which inserts a
> new Q2 on the posterior-targeting objective. "§7.1 Q2 — partially answered"
> and "Q3 partially answered" are the *same* finding on the *same* evidence.

### The DA-venue papers (JAMES)

| doc | status |
|---|---|
| **`docs/plans/paper/multi_p1_structural_hypotheses.md`** | **CURRENT**, SCOPING v2 2026-09-20. P1, the diagnosis paper: the structural hypotheses of classical DA — **H1** Markovianity; **H2** model fidelity as the route to skill; **H3a** ODE state representation; **H3b** autoregressive error propagation — and how they explain DA's limited return on sparse observations and its impoverished posteriors. It names **H4** (the value of gradients/adjoints) but keeps only its classical corollary C7; the constructive half is **assigned to P2, which has not yet accepted it** (§10 decision 6). Draft: `docs/papers/p1_structural_hypotheses/`. |
| **`docs/plans/paper/l96_p2_unrolled_solvers_and_flows.md`** | **CURRENT**, SCOPING v1 2026-09-18. P2, the method paper and sequel to the 4DVarNet line: an unrolled variational solver is the degenerate (`Ψ_mean`-only) member of the conditional-flow family; the continuous blend; the negatives. **S0 only.** |

**The three-paper split is settled** (P2 §9): the ML paper owns families
A/B/C1/D′ and is **deliberately non-DA**; **P1** owns the model-error axis on
L96/QG; **P2** owns the operator family and the observation-sparsity axis at S0.
Two boundaries must not blur — calibration (P1 diagnoses, P2 measures a sampler)
and observation density (P1's *density × model-error* grid vs P2's *density
sweep at S0*, which **must share one protocol**).

> **Standing caveat: these docs quote the benchmark by digit**, mostly from
> `reports/l96/outputs/l96_consolidated_benchmark.md`. That report is
> **regenerated**, and a regeneration can move them: #219 shifted the three
> Stier values by 1 in the last decimal (0.4011→0.4012, 0.3728→0.3729,
> 0.3410→0.3411). Nothing guards the docs against the report —
> `tests/test_l96_report_consistency.py` guards the report against the archive.
> **After any regeneration, re-grep these docs for the values they cite.**

> **Standing caveat for both DA papers.** The L96/QG `S0`/`S1` axis prices model
> error **only for methods that run a forward model at inference**. `param_bias`
> reaches the DA model's parameters, not the truth, so a model-free neural row's
> `S1/S0 ~ 1.00` is definitional, not a robustness result. Never table it as a
> cross-class robustness comparison
> (`docs/plans/paper/multi_p1_structural_hypotheses.md` §4.1, R1).

## Analysis plans

| doc | status |
|---|---|
| `docs/plans/analysis/l96_cfm_tau_consistency_next_steps.md` | **CLOSED** 2026-09-25 (last revision v4). Outcomes, the one identified model failure (PredictStateCFM's late-τ Jacobian) and open items are in **`docs/results/l96_cfm_tau_consistency_synthesis.md`** — cite that, not the seven notes it summarises. |
| `docs/plans/analysis/multi_phaseD_physics_priors.md` | DESIGN; live work packages (see the ML-paper chain above). |
| `docs/plans/analysis/l96_phaseB_cfm_variants.md` | DESIGN + PARTIAL IMPLEMENTATION (V2/V3 CFM variants). |
| `docs/plans/analysis/l96_phaseC_joint_da.md` | PLAN, designed for execution — branch `feature/l96-joint-da-benchmark`. |
| `docs/plans/analysis/l96_phaseC_joint_neural.md` | PLAN, designed for execution. |
| `docs/plans/analysis/l96_joint_additional_metrics.md` | PLAN, in progress since 2026-08-26. |
| `docs/plans/analysis/l63_expG_tau0_cfm.md` | PLAN only — motivation, expected outcomes, run command; no results section. Outcomes live in the L63 reports. |

## Case-study proposals

| doc | status |
|---|---|
| `docs/plans/case_study/cgoa_coupled_gyrostats.md` | **SCOPING** v2 2026-09-24. Coupled ocean–atmosphere from coupled gyrostats (Tier A) plus MAOOAM through qgs (Tier B), on one quadratic-tensor torch engine, built on shared pieces extracted in a no-op PR rather than mirroring the L96 stack. Nothing implemented. Paper fit (P1 / P2 / standalone) is open, §9. |
| `docs/plans/case_study/qg_gyrostat_synoptic_forcing.md` | **SCOPING** v1 2026-09-25. Extends the QG case study (not a new testbed): replace the kinematic OU storm with a gyrostat low-order atmosphere to study the upper-ocean response to synoptic variability. Five options, from a one-way parametric driver (recommended first, with a phase-randomized surrogate control) to two-way mechanical coupling and an ocean gyrostat reduced model. The QG benchmark default is unchanged. Nothing implemented. |

## Tech

| doc | status |
|---|---|
| `docs/plans/tech/multi_refactor_plan.md` | **CURRENT.** Contributor-facing refactor (codebase structure). Phase 0 landed 2026-09-17; Phases 1-4 not started. |
| `docs/plans/tech/multi_test_suite_redesign.md` | PROPOSAL v1 2026-09-22; Phase A1 implemented. |
| `docs/plans/tech/qg_hydra_migration.md` | DRAFT v1 2026-09-19. Moving QG's argparse entry point onto Hydra without orphaning the archived checkpoints; nothing implemented. |
| `docs/plans/tech/l96_cond_extra_dim.md` | DESIGN — `cond_extra_dim` conditioning separation (Option B1). |
| `docs/plans/tech/l96_fdv_torch_compile_notes.md` | NOTES — why `torch.compile` / JAX were shelved for FDV. |
| `docs/plans/tech/archive.md` | NOTES — run-artifact layout and the `evaluation.archive` resolver. |
| `docs/plans/tech/worktrees.md` | NOTES — one worktree per topic branch. |

## Results

Filed by case study; read the plan that produced a note for its context. Where a
thread has a synthesis note, cite it rather than the notes it summarises:

| thread | synthesis |
|---|---|
| L96 τ-consistency of the CFM operators | `docs/results/l96_cfm_tau_consistency_synthesis.md` (#261) |
 The
L96/QG/L63 number tables are under `reports/*/outputs/`.
