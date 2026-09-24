# Scoping & design docs — what is current

Scoping docs here **supersede each other in chains**: a newer one takes over the
framing while the older one's work packages or evidence survive. Nothing in the
filenames tells you which link is live, so this index does.

## The "is the true prior the best prior?" chain

Read **top to bottom**; only the first is current.

| doc | status |
|---|---|
| **`ml_paper_exploiting_prior_knowledge.md`** | **CURRENT.** ML-venue scoping, v1 2026-09-16, revised 2026-09-17 to absorb the FM-prior sampler results (#211, #212). |
| `best_prior_approximate_inference.md` | Superseded **in framing** by the above; its testbed ladder and risk register were folded in. Retained for provenance. |
| `phase_D_physics_priors_under_uncertainty.md` | Framing superseded in turn, but its **work packages are live** — it is the dynamical-systems instantiation and evidence base. |

> **Do not cross-cite these two by question number.** The
> observation-configuration question ("does the optimal prior depend on `H`?")
> is **Q2** in `best_prior_approximate_inference.md` and **Q3** in
> `ml_paper_exploiting_prior_knowledge.md`, which inserts a new Q2 on the
> posterior-targeting objective. "§7.1 Q2 — partially answered" and "Q3
> partially answered" are the *same* finding on the *same* evidence.

## The DA-venue papers (JAMES)

| doc | status |
|---|---|
| **`da_paper_structural_hypotheses.md`** | **CURRENT**, SCOPING v2 2026-09-20. Diagnosis paper: the structural hypotheses of classical DA — four argued (**H1** Markovianity; **H2** model fidelity as the route to skill; **H3a** ODE state representation; **H3b** autoregressive error propagation) and how they explain DA's limited return on sparse observations and its impoverished posteriors. v2 re-cut the hypothesis set — no evidence changed. It also names **H4** (the value of gradients/adjoints) but keeps only its classical corollary C7; the constructive half is **assigned to P2, which has not yet accepted it** (§10 decision 6). |
| **`p2_unrolled_solvers_and_flows.md`** | **CURRENT**, SCOPING v1 2026-09-18. Method paper, sequel to the 4DVarNet line: an unrolled variational solver is the degenerate (`Ψ_mean`-only) member of the conditional-flow family; the continuous blend; the negatives. **S0 only.** |

**The three-paper split is now settled** (P2 §9): the ML doc owns
families A/B/C and is **deliberately non-DA**; **P1** owns the model-error axis
on L96/QG; **P2** owns the operator family and the observation-sparsity axis at
S0. Two boundaries must not blur — calibration (P1 diagnoses, P2 measures a
sampler) and observation density (P1's *density × model-error* grid vs P2's
*density sweep at S0*, which **must share one protocol**).

> **Standing caveat: these docs quote the benchmark by digit.** The scoping docs
> cite exact values from `reports/l96/outputs/l96_consolidated_benchmark.md`
> (DA baselines, the Stier rows, the SDA monai trio). That report is
> **regenerated**, and a regeneration can move them: #219 shifted the three Stier
> values by 1 in the last decimal (0.4011→0.4012, 0.3728→0.3729,
> 0.3410→0.3411), which the docs were refreshed for on 2026-09-18. Nothing
> guards the docs against the report — `tests/test_l96_report_consistency.py`
> guards the report against the archive. **After any regeneration, re-grep these
> docs for the values they cite.**

> **Standing caveat for both DA papers.** The L96/QG `S0`/`S1` axis prices model
> error **only for methods that run a forward model at inference**. `param_bias`
> reaches the DA model's parameters, not the truth, so a model-free neural row's
> `S1/S0 ~ 1.00` is definitional, not a robustness result. Never table it as a
> cross-class robustness comparison. (`da_paper_structural_hypotheses.md` §4.1,
> R1.)

## Standalone plans

| doc | status |
|---|---|
| `phase_B_l96_cfm_variants.md` | DESIGN + PARTIAL IMPLEMENTATION (V2/V3 CFM variants). |
| `phase_C_l96_joint_da.md` | Designed for execution — branch `feature/l96-joint-da-benchmark`. |
| `phase_C_l96_joint_neural.md` | Designed for execution. |
| `experiment_G_tau0_cfm.md` | Plan only — motivation, expected outcomes, run command; **no results section**. Outcomes live in the L63 reports. **No `Status:` line**; filed by content. |
| `cond_extra_dim_plan.md` | **No `Status:` line** — unclassified, filed here on its `_plan` name alone. |
| `refactor_plan.md` | Contributor-facing codebase refactor plan (added by #215). Phase 0 (CI gates the whole `tests/` tree) done; Phases 1-4 open. Not a science doc — filed here because it is a forward-looking plan. |
| `joint_additional_metrics_plan.md` | **No `Status:` line** — unclassified, filed here on its `_plan` name alone. |
| `cfm_tau_consistency_next_steps.md` | **DRAFT**, v4 2026-09-24. Re-targeted at defect 2 (variance collapse) after Batch 1 came back negative. S1 sampler test done: stochasticity is rejected, and early-fine Euler steps are a free win. Next: T5, a second-order (per-channel Jacobian-trace) consistency loss, 2 values of λ × 2 seeds, plus the T0b reseed. T2a′ is optional. Decisions in §6. |
| `qg_hydra_migration.md` | **DRAFT**, v1 2026-09-19. Moving QG's argparse entry point onto Hydra without orphaning the archived checkpoints. Nothing implemented; three staged PRs proposed. Not a science doc. |
| `refactor_plan.md` | **CURRENT.** Contributor-facing refactor (codebase structure, not science). Phase 0 landed 2026-09-17; Phases 1-4 not started. |

Measurement records for this work are in `docs/results/`; the L96/QG/L63 number
tables are under `reports/*/outputs/`.
