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

## The DA-venue paper (JAMES)

| doc | status |
|---|---|
| **`da_paper_structural_hypotheses.md`** | **CURRENT**, SCOPING v1 2026-09-17. Diagnosis paper: three structural hypotheses of classical DA (sequential; model fidelity as the route to skill; ODE state representation) and how they explain DA's limited return on sparse observations and its impoverished posteriors. |

Its relationship to the ML chain above is **not yet settled** — see its §10
decision 1. Working split: the ML doc owns the architecture/inductive-bias axis
at fixed model error; this one owns the model-error and observation-sparsity
axes. They share the L96/QG evidence base, so decide the boundary before running
any experiment twice.

> **Standing caveat for both papers.** The L96/QG `S0`/`S1` axis prices model
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
| `refactor_plan.md` | **CURRENT.** Contributor-facing refactor (codebase structure, not science). Phase 0 landed 2026-09-17; Phases 1-4 not started. |

Measurement records for this work are in `docs/results/`; the L96/QG/L63 number
tables are under `reports/*/outputs/`.
