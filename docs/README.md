# `docs/` layout

Three folders, split by **lifecycle** rather than by topic — the three behave
differently and mixing them is what made the flat directory hard to read.

| folder | contents | lifecycle |
|---|---|---|
| `scoping/` | design docs, scoping docs, phase plans — *what we intend to do* | **superseded in chains.** A newer doc replaces an older one's framing while parts of the old one survive. `scoping/README.md` is the index of which link is live. |
| `results/` | measurement records, investigation notes — *what was found* | **append-only.** Do not edit one to reflect a later finding; write a new doc and have it state the retraction (as `psi_decomposition_results.md` does for conclusion 5 of `cfm_affine_velocity_decomposition.md`). |
| `papers/` | draft PDFs, `case_studies.tex` | external artifacts, not repo prose |

`worktrees.md` stays at this level: it describes the repo, not the science.

## Conventions

- **Every doc should carry a `**Status:**` line** directly under its title,
  saying what it is (SCOPING / DESIGN / RESULTS / NOTES) and, when superseded,
  by what. **Five currently do not** — `scoping/cond_extra_dim_plan.md`,
  `scoping/experiment_G_tau0_cfm.md`,
  `scoping/joint_additional_metrics_plan.md`,
  `results/fm_prior_sampler_results.md` and
  `results/joint_estimation_progress.md`. They were filed by content, not by a
  declared status; add one when you next touch them.
- **Cite docs by repo-relative path** (`docs/scoping/foo.md`), not by bare
  filename. Bare names were what let `models/sda.py:24` come to cite
  `docs/phase_D_l96_sda.md`, a file that has never existed on any branch.
- Report outputs under `reports/*/outputs/` are generated; fix the generator
  **and** the checked-in output together, or the next regeneration reverts you.
