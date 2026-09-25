# 4DVarNet-FM

Flow matching for variational data assimilation, benchmarked against classical DA
on three dynamical systems.

The question: can a learned generative prior (conditional flow matching, Tweedie
decompositions, unrolled 4DVarNet solvers) match or beat EnKF / ETKF /
strong- and weak-constraint 4D-Var on the same observation geometry and the same
error metrics?

## Case studies

| system | role |
|---|---|
| **Lorenz-63** | smallest testbed; parameter-estimation and randomized-parameter studies |
| **Lorenz-96** (two-scale) | the main benchmark — model error, observation-density sweeps, most neural schemes |
| **QG** (two-layer quasi-geostrophic) | spatially extended reference case; DA baselines on a Phillips channel |

## Getting started

Read **[`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)** — it walks one full
Lorenz-96 loop end to end: generate data, run the DA baselines, train a neural
scheme, evaluate it against them, render the benchmark table.

## Layout

```
data/        per-system truth simulation, observation geometry, datasets
models/      dynamics (per system) + neural architectures
training/    PyTorch Lightning wrappers and the two-stage training pattern
evaluation/  DA baselines (EnKF/ETKF/4DVar) and neural inference + metrics
conf/        Hydra structured-config schemas
config/      YAML experiment presets
reports/     per-case-study report generators and their result JSONs
tests/       pytest suite ('not slow' subset is the merge gate)
batch/       SLURM scripts
```

Results and their provenance live in `reports/<system>/outputs/`. The report
generators render from stored JSON, so a report can always be regenerated
without re-running the science.

## Documentation

| doc | what it covers |
|---|---|
| [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) | worked path, conventions, gotchas |
| [`docs/plans/tech/multi_refactor_plan.md`](docs/plans/tech/multi_refactor_plan.md) | where the code structure is heading |
| [`AGENTS.md`](AGENTS.md) | session workflow, git/PR rules, changelog policy |
| [`PLAN.md`](PLAN.md) | the running scientific design plan |
| [`docs/plans/tech/worktrees.md`](docs/plans/tech/worktrees.md) | the one-worktree-per-topic setup |
