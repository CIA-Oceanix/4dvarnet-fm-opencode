# 4DVarNet-FM Project Guidelines

## Session Workflow

Every opencode session in this repository MUST follow this workflow:

1. **Read PLAN.md first** — Before making any changes, read `PLAN.md` to understand the current design plan.
2. **Implement changes** — Make the requested code modifications.
3. **Verify** — Run the relevant test/lint commands (see below).
4. **Log changes** — Append a dated entry to `CHANGELOG.md` describing what was implemented, why, and any notable design decisions.

### Working tree selection (IMPORTANT)

This project uses one physical worktree per topic branch:

| Worktree dir | Branch | Topic |
|---|---|---|
| `/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode` | `master` | Integration / PR landing |
| `.../4dvarnet-fm-opencode/4dvarnet-fm-joint-da` | `feature/l96-joint-da-benchmark` | Joint DA (ETKF) topic |
| `.../4dvarnet-fm-opencode/4dvarnet-fm-cfm-v2v3` | `feature/l96-v2v3-pure` | V2/V3 (TweedieCFM + PredictStateCFM) topic |

**Every opencode session MUST start in the appropriate working tree.** Branch-pinning guarantees you cannot work on the wrong branch, but this only works if you `cd` into the topic directory first:

- **Joint-DA session**: `cd .../4dvarnet-fm-opencode/4dvarnet-fm-joint-da && opencode`
- **V2/V3 session**: `cd .../4dvarnet-fm-opencode/4dvarnet-fm-cfm-v2v3 && opencode`
- **Main repo changes**: `cd .../4dvarnet-fm-opencode && opencode`

**Worktree bootstrap rule:** When starting a topic session, if `AGENTS.md`, `PLAN.md`, or `CHANGELOG.md` were last updated on the master branch **before the topic was forked** (check their timestamps), manually sync them from master first:

```bash
git pull origin master --dry-run  # fast check: origin/master newer?
# If newer: git pull origin master    # sync merged docs first
```

Commit-before-switch rule: **Never leave uncommitted edits when switching worktrees.** Always commit or stash your unfinished work before checking out a different branch.

## Git / PR Workflow

These general requirements apply to code changes in **every** session (not just L96).
The canonical, current version of these rules lives here; PLAN.md may carry
L96-specific details and pointers.

### Branching

- New work branches use the `feature/<topic>` prefix (e.g. `feature/session-workflow-automation`).
- `feat/*` names are **reserved for integration branches** (e.g. `feat/l96-neural-eval-fix`).
  A repo ruleset **blocks direct pushes of new `feat/l96-*` branches**, so always create
  `feature/...` branches for new work.
- Check the active integration branch (remote / PLAN.md) and use it as the PR base.

### Run-to-completion policy (IMPORTANT)

Once the user gives a go-ahead (e.g. "go", "proceed", "approved"), the implementing agent
drives a subtask **all the way to a merged PR without pausing for another approval between
the create → CI → review → merge steps**. The loop is fully automated (approval is done by
the `rfablet-review` identity and the `pytest` CI check is the merge gate), so nothing a human
must decide sits in the middle. Concretely, after pushing and opening the PR, continue:
wait for the `pytest` check to pass, run the reviewer approval, then verify and squash-merge.
Do **NOT** treat "PR created" as a natural stopping point.

Stop for user input only on a genuine external blocker:

- a reviewer **request-changes** (must be fixed with a new commit)
- a **non-informational CI failure** (informational ruff `continue-on-error` does not block)
- a **merge conflict**
- a **checkpoint the user explicitly asked to review**

### Review + merge

- Approvals come from the second GitHub account **`rfablet-review`** — not the author
  (GitHub blocks self-approval). Use `scripts/open_pr.sh review <PR#>` with the reviewer
  token at `~/.config/opencode/reviewer-token`.
- Pipeline: `scripts/open_pr.sh create "<msg>"` → wait for CI → `review <PR#>` →
  `verify <PR#>` (squash-merge).
- **Merge gate = `pytest -m "not slow"`** on the required test files. Ruff is informational
  (`continue-on-error: true`) so `mergeStateStatus: UNSTABLE` does not block the merge.
- Before merging, run the fast tests and `ruff check` on touched files locally.

### Hygiene

- Never commit artifacts/checkpoints or untracked scratch files (`experiments/` is gitignored).
- Add a CHANGELOG.d/ fragment (see format below) for every merged change.
- Don't reformat or restyle unrelated code in the same change — keep diffs scoped to
  the requested task, even when touching a file that could use a broader cleanup.

## Changelog Format

**Do NOT edit `CHANGELOG.md` directly.** Editing a single shared file's top
section is exactly what makes every concurrently-open PR conflict with every
other one (each inserts its entry at the same spot). Instead, add a new file
to `CHANGELOG.d/` as part of the same PR that makes the change — new files
never conflict with each other in git, which removes that conflict class
entirely. See `CHANGELOG.d/README.md` for the full convention.

Filename: `YYYY-MM-DD-<short-slug>.md`. Content — the same format as before,
just in its own file now:

```
## YYYY-MM-DD: Short Title

**Summary:** 1-2 sentence description of changes.
**Files modified:** `path/to/file.py` — brief note
**Rationale:** Why this change was made.
**Verification:** Test command run and result.
```

Periodically (not part of any individual PR), run
`python scripts/assemble_changelog.py` to fold all pending fragments into
`CHANGELOG.md` (newest first) and delete the consumed fragment files.

## Conda environment

**`fdv-monai-proto` is the project's single default env** (torch 2.8.0+cu126 +
monai 1.6.0, pinned together in `requirements.txt`):

```bash
/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin/python ...
```

Batch scripts default to it (`CONDA_ENV="${CONDA_ENV:-fdv-monai-proto}"`, or a
direct `envs/fdv-monai-proto/bin/python` path); override per-run with
`CONDA_ENV=<other>` where a script reads it.

The older `fdv` env (torch 2.4.1+cu121, no monai) is **kept frozen as a
rollback path** — two past incidents had a concurrent process break the shared
env under running jobs — but nothing should point at it any more. Before the
switch, `fdv-monai-proto` was verified to be a strict superset of `fdv`
(identical Python and all 356 packages at identical versions apart from
torch/triton/CUDA libs + monai), with working CUDA on every GPU tier this
project uses (sm_75 RTX8000 -> sm_90 H100/H200), bit-identical numerics, and
no test or throughput regression.

**`torch.load` note:** torch >= 2.6 defaults to `weights_only=True`, which
rejects pickled Dataset objects and numpy arrays (e.g. the L96
`dataset_cache/*.pt` caches). Project code passes `weights_only=False`
explicitly when loading its own artifacts — keep doing so in new code.

## Build, Lint, and Test Commands

- **Lint:** `ruff check .` — check code quality
- **Type check:** `mypy .` — static type analysis
- **Tests:** `pytest tests/ -v` — run full test suite
- **Quick test:** `pytest tests/ -v -m "not slow"` — skip slow tests
- **Coverage:** `pytest tests/ --cov=. --cov-report=term`

Always run tests after making changes.

## Project Structure

- `data/` — Lorenz-63 SDE simulation, datasets, dataloaders
- `models/` — Neural network architectures (UNet1D, TweedieSolver, etc.)
- `training/` — Training pipelines (Lightning-based)
- `evaluation/` — Baselines (4D-Var, EnKF, ETKF) and metrics
- `conf/` — Hydra structured config schemas
- `config/` — YAML configuration presets
- `reports/` — Report generation scripts
- `batch/` — SLURM batch scripts for HPC
- `tests/` — Unit and integration tests
- `docs/` — `plans/{paper,analysis,case_study,tech}/`, `results/`, `papers/`; layout, naming
  (`<case>_<topic>.md`) and the index of live docs in `docs/README.md`, enforced by
  `tests/test_docs_layout.py`
- Top-level entry points: `train.py` (Hydra-driven training), `run_experiment.py` /
  `run_experiments.py` (single/batch experiment runners), and per-case-study
  `eval_*.py` / `evaluate_all*.py` scripts (e.g. `eval_baselines.py`,
  `evaluate_all_l96.py`, `train_qg_neural.py`) — these multiply per topic branch,
  so this list isn't exhaustive; `ls *.py` in the relevant worktree is authoritative.

## Key Conventions

- **Python 3.10+** with `torch`, `numpy`, `hydra-core`, `pytorch-lightning`
- **Configuration** uses Hydra/OmegaConf (see `conf/schema.py` for dataclass schemas)
- **No comments** in code unless absolutely necessary (prefer self-documenting names)
- **Type hints** should be used for all function signatures
- **Training** uses PyTorch Lightning (`LitModel` wrapper in `training/lightning_module.py`)
- **Two-stage training** pattern: Stage 1 trains the mean estimator, Stage 2 freezes it and trains the residual
- **Data** is generated on-the-fly; no large data files committed to git
- **Tests** use `pytest` with markers (`@pytest.mark.slow`) for expensive tests
- **LR scheduling default (as of 2026-09-10):** cosine-annealed LR (`use_cosine_scheduler:
  true`) is the deliberate, standing default for training runs -- not opt-in. This is a
  project policy decision, not an accident: an earlier *unintentional* version of this same
  flip was caught and reverted during PR #176's review specifically because it would have
  silently retrained ~71 untouched configs with no explanation. See CHANGELOG.md's
  2026-09-10 "FDV2 gradient-channel NaN fix + cosine LR scheduler now the deliberate default"
  entry before reverting this default on sight.

- **L96 DA benchmark inflation (as of 2026-09-23):** ETKF/EnKF use inflation **1.5 on S0 and 2.0
  on S1** (`evaluation/run_l96.py::L96_DA_INFLATION`, the default of `evaluate_all_l96.py` and
  `eval_da_random_layout_l96.py`). The earlier single value 2.0 is S1-tuned and over-disperses S0; see
  the 2026-09-23 "L96 DA benchmark default inflation per case" changelog entry before changing it.

- **Flow sampler step schedule (as of 2026-09-24):** `VanillaCFM.sample` / `PredictStateCFM.sample`
  (and their monai subclasses) integrate on the early-fine Euler grid `tau_k = 1 - (1 - k/N)^0.5`
  (`models.vanilla_cfm.DEFAULT_STEP_POWER`), not the uniform grid. This is a deliberate default: at the
  same N = 10 it improved CRPS by 4.4% / 2.0% and RMSE slightly on the P1 PSC-M / VanillaCFM-M
  checkpoints (`docs/results/l96_cfm_sampler_schedule.md`). **Numbers produced before this date used the
  uniform grid**; reproduce them with `eval_neural_l96.py --step-power 1` (or `model.step_power: 1.0`).
  `eval_neural_l96.py` records the grid used under `sampling.step_power`.
- **L96 benchmark flow sampler (as of 2026-09-24):** benchmark flows (VanillaCFM / PredictStateCFM) are
  scored with **30 members × 20 early-fine steps** (`--n-members 30 --n-outer 20`, p = 0.5), eval
  sub-directory `ens30_no20` (was `ens30_no10`, uniform). On the benchmark-default checkpoints this gives
  spread/RMSE ≈ 0.86-1.02 and beats uniform N = 80 on CRPS for PredictStateCFM
  (`docs/results/l96_cfm_tau_consistency_l96b.md`). SDA keeps its own guided protocol (`ens30_gw20`, 10 steps).

## When Making Model Changes

- Update the corresponding config in `config/experiment/` if training parameters change
- Ensure `LitModel` (in `training/lightning_module.py`) handles the new model type correctly
- Add tests for any new model, loss, or dataset in `tests/`
- Document the change in `CHANGELOG.md`

