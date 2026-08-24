# 4DVarNet-FM Project Guidelines

## Session Workflow

Every opencode session in this repository MUST follow this workflow:

1. **Read PLAN.md first** — Before making any changes, read `PLAN.md` to understand the current design plan.
2. **Implement changes** — Make the requested code modifications.
3. **Verify** — Run the relevant test/lint commands (see below).
4. **Log changes** — Append a dated entry to `CHANGELOG.md` describing what was implemented, why, and any notable design decisions.

## PR Workflow (Run-to-Completion)

QG work is delivered as PRs into an integration branch (see `PLAN.md`). The
`feat/qg-*` repo ruleset requires **1 approving review by a reviewer with write
access** plus a green `pytest` check.

### Run-to-completion policy

Once the user gives a go-ahead (e.g. "go", "proceed"), drive the subtask **all
the way to a merged PR** without pausing between the create → CI → review →
merge steps. After pushing and opening the PR, continue: wait for the `pytest`
check to pass, run the reviewer approval, then verify and squash-merge. Do
**NOT** treat "PR created" as a stopping point.

Stop for user input only on a genuine external blocker:
- a reviewer **request-changes** (must be fixed with a new commit)
- a **non-informational CI failure** (informational ruff is `continue-on-error`, non-blocking)
- a merge conflict
- a checkpoint the user explicitly asked to review

### Review identity is critical

GitHub blocks self-approval, so approvals must come from the second account
**`rfablet-review`** — never from the PR author. The reviewer agent MUST use
`scripts/open_pr.sh review <PR#>` (or inject the token manually), NOT a plain
`gh pr review`. The reviewer token lives at
`~/.config/opencode/reviewer-token` and is auto-loaded by the script.

- Pipeline: `scripts/open_pr.sh create "<msg>"` → wait CI → `review <PR#>` →
  `verify <PR#>` (squash-merge).
- `review` runs as `rfablet-review` (via `GH_TOKEN`); `create`/`verify` run as
  the default `rfablet` account. That split is what makes approvals legal.
- **Merge gate** = `pytest -m "not slow"` green on the required test files.
  Ruff is informational, so `mergeStateStatus: UNSTABLE` (due to ruff) does
  **not** block the merge.
- Before deciding to `verify`, confirm `gh pr view <PR#>` shows
  `reviewDecision: APPROVED` and the passed `pytest` checks. If it shows
  `REVIEW_REQUIRED` (or a plain `gh pr review` was attempted with self-approval
  blocked), rerun through `scripts/open_pr.sh review <PR#>` rather than trying
  `--admin` (the ruleset rejects admin bypass).
- Base the PR on the QG integration branch named in `PLAN.md` (`feat/qg-case-study`).
  Prepare new work on a `feat/qg-*` branch created off that base.

### Hygiene

- Never commit artifacts/checkpoints or untracked scratch files.
- Add a `CHANGELOG.md` entry (format below) for every merged change.

## Changelog Format

Each entry in `CHANGELOG.md` should follow this format:

```
## YYYY-MM-DD: Short Title

**Summary:** 1-2 sentence description of changes.
**Files modified:** `path/to/file.py` — brief note
**Rationale:** Why this change was made.
**Verification:** Test command run and result.
```

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

## Key Conventions

- **Python 3.10+** with `torch`, `numpy`, `hydra-core`, `pytorch-lightning`
- **Configuration** uses Hydra/OmegaConf (see `conf/schema.py` for dataclass schemas)
- **No comments** in code unless absolutely necessary (prefer self-documenting names)
- **Type hints** should be used for all function signatures
- **Training** uses PyTorch Lightning (`LitModel` wrapper in `training/lightning_module.py`)
- **Two-stage training** pattern: Stage 1 trains the mean estimator, Stage 2 freezes it and trains the residual
- **Data** is generated on-the-fly; no large data files committed to git
- **Tests** use `pytest` with markers (`@pytest.mark.slow`) for expensive tests

## When Making Model Changes

- Update the corresponding config in `config/experiment/` if training parameters change
- Ensure `LitModel` (in `training/lightning_module.py`) handles the new model type correctly
- Add tests for any new model, loss, or dataset in `tests/`
- Document the change in `CHANGELOG.md`
