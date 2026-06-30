# 4DVarNet-FM Project Guidelines

## Session Workflow

Every opencode session in this repository MUST follow this workflow:

1. **Read PLAN.md first** — Before making any changes, read `PLAN.md` to understand the current design plan.
2. **Implement changes** — Make the requested code modifications.
3. **Verify** — Run the relevant test/lint commands (see below).
4. **Log changes** — Append a dated entry to `CHANGELOG.md` describing what was implemented, why, and any notable design decisions.

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
