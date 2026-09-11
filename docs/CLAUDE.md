# CLAUDE.md — 4DVarNet-FM OpenCode

> This file tells Claude Code about the project so it can work effectively without
> re-exploring the codebase from scratch each session. Keep it up to date as the
> project evolves.

## Project overview

**4DVarNet with Flow Matching** — a research codebase implementing data-driven variational
data assimilation for fluid systems (ocean, Lorenz-96). The model combines 4DVarNet
(gradient-based variational solver learned end-to-end) with Flow Matching (generative
modelling of the prior). Developed at IMT Atlantique / CIA-OceaniX.

**Three active topic branches / worktrees** (see `docs/worktrees.md`):
- `main` — core implementation (`4dvarnet-fm-opencode/`)
- `joint-da` — joint data assimilation variant (`4dvarnet-fm-joint-da/`)
- `cfm-v2v3` — continuous flow matching v2/v3 variants (`4dvarnet-fm-cfm-v2v3/`)

---

## Repo structure

```
.
├── train.py                   # Main training entry point
├── run_experiment.py          # Single-experiment runner
├── run_experiments.py         # Batch experiment orchestration
├── eval_baselines.py          # Baseline evaluation
├── evaluate_all.py            # Full evaluation sweep
├── evaluate_all_l96.py        # Lorenz-96 evaluation sweep
├── eval_joint_comparison.py   # Joint DA comparison
├── eval_neural_l96.py         # Neural solver eval on L96
├── eval_joint_neural_l96.py   # Joint + neural on L96
│
├── conf/                      # Hydra config tree (override with +key=value)
├── config/                    # Additional / legacy config files
├── training/                  # LightningModule definitions, loss, optimisers
├── models/                    # Model architectures (4DVarNet, FM, solver)
├── evaluation/                # Metrics, diagnostics, plotting
├── data/                      # Data loaders and preprocessing
├── notebooks/                 # Exploratory Jupyter notebooks
├── demos/                     # Demo scripts / reproducibility
├── scripts/                   # Utility scripts (download, preprocess, etc.)
├── batch/                     # Cluster job scripts (SLURM / PBS)
├── tests/                     # pytest test suite
├── docs/                      # Documentation (worktrees, conventions)
├── reports/                   # Generated reports and figures
├── logs/                      # Training logs (gitignored)
└── archive/                   # Old experiments kept for reference
```

---

## Stack

| Component | Library |
|---|---|
| Deep learning | PyTorch |
| Training loop | PyTorch Lightning |
| Config management | Hydra + OmegaConf |
| Experiments | `run_experiment.py` / `run_experiments.py` |
| Evaluation & plots | custom (`evaluation/`) + matplotlib |
| Tests | pytest |
| CI | GitHub Actions (`.github/workflows/`) |

---

## How to run things

### Training

```bash
# Basic training run (Hydra config from conf/)
python train.py

# Override config keys
python train.py model=fm_v1 data=swot trainer.max_epochs=100

# Run a named experiment
python run_experiment.py experiment=<name>

# Batch of experiments
python run_experiments.py
```

### Evaluation

```bash
python evaluate_all.py         # full sweep
python evaluate_all_l96.py     # Lorenz-96
python eval_baselines.py       # baselines only
python eval_joint_comparison.py
```

### Tests

```bash
pytest                         # all tests
pytest tests/ -x -q            # fail-fast, quiet
pytest tests/ -k "l96"         # filter by name
```

### Cluster (SLURM)

```bash
# TODO: fill in the standard sbatch command for your cluster
# sbatch batch/<job_script>.sh
```

---

## Configuration system (Hydra)

Config lives under `conf/`. Override on the command line with `key=value`; compose
experiments by combining config groups. Check `conf/` for available groups (model, data,
trainer, callbacks, …).

```bash
# List available configs
python train.py --cfg job

# Multirun (grid search)
python train.py -m model=fm_v1,fm_v2 data=swot,l96
```

---

## Data

```
data/
├── <TODO: describe datasets — SWOT, L96, NATL60, …>
└── <TODO: note where raw data lives on the cluster / local cache>
```

> **Note for Claude:** Do not generate synthetic data or modify files under `data/` without
> explicit instruction. Data paths are usually set via Hydra config.

---

## Key design conventions

- **LightningModule** is the unit of a model: training step, validation step, and
  optimiser config all live in `training/`.
- **Hydra configs** are the single source of truth for hyperparameters — never hardcode
  numbers in Python that belong in config.
- **Evaluation is decoupled from training** — run eval scripts separately after a
  checkpoint is saved.
- **Worktrees** keep topic branches isolated on disk; see `docs/worktrees.md` before
  creating or switching branches.

---

## Common Claude Code tasks on this repo

- Implement or refactor a model in `models/`
- Add a new Hydra config group under `conf/`
- Write or fix a pytest test in `tests/`
- Add a metric or diagnostic to `evaluation/`
- Debug a training run from a log excerpt
- Draft a CHANGELOG entry or update PLAN.md

**When editing models or training code, always:**
1. Check whether a Hydra config parameter needs to be added/changed alongside the code.
2. Run `pytest -x -q` before marking work done.
3. Do not modify `archive/` or `logs/` unless explicitly asked.

---

## What NOT to do

- Don't run full training unless asked — prefer `trainer.fast_dev_run=true` for quick checks.
- Don't commit to `main` directly — open a PR or use the topic worktree.
- Don't reformat the entire codebase in one pass.
- Don't install packages without checking `requirements.txt` first.

---

## TODO / open questions for Ronan to fill in

- [ ] Exact Python version and conda/venv setup command
- [ ] Data download / access instructions
- [ ] Cluster name and standard `sbatch` invocation
- [ ] Which Hydra config groups are most commonly overridden
- [ ] Any environment variables needed (`WANDB_API_KEY`, data paths, …)
- [ ] Primary evaluation metric to optimise for
