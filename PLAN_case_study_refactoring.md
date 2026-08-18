# Multi-Case-Study Refactoring Plan

## Objective
Refactor the codebase to support multiple dynamical systems (Lorenz63, Lorenz96, Shallow Water) behind a common interface, starting with L63-only abstraction (Phase 1), then adding Shallow Water (Phase 2+).

**Branch**: `feat/multi-case-study` (created from master @ `6dd4984`)

---

## Current Status (Jul 11)

- **Phase 1** — fully committed and tagged (p1-config through p1-fixes)
- **Bug fix** — EnKF/ETKF batch forecast path now flattens `(B,N,D)` → `(B*N,D)` before `dynamics.step()` to handle per-window sigma/rho/beta correctly. Committed as `0a5c35f`.
- **Full S0/S1 DA baseline run** — submitted as SLURM job **42642** (200 test windows, report-matched config, batch_size=50, A40 GPU)
- **Edge-effect sensitivity** — trimmed RMSE (steps 50–250) vs full RMSE compared; differences ≤ 0.053, no ranking changes (`bc50a64`)
- **Phase 2 (Lorenz96)** — initial implementation committed (dynamics, data, config, factory, baseline kwargs refactor)
- **Remaining issues**:
  - `DataConfig` in `conf/schema.py` still has `forcing_coupling: str` — should add `coupling_exponent_truth` and `coupling_exponent_da` fields; `to_lorenz63_config()` should pass them _(low priority — non-breaking, only affects Hydra pipeline)_
  - Configs still use `forcing_coupling: linear/quartic` — should migrate to `coupling_exponent_truth: 1.6` / `coupling_exponent_da: 1.0` _(same)_
  - `eval_baselines.py` (Hydra pipeline) doesn't create dynamics with correct coupling exponents — uses `get_dynamics()` which now needs `coupling_exponent_truth` from config _(same)_

---

## Phase 1: Architecture Abstraction (L63 continues working identically)

Remove hardcoded L63-specific assumptions behind abstract interfaces. No functional changes to existing experiments.

### Progress Tracking

| # | Agent | Description | Files | Est. | Status | Git tag |
|---|-------|-------------|-------|------|--------|---------|
| 1 | Config | Add `data.system` to `lorenz63_default.yaml`; create `config.yaml` + `case_study/lorenz63.yaml` skeleton | `config/` | 1h | completed | `p1-config` |
| 2 | Dynamics | Create `DynamicsBase(ABC)` + refactor `Lorenz63Dynamics` as subclass + factory | `models/dynamics.py`, `models/lorenz63_dynamics.py` | 1.5h | completed | `p1-dynamics` |
| 3 | Model dims | Parameterize `obs_dim`, `param_dim`, `FlowMatchingBatch` + `collate_fm` from config | `models/direct_unet.py`, `models/vanilla_cfm.py`, `data/dataloader.py`, `train.py` | 1h | completed | `p1-model-dims` |
| 4 | Metrics | Generalize RMSE keys from X/Y/Z to config-driven `state_names` | `evaluation/metrics.py`, `train.py` (partial) | 0.5h | completed | `p1-metrics` |
| 5 | Datasets | Refactor datasets to use `DynamicsBase` internally; `generate_full_trajectory` on `Lorenz63Dynamics` | `data/lorenz63.py`, `data/random_bias_dataset.py`, `data/random_param_dataset.py` | 2h | completed | `p1-datasets` |
| 6 | DA baselines | Parameterize `Weak4DVar`, `Strong4DVar`, `EnKF`, `ETKF` to accept `DynamicsBase` | `evaluation/baselines.py` | 3h | completed | `p1-baselines` |
| 7 | Pipeline | Wire `model_factory`, `get_dynamics`, evaluation loop + full validation | `train.py`, `eval_baselines.py`, `training/`, `evaluate_all.py` | 1.5h | completed | `p1-pipeline` |
| — | Fixes | `coupling_exponent_truth` bugfix, dynamics pooling, numerical equivalence tests, `get_dynamics` API fix | `data/*.py`, `evaluation/run.py`, `models/*.py`, `tests/*.py` | — | completed | `p1-fixes` |
| — | Batch fix | Flatten ensemble dim in EnKF/ETKF batch forecast step for `dynamics.step()` API compatibility | `evaluation/baselines.py` | — | completed | `p1-batch-fix` |

### Verifications (run after each agent)

```bash
# Syntax + imports
ruff check . --select=E,F  # no errors
python -c "import ast; ast.parse(open('FILE').read())"  # for each modified file

# Config validation
python -c "from hydra import compose, initialize; initialize(config_path='config'); compose('config')"

# Test suite (fast tests only)
pytest tests/ -v -m "not slow"

# Smoke test (GPU)
python train.py --config-name experiment/S1_direct_unet_s0s1 hydra.run.dir=. hydra.output_subdir=null
```

### Dependency Graph for Parallel Execution

```
Batch 1 (parallel):
  Agent 1 (Config)  —  no deps
  Agent 2 (Dynamics)  —  no deps

Batch 2 (parallel, after Agent 1):
  Agent 3 (Model dims)  —  needs Agent 1
  Agent 4 (Metrics)  —  needs Agent 1 (quick, can merge into Agent 3)

Batch 3 (after Agent 1 + 2):
  Agent 5 (Datasets)  —  needs Agents 1 + 2

Batch 4 (after Agent 2):
  Agent 6 (Baselines)  —  needs Agent 2

Batch 5 (after Agents 1-6):
  Agent 7 (Pipeline)  —  needs all
```

### Commands to Resume After Crash

If session crashes mid-phase:

```bash
# 1. Check current branch and git log
git log --oneline -10

# 2. Check which git tags exist to see completed phases
git tag -l 'p1-*'

# 3. Check this plan file for status
grep '|' PLAN_case_study_refactoring.md | grep -E '(pending|in_progress)'

# 4. Resume from next pending agent
```

---

## Phase 2: Lorenz96 Case Study (pending)

**Lorenz '96 system** — multi-scale lattice with NO slow variables and J fast variables per slow node. The dynamics follow a two-scale coupling pattern:
- Slow: `dX_k/dt = X_{k-1}(X_{k+1} - X_{k-2}) - X_k + F - h * sum_j(Y_{j,k})`
- Fast: `dY_{j,k}/dt = (1/eps) * (Y_{j+1,k}*(Y_{j-1,k} - Y_{j+2,k}) - Y_{j,k}) + (h_x/eps) * X_k`
- Forcing coupling: `W = c1 * sign(W_raw) * |W_raw|^exponent` (same pattern as L63)

Parameters: NO=8, J=4, FO=8, FA=8, coupling parameters matching the L96 specification.

### Agents

| # | Name | Description | Files | Est. | Status |
|---|------|-------------|-------|------|--------|
| 0 | Baseline API | Refactor all `dynamics.step()` calls to use `**kwargs` pattern; remove vestigial `_apply_coupling` from baselines.py | `evaluation/baselines.py`, `models/lorenz63_dynamics.py` | 1h | committed |
| 1 | Dynamics | `Lorenz96Dynamics` subclass of `DynamicsBase` with `step()` + forcing generation + `generate_full_trajectory()` | `models/lorenz96_dynamics.py` | 1.5h | committed |
| 2 | Config | `config/case_study/lorenz96.yaml` with system defaults | `config/case_study/lorenz96.yaml` | 0.5h | committed |
| 3 | Data | `data/lorenz96.py` — dataset classes using `Lorenz96Dynamics`; `Lorenz96Config` dataclass | `data/lorenz96.py` | 2h | committed |
| 4 | Factory | Update `get_dynamics()` to dispatch to `Lorenz96Dynamics` based on `data.system: lorenz96` | `models/dynamics.py` | 0.5h | committed |
| 5 | Pipeline | End-to-end validation: training + baselines + evaluation with L96 config | `train.py`, `eval_baselines.py`, `evaluate_all.py` | 1h | pending |
| — | Lint | `ruff check . --select=E,F` — no functional errors (only pre-existing E501) | — | — | verified |

### Verifications
- `ruff check . --select=E,F` — no errors
- Single-step numerical test: `Lorenz96Dynamics.step()` matches inline L96 reference
- Dataset generation: `Lorenz96Dataset` produces valid trajectories
- Baseline run: all 4 DA methods complete without errors on small test set

## Phase 3: Shallow Water Case Study (future)

- `models/shallow_water_dynamics.py` — 2D PDE solver (Eqs. 10-22 from PDF)
- `models/unet2d.py` — 2D U-Net for spatial fields
- `config/case_study/shallow_water.yaml` — 64x64 grid, g=9.81, f=1e-4
- Sparse spatial observation model
- New experiment configs

## Key Design Decisions

1. **Config inheritance**: `config.yaml` → `case_study/{name}.yaml` → `experiment/{name}.yaml`
2. **`data.system` field**: Already exists in `lorenz63_default.yaml` as `data.system: lorenz63` — will be used as the case study selector
3. **`DynamicsBase` interface**:
   ```python
   class DynamicsBase(ABC):
       state_dim: int
       param_names: list[str]
       param_dim: int
       @abstractmethod
       def step(self, state, forcing, params) -> Tensor: ...
       def rollout(self, x0, forcing, steps, params) -> Tensor: ...
   ```
4. **`get_dynamics(cfg)` factory**: Dispatches `data.system` → `Lorenz63Dynamics`, `Lorenz96Dynamics`, `ShallowWaterDynamics`
5. **Checkpoint compatibility**: Model architecture is deterministically determined by config dimensions, so same config = same architecture = compatible checkpoints.
6. **Dataset caching**: Cache key includes `data.system` to prevent cross-case-study cache collisions.
## Phase 3: Shallow Water Case Study (in progress)

| # | Agent | Description | Files | Est. | Status | Git tag |
|---|-------|-------------|-------|------|--------|---------|
| 0 | Baseline fix | Replace hardcoded L63 params with **params | `evaluation/baselines.py` | 5min | DONE | psw-fix |
| 1 | Metrics | Add EV + per-component metrics | `evaluation/metrics.py` | 1h | DONE | psw-metrics |
| 2 | Dynamics | `ShallowWaterDynamics` class | `models/shallow_water_dynamics.py` | 3h | DONE | psw-dynamics |
| 3 | Config & Data | `ShallowWaterConfig`, datasets, YAML | `data/shallow_water.py`, config/, conf/ | 2h | DONE | psw-config |
| 4 | DA + Eval | `run_sw.py`, `evaluate_all_sw.py` | `evaluation/run_sw.py`, `evaluate_all_sw.py` | 2h | DONE | psw-da |
| 5 | Tests | SW tests with EV performance targets | `tests/test_shallow_water.py` | 2h | DONE | psw-tests |
| 6 | SLURM + Docs | SBATCH scripts, PLAN update, CHANGELOG | `batch/run_sw_*.sbatch` | 1h | DONE | psw-sbatch |

### Resume after crash
```bash
# Check current state
git log --oneline feat/multi-case-study | head -20
git tag -l 'psw-*'
grep '| pending |' PLAN_case_study_refactoring.md
```
