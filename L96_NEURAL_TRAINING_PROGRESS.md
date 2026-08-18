# L96 Neural Training Progress

Branch: `feat/l96-neural-training`
Base: `master` @ `0687e07`
Commits: `3a1c8d5` (implementation), `c68ea9d` (docs)
Last updated: 2026-08-19

## Overview

Goal: All-5-param randomization of the two-scale Lorenz-96 system + train neural
models (DirectUNet, VanillaCFM-τ=0) with S0/S1 data, comparing against DA baselines
on the same test configuration.

Key design decisions:
- **Randomized params**: F, c₁, h, hx, ε — all 5, each ±20% of its reference value.
- **Neural conditioning**: `param_dim=0` — models see obs + corrupted forcing only.
- **S0 test**: all 5 params U(0.8·ref, 1.2·ref) independently (clean, no model error).
- **S1 test**: same ±20% + an extra per-param bias of ±10% (`param_bias=0.1`); DA forward
  model uses the biased `*_da` params (model-error scenario).
- **DA / neural parity**: same randomized test data; DA uses per-window `*_da` params.

## WP Status

| WP | Description | Status | Commit | Notes |
|---|---|---|---|---|
| WP1 | `models/lorenz96_dynamics.py` — all-5-param `step()` | ✅ | `3a1c8d5` | kwargs `c1,h,hx,eps` + `F`; verified scalar/batch/traj |
| WP2 | `data/lorenz96.py` — extra helpers, RandomParam/RandBias all-5, `make_l96_s0_s1_trainval` | ✅ | `3a1c8d5` | windows store true + `*_da` params |
| WP3 | `models/direct_unet.py`, `models/vanilla_cfm.py` — `param_dim=0` | ✅ | `3a1c8d5` | `obs_dim = state_dim + 1` |
| WP4 | `config/lorenz96_default.yaml`, `L1_*.yaml`, `L2_*.yaml` | ✅ | `3a1c8d5` | 2 configs, both param_dim=0 |
| WP5 | `data/dataloader.py` — verified working (no core change) | ✅ | `3a1c8d5` | `with_params=False` → params=None |
| WP6 | `train.py` — L96 dispatch + eval batch `param_dim` | ✅ | `3a1c8d5` | fixed pre-existing `to_lorenz96_config` DictConfig bug |
| Step 11a | `evaluation/baselines.py` — all-5 forwarding | ✅ | `3a1c8d5` | std DA methods forward `**kwargs` already; no change needed |
| Step 11b | `evaluation/run_l96.py` — per-window all-5 params | ✅ | `3a1c8d5` | `_per_window_params` uses `*_da` when present |
| WP7 | `tests/test_lorenz96_training.py` — expanded | ✅ | `3a1c8d5` | 11 tests pass |
| — | `L96_NEURAL_TRAINING_PROGRESS.md` | ✅ | `3a1c8d5` | this file |
| Step 11c | DA baseline consistency re-run | ⬜ pending | — | `batch/run_l96_da_consistency.sbatch` |
| Step 12 | Neural training L1 + L2 (full) | ⬜ pending | — | `batch/run_l96_neural_training.sbatch` |
| WP8 | Results comparison + iteration | ⬜ pending | — | `batch/run_l96_evaluate_all.sbatch` |

## File Change Log

| File | Status | Change summary | Verified |
|---|---|---|---|
| `models/lorenz96_dynamics.py` | ✅ | all-5-param step/derivative/trajectories | ✅ |
| `data/lorenz96.py` | ✅ | all-5 randomization + trainval factory + `*_da` | ✅ |
| `models/direct_unet.py` | ✅ | param_dim=0 guard | ✅ |
| `models/vanilla_cfm.py` | ✅ | param_dim=0 guard | ✅ |
| `config/lorenz96_default.yaml` | ✅ | new base default | ✅ |
| `config/experiment/L1_direct_unet_s0s1.yaml` | ✅ | DirectUNet, param_dim=0 | ✅ |
| `config/experiment/L2_vanilla_cfm_s0s1.yaml` | ✅ | VanillaCFM τ=0, param_dim=0 | ✅ |
| `train.py` | ✅ | L96 dispatch + eval param_dim | ✅ |
| `evaluation/run_l96.py` | ✅ | per-window all-5 params | ✅ |
| `tests/test_lorenz96_training.py` | ✅ | +6 tests | ✅ |

## Iteration Log

### Iteration 1 (in progress)
- DA baseline consistency re-run: **pending** (needs sbatch)
- L1 (DirectUNet) results: **pending**
- L2 (VanillaCFM-τ=0) results: **pending**

## Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-08-19 | All 5 params randomized ±20% | Match L63 S-series design, extended |
| 2026-08-19 | `param_dim=0` (no conditioning) | Model learns from obs+forcing only |
| 2026-08-19 | S1 = ±20% + ±10% per-param bias | Systematic model-error test |
| 2026-08-19 | DA uses `*_da` (biased) params at S1 | Same test config as neural |

## Next steps (handoff)

1. **Commit** the current implementation on `feat/l96-neural-training`.
2. **Create sbatch scripts**: `run_one_epoch_tests_l96.sbatch`,
   `run_l96_da_consistency.sbatch`, `run_l96_neural_training.sbatch`,
   `run_l96_evaluate_all.sbatch`.
3. **Run lint/test**: `ruff check .`, `pytest tests/ -m "not slow"`.
4. **Step 11c**: re-run DA baselines (consistency check) via sbatch.
5. **Step 12**: train L1 + L2 (1000/100/200 windows) via sbatch.
6. **WP8**: compare DA vs neural on S0/S1; iterate if neural underperforms.

## Session notes

- The `to_lorenz96_config` method on `DataConfig` exists but does NOT work on Hydra
  DictConfig (`dc` is a plain dict). `train.py` now builds `Lorenz96Config` manually.
- Pre-existing master test failures (unchanged by this branch):
  `test_lorenz63.py::test_observations_noise`,
  `test_random_param_dataset.py::test_getitem_keys`,
  `test_random_param_dataset.py::test_deterministic_with_seed`,
  `test_numerical_equivalence.py` (collection error).
- Standard DA baselines (Weak/Strong-4DVar, EnKF, ETKF) forward `**kwargs` to
  `dynamics.step()`, so no baselines.py edits were needed for all-5 forwarding.
