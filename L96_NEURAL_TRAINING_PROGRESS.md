# L96 Neural Training Progress

Branch: `feat/l96-neural-training`
Base: `master` @ `0687e07`
Commits: `3a1c8d5` (implementation), `c68ea9d` (docs)
Last updated: 2026-08-19

## Overview

Goal: All-5-param randomization of the two-scale Lorenz-96 system with **partial
observations** (obs_j=2) + train neural models (DirectUNet, VanillaCFM-τ=0) in 24D
observed space, comparing against DA baselines on the same test configuration.

Key design decisions:
- **Partial observations**: obs_j=2 → 24D observed state (8 slow X + 16 fast Y1,Y2).
  Truth is 40D (J=4) with `fast_weights=[1,1,0.1,0.1]`.
- **Randomized params**: F, c₁, h, hx, ε — all 5, each ±20% of its reference value.
- **Neural conditioning**: `param_dim=0` — models see obs + corrupted forcing only.
- **Neural state_dim**: 24 — operates in observed subspace, no padding.
- **S0 DA**: 40D dynamics (J=4), `ObsOperator(40, obs_var_indices)` → rectangular H.
- **S1 DA**: 24D dynamics (J=2), `ObsOperator(24, identity)` → all dims observed.
- **S0 test**: all 5 params U(0.8·ref, 1.2·ref) independently (clean, no model error).
- **S1 test**: same ±20% + an extra per-param bias of ±10% (`param_bias=0.1`); DA forward
  model uses the biased `*_da` params (model-error scenario).
- **Per-group scoring**: slow (8D), obs_fast (16D), all_obs (24D), plus explained variance.

## WP Status

| WP | Description | Status | Commit | Notes |
|---|---|---|---|---|
| WP1 | `models/lorenz96_dynamics.py` — all-5-param `step()` | ✅ | `3a1c8d5` | kwargs `c1,h,hx,eps` + `F`; verified scalar/batch/traj |
| WP2 | `data/lorenz96.py` — RandomParam/RandBias all-5, `make_l96_s0_s1_trainval` | ✅ | `3a1c8d5` | windows store true + `*_da` params; obs_var_indices propagated |
| WP3 | `models/direct_unet.py`, `models/vanilla_cfm.py` — `param_dim=0` | ✅ | `3a1c8d5` | `obs_dim = state_dim + 1` |
| WP4 | `config/lorenz96_default.yaml`, `L1_*.yaml`, `L2_*.yaml` | ✅ | `3a1c8d5` | state_dim=24, obs_j=2, fast_weights |
| WP5 | `data/dataloader.py` — `obs_var_indices` subsampling | ✅ | this session | `FlowMatchingDataset` subsamples true_state to24D |
| WP6 | `train.py` — obs_j → obs_var_indices + per-group RMSE | ✅ | this session | computes indices, passes to data/eval |
| Step 11a | `evaluation/baselines.py` — all-5 forwarding + ObsOperator | ✅ | `3a1c8d5` | std DA methods forward `**kwargs` already; ObsOperator already existed |
| Step 11b | `evaluation/run_l96.py` — ObsOperator + S1 J=2 + per-group | ✅ | this session | S0: ObsOperator(40,obs_var_indices); S1: J=2 dynamics + identity |
| WP7 | `tests/test_lorenz96_training.py` — partial-obs tests | ✅ | this session | 22 tests pass (11 original + 11 new) |
| — | `L96_NEURAL_TRAINING_PROGRESS.md` | ✅ | this session | this file |
| Step 11c | DA baseline consistency re-run (partial obs) | ✅ | — | sbatch 48683; 200-window obs_j=2 EnKF/ETKF/Strong-4DVar + EV backfill. S0 EnKF EV +0.544, ETKF +0.538, Strong +0.586; S1 EnKF +0.022, ETKF +0.036, Strong +0.205 |
| Step 11d | Obs-density variation (S0-Obs100/S1-Obs100) | 🔄 in progress | — | sbatch 48688 (OBS_INTERVAL=100); trajectory-reuse cache, 2× denser obs (~30/window), EV captured |
| Step 11e | Fast-weights randomization experiment (S0b/S1b: F + fast_weights) | ⬜ planned | — | requires threading per-window fast_weights through dynamics + randomization |
| Step 12 | Neural training L1 + L2 (full) | 🔄 in progress | — | `batch/run_l96_neural_training.sbatch`, job 49013 (L1 running, L2 pending) |
| WP8 | Results comparison + iteration | ⬜ pending | — | `batch/run_l96_evaluate_all.sbatch` |

## File Change Log

| File | Status | Change summary | Verified |
|---|---|---|---|
| `models/lorenz96_dynamics.py` | ✅ | all-5-param step/derivative/trajectories | ✅ |
| `data/lorenz96.py` | ✅ | all-5 randomization + trainval factory + obs_var_indices | ✅ |
| `models/direct_unet.py` | ✅ | param_dim=0 guard | ✅ |
| `models/vanilla_cfm.py` | ✅ | param_dim=0 guard | ✅ |
| `conf/schema.py` | ✅ | `obs_j` field + `_compute_obs_var_indices()` in `to_lorenz96_config` | ✅ |
| `config/lorenz96_default.yaml` | ✅ | obs_j=2, fast_weights, state_dim=24 | ✅ |
| `config/experiment/L1_direct_unet_s0s1.yaml` | ✅ | DirectUNet, state_dim=24, param_dim=0 | ✅ |
| `config/experiment/L2_vanilla_cfm_s0s1.yaml` | ✅ | VanillaCFM τ=0, state_dim=24, param_dim=0 | ✅ |
| `data/dataloader.py` | ✅ | `obs_var_indices` param on FlowMatchingDataset/ConcatFMDataset/make_dataloaders | ✅ |
| `train.py` | ✅ | obs_j → obs_var_indices, per-group RMSE, obs_var_indices in evaluate_model/save | ✅ |
| `evaluation/run_l96.py` | ✅ | make_obs_j_indices, ObsOperator S0/S1, S1 J=2 dynamics, per-group fmt_rmse, EV (fmt_ev/_per_group_ev), obs_interval cache key | ✅ |
| `evaluate_all_l96.py` | ✅ | --obs-j CLI, obs_var_indices, per-group table, dataset caching (obsj/nwin .pt, --regenerate-data), int{obs_interval} cache key, trajectory-reuse | ✅ |
| `backfill_l96_baselines_ev.py` | ✅ | one-off CPU back-compute of EV into existing baseline caches | ✅ |
| `tests/test_lorenz96_training.py` | ✅ | +11 partial-obs + 3 EV tests (25 total) | ✅ |
| `batch/run_l96_da_consistency.sbatch` | ✅ | OBS_INTERVAL env → --obs-interval | ✅ |

## Iteration Log

### Iteration 6 — obs-density variation (S0-Obs100/S1-Obs100)
- Threaded `obs_interval` through the L96 dataset + baseline caches so observation density is
  parametrizable. Dataset cache key now `..._obsj{obs_j}_int{obs_interval}_nwin{nwin}.pt`; baseline
  cache key now `..._obsj{obs_j}_int{obs_interval}.json`.
- Added trajectory-reuse: when `obs_interval` differs but a same-seed cache exists, load its
  trajectories and regenerate only `obs`/`obs_mask` via `_generate_observations` (reusing `obs_seed`),
  cutting dataset prep from ~73 min to ~2 s.
- `batch/run_l96_da_consistency.sbatch` takes `OBS_INTERVAL`. Job 48688 (OBS_INTERVAL=100) reuses
  cached trajectories (1.6 s) then runs EnKF/ETKF/Strong-4DVar on GPU → S0-Obs100/S1-Obs100.
- 25 tests pass; only pre-existing ruff E401/F541 remain. Smoke confirms 30 obs/window (vs 15).

### Iteration 5 — add EV to S0/S1 DA baseline cache
- `run_and_cache_baselines` discarded the pooled EV already computed by `evaluate_baseline`.
  Captured it, added `fmt_ev`/`_per_group_ev`, and stored `ev` (per-dim + grouped) in each
  method's JSON cache entry. Backfilled existing 200-window caches via `backfill_l96_baselines_ev.py`.
- Backfilled 200-window all_obs EV: S0 EnKF +0.544, ETKF +0.538, Strong-4DVar +0.586;
  S1 EnKF +0.022, ETKF +0.036, Strong-4DVar +0.205.
- 3 tests added (25 total). `pytest tests/test_lorenz96_training.py -m "not slow"` pass.

### Iteration 4 — cache S0/S1 dataset in evaluate_all_l96
- `evaluate_all_l96.py` regenerated the 200-window S0/S1 test dataset every run (~17 min) even though
  DA baselines were cached. Added dataset caching to `experiments/l96_datasets_obsj{obs_j}_nwin{nwin}.pt`
  (torch.save/load), with `--regenerate-data` to force regeneration.
- Verified `torch.save`/`torch.load` round-trip on a 2-window smoke. Resubmitted DA consistency as
  job 48674 (GPU/sbatch) to generate + cache the full 200-window dataset and run EnKF/ETKF.

### Iteration 3 — S0 RMSE/EV 24D subspace fix
- Found & fixed a bug in `evaluate_baseline`: S0 DA analyses are 40D (full state), matching the 40D
  `true_state`, so the old subsampling guard `analysis.shape[-1] != truth.shape[-1]` never fired and
  RMSE/EV included the 16 unobserved fast vars (Y3,Y4), inflating RMSE and deflating EV vs the 24D neural space.
- `evaluate_baseline` now subsamples both `analysis` and `ref` to `obs_var_indices` whenever it is
  provided (analysis only if its dim > obs count), and always overrides `result.rmse`. Full analysis kept for plots.
- S0 smoke (3 win): EnKF RMSE 1.452→1.264, ETKF 1.398→1.297; EV EnKF +0.512, ETKF +0.487. S1 unchanged.

### Iteration 2 — partial observation integration
- Added `obs_j=2` config → 24D observed subspace (8 slow + 16 fast Y1,Y2)
- Neural models now operate in 24D (`state_dim=24`, no padding)
- S0 DA: `ObsOperator(40, obs_var_indices)` → rectangular H
- S1 DA: `Lorenz96Dynamics(J=2)` + `ObsOperator(24, identity)`
- Per-group RMSE scoring: slow / obs_fast / all_obs
- `--obs-j` CLI arg on `evaluate_all_l96.py`

### Iteration 1 (previous, superseded)
- DA baseline consistency re-run with all-5 full-state: **superseded** by partial-obs

## Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-08-19 | All 5 params randomized ±20% | Match L63 S-series design, extended |
| 2026-08-19 | `param_dim=0` (no conditioning) | Model learns from obs+forcing only |
| 2026-08-19 | S1 = ±20% + ±10% per-param bias | Systematic model-error test |
| 2026-08-19 | DA uses `*_da` (biased) params at S1 | Same test config as neural |
| 2026-08-19 | obs_j=2 default (24D obs subspace) | Observe Y1,Y2 per node; Y3,Y4 unobserved with fast_weights=[1,1,0.1,0.1] |
| 2026-08-19 | Neural state_dim=24 (no padding) | Target = 24D observed subset of true_state |
| 2026-08-19 | S1 DA uses J=2 dynamics (24D) | No unobserved fast vars; identity H |
| 2026-08-19 | S0 DA uses J=4 dynamics (40D) + ObsOperator | Full dynamics with rectangular obs mapping |

## Next steps (handoff)

1. **Commit** partial-obs implementation on `feat/l96-neural-training`.
2. **Re-run DA baselines** with `--obs-j=2` via sbatch (Step 11c).
3. **Train L1 + L2** (state_dim=24) via `batch/run_l96_neural_training.sbatch` (Step 12).
4. **WP8**: compare DA vs neural on S0/S1 with per-group scoring.

## Session notes

- The `to_lorenz96_config` method on `DataConfig` exists but does NOT work on Hydra
  DictConfig (`dc` is a plain dict). `train.py` now builds `Lorenz96Config` manually.
- Pre-existing master test failures (unchanged by this branch):
  `test_lorenz63.py::test_observations_noise`,
  `test_random_param_dataset.py::test_getitem_keys`,
  `test_random_param_dataset.py::test_deterministic_with_seed`,
  `test_numerical_equivalence.py` (collection error).
- `ObsOperator` and `make_obs_j_indices` already existed in `evaluation/baselines.py`
  and `run_l96_sweep2.py` respectively; now wired into the main eval path.
- `data/lorenz96.py` already propagated `obs_var_indices` through `_generate_observations`;
  the new code adds subsampling of `true_state` in `dataloader.py` and config wiring.
