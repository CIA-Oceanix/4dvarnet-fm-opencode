# 4DVarNet-FM: Implementation Plan

## Overview

Three model families + CS3/CS4 randomized-parameter tests + Experiment G ablation.

- **DirectUNet**: Single UNet pass `obs → state` via MSE (no flow matching) — implemented
- **VanillaCFM**: Standard conditional flow matching, no Tweedie decomposition — implemented
- **TweedieSolver**: Original two-stage solver — legacy, maintained
- **RandomParamDataset**: Per-window randomized `σ,ρ,β` ±20% for robust training — implemented
- **CS3/CS4**: Evaluation on unseen random parameter draws — implemented
- **Exp G (τ=0 CFM)**: Ablate multi-τ training to isolate CFM's source of advantage — implemented (superseded by S9/S10)

## L96 (two-scale Lorenz-96) — merged to master 2026-08-18

- **Dynamics/DA baselines** (`feat/weighted-fast-coupling` merged into master, SW/MAOOAM excluded):
  - `models/lorenz96_dynamics.py`, `data/lorenz96.py` — two-scale L96 (NO=8, J=4, state_dim=40, weighted fast coupling)
  - `evaluation/run_l96*.py` + `reports/outputs/l96_baseline_report.md` — Waves 1-4 DA sweeps + ETKF ablation, pooled-EV metric
  - `models/dynamics.py` `get_dynamics()` supports only `lorenz63`/`lorenz96` (SW/MAOOAM deferred)
- **Training infrastructure** (built, no runs launched):
  - `conf/schema.py` → `DataConfig.to_lorenz96_config()`
  - `train.py` → `data.system` dispatch (`lorenz96`), `param_names`-generalized eval
  - configs: `config/experiment/L1_direct_unet_s0s1.yaml`, `L2_vanilla_cfm_s0s1.yaml`
  - test: `tests/test_lorenz96_training.py`
- **To do (next)**: launch L96 UNet/CFM training (L1/L2) and compare vs DA baselines.

## QG (two-layer quasi-geostrophic) — branch `feat/qg-case-study` (Phase A)

Two-layer Phillips channel (double-periodic β-plane), pyqg-compatible dimensional units.
pyqg used as **reference + validation only**; the engine is a native torch port for autograd/batching.
Parallel-session isolation via worktrees (`../4dvarnet-fm-qg`); shared clone stays on master.

- **Dynamics** `models/qg_dynamics.py` (`QGDynamics(DynamicsBase)`):
  - torch port of pyqg v0.4.0 formulation: RK4 (vs pyqg AB3, documented deviation), flux-form advection with total u=u′+U_k, pyqg exponential filter (filterfac=23.6), masked PV inversion (K2==0 → ψ̂=0, PV anomalies zero-mean by construction).
  - state flattened `[..., 2·ny·nx]` layer-major; `forcing` accepted for DynamicsBase compat, ignored (autonomous).
  - `param_names=["beta","rd","rek","U1","U2"]`; runtime overrides beta/rek/U1/U2 only (rd/delta fixed at construction).
- **Calibration** `reports/calibrate_qg_nominal.py` (PRESETS A/B/C + `--with-pyqg`):
  - Nominal config = **preset B**: U₁=0.05, U₂=0, rd=15km, β=1.5e-11, δ=0.25, rek=5.787e-7, dt=7200s, nx=64. Near equilibrium (−16% KE drift @2y), KE 3.6e-3, spectral peak 0.44·k_d; torch≈pyqg (spectral corr 0.984).
  - Validation: discrete max growth 0.0145/day vs continuous 0.0147/day; tendency equivalence vs pyqg at rtol=1e-5.
  - Figures/JSON in `reports/outputs/figs/qg_*`.
- **Data** `data/qg.py` (`QGConfig`, `QGDataset`, `make_qg_datasets`):
  - Windows sliced from a single post-spinup trajectory; default spinup_years=2.0, window_days=60, obs_interval=6.
  - Obs: `_generate_observations` on flattened state (all grid points), `R_var=1e-12` (~4% of equilibrated q₁ std≈2.6e-5).
  - Window keys: `true_state`, `obs`, `obs_mask`, `forcing_true/corrupted`, `wind_curl` (randomization/S0–S1 deferred).
- **Tests** `tests/test_qg_dynamics.py` (18), `tests/test_qg_data.py` (13, 1 slow).

### QG Phase A.2 — moving-storm wind forcing (PRs #30/#32/#34/#37/#38 + moving-storm PRs)

Adds an atmosphere-like **wind-stress curl** as an upper-layer PV source (Ekman pumping, Vallis §9.2), shaped as a **localized Gaussian storm on a moving storm track**:
- `models/qg_dynamics.py`: `dq1/dt += curl_τ` where `curl_τ(x,y,t) = A(t)·(1 − r²/2σ²)·exp(−r²/2σ²)` is a Witch-of-Agnesi profile centered at a moving storm center `(xc, yc)`. Amplitude `A(t)` is an Ornstein–Uhlenbeck process (`wind_amp` = OU σ, `wind_tau_days`; default `τ=15 d` ≈ storm-passage timescale so the storm is coherent, not flickering) and the center follows a storm track `xc(t) = (L/2 + wind_cx·t + wx(t)) mod L`, `yc(t) = (W/2 + wind_cy·t + wy(t)) mod W`, with OU position jitter `wx, wy` (`wind_drift_tau_days`, `wind_drift_sigma`). Mean drift `(wind_cx, wind_cy) = (0.5, 0.03) m/s` gives a ~23-day zonal crossing with a slow NE track; `wind_sigma = 250 km` storm width. `generate_wind_state(num_steps)` returns a `(T,3)` array `[A, xc, yc]`; `wind_curl_field(wind_state)` builds the per-step field; `generate_full_trajectory`/`generate_batch_trajectories` return `(traj, wind_state)`. `wind_amp=0` reproduces the unforced trajectory **bitwise** (regression test); spinup stays unforced; held constant within each RK4 step.
- **Calibration** `reports/calibrate_qg_wind.py`: nx=64 sweep over `wind_amp ∈ {0, 3e-12, 1e-11, 2e-11, 3e-11}` under the moving-storm defaults at `τ=15` → `wind_amp=1e-11` gives KE +30% vs unforced (3.51e-3), nearest the +32% comparable-contribution target; 2e-11 → +158%, 3e-11 → +350%.
- **Data** `data/qg.py`: `QGConfig` wind params (`wind_amp`, `wind_tau_days`, `wind_sigma`, `wind_cx`, `wind_cy`, `wind_drift_tau_days`, `wind_drift_sigma`, `wind_seed`); `QGDataset` stores per-window `wind_curl` `(T,ny,nx) = wind_curl_field(wind_state_slice)` and `forcing_true/corrupted` = the per-step wind amplitude column `wind_state[:,0]`.
- **Animation** `reports/animate_qg_wind.py` → `reports/outputs/figs/qg_wind_animation.gif` (now shows the moving storm) + strong-amplitude `qg_wind_animation_strong.gif`.
- **Impact diagnostics** `reports/diagnose_qg_wind_impact.py` → `reports/outputs/figs/qg_wind_impact.png`: combined 4-panel (KE time series, isotropized KE spectrum, PV anomaly `q1_forced−q1_unforced`, wind work `⟨τ_curl·ψ₁⟩`) comparing unforced vs default (1e-11) vs strong (3e-11), making the wind's ~+33%/+283% KE imprint diagnostic.
- **PR workflow** active: `feat/qg-*` ruleset (1 review + pytest, `do_not_enforce_on_create` on), reviewer/analyst → `cortecs/glm-5.2`.
- **To do (next)**: QG S0/S1 along-track evaluation dataset (now in Phase A.3) → then QG DA baselines (EnKF/ETKF/4DVar on ψ/q), then train.py dispatch + neural training.

### QG Phase A.3 — S0/S1 along-track evaluation dataset (branch `feat/qg-s0s1-alongtrack-dataset`)

S0/S1 evaluation mirroring the L96 case-study pattern, on a **nadir-altimetry-style** observation layout:
- **Obs** `_generate_alongtrack_observations` (`data/qg.py`): one full **meridional column** of the upper layer per repeat cycle (`track_repeat_days=5`), advancing zonally by `track_advance_pts=4` grid points per pass (`~62.5 km` cross-track spacing at nx=64), ±Gaussian noise at `obs_noise_std_frac=0.05` of the field's std; both `psi` and `q` targets stored (`obs_field`, `store_targets`). Compact `(T, ny)` obs + `track_x_index` per window; `expand_obs_to_grid` maps back to a NaN-padded `(T, ny*nx)` field. Coverage ≈ 0.026% (768 vs 2,949,120 dense).
- **Truth** `QGS01Dataset._generate_truth`: **20 (test) windows shared by all scenarios**, per-window draws `U1, rd, rek ~ U[1±param_range]` (`param_range=0.15`; β/δ fixed), independent storm per window — start `(x0,y0)~U(0,L)²`, track `cx~U[0.25,0.75]`, `cy~U[−0.06,0.06]`, and `wind_amp` drawn from discrete levels `{0, 3e-12, 1e-11, 2e-11, 3e-11}` round-robin `i % 5` (`wind_amp` stored per window for the KE diagnostic). *Log-uniform `wind_amp` draws are noted for the future training set.*
- **Scenarios** `_scenario_window`:
  - **S0** (`test_s0`): `da_params = true_params`, `da_model="qg2l"`, `wind_state_corrupted = wind_state_true` (model-error-free).
  - **S1a** (`test_s1a`): 2-layer DA model with **param bias** `rd, rek ← ·(1 − s1_param_bias)` (`s1_param_bias=0.15`) and **corrupted wind** `_make_corrupted_wind_state` — storm centre OU jitter σ=`s1_loc_sigma_frac·wind_sigma` (=0.25·250 km≈62 km, `s1_tau_days=10` d), amplitude `A·(1+s1_amp_bias)+η` with OU `η` (σ=`s1_sigma_eta_frac·A_std`, τ=`s1_tau_days`). OU paths built via `_ou_series` (unit-std, `N(0,1)` increments matching the dynamics convention).
  - **S1b** (`test_s1b`): **structural model error** — reduced-gravity single-layer `da_model="qg1l"` (`models/qg1l_dynamics.py`), params unbiased (= true draws). Free-divergence calibration projects the truth upper-layer ψ₁ into the 1-layer model (`state_from_streamfunction`) and compares the roll-out ψ₁ against the truth target; S1b divergence `[1.13 … 1.36]` > S1a, both ≫ S0.
- **Reduced-gravity model** `models/qg1l_dynamics.py` (`QG1LDynamics(DynamicsBase)`): one active upper layer, `q = lap psi - psi/rd^2`, masked inversion `psi_hat = -q_hat/(K2 + rd^-2)`, `dq/dt = -J(psi, q + beta*y) - rek*lap psi + curl_tau`. Mirrors `QGDynamics`'s spectral/RK4/filter + moving-storm wind machinery; `param_names=["beta","rd","rek","U1"]`; `state_dim = ny*nx`; `wind_amp=0` bitwise-unforced guard; `state_from_streamfunction` builds the model state from an upper-layer streamfunction.
- **Model op** `models/qg_dynamics.py`: `generate_wind_state` gained optional `x0`/`y0` (storm-start), and a new **`rollout_trajectory(state, steps, wind_state)`** returning the full `(steps+1, D)` path (index-aligned with `generate_full_trajectory`; reproduces it bitwise given the same IC+wind — regression-tested).
- **Calibration** `reports/calibrate_qg_alongtrack.py` (`--device`, default cuda): builds the S0/S1 datasets then measures **free divergence** `‖rollout_trajectory(true_ic, corrupted_wind) − true_state‖/‖true_state‖` under each scenario's `da_model`/`da_params`/corrupted wind. Output `reports/outputs/figs/qg_alongtrack_calibration.{png,json}` (5-panel: psi1 snapshot + tracks, obs Hovmöller, coverage, KE-by-`wind_amp`, divergence table). A sanity gate asserts `max(S0) < min(S1a)`.
  - Verified separation: S0 divergence `[3.7e-07 … 2.4e-04]` (round-trip) ≪ S1a `[0.48 … 1.35]` — **2033× separation**. Free-divergence `_build_dyn` must carry the window's `wind_amp`/`wind_sigma` so the wind-forcing gate (`_wind_curl_spectral` short-circuits on `wind_amp==0`) opens; the original neglect produced spurious S0 divergences ~0.5–1.1 for forced windows.
- **Tests** `tests/test_qg_s0s1.py` (14): window schema, shared-truth across scenarios, S0 da_params==true, S1a param-bias sign, corrupted-wind OU stats (loc jitter σ, amp bias/η with expected ratio `√((1+b)²+σ_frac²)`), track geometry (advance, mask, noise scale), density, determinism; plus `rollout_trajectory` regression tests in `tests/test_qg_dynamics.py`.
- **To do (next)**: QG DA baselines (EnKF/ETKF/4DVar on ψ/q) on the S0/S1 with held-out truth; then `train.py` dispatch + neural training on psi/q; parameter randomization as the final S0/S1 analog step.

## Experiments

| ID | Model | Hidden | Epochs | Train mix | Status |
|---|---|---|---|---|---|
| E1_direct_unet_default | DirectUNet | [64,128,256] | 200 | cs1+cs2 | config ready |
| E2_direct_unet_small | DirectUNet | [32,64,128] | 200 | cs1+cs2 | config ready |
| E3_direct_unet_rand | DirectUNet | [32,64,128] | 200 | cs1_rand+cs2_rand | config ready |
| F1_vanilla_cfm_default | VanillaCFM | [64,128,256] | 400 | cs1+cs2 | config ready |
| F2_vanilla_cfm_small | VanillaCFM | [32,64,128] | 400 | cs1+cs2 | config ready |
| F3_vanilla_cfm_rand | VanillaCFM | [32,64,128] | 400 | cs1_rand+cs2_rand | config ready |
| G1_vanilla_cfm_t0_default | VanillaCFM (τ=0) | [64,128,256] | 400 | cs1+cs2 | **to implement** |
| G2_vanilla_cfm_t0_small | VanillaCFM (τ=0) | [32,64,128] | 400 | cs1+cs2 | **to implement** |
| G3_vanilla_cfm_t0_rand | VanillaCFM (τ=0) | [32,64,128] | 400 | cs1_rand+cs2_rand | **to implement** |

## Phases

### Phase 0: Plan
- [x] Initial PLAN.md created
- [x] CS3/CS4 experiment plan in `docs/case_studies.tex`
- [x] Exp G (τ=0) experiment plan in `docs/experiment_G_tau0_cfm.md`

### Phase 1: Implementation (complete)
- [x] `models/direct_unet.py` — DirectUNet nn.Module
- [x] `models/vanilla_cfm.py` — VanillaCFM nn.Module with CFM loss + sampling
- [x] `data/random_param_dataset.py` — Randomized Lorenz-63 parameters per window
- [x] `conf/schema.py` — `DirectUNetConfig`, `VanillaCFMConfig`, `DataConfig` with CS3/CS4 fields
- [x] `training/lightning_module.py` — `LitModel` dispatches all 3 model types
- [x] `training/pipeline.py` — `create_trainer`, `train_stage`, `run_2stage_pipeline`
- [x] `train.py` — `model_factory`, `evaluate_model`, CS3/CS4 evaluation
- [x] 6 experiment YAML configs (E1-E3, F1-F3)
- [x] CS3/CS4 test cases in data generation, evaluation, and report
- [x] `evaluate_all.py` — Unified baseline + CFM comparison script
- [x] `reports/generate_unet_cfm_report.py` — CS3/CS4 report

### Phase 2: sbatch Infrastructure (this session)
- [x] `batch/run_lint.sbatch` — ruff + mypy in batch
- [x] `batch/run_test_suite.sbatch` — pytest (fast) in batch
- [x] `batch/run_config_validation.sbatch` — Hydra config + model factory validation
- [x] Deprecated duplicate `run_vanilla_experiments.sbatch` and interactive `run_tests.sh`

### Phase 3: τ=0 CFM Ablation (Exp G)
- [ ] `conf/schema.py` — add `train_tau_0_only: bool = False` to `VanillaCFMConfig`
- [ ] `models/vanilla_cfm.py` — τ=0 logic in `compute_cfm_loss` and `sample`
- [ ] `train.py` — wire `train_tau_0_only` flag through `model_factory`
- [ ] 3 config YAMLs: G1_vanilla_cfm_t0_default, G2_vanilla_cfm_t0_small, G3_vanilla_cfm_t0_rand
- [ ] Update `batch/run_one_epoch_tests.sbatch` + `batch/run_new_experiments.sbatch` with G1-G3
- [ ] Tests for τ=0 mode

### Phase 4: Verify (all via sbatch)
- [ ] `sbatch batch/run_config_validation.sbatch` — all 10 configs load
- [ ] `sbatch batch/run_lint.sbatch` — ruff + mypy pass
- [ ] `sbatch batch/run_test_suite.sbatch` — all fast tests pass
- [ ] `sbatch batch/run_one_epoch_tests.sbatch` — GPU smoke test (E1-F3 + G1-G3, 1 epoch)

### Phase 5: Launch
- [ ] `sbatch batch/run_new_experiments.sbatch` — full E1-F3 + G1-G3
- [ ] Collect results → `python reports/generate_experiment_report.py`
- [ ] Merge to master, push
- [ ] Update CHANGELOG.md

## Interfaces

### Model forward signatures (for LightningModule dispatch):
```
TweedieSolver:
  training_step(stage=1): model.estimate_mean(obs) → (B,T,D)
  training_step(stage=2): model(obs) → (B,T,D)
  config_optim(stage=1): model.mean_estimator.parameters()
  config_optim(stage=2): model.non_gaussian.parameters()

DirectUNet:
  training_step: model(obs) → (B,T,D)
  loss: StateMSELoss(pred, batch.states)
  config_optim: model.parameters()

VanillaCFM:
  training_step: compute_cfm_loss(batch) → scalar
  config_optim: model.parameters()
  sampling: model.sample(obs, N_outer) → (B,T,D)
```

### Dataset output format:
```python
{
    "true_state": Tensor(T, 3),
    "obs": Tensor(T, 3),
    "obs_mask": Tensor(T,),
    "forcing_true": Tensor(T,),
    "forcing_corrupted": Tensor(T,),
}
```

### results.json format (per experiment):
```json
{
  "experiment_id": "...",
  "config": {...},
  "epochs_trained": ...,
  "total_time_seconds": ...,
  "train_time_seconds": ...,
  "eval_time_seconds": ...,
  "fm_cs1": {"X": {"mean": ..., "std": ...}, "Y": ..., "Z": ..., "mean": ...},
  "fm_cs2": {...},
  "fm_cs3": {...},
  "fm_cs4": {...},
  "fm_degradation": ...,
  "fm_degradation_cs3cs4": ...
}
```
