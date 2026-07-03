# Joint State-Parameter Estimation: Progress Log

## Status: In Progress | Started: 2026-07-02

## Branch: `feature/joint-estimation`

---

## Implementation Checklist

### Wave 1A: UNet output_dim (`models/unet.py`)
- [x] Add `output_dim` param to `UNet1D.__init__` (default `None` → `state_dim`)
- [x] Modify `enc_out` final conv to use `output_dim`
- [x] **Test:** `UNet1D(state_dim=3, output_dim=6)(x_in, obs, tau)` → `(B, 6, T)`

### Wave 1B: Data + Config
- [x] **`data/random_param_dataset.py`** — support `param_noise` as tuple `(lo, hi)`
- [x] **`data/dataloader.py`** — add `params` field to `FlowMatchingBatch`
- [x] **`data/dataloader.py`** — extend `collate_fm` to stack params
- [x] **`conf/schema.py`** — add `JointCFMConfig` dataclass

### Wave 2A: JointCFM model (`models/vanilla_cfm.py`)
- [x] `JointCFM(VanillaCFM).__init__` — UNet with `output_dim=6`, store `param_loss_weight`
- [x] `forward(x_t, obs, tau)` → `(v_state (B,T,3), param_feats (B,T,3))`
- [x] `estimate_params(obs)` → `(B, 3)` with softplus
- [x] `compute_cfm_loss(batch)` — CFM + param MSE
- [x] `sample(obs)` — Euler integration + param estimation
- [x] τ=0 mode: `train_tau_0_only` in loss and sample
- [x] **Fast test:** forward shapes, loss scalar, backward
- [ ] **GPU test:** JointCFM on CS4, state RMSE < 3.0

### Wave 2B: Joint DA baselines (`evaluation/baselines.py`)
- [x] **JointWeak4DVar** — optimize `x0, q, log(σ), log(ρ), log(β)`
- [x] **JointStrong4DVar** — optimize `x0, log(σ), log(ρ), log(β)` (no q)
- [x] **JointEnKF** — 6D state `[X,Y,Z,σ,ρ,β]`, H = [I₃, 0₃ₓ₃]
- [x] **JointETKF** — 6D state, SVD on observed part
- [ ] **GPU test (all 4 DA methods):** state RMSE within 20% of non-joint variant

### Wave 3: Pipeline integration
- [x] **`training/lightning_module.py`** — add `"joint_cfm"` dispatch
- [x] **`train.py`** — add `joint_cfm` to `model_factory`, `evaluate_model`, `save_trajectories`
- [x] **`evaluation/metrics.py`** — add `param_rmse` function
- [x] **`evaluation/run.py`** — add joint baseline methods to pool + method list

### Wave 4: Tests + Configs
- [x] **`tests/test_joint_estimation.py`** — 11 fast tests passing, 4 slow tests
- [x] **`config/experiment/H1_joint_cfm_default.yaml`**
- [x] **`config/experiment/H2_joint_cfm_tau0.yaml`**
- [ ] **`docs/joint_estimation_design.md`** — design document

### Wave 5: Comparison + Evaluation
- [x] `BaselineResult.params` field (shape `(num_steps, 3)`)
- [x] All 4 joint methods save param estimates in `assimilate` + `assimilate_batch`
- [x] `eval_joint_comparison.py` — vanilla vs joint state/param RMSE table on CS3/CS4

### Integration Smoke Test (RTX8000)
- [x] `pytest tests/test_joint_estimation.py -v -m "fast"` — 12/12 pass
- [x] `pytest tests/test_joint_estimation.py -v -m "slow"` — 4/4 pass
- [x] State RMSE comparison on CS3/CS4 (da_window_steps=50, batch_size=200)

---

## Test Results Log

| Date | Test | Result | Notes |
|------|------|--------|-------|
| 2026-07-02 | `test_joint_estimation.py -k "not slow"` | 12/12 pass | UNet output_dim, JointCFM forward/loss/sample/tau0 all OK |
| 2026-07-02 | Full test suite `-k "not slow"` | 119/119 pass | No regressions in existing tests |
| 2026-07-02 | `test_joint_estimation.py -m "slow"` | 4/4 pass | All 4 joint DA methods (6.72s) |
| 2026-07-02 | `eval_joint_comparison.py` | Run on GPU | CS3/CS4, 100 windows, batch_size=200, dws=50 |

## State RMSE Comparison (dws=50, batch_size=200)

### opt_steps=60 / max_iter=15 (original)

| Method | CS3 Joint/Vanilla | CS4 Joint/Vanilla | Param RMSE CS3 | Param RMSE CS4 |
|--------|:---:|:---:|:---:|:---:|
| Weak-4DVar | 1.20x | 1.20x | 1.68/1.90/0.29 | 1.91/1.77/0.26 |
| Strong-4DVar | 1.98x | 1.82x | 20.63/3.96/0.65 | 12.93/4.07/0.70 |
| EnKF | 0.77x | 0.49x | 0.64/0.47/0.12 | 0.97/0.74/0.20 |
| ETKF | 1.21x | 0.91x | 0.41/0.62/0.11 | 0.92/0.76/0.18 |

### opt_steps=200 / max_iter=50

All 4DVar methods updated; EnKF/ETKF methods unchanged (use one-step filter, no iteration).

| Method | CS3 State RMSE | CS4 State RMSE | CS3 Joint/Vanilla | CS4 Joint/Vanilla | Param RMSE CS3 | Param RMSE CS4 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Weak-4DVar | **0.71** | **0.81** | 1.00 | 1.00 | — | — |
| Joint-Weak-4DVar | **0.95** | **0.96** | 1.35x | 1.26x | 2.24/1.95/0.33 | 2.66/1.72/0.30 |
| Strong-4DVar | **0.68** | **0.79** | 1.00 | 1.00 | — | — |
| Joint-Strong-4DVar | **1.04** | **1.34** | 1.54x | 1.71x | 15.05/3.66/0.44 | 13.39/3.56/0.61 |
| EnKF | **1.26** | **2.48** | 1.00 | 1.00 | — | — |
| Joint-EnKF | **1.04** | **1.24** | 0.82x | 0.49x | 0.64/0.50/0.11 | 0.94/0.78/0.19 |
| ETKF | **1.15** | **2.17** | 1.00 | 1.00 | — | — |
| Joint-ETKF | **1.36** | **2.01** | 1.17x | 0.93x | 0.40/0.58/0.12 | 1.05/0.83/0.18 |

### Effect of more Adam steps (60 → 200) on 4DVar methods

| Method | CS3 State RMSE Δ | CS4 State RMSE Δ | Joint/Vanilla ratio Δ | Param RMSE Δ |
|--------|:---:|:---:|:---:|:---:|
| Vanilla Weak-4DVar | 0.84→**0.71** (↓16%) | 0.81→**0.81** (no change) | — | — |
| Joint-Weak-4DVar | 1.01→**0.95** (↓6%) | 0.98→**0.96** (↓2%) | 1.20→**1.35** (↑) | σ RMSE ↑ (overfit to window) |
| Vanilla Strong-4DVar | 0.64→**0.68** (↑6%) | 0.81→**0.79** (↓2%) | — | — |
| Joint-Strong-4DVar | 1.45→**1.04** (↓28%) | 1.48→**1.34** (↓9%) | 1.98→**1.54** (↓) | σ RMSE ↓ 20.6→15.0 |

Key findings:
- **More steps helps vanilla Weak-4DVar on CS3** (16% improvement) — the smoother converges better with more iterations
- **More steps helps Joint-Strong-4DVar significantly** (ratio 1.98→1.54) — the params converge closer to truth, reducing trajectory corruption
- **More steps hurts Joint-Weak-4DVar's param RMSE** — the q-control + params have too many degrees of freedom; 200 steps lets the optimizer overfit param estimates to compensate for q
- **Joint-EnKF remains the best joint method** — 0.49x (CS4) and 0.82x (CS3) ratio vs vanilla, with best param RMSE

## Blocker History

| Date | Issue | Status | Resolution |
|------|-------|--------|------------|
|      |       |        |            |

## Commits

| Hash | Time | Changes |
|------|------|---------|
|      |      |         |
