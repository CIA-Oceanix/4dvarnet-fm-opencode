# V3 PredictStateCFM Implementation

## Overview

**V3 PredictStateCFM** is a single-stage CFM variant where the network predicts the final mean state `μ = E[x1|xt,y]` directly rather than the velocity residual `v = (x1-x0) / (1-τ)`.

### ODE Formulation

```
v = (μ - x) / (1 - τ) where μ = E[x1 | x_τ, y]
```

This represents a backward-drift mechanism that pulls the state toward the predicted final state.

## Branch

`feature/l96-predict-state-cfm-clean` (ready for merge to master)

## Key Characteristics

- **Single-stage**: Trains end-to-end; no freezing or stage switching
- **ODE formulation**: Forward integration with tau-rescaled velocity
- **Backward-drift**: Pulls states toward predicted final state
- **Same interface as VanillaCFM**: Passes obs + forcing/params as input, outputs state

## Required Changes for V3

✅ `PredictStateCFM` class in `models/vanilla_cfm.py` — 80 lines
✅ `PredictStateCFMConfig` schema in `conf/schema.py`
✅ `train.py` dispatch (model_factory, evaluate_model, save_trajectories)
✅ `training/lightning_module.py` dispatch (_forward_and_loss, forward)
✅ Config: `config/experiment/V3_predict_state_cfm_l96.yaml`
✅ Sbatch: `batch/run_l96_cfm_v3_train.sbatch`
✅ Tests: `tests/test_predict_state_cfm.py` (8/8 pass)
✅ CI: Added to test path + feature/* triggers

## Implementation Details

### `PredictStateCFM.__init__()`
```python
def __init__(self, state_dim=3, hidden_channels=None, time_emb_dim=64,
             N_outer=10, sigma_prior=0.5, dropout=0.1,
             train_tau_0_only=False, param_dim=4, cond_extra_dim=0)
```

### `forward(x_t, batch, tau)`
- Takes current state `x_t`, observations `batch`, time interpolation `tau`
- Predicts final state mean `μ = E[x1|x_τ, batch]`
- ODE: `v = (μ - x) / (1 - τ)`

### `compute_loss(batch)`
- Single-stage CFM loss: `MSE(μ, x1)`
- Training: `τ ~ Uniform(0, 1)` per-sample, or `τ = 0` if `train_tau_0_only=True`

### `sample(batch, N_outer=None)`
- Sample trajectories via forward ODE integration
- Integration loop: for `step in range(N_outer)`: `x += dt * (μ - x) / (1 - τ_step)`
- Final: `x1` is sampled trajectory

## Architecture

```
Input: x_t (B,T,D), batch.obs (B,T,D), batch.forcing, batch.params
      ↓
ConditionEncoder: [obs, forcing, params] cond
      ↓
UNet1D: cond → μ ~ N(μ_pred, σ²)
      ↓
Output: μ (B,T,D)
```

## Training Config

L96 benchmark:
- `state_dim=24`, `param_dim=0` (obs-only training)
- `hidden_channels=[64,128,256]`, `time_emb_dim=64`
- `N_outer=10`, `sigma_prior=0.5`, `dropout=0.1`
- `cond_extra_dim=0` (no forcing/params conditioning)
- `train_tau_0_only=False` (multi-τ training)
- `epochs=400`

## Testing

```
pytest tests/test_predict_state_cfm.py -v -m "not slow"
```

All 8 tests pass:
1. `test_forward_shape`
2. `test_forward_cond_extra_dim_gt0`
3. `test_compute_loss_shape`
4. `test_sample_shape`
5. `test_sample_finite`
6. `test_sample_train_tau_0_only`
7. `test_init_params`
8. `test_init_default_channels`

## Smoke Test Results

| Config | Loss (epoch 0) | Loss (epoch 5) | Sample Shape |
|--------|----------------|----------------|--------------|
| L63 default (3D) | 1.02 | 0.97 | (4, 50, 3) |
| L96 default (24D) | 0.99 | 0.98 | (4, 50, 24) |
| L96 small ([32,64]) | 0.94 | 0.95 | (4, 50, 24) |

V3 trades correctly and samples correctly on all configurations.

## Next Steps

1. **Phase 1 (DONE)**: Implement V3 and verify tests
2. **Phase 2**: Merge V3 to master (current branch)
3. **Phase 3**: Implement V2 TweedieCFM (two-stage variant)
   - See `PHASE2_L96_TWEEDIE_CFM_PLAN.md` for detailed 9-step plan
   - Critical bug to avoid: `velocity_unet obs_dim = 2*state_dim`
   - Two-stage training: freeze mean_estimator → train velocity (or vice versa)
4. **Phase 4**: Run V3 V2 cluster training and evaluation

## References

- Implementation: `models/vanilla_cfm.py` lines 156-235
- Schema: `conf/schema.py` lines 168-177 (`PredictStateCFMConfig`)
- Tests: `tests/test_predict_state_cfm.py`
- Config: `config/experiment/V3_predict_state_cfm_l96.yaml`
- Sbatch: `batch/run_l96_cfm_v3_train.sbatch`
- Plan: `PHASE2_L96_TWEEDIE_CFM_PLAN.md`
- Plan: `docs/phase_B_l96_cfm_variants.md`
