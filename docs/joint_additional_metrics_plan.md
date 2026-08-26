# Step 1: L96 Joint Benchmark — Additional Metrics

Status: in progress 2026-08-26 · branch `feature/l96-joint-additional-metrics`

## Objective

Complement the L96 joint state-parameter neural benchmark
(`reports/l96/outputs/l96_joint_neural_benchmark.md`) with two additional
metrics that quantify the **parameter estimation quality** beyond the state
RMSE already reported:

1. **NRMSE** (normalized parameter RMSE) — `param_RMSE / mean(|true_param|)`
   per parameter and as a mean across the 8 parameters.
2. **Trajectory forecast skill** — MSE/RMSE/EV between a short 300-step rollout
   simulated with the **true** parameters and one with the **estimated**
   parameters, from the **same initial state and forcing** (sensitivity of
   short-term forecast quality to parameter error).

## Why 300 steps

Empirically measured with the real L96 dynamics (dt=0.001, Lyapunov
e-folding ~700-1100 steps), on the 24D observed subspace:

| scenario             | 100 steps | 300 steps | 500 steps |
|----------------------|-----------|-----------|-----------|
| L9-like (good params)| EV 0.994  | EV 0.965  | EV 0.920  |
| L7-like (poor params)| EV 0.978  | EV 0.843  | EV 0.692  |
| large-F error only   | EV 1.000  | EV 0.997  | EV 0.986  |

- 100 steps is too short (EV ~1.0 even for poor params — no discrimination).
- 500 steps puts the poor-param case below the preferred 0.5-0.9 range.
- **300 steps is the sweet spot**: good params stay ~0.96, poor params drop to
  ~0.84, clearly distinguished, both within the desired 0.5-0.9 band.

The divergence is driven mostly by the fast variables (Y1,Y2; sensitive to
eps/hx/fast_weights), which is exactly the physics the joint models must learn.

## Data flow

- The cached dataset windows store the full 40D `true_state` and `forcing_true`.
- The neural eval already returns `params_pred`/`params_true` as `(W, 8)`.
- For the rollout we need the 40D `x0 = true_state[0]` and clean `forcing_true`
  per window — `collate_joint_eval` currently drops `forcing_true`, so it must
  be threaded through.

## Files

| # | File | Change |
|---|------|--------|
| 1 | `evaluation/estimate_metrics.py` | Add `nrmse_param(pred_params, true_params)`, `trajectory_forecast_skill(dynamics, x0_batch, forcing_batch, params_true, params_est, n_steps, obs_var_indices)` |
| 2 | `evaluation/neural_inference.py` | `collate_joint_eval`: add `"forcing_true"`. `_run_case_inference` + `run_inference`: collect/return it. |
| 3 | `eval_joint_neural_l96.py` | Add `--n-compare-steps` (default 300); instantiate `Lorenz96Dynamics`; call new metrics; store `nrmse_param` + `traj_forecast` in JSON. |
| 4 | `reports/l96/generate_l96_joint_neural_report.py` | Add NRMSE table + Trajectory forecast skill table. |
| 5 | `tests/test_estimate_metrics.py` (new) | Unit tests for both functions. |
| 6 | `.github/workflows/ci.yml` | Add `tests/test_estimate_metrics.py` to the pytest gate. |

## Metrics

### NRMSE

```python
nrmse_d = sqrt(mean((pred - true)^2, axis=0)) / mean(|true|, axis=0)  # per-param
nrmse_mean = mean(nrmse_d)
```

Normalizes away the scale difference between F (~8) and eps (~0.1) so each
parameter contributes equally.

### Trajectory forecast skill

For each of the W windows: roll two 40D trajectories out of `x0` with the
forced sequence `forcing_true`, using true params vs estimated params, for
`n_steps`; subsample both to `obs_var_indices` (24D); pool across windows and
compute per-dim MSE/EV then group by slow / obs_fast / all_obs.

```json
"traj_forecast": {
  "n_steps": 300,
  "rmse": {"groups": {"slow", "obs_fast", "all_obs"}, "mean"},
  "ev":   {"groups": {"slow", "obs_fast", "all_obs"}, "mean"},
}
```

## Compute

- Rollouts: CPU RK4, ~0.05 s per 300-step window x 200 windows = ~10 s per
  model. Negligible; same sbatch eval jobs as before, just with the extra step.

## Verification

- Unit tests for `nrmse_param` and `trajectory_forecast_skill`.
- Re-run single-sample eval on L7/L8/L9 and ens30 eval on L7/L9; check new JSON
  keys present and sensible; regenerate the report.

## Note on the shared working tree (2026-08-26)

The main repo working tree is shared with other opencode sessions, so concurrent
branch switches can wipe uncommitted tracked-file edits. This work is done in an
**isolated git worktree** (`/tmp/opencode/l96-joint-additional-metrics`) to avoid
collisions; commits go to the shared remote ref `feature/l96-joint-additional-metrics`
and the PR targets `master`.

## Next step (after this PR merges)

Step 2: `feature/l96-joint-etkf-da` — run `JointETKFL96` on S0/S1 on GPU, read
vanilla ETKF from cache
(`experiments/l96_baselines_dws500_s0c_inf2.0_etkf_inf2.0_obsj2_int100_fw.json`,
S0 RMSE 0.866 / S1 1.471), populate the DA rows in the same benchmark report.
