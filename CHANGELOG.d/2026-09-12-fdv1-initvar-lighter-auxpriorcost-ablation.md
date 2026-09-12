## 2026-09-12: L96 — FDV1 ablation: random init + 1/10th aux prior-cost weight

**Summary:** New experiment config
`FDV1_obsstate_monai_l96_initvar01_auxpriorcost01.yaml`: FDV1's own
`update_input=obs+state` architecture with `init_state_var=0.1` (random
initial condition, from the 2026-09-12 init-state-var feature) AND
`aux_var_cost_weight=0.01` -- 1/10th the weight used in the killed job
53287's `aux_var_cost_weight=0.1` config.
**Files modified:** new
`config/experiment/FDV1_obsstate_monai_l96_initvar01_auxpriorcost01.yaml`,
`batch/run_l96_fdv1_obsstate_monai_initvar01_auxpriorcost01_train.sbatch`.
No code changes -- both `init_state_var` and `aux_var_cost_weight` already
existed as configurable `FourDVarNetSolver` params.
**Rationale:** Two FDV1 ablations run in parallel this session, both
keeping `update_input=obs+state` untouched: (1) job 53288
(`init_state_var=0.1` alone) tracked/beat plain FDV1's own convergence pace
-- random init is NOT harmful; (2) job 53287 (`aux_var_cost_weight=0.1`
alone) plateaued at val_loss ~1.22-1.32 by epoch 104-122 vs plain FDV1's own
~0.86-0.95 at the same epochs -- a real, substantial degradation directly
attributable to the auxiliary prior-consistency loss term. Job 53287 was
killed once this was clear. This run combines the (harmless) random init
with a 10x lighter aux weight, to test whether a lighter touch on the aux
term avoids the degradation seen at 0.1.
**Verification:** Manual smoke test with matched random seeds across two
`forward()` calls: confirmed `compute_loss()`'s output matches
`MSE + 0.01*(prior_cost(pred)+prior_cost(true))` to numerical precision,
`init_state_var=0.1` produces a stochastic `x_0` (two calls on the same
batch produce different outputs), and gradients reach both `unet` and
`prior_unet`. No new tests needed (both features individually tested when
introduced); this is a config-only combination of two already-verified
knobs.
