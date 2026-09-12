## 2026-09-12: L96 — FDV1 ablation: S-tier UNet + aux prior-cost weight

**Summary:** New experiment config
`FDV1_obsstate_monai_l96_initvar01_Stier_auxpriorcost01.yaml`: same as
`FDV1_obsstate_monai_l96_initvar01_Stier.yaml` (S-tier MonaiUNet1D,
`hidden_channels=[32,64,128]`, `monai_num_res_blocks=1`, 1,055,544 params;
`init_state_var=0.1`) plus `aux_var_cost_weight=0.01`, the same weight used
in the M-tier ablation (job 53289).
**Files modified:** new
`config/experiment/FDV1_obsstate_monai_l96_initvar01_Stier_auxpriorcost01.yaml`,
`batch/run_l96_fdv1_obsstate_monai_initvar01_Stier_auxpriorcost01_train.sbatch`.
No code changes -- all three knobs (`init_state_var`, `aux_var_cost_weight`,
`monai_num_res_blocks`) already exist as configurable `FourDVarNetSolver`
params.
**Rationale:** Job 53292 (S-tier + random init, no aux loss) just completed
with a FULL eval confirming genuinely healthy fast-Y reconstruction
(variance ratio 0.975/0.976 vs FDV1 baseline's 0.926, correlation
0.951/0.952 vs 0.948) and a final val_loss (0.197) that actually beats
plain zero-init FDV1's own 0.205 -- the S tier is a real improvement, not
an artifact. Separately, job 53289 (M-tier + random init +
`aux_var_cost_weight=0.01`) plateaued at val_loss ~0.39 by epoch 323, a
substantial, lasting degradation vs every aux-loss-free FDV1 variant
(0.20-0.29 range) despite the aux term contributing only ~2% of the total
loss value (measured directly on that checkpoint) -- likely gradient
cross-talk through `x_final` (which depends on the main solver UNet's own
parameters) rather than the loss magnitude itself. This run tests whether
that same degradation appears on the S tier too, isolating whether it's
specific to M-tier's particular prior_unet/solver-unet capacity
combination or a more general property of the aux loss term.
**Verification:** Manual smoke test with matched random seeds: confirmed
`compute_loss()` output matches `MSE + 0.01*(prior_cost(pred)+prior_cost(true))`
exactly, `unet` param count is 1,055,544 (S tier), `prior_unet` param count
is 1,053,240, gradients reach both `unet` and `prior_unet`. No new tests
needed -- config-only combination of three already-tested knobs.
