## 2026-09-12: L96 — FDV1 ablation: ~half-S-tier UNet + aux prior-cost weight

**Summary:** New experiment config
`FDV1_obsstate_monai_l96_initvar01_XStier_auxpriorcost01.yaml`: same as
`FDV1_obsstate_monai_l96_initvar01_Stier_auxpriorcost01.yaml`
(`update_input=obs+state`, `init_state_var=0.1`, `aux_var_cost_weight=0.01`)
but on `hidden_channels=[24,48,96]`, `monai_num_res_blocks=1`,
`monai_norm_num_groups=8` -- 596,784 params, the closest achievable ratio to
"half the S tier's 1,055,544 params" among candidates tried
(`[16,32,64]`x1=267,944 [25%], `[20,40,80]`x1=416,124 [39%],
`[24,48,96]`x1=596,784 [57%], `[28,56,112]`x1=809,924 [77%]). Not an
official tier name in `project_l96_monai_unet_complexity_tiers` -- an
interpolated point, added on request.
**Files modified:** new
`config/experiment/FDV1_obsstate_monai_l96_initvar01_XStier_auxpriorcost01.yaml`,
`batch/run_l96_fdv1_obsstate_monai_initvar01_XStier_auxpriorcost01_train.sbatch`.
No code changes -- all knobs already exist as configurable
`FourDVarNetSolver` params.
**Rationale:** Job 53289 (M-tier, 5,889,048 params, random init +
`aux_var_cost_weight=0.01`) plateaued permanently at val_loss ~0.384 --
severe, lasting degradation vs every aux-loss-free FDV1 variant
(0.20-0.29). Job 53315 (S-tier, 1,055,544 params, same random init + aux
weight) instead converged cleanly, tracking well below 53289's final
plateau by epoch ~50 and still improving -- suggesting the degradation is
tier/capacity-dependent, not an intrinsic property of the aux loss term
itself (consistent with a gradient-cross-talk-through-`x_final` mechanism
that scales with the main solver UNet's own size/capacity). This run tests
whether an even smaller UNet continues that trend (degradation shrinking
further) or there's a floor below which the interference reappears.
**Verification:** Manual smoke test with matched random seeds: confirmed
`compute_loss()` output matches `MSE + 0.01*(prior_cost(pred)+prior_cost(true))`
exactly, `unet` param count is 596,784, gradients reach both `unet` and
`prior_unet`. No new tests needed -- config-only combination of
already-tested knobs (`init_state_var`, `aux_var_cost_weight`,
`hidden_channels`/`monai_num_res_blocks`/`monai_norm_num_groups`).
