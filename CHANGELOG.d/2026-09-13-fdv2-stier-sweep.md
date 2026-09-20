## 2026-09-13: L96 — FDV2 grad+state/subgrad+state at S-tier, corrected fixes

**Summary:** New experiment configs
`FDV2_grad_state_monai_l96_Stier.yaml`/`FDV2_subgrad_state_monai_l96_Stier.yaml`:
FDV2's two gradient-conditioned `update_input` modes on the S-tier
MonaiUNet1D (`hidden_channels=[32,64,128]`, `monai_num_res_blocks=1`,
1,055,544 main-solver params, vs the M tier's `hidden_channels=[64,128,256]`,
`monai_num_res_blocks=2`, 5,889,048 params every earlier FDV2 attempt this
session used), combined with `init_state_var=0.1` (random initial
condition) and `aux_var_cost_weight=0.01` (the corrected pure
`prior_cost(x_final)+prior_cost(states)` auxiliary loss, no `prior_weight`/
`obs_cost` mixed in -- 2026-09-11 fix). `grad+state` additionally carries
the earlier `grad_clip_range=5.0` tanh soft-clip fix.
**Files modified:** new `config/experiment/FDV2_grad_state_monai_l96_Stier.yaml`,
`config/experiment/FDV2_subgrad_state_monai_l96_Stier.yaml`,
`batch/run_l96_fdv2_grad_state_monai_Stier_train.sbatch`,
`batch/run_l96_fdv2_subgrad_state_monai_Stier_train.sbatch`. No code
changes -- all knobs already exist as configurable `FourDVarNetSolver`
params (PR #196).
**Rationale:** An FDV1 (`update_input=obs+state`) capacity sweep this
session found the auxiliary prior-consistency loss term severely and
permanently degrades val_loss at the M-tier capacity (`aux_var_cost_weight
=0.01` -> plateau ~0.38 vs ~0.20-0.29 for aux-loss-free variants, job
53289) but causes no degradation at all at the S tier (job 53315, matched
its own clean control exactly) -- and the S-tier + this aux weight
combination went on to set a new best RMSE for the whole L96 consolidated
benchmark once hybridized with SDA3 (PR #200). Every earlier FDV2
gradient-conditioned attempt this session (`grad+state`: jobs 52926, 53104,
53258; `subgrad+state`: jobs 53011, 53045, 53077, 53078, 53254, 53259) was
at M-tier (or the priorS+tbptt narrow-prior variant) only, and none got a
clean run under the normalization fix, the corrected aux-loss formula, AND
a capacity small enough to plausibly tolerate that term -- this tests
whether the same tier-dependent story holds for FDV2's gradient-conditioned
update rules, not just FDV1's plain residual-UNet one.
**Verification:** Manual smoke test for both configs: confirmed
`aux_var_cost_weight=0.01`/`init_state_var=0.1` correctly wired, `unet`
param count matches the S tier (1,055,544 for `grad+state`; 1,057,848 for
`subgrad+state` -- expected, its input channel multiplier differs, 3x
state_dim vs `grad+state`'s 2x), one `compute_loss()`+`backward()` step
runs cleanly with gradients reaching both `unet` and `prior_unet` for both
configs.
