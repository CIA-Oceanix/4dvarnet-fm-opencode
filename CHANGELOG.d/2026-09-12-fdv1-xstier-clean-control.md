## 2026-09-12: L96 — FDV1 clean control at the ~half-S-tier capacity

**Summary:** New experiment config `FDV1_obsstate_monai_l96_initvar01_XStier.yaml`:
`update_input=obs+state`, `init_state_var=0.1`, the same
`hidden_channels=[24,48,96]`/`monai_num_res_blocks=1`/`monai_norm_num_groups=8`
(596,784 params) tier as
`FDV1_obsstate_monai_l96_initvar01_XStier_auxpriorcost01.yaml` (job 53316),
but `aux_var_cost_weight=0.0` (default -- no `prior_unet`, no auxiliary
loss). Mirrors the existing clean-vs-aux paired-comparison pattern already
run at M-tier (`FDV1_obsstate_monai_l96_initvar01.yaml`, job 53288) and
S-tier (`FDV1_obsstate_monai_l96_initvar01_Stier.yaml`, job 53292).
**Files modified:** new `config/experiment/FDV1_obsstate_monai_l96_initvar01_XStier.yaml`,
`batch/run_l96_fdv1_obsstate_monai_initvar01_XStier_train.sbatch`. No code
changes.
**Rationale:** Job 53316 (same ~half-S-tier + `aux_var_cost_weight=0.01`)
is tracking meaningfully behind job 53315 (S-tier + same aux weight) at
matched epochs (e.g. epoch 32: val_loss 0.409 [S-tier] vs 0.947 [half-S-tier],
epoch 52: 0.302 vs 0.417) -- a real, widening gap. Without this clean
control it's ambiguous whether that gap is the aux-loss degradation
(confirmed severe at M-tier, absent at S-tier) reappearing at this smaller
capacity, or simply this tier's own natural (aux-loss-free) convergence
pace being slower for unrelated reasons. This run isolates that.
**Verification:** Manual smoke test: confirmed `unet` param count is
596,784, `prior_unet is None` (aux term correctly inactive at weight 0.0),
one `compute_loss()`+`backward()` step runs, two consecutive `forward()`
calls differ (random init confirmed active). No new automated tests --
config-only combination of already-tested knobs.
