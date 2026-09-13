## 2026-09-12: L96 — configurable monai_num_res_blocks (S-tier ablation)

**Summary:** New `FourDVarNetSolver`/`_build_backbone_unet` param
`monai_num_res_blocks` (default `2`, `MonaiUNet1D`'s own existing default --
backward-compatible, matches every current FDV1/FDV2 config's actual
behavior exactly). Together with `hidden_channels` this selects a
`MonaiUNet1D` capacity tier from the `project_l96_monai_unet_complexity_tiers`
memory's ladder: `hidden_channels=[64,128,256]` + `monai_num_res_blocks=2` is
"M" (5,889,048 params, today's default everywhere); `[32,64,128]` +
`monai_num_res_blocks=1` is "S" (1,055,544 params). New experiment config
`FDV1_obsstate_monai_l96_initvar01_Stier.yaml`: same as
`FDV1_obsstate_monai_l96_initvar01.yaml` (`update_input=obs+state`,
`init_state_var=0.1`) but on the S tier instead of M.
**Files modified:** `models/fourdvarnet.py` (`_build_backbone_unet`,
`FourDVarNetSolver.__init__` -- threaded to both `self.unet` and
`self.prior_unet`); `conf/schema.py` (`FourDVarNetConfig.monai_num_res_blocks`);
`train.py` (model_factory threading); `tests/test_fourdvarnet_monai.py`
(new `TestMonaiNumResBlocks`: default reproduces the exact M-tier param
count 5,889,048; `monai_num_res_blocks=1` + S-tier `hidden_channels`
reproduces the exact S-tier count 1,055,544 for the main solver unet and
1,053,240 for `prior_unet` -- a real, expected difference since `prior_unet`
takes state-only 24ch input vs. the main unet's `obs+state` 48ch input, not
a bug); new `config/experiment/FDV1_obsstate_monai_l96_initvar01_Stier.yaml`,
`batch/run_l96_fdv1_obsstate_monai_initvar01_Stier_train.sbatch`.
**Rationale:** User asked to confirm FDV1's UNet is M-tier (confirmed: yes,
verified by direct instantiation, 5,889,048 params, `num_res_blocks` was
never previously exposed/overridden anywhere in the FDV call chain) and
then asked for a new FDV1 run (same as the confirmed-harmless
`init_state_var=0.1` ablation) on the smaller S tier -- to test whether a
much smaller UNet still reproduces FDV1's healthy fast-Y reconstruction, as
a capacity-sensitivity check.
**Verification:** `pytest tests/test_fourdvarnet.py tests/test_fourdvarnet_monai.py
tests/test_lightning_module.py` -- 120 passed (117 pre-existing + 3 new).
`ruff check` clean (pre-existing E402 in test_fourdvarnet_monai.py from its
`importorskip` pattern, unrelated to this change). Manual smoke test: built
the model via `model_factory` from the new config, confirmed
`sum(p.numel() for p in model.unet.parameters()) == 1,055,544` exactly, ran
one `compute_loss()`+`backward()` step, confirmed the random init is
correctly still active (two `forward()` calls differ).
