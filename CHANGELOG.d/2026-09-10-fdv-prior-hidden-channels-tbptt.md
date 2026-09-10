## 2026-09-10: L96 — FourDVarNetSolver prior_hidden_channels + truncated-BPTT blocks

**Summary:** Added two configurable architecture/training options to
`FourDVarNetSolver`, tested together in a new
`FDV2_subgrad_state_monai_l96_priorS_tbptt.yaml` config (job 53045). (1)
`prior_hidden_channels` (`None` default -- `prior_unet` shares the main
solver unet's `hidden_channels`, unchanged behavior): lets the prior
operator be narrower than the solver, matching
`CIA-Oceanix/4dvarnet-global-mapping`'s `ronan_devs` branch convention
(`glo12-sla-4th-unrolling-ossev1.yaml`: prior `model_channels=32` vs solver
`model_channels=64`) -- this codebase previously always gave both networks
equal capacity. (2) `tbptt_n_blocks`/`tbptt_block_size` (default `1`/`None`
-> derived as `N_outer`, unchanged single-continuous-graph behavior): splits
the `N_outer` unroll into detached blocks, ported from `ronan_devs`'
`Lit4dVarNetTwoSolvers.base_step` detach()-at-a-boundary +
averaged-multi-stage-loss pattern, adapted to one weight-tied solver called
repeatedly instead of two distinct solver instances. `compute_loss()`
averages an MSE term over every block's end-of-block state only in
training mode; validation/eval always scores the final iteration's answer
alone, so `val_loss`/`stage1_best.ckpt` checkpoint selection is unaffected
by `tbptt_n_blocks`.
**Files modified:** `conf/schema.py` (`FourDVarNetConfig` gains the 3 new
fields); `models/fourdvarnet.py` (`FourDVarNetSolver.__init__` validates
`tbptt_n_blocks*tbptt_block_size==N_outer`; `prior_unet` construction uses
`prior_hidden_channels` when given; `forward()`/`compute_loss()` refactored
around a shared `_unrolled_blocks()` helper); `train.py` (model_factory
threads the 3 fields through, backward-compatible defaults);
`tests/test_fourdvarnet.py` (`TestPriorHiddenChannels`,
`TestTruncatedBPTT`); new `config/experiment/FDV2_subgrad_state_monai_l96_priorS_tbptt.yaml`
+ `batch/run_l96_fdv2_subgrad_state_monai_priorS_tbptt_train.sbatch`.
**Rationale:** User asked to test whether giving the solver more relative
capacity than the prior (matching the reference implementation's own
design) speeds up training, and separately requested truncated-BPTT over
2 blocks of 5 iterations (both block count and block size configurable, not
hardcoded) to bound backward-pass memory/compute -- explicitly requiring
`val_loss` to keep reflecting only the true final-iteration answer, not a
block average, so checkpoint selection stays meaningful.
**Verification:** `pytest tests/test_fourdvarnet.py tests/test_fourdvarnet_monai.py
tests/test_lightning_module.py tests/test_l96_normalization_configs.py` --
144 passed. `ruff check` clean. New config: 7.4M params (vs the
equal-capacity `FDV2_subgrad_state_monai_l96`'s 11.8M); 1- and 2-epoch smoke
tests ran cleanly with `train_loss`/`val_loss` visibly distinct per epoch as
expected; real 400-epoch run launched as job 53045 (A100), training cleanly.
