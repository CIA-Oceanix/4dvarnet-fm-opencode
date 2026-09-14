## 2026-09-14: Q6 redesign -- `monai2d` backbone (2D circular convs, T merged into channels)

**Summary:** Q6's first launch (`unet_backbone="monai2d"` -- previously
`"monai"`, MonaiUNet1D) trained for 126/400 epochs with a flat,
non-decreasing loss (killed, not a crash -- see PLAN.md's 2026-09-14
note). Redesigned to use a genuinely 2D circular-conv backbone (matching
Q1's own `MonaiDirectUNetQG` architecture) instead of `MonaiUNet1D`'s
flattened 1D treatment of the whole state vector, merging the T (days)
axis into the *channel* dimension rather than either the batch dimension
(`MonaiDirectUNetQG`'s own convention, fully independent per-day
processing) or a downsampled 1D-sequence axis (the old backbone's
convention, which needed a pad/crop workaround for QG's 30-day windows).

**Files modified:**
- `models/monai_unet_qg2d.py` -- new `MonaiUNet2DQGSolver`: drop-in
  backbone for `FourDVarNetSolver` matching `MonaiUNet1D`'s exact
  `forward(x, tau=...)` `(B, C, T)` in/out convention, internally
  reshaping `(B, C, T) -> (B, T*channels_per_day, ny, nx)` and back
  through `MonaiUNet2DCircular`.
- `models/fourdvarnet.py` -- new `unet_backbone="monai2d"` option in
  `_build_backbone_unet`/`FourDVarNetSolver.__init__` (new `qg_T`/
  `qg_ny`/`qg_nx` constructor args, required only for this backbone,
  threaded through both `self.unet` and `self.prior_unet`). Zero changes
  needed to `_solver_iteration`/`_prior_ae`/`forward()` (all shared with
  L96) since the new backbone matches the existing call convention
  exactly.
- `train_qg_neural.py` -- `build_model()` passes `qg_T=num_days(cfg),
  qg_ny=cfg.ny, qg_nx=cfg.nx` automatically for `model_type="fourdvarnet"`;
  the T-axis padding wrapper (`_pad_batch_for_monai1d`/`_padded_prior_cost`,
  needed only for the old `unet_backbone="monai"`) is now conditional on
  `unet_backbone == "monai"` specifically, with a new `_plain_prior_cost`
  for backbones (`unet1d`, `monai2d`) that have no T-divisibility
  constraint to work around.
- `config/experiment/Q6_fourdvarnet_s0.yaml` -- `unet_backbone: monai2d`,
  `monai_norm_num_groups: 4` (was 32 -- MONAI requires every channel
  count, including this backbone's internal `T*channels_per_day` totals,
  divisible by this).
- `tests/test_monai_unet_qg2d.py`, `tests/test_qg_neural.py` -- 5 new
  tests: `MonaiUNet2DQGSolver` construction/forward/backward + its two
  `ValueError` guards, a `FourDVarNetSolver(unet_backbone="monai2d")`
  end-to-end Lightning forward/backward test, and the Q6 YAML config
  sanity check (including the `monai_norm_num_groups` divisibility check).

**Rationale:** The flattened 1D backbone has no 2D locality/periodicity
prior at all for QG's genuinely 2D domain -- user feedback identified this
as the likely cause of the flat-loss run (not confirmed to be a bug, but a
plausible architecture mismatch) and proposed channel-merging T instead of
batch-folding it, which also elegantly removes the T-divisibility
constraint that needed the earlier pad/crop workaround.

**Verification:** `pytest tests/test_qg_neural.py tests/test_fourdvarnet.py
tests/test_monai_unet_qg2d.py -m "not slow"` (fdv + fdv-monai-proto envs)
-- all passing except the same pre-existing, order-dependent flaky test
already documented in the prior Q6 fragment (confirmed unrelated).
`ruff check` clean. Verified full production-scale construction/forward/
backward on CPU (nx=64, state_dim=8192, T=30: 5,157,432 params, finite
loss). 2-epoch smoke test (job 53476) confirms real learning (psi loss
decreasing epoch-over-epoch, S0 psi EV=0.679 already after 2 epochs, vs.
the old backbone's -0.12). Full 400-epoch training launched (job 53477).
