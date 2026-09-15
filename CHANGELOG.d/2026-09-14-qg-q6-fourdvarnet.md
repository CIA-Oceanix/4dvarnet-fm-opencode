## 2026-09-14: Q6 (FourDVarNetSolver / FDV1) on QG S0

**Summary:** New `model_type="fourdvarnet"` for `train_qg_neural.py` --
QG's first unrolled-variational-solver scheme, same obs-only role as Q1.
Hyperparameters mirror L96's current winning "S"-tier config
(`aux_var_cost_weight=0.01`), adapted to QG's `state_dim=8192`.

**Files modified:**
- `models/fourdvarnet.py` -- `_unrolled_blocks` now only `.unsqueeze(-1)`s
  a 2D `(B,T)` obs_mask, using an already-3D `(B,T,D)` mask as-is
  (backward-compatible with L96's own convention; verified via its full
  85-test suite, unchanged).
- `data/qg_neural.py` -- `QGBatch.obs_mask` is now a genuine per-cell mask
  (matches `obs`/`states` shape exactly), not a collapsed per-day
  any-obs boolean -- unused by direct_unet/vanilla_cfm, which never read
  it.
- `train_qg_neural.py` -- `build_model()`'s new `fourdvarnet` branch;
  `--model-type`'s choices now include it (a real bug: this was missed
  initially, caught only at job-launch time); `QGNeuralLightning.
  _estimate_and_psi_loss`'s new fourdvarnet branch (calls `forward()`
  directly, replicating `compute_loss()`'s optional prior-consistency
  term by hand so it can feed into the existing PV-q auxiliary loss);
  `_pad_batch_for_monai1d`/`_padded_prior_cost`/`_pad_time_axis` -- pad
  the T (days) axis to a multiple of 8 before any `unet_backbone="monai"`
  call (MonaiUNet1D's downsampling needs T divisible by
  `2**(len(hidden_channels)-1)`; QG's 30-day windows aren't), cropping
  the output/reconstruction back before any loss is computed.
- `config/experiment/Q6_fourdvarnet_s0.yaml`,
  `batch/run_qg_q6_fourdvarnet_train.sbatch` -- new.
- `tests/test_qg_neural.py`, `tests/test_fourdvarnet.py` -- new tests:
  fourdvarnet forward/backward + aux-cost-weight, the per-cell obs_mask
  shape/content, the MONAI T-axis padding fix (using Q6's actual
  aux_var_cost_weight=0.01 so both padding paths are exercised), the
  `--model-type`/`build_model()` choices-sync check, and (in
  `test_fourdvarnet.py`) 2D-mask-still-broadcasts / 3D-mask-is-per-cell
  regression tests for the shared `models/fourdvarnet.py` change.

**Rationale:** `FourDVarNetSolver`'s obs cost only supported a
per-timestep mask -- correct for L96's fixed observable-channel subset,
but wrong for QG's per-day-and-per-cell-varying `cols_per_day` geometry;
feeding it the old day-level mask would have silently treated unobserved
cells' zero-fill as real obs=0 measurements. Caveat: Q6's own config uses
`update_input="obs+state"`, which never references `obs_mask` at all, so
this fix has no effect on Q6's own results -- only on any future
grad-conditioned QG config. See PLAN.md's 2026-09-13/14 Q6 section for the
full bug-by-bug story (3 real bugs found and fixed via smoke testing
before the first clean run).

**Verification:** `pytest tests/test_qg_neural.py tests/test_fourdvarnet.py
-m "not slow"` (fdv + fdv-monai-proto envs) -- all passing except one
pre-existing, order-dependent flaky test unrelated to this change
(`test_ensure_truth_cache_redrawn_matches_plain_when_cfg_unchanged`,
confirmed passes reliably in isolation). `ruff check` clean. 2-epoch smoke
test (job 53470) completed cleanly end-to-end (train+val+test eval,
results.json written, ~222s/epoch). Full 400-epoch training launched (job
53473).
