## 2026-09-14: Q7 -- DirectUNet with the T-merged-into-channels backbone

**Summary:** New `model_type="direct_unet_tchannels"` -- same role as Q1
(single-pass, obs-only, M-tier), but built on Q6's `MonaiUNet2DQGSolver`
backbone (T merged into channels) instead of `MonaiDirectUNetQG`'s
per-day-independent batch-folding. Answers whether Q6's redesign helps the
simple single-pass scheme too, not just the unrolled solver.

**Files modified:**
- `models/monai_unet_qg2d.py` -- new `MonaiDirectUNetQGChannelTime`:
  `forward(batch) -> (B, T, D)`, same external contract as
  `MonaiDirectUNetQG`, built entirely on the already-existing
  `MonaiUNet2DQGSolver`. Obs-only (no forcing/param/IC hooks).
- `train_qg_neural.py` -- `build_model()`'s new branch (hardcoded
  `hidden_channels=[64,128,256]`, `norm_num_groups=4` -- MONAI channel-
  divisibility, same reasoning as Q6's S-tier); `epochs_for()` (200, same
  as Q1); `--model-type` choices; `QGNeuralLightning.
  _estimate_and_psi_loss`/`estimate_windows` share `direct_unet`'s own
  branch (identical loss/eval logic, only the model class differs).
- `config/experiment/Q7_direct_unet_tchannels_s0.yaml`,
  `batch/run_qg_q7_direct_unet_tchannels_train.sbatch` -- new.
- `tests/test_monai_unet_qg2d.py`, `tests/test_qg_neural.py` -- 4 new
  tests: forward shape/gradient, NaN-obs handling, wrong-T `ValueError`,
  and the YAML config sanity check.

**Rationale:** See PLAN.md's 2026-09-14 Q7 section for the full design
discussion (follow-up to Q6's `monai2d` backbone redesign).

**Verification:** `pytest tests/test_qg_neural.py tests/test_monai_unet_qg2d.py
-m "not slow"` (fdv + fdv-monai-proto envs) -- all passing except the same
pre-existing, order-dependent flaky test already documented in the Q6
fragments (confirmed unrelated). `ruff check` clean. 2-epoch smoke test
(job 53481): clean, ~50.7s/epoch (much faster than Q1's own ~222s/epoch --
single-pass, no unrolling), S0 psi EV=0.837/q EV=0.032 already after 2
epochs. Full 200-epoch training launched.
