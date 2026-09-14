## 2026-09-14: Q8/Q9/Q10 -- conditioning ported onto the T-channels backbone

**Summary:** `MonaiDirectUNetQGChannelTime` (Q7's backbone) gains the same
`param_dim`/`cond_extra_dim`/`ic_dim` forcing/param/IC conditioning hooks
as `MonaiDirectUNetQG` (Q1/Q3/Q4/Q5's backbone), so the Q3 (oracle cond.),
Q4 (noisy cond.), and Q5 (noisy cond. + IC) ablations can be retrained on
Q7's stronger T-merged-into-channels backbone instead of the older
per-day-independent batch-folding one. New configs Q8/Q9/Q10 mirror Q3/
Q4/Q5 respectively.

**Files modified:**
- `models/monai_unet_qg2d.py` -- `MonaiDirectUNetQGChannelTime.__init__`
  gains `param_dim=0, cond_extra_dim=0, ic_dim=0` (all-zero default keeps
  Q7 itself unaffected); `forward()` folds forcing/params/ic into the
  same per-day channel block as the state placeholder + obs, before T is
  merged into channels.
- `train_qg_neural.py` -- `build_model()`'s `direct_unet_tchannels` branch
  now passes `param_dim`/`cond_extra_dim`/`ic_dim` through (previously
  hardcoded to obs-only).
- `config/experiment/Q8_direct_unet_tchannels_s0_oracle_cond.yaml`,
  `Q9_direct_unet_tchannels_s1_noisy_cond.yaml`,
  `Q10_direct_unet_tchannels_s1_noisy_ic_cond.yaml`,
  `batch/run_qg_q{8,9,10}_direct_unet_tchannels_*_train.sbatch` -- new.
  All three train at the current fallback default (lag=5.0d/noise=0.05,
  the DA reference case) since they're genuinely new configs, with
  `gradient_clip_val: 1.0` set proactively (same noise level that caused
  a real collapse for Q3-noise0.05 at the default clip=10.0).
- `config/experiment/Q7_direct_unet_tchannels_s0.yaml` -- comment updated
  (Q7 stays obs-only; conditioning support now exists but isn't used here).
- `tests/test_monai_unet_qg2d.py` -- 4 new tests: forcing+params
  conditioning changes the output, IC conditioning changes the output,
  and the full combination (forcing+params+ic) constructs/forwards/
  backwards cleanly.
- `tests/test_qg_neural.py` -- 3 new YAML config sanity tests (Q8/Q9/Q10).

**Rationale:** Q7 beat Q1 outright on both psi and q EV (see PLAN.md's Q7
section) -- the user asked to replace the whole DirectUNet benchmark
family (Q1/Q3/Q4/Q3-noise0.05/Q5), not just the obs-only baseline, once
offered the choice of scope. See PLAN.md's 2026-09-14 "T-channels bench
refresh" section for the full design/verification writeup.

**Verification:** `pytest tests/test_qg_neural.py tests/test_monai_unet_qg2d.py
-m "not slow"` (fdv-monai-proto env) all passing; `ruff check` clean.
CPU smoke test at full production scale (nx=64, T=30) for all three new
configs: model construction, forward, and backward all finite; separately
verified the YAML-load -> `build_model()` path wires param_dim/
cond_extra_dim/ic_dim correctly for each. Full 200-epoch GPU training
launched for Q8/Q9/Q10; not yet evaluated.
