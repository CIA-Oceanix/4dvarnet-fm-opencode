## 2026-09-14: Q7 retrain at the DA-matched noise/lag (Q7-noise05)

**Summary:** New config `Q7_direct_unet_tchannels_s0_noise05.yaml` retrains
Q7 (T-channels DirectUNet, obs-only) at lag=5.0d/noise=0.05 -- the same
observation difficulty Q8/Q9/Q10 now train at -- so the final T-channels
bench table has a genuinely matched obs-only reference row instead of
comparing Q8/Q9/Q10's harder-obs numbers against the original Q7's
easier-obs ones.

**Files modified:**
- `config/experiment/Q7_direct_unet_tchannels_s0_noise05.yaml`,
  `batch/run_qg_q7_direct_unet_tchannels_noise05_train.sbatch` -- new. No
  `obs_noise_std_frac`/`init_lag_days` override (picks up the current
  0.05/5.0 fallback default directly, like Q8/Q9/Q10); `gradient_clip_val:
  1.0` set proactively (same reasoning as Q8/Q9/Q10).
- `tests/test_qg_neural.py` -- 1 new YAML config sanity test.

**Rationale:** Mid-training comparison showed Q8/Q9/Q10 with substantially
lower val psi loss than Q7 at matching epochs, but Q7 was trained at the
old easy lag=1.0/noise=0.01 default while Q8/Q9/Q10 train at the new
DA-matched default -- an apples-to-apples violation caught before the
final bench table was built. Mirrors the Q3/Q3-noise0.05 precedent. See
PLAN.md's 2026-09-14 "T-channels bench refresh" section.

**Verification:** `pytest tests/test_qg_neural.py -k q7 -m "not slow"`
(fdv env) passing; `ruff check tests/test_qg_neural.py` clean. Full
200-epoch GPU training launched; not yet evaluated.
