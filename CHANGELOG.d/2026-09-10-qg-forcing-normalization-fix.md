## 2026-09-10: QG Q3/Q4 forcing-normalization fix (training collapse)

**Summary:** A full 200-epoch Q3 training run collapsed at epoch 28 --
train/val loss froze at an exact constant for 13+ consecutive epochs after
declining healthily. The frozen value matched exactly what a model gets
from predicting the (normalized) zero mean for both targets, i.e. the
network died. Root cause: the forcing field was left unnormalized on an
unverified assumption that its scale was "already comparable" to the
z-scored channels it's concatenated with -- checked after the collapse and
found false by ~12-13 orders of magnitude (raw `wind_curl` is
`O(1e-13)-O(1e-12)` vs. unit-variance psi/obs/param channels).

**Files modified:**
- `data/qg_neural.py` -- `QGNeuralDataset` gained `forcing_norm_stats`
  (required whenever `cond_mode != "none"`, mirroring `param_norm_stats`);
  the forcing field is now z-scored with a single global scalar (not
  per-grid-cell, to preserve the storm-location spatial pattern).
- `precompute_qg_norm_stats.py` -- new `--output-forcing` flag.
- `train_qg_neural.py` -- new `--forcing-norm-stats-path` CLI flag, wired
  through dataset construction and `estimate_windows`.
- `config/experiment/Q3_direct_unet_s0_oracle_cond.yaml`,
  `Q4_direct_unet_s1_noisy_cond.yaml` -- new `forcing_norm_stats_path`.
- `batch/run_qg_param_stats.sbatch` -- also computes the forcing stats.
- Tests: `tests/test_qg_neural.py` -- forcing-stats-required regression
  test, and a test asserting normalized forcing is actually O(1)-ish (not
  still ~1e-12) -- the previous tests only checked normalization *ran*,
  not that it meaningfully rescaled the value, which would not have caught
  this bug.

**Rationale:** A 2-epoch smoke test cannot catch a divergence that only
manifests dozens of epochs into training -- this bug was invisible to
every prior smoke-test validation. Both in-flight Q3/Q4 full runs were
stopped (Q4 hadn't reached a comparable epoch count to know if it would
have failed the same way, but the same latent bug applied to it too)
pending a longer validation run before relaunching.

**Verification:** `pytest tests/test_qg_neural.py -m "not slow"` (fdv env)
-- 28 passed, 1 skipped (monai-gated). `ruff check` clean on all touched
`.py` files.
