## 2026-09-12: fix Q5/Q3-lag5 test-split cache-key bug (silent full rollout)

**Summary:** Q5's and Q3-lag5's smoke test jobs had to be killed after ~10
minutes of silent, expensive truth regeneration. Root cause:
`train_qg_neural.py`'s test-split loading called
`ensure_truth_cache(test_cfg, ...)` directly with a `test_cfg` carrying
Q5's obs/IC-protocol overrides (`obs_noise_std_frac=0.05`,
`init_lag_days=5.0`, `s1_param_bias=0.1`) -- `_truth_cache_path` hashes the
*whole* `QGConfig`, so any of these fields changes the cache key and
silently misses the pre-generated production cache, triggering a full
from-scratch 100-window truth rollout instead of the intended cheap
obs/IC redraw.

**Files modified:**
- `data/qg_neural.py` -- new `ensure_truth_cache_redrawn(cache_cfg,
  eval_cfg, ...)`: loads cached truth at `cache_cfg`'s key (unaffected by
  obs/IC overrides), then redraws obs/init-state via `eval_cfg`'s actual
  settings -- mirrors `eval_qg_neural_s0_s1.py`'s already-proven fix for
  the exact same class of bug.
- `train_qg_neural.py` -- test-split loading now builds a plain
  `test_cache_cfg` (nx/seed/num_test only, like `train_cfg`/`val_cfg`
  already do) and calls `ensure_truth_cache_redrawn` instead of
  `ensure_truth_cache(test_cfg, ...)` directly.
- Tests: `tests/test_qg_neural.py` -- confirms `ensure_truth_cache_redrawn`
  reproduces `ensure_truth_cache`'s output bit-for-bit when `eval_cfg ==
  cache_cfg` (the Q1/Q3/Q4 case, no behavior change), and that a different
  `eval_cfg` doesn't change which cache file is loaded.

**Rationale:** Train/val were unaffected (their `train_cfg`/`val_cfg` were
already plain nx/seed/num_windows configs with no obs-protocol fields, and
on-the-fly obs redraws already correctly used `test_cfg`'s real settings) --
only the fixed test split's cache lookup used the overridden cfg directly.

**Verification:** `pytest tests/test_qg_neural.py -m "not slow"` (fdv env)
-- 34 passed, 1 skipped (monai-gated). `ruff check` clean.
