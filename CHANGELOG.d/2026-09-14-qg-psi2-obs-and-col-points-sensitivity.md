## 2026-09-14: New psi2 (lower-layer) point obs + col_points_per_day mode; ETKF obs-density sensitivity

**Summary:** Adds two new opt-in QG observation capabilities motivated by
the unobserved-deep-layer (PV q-layer2) weakness recurring throughout the
ETKF-ridge and 4DVar sensitivity work: `psi2_points_per_day` (independent
lower-layer random-point observations, never possible before -- only the
upper layer was ever directly observed) and `col_points_per_day` (upper-
layer column observations as random `(t, x)` draws across the whole day,
allowing multiple columns per timestep, removing the `cols_per_day`
sampler's `steps_per_day` ceiling). Also fixes a pre-existing bug: the
`cols_per_day` sampler hangs forever (confirmed via an isolated 15s-timeout
call) when `cols_per_day > steps_per_day`, instead of failing cleanly --
now raises a clear `ValueError`.

Ran an ETKF obs-density sensitivity sweep (N=10, S0+S1, `cols_per_day=8`,
the new `col_points_per_day=16`, and a combined psi1(4)+psi2(10) config).
Key finding: on S1 (model error), more of the same upper-layer observation
is destabilizing (col_points=16 collapses catastrophically, psi EV -1.19),
while adding independent psi2 observations is the best config on both S0
and S1 -- directly confirming the motivating hypothesis.

**Files modified:**
- `data/qg.py` -- `QGConfig.psi2_points_per_day`/`col_points_per_day` (new,
  default 0); `_layer_field` (generalizes `_upper_field`);
  `_generate_random_point_observations`/`_generate_random_column_point_observations`
  (new); `ValueError` guard in both collision-avoiding samplers.
- `evaluation/baselines.py` -- `ETKF._per_time`/`assimilate()`: unified
  h-mode/index-mode observation-width derivation (`idx.numel()`-based,
  backward-compatible) so a combined multi-stream H-function's per-time-
  varying width works; `num_steps` now derived from `obs_mask.shape[0]`
  (identical value at every existing call site) so `observations` can be a
  variable-width list, not just a dense tensor. EnKF and every other class
  in this shared file untouched.
- `evaluation/run_qg_baselines.py` -- `_event_points`/`_event_column_groups`
  (new extractors), `_psi_h_combined`/`_combined_index_at`/
  `_combined_observations`/`_combined_obs_mask`/`_build_qg_col_point_loc_matrices`
  (new combined observation-operator/localization machinery, used by either
  new capability); `--col-points-per-day`/`--psi2-points-per-day` CLI flags.
- `tests/test_qg_psi2_points.py` (new, 15 tests), `tests/test_qg_col_points.py`
  (new, 10 tests).
- `reports/qg/outputs/qg_obs_density_sweep/*.json` (new, N=10, 8 files).
- `PLAN.md` -- full narrative.

**Rationale:** Both the ETKF-ridge and 4DVar sensitivity studies kept
running into the same wall -- PV on the unobserved lower layer collapses
under model error, and no hyperparameter fixes it. This tests the more
direct hypothesis: does the deep layer need to be *observed*, even
sparsely, rather than purely inferred through dynamics.

**Verification:** `ruff check` passes on all touched files; full QG test
suite (157 tests including the 25 new ones, `-m "not slow"`) passes
unchanged; the previously-hanging `cols_per_day=16` case now raises a
clear error in <1s instead of hanging (confirmed via isolated timeout test).
