## 2026-09-14: New psi2 (lower-layer) point obs + cols_sampling="random" mode; ETKF/EnKF obs-density sensitivity (with a major correction)

**Summary:** Adds two new opt-in QG observation capabilities motivated by
the unobserved-deep-layer (PV q-layer2) weakness recurring throughout the
ETKF-ridge and 4DVar sensitivity work: `psi2_points_per_day` (independent
lower-layer random-point observations -- the lower layer was never
directly observed before this) and `cols_sampling="random"` (upper-layer
column observations as random `(t, x)` draws across the whole day,
allowing multiple columns per timestep -- an alternate sampling mode for
the same `cols_per_day` density knob, removing its `steps_per_day`
ceiling). Also fixes a pre-existing bug: the default `cols_sampling=
"sequential"` sampler hangs forever (confirmed via an isolated 15s-timeout
call) when `cols_per_day > steps_per_day`, instead of failing cleanly --
now raises a clear `ValueError`.

**The obs-density sensitivity result went through a real correction.** An
initial N=10 sweep (ETKF, later cross-checked on EnKF too) found more
upper-layer density "destabilizing" S1 (`cols_sampling="random"` at
density 16 collapsing to psi EV -1.19), while adding psi2 observations
looked like the best config on both scenarios. The user pushed back
(correctly) and asked for an inflation check and a `cols=64` run,
expecting *more* data to help, not hurt. That follow-up found: (1)
inflation=1.0 was already optimal, ruling out inflation mismatch; (2)
`cols=64` was worse still (psi -17.6) but S0 (no model error) *also*
degraded, which a pure-model-error story can't explain; (3) sweeping
`loc_radius` down (N_ensemble=80 fixed) fully recovered and then exceeded
the original baseline at both cols=16 and cols=64 -- `loc_radius=6.0` was
tuned for the sparse 4-8/day regime and became an ensemble-conditioning
bottleneck at higher density (classic sampling-error/rank-deficiency,
the standard fix being to shrink localization when ensemble size can't
grow). The psi1+psi2 combined config, re-swept, was already near its own
`loc_radius` optimum (its density is much lower) -- but once *psi1-only*
configs get their own fair, density-matched `loc_radius`, they clearly
outperform the psi1+psi2 mix at its lower density. **Both original
headline conclusions ("more psi1 destabilizes S1", "psi2 obs uniquely
valuable") are therefore retracted** -- see `PLAN.md`'s "MAJOR CORRECTION"
subsection for the full trail. The underlying capabilities (psi2 point
obs, `cols_sampling="random"`, the sequential-sampler bug fix) remain
correct and useful; only the scientific interpretation built on top of
them, at the originally-used fixed `loc_radius=6.0`, was wrong.

**Genuinely open question**: does psi2 information help at *matched*
total observation density against a properly-localized pure-psi1 config?
Not yet tested.

**Files modified:**
- `data/qg.py` -- `QGConfig.psi2_points_per_day`/`cols_sampling` (new,
  defaults 0/"sequential"); `_layer_field` (generalizes `_upper_field`);
  `_generate_random_point_observations`/`_generate_random_column_point_observations`
  (new); `ValueError` guard in both collision-avoiding samplers.
- `evaluation/baselines.py` -- `ETKF`/`EnKF`'s `_per_time`/`assimilate()`:
  unified h-mode/index-mode observation-width derivation (`idx.numel()`-
  based, backward-compatible) so a combined multi-stream H-function's
  per-time-varying width works for both filters; `num_steps` now derived
  from `obs_mask.shape[0]` (identical value at every existing call site)
  so `observations` can be a variable-width list, not just a dense tensor.
  Every other class in this shared file untouched.
- `evaluation/run_qg_baselines.py` -- `_event_points`/`_event_column_groups`
  (new extractors), `_psi_h_combined`/`_combined_index_at`/
  `_combined_observations`/`_combined_obs_mask`/`_build_qg_col_point_loc_matrices`
  (new combined observation-operator/localization machinery, used by either
  new capability); `--cols-sampling`/`--psi2-points-per-day` CLI flags.
- `tests/test_qg_psi2_points.py` (new, 15 tests), `tests/test_qg_cols_sampling.py`
  (new, 10 tests, renamed from an earlier `test_qg_col_points.py`) -- both
  check mechanical correctness only (shapes, determinism, no-hang,
  H-function/localization correctness), no scientific claims baked in.
- `reports/qg/outputs/qg_obs_density_sweep/*.json` (new, N=10, 45 files:
  the original ETKF/EnKF density sweep, the EnKF cross-check, the
  inflation and `loc_radius` sweeps at cols=16/64, and the psi1+psi2
  `loc_radius` sweep).
- `PLAN.md` -- full narrative including the correction trail.

**Rationale:** Both the ETKF-ridge and 4DVar sensitivity studies kept
running into the same wall -- PV on the unobserved lower layer collapses
under model error, and no hyperparameter fixes it. This tests the more
direct hypothesis: does the deep layer need to be *observed*, even
sparsely, rather than purely inferred through dynamics. The answer, after
correction, is: inconclusive at matched density, and the apparent
"more-of-the-same-hurts" effect was a separate, fixable localization
issue, not evidence either way.

**Verification:** `ruff check` passes on all touched files; full QG test
suite (173+ tests including the 25 new ones, `-m "not slow"`) passes
unchanged; the previously-hanging `cols_per_day=16` case now raises a
clear error in <1s instead of hanging (confirmed via isolated timeout
test); the `cols_sampling="random"` sanity check at matched density
(cols=4) reproduces the default sampler's performance within N=10 noise;
the `loc_radius` sweep at cols=16/64 (both scenarios) and at the psi1+psi2
config confirms the corrected conclusions above.
