## 2026-09-13: Q1-obsdensity (obs-density-augmented Q1 training)

**Summary:** New Q1 variant where the TRAIN split resamples `cols_per_day`
uniformly on every draw (instead of one fixed value), mirroring L96's
obs-density-augmented training, so one checkpoint generalizes across
observing-network densities. Val/test stay at the fixed `cols_per_day=4`
reference density.

**Files modified:**
- `data/qg_neural.py` -- `QGNeuralDataset` gained `cols_per_day_range=(lo,
  hi)`: requires `on_the_fly_obs=True`, validates `1 <= lo <= hi`, and
  (new, see Rationale) validates `hi <= steps_per_day(cfg)` for
  `obs_geometry="random_columns"`. `_resolved_window` samples a fresh
  `cols_per_day` per draw via `dataclasses.replace(cfg, cols_per_day=...)`
  (mirrors `_noisy_forcing_and_params`'s existing per-draw-cfg pattern).
- `data/qg.py` -- `_generate_random_column_observations` now raises
  `ValueError` if `cols_per_day > steps_per_day` instead of silently
  hanging.
- `train_qg_neural.py` -- new `--cols-per-day-min`/`--cols-per-day-max`
  (YAML-fallback pattern), wired into the train dataset only.
- `config/experiment/Q1_direct_unet_s0_obsdensity_aug.yaml`,
  `batch/run_qg_q1_obsdensity_aug_train.sbatch` -- new.
- `tests/test_qg_neural.py`, `tests/test_qg_random_columns.py` -- 4 new
  tests (2 reproduce the hang condition as a fast `ValueError` check).

**Rationale:** The originally-requested range `[4, 24]` hung a real GPU
job (job 53434) -- `_generate_random_column_observations` assigns each
column its own distinct intra-day time slot via a collision-avoidance
loop, and at the production config's `dt=7200s` (`steps_per_day=12`), any
value above 12 can never find a free slot, so the loop spins forever.
Fixed with a defensive `ValueError` at two levels (the low-level function,
and dataset-construction time) plus capping the config's range at `[4,
12]` (the widest this config physically supports). See PLAN.md's
2026-09-13 Q1-obsdensity section for the full story.

**Verification:** `pytest tests/test_qg_neural.py tests/test_qg_random_columns.py
-m "not slow"` (fdv env) -- all passing, including the 2 new hang-condition
regression tests. `ruff check` clean. Re-ran the same 2-epoch smoke test at
the corrected `[4, 12]` range on a real GPU node (job 53439): completed
cleanly, ~217s/epoch (in line with Q1's own historical per-epoch cost),
full `results.json` written. Full 200-epoch training launched (job 53444).
