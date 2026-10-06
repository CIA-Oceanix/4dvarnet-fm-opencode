## 2026-09-22: Add per-channel normalization support for L63

**Summary:** L63 had no normalization path at all -- `data.normalize` was a
dead config key for any non-L96 system (`train.py`'s data/eval helpers only
threaded `norm_stats` through the `system == "lorenz96"` branch), so raw,
per-channel-heterogeneous-scale states (X: mean 0.11/std 8.25, Y: mean
0.13/std 9.32, Z: mean 24.9/std 8.85, from `precompute_l63_norm_stats.py`)
were fed straight into the network regardless of the flag. Added an
equivalent per-channel z-score pipeline for L63, reusing L96's existing
`data/normalization.py`/`make_collate_fm` machinery rather than duplicating
it.

**Files modified:**
- `precompute_l63_norm_stats.py` (new) -- computes/saves 3-channel (X,Y,Z)
  mean/std from the S0 train split to `experiments/l63_norm_stats.pt`,
  mirroring `precompute_l96_norm_stats.py`'s convention (single shared
  stats file reused by S0 and S1 training, since L63 has no
  partial-observation subspace like L96's slow/fast split).
- `train.py`:
  - `make_experiment_dataloaders` (the L63/non-L96 training dataloader
    factory) now accepts `norm_stats` and uses `make_collate_fm(norm_stats)`
    instead of always the raw `collate_fm`.
  - `run_experiment`'s `data.normalize`/`norm_stats_path` loading is
    generalized beyond `system == "lorenz96"` to also cover
    `lorenz63` (default stats path `experiments/l63_norm_stats.pt`).
  - `_make_eval_batch`/`evaluate_model`/`save_trajectories` now accept
    `norm_stats`: `obs` (the model's input) is normalized before the
    forward/sample call, and the model's state-space prediction is
    denormalized back to raw physical units before any RMSE/R2/CRPS/
    trajectory is computed -- `true_state` stays raw always (mirrors
    `evaluation/neural_inference.py`'s L96 eval convention: train.py's own
    in-process eval is the *only* results.json source for L63, unlike L96
    which has separate `eval_neural_l96.py`/`eval_monai_l96.py` scripts, so
    this denormalize step is load-bearing here, not just a nicety).

**Rationale:** The L63 monai DirectUNet/CFM S+/M/L tier sweep
(`config/models/monai_direct_unet_*.yaml`/`monai_vanilla_cfm_*.yaml`) was
built to mirror the L96 tier ladder's architecture/scheduler recipe exactly,
but L96's recipe trains on `data.normalize: true`. Without this change, that
parity claim was false on the data side. All 12 existing tier configs still
default to `data.normalize` unset (`false`) -- this change only adds the
capability; none of the in-flight/completed tier runs were touched or
need rerunning as a result of this commit alone.

**Verification:** `pytest tests/test_lorenz96_training.py
tests/test_baselines_hydra.py tests/test_data_leakage.py -q` (56 passed, 2
pre-existing unrelated failures in `evaluation/baselines.py`'s Strong4DVar
dynamics wiring, confirmed present before this change too via `git stash`).
Manually smoke-tested a throwaway 2-epoch/20-window L63 DirectUNet config
with `data.normalize: true` end-to-end (train -> eval -> results.json):
ran without error and produced physical-scale RMSE (X/Y/Z ~ 8-10, consistent
with an undertrained 2-epoch model against the ~8-9 std raw data), confirming
the normalize/denormalize round-trip is wired correctly rather than
producing garbage-scale or NaN output.
