## 2026-09-10: QG Q4 training perf fix (cached QGDynamics + num_workers)

**Summary:** A 2-epoch smoke test showed Q4 (noisy forcing+param
conditioning) running ~2.4x slower per epoch than Q3/Q1 (~504-533s vs
~220-225s). Root cause: `_noisy_forcing_and_params` rebuilt a full
`QGDynamics` object (spectral PV-inversion machinery) from scratch on every
single training draw just to compute the wind-curl field, serialized with
GPU training by a hardcoded `num_workers=1`.

**Files modified:**
- `data/qg_neural.py` -- new `_cached_qg_dynamics(cfg)` caches the
  `QGDynamics` object per-cfg instead of rebuilding it every draw
  (deterministic given `cfg`, confirmed bitwise-identical output).
- `train_qg_neural.py` -- new `--num-workers` CLI flag (default 4, was
  hardcoded 1) + `persistent_workers=True` for both train/val DataLoaders.
- `batch/run_qg_q3_train.sbatch`, `run_qg_q4_train.sbatch` -- bumped
  `--cpus-per-task` to 6 to back the new default worker count.
- `tests/test_qg_neural.py` -- regression test for the cache (same object
  reused, bitwise-identical `wind_curl_field` output vs. uncached).

**Rationale:** The bottleneck was CPU-side object-reconstruction overhead
and lack of DataLoader parallelism, not GPU floating-point throughput --
confirmed by relaunching on an A100: Q3 (GPU-bound) got a real ~2.7x
speedup, Q4 (CPU-bound before this fix) only ~1.28x, matching the
diagnosis. Fixing the actual bottleneck (redundant object rebuild + no
worker parallelism) is a much better lever than moving the curl computation
to GPU, which wouldn't have addressed either cause and is awkward inside
forked DataLoader worker processes anyway.

**Verification:** `pytest tests/test_qg_neural.py -m "not slow"` (fdv env)
-- 26 passed, 1 skipped (monai-gated). `ruff check` clean on all touched
`.py` files.
