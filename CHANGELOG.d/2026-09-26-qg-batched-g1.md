## 2026-09-26: G1 — batched QG dynamics and batched spectral wind on GPU

**Summary:** Implements G1 of `docs/plans/tech/qg_batched_generation_datasets.md`:
- **`models/qg_batched.py`:** `BatchedQGDynamics`, with the same numerics
  as `QGDynamics` but `rd, U1, U2, beta, rek` held as `(B,)` tensors, plus the
  relative-wind eddy drag `-r_cf * zeta_1` (decision 4). Wind enters through
  `SpectralWindForcing` or `StormWindForcing`. `QGDynamics` is untouched.
- **`models/qg_wind_batched.py`:** `BatchedGyrostat`, batched unit OU, a
  batched multivariate phase surrogate, and `batched_spectral_wind`. Each
  window has its own seed, level, drift, start phase and gyrostat time
  unit, and gets its own 50-unit burn-in.
- **`scripts/bench_qg_batched.py`:** a benchmark for batched generation.

**Files modified:** the three new files above; `tests/test_qg_batched.py`
(16 tests); `docs/plans/tech/qg_batched_generation_datasets.md` (§2.5
measured costs, §3.2 revised burn-in).
**Rationale:** The critical path for the settled 5000/500/500 datasets.
Design points:
- **Parity:** each window of a batch matches a `QGDynamics` rollout with
  its own parameters to 1e-10 (float64, unforced, spectral and storm
  forcing).
- **Bit-exact batch invariance of the wind.** The gyrostat right-hand side
  uses gathers and explicit elementwise adds over a fixed role table (no
  atomics, no reductions), noise and phases come from per-window
  generators, and the surrogate FFT is per window. This is needed because
  the chaotic burn-in amplifies roundoff by about e^60. A batched FFT over
  windows was found to round differently with batch size (2e-13), hence the
  per-window FFT.
- **The attractor-state cache in the plan is replaced by per-window
  burn-ins** run as one GPU batch: independent by construction, and the
  cost does not grow with the number of windows.
- **Vectorizing the gyrostat right-hand side cut the test time from 70 s to
  19 s.**

Measured on an RTX 8000 at batch size 256:
- 0.028 ms per window-step (the plan assumed 0.027);
- 0.37 s per window for a full forced 2-year spin-up plus window (about
  0.26 s with split-wide wind batches), i.e. about 26 min for
  5000/500/500;
- ocean trajectories bit-identical alone or in a batch.
**Verification:** `pytest tests/test_qg_batched.py`: 16 passed (19 s,
including the CUDA test). `ruff check` on the new files: clean. The
benchmark JSON is in the PR.

**Review follow-up (same PR):** `BatchedGyrostat.integrate` took the RK4
substep count from the largest `dt` in the batch, so a window's step size
could depend on its batch-mates (the reviewer noted this does not happen in
the production range). Windows are now grouped by their own substep count.
The new test `test_window_series_invariant_when_substeps_differ_across_the_batch`
(time units 30 and 90 d at 12 h steps, so 4 and 2 substeps) covers it.
