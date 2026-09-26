## 2026-09-26: G2 — QG dataset framework (specs, independent splits, leakage checks) + first dataset

**Summary:** Implements G2 of `docs/plans/tech/qg_batched_generation_datasets.md`.

`data/qg_datasets.py` provides:
- a versioned `QGDatasetSpec`, with the production spec
  `qg_specwind_gyrostat_v1` (5000/500/500) and a demo spec (256/64/64);
- per-window `SeedSequence` spawn keys `(split_id, index)`, with a separate
  test entropy (the test set does not depend on the train or val sizes);
- an independent Latin-hypercube factor design per split, with a 20% calm
  stratum and the gyrostat time unit (30–90 d);
- batched GPU generation into contiguous shards, in the lean storage format
  (train/val every 12 h, test every step);
- manifests, a read-only test directory, and a loader that refuses test
  data unless `purpose='test'`;
- five leakage checks and a diversity report.

Also adds:
- `scripts/generate_qg_dataset.py` (generate, assemble, all, estimate) and
  `batch/generate_qg_dataset.sbatch` (a SLURM array over shards);
- `reports/qg/visualize_qg_dataset.py`, which exports the overview-page
  data (train and val only; test withheld by protocol);
- `return_modes` on `batched_spectral_wind`.

**Files modified:**
- new: `data/qg_datasets.py`, `scripts/generate_qg_dataset.py`,
  `batch/generate_qg_dataset.sbatch`, `reports/qg/visualize_qg_dataset.py`,
  `tests/test_qg_datasets.py` (11 tests),
  `docs/results/qg_specwind_demo_dataset.md`;
- `models/qg_wind_batched.py` (`return_modes`);
- `tests/test_qg_batched.py` (+1).

**Rationale:** Decisions 1–3 and 6 require a reproducible generator whose
test set is independent by construction and whose independence is
checked, not assumed. Findings from the first dataset (demo instance):
- **All independence checks pass.** Val and test nearest-neighbour distances
  to train match train's own; the largest start-state correlation across
  splits is 0.66.
- **Val matches train** (KS p-values 0.97–1.00).
- **Gyrostat regimes synchronize.** The ring coupling puts all four copies
  in the same lobe in about 50% of windows, and the alternating codes are
  nearly absent. This bears on the pending ring-coupling factor.

Design choices made during implementation:
- contiguous shards, since strided shards made sequential reads reload a
  685 MB shard per window;
- a KD-tree for nearest neighbours instead of an O(N²) distance matrix.

The full 5000/500/500 run (~23 GB) is not launched: `/Odyssey` was at 100%
(170 GB free).

**Verification:**
- `pytest tests/test_qg_datasets.py tests/test_qg_batched.py`: 29 passed.
- The demo dataset was generated end to end on the GPU (train 256 in
  96.5 s, val and test 64 each in about 58 s); the independence report
  passes.
- `ruff check` on the new files: clean.
