## 2026-09-26: Coupled (Option C) dataset driver + first coupled dataset (train/val)

**Summary:** The G2 dataset framework now generates datasets from the
two-way coupled system:
- `QGDatasetSpec.kappa_fb`, hashed only for `driver="coupled"`, so existing
  split hashes are unchanged;
- specs `qg_coupled_gyrostat_v1` (paired with the forced v1: same entropies,
  factor design and seeds) and `qg_coupled_gyrostat_demo`;
- `write_shard` dispatches the coupled driver to
  `models.qg_coupled.generate_coupled_windows`, which now returns the full
  G2 format (regime at window start, rms curl, KE, hashes, design units)
  plus a per-window feedback ratio;
- manifests record the feedback ratio.

`reports/qg/visualize_qg_dataset.py` handles a missing test split and the
feedback ratio, and gains `--pair-spec` for a window-by-window comparison
with a paired dataset. Generated `qg_coupled_gyrostat_v1` train/val
(5000/500) at κ_fb = 1.

**Files modified:** `data/qg_datasets.py`; `models/qg_coupled.py`
(`generate_coupled_windows` in G2 format; rollout captures state and
diagnoses from a given step); `reports/qg/visualize_qg_dataset.py`;
`tests/test_qg_coupled.py` (+2 tests); `docs/results/qg_coupled_dataset.md`
— new.
**Rationale:** The coupled system is the configuration closest to real
coupled reanalyses, so it needs the same dataset protocol as the forced
one. Findings:
- **Generation:** 44 min on one RTX 8000 (0.48 s per window, 1.26× forced),
  14.3 GB.
- **Independence:** passes (0.6% of val below train's 1% NN quantile; max
  start-state correlation 0.84).
- **Val matches train:** KS p ≥ 0.90.
- **Feedback ratio:** median 1.2% (p90 3.7%) of the gyrostat's own
  tendency.
- **Paired with the forced dataset:** median KE ratio 1.00 (p10–p90
  0.73–1.42), KE correlation 0.80. Physical feedback leaves the energy level
  unchanged on average; the scatter is chaos.

**Verification:**
- `pytest tests/test_qg_coupled.py tests/test_qg_datasets.py
  tests/test_qg_specwind_neural.py tests/test_qg_batched.py` plus the docs
  and reports index tests: 95 passed.
- `ruff check` on the touched files: clean.
- The dataset's independence report passes.

**Additions (same PR):**
- **Coupled test split** (500 windows, 247 s, 7.4 GB, read-only). All
  independence pairs pass (train–test and val–test: 0.6% and 0.8% below
  the reference 1% NN quantile; max start-state correlation 0.79 and 0.78).
- **`docs/plans/analysis/qg_specwind_da_s0.md`** (DRAFT v1): the design of
  ETKF/EnKF S0 experiments on the forced and coupled test sets.
  - Scope: ocean state only, with the true wind prescribed, which is a
    perfect model for both datasets.
  - Observations: ψ₁ on 3 random meridional columns per day (4.7% of the
    grid, each once per day), noise 5%, 5-day lagged initial conditions;
    the QG defaults are re-checked on val.
  - Implementation: the minimal `QGDynamics` spectral hook of PR-2 plus a
    thin driver over `run_qg_baselines.run(ds=...)`.
  - About 21 GPU-hours.
