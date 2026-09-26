## 2026-09-26: DA-1 — spectral wind hook in QGDynamics + S0 DA driver for the spectral-wind datasets

**Summary:** Implements DA-1 of `docs/plans/analysis/qg_specwind_da_s0.md`.
- **`QGDynamics`:** opt-in `wind_driver` (spectral drivers use a
  `FourierWindBasis` PV source; `wind_state_dim` = 12 for `|k| <= 2`),
  `wind_kmax`, and the relative-wind eddy drag `r_cf`. The defaults
  (`"ou"`, `r_cf = 0`) leave the legacy path bit-identical; the legacy QG
  dynamics and data tests pass unchanged. This is the minimal
  `QGDynamics` part of the Option B plan's PR-2.
- **`evaluation/run_qg_baselines._build_dyn`:** builds the spectral-forced
  DA model, with the truth's `r_cf`, for windows carrying `specwind`
  metadata. The one-layer model refuses them until PR-2.
- **`data/qg_specwind_neural.py`:**
  - `load_full_res_windows` for selected indices: test is read directly;
    thinned splits are regenerated and checked against storage with a
    relative tolerance;
  - `with_fixed_obs` is seeded by dataset index, so observations do not
    depend on the shard;
  - the window metadata carries `r_cf` and `kmax`.
- **`evaluation/run_qg_specwind_da.py`:** loads the windows, adds fixed
  nadir-like ψ₁ observations (`random_columns`, 3 columns per day by
  default) and the S0 fields (true parameters, the truth's wind), then
  calls `run(..., ds=...)`.
- **`batch/run_qg_specwind_da.sbatch`:** a SLURM array over window shards,
  reading the staged copy under `experiments/qg_datasets`.

**Files modified:** `models/qg_dynamics.py`, `evaluation/run_qg_baselines.py`,
`data/qg_specwind_neural.py`; new `evaluation/run_qg_specwind_da.py`,
`batch/run_qg_specwind_da.sbatch`, `tests/test_qg_specwind_da.py` (5 tests).
**Rationale:** The DA runs (DA-2, DA-3) need a DA model that matches the
truth exactly under S0. The truth has spectral forcing and the eddy drag,
which the legacy `QGDynamics` had neither of. Found along the way:
`_sample_init_state` indexes out of range when the sampled lag is below one
model step (`kk = 0`). This is unreachable at the reference 5-day lag,
noted in the test and left unchanged here.
**Verification:**
- `pytest tests/test_qg_specwind_da.py` (5 passed); the spectral
  `QGDynamics` matches `BatchedQGDynamics` with eddy drag to 1e-10 in
  float64. Legacy `tests/test_qg_dynamics.py` and
  `tests/test_qg_data.py`: passed.
- Smoke run: ETKF, N = 80, 3 columns per day, 2 forced test windows from
  the staged copy. DA took 43 s per window. q EV 0.20 against −0.37 for
  the free forecast, with RMSE improved 1.37×; 2 windows, not a result.
