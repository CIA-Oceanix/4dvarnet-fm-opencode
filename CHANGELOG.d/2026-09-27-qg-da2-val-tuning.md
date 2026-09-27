## 2026-09-27: DA-2 — vertical localization, bred initial ensemble, per-window seeds, val tuning sweep

**Summary:** Adds the two DA-2 axes and the sweep driver for tuning on val.
All new options default off, so existing results are unchanged.
- **Vertical localization:** `cross_layer` in `_build_qg_col_loc_matrices`
  and `_build_qg_col_point_loc_matrices`; `loc_cross_layer` in
  `run_qg_baselines.run`. The column localization gave state points in the
  other layer weight 0, so the filter never updated the lower-layer PV
  directly. A weight c > 0 gives them c times the horizontal Gaspari–Cohn
  weight. At 0 the matrices are bit-identical to before. q observations
  refuse the option.
- **Bred initial ensemble:** `_bred_ensemble` and
  `run(init_ensemble_kind="bred", breed_days=3.0)`:
  1. add white noise to the truth `breed_days` before the initial state;
  2. integrate the members with the DA model and the truth's lead-period
     wind (new window field `wind_lead`);
  3. rescale the anomalies to the white-noise amplitude and recentre them
     on the shared initial state.
  No truth later than the initial state is used.
- **Per-window seeds:** `_sample_init_state` and `_ensemble_from_init` take
  `seed_key`. The spectral-wind driver passes the dataset index, so each
  window gets its own lag and perturbations. Without a key the legacy seeds
  are unchanged.
- **Driver:** `evaluation/run_qg_specwind_da.py` gains `--loc-cross-layer`,
  `--init-ensemble`, `--breed-days` and `--disp-frac`.
- **Sweep:** new `evaluation/qg_specwind_da2_sweep.py` and
  `batch/run_qg_specwind_da2.sbatch`: 72 configurations (54 ETKF, 18 EnKF)
  on 20 val windows. It computes per-window ψ₁/ψ₂/q₁/q₂ EV and RMSE from the
  saved trajectories with each window's own parameters, and summarizes
  against the defaults with a paired bootstrap.
- **Design doc** §4.1 and §6: the new grid, why, selection rule, cost.

**Files modified:** `evaluation/baselines.py`, `evaluation/run_qg_baselines.py`,
`evaluation/run_qg_specwind_da.py`, `data/qg_specwind_neural.py`,
`tests/test_qg_specwind_da.py` (+4 tests), `docs/plans/analysis/qg_specwind_da_s0.md`;
new `evaluation/qg_specwind_da2_sweep.py`, `batch/run_qg_specwind_da2.sbatch`.
**Rationale:** In the DA-1 check, the unobserved ψ₂ degraded in a calm
window. The zero cross-layer localization weight and the grid-scale,
vertically uncorrelated white-noise ensemble are the two direct suspects,
so both become tuning axes.
Also found: `run()`'s per-field ψ metrics invert every window with window
0's parameters. That is approximate for datasets whose parameters vary per
window, so the sweep computes its own metrics. The legacy path is left
unchanged.
**Verification:**
- `pytest -m "not slow" tests/test_qg_specwind_da.py tests/test_qg_baselines.py
  tests/test_qg_baselines_4dvar.py tests/test_qg_specwind_neural.py`: 53 passed.
- `ruff check` on the touched files: clean.
